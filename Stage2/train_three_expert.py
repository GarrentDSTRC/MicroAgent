# -*- coding: utf-8 -*-
"""
三专家融合训练 - CNN + 手工特征 + VLM
"""
import os
import sys
import cv2
import numpy as np
import torch
import random
import json
import handcraft
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cpu")
embedding = 200

VLMPROMPT = """请分析两张连续拍摄的显微镜图像，判断当前聚焦状态的变化趋势，并据此推断电机应向哪个方向移动多少步以接近最佳聚焦位置。 - 如果第二张图像比第一张更模糊，说明焦点正在远离最佳位置，电机应向负方向（"-"）移动； - 如果第二张图像比第一张更清晰，说明焦点正在接近最佳位置，电机应继续向正方向（"+"）移动。 请基于图像清晰度变化，估计电机需移动的步数（取整数），并严格按照以下 JSON 格式返回结果，不要包含任何额外文本或解释： {"analysis": "电机应该向{x}方向移动{y}步。", "direction": "{x}", "distance": {y}} 注意： - "direction" 只能是 "+" 或 "-"； - "distance" 必须是非负整数（如 0, 1, 2, ...）； - "analysis" 中的方向和步数必须与 direction 和 distance 字段一致； - 输出必须是纯 JSON，无 Markdown、无注释、无多余空格或换行。"""


class VLMExpert:
    """VLM专家 - 加载Qwen3VL + LoRA"""
    
    def __init__(self, model_path, adapter_path):
        print("[INFO] 加载VLM模型...")
        
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError:
            print("[WARN] Qwen3VL不可用，使用模拟VLM")
            self.model = None
            self.processor = None
            return
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        self.processor = Qwen3VLProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        print("[INFO] VLM加载完成")
    
    def predict(self, img1_path, img2_path, instruction):
        """预测单个样本"""
        if self.model is None:
            return "+", 5
        
        pil_img1 = Image.open(img1_path).convert('RGB')
        pil_img2 = Image.open(img2_path).convert('RGB')
        
        # 使用instruction中已有的<image><image>标记
        content_list = [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
        
        messages = [{
            "role": "user",
            "content": content_list
        }]
        
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text,
            images=[pil_img1, pil_img2],
            return_tensors="pt"
        )
        
        inputs = {k: v for k, v in inputs.items()}
        
        # 使用generation_config而不是直接传参
        self.model.generation_config.max_new_tokens = 128
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
        
        generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.tokenizer.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True
        ).strip()
        
        return self._parse_response(response)
    
    def _parse_response(self, response):
        """解析VLM输出"""
        print("response",response)
        dir_match = re.search(r'"direction":\s*"([+-])"', response)
        dist_match = re.search(r'"distance":\s*(\d+)', response)
        
        if dir_match and dist_match:
            direction = dir_match.group(1)
            distance = int(dist_match.group(1))
            return direction, distance
        return "+", 5


class DoubleConv(nn.Module):
    """CNN专家 - 与train_adapted.py对齐"""
    def __init__(self, in_channels=3, out_channels=16, embeding=200, num_act=2):
        super().__init__()
        self.embeding = embeding
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.L = nn.Linear(out_channels * embeding * embeding, 50)
        
        # 与train_adapted.py一致的classifier结构
        self.classifier = nn.Sequential(
            nn.Linear(50 * 2 + 8, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, num_act)
        )
    
    def extract_handcraft(self, x):
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3:
            img = x.cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = x.cpu().numpy()
        
        img = img.astype(np.float64)
        
        def safe_log(x):
            return np.log(max(x, 1e-8))
        
        try:
            e = handcraft.entropy(img)
            b = safe_log(max(handcraft.brenner(img), 1e-8))
            v = safe_log(max(handcraft.variance(img), 1e-8))
            en = safe_log(max(handcraft.energy(img), 1e-8))
        except:
            e, b, v, en = 0.0, 0.0, 0.0, 0.0
        
        features = np.array([e, b, v, en], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, state):
        x0 = state[0].to(device)
        x1 = state[1].to(device)
        
        handcraft_feature = self.extract_handcraft(x0).unsqueeze(0)
        handcraft_feature2 = self.extract_handcraft(x1).unsqueeze(0)
        
        X0 = self.double_conv(x0)
        X1 = self.double_conv(x1)
        
        X2 = X0.view(-1, 16 * self.embeding * self.embeding)
        X3 = X1.view(-1, 16 * self.embeding * self.embeding)
        
        X4 = self.L(X2)
        X5 = self.L(X3)
        
        # 与train_adapted.py一致：拼接后过classifier
        features = torch.cat([X4, X5, handcraft_feature, handcraft_feature2], dim=1)
        output = self.classifier(features)
        
        return output, handcraft_feature, handcraft_feature2


class ThreeExpertFusion(nn.Module):
    """三专家融合网络 - 晚期融合策略"""
    def __init__(self):
        super().__init__()
        
        self.cnn_expert = DoubleConv()
        
        # CNN头: 100维 -> 32维
        self.cnn_head = nn.Linear(100, 32)
        
        # VLM头: 2维 -> 32维 (增加权重)
        self.vlm_head = nn.Linear(2, 32)
        
        # 手工特征头: 8维 -> 32维
        self.handcraft_head = nn.Linear(8, 32)
        
        # 融合网络: 32*3=96 -> 64 -> 32 -> 2
        self.fusion = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, state, vlm_feat):
        # CNN专家
        x0 = state[0].to(device)
        x1 = state[1].to(device)
        
        hand_feat1 = self.cnn_expert.extract_handcraft(x0).unsqueeze(0)
        hand_feat2 = self.cnn_expert.extract_handcraft(x1).unsqueeze(0)
        
        X0 = self.cnn_expert.double_conv(x0)
        X1 = self.cnn_expert.double_conv(x1)
        
        X2 = X0.view(-1, 16 * 200 * 200)
        X3 = X1.view(-1, 16 * 200 * 200)
        
        X4 = self.cnn_expert.L(X2)
        X5 = self.cnn_expert.L(X3)
        
        cnn_combined = torch.cat([X4, X5], dim=1)
        cnn_feat = self.cnn_head(cnn_combined)
        
        # VLM特征
        vlm_input = torch.tensor(vlm_feat, dtype=torch.float32).unsqueeze(0).to(device)
        vlm_feat_processed = self.vlm_head(vlm_input)
        
        # 手工特征
        hand_feat_combined = torch.cat([hand_feat1, hand_feat2], dim=1)
        hand_feat_processed = self.handcraft_head(hand_feat_combined)
        
        # 晚期融合: 先归一化，再拼接
        cnn_feat = F.normalize(cnn_feat, dim=1)
        vlm_feat_processed = F.normalize(vlm_feat_processed, dim=1)
        hand_feat_processed = F.normalize(hand_feat_processed, dim=1)
        
        # 拼接
        combined = torch.cat([cnn_feat, vlm_feat_processed, hand_feat_processed], dim=1)
        
        output = self.fusion(combined)
        
        return output
    
    def predict(self, state, vlm_feat):
        output = self.forward(state, vlm_feat)
        direction = "+" if output[0, 1] > 0.5 else "-"
        distance = int(torch.sigmoid(output[0, 0]).item() * 15)
        return direction, distance, output


class VLMDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.data = self._prepare_data(json_path)
    
    def _prepare_data(self, json_path):
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for item in json_data:
            images = item.get('images', [])
            output = item.get('output', '')
            instruction = item.get('instruction', '')
            
            if not images or len(images) < 2:
                continue
            
            img1_name = os.path.basename(images[0])
            img2_name = os.path.basename(images[1])
            
            label_dir = 0.0
            label_dist = 1.0
            
            if output:
                try:
                    output_str = str(output)
                    dir_match = re.search(r'"direction":\s*"([+-])"', output_str)
                    dist_match = re.search(r'"distance":\s*(\d+)', output_str)
                    
                    if dir_match and dist_match:
                        direction = dir_match.group(1)
                        distance = int(dist_match.group(1))
                        label_dir = 1.0 if direction == '+' else 0.0
                        label_dist = distance / 15.0
                except:
                    pass
            
            label = torch.tensor([label_dist, label_dir], dtype=torch.float32)
            
            data.append({
                'img1': img1_name,
                'img2': img2_name,
                'label': label,
                'instruction': instruction
            })
        
        print(f"Loaded {len(data)} samples from {json_path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        img1 = Image.open(os.path.join(self.images_dir, sample['img1'])).convert('RGB')
        img2 = Image.open(os.path.join(self.images_dir, sample['img2'])).convert('RGB')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return [img1, img2], sample['label'], sample['img1'], sample['img2'], sample.get('instruction', '')


def extract_vlm_features(dataset, vlm_expert, images_dir, cache_path="vlm_features_cache.json"):
    """预提取VLM特征（带缓存）"""
    
    # 检查缓存是否存在
    if os.path.exists(cache_path):
        print(f"\n[INFO] 发现VLM特征缓存: {cache_path}")
        print("[INFO] 加载缓存...")
        with open(cache_path, 'r', encoding='utf-8') as f:
            vlm_features = json.load(f)
        print(f"[INFO] 缓存加载完成: {len(vlm_features)} samples")
        return vlm_features
    
    vlm_features = []
    
    print("\n[INFO] 提取VLM特征...")
    for i in range(len(dataset)):
        _, _, img1_name, img2_name, instruction = dataset[i]
        
        img1_path = os.path.join(images_dir, img1_name)
        img2_path = os.path.join(images_dir, img2_name)
        
        try:
            direction, distance = vlm_expert.predict(img1_path, img2_path, instruction)
            
            dir_val = 1.0 if direction == '+' else 0.0
            dist_val = distance / 15.0
            
            vlm_features.append([dist_val, dir_val])
            print(f"  Sample {i}: VLM pred={direction}{distance}")
        except Exception as e:
            print(f"  Sample {i}: VLM error, using default - {e}")
            vlm_features.append([0.5, 0.5])
    
    # 保存缓存
    print(f"\n[INFO] 保存VLM特征缓存到: {cache_path}")
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(vlm_features, f)
    print(f"[INFO] VLM特征提取完成: {len(vlm_features)} samples")
    return vlm_features


def train_fusion():
    print("=" * 60)
    print("Three Expert Fusion Training")
    print("=" * 60)
    
    train_json = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\train_only.json"
    images_dir = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\images"
    model_path = r"D:\MicroAgent\Qwen3-VL-2B-Instruct"
    adapter_path = r"D:\MicroAgent\Stage1\ckpt\checkpoint-60"
    
    transform = transforms.Compose([
        transforms.Resize([embedding, embedding]),
        transforms.ToTensor(),
    ])
    
    dataset = VLMDataset(train_json, images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    

    
    model = ThreeExpertFusion().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    vlm_expert = VLMExpert(model_path, adapter_path)
    vlm_features = extract_vlm_features(dataset, vlm_expert, images_dir, cache_path="vlm_train_cache.json")
    
    print(f"Dataset size: {len(dataset)}")
    
    epochs = 6
    
    # 只提取前10个VLM特征用于快速测试
    max_samples = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, labels, _, _, _) in enumerate(dataloader):
            img1 = images[0].to(device)
            img2 = images[1].to(device)
            labels = labels.to(device)
            
            vlm_feat = vlm_features[batch_idx % len(vlm_features)]
            
            optimizer.zero_grad()
            
            output = model([img1, img2], vlm_feat)
            
            loss = F.mse_loss(output, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / max(1, len(dataloader))
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), r"D:\MicroAgent\Stage2\three_expert_model.pt")
    print("Model saved!")
    
    return model, vlm_expert, vlm_features


def evaluate():
    print("\n=== Three Expert Evaluation ===")
    
    test_json = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\testset.json"
    images_dir = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\images"
    model_path = r"D:\MicroAgent\Qwen3-VL-2B-Instruct"
    adapter_path = r"D:\MicroAgent\Stage1\ckpt\checkpoint-60"
    model_path_ckpt = r"D:\MicroAgent\Stage2\three_expert_model.pt"
    
    transform = transforms.Compose([
        transforms.Resize([embedding, embedding]),
        transforms.ToTensor(),
    ])
    
    dataset = VLMDataset(test_json, images_dir, transform=transform)
    
    model = ThreeExpertFusion()
    model.load_state_dict(torch.load(model_path_ckpt, map_location=device, weights_only=False))
    model.eval()
    
    vlm_expert = VLMExpert(model_path, adapter_path)
    vlm_features = extract_vlm_features(dataset, vlm_expert, images_dir, cache_path="vlm_test_cache.json")
    
    total_error = 0
    correct_dir = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            images, labels, _, _, _ = dataset[i]
            
            direction, distance, _ = model.predict(
                [images[0].unsqueeze(0), images[1].unsqueeze(0)],
                vlm_features[i]
            )
            
            true_dir = "+" if labels[1] > 0.5 else "-"
            true_dist = int(labels[0].item() * 15)
            
            error = abs(distance - true_dist)
            total_error += error
            if direction == true_dir:
                correct_dir += 1
            
            print(f"Sample {i}: Pred={direction}{distance}, True={true_dir}{true_dist}, Error={error}")
    
    print(f"\nAvg Error: {total_error/len(dataset):.2f} steps")
    print(f"Direction Acc: {correct_dir/len(dataset)*100:.0f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_fusion()
    else:
        evaluate()
