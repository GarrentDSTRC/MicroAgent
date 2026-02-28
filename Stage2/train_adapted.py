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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cpu")
embedding = 200

class DoubleConv(nn.Module):
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
        self.classifier = nn.Sequential(
            nn.Linear(50 * 2 + 8, 40), 
            nn.ReLU(), 
            nn.Linear(40, 10), 
            nn.ReLU(),
            nn.Linear(10, num_act)
        )

    def extract_handcraft(self, x):
        """提取手工特征 - 老代码版本"""
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
        
        features = torch.cat([X4, X5, handcraft_feature, handcraft_feature2], dim=1)
        o = self.classifier(features)
        return o
    
    def predict(self, state):
        output = self.forward(state)
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
            
            if not images or len(images) < 2:
                continue
            
            img1_name = os.path.basename(images[0])
            img2_name = os.path.basename(images[1])
            
            label_dir = 0.0
            label_dist = 1.0
            
            if output:
                try:
                    import re
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
                'label': label
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
        
        return [img1, img2], sample['label']


def train():
    print("=" * 60)
    print("Old CV Code - Adapted for New Dataset")
    print("=" * 60)
    
    train_json = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\train_only.json"
    images_dir = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\images"
    
    agent = DoubleConv().to(device)
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.001)
    
    transform = transforms.Compose([
        transforms.Resize([embedding, embedding]),
        transforms.ToTensor(),
    ])
    
    dataset = VLMDataset(train_json, images_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    epochs = 15
    for epoch in range(epochs):
        agent.train()
        epoch_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            img1 = images[0].to(device)
            img2 = images[1].to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = agent([img1, img2])
            
            loss = F.mse_loss(output, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / max(1, len(dataloader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
    
    torch.save(agent.state_dict(), r"D:\MicroAgent\Stage2\model_adapted.pt")
    print("Model saved!")
    
    return agent


def evaluate():
    print("\n=== Evaluation ===")
    
    test_json = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\testset.json"
    images_dir = r"D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\images"
    model_path = r"D:\MicroAgent\Stage2\model_adapted.pt"
    
    agent = DoubleConv()
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    agent.eval()
    
    transform = transforms.Compose([
        transforms.Resize([embedding, embedding]),
        transforms.ToTensor(),
    ])
    
    dataset = VLMDataset(test_json, images_dir, transform=transform)
    
    total_error = 0
    correct_dir = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            images, labels = dataset[i]
            
            direction, distance, _ = agent.predict([images[0].unsqueeze(0), images[1].unsqueeze(0)])
            
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
        train()
    else:
        evaluate()
