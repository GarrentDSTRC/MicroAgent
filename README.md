# Micro-Agent: Mitigating Catastrophic Forgetting in Spatial Prediction Tasks through Two-Stage Adaptive Optimization

## Project Overview

An automated microscope focusing system based on a deep learning multi-expert fusion approach.
![Micro-Agent](./Stage2/docs/MicroAgent.png)
## Project Structure

```
MicroAgent/
├── Qwen3-VL-2B-Instruct/      # Qwen3-VL-2B base model
├── Stage1/                    # VLM fine-tuning phase
│   ├── config/               # Configuration files
│   ├── ckpt/                 # LoRA weights
│   ├── MicroAgent.py        # VLM capability demonstration
│   └── evaluate_finetuning_final.py
├── Stage2/                    # Three-expert fusion phase
│   ├── data/                 # Dataset
│   │   └── vlm_finetune_dataset_fixed/
│   │       ├── train_only.json   # Training set: 126 samples
│   │       ├── testset.json      # Test set: 14 samples
│   │       └── images/           # Image files
│   ├── docs/                  # Experimental result visualizations
│   ├── train_adapted.py       # CNN + handcrafted feature training
│   ├── train_three_expert.py # Three-expert fusion training
│   ├── model_adapted.pt      # CNN expert model
│   └── three_expert_model.pt # Three-expert fusion model
└── environment.yaml          # Environment configuration
```

## Dataset

- **Training Set**: 300+ samples, incorporating include resampled instances curated from our self-collected data
- **Test Set**: 5% of desensitized self-collected data, openly released to foster reproducible research `Stage2/data/vlm_finetune_dataset_fixed/testset.json`
- **Image Repository**: `Stage2/data/vlm_finetune_dataset_fixed/images/`


## Experimental Stages

### Stage 1: VLM Fine-tuning (`Stage1/`)

LoRA-based fine-tuning of the Qwen3-VL-2B-Instruct model.

**Model Paths**:
- Base Model: `Qwen3-VL-2B-Instruct/` (requires manual download)
- LoRA Weights: `Stage1/ckpt/checkpoint-60/`

**Weights Download**: https://pan.zju.edu.cn/share/843a3a8b6f4cc6e774b1cb3075

#### 1. `MicroAgent.py` – VLM Capability Demonstration

Demonstrates three core capabilities of the VLM:
- Pure text dialogue functionality
- Vision-language dialogue (image content description)
- Microscope focusing action prediction (dual-frame analysis)

**Execution Example**:
```bash
cd D:\MicroAgent\Stage1
conda activate qwen3vl
python MicroAgent.py config/evaluation_config.yaml
```

**Sample Output**:
```
[TEST 1] Text Chat Test
[User] Microscope operation procedures and precautions
[Model] Response: The microscope is an indispensable experimental instrument across numerous fields including biology, medicine, and materials science...

[TEST 2] Image Chat Test
[Image] Path: D:\MicroAgent\Stage2\data\vlm_finetune_dataset_fixed\images\sample_0_0_0.png
[User] Please describe the metallic workpiece shown in this image
[Model] Response: This image depicts a metallic workpiece featuring specific geometric shapes and surface characteristics...

[TEST 3] Action Prediction Test
Frame 1: sample_0_0_0.png
Frame 2: sample_0_0_1.png
[Result]: Direction: Left (-), Distance: 1 pixels
```

#### 2. `evaluate_finetuning_final.py` – Stage 1 Quantitative Evaluation

Evaluates the fine-tuned VLM on the test set, outputting quantitative metrics including directional accuracy and distance error.

```bash
cd D:\MicroAgent\Stage1
conda activate qwen3vl
python evaluate_finetuning_final.py config/evaluation_config.yaml
```

### Stage 2: Three-Expert Fusion (`Stage2/`)

Fusion training integrating CNN expert, handcrafted feature expert, and VLM expert.

#### 2.1 CNN + Handcrafted Features (Ablation Study: Excluding VLM)

```bash
cd D:\MicroAgent\Stage2
conda activate qwen3vl
python train_adapted.py --mode train
python train_adapted.py --mode eval
```

#### 2.2 Three-Expert Fusion (CNN + Handcrafted + VLM)

```bash
cd D:\MicroAgent\Stage2
conda activate qwen3vl
python train_three_expert.py --mode train
python train_three_expert.py --mode eval
```

## Model Architecture

### CNN Expert + Handcrafted Features

```
Input Image -> 3-layer CNN (16 channels) -> Linear(16*200*200 → 50)
                      ↓
        Handcrafted Features (4-dim: entropy, log(brenner), log(variance), log(energy))
                      ↓
        Concatenation(50*2 + 8 = 108) -> Linear(108→40) -> Linear(40→10) -> Linear(10→2)
        Output: [distance, direction]
```

### Three-Expert Fusion Architecture

```
CNN Expert: 3-layer CNN + Linear → 100-dim → 32-dim
Handcrafted Features: 4-dim × 2 frames = 8-dim → 32-dim  
VLM Expert: Qwen3VL-2B + LoRA → 2-dim (direction, distance) → 32-dim

Normalized expert features concatenated: 32×3 = 96-dim → 64 → 32 → 2
```

## Environment Configuration

```bash
# Create environment
conda env create -f environment.yaml

# Or activate existing environment
conda activate qwen3vl
```

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy
pillow
opencv-python
transformers>=4.47.0
peft
accelerate
sentencepiece
protobuf
safetensors
huggingface-hub
scipy
matplotlib
scikit-learn
```

## Important Notes

1. VLM training requires activation of the `qwen3vl` conda environment (`conda activate qwen3vl`)
2. All data paths have been unified under the `Stage2/data/` directory
3. Label extraction utilizes regular expression: `"direction":\s*"([+-])"`
4. VLM feature caching files: `vlm_train_cache.json`, `vlm_test_cache.json`