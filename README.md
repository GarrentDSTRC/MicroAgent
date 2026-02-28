# Micro-Agent: Mitigating Catastrophic Forgetting in Spatial Prediction Tasks through Two-Stage Adaptive Optimization

## Overview

This project implements a vision-language model (VLM) based micro-agent for microscope image analysis, specifically designed for predicting motor movement direction and step count in autofocus applications. The proposed method addresses **catastrophic forgetting** in spatial prediction tasks through a novel **Two-Stage Adaptive Optimization** approach.

## Key Features

- **Base Model**: Qwen3-VL-2B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Microscope image analysis - predict motor direction (+/-) and distance (steps)
- **Two-Stage Adaptive Optimization**: Combines CNN-based feature extraction with VLM fine-tuning to preserve foundational knowledge while learning new spatial prediction tasks

## Project Structure

```
MicroAgent/
├── Stage1/                    # VLM Fine-tuning Stage
│   ├── MicroAgent.py          # Main inference script
│   ├── qwen3vl_finetune_proper_multi_gpu.py  # Fine-tuning script
│   ├── evaluate_finetuning_final.py          # Evaluation script
│   └── ckpt/checkpoint-60/    # Fine-tuned LoRA weights
│
├── Stage2/                    # CNN + Adaptive Optimization Stage
│   ├── train_adapted.py       # Two-stage training script
│   ├── train_three_expert.py  # Multi-expert training
│   └── data/                  # Training datasets
│
├── Qwen3-VL-2B-Instruct/      # Base model files
└── config/                    # Configuration files
```

## Installation

```bash
# Create conda environment
conda env create -f environment.yaml
conda activate microagent

# Or install dependencies manually
pip install torch transformers peft qwen-vl-utils
pip install opencv-python numpy pillow
```

## Model and Weights Download

Download the base Qwen3-VL-2B-Instruct model:

```bash
modelscope download --model qwen/Qwen3-VL-2B-Instruct
```
Weights:https://pan.zju.edu.cn/share/843a3a8b6f4cc6e774b1cb3075
## Usage

### 1. Fine-tuning (Stage 1)

```bash
cd Stage1
python qwen3vl_finetune_proper_multi_gpu.py --config config/multi_gpu_config.yaml
```

### 2. Two-Stage Training (Stage 2)

```bash
cd Stage2
python train_adapted.py
```

### 3. Evaluation

```bash
cd Stage1
python evaluate_finetuning_final.py config/evaluation_config.yaml
```

### 4. Inference

```python
from MicroAgent import ActionPredictor
from PIL import Image

# Initialize model
predictor = ActionPredictor(
    model_path="D:/MicroAgent/Qwen3-VL-2B-Instruct",
    adapter_path="D:/MicroAgent/Stage1/ckpt/checkpoint-60"
)

# Load microscope images
img1 = Image.open("microscope_image_1.png")
img2 = Image.open("microscope_image_2.png")

# Predict motor movement
result = predictor.predict_direction_and_distance([img1, img2])
print(result)
```

## Experimental Results

### Evaluation on Test Set (14 samples)

| Model | Direction Accuracy | Avg Distance Error |
|-------|-------------------|-------------------|
| Fine-tuned | **71.43%** | **5.29** |
| Base | 14.29% | 7.00 |

**Improvements:**
- Direction accuracy: +57.14% (400% relative improvement)
- Distance error: -1.71 steps reduction

## Two-Stage Adaptive Optimization

The proposed method mitigates catastrophic forgetting through:

1. **Stage 1 - Feature Extraction**: CNN-based handcrafted feature extraction for spatial information
2. **Stage 2 - VLM Fine-tuning**: LoRA-based adaptation with frozen visual encoder to preserve pre-trained knowledge
3. **Adaptive Learning Rate**: Separate optimization schedules for different model components

## Citation

```bibtex
@article{microagent2026,
  title={Micro-Agent: Mitigating Catastrophic Forgetting in Spatial Prediction Tasks through Two-Stage Adaptive Optimization},
  author={},
  year={2026}
}
```

## License

MIT License
