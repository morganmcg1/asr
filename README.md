# Parakeet-v3 Fine-tuning with NeMo 2.5+

This repository contains updated code for fine-tuning the NVIDIA Parakeet-v3 ASR model using NeMo 2.5+, adapted from the original NVIDIA Riva tutorial that used NeMo 1.23.

## 🚀 Key Updates from Original Tutorial

### Model Changes
- **Updated Model**: Uses `nvidia/parakeet-tdt-0.6b-v3` instead of the original Parakeet model
- **Architecture**: FastConformer-TDT with 600M parameters
- **Languages**: Supports 25 European languages with automatic language detection
- **License**: CC BY 4.0 (commercial/non-commercial use allowed)

### NeMo Version Changes (1.23 → 2.5+)
- **API Updates**: Updated import statements and configuration structure
- **Configuration**: Uses OmegaConf for configuration management
- **Training Setup**: Simplified trainer and experiment manager setup
- **Model Loading**: Direct loading from HuggingFace Hub
- **Logging**: Integrated Weights & Biases support

## 📁 Files Overview

### Core Files
- `parakeet_v3_finetune_nemo2.ipynb` - Complete Jupyter notebook with step-by-step tutorial
- `parakeet_v3_finetune_simple.py` - Standalone Python script for local testing
- `modal_parakeet_finetune.py` - Modal GPU infrastructure script for cloud training

### Key Features
- ✅ **NeMo 2.5+ Compatibility**: Updated for latest NeMo version
- ✅ **Parakeet-v3 Model**: Uses the latest multilingual model
- ✅ **Modal GPU Support**: Cloud training on H100/H200 GPUs
- ✅ **Weights & Biases Integration**: Experiment tracking and logging
- ✅ **Comprehensive Error Handling**: Robust data processing and training
- ✅ **Production Ready**: Includes model export and evaluation

## 🛠️ Installation

### Local Setup
```bash
# Install system dependencies
sudo apt-get update && apt-get install -y sox libsndfile1 ffmpeg libsox-fmt-mp3 jq wget

# Install Python dependencies
pip install nemo_toolkit[asr]>=2.5.0
pip install text-unidecode matplotlib>=3.3.2 librosa soundfile
pip install huggingface-hub>=0.23.2 omegaconf pytorch-lightning
pip install wandb  # Optional for experiment tracking
```

### Modal Setup (for GPU training)
```bash
# Install Modal
pip install modal

# Set up Modal token (replace with your credentials)
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET

# Create environment
modal environment create asr
```

## 🚀 Quick Start

### 1. Local Testing
```bash
# Run simple fine-tuning script
python parakeet_v3_finetune_simple.py
```

### 2. Jupyter Notebook
```bash
# Start Jupyter and open the notebook
jupyter notebook parakeet_v3_finetune_nemo2.ipynb
```

### 3. Modal GPU Training
```bash
# Run on Modal with Weights & Biases logging
MODAL_ENVIRONMENT=asr modal run modal_parakeet_finetune.py --max-epochs 5 --batch-size 8 --use-wandb
```

## 📊 Model Performance

### Parakeet-v3 Features
- **Automatic Punctuation & Capitalization**: No post-processing needed
- **Word-level Timestamps**: Precise timing information
- **Long Audio Support**: Up to 24 minutes with full attention
- **Multilingual**: 25 European languages supported
- **High Accuracy**: State-of-the-art performance on benchmarks

### Training Configuration
- **Learning Rate**: 1e-4 (lower for fine-tuning)
- **Batch Size**: 4-8 (adjustable based on GPU memory)
- **Precision**: bf16-mixed for faster training
- **Optimizer**: AdamW with cosine annealing scheduler

## 🔧 Configuration Options

### Training Parameters
```python
# Basic configuration
max_epochs = 10          # Number of training epochs
batch_size = 8           # Training batch size
learning_rate = 1e-4     # Learning rate for fine-tuning
use_wandb = True         # Enable Weights & Biases logging

# Advanced configuration
precision = 'bf16-mixed' # Mixed precision training
gradient_clip_val = 1.0  # Gradient clipping
val_check_interval = 1.0 # Validation frequency
```

### Model Configuration
```python
# Data configuration
train_ds:
  manifest_filepath: "path/to/train_manifest.json"
  batch_size: 8
  shuffle: true
  num_workers: 4

validation_ds:
  manifest_filepath: "path/to/val_manifest.json"
  batch_size: 8
  shuffle: false
  num_workers: 4

# Optimizer configuration
optim:
  name: "adamw"
  lr: 1e-4
  weight_decay: 0.001
  sched:
    name: "CosineAnnealing"
    warmup_steps: 100
    min_lr: 1e-6
```

## 📈 Weights & Biases Integration

The Modal script includes full W&B integration for experiment tracking:

```bash
# Set environment variables
export WANDB_API_KEY="your_api_key"
export WANDB_PROJECT="asr"
export WANDB_ENTITY="your_entity"

# Run with logging
modal run modal_parakeet_finetune.py --use-wandb
```

### Logged Metrics
- Training/Validation Loss
- Word Error Rate (WER)
- Learning Rate Schedule
- GPU Utilization
- Training Time per Epoch
- Model Transcription Examples

## 🎯 Use Cases

### Ideal For
- **Domain Adaptation**: Adapt to specific vocabulary or speaking styles
- **Language Variants**: Fine-tune for specific regional accents
- **Noisy Environments**: Improve performance on specific audio conditions
- **Custom Vocabulary**: Add domain-specific terms and phrases

### Example Domains
- Medical transcription
- Legal documentation
- Technical presentations
- Customer service calls
- Educational content

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 4

# Use gradient accumulation
--accumulate-grad-batches 2
```

#### 2. Manifest File Errors
- Ensure audio files exist at specified paths
- Check JSON format (one object per line)
- Verify duration calculations are correct

#### 3. Model Loading Issues
```python
# Clear cache if needed
import torch
torch.cuda.empty_cache()

# Verify model name
model_name = "nvidia/parakeet-tdt-0.6b-v3"
```

#### 4. Training Convergence
- Lower learning rate for better stability
- Increase warmup steps for large datasets
- Monitor validation metrics for overfitting

## 📚 Additional Resources

### Documentation
- [NeMo ASR Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html)
- [Parakeet-v3 Model Card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Modal Documentation](https://modal.com/docs)

### Research Papers
- [FastConformer: Local Augmentation for Efficient Conformer](https://arxiv.org/abs/2305.05084)
- [TDT: Token-level Direct Transducer](https://arxiv.org/abs/2304.01556)
- [Parakeet Technical Report](https://arxiv.org/abs/2509.14128)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/asr.git
cd asr

# Install development dependencies
pip install -e .
pip install pytest black flake8
```

## 📄 License

This project is licensed under the MIT License. The Parakeet-v3 model is licensed under CC BY 4.0.

## 🙏 Acknowledgments

- NVIDIA NeMo Team for the excellent framework
- NVIDIA Riva Team for the original tutorial
- Modal Labs for GPU infrastructure
- Weights & Biases for experiment tracking

---

**Note**: This is an updated version of the original NVIDIA Riva ASR fine-tuning tutorial, adapted for NeMo 2.5+ and the Parakeet-v3 model. The original tutorial can be found in the [NVIDIA Riva documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-finetune-parakeet-nemo.html).