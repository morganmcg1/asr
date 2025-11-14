# ASR Fine-tuning with Parakeet-v3 and NeMo 2.5+

This repository contains updated code for fine-tuning NVIDIA's Parakeet-v3 ASR model using NeMo 2.5+ (updated from the original NeMo 1.23 tutorial).

## 🎯 Overview

The original NVIDIA Riva tutorial used NeMo 1.23 and the original Parakeet model. This repository provides:

- **✅ Updated NeMo version**: From 1.23 to 2.5.3
- **✅ Updated model**: From original Parakeet to `nvidia/parakeet-tdt-0.6b-v3` (627M parameters)
- **✅ Modal GPU support**: Training on H100/H200/A100 GPUs via Modal
- **✅ Weights & Biases integration**: Experiment tracking and model artifacts to "asr" project
- **✅ Fast subset training**: Quick experimentation with AN4 dataset

## 📁 Files

### Core Files
- **`parakeet_v3_finetune_nemo2.ipynb`** - Complete Jupyter notebook with fine-tuning pipeline
- **`modal_parakeet_final.py`** - Modal script for H100 GPU training with W&B logging
- **`requirements.txt`** - Complete dependency list

### Debugging & Development  
- **`modal_debug_minimal.py`** - Debugging script to test model loading and functionality
- **`README.md`** - This documentation
- **`LICENSE`** - MIT License

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Modal GPU Training (Recommended)

```bash
# Install and setup Modal
pip install modal
python -m modal setup

# Set up W&B credentials  
modal secret create wandb-secret WANDB_API_KEY=your_key_here

# Run debugging test first
modal run modal_debug_minimal.py

# Run full training on H100
modal run modal_parakeet_final.py --max-epochs 5 --batch-size 8 --subset-size 100
```

### 3. Jupyter Notebook Development

```bash
jupyter notebook parakeet_v3_finetune_nemo2.ipynb
```

## 🔧 Key Changes from Original Tutorial

### NeMo API Updates (1.23 → 2.5.3)
- ✅ Updated model loading: `ASRModel.from_pretrained()`
- ✅ New configuration structure with OmegaConf  
- ✅ Updated trainer setup and data loading
- ✅ Compatible PyTorch Lightning version (2.4.0)

### Model Updates
- **Original**: `nvidia/parakeet-rnnt-1.1b` 
- **New**: `nvidia/parakeet-tdt-0.6b-v3` (627M parameters)
- ✅ TDT (Token-and-Duration Transducer) architecture
- ✅ Better performance and efficiency

### Infrastructure
- ✅ **Modal GPU support**: H100, H200, A100 training
- ✅ **W&B integration**: Logs to "asr" project with model name tags
- ✅ **Containerized training**: Reproducible environments
- ✅ **Fast subset training**: 100 samples for quick experimentation

## 🤖 Model Details

**Parakeet-v3 (nvidia/parakeet-tdt-0.6b-v3)**
- **Parameters**: 627,008,134 (627M)
- **Architecture**: Token-and-Duration Transducer (TDT)
- **Sample Rate**: 16kHz
- **Tokenizer**: SentencePiece (8192 tokens)
- **Loss**: TDT with configurable durations

## ⚙️ Training Configuration

### Default Settings
- **Epochs**: 5
- **Batch Size**: 8  
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing with warmup
- **Precision**: BF16 mixed precision
- **Dataset**: AN4 (100 samples for fast training)

### W&B Logging
- **Project**: `asr` (always)
- **Tags**: `parakeet-v3`, `nvidia/parakeet-tdt-0.6b-v3`, `h100`, `fine-tuning`
- **Metrics**: Training/validation loss, learning rate
- **Artifacts**: Fine-tuned model with metadata

## 💻 Hardware Requirements

### Modal GPU Training (Recommended)
- **H100/H200/A100** GPU via Modal
- **40GB+** GPU memory for full model
- Fast internet for dataset download

### Local Development
- **8GB RAM** minimum
- **Python 3.8+**
- CPU-only for debugging

## 🐛 Troubleshooting

### Common Issues

1. **PyTorch Lightning compatibility**
   ```bash
   # Use exact versions
   pip install pytorch-lightning==2.4.0 nemo_toolkit[asr]==2.5.3
   ```

2. **Model loading errors**
   - Ensure internet connection for HuggingFace downloads
   - Check CUDA availability: `torch.cuda.is_available()`

3. **Memory issues**
   - Reduce `--batch-size` for smaller GPUs
   - Use gradient accumulation for effective larger batches

4. **Modal setup**
   ```bash
   modal setup  # For authentication
   modal secret create wandb-secret WANDB_API_KEY=your_key
   ```

## 📊 Results

### Training Progress
The training logs show:
- ✅ Model loading: 627M parameters
- ✅ Dataset preparation: AN4 conversion and manifest creation
- ✅ W&B logging: Proper project and tagging
- ✅ GPU utilization: H100 with BF16 mixed precision

### W&B Dashboard
Check your results at: `https://wandb.ai/milieu/asr`

Tags include: `parakeet-v3`, `nvidia/parakeet-tdt-0.6b-v3`, `h100`, `fine-tuning`

## 🔄 Development Workflow

1. **Debug first**: `modal run modal_debug_minimal.py`
2. **Develop interactively**: Use Jupyter notebook
3. **Train on GPU**: `modal run modal_parakeet_final.py`
4. **Monitor progress**: Check W&B dashboard
5. **Download artifacts**: Fine-tuned model from W&B

## 📚 References

- [Original NVIDIA Riva Tutorial](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-finetune-parakeet-nemo.html)
- [NeMo 2.5+ Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Parakeet-v3 Model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Modal Documentation](https://modal.com/docs)

## 📄 License

MIT License - see LICENSE file for details.