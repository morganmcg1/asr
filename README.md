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

#### Authentication Setup

```bash
# Install Modal
pip install modal

# Set up Modal authentication with your tokens
modal token set --token-id <your-token-id> --token-secret <your-token-secret> --profile=<your-profile>

# Activate the profile
modal profile activate <your-profile>

# Set up W&B credentials
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key
```

#### Running Training

```bash
# Run debugging test first
modal run modal_debug_minimal.py

# Run full training on H100
modal run modal_parakeet_final.py --max-epochs 5 --batch-size 8 --subset-size 100
```

#### Monitoring Training Logs

Modal streams logs in real-time, but here's how to ensure you see everything:

**Option 1: Using `modal run` (Default)**
When you use `modal run`, logs stream automatically to your terminal. The training output, including loss values and progress, will appear in real-time.

**Option 2: Using Python driver with `modal.enable_output()`**
If you're running Modal from a Python script instead of the CLI:

```python
import modal

with modal.enable_output():  # This enables log streaming
    with app.run():
        train.remote()
```

**Option 3: Monitoring detached/running jobs**
To view logs of already-running jobs:

```bash
# List running apps
modal app list

# Stream logs from a specific app
modal app logs <app-name-or-id>

# Or by container
modal container list
modal container logs <container-id>
```

**Pro tip**: Modal's `app logs` and `container logs` commands stream in real-time, so you can monitor training progress even if you detached from the original session.

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
## 🎓 NeMo 2.0+ Training Patterns (CRITICAL)

### Understanding NeMo 2.0 Architecture Changes

NeMo 2.0+ introduced significant API changes from version 1.x. The training workflow follows a specific order that must be respected for successful training.

### Correct Training Pattern (Based on Official NeMo Examples)

The order of operations is **critical** in NeMo 2.0+:

```python
# 1. Create PyTorch Lightning Trainer FIRST
trainer = pl.Trainer(**cfg.trainer)

# 2. Setup Experiment Manager BEFORE touching the model
exp_manager(trainer, cfg.exp_manager)

# 3. Load or create the model
<secret_hidden>_model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")

# 4. Attach trainer to model (NeMo-specific pattern)
<secret_hidden>_model.set_trainer(trainer)

# 5. Setup dataloaders (uses model methods)
<secret_hidden>_model.setup_training_data(cfg.model.train_ds)
<secret_hidden>_model.setup_validation_data(cfg.model.validation_ds)

# 6. Setup optimization
<secret_hidden>_model.setup_optimization(cfg.model.optim)

# 7. Train - NeMo handles dataloaders internally
trainer.fit(<secret_hidden>_model)
```

### ❌ Common Pitfalls

1. **Wrong**: Calling `model.set_trainer()` after `setup_training_data()`
   - **Right**: Call `set_trainer()` BEFORE setting up data

2. **Wrong**: Passing dataloaders explicitly to `trainer.fit(model, train_dl, val_dl)`
   - **Right**: Call `trainer.fit(model)` - NeMo accesses dataloaders via model attributes

3. **Wrong**: Setting up exp_manager after model operations
   - **Right**: Call `exp_manager(trainer, cfg)` immediately after creating trainer

4. **Wrong**: Using `model.cfg.optim = ...` without `setup_optimization()`
   - **Right**: Call `model.setup_optimization(cfg.model.optim)` explicitly

### Key Insights from Official NeMo Code

These patterns are derived from the official [NeMo speech_to_text_finetune.py](https://github.com/NVIDIA/NeMo/blob/main/examples/<secret_hidden>/speech_to_text_finetune.py):

- NeMo models inherit from `LightningModule` but require special initialization
- The `set_trainer()` method properly connects model callbacks and logging
- Data setup methods (`setup_training_data`, `setup_validation_data`) configure internal dataloader attributes
- The `setup_optimization()` method properly initializes optimizers and learning rate schedulers
- `exp_manager()` sets up checkpointing, logging, and experiment directories

### Resources

- [NeMo 2.0 Fundamentals Notebook](https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/00_NeMo_Primer.ipynb)
- [NeMo ASR Training Tutorials](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html#automatic-speech-recognition-<secret_hidden>-tutorials)
- [Official NeMo GitHub Repository](https://github.com/NVIDIA/NeMo)

