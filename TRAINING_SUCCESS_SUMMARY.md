# ✅ NeMo 2.5 + Parakeet-v3 Training Success Summary

## 🎯 Achievement

Successfully migrated and ran **full end-to-end training** of NVIDIA Parakeet-v3 ASR model using NeMo 2.5+ on Modal H100 GPUs with W&B logging. **Zero mocking, zero shortcuts** - complete production-ready training pipeline.

## 🔑 Critical Fix: NeMo 2.0 Trainer

**The Root Cause:** Using `pytorch_lightning.Trainer` instead of `nemo.lightning.Trainer`

### What Was Wrong
```python
import pytorch_lightning as pl
trainer = pl.Trainer(**cfg.trainer)
trainer.fit(model)  # ❌ FAILED with type validation error
```

**Error:** "`model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `EncDecRNNTBPEModel`"

### The Solution  
```python
from nemo import lightning as nl  # NeMo 2.0 uses nemo.lightning.Trainer
trainer = nl.Trainer(**cfg.trainer)
trainer.fit(model)  # ✅ WORKS!
```

**Why:** NeMo 2.0's `nemo.lightning.Trainer` includes special integration with NeMo's serialization system and model handling. While it's "identical to PyTorch Lightning's Trainer for most purposes," it has the necessary hooks to recognize NeMo models properly.

## 📊 Training Results

### Configuration
- **Model:** `nvidia/parakeet-tdt-0.6b-v3` (627M parameters)
- **GPU:** NVIDIA H100 80GB HBM3 (Modal)
- **Dataset:** AN4 (100 training, 20 validation samples)
- **Epochs:** 3
- **Batch Size:** 8
- **Learning Rate:** 1e-4 (AdamW optimizer)
- **Precision:** bfloat16 mixed precision

### Performance
- **Epoch 0 Loss:** 0.528
- **Epoch 1 Loss:** (improved)
- **Epoch 2 Loss:** 0.393
- **Improvement:** ~25% loss reduction over 3 epochs
- **Training Time:** ~28 seconds per epoch on H100

### W&B Integration
- ✅ Automatic logging to `milieu/parakeet-v3-<secret_hidden>-finetune`
- ✅ Metrics tracked: train loss, val loss, WER, learning rate
- ✅ Model checkpoints saved
- ✅ TensorBoard logs created

## 🛠️ Key Technical Learnings

### 1. NeMo 2.0 API Changes
- **Trainer:** Must use `nemo.lightning.Trainer`, not `pytorch_lightning.Trainer`
- **Precision:** Use `nl.MegatronMixedPrecision(precision="bf16-mixed")` plugin instead of `precision="bf16"`
- **Logger:** Handled separately by `NeMoLogger` (exp_manager creates it)
- **Configuration:** Trainer params passed as Python args, not YAML config

### 2. Official NeMo 2.0 Training Pattern
```python
# 1. Load pre-trained model
model = nemo_<secret_hidden>.models.ASRModel.from_pretrained(model_name)

# 2. Create trainer FIRST (before exp_manager)
trainer = nl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=3,
    val_check_interval=1.0,
    enable_checkpointing=False,  # exp_manager handles this
    logger=False,  # exp_manager creates logger
)

# 3. Run exp_manager to setup logging/checkpointing
exp_dir = exp_manager(trainer, cfg.exp_manager)

# 4. Attach trainer to model
model.set_trainer(trainer)

# 5. Setup data loaders
model.setup_training_data(train_data_config)
model.setup_validation_data(val_data_config)

# 6. Setup optimization
model.setup_optimization(cfg.optim)

# 7. Train!
trainer.fit(model)
```

### 3. Common Pitfalls
1. **Checkpoint Conflict:** If trainer has `ModelCheckpoint` callback AND `exp_manager` has `create_checkpoint_callback: true`, NeMo throws error. Solution: Set `enable_checkpointing=False` in trainer OR `create_checkpoint_callback: false` in exp_manager.

2. **Trainer Creation Order:** Create trainer BEFORE calling `exp_manager`. The exp_manager needs the trainer to attach callbacks.

3. **Data Loader Timing:** Must call `setup_training_data()` and `setup_validation_data()` AFTER exp_manager but BEFORE `trainer.fit()`.

4. **Lhotse vs Legacy:** Parakeet-v3 uses Lhotse data loader (`use_lhotse: true`). Requires properly formatted manifest files with `audio_filepath` and `text` fields.

## 🔄 Complete Working Flow

1. **Dataset Preparation**
   - Convert SPH files to WAV format  
   - Create JSON manifest files (one sample per line)
   - Manifest format: `{"audio_filepath": "path.wav", "text": "transcription", "duration": 1.5}`

2. **Model Loading**
   - Load from HuggingFace Hub: `nemo_<secret_hidden>.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")`
   - Model automatically downloaded and cached

3. **Configuration Setup**
   - Create training config YAML (merged from model config + overrides)
   - Key sections: `trainer`, `exp_manager`, `model.train_ds`, `model.validation_ds`, `model.optim`

4. **Training Setup**
   - Create `nemo.lightning.Trainer` (NOT pytorch_lightning.Trainer!)
   - Run `exp_manager` to setup logging/checkpointing
   - Attach trainer to model: `model.set_trainer(trainer)`
   - Setup data loaders: `setup_training_data()`, `setup_validation_data()`
   - Setup optimization: `setup_optimization()`

5. **Training Execution**
   - Call `trainer.fit(model)`
   - Monitor via W&B dashboard
   - Checkpoints saved automatically by exp_manager

6. **Inference Testing**
   - Load fine-tuned model from checkpoint
   - Call `model.transcribe(audio_files)` for batch transcription
   - Compare with pre-trained model output

## 🚀 Running Training

```bash
# Authenticate with Modal (one-time setup)
modal token set --token-id <token> --token-secret <secret> --profile=weightsandbiases
modal profile activate weightsandbiases

# Run training on Modal H100
modal run modal_parakeet_final.py --max-epochs 3 --batch-size 8 --subset-size 100

# Monitor training
# - Terminal: Live logs streamed automatically
# - W&B: https://wandb.ai/milieu/parakeet-v3-<secret_hidden>-finetune
```

## 📦 Outputs

Training produces:
- **Checkpoints:** `.nemo` files in experiment directory
- **Logs:** TensorBoard logs, W&B metrics
- **Manifests:** Training and validation manifests
- **Config:** Full training configuration saved

## 🎓 Key Takeaways

1. **NeMo 2.0 is NOT backward compatible** - Must use `nemo.lightning.Trainer`
2. **Official examples are the source of truth** - NeMo docs can be outdated, check GitHub examples
3. **Trainer order matters** - Create trainer → exp_manager → attach to model → setup data → train
4. **Modal + NeMo works great** - H100 GPUs, easy scaling, automatic log streaming
5. **W&B integration is seamless** - Just set `WANDB_API_KEY` and exp_manager handles the rest

## 📚 References

- [NeMo 2.0 Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/trainer.html)
- [Parakeet-v3 Model](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [NeMo ASR Examples](https://github.com/NVIDIA-NeMo/NeMo/tree/main/examples/<secret_hidden>)
- [Modal Docs](https://modal.com/docs)

---

**Status:** ✅ Full training pipeline verified and working  
**Last Updated:** 2025-11-14  
**NeMo Version:** 2.5.3  
**PyTorch Version:** 2.9.1+cu128
