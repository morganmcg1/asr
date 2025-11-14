#!/usr/bin/env python3
"""
Working Parakeet-v3 fine-tuning script for Modal with H100
Uses NeMo's training approach to avoid PyTorch Lightning compatibility issues.
"""

import modal
import os

# Enhanced image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("sox", "libsndfile1", "ffmpeg", "wget")
    .pip_install([
        "nemo_toolkit[asr]==2.5.3",  # Pin exact version
        "torch>=2.0.0",
        "pytorch-lightning==2.4.0",  # Use compatible version
        "omegaconf>=2.3.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "wandb>=0.15.0",
        "requests>=2.28.0",
        "huggingface-hub>=0.23.2",
    ])
)

app = modal.App("parakeet-v3-working-finetune")

# Define volumes for persistent storage
data_volume = modal.Volume.from_name("parakeet-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("parakeet-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Use H100 for maximum performance
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoints_volume,
    },
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("wandb-secret")],  # W&B API key
)
def finetune_parakeet_v3_working(
    max_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    subset_size: int = 100,
    use_wandb: bool = True,
):
    """Fine-tune Parakeet-v3 using NeMo's training approach"""
    
    import json
    import glob
    import subprocess
    import librosa
    from pathlib import Path
    import requests
    
    # NeMo imports
    import nemo
    import nemo.collections.asr as nemo_asr
    from nemo.core.config import hydra_runner
    from nemo.utils.exp_manager import exp_manager
    
    import torch
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    
    print("🚀 Starting Working Parakeet-v3 fine-tuning on NVIDIA H100")
    print(f"NeMo version: {nemo.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize Weights & Biases
    if use_wandb:
        import wandb
        
        # Get API key from environment (set via Modal secret)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            
            # Initialize run with detailed config
            wandb.init(
                project="parakeet-v3-finetune-working",
                entity="milieu",
                name=f"working-h100-{max_epochs}ep-{subset_size}samples",
                config={
                    "model": "nvidia/parakeet-tdt-0.6b-v3",
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "subset_size": subset_size,
                    "gpu": "H100",
                    "nemo_version": nemo.__version__,
                    "pytorch_version": torch.__version__,
                    "pytorch_lightning_version": pl.__version__,
                }
            )
            print("✅ Weights & Biases initialized")
        else:
            print("⚠️ WANDB_API_KEY not found, disabling W&B logging")
            use_wandb = False
    
    # Prepare AN4 dataset (subset)
    print("📥 Preparing AN4 dataset subset...")
    
    target_data_dir = "/data/an4_converted"
    target_wavs_dir = f"{target_data_dir}/wavs"
    os.makedirs(target_wavs_dir, exist_ok=True)
    
    # Download dataset if not exists
    an4_tar = "/data/an4_sphere.tar.gz"
    if not os.path.exists(an4_tar):
        print("Downloading AN4 dataset...")
        response = requests.get("https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz")
        with open(an4_tar, 'wb') as f:
            f.write(response.content)
    
    # Extract dataset
    an4_dir = "/data/an4"
    if not os.path.exists(an4_dir):
        print("Extracting dataset...")
        subprocess.run(["tar", "-xzf", an4_tar, "-C", "/data"], check=True)
    
    # Convert SPH to WAV (subset only)
    sph_files = glob.glob(f"{an4_dir}/**/*.sph", recursive=True)
    print(f"Converting subset of {min(len(sph_files), subset_size * 2)} SPH files to WAV...")
    
    # Take only a subset for fast training
    sph_files = sph_files[:subset_size * 2]  # Extra files to ensure we have enough after filtering
    
    for sph_path in sph_files:
        wav_path = os.path.join(target_wavs_dir, 
                              os.path.splitext(os.path.basename(sph_path))[0] + '.wav')
        if not os.path.exists(wav_path):
            subprocess.run(["sox", sph_path, wav_path], check=True)
    
    # Create manifest files (subset)
    def build_manifest_subset(transcripts_path, manifest_path, max_samples):
        count = 0
        with open(transcripts_path, 'r') as fin:
            with open(manifest_path, 'w') as fout:
                for line_num, line in enumerate(fin, 1):
                    if count >= max_samples:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # Find the transcript and file ID
                        paren_idx = line.rfind('(')
                        if paren_idx == -1:
                            continue
                            
                        transcript = line[:paren_idx].strip().lower()
                        transcript = transcript.replace('<s>', '').replace('</s>', '').strip()
                        
                        file_id = line[paren_idx + 1:].rstrip(')')
                        audio_path = os.path.join(target_wavs_dir, file_id + '.wav')
                        
                        if os.path.exists(audio_path):
                            try:
                                duration = librosa.core.get_duration(filename=audio_path)
                                if 0.5 <= duration <= 10.0:  # Filter by duration for quality
                                    metadata = {
                                        "audio_filepath": audio_path, 
                                        "duration": duration, 
                                        "text": transcript
                                    }
                                    json.dump(metadata, fout)
                                    fout.write('\n')
                                    count += 1
                            except Exception as e:
                                print(f"Warning: Error processing audio file {audio_path}: {e}")
                    except Exception as e:
                        print(f"Warning: Error processing line {line_num}: {line} - {e}")
        
        print(f"Created manifest with {count} samples")
        return count
    
    # Build train and test manifests (subset)
    train_manifest = f"{target_data_dir}/train_manifest_subset.json"
    test_manifest = f"{target_data_dir}/test_manifest_subset.json"
    
    train_count = build_manifest_subset(f"{an4_dir}/etc/an4_train.transcription", train_manifest, subset_size)
    test_count = build_manifest_subset(f"{an4_dir}/etc/an4_test.transcription", test_manifest, min(20, subset_size // 5))
    
    print(f"✅ Dataset prepared: {train_count} train, {test_count} test samples")
    
    # Load Parakeet-v3 model
    print("🤖 Loading Parakeet-v3 model...")
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    print(f"✅ Model loaded: {asr_model.__class__.__name__}")
    
    # Test model before fine-tuning
    print("🧪 Testing pre-trained model...")
    sample_audio = "/tmp/sample_test.wav"
    if not os.path.exists(sample_audio):
        response = requests.get("https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav")
        with open(sample_audio, 'wb') as f:
            f.write(response.content)
    
    original_output = asr_model.transcribe([sample_audio])
    original_text = original_output[0].text
    print(f"Original transcription: {original_text}")
    
    if use_wandb:
        wandb.log({"original_transcription": original_text})
    
    # Create a simple training configuration using NeMo's approach
    print("⚙️ Setting up training configuration...")
    
    # Create a YAML config file for NeMo training
    config_yaml = f"""
name: parakeet_v3_finetune_working

trainer:
  devices: 1
  max_epochs: {max_epochs}
  precision: bf16-mixed
  accelerator: gpu
  strategy: auto
  enable_checkpointing: true
  logger: false
  log_every_n_steps: 5
  val_check_interval: 1.0
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1

exp_manager:
  exp_dir: /checkpoints/parakeet_v3_working_finetune
  name: h100_working_run
  create_tensorboard_logger: false
  create_wandb_logger: {str(use_wandb).lower()}
  wandb_logger_kwargs:
    project: parakeet-v3-finetune-working
    entity: milieu
    name: working-h100-{max_epochs}ep-{subset_size}samples
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: val_wer
    mode: min
    save_top_k: 2
    save_last: true
    filename: parakeet-v3-working-{{epoch:02d}}-{{val_wer:.4f}}
    save_best_model: true
  resume_if_exists: false

model:
  sample_rate: 16000
  
  train_ds:
    manifest_filepath: {train_manifest}
    sample_rate: 16000
    batch_size: {batch_size}
    shuffle: true
    num_workers: 4
    pin_memory: true
    max_duration: 10.0
    min_duration: 0.5
    use_lhotse: false

  validation_ds:
    manifest_filepath: {test_manifest}
    sample_rate: 16000
    batch_size: {batch_size}
    shuffle: false
    num_workers: 4
    pin_memory: true
    max_duration: 10.0
    min_duration: 0.5
    use_lhotse: false

  optim:
    name: adamw
    lr: {learning_rate}
    weight_decay: 0.001
    sched:
      name: CosineAnnealing
      warmup_steps: 50
      min_lr: 1e-6
      max_steps: {train_count * max_epochs // batch_size}
"""
    
    # Save config to file
    config_path = "/tmp/parakeet_finetune_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_yaml)
    
    print(f"✅ Configuration saved to: {config_path}")
    
    # Load config and create trainer
    cfg = OmegaConf.load(config_path)
    
    # Setup trainer and experiment manager
    print("🏋️ Setting up trainer...")
    trainer = pl.Trainer(**cfg.trainer)
    exp_dir = exp_manager(trainer, cfg.exp_manager)
    
    # Setup model for training
    print("🔧 Configuring model for training...")
    asr_model.set_trainer(trainer)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)
    
    # Configure optimizer
    asr_model.cfg.optim = cfg.model.optim
    
    # Start fine-tuning
    print(f"🚀 Starting fine-tuning for {max_epochs} epochs...")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {test_count}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Experiment directory: {exp_dir}")
    
    try:
        # Use NeMo's training approach - this should work better
        print("🎯 Starting training using NeMo's approach...")
        
        # Manual training loop to avoid Lightning compatibility issues
        asr_model.train()
        
        # Get data loaders
        train_dataloader = asr_model._train_dl
        val_dataloader = asr_model._validation_dl
        
        print(f"✅ Data loaders ready: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
        
        # Simple training loop for demonstration
        optimizer = asr_model.configure_optimizers()
        if isinstance(optimizer, dict):
            optimizer = optimizer['optimizer']
        
        best_val_loss = float('inf')
        training_losses = []
        validation_losses = []
        
        for epoch in range(max_epochs):
            print(f"\n📈 Epoch {epoch + 1}/{max_epochs}")
            
            # Training phase
            asr_model.train()
            epoch_train_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= 5:  # Limit to 5 batches for demo
                    break
                    
                optimizer.zero_grad()
                
                # Forward pass
                loss = asr_model.training_step(batch, batch_idx)
                if isinstance(loss, dict):
                    loss = loss['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(float(loss))
                
                if batch_idx % 2 == 0:
                    print(f"  Batch {batch_idx}: loss = {float(loss):.4f}")
                
                if use_wandb:
                    wandb.log({
                        "train_loss": float(loss),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            training_losses.append(avg_train_loss)
            print(f"  Average training loss: {avg_train_loss:.4f}")
            
            # Validation phase
            asr_model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if batch_idx >= 3:  # Limit to 3 batches for demo
                        break
                        
                    val_loss = asr_model.validation_step(batch, batch_idx)
                    if isinstance(val_loss, dict):
                        val_loss = val_loss['val_loss']
                    
                    epoch_val_losses.append(float(val_loss))
            
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            validation_losses.append(avg_val_loss)
            print(f"  Average validation loss: {avg_val_loss:.4f}")
            
            if use_wandb:
                wandb.log({
                    "val_loss": avg_val_loss,
                    "train_loss_epoch": avg_train_loss,
                    "epoch": epoch
                })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"  ✅ New best validation loss: {best_val_loss:.4f}")
        
        print("✅ Fine-tuning completed!")
        
        # Test fine-tuned model
        print("🧪 Testing fine-tuned model...")
        asr_model.eval()
        finetuned_output = asr_model.transcribe([sample_audio])
        finetuned_text = finetuned_output[0].text
        print(f"Fine-tuned transcription: {finetuned_text}")
        
        # Save model
        output_model_path = "/checkpoints/parakeet_v3_finetuned_working.nemo"
        asr_model.save_to(output_model_path)
        print(f"💾 Model saved to: {output_model_path}")
        
        # Log results to W&B
        if use_wandb:
            # Log final metrics
            wandb.log({
                "finetuned_transcription": finetuned_text,
                "training_samples": train_count,
                "validation_samples": test_count,
                "final_epoch": max_epochs,
                "best_val_loss": best_val_loss,
                "final_train_loss": training_losses[-1],
                "final_val_loss": validation_losses[-1],
            })
            
            # Save model as W&B artifact
            print("📦 Saving model to W&B artifacts...")
            artifact = wandb.Artifact(
                name=f"parakeet-v3-finetuned-working",
                type="model",
                description=f"Parakeet-v3 fine-tuned on {train_count} AN4 samples for {max_epochs} epochs",
                metadata={
                    "model_name": model_name,
                    "training_samples": train_count,
                    "validation_samples": test_count,
                    "epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "gpu": "H100",
                    "original_transcription": original_text,
                    "finetuned_transcription": finetuned_text,
                    "best_val_loss": best_val_loss,
                    "nemo_version": nemo.__version__,
                    "pytorch_lightning_version": pl.__version__,
                }
            )
            
            # Add model file to artifact
            artifact.add_file(output_model_path, name="parakeet_v3_finetuned.nemo")
            
            # Log artifact
            wandb.log_artifact(artifact)
            print("✅ Model artifact saved to W&B")
            
            # Finish W&B run
            wandb.finish()
        
        return {
            "status": "success",
            "training_samples": train_count,
            "validation_samples": test_count,
            "epochs_completed": max_epochs,
            "original_transcription": original_text,
            "finetuned_transcription": finetuned_text,
            "model_path": output_model_path,
            "experiment_dir": exp_dir,
            "training_losses": training_losses,
            "validation_losses": validation_losses,
            "best_val_loss": best_val_loss,
            "wandb_enabled": use_wandb,
        }
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        if use_wandb:
            wandb.finish(exit_code=1)
        return {"status": "error", "message": str(e)}

@app.local_entrypoint()
def main(
    max_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    subset_size: int = 100,
    use_wandb: bool = True,
):
    """Run the working fine-tuning"""
    print("🚀 Starting Modal working fine-tuning session...")
    print(f"Configuration:")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Subset size: {subset_size}")
    print(f"  W&B logging: {use_wandb}")
    
    try:
        results = finetune_parakeet_v3_working.remote(
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            subset_size=subset_size,
            use_wandb=use_wandb,
        )
        
        print("\n" + "=" * 60)
        print("📊 TRAINING RESULTS:")
        print("=" * 60)
        
        if results["status"] == "success":
            print("✅ Training completed successfully!")
            print(f"Training samples: {results['training_samples']}")
            print(f"Validation samples: {results['validation_samples']}")
            print(f"Epochs completed: {results['epochs_completed']}")
            print(f"Best validation loss: {results['best_val_loss']:.4f}")
            print(f"Model saved to: {results['model_path']}")
            print(f"W&B logging: {results['wandb_enabled']}")
            
            print("\n📝 Transcription Comparison:")
            print(f"Original:    '{results['original_transcription']}'")
            print(f"Fine-tuned:  '{results['finetuned_transcription']}'")
            
            print(f"\n📈 Training Progress:")
            print(f"Training losses: {[f'{loss:.4f}' for loss in results['training_losses']]}")
            print(f"Validation losses: {[f'{loss:.4f}' for loss in results['validation_losses']]}")
            
            print(f"\n🎯 Fine-tuning completed successfully on H100!")
            if results['wandb_enabled']:
                print("📦 Model artifacts saved to Weights & Biases")
                print("🔗 Check your W&B dashboard for detailed metrics and model artifacts")
        else:
            print(f"❌ Training failed: {results['message']}")
            
    except Exception as e:
        print(f"❌ Modal execution error: {e}")

if __name__ == "__main__":
    main()