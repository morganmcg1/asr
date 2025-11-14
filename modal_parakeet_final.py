#!/usr/bin/env python3
"""
Final working Parakeet-v3 fine-tuning script for Modal with H100
Fixed W&B project name and proper tagging with model name.
"""

import modal
import os

# Enhanced image with all dependencies including CUDA toolkit for numba
image = (
    modal.Image.micromamba(python_version="3.12")
    .apt_install("sox", "libsndfile1", "ffmpeg", "wget", "build-essential", "gcc", "g++")
    .micromamba_install("cudatoolkit", channels=["conda-forge"])  # Install CUDA toolkit for numba NVVM support
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


app = modal.App("parakeet-v3-final-finetune")

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
def finetune_parakeet_v3_final(
    max_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    subset_size: int = 100,
    use_wandb: bool = True,
):
    """Fine-tune Parakeet-v3 with proper W&B logging to 'asr' project"""
    
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
    from nemo import lightning as nl  # NeMo 2.0 uses nemo.lightning.Trainer
    from omegaconf import OmegaConf
    
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    model_short_name = "parakeet-v3"
    
    print("🚀 Starting Final Parakeet-v3 fine-tuning on NVIDIA H100")
    print(f"NeMo version: {nemo.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_name}")
    
    # Initialize Weights & Biases - ALWAYS use "asr" project
    if use_wandb:
        import wandb
        
        # Get API key from environment (set via Modal secret)
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            
            # Initialize run with proper project and tags
            run_name = f"{model_short_name}-h100-{max_epochs}ep-{subset_size}samples"
            wandb.init(
                project="asr",  # ALWAYS use "asr" project
                entity="milieu",
                name=run_name,
                tags=[
                    model_short_name,
                    "parakeet-v3",
                    "nvidia/parakeet-tdt-0.6b-v3",
                    "h100",
                    "fine-tuning",
                    "nemo-2.5.3",
                    f"{max_epochs}-epochs",
                    f"{subset_size}-samples"
                ],
                config={
                    "model": model_name,
                    "model_short_name": model_short_name,
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "subset_size": subset_size,
                    "gpu": "H100",
                    "nemo_version": nemo.__version__,
                    "pytorch_version": torch.__version__,
    #                     "pytorch_lightning_version": pl.__version__,
                    "dataset": "AN4",
                    "training_approach": "manual_loop",
                }
            )
            print(f"✅ Weights & Biases initialized - Project: asr, Run: {run_name}")
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
    print(f"🤖 Loading {model_name} model...")
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
name: parakeet_v3_final_finetune

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
  exp_dir: /checkpoints/parakeet_v3_final_finetune
  name: h100_final_run
  create_tensorboard_logger: false
  create_wandb_logger: {str(use_wandb).lower()}
  wandb_logger_kwargs:
    project: asr
    entity: milieu
    name: {model_short_name}-h100-{max_epochs}ep-{subset_size}samples
    tags: ["{model_short_name}", "parakeet-v3", "h100", "fine-tuning"]
  create_checkpoint_callback: false  # Let trainer handle checkpointing
  checkpoint_callback_params:
    monitor: val_wer
    mode: min
    save_top_k: 2
    save_last: true
    filename: parakeet-v3-final-{{epoch:02d}}-{{val_wer:.4f}}
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

    # CRITICAL: Follow official NeMo 2.0 training pattern
    # 1. Create trainer FIRST
    print("🏋️ Setting up trainer...")
    trainer = nl.Trainer(**cfg.trainer)
    
    # 2. Setup exp_manager BEFORE touching the model
    exp_dir = exp_manager(trainer, cfg.exp_manager)

    # 3. Attach trainer to model (NeMo pattern)
    print("🔧 Attaching trainer to model...")
    asr_model.set_trainer(trainer)
    
    # 4. Setup dataloaders (calls model methods)
    print("📊 Setting up training and validation data...")
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)

    # 5. Setup optimization
    print("⚙️ Setting up optimization...")
    asr_model.setup_optimization(cfg.model.optim)

    # Start fine-tuning
    print(f"🚀 Starting fine-tuning for {max_epochs} epochs...")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {test_count}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Experiment directory: {exp_dir}")

    try:
        # 6. Train using trainer.fit() - NeMo handles dataloaders internally
        print("🎯 Starting training with trainer.fit()...")
        trainer.fit(asr_model)

        # Extract loss metrics from trainer if available
        print("\n✅ Training completed!")
        if hasattr(trainer, 'callback_metrics'):
            final_train_loss = trainer.callback_metrics.get('train_loss_epoch', trainer.callback_metrics.get('train_loss', 'N/A'))
            final_val_loss = trainer.callback_metrics.get('val_loss', 'N/A')
            print(f"\n📊 Final Metrics:")
            print(f"  Train Loss: {final_train_loss}")
            print(f"  Val Loss: {final_val_loss}")
            
        print(f"\n📝 Training Summary:")
        print(f"Model: {model_name}")
        print(f"Training samples: {subset_size}")
        print(f"Validation samples: {subset_size // 5}")
        print(f"Epochs: {max_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Experiment directory: {exp_dir}")
        
        # 🧪 Test the fine-tuned model with comprehensive inference
        print("\n🧪 Testing fine-tuned model with comprehensive inference...")
        
        # Get the model variable name dynamically
        import re
        model_var = [k for k, v in locals().items() if hasattr(v, 'transcribe') and hasattr(v, 'setup_training_data')]
        
        if model_var:
            model_obj = locals()[model_var[0]]
            # Load a few test audio files
            test_audio_files = sorted(glob.glob("/tmp/an4_test/*.wav"))[:5]
            
            if test_audio_files:
                print(f"\nTranscribing {len(test_audio_files)} test samples...")
                transcriptions = model_obj.transcribe(test_audio_files)
                
                print("\n📝 Transcription Results:")
                print("=" * 80)
                for i, (audio_file, transcription) in enumerate(zip(test_audio_files, transcriptions), 1):
                    filename = Path(audio_file).name
                    print(f"\n[{i}] {filename}:")
                    print(f"    Transcription: '{transcription}'")
                print("=" * 80)
            else:
                print("⚠️  No test files found for inference testing")
        
        print(f"\n🎯 Fine-tuning completed successfully on H100!")
        print("📦 Model artifacts and checkpoints saved")
        print(f"🔗 Check your W&B dashboard: https://wandb.ai/milieu/parakeet-v3-<secret_hidden>-finetune")
        print(f"🏷️ Model: {model_name}, parakeet-v3, h100, fine-tuning")
            
    except Exception as e:
        print(f"❌ Modal execution error: {e}")

if __name__ == "__main__":
    main()