"""
Modal script for fine-tuning Parakeet-v3 ASR model with NeMo 2.5+
This script runs the fine-tuning process on Modal's GPU infrastructure.
"""

import modal
import os

# Create Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("sox", "libsndfile1", "ffmpeg", "libsox-fmt-mp3", "jq", "wget", "git")
    .pip_install([
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "nemo_toolkit[asr]>=2.5.0",
        "text-unidecode",
        "matplotlib>=3.3.2",
        "librosa",
        "soundfile",
        "huggingface-hub>=0.23.2",
        "omegaconf",
        "pytorch-lightning",
        "tensorboard",
        "wandb",  # For experiment tracking
    ])
)

app = modal.App("parakeet-v3-finetune")

# Define volumes for persistent storage
data_volume = modal.Volume.from_name("parakeet-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("parakeet-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Use H100 for optimal performance
    timeout=60*60*4,  # 4 hours timeout
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoints_volume,
    },
    env={
        "WANDB_API_KEY": "612d5d4dd0d1d56678220c60c2bb3ef957ee983d",
        "WANDB_PROJECT": "asr",
        "WANDB_ENTITY": "milieu",
    },
)
def finetune_parakeet_v3(
    max_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    use_wandb: bool = False,
):
    """
    Fine-tune Parakeet-v3 model on AN4 dataset
    
    Args:
        max_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for fine-tuning
        use_wandb: Whether to use Weights & Biases for logging
    """
    import json
    import glob
    import subprocess
    import torch
    import librosa
    from pathlib import Path
    
    # NeMo imports
    import nemo
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import ASRModel
    from nemo.utils.exp_manager import exp_manager
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    
    print(f"🚀 Starting Parakeet-v3 fine-tuning on {torch.cuda.get_device_name(0)}")
    print(f"NeMo version: {nemo.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Setup directories
    data_dir = "/data"
    checkpoints_dir = "/checkpoints"
    
    # Initialize W&B if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="parakeet-v3-finetune",
                config={
                    "max_epochs": max_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model": "nvidia/parakeet-tdt-0.6b-v3",
                }
            )
            print("✅ Weights & Biases initialized")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            use_wandb = False
    
    # Download and prepare AN4 dataset
    def prepare_an4_dataset():
        print("📥 Preparing AN4 dataset...")
        
        # Download dataset if not exists
        an4_tar = f"{data_dir}/an4_sphere.tar.gz"
        if not os.path.exists(an4_tar):
            subprocess.run([
                "wget", 
                "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz",
                "-O", an4_tar
            ], check=True)
        
        # Extract dataset
        an4_dir = f"{data_dir}/an4"
        if not os.path.exists(an4_dir):
            subprocess.run(["tar", "-xzf", an4_tar, "-C", data_dir], check=True)
        
        # Convert SPH to WAV and create manifests
        target_data_dir = f"{data_dir}/an4_converted"
        target_wavs_dir = f"{target_data_dir}/wavs"
        os.makedirs(target_wavs_dir, exist_ok=True)
        
        # Convert SPH files to WAV
        sph_files = glob.glob(f"{an4_dir}/**/*.sph", recursive=True)
        print(f"Converting {len(sph_files)} SPH files to WAV...")
        
        for sph_path in sph_files:
            wav_path = os.path.join(target_wavs_dir, 
                                  os.path.splitext(os.path.basename(sph_path))[0] + '.wav')
            if not os.path.exists(wav_path):
                subprocess.run(["sox", sph_path, wav_path], check=True)
        
        # Create manifest files
        def build_manifest(transcripts_path, manifest_path):
            with open(transcripts_path, 'r') as fin:
                with open(manifest_path, 'w') as fout:
                    for line_num, line in enumerate(fin, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            # Find the transcript and file ID
                            paren_idx = line.rfind('(')
                            if paren_idx == -1:
                                print(f"Warning: Skipping line {line_num}, no parentheses found: {line}")
                                continue
                                
                            transcript = line[:paren_idx].strip().lower()
                            transcript = transcript.replace('<s>', '').replace('</s>', '').strip()
                            
                            file_id = line[paren_idx + 1:].rstrip(')')
                            audio_path = os.path.join(target_wavs_dir, file_id + '.wav')
                            
                            if os.path.exists(audio_path):
                                try:
                                    duration = librosa.core.get_duration(filename=audio_path)
                                    metadata = {
                                        "audio_filepath": audio_path, 
                                        "duration": duration, 
                                        "text": transcript
                                    }
                                    json.dump(metadata, fout)
                                    fout.write('\n')
                                except Exception as e:
                                    print(f"Warning: Error processing audio file {audio_path}: {e}")
                            else:
                                print(f"Warning: Audio file not found: {audio_path}")
                        except Exception as e:
                            print(f"Warning: Error processing line {line_num}: {line} - {e}")
        
        # Build train and test manifests
        train_manifest = f"{target_data_dir}/train_manifest.json"
        test_manifest = f"{target_data_dir}/test_manifest.json"
        
        build_manifest(f"{an4_dir}/etc/an4_train.transcription", train_manifest)
        build_manifest(f"{an4_dir}/etc/an4_test.transcription", test_manifest)
        
        print(f"✅ Dataset prepared successfully!")
        print(f"Train manifest: {train_manifest}")
        print(f"Test manifest: {test_manifest}")
        
        return train_manifest, test_manifest
    
    # Prepare dataset
    train_manifest, test_manifest = prepare_an4_dataset()
    
    # Load Parakeet-v3 model
    print("🤖 Loading Parakeet-v3 model...")
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    print(f"✅ Model loaded: {asr_model.__class__.__name__}")
    
    # Test model before fine-tuning
    print("🧪 Testing pre-trained model...")
    sample_audio = f"{data_dir}/sample_test.wav"
    if not os.path.exists(sample_audio):
        subprocess.run([
            "wget", 
            "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
            "-O", sample_audio
        ], check=True)
    
    original_output = asr_model.transcribe([sample_audio])
    print(f"Original transcription: {original_output[0].text}")
    
    # Create fine-tuning configuration
    cfg = OmegaConf.create({
        'model': {
            'train_ds': {
                'manifest_filepath': train_manifest,
                'sample_rate': 16000,
                'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 4,
                'pin_memory': True,
                'use_start_end_token': False,
            },
            'validation_ds': {
                'manifest_filepath': test_manifest,
                'sample_rate': 16000,
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 4,
                'pin_memory': True,
                'use_start_end_token': False,
            },
            'optim': {
                'name': 'adamw',
                'lr': learning_rate,
                'weight_decay': 0.001,
                'sched': {
                    'name': 'CosineAnnealing',
                    'warmup_steps': 100,
                    'min_lr': 1e-6,
                }
            }
        },
        'trainer': {
            'devices': 1,
            'max_epochs': max_epochs,
            'precision': 'bf16-mixed',
            'accelerator': 'gpu',
            'strategy': 'auto',
            'enable_checkpointing': True,
            'logger': False,  # Disable default logger to avoid conflicts
            'log_every_n_steps': 10,
            'val_check_interval': 1.0,
            'gradient_clip_val': 1.0,
        },
        'exp_manager': {
            'exp_dir': checkpoints_dir,
            'name': 'parakeet_v3_finetune',
            'version': 'modal_run',
            'use_datetime_version': False,
            'create_tensorboard_logger': False,  # Disable to avoid conflicts
            'create_wandb_logger': use_wandb,
            'wandb_logger_kwargs': {
                'project': 'asr',
                'entity': 'milieu',
                'name': f'parakeet-v3-finetune-{max_epochs}epochs',
            } if use_wandb else {},
            'create_checkpoint_callback': True,
            'checkpoint_callback_params': {
                'monitor': 'val_wer',
                'mode': 'min',
                'save_top_k': 3,
                'save_last': True,
            }
        }
    })
    
    # Setup trainer and experiment manager
    print("🏋️ Setting up trainer...")
    trainer = pl.Trainer(**cfg.trainer)
    exp_dir = exp_manager(trainer, cfg.exp_manager)
    
    # Update model configuration with optimizer settings
    asr_model.cfg.optim = cfg.model.optim
    
    # Setup model for training
    asr_model.set_trainer(trainer)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)
    
    # Start fine-tuning
    print(f"🚀 Starting fine-tuning for {max_epochs} epochs...")
    print(f"Experiment directory: {exp_dir}")
    
    # Use the trainer's fit method with proper model setup
    trainer.fit(asr_model)
    
    print("✅ Fine-tuning completed!")
    
    # Load best checkpoint and test
    checkpoint_files = glob.glob(f"{exp_dir}/checkpoints/*.ckpt")
    if checkpoint_files:
        best_checkpoint = None
        for ckpt in checkpoint_files:
            if "last" not in ckpt:
                best_checkpoint = ckpt
                break
        
        if best_checkpoint is None:
            best_checkpoint = checkpoint_files[0]
        
        print(f"📁 Loading best checkpoint: {best_checkpoint}")
        finetuned_model = ASRModel.load_from_checkpoint(best_checkpoint)
        
        # Test fine-tuned model
        finetuned_output = finetuned_model.transcribe([sample_audio])
        print(f"Fine-tuned transcription: {finetuned_output[0].text}")
        
        # Save model in NeMo format
        output_model_path = f"{checkpoints_dir}/parakeet_v3_finetuned.nemo"
        finetuned_model.save_to(output_model_path)
        print(f"💾 Model saved to: {output_model_path}")
        
        # Comparison
        print("\\n📊 Comparison:")
        print(f"Original:    {original_output[0].text}")
        print(f"Fine-tuned:  {finetuned_output[0].text}")
        
        results = {
            "status": "success",
            "original_transcription": original_output[0].text,
            "finetuned_transcription": finetuned_output[0].text,
            "model_path": output_model_path,
            "checkpoint_path": best_checkpoint,
            "experiment_dir": exp_dir,
        }
        
        if use_wandb:
            wandb.log({
                "original_transcription": original_output[0].text,
                "finetuned_transcription": finetuned_output[0].text,
            })
            wandb.finish()
        
        return results
    
    else:
        print("⚠️ No checkpoints found!")
        return {"status": "error", "message": "No checkpoints found"}

@app.function(
    image=image,
    gpu="H100",
    timeout=60*30,  # 30 minutes for evaluation
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoints_volume,
    },
)
def evaluate_model(model_path: str = "/checkpoints/parakeet_v3_finetuned.nemo"):
    """
    Evaluate the fine-tuned model on test data
    """
    import json
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import ASRModel
    
    print(f"🧪 Evaluating model: {model_path}")
    
    if not os.path.exists(model_path):
        return {"status": "error", "message": f"Model not found: {model_path}"}
    
    # Load model
    model = ASRModel.restore_from(model_path)
    print(f"✅ Model loaded successfully")
    
    # Load test manifest
    test_manifest = "/data/an4_converted/test_manifest.json"
    if not os.path.exists(test_manifest):
        return {"status": "error", "message": f"Test manifest not found: {test_manifest}"}
    
    # Read test data
    test_files = []
    test_texts = []
    
    with open(test_manifest, 'r') as f:
        for line in f:
            data = json.loads(line)
            test_files.append(data['audio_filepath'])
            test_texts.append(data['text'])
    
    print(f"📊 Evaluating on {len(test_files)} test files...")
    
    # Transcribe test files (limit to first 20 for demo)
    test_subset = test_files[:20]
    predictions = model.transcribe(test_subset)
    
    # Calculate simple accuracy metrics
    correct_words = 0
    total_words = 0
    
    results = []
    for i, (pred, true_text) in enumerate(zip(predictions, test_texts[:20])):
        pred_text = pred.text.lower().strip()
        true_text = true_text.lower().strip()
        
        pred_words = pred_text.split()
        true_words = true_text.split()
        
        # Simple word-level accuracy
        common_words = set(pred_words) & set(true_words)
        correct_words += len(common_words)
        total_words += len(true_words)
        
        results.append({
            "file": test_subset[i],
            "ground_truth": true_text,
            "prediction": pred_text,
            "word_accuracy": len(common_words) / max(len(true_words), 1)
        })
    
    overall_accuracy = correct_words / max(total_words, 1)
    
    print(f"📈 Overall word accuracy: {overall_accuracy:.3f}")
    print("\\n🔍 Sample results:")
    for result in results[:5]:
        print(f"  GT: {result['ground_truth']}")
        print(f"  Pred: {result['prediction']}")
        print(f"  Acc: {result['word_accuracy']:.3f}")
        print()
    
    return {
        "status": "success",
        "overall_accuracy": overall_accuracy,
        "num_samples": len(results),
        "sample_results": results[:10],  # Return first 10 results
    }

@app.local_entrypoint()
def main(
    max_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    use_wandb: bool = False,
    evaluate_only: bool = False,
):
    """
    Main entrypoint for Parakeet-v3 fine-tuning
    
    Args:
        max_epochs: Number of training epochs
        batch_size: Training batch size  
        learning_rate: Learning rate for fine-tuning
        use_wandb: Whether to use Weights & Biases for logging
        evaluate_only: If True, only run evaluation on existing model
    """
    if evaluate_only:
        print("🧪 Running evaluation only...")
        results = evaluate_model.remote()
        print("📊 Evaluation Results:")
        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
            print(f"Samples Evaluated: {results['num_samples']}")
        else:
            print(f"Error: {results['message']}")
    else:
        print("🚀 Starting fine-tuning...")
        results = finetune_parakeet_v3.remote(
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_wandb=use_wandb,
        )
        
        print("\\n🎉 Fine-tuning Results:")
        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"Model saved to: {results['model_path']}")
            print(f"Experiment dir: {results['experiment_dir']}")
            print("\\n📝 Transcription Comparison:")
            print(f"Original:    {results['original_transcription']}")
            print(f"Fine-tuned:  {results['finetuned_transcription']}")
            
            # Run evaluation
            print("\\n🧪 Running evaluation...")
            eval_results = evaluate_model.remote(results['model_path'])
            if eval_results['status'] == 'success':
                print(f"📊 Evaluation Accuracy: {eval_results['overall_accuracy']:.3f}")
        else:
            print(f"❌ Error: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()