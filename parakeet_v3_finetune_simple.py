#!/usr/bin/env python3
"""
Simple Parakeet-v3 fine-tuning script for NeMo 2.5+
This script demonstrates how to fine-tune the Parakeet-v3 model using the updated NeMo API.
"""

import os
import json
import glob
import subprocess
import librosa
from pathlib import Path

# NeMo imports
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

def prepare_an4_dataset(data_dir="/tmp/an4_data"):
    """Download and prepare the AN4 dataset"""
    print("📥 Preparing AN4 dataset...")
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset if not exists
    an4_tar = f"{data_dir}/an4_sphere.tar.gz"
    if not os.path.exists(an4_tar):
        print("Downloading AN4 dataset...")
        subprocess.run([
            "wget", 
            "https://dldata-public.s3.us-east-2.amazonaws.com/an4_sphere.tar.gz",
            "-O", an4_tar
        ], check=True)
    
    # Extract dataset
    an4_dir = f"{data_dir}/an4"
    if not os.path.exists(an4_dir):
        print("Extracting dataset...")
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

def main():
    print(f"🚀 Starting Parakeet-v3 fine-tuning")
    print(f"NeMo version: {nemo.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Prepare dataset
    train_manifest, test_manifest = prepare_an4_dataset()
    
    # Load Parakeet-v3 model
    print("🤖 Loading Parakeet-v3 model...")
    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    print(f"✅ Model loaded: {asr_model.__class__.__name__}")
    
    # Test model before fine-tuning
    print("🧪 Testing pre-trained model...")
    sample_audio = "/tmp/sample_test.wav"
    if not os.path.exists(sample_audio):
        subprocess.run([
            "wget", 
            "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
            "-O", sample_audio
        ], check=True)
    
    original_output = asr_model.transcribe([sample_audio])
    print(f"Original transcription: {original_output[0].text}")
    
    # Create a simple training configuration
    print("⚙️ Setting up training configuration...")
    
    # Update model's train and validation data configuration
    asr_model.cfg.train_ds.manifest_filepath = train_manifest
    asr_model.cfg.train_ds.batch_size = 4
    asr_model.cfg.train_ds.num_workers = 2
    
    asr_model.cfg.validation_ds.manifest_filepath = test_manifest
    asr_model.cfg.validation_ds.batch_size = 4
    asr_model.cfg.validation_ds.num_workers = 2
    
    # Setup optimizer configuration
    asr_model.cfg.optim.name = "adamw"
    asr_model.cfg.optim.lr = 1e-4
    asr_model.cfg.optim.weight_decay = 0.001
    
    # Setup trainer
    trainer_cfg = {
        'devices': 1 if torch.cuda.is_available() else 'auto',
        'max_epochs': 2,
        'precision': 'bf16-mixed' if torch.cuda.is_available() else 32,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'enable_checkpointing': True,
        'logger': False,  # Disable default logger
        'log_every_n_steps': 10,
        'val_check_interval': 1.0,
    }
    
    trainer = pl.Trainer(**trainer_cfg)
    
    # Setup experiment manager
    exp_manager_cfg = {
        'exp_dir': '/tmp/parakeet_checkpoints',
        'name': 'parakeet_v3_finetune',
        'version': 'simple_test',
        'create_tensorboard_logger': False,
        'create_wandb_logger': False,
        'create_checkpoint_callback': True,
        'checkpoint_callback_params': {
            'monitor': 'val_wer',
            'mode': 'min',
            'save_top_k': 1,
            'save_last': True,
        }
    }
    
    exp_dir = exp_manager(trainer, OmegaConf.create(exp_manager_cfg))
    
    # Setup model for training
    asr_model.set_trainer(trainer)
    asr_model.setup_training_data(asr_model.cfg.train_ds)
    asr_model.setup_validation_data(asr_model.cfg.validation_ds)
    
    print("🚀 Starting fine-tuning...")
    print(f"Experiment directory: {exp_dir}")
    
    try:
        # Start training
        trainer.fit(asr_model)
        print("✅ Fine-tuning completed!")
        
        # Test fine-tuned model
        finetuned_output = asr_model.transcribe([sample_audio])
        print(f"Fine-tuned transcription: {finetuned_output[0].text}")
        
        # Save model
        output_model_path = "/tmp/parakeet_v3_finetuned.nemo"
        asr_model.save_to(output_model_path)
        print(f"💾 Model saved to: {output_model_path}")
        
        # Comparison
        print("\n📊 Comparison:")
        print(f"Original:    {original_output[0].text}")
        print(f"Fine-tuned:  {finetuned_output[0].text}")
        
        return {
            "status": "success",
            "original_transcription": original_output[0].text,
            "finetuned_transcription": finetuned_output[0].text,
            "model_path": output_model_path,
            "experiment_dir": exp_dir,
        }
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    main()