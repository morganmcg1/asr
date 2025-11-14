#!/usr/bin/env python3
"""
Minimal debugging script for Parakeet-v3 fine-tuning with NeMo 2.5+
This script tests basic functionality with minimal resources.
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path

def create_tiny_dataset():
    """Create a tiny synthetic dataset for testing"""
    print("📝 Creating tiny synthetic dataset...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")
    
    # Create a simple manifest with just a few entries
    manifest_data = [
        {
            "audio_filepath": "/dev/null",  # Dummy path for testing
            "duration": 1.0,
            "text": "hello world"
        },
        {
            "audio_filepath": "/dev/null",
            "duration": 1.5,
            "text": "this is a test"
        },
        {
            "audio_filepath": "/dev/null", 
            "duration": 2.0,
            "text": "testing speech recognition"
        }
    ]
    
    # Write train manifest
    train_manifest = os.path.join(temp_dir, "train_manifest.json")
    with open(train_manifest, 'w') as f:
        for item in manifest_data:
            json.dump(item, f)
            f.write('\n')
    
    # Write validation manifest (same data for simplicity)
    val_manifest = os.path.join(temp_dir, "val_manifest.json")
    with open(val_manifest, 'w') as f:
        for item in manifest_data[:2]:  # Just 2 items for validation
            json.dump(item, f)
            f.write('\n')
    
    print(f"✅ Created manifests:")
    print(f"  Train: {train_manifest}")
    print(f"  Val: {val_manifest}")
    
    return train_manifest, val_manifest

def test_nemo_import():
    """Test basic NeMo imports"""
    print("🔍 Testing NeMo imports...")
    
    try:
        import nemo
        print(f"✅ NeMo version: {nemo.__version__}")
        
        import nemo.collections.asr as nemo_asr
        print("✅ NeMo ASR collections imported")
        
        from nemo.collections.asr.models import ASRModel
        print("✅ ASR Model imported")
        
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test loading a small model"""
    print("🤖 Testing model loading...")
    
    try:
        import nemo.collections.asr as nemo_asr
        
        # Try to load a smaller model first for testing
        print("Loading Parakeet-v3 model...")
        model_name = "nvidia/parakeet-tdt-0.6b-v3"
        
        # Just test the model loading without full initialization
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        print(f"✅ Model loaded: {asr_model.__class__.__name__}")
        
        # Test basic model info
        print(f"Model config keys: {list(asr_model.cfg.keys())[:5]}...")  # Show first 5 keys
        
        return asr_model
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return None

def test_basic_transcription(model):
    """Test basic transcription functionality"""
    print("🎤 Testing basic transcription...")
    
    try:
        # Download a small test audio file
        test_audio = "/tmp/test_audio.wav"
        if not os.path.exists(test_audio):
            print("Downloading test audio...")
            subprocess.run([
                "wget", 
                "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
                "-O", test_audio
            ], check=True)
        
        # Test transcription
        print("Running transcription...")
        output = model.transcribe([test_audio])
        print(f"✅ Transcription result: {output[0].text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return False

def test_data_setup(model, train_manifest, val_manifest):
    """Test data loader setup"""
    print("📊 Testing data setup...")
    
    try:
        # Create minimal config
        from omegaconf import OmegaConf
        
        train_config = OmegaConf.create({
            'manifest_filepath': train_manifest,
            'sample_rate': 16000,
            'batch_size': 1,  # Minimal batch size
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': False,
        })
        
        val_config = OmegaConf.create({
            'manifest_filepath': val_manifest,
            'sample_rate': 16000,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': False,
        })
        
        print("Setting up training data...")
        # This will likely fail due to dummy audio paths, but we can catch it
        try:
            model.setup_training_data(train_config)
            print("✅ Training data setup successful")
        except Exception as e:
            print(f"⚠️ Training data setup failed (expected with dummy data): {e}")
        
        try:
            model.setup_validation_data(val_config)
            print("✅ Validation data setup successful")
        except Exception as e:
            print(f"⚠️ Validation data setup failed (expected with dummy data): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data setup error: {e}")
        return False

def test_trainer_setup():
    """Test PyTorch Lightning trainer setup"""
    print("🏋️ Testing trainer setup...")
    
    try:
        import pytorch_lightning as pl
        
        # Minimal trainer config
        trainer = pl.Trainer(
            max_epochs=1,
            devices=1,
            accelerator='cpu',  # Use CPU for debugging
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        
        print(f"✅ Trainer created: {trainer.__class__.__name__}")
        return trainer
        
    except Exception as e:
        print(f"❌ Trainer setup error: {e}")
        return None

def main():
    """Run all debugging tests"""
    print("🐛 Starting minimal debugging tests for Parakeet-v3 + NeMo 2.5+")
    print("=" * 60)
    
    # Test 1: Basic imports
    if not test_nemo_import():
        print("❌ Basic imports failed - stopping here")
        return
    
    print("\n" + "=" * 60)
    
    # Test 2: Create tiny dataset
    try:
        train_manifest, val_manifest = create_tiny_dataset()
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return
    
    print("\n" + "=" * 60)
    
    # Test 3: Model loading
    model = test_model_loading()
    if model is None:
        print("❌ Model loading failed - stopping here")
        return
    
    print("\n" + "=" * 60)
    
    # Test 4: Basic transcription
    if not test_basic_transcription(model):
        print("⚠️ Transcription test failed, but continuing...")
    
    print("\n" + "=" * 60)
    
    # Test 5: Data setup
    if not test_data_setup(model, train_manifest, val_manifest):
        print("⚠️ Data setup test failed, but continuing...")
    
    print("\n" + "=" * 60)
    
    # Test 6: Trainer setup
    trainer = test_trainer_setup()
    if trainer is None:
        print("❌ Trainer setup failed")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Debugging tests completed!")
    print("\nSummary:")
    print("✅ NeMo imports working")
    print("✅ Model loading working") 
    print("✅ Basic functionality accessible")
    print("\nNext steps:")
    print("1. Fix data loading with real audio files")
    print("2. Test actual training loop")
    print("3. Scale up to full dataset")

if __name__ == "__main__":
    main()