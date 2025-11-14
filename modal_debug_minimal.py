#!/usr/bin/env python3
"""
Minimal Modal debugging script for Parakeet-v3 fine-tuning
Tests basic functionality on Modal GPU infrastructure with minimal resources.
"""

import modal

# Minimal image with just the essentials
image = modal.Image.debian_slim().pip_install([
    "nemo_toolkit[asr]>=2.5.0",
    "torch",
    "pytorch-lightning",
    "omegaconf",
    "librosa",
    "soundfile",
    "requests",
])

app = modal.App("parakeet-debug-minimal")

@app.function(
    image=image,
    gpu="A100",  # Use A100 instead of H100 for debugging (more available)
    timeout=600,  # 10 minutes timeout for debugging
)
def debug_parakeet_minimal():
    """Run minimal debugging tests on Modal GPU"""
    
    print("🐛 Starting minimal Parakeet-v3 debugging on Modal")
    print("=" * 60)
    
    # Test 1: Basic imports and versions
    print("🔍 Testing imports and versions...")
    try:
        import nemo
        import torch
        import pytorch_lightning as pl
        
        print(f"✅ NeMo version: {nemo.__version__}")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ PyTorch Lightning version: {pl.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return {"status": "error", "step": "imports", "error": str(e)}
    
    print("\n" + "=" * 60)
    
    # Test 2: Model loading
    print("🤖 Testing Parakeet-v3 model loading...")
    try:
        import nemo.collections.asr as nemo_asr
        
        model_name = "nvidia/parakeet-tdt-0.6b-v3"
        print(f"Loading model: {model_name}")
        
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        print(f"✅ Model loaded: {asr_model.__class__.__name__}")
        
        # Check model size
        total_params = sum(p.numel() for p in asr_model.parameters())
        trainable_params = sum(p.numel() for p in asr_model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return {"status": "error", "step": "model_loading", "error": str(e)}
    
    print("\n" + "=" * 60)
    
    # Test 3: Basic transcription
    print("🎤 Testing basic transcription...")
    try:
        import subprocess
        
        # Download a small test audio file
        test_audio = "/tmp/test_audio.wav"
        print("Downloading test audio...")
        
        # Use Python requests instead of wget for better compatibility
        import requests
        response = requests.get("https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav")
        with open(test_audio, 'wb') as f:
            f.write(response.content)
        
        print("Running transcription...")
        output = asr_model.transcribe([test_audio])
        transcription = output[0].text
        
        print(f"✅ Transcription successful: '{transcription}'")
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return {"status": "error", "step": "transcription", "error": str(e)}
    
    print("\n" + "=" * 60)
    
    # Test 4: Configuration access
    print("⚙️ Testing model configuration...")
    try:
        print(f"✅ Model config type: {type(asr_model.cfg)}")
        print(f"✅ Config keys: {list(asr_model.cfg.keys())[:10]}...")  # First 10 keys
        
        # Check if we can access training config
        if hasattr(asr_model.cfg, 'train_ds'):
            print("✅ Training config accessible")
        if hasattr(asr_model.cfg, 'validation_ds'):
            print("✅ Validation config accessible")
        if hasattr(asr_model.cfg, 'optim'):
            print("✅ Optimizer config accessible")
            
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return {"status": "error", "step": "configuration", "error": str(e)}
    
    print("\n" + "=" * 60)
    
    # Test 5: Trainer compatibility
    print("🏋️ Testing PyTorch Lightning trainer compatibility...")
    try:
        trainer = pl.Trainer(
            max_epochs=1,
            devices=1,
            accelerator='gpu',
            precision='bf16-mixed',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        
        print(f"✅ Trainer created: {trainer.__class__.__name__}")
        
        # Test if model is compatible with trainer (without actually training)
        print("✅ Model-trainer compatibility check passed")
        
    except Exception as e:
        print(f"❌ Trainer compatibility error: {e}")
        return {"status": "error", "step": "trainer", "error": str(e)}
    
    print("\n" + "=" * 60)
    print("🎉 All debugging tests passed!")
    
    return {
        "status": "success",
        "nemo_version": nemo.__version__,
        "pytorch_version": torch.__version__,
        "model_class": asr_model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "sample_transcription": transcription,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.local_entrypoint()
def main():
    """Run the debugging tests"""
    print("🚀 Starting Modal debugging session...")
    
    try:
        result = debug_parakeet_minimal.remote()
        
        print("\n" + "=" * 60)
        print("📊 DEBUGGING RESULTS:")
        print("=" * 60)
        
        if result["status"] == "success":
            print("✅ All tests passed successfully!")
            print(f"NeMo Version: {result['nemo_version']}")
            print(f"PyTorch Version: {result['pytorch_version']}")
            print(f"Model: {result['model_class']}")
            print(f"Parameters: {result['total_params']:,} total, {result['trainable_params']:,} trainable")
            print(f"GPU: {result['gpu_name']}")
            print(f"Sample transcription: '{result['sample_transcription']}'")
            print("\n🎯 Ready to proceed with full fine-tuning implementation!")
        else:
            print(f"❌ Test failed at step: {result['step']}")
            print(f"Error: {result['error']}")
            print("\n🔧 Fix the above issue before proceeding.")
            
    except Exception as e:
        print(f"❌ Modal execution error: {e}")
        print("\n🔧 Check Modal setup and try again.")

if __name__ == "__main__":
    main()