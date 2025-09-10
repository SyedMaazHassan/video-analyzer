#!/usr/bin/env python3
"""
Google Colab Training Setup
==========================
Complete setup script for training surgical AI models on Google Colab.
Handles data upload, environment setup, and model training.
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def install_requirements():
    """Install required packages for Colab."""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "opencv-python>=4.8.0",
        "numpy<2.0.0",
        "pandas>=2.0.0",
        "lxml==5.3.0",
        "tqdm>=4.65.0"
    ]
    
    print("🔧 Installing requirements...")
    for req in requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)
    print("✅ Requirements installed!")

def setup_project_structure():
    """Create necessary project directories."""
    dirs = [
        "surgical_ai_system/models/phase_detection",
        "surgical_ai_system/models/instrument_detection", 
        "surgical_ai_system/models/event_detection",
        "surgical_ai_system/models/motion_analysis",
        "surgical_ai_system/training",
        "surgical_ai_system/trained_models",
        "videos",
        "xml_path"
    ]
    
    print("📁 Creating project structure...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Project structure created!")

def clone_and_setup():
    """Clone repository and setup for Colab training."""
    print("🚀 Setting up Surgical AI Training on Google Colab")
    print("="*60)
    
    # Check if running in Colab
    if 'google.colab' not in sys.modules:
        print("⚠️  This script is designed for Google Colab")
        return False
    
    # Install requirements
    install_requirements()
    
    # Clone repository (without large files)
    print("📥 Cloning repository...")
    repo_url = "https://github.com/your-username/video-analyzer.git"  # Update this
    os.system(f"git clone {repo_url} /content/surgical_ai")
    
    # Change to project directory
    os.chdir("/content/surgical_ai")
    
    # Setup structure
    setup_project_structure()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Upload your videos using one of these methods:")
    print("   - Google Drive: Mount drive and copy videos")
    print("   - Upload widget: Upload zip file with videos")
    print("   - External URL: Download from hosted location")
    print("2. Upload corresponding XML annotations")
    print("3. Run training!")
    
    return True

def upload_from_drive():
    """Helper to mount Google Drive and copy videos."""
    from google.colab import drive
    
    print("🗂️  Mounting Google Drive...")
    drive.mount('/content/drive')
    
    print("\nTo copy your data:")
    print("# Copy videos:")
    print("!cp -r '/content/drive/MyDrive/YourVideoFolder/*' /content/surgical_ai/videos/")
    print("# Copy XML annotations:")
    print("!cp -r '/content/drive/MyDrive/YourXMLFolder/*' /content/surgical_ai/xml_path/")

def upload_from_widget():
    """Helper to upload files using Colab widget."""
    from google.colab import files
    
    print("📤 Upload your videos.zip file:")
    uploaded = files.upload()
    
    # Extract first uploaded file
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            print(f"📦 Extracting {filename}...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('/content/surgical_ai/videos/')
            print("✅ Videos extracted!")
            break

def start_training():
    """Start the training process."""
    print("🏥 Starting Surgical AI Training...")
    print("This will train 4 models: Phase, Instrument, Event, Motion")
    
    # Change to project directory
    os.chdir("/content/surgical_ai")
    
    # Run training
    result = subprocess.run([
        sys.executable, 
        "surgical_ai_system/training/practical_master_trainer.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("🎉 Training completed successfully!")
        print("📋 Generated models in: surgical_ai_system/trained_models/")
        return True
    else:
        print("❌ Training failed:")
        print(result.stderr)
        return False

if __name__ == "__main__":
    success = clone_and_setup()
    if success:
        print("\n🔗 Use these helper functions:")
        print("- upload_from_drive() - Copy from Google Drive")
        print("- upload_from_widget() - Upload zip file")
        print("- start_training() - Begin training process")