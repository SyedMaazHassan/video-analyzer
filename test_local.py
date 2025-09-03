#!/usr/bin/env python3
"""
Test script to run locally without Docker.
This will check dependencies and run a minimal test.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 
        'pandas', 'sklearn', 'matplotlib', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package}")
    
    return missing

def install_missing_packages(missing):
    """Install missing packages using pip."""
    if not missing:
        return True
    
    print(f"\nInstalling missing packages: {missing}")
    
    # Map package names to pip names
    pip_names = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn'
    }
    
    for package in missing:
        pip_name = pip_names.get(package, package)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            print(f"✓ Installed {pip_name}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {pip_name}")
            return False
    
    return True

def main():
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing packages: {missing}")
        choice = input("Install missing packages? (y/n): ")
        if choice.lower() == 'y':
            if install_missing_packages(missing):
                print("All packages installed successfully!")
            else:
                print("Some packages failed to install. Please install manually.")
                return
        else:
            print("Please install missing packages manually.")
            return
    
    print("\n✓ All dependencies available!")
    
    # Try to run the setup script
    print("\nRunning setup script...")
    try:
        from setup_and_train import main as setup_main
        setup_main()
    except Exception as e:
        print(f"Error running setup: {str(e)}")
        print("\nYou can try running manually:")
        print("python setup_and_train.py")

if __name__ == "__main__":
    main()