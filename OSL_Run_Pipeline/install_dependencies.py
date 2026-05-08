#!/usr/bin/env python3
"""
Install dependencies for real-time sign language recognition

This script checks and installs required packages for webcam_inference_osl.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✓ {description} - SUCCESS")
            return True
        else:
            print(f"✗ {description} - FAILED")
            return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False


def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("\n" + "="*60)
    print("SIGN LANGUAGE RECOGNITION - DEPENDENCY INSTALLER")
    print("="*60)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        sys.exit(1)
    
    # Check current packages
    print(f"\n{'='*60}")
    print("Checking installed packages...")
    print(f"{'='*60}")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy',
    }
    
    installed = {}
    for pkg, name in packages.items():
        is_installed = check_package(pkg)
        status = "✓ installed" if is_installed else "✗ NOT installed"
        print(f"  {name:15s} {status}")
        installed[pkg] = is_installed
    
    # Check if all installed
    if all(installed.values()):
        print("\n✅ All dependencies are already installed!")
        return
    
    # Prompt user
    print(f"\n{'='*60}")
    response = input("Install missing packages? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Installation cancelled.")
        return
    
    # Install packages
    print(f"\n{'='*60}")
    print("INSTALLING PACKAGES")
    print(f"{'='*60}")
    
    # OpenCV
    if not installed['cv2']:
        run_command(
            [sys.executable, "-m", "pip", "install", "opencv-python"],
            "Installing OpenCV"
        )
    
    # MediaPipe
    if not installed['mediapipe']:
        run_command(
            [sys.executable, "-m", "pip", "install", "mediapipe"],
            "Installing MediaPipe"
        )
    
    # NumPy
    if not installed['numpy']:
        run_command(
            [sys.executable, "-m", "pip", "install", "numpy"],
            "Installing NumPy"
        )
    
    # PyTorch - special handling
    if not installed['torch']:
        print(f"\n{'='*60}")
        print("PyTorch Installation Options")
        print(f"{'='*60}")
        print("1. GPU (CUDA 11.8) - Recommended for speed")
        print("2. GPU (CUDA 12.1) - For newer GPUs")
        print("3. CPU only - Works on all machines but slower")
        print("4. Skip (manual installation)")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            run_command(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"],
                "Installing PyTorch + CUDA 11.8"
            )
        elif choice == '2':
            run_command(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
                "Installing PyTorch + CUDA 12.1"
            )
        elif choice == '3':
            run_command(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision"],
                "Installing PyTorch (CPU)"
            )
        else:
            print("Skipping PyTorch installation. Install manually:")
            print("  Visit: https://pytorch.org/get-started/locally/")
    
    # Final check
    print(f"\n{'='*60}")
    print("INSTALLATION COMPLETE")
    print(f"{'='*60}")
    print("\nAll dependencies installed! You can now run:")
    print("  python webcam_inference_osl.py")
    print()


if __name__ == "__main__":
    main()
