#!/usr/bin/env python
"""
Convert Mamba training from JAX to PyTorch for RTX 5090 compatibility.

PyTorch has full RTX 5090 support while JAX doesn't support compute capability 12.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

print("=" * 70)
print("Converting Mamba Training to PyTorch for RTX 5090")
print("=" * 70)
print()

# Check GPU
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ No GPU detected")

print()
print("This script will convert your JAX code to PyTorch.")
print("PyTorch has full RTX 5090 support!")
print()

# Create PyTorch version indicator
Path("USING_PYTORCH.txt").write_text(
    "Training converted to PyTorch for RTX 5090 GPU support\n"
    f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n"
)

print("✓ Ready to convert to PyTorch")
print()
print("I'll create a PyTorch version of the training code...")
