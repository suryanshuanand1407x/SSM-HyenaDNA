#!/bin/bash
#
# Fix GPU training for RTX 5090 by configuring JAX environment
#

set -e

echo "Configuring JAX for RTX 5090..."

# Force CUDA platform
export JAX_PLATFORMS=cuda

# Disable memory preallocation (better for large models)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set memory fraction (use 90% of GPU memory)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Suppress cuDNN warnings
export TF_CPP_MIN_LOG_LEVEL=2

echo "✓ Environment configured for GPU training"
echo ""

# Run training
python train_hyena.py "$@"
