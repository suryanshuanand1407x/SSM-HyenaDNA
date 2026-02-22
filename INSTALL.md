# Installation Guide - RTX 5090

## Prerequisites

- **Hardware**: NVIDIA RTX 5090 (or any CUDA-capable GPU)
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **CUDA**: 12.0 or higher
- **Python**: 3.10 or higher
- **Disk Space**: ~20GB (for dependencies, datasets, checkpoints)

## Quick Install (RTX 5090 Machine)

```bash
# Navigate to project
cd /path/to/p2

# Install all dependencies (automatic CUDA detection)
bash install_requirements.sh

# Verify GPU setup
python setup_gpu.py
```

**That's it!** If verification passes, you're ready to train.

## Detailed Installation Steps

### Step 1: Python Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n hyena python=3.10 -y
conda activate hyena
```

**Option B: Using venv**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Automatic installation (detects GPU)
bash install_requirements.sh
```

**Manual Installation** (if script fails):

```bash
# Upgrade pip
pip install --upgrade pip

# Install JAX with CUDA 12 (for RTX 5090)
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check CUDA
nvidia-smi

# Check JAX sees GPU
python -c "import jax; print(jax.devices())"

# Full verification
python setup_gpu.py
```

Expected output:
```
✓ Found 1 GPU(s)
✓ GPU computation test passed
✓ bfloat16 computation works
✓ All checks passed!
```

## Dependency List

### Core ML Stack
- **JAX** (0.4.25+): GPU-accelerated array computing
- **Flax** (0.8.0+): Neural network library
- **Optax** (0.1.9+): Optimization algorithms

### HuggingFace Ecosystem
- **transformers** (4.37.0+): Pre-trained HyenaDNA models
- **datasets** (2.18.0+): Genomic dataset loading
- **huggingface-hub** (0.20.0+): Model hub access
- **tokenizers** (0.15.0+): Fast tokenization
- **safetensors** (0.4.0+): Safe model loading
- **accelerate** (0.26.0+): Training acceleration

### Utilities
- **numpy** (1.26.0+): Array operations
- **scipy** (1.11.0+): Scientific computing
- **matplotlib** (3.8.0+): Visualization
- **seaborn** (0.13.0+): Statistical plots
- **tqdm** (4.66.0+): Progress bars

## Troubleshooting

### Issue: "No module named 'jax'"

**Solution:**
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: "RuntimeError: CUDA not available"

**Check CUDA installation:**
```bash
nvcc --version  # Should show CUDA 12.x
nvidia-smi      # Should show GPU
```

**Reinstall JAX with CUDA:**
```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Issue: "ImportError: libcudnn.so.8"

**Install cuDNN:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libcudnn8 libcudnn8-dev

# Or download from NVIDIA: https://developer.nvidia.com/cudnn
```

### Issue: "Out of memory" during installation

**Solution:**
```bash
# Install packages one by one
pip install jax jaxlib
pip install flax optax
pip install transformers datasets
# ... etc
```

### Issue: "Killed" during pip install

**Increase swap space:**
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Alternative: Docker Installation

If you prefer Docker:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy project
COPY . /workspace
WORKDIR /workspace

# Install dependencies
RUN bash install_requirements.sh

# Run
CMD ["python", "train_hyena.py", "--config", "tustin"]
```

Build and run:
```bash
docker build -t hyena-mamba .
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints hyena-mamba
```

## Version Compatibility

### Tested Configurations

| Component | Version | Status |
|-----------|---------|--------|
| CUDA | 12.1 | ✅ Tested |
| Python | 3.10 | ✅ Tested |
| JAX | 0.4.25 | ✅ Tested |
| RTX 5090 | - | ✅ Optimized |

### Known Compatible Versions

- **CUDA**: 12.0, 12.1, 12.2
- **Python**: 3.10, 3.11
- **JAX**: 0.4.25+
- **Flax**: 0.8.0+

## Minimal Installation (for testing)

If you just want to test the code without GPU:

```bash
# CPU-only (fast install)
pip install jax jaxlib flax optax numpy matplotlib tqdm

# Run tests
python test_hyena_data.py
```

**Note**: Training will be slow without GPU.

## Post-Installation

After successful installation:

1. **Set environment variables:**
   ```bash
   source .env
   ```

2. **Run tests:**
   ```bash
   python test_hyena_data.py
   ```

3. **Quick validation:**
   ```bash
   bash run_experiment.sh quick
   ```

4. **Start full training:**
   ```bash
   bash run_experiment.sh tustin
   ```

## Getting Help

If installation fails:

1. Check CUDA installation: `nvcc --version`
2. Check GPU visibility: `nvidia-smi`
3. Verify Python version: `python --version`
4. Check error logs in terminal output
5. Try manual installation steps above

## Verification Checklist

Before starting training, verify:

- [ ] `nvidia-smi` shows RTX 5090
- [ ] `python setup_gpu.py` passes all checks
- [ ] `python test_hyena_data.py` completes successfully
- [ ] GPU utilization reaches >95% during quick test
- [ ] bfloat16 works (checked by `setup_gpu.py`)

---

**Once all checks pass, you're ready to train! 🚀**

See [QUICKSTART.md](QUICKSTART.md) for training instructions.
