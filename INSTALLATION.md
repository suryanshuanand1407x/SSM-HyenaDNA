# Installation Guide - Tustin-Mamba for HG38

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Install JAX with GPU Support (Recommended)

**For CUDA 12.x:**
```bash
pip install --upgrade "jax[cuda12]"
```

**For CUDA 11.x:**
```bash
pip install --upgrade "jax[cuda11]"
```

**Using pip wheels (alternative):**
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Verify Installation

```bash
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

**Expected output (GPU):**
```
JAX version: 0.4.25
Devices: [cuda(id=0)]
```

**Expected output (CPU):**
```
JAX version: 0.4.25
Devices: [cpu(id=0)]
```

### 4. Test Dataset Download

```bash
python test_dataset_download.py
```

### 5. Start Training

```bash
# RC-equivariant training (recommended)
python train_20k_rc_equivariant.py

# Or standard training (no RC)
python train_20k_stable.py
```

---

## Detailed Installation

### Prerequisites

- **Python:** 3.9+ (3.10 or 3.11 recommended)
- **GPU (optional but recommended):** NVIDIA GPU with CUDA 11.8+ or 12.x
- **RAM:** 16 GB minimum, 32 GB recommended
- **Disk Space:** ~10 GB (for hg38 dataset + models)

### Step-by-Step Setup

#### Option 1: Fresh Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv_tustin

# Activate (Linux/Mac)
source venv_tustin/bin/activate

# Activate (Windows)
venv_tustin\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install JAX with GPU
pip install --upgrade "jax[cuda12]"
```

#### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n tustin python=3.11 -y
conda activate tustin

# Install dependencies
pip install -r requirements.txt

# Install JAX with GPU
pip install --upgrade "jax[cuda12]"
```

#### Option 3: System-wide Installation (Not Recommended)

```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-dev python3-pip

# Install CUDA (if using GPU)
# Follow: https://developer.nvidia.com/cuda-downloads

# Install Python packages
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install dependencies
pip3 install -r requirements.txt

# JAX (CPU only on macOS)
pip3 install --upgrade jax jaxlib
```

**Note:** macOS doesn't support NVIDIA GPUs. JAX will run on CPU.

### Windows (WSL2 Recommended)

**Option 1: WSL2 (Recommended)**
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install -y python3-dev python3-pip

# Install dependencies
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

**Option 2: Native Windows**
```powershell
# Install Python 3.11 from python.org

# Install dependencies
pip install -r requirements.txt

# JAX on Windows (experimental)
pip install --upgrade jax jaxlib
```

**Note:** GPU support on Windows is experimental. WSL2 is recommended.

---

## GPU Setup

### Check CUDA Version

```bash
nvidia-smi
```

**Look for:** `CUDA Version: 12.x` or `CUDA Version: 11.x`

### Install Matching JAX Version

**CUDA 12.x:**
```bash
pip install --upgrade "jax[cuda12]"
```

**CUDA 11.x:**
```bash
pip install --upgrade "jax[cuda11]"
```

### Verify GPU Detection

```bash
python -c "import jax; print(jax.devices())"
```

**Expected:** `[cuda(id=0)]`

**If CPU only:** `[cpu(id=0)]`
- Check CUDA installation
- Reinstall JAX with correct CUDA version
- Check GPU drivers

---

## Dependency Details

### Core Requirements (Must Install)

| Package | Purpose | Version |
|---------|---------|---------|
| `jax` | JAX core | ≥0.4.25 |
| `flax` | Neural networks | ≥0.8.0 |
| `optax` | Optimizers | ≥0.1.9 |
| `pyfaidx` | **FASTA parsing** | ≥0.7.2.1 |
| `pandas` | BED files, metrics | ≥2.0.0 |
| `transformers` | HyenaDNA weights | ≥4.37.0 |
| `tqdm` | Progress bars | ≥4.66.0 |

### Optional but Recommended

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `tensorboard` | Training visualization | `pip install tensorboard` |
| `wandb` | Experiment tracking | `pip install wandb` |
| `jupyter` | Interactive notebooks | `pip install jupyter` |
| `psutil` | System monitoring | `pip install psutil` |

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pyfaidx'`

**Solution:**
```bash
pip install pyfaidx
```

### Issue: JAX doesn't detect GPU

**Check CUDA:**
```bash
nvidia-smi
nvcc --version  # CUDA compiler version
```

**Reinstall JAX:**
```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]"
```

**Check installation:**
```bash
python -c "import jax; print(jax.devices())"
```

### Issue: `CUDA error: out of memory`

**Solution:** Reduce batch size or sequence length

```python
# Edit train_20k_rc_equivariant.py
batch_size=8       # Reduce from 16
seq_len=512        # Reduce from 1024
```

### Issue: `ImportError: cannot import name 'Fasta' from 'pyfaidx'`

**Solution:** Update pyfaidx
```bash
pip install --upgrade pyfaidx>=0.7.2.1
```

### Issue: Slow training (CPU mode)

**Check if GPU is being used:**
```bash
python -c "import jax; print(jax.devices())"
```

**If showing CPU, install GPU version:**
```bash
pip install --upgrade "jax[cuda12]"
```

### Issue: Version conflicts

**Solution:** Create fresh virtual environment
```bash
# Deactivate current environment
deactivate

# Remove old environment
rm -rf venv_tustin

# Create new environment
python -m venv venv_tustin
source venv_tustin/bin/activate

# Install fresh
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]"
```

---

## Verification Checklist

Run these commands to verify installation:

```bash
# 1. Check Python version
python --version  # Should be 3.9+

# 2. Check JAX
python -c "import jax; print(f'JAX: {jax.__version__}')"

# 3. Check GPU
python -c "import jax; print(f'Devices: {jax.devices()}')"

# 4. Check genomics libraries
python -c "import pyfaidx; print('pyfaidx: OK')"

# 5. Check HuggingFace
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 6. Test dataset
python test_dataset_download.py

# 7. Test RC module
python rc_equivariance.py
```

**All checks passed?** ✅ Ready to train!

---

## Disk Space Requirements

| Component | Size | Notes |
|-----------|------|-------|
| HG38 Dataset | ~3.1 GB | Auto-downloaded on first run |
| HyenaDNA Weights | ~500 MB | Auto-downloaded from HuggingFace |
| Checkpoints | ~2-5 GB | Per training run (20K steps) |
| Python Packages | ~2 GB | Installed in venv |
| **Total** | **~8-11 GB** | |

---

## Next Steps

After installation:

1. **Test setup:**
   ```bash
   python test_dataset_download.py
   ```

2. **Quick training test:**
   ```bash
   python train_20k_stable.py  # ~10 min on GPU
   ```

3. **Full RC-equivariant training:**
   ```bash
   python train_20k_rc_equivariant.py  # ~2-3 hours on GPU
   ```

4. **View metrics:**
   ```bash
   python view_metrics.py
   ```

---

## Support

**Documentation:**
- `README.md` - Project overview
- `RC_EQUIVARIANCE_GUIDE.md` - RC training guide
- `TUSTIN_MAMBA_ANALYSIS.md` - Technical details
- `TRAINING_GUIDE.md` - Training guide

**Quick References:**
- `START_20K_TRAINING.txt` - Quick start
- `RC_INTEGRATION_COMPLETE.txt` - RC setup

**Common Issues:**
- GPU not detected → Reinstall JAX with CUDA
- Out of memory → Reduce batch_size
- Slow training → Check if using GPU
- Missing pyfaidx → `pip install pyfaidx`

---

✅ **Installation complete!** Ready to train your Tustin-Mamba model.
