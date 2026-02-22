# HyenaDNA → Tustin Mamba Training Guide

## Overview

This setup automatically:
1. **Downloads HG38 dataset** (~3.1 GB) from Google Cloud Storage
2. **Downloads HyenaDNA-medium weights** from HuggingFace
3. **Swaps Hyena blocks with Tustin Mamba blocks**
4. **Trains with 2-phase fine-tuning** (freeze embeddings → full fine-tuning)

## Quick Start

### Option 1: Automatic Training (Recommended)

The easiest way to start training:

```bash
# Quick test (small model, 1000 steps)
python train_with_auto_download.py

# Full Tustin training (512d, 6 layers, 100K steps)
python train_with_auto_download.py --config tustin

# Resume from checkpoint
python train_with_auto_download.py --resume --config tustin
```

**What happens automatically:**
- ✓ Checks if HG38 dataset exists, downloads if missing (~3.1 GB)
- ✓ Downloads HyenaDNA-medium weights from HuggingFace
- ✓ Initializes Tustin Mamba blocks
- ✓ Trains with 2-phase strategy
- ✓ Saves checkpoints every 200 steps

### Option 2: Standard Training Script

If you prefer the original training script:

```bash
# Quick test
python train_hyena.py --config quick

# Tustin Mamba (recommended)
python train_hyena.py --config tustin

# Custom hyperparameters (edit top of train_hyena.py)
python train_hyena.py --config custom

# Resume training
python train_hyena.py --resume --config tustin
```

## Dataset Information

### HG38 Human Reference Genome

**Files:**
- `hg38.ml.fa` - Human reference genome (3.1 GB)
- `human-sequences.bed` - Sequence intervals with train/val/test splits

**Download:**
The dataset will download automatically when you run training. You can also manually download:

```bash
# Manual download
python download_hg38.py

# Or using bash script
bash download_hg38_data.sh
```

**Data Source:**
- Google Cloud Storage (basenji_barnyard2 bucket)
- Same dataset used by official HyenaDNA

## Model Architecture

### HyenaDNA-Medium → Tustin Mamba

**What gets loaded from HyenaDNA:**
- ✓ Token embeddings (pretrained on genomic data)
- ✓ Layer norms (if compatible)

**What gets replaced:**
- Hyena operators → **Tustin Mamba blocks** (your custom SSM)

**Architecture:**
```
Input Tokens (DNA: A, C, G, T, N)
    ↓
[HyenaDNA Embeddings]  ← Pretrained weights loaded
    ↓
[Tustin Mamba Block 1]  ← Your custom Tustin discretization
[Tustin Mamba Block 2]
    ...
[Tustin Mamba Block N]
    ↓
[Output Layer]  ← Randomly initialized
    ↓
Logits (vocab_size=12)
```

## Configuration Presets

### Quick Config (Testing)
```python
d_model=128
n_layers=2
batch_size=16
seq_len=1024
max_steps=1000
```
**Use for:** Quick validation, debugging

### Tustin Config (Recommended)
```python
d_model=512
n_layers=6
batch_size=32
seq_len=4096
max_steps=100000
phase1_steps=20000  # Freeze embeddings
phase2_steps=80000  # Full fine-tuning
```
**Use for:** Full training run, comparison studies

### Custom Config
Edit the **top of `train_hyena.py`** to set custom hyperparameters:

```python
# At top of train_hyena.py
MODEL_D_MODEL = 512
MODEL_N_LAYERS = 6
MODEL_MODE = "tustin"  # Your Tustin Mamba
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
SEQ_LEN = 4096
```

Then run:
```bash
python train_hyena.py --config custom
```

## Training Strategy

### Two-Phase Fine-tuning

**Phase 1: Linear Probing (20% of training)**
- Freeze HyenaDNA embeddings (pretrained)
- Train only Mamba blocks
- Fast adaptation to new architecture

**Phase 2: Full Fine-tuning (80% of training)**
- Unfreeze all parameters
- Train end-to-end
- Fine-tune embeddings for your task

### Checkpointing

Checkpoints saved every **200 steps** to:
```
./checkpoints/hyena_mamba_tustin/
├── checkpoint_000200.pkl  # Step 200
├── checkpoint_000400.pkl  # Step 400
├── ...
└── checkpoint_100000.pkl  # Final
```

Each checkpoint includes:
- Model parameters
- Optimizer state
- Training step
- Metrics (train/val loss, accuracy)

## Monitoring Training

### Real-time Logs

Watch training progress:
```bash
# Loss curves
tail -f checkpoints/hyena_mamba_tustin/metrics.csv

# Training logs
# (automatically printed during training)
```

### Metrics Saved
- Train loss
- Validation loss
- Train accuracy (if available)
- Validation accuracy (if available)

## Hardware Requirements

### GPU (Recommended)
- **RTX 5090** (32 GB) - Full training, long sequences
- **RTX 4090** (24 GB) - Medium model
- **RTX 3090** (24 GB) - Smaller batch size

### CPU (Fallback)
- Works on CPU but **much slower**
- JAX will automatically detect and use CPU

## Troubleshooting

### Dataset Download Fails

**Problem:** Network error during download

**Solution:**
```bash
# Manual download using bash script
bash download_hg38_data.sh

# Or download directly
python download_hg38.py --force
```

### HuggingFace Download Fails

**Problem:** Can't download HyenaDNA weights

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME=./cache/huggingface

# Or edit config_hyena.py
cache_dir: str = "./cache/hyenadna"  # Change this path
```

### Out of Memory (OOM)

**Problem:** GPU runs out of memory

**Solution:** Reduce batch size or sequence length
```python
# Edit train_hyena.py
BATCH_SIZE = 16  # Reduce from 32
SEQ_LEN = 2048   # Reduce from 4096
```

### JAX/CUDA Issues

**Problem:** JAX doesn't detect GPU

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall JAX with CUDA support
pip install --upgrade "jax[cuda12]"
```

## File Structure

```
hyena2/
├── train_hyena.py                  # Main training script
├── train_with_auto_download.py    # Auto-download wrapper
├── download_hg38.py                # Dataset downloader
├── download_hg38_data.sh           # Bash download script
├── hyena_data_hg38.py             # HG38 data loader
├── model_hybrid.py                 # HyenaDNA + Mamba hybrid
├── mamba_core.py                   # Tustin Mamba implementation
├── config_hyena.py                 # Configuration presets
├── checkpoint_utils.py             # Checkpoint management
├── mamba_optim.py                  # Optimized training functions
├── data/
│   └── hg38/                       # Dataset (auto-downloaded)
│       ├── hg38.ml.fa              # Human genome
│       └── human-sequences.bed     # Sequence intervals
└── checkpoints/
    └── hyena_mamba_tustin/         # Training checkpoints
```

## Next Steps

1. **Test dataset download:**
   ```bash
   python test_dataset_download.py
   ```

2. **Quick training test:**
   ```bash
   python train_with_auto_download.py
   ```

3. **Full training run:**
   ```bash
   python train_with_auto_download.py --config tustin
   ```

4. **Monitor progress:**
   Watch the terminal for loss curves and metrics

5. **Resume if interrupted:**
   ```bash
   python train_with_auto_download.py --resume --config tustin
   ```

## Comparison: Tustin vs ZOH

To compare Tustin vs ZOH discretization:

```bash
# Train Tustin
python train_hyena.py --config tustin

# Train ZOH
python train_hyena.py --config zoh

# Compare results in:
./results/tustin_comparison/
./results/zoh_comparison/
```

## Questions?

- Check logs in `./checkpoints/hyena_mamba_tustin/`
- Verify dataset: `python download_hg38.py`
- Test data loader: `python test_dataset_download.py`
