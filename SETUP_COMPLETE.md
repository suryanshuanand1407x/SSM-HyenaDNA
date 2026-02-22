# 🎉 Setup Complete! HyenaDNA → Tustin Mamba Training

## ✅ What Was Configured

### 1. Automatic Dataset Download
- **Created:** `download_hg38.py` - Python module for automatic HG38 dataset download
- **Modified:** `hyena_data_hg38.py` - Data loader now auto-downloads dataset
- **Feature:** Downloads ~3.1 GB HG38 human genome automatically on first run
- **Location:** `./data/hg38/`

### 2. Automatic HyenaDNA Weights Loading
- **Existing:** `model_hybrid.py` - Already loads HyenaDNA-medium weights
- **Source:** HuggingFace `LongSafari/hyenadna-medium-160k-seqlen`
- **Loaded:** Token embeddings, layer norms (pretrained on genomic data)

### 3. Tustin Mamba Integration
- **Existing:** `mamba_core.py` - Your Tustin Mamba implementation
- **Configuration:** Set `MODEL_MODE = "tustin"` in `train_hyena.py`
- **Feature:** Swaps HyenaDNA Hyena blocks with your Tustin Mamba blocks

### 4. Training Scripts
- **Created:** `train_with_auto_download.py` - Wrapper with automatic setup
- **Created:** `START_TRAINING.sh` - One-command launcher
- **Created:** `test_dataset_download.py` - Test dataset download
- **Existing:** `train_hyena.py` - Main training script (already configured)

### 5. Documentation
- **Created:** `TRAINING_GUIDE.md` - Comprehensive training guide
- **Created:** `SETUP_COMPLETE.md` - This file

---

## 🚀 Quick Start (3 Commands)

### Easiest Way (One Command):
```bash
bash START_TRAINING.sh
```

### Alternative (Python):
```bash
python train_with_auto_download.py
```

### Standard Training:
```bash
python train_hyena.py --config quick
```

**All three methods will:**
1. ✓ Auto-download HG38 dataset if missing (~3.1 GB)
2. ✓ Auto-download HyenaDNA-medium weights from HuggingFace
3. ✓ Initialize Tustin Mamba blocks
4. ✓ Start training with 2-phase fine-tuning

---

## 📊 Training Modes

### Quick Test (Recommended First)
```bash
bash START_TRAINING.sh quick
```
- Small model (128d, 2 layers)
- 1000 steps
- Fast validation (~10 minutes on GPU)

### Full Tustin Training
```bash
bash START_TRAINING.sh tustin
```
- Large model (512d, 6 layers)
- 100K steps
- Full training run (~hours on RTX 5090)

### Custom Hyperparameters
1. Edit top of `train_hyena.py`:
   ```python
   MODEL_D_MODEL = 512
   MODEL_N_LAYERS = 6
   MODEL_MODE = "tustin"
   BATCH_SIZE = 32
   SEQ_LEN = 4096
   ```

2. Run:
   ```bash
   bash START_TRAINING.sh custom
   ```

### Resume Training
```bash
bash START_TRAINING.sh resume
```

---

## 🔍 How It Works

### Step-by-Step Flow

1. **Run training script**
   ```bash
   python train_with_auto_download.py
   ```

2. **Auto-download HG38 dataset** (if not present)
   - Downloads `hg38.ml.fa.gz` from Google Cloud (~800 MB compressed)
   - Decompresses to `hg38.ml.fa` (~3.1 GB)
   - Downloads `human-sequences.bed` (~35 MB)
   - Saves to `./data/hg38/`

3. **Auto-download HyenaDNA weights** (via HuggingFace)
   - Model: `LongSafari/hyenadna-medium-160k-seqlen`
   - Loads pretrained embeddings
   - Caches to `./data/hyenadna/`

4. **Initialize Hybrid Model**
   ```
   HyenaDNA Embeddings (pretrained)
         ↓
   Tustin Mamba Block 1 (random init)
   Tustin Mamba Block 2 (random init)
         ...
   Tustin Mamba Block N (random init)
         ↓
   Output Layer (random init)
   ```

5. **Two-Phase Training**
   - **Phase 1** (20% of steps): Freeze embeddings, train Mamba blocks
   - **Phase 2** (80% of steps): Unfreeze all, full fine-tuning

6. **Save Checkpoints**
   - Every 200 steps
   - Location: `./checkpoints/hyena_mamba_tustin/`
   - Includes: params, optimizer state, metrics

---

## 📁 Dataset Information

### HG38 Human Reference Genome

**What is it?**
- Complete human genome reference (GRCh38/hg38)
- Used for pretraining HyenaDNA
- Real genomic sequences from 24 chromosomes

**Files:**
```
./data/hg38/
├── hg38.ml.fa              # Human genome FASTA (3.1 GB)
└── human-sequences.bed     # Train/val/test intervals (35 MB)
```

**Download:**
- **Automatic:** Happens on first training run
- **Manual:** `python download_hg38.py` or `bash download_hg38_data.sh`
- **Source:** Google Cloud Storage (basenji_barnyard2)

**Verification:**
```bash
python test_dataset_download.py
```

---

## 🎯 Model Architecture

### HyenaDNA-Medium (Pretrained)
- **Vocab size:** 12 (DNA tokens: A, C, G, T, N, + special)
- **Hidden dim:** 256 (medium model)
- **Sequence length:** Up to 160K (for pretraining)
- **Pretrained on:** Human genome data

### Your Tustin Mamba Model
```python
# Configuration (in train_hyena.py)
MODEL_D_MODEL = 512        # Hidden dimension
MODEL_N_LAYERS = 6         # Number of Tustin Mamba blocks
MODEL_MODE = "tustin"      # Your Tustin discretization
```

**What gets loaded from HyenaDNA:**
- ✓ Token embeddings (12 vocab × hidden_dim)
- ✓ Layer norms (if compatible)

**What gets replaced:**
- ✗ Hyena operators → ✓ **Tustin Mamba blocks**

**What gets randomly initialized:**
- Tustin Mamba SSM parameters
- Output layer

---

## 📈 Monitoring Training

### Real-time Progress
Training automatically prints:
```
Phase 1 (Freeze Embeddings): 100%|████| 20000/20000 [loss: 1.234]
Phase 2 (Full Fine-tuning):  100%|████| 80000/80000 [loss: 0.567]
```

### Metrics Saved
```
./checkpoints/hyena_mamba_tustin/
├── metrics.csv              # Loss curves
├── checkpoint_000200.pkl    # Every 200 steps
├── checkpoint_000400.pkl
└── ...
```

### View Metrics
```bash
# Watch loss in real-time
tail -f ./checkpoints/hyena_mamba_tustin/metrics.csv

# Plot metrics (if you have visualization script)
python plot_training_metrics.py
```

---

## 🛠️ Troubleshooting

### Issue: Dataset download fails
**Solution:**
```bash
# Manual download
python download_hg38.py --force

# Or using bash
bash download_hg38_data.sh
```

### Issue: HuggingFace download fails
**Solution:**
```bash
# Set cache directory
export HF_HOME=./cache/huggingface

# Clear cache and retry
rm -rf ./data/hyenadna/
python train_with_auto_download.py
```

### Issue: Out of memory (OOM)
**Solution:** Reduce batch size or sequence length
```python
# Edit train_hyena.py
BATCH_SIZE = 16      # Reduce from 32
SEQ_LEN = 2048       # Reduce from 4096
```

### Issue: JAX doesn't detect GPU
**Solution:**
```bash
# Check GPU
nvidia-smi

# Reinstall JAX with CUDA
pip install --upgrade "jax[cuda12]"
```

### Issue: Missing dependencies
**Solution:**
```bash
pip install jax flax optax transformers pyfaidx pandas tqdm numpy
```

---

## 🧪 Testing

### Test Dataset Download
```bash
python test_dataset_download.py
```

**Output:**
```
✓ Dataset download test PASSED
✓ Dataset verification PASSED
✓ Data loader test PASSED
✓ All tests passed! Ready for training.
```

### Test Training (Quick)
```bash
bash START_TRAINING.sh quick
```

**Expected:**
- Downloads dataset (~3-10 minutes)
- Downloads HyenaDNA weights (~1-2 minutes)
- Trains for 1000 steps (~5-10 minutes on GPU)
- Saves checkpoints to `./checkpoints/hyena_mamba_quick/`

---

## 📋 Command Reference

### Training Commands
```bash
# Quick test
bash START_TRAINING.sh quick
python train_with_auto_download.py

# Full Tustin training
bash START_TRAINING.sh tustin
python train_with_auto_download.py --config tustin

# Custom hyperparameters
bash START_TRAINING.sh custom
python train_with_auto_download.py --config custom

# Resume training
bash START_TRAINING.sh resume
python train_with_auto_download.py --resume --config tustin

# ZOH comparison
bash START_TRAINING.sh zoh
python train_with_auto_download.py --config zoh
```

### Dataset Commands
```bash
# Download dataset
python download_hg38.py

# Force re-download
python download_hg38.py --force

# Test dataset
python test_dataset_download.py
```

### Standard Training (without auto-download wrapper)
```bash
python train_hyena.py --config quick
python train_hyena.py --config tustin
python train_hyena.py --config custom
python train_hyena.py --resume --config tustin
```

---

## 🎓 What Happens During Training

### Phase 1: Linear Probing (20%)
- **Steps:** 0 → 20,000
- **Strategy:** Freeze HyenaDNA embeddings, train Mamba blocks
- **Goal:** Adapt Mamba blocks to genomic data distribution
- **Checkpoints:** Every 200 steps

### Phase 2: Full Fine-tuning (80%)
- **Steps:** 20,000 → 100,000
- **Strategy:** Unfreeze all parameters, end-to-end training
- **Goal:** Fine-tune entire model for optimal performance
- **Checkpoints:** Every 200 steps

### Checkpointing
- **Frequency:** Every 200 steps
- **Retention:** Last 20 checkpoints (4000 steps of history)
- **Contents:** Model params, optimizer state, step, metrics
- **Location:** `./checkpoints/hyena_mamba_[config]/`

---

## 📊 Expected Results

### Quick Test (1000 steps)
- **Train Loss:** ~2.5 → ~1.5
- **Val Loss:** ~2.5 → ~1.6
- **Time:** ~10 minutes (GPU)

### Full Tustin (100K steps)
- **Train Loss:** ~2.5 → ~0.5-0.8
- **Val Loss:** ~2.5 → ~0.6-0.9
- **Time:** Several hours (RTX 5090)

---

## 🔄 Next Steps

1. **Verify setup:**
   ```bash
   python test_dataset_download.py
   ```

2. **Quick test:**
   ```bash
   bash START_TRAINING.sh quick
   ```

3. **Monitor results:**
   - Watch terminal for loss curves
   - Check `./checkpoints/hyena_mamba_quick/metrics.csv`

4. **Full training:**
   ```bash
   bash START_TRAINING.sh tustin
   ```

5. **Compare Tustin vs ZOH:**
   ```bash
   bash START_TRAINING.sh tustin
   bash START_TRAINING.sh zoh
   # Compare results in ./results/
   ```

---

## 📚 Files Created

```
hyena2/
├── download_hg38.py                ✨ NEW - Auto dataset downloader
├── train_with_auto_download.py    ✨ NEW - Training wrapper
├── START_TRAINING.sh               ✨ NEW - One-command launcher
├── test_dataset_download.py        ✨ NEW - Dataset test script
├── TRAINING_GUIDE.md               ✨ NEW - Comprehensive guide
├── SETUP_COMPLETE.md               ✨ NEW - This file
├── hyena_data_hg38.py              🔧 MODIFIED - Added auto-download
└── [existing files unchanged]
```

---

## ✅ Ready to Train!

Everything is set up. Just run:

```bash
bash START_TRAINING.sh
```

The script will automatically:
1. ✓ Download HG38 dataset (~3.1 GB)
2. ✓ Download HyenaDNA-medium weights
3. ✓ Initialize Tustin Mamba blocks
4. ✓ Train with 2-phase fine-tuning
5. ✓ Save checkpoints every 200 steps

**Questions?** Check `TRAINING_GUIDE.md` for detailed documentation.

---

**Happy Training! 🚀**
