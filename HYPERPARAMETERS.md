# Hyperparameter Tuning Guide

## Quick Tweaking (Recommended)

**Edit the top of `train_hyena.py`** - all hyperparameters are there!

```python
# Example: train_hyena.py (lines 1-50)

# Model Architecture
MODEL_D_MODEL = 512              # ← Change this!
MODEL_N_LAYERS = 6               # ← And this!
MODEL_MODE = "tustin"            # ← Or this!

# Training Strategy
LEARNING_RATE = 1e-4             # ← Easy tweaking!
BATCH_SIZE = 32
SEQ_LEN = 4096
```

Then run:
```bash
python train_hyena.py --config custom
# or
bash run_experiment.sh custom
```

## All Hyperparameters Explained

### Model Architecture

```python
MODEL_D_MODEL = 512              # Hidden dimension
```
- **128**: Tiny model, fast training, ~2M params
- **256**: Small model, ~10M params (good for debugging)
- **512**: Medium model, ~50M params (production quality)
- **1024**: Large model, ~200M params (research scale)
- **RTX 5090 recommendation**: 512-1024

```python
MODEL_N_LAYERS = 6               # Number of Mamba blocks
```
- **2**: Very shallow, fast training
- **4**: Shallow, good for quick experiments
- **6**: Medium depth (good balance)
- **8**: Deep, better accuracy
- **12**: Very deep, best accuracy but slower
- **RTX 5090 recommendation**: 6-12

```python
MODEL_D_STATE = 16               # SSM state dimension
```
- **16**: Standard (Mamba paper default)
- **32**: Higher capacity (more memory)
- **64**: Very high capacity (experimental)
- **Recommendation**: Keep at 16 unless experimenting

```python
MODEL_D_CONV = 4                 # Convolution width
```
- **4**: Standard (Mamba paper default)
- **8**: Wider receptive field (more memory)
- **Recommendation**: Keep at 4

```python
MODEL_EXPAND = 2                 # Expansion factor
```
- **2**: Standard (Mamba paper default)
- **4**: Wider intermediate layer (more capacity)
- **Recommendation**: Keep at 2

```python
MODEL_MODE = "tustin"            # Discretization method
```
- **"tustin"**: Your Tustin discretization (comparison study)
- **"zoh"**: Zero-Order Hold baseline
- **"vanilla"**: Raw Tustin without guards

### Training Strategy

```python
LEARNING_RATE = 1e-4             # Learning rate
```
- **5e-5**: Conservative (stable, slower)
- **1e-4**: Standard (good default)
- **2e-4**: Aggressive (faster but less stable)
- **If training diverges**: Lower to 5e-5 or 3e-5
- **If too slow**: Increase to 2e-4

```python
BATCH_SIZE = 32                  # Batch size
```
- **8**: Small (fits on any GPU, slower training)
- **16**: Medium (good for 16GB GPUs)
- **32**: Large (good for RTX 5090)
- **64**: Very large (best for RTX 5090 with short sequences)
- **Memory constraint**: Reduce if OOM
- **Throughput**: Larger is usually faster

```python
SEQ_LEN = 4096                   # Sequence length (tokens)
```
- **1024**: Short (fast, less context)
- **2048**: Medium (good balance)
- **4096**: Long (better genomic context)
- **8192**: Very long (max quality, more memory)
- **RTX 5090 @ 512d×6**: Can handle 8192
- **RTX 5090 @ 1024d×12**: Reduce to 4096 or 2048

```python
MAX_STEPS = 100000               # Total training steps
```
- **1000**: Quick test (~5 min)
- **10000**: Short training (~30 min)
- **50000**: Medium training (~2 hours)
- **100000**: Full training (~4 hours)
- **200000**: Extended training (~8 hours)

```python
WARMUP_STEPS = 2000              # LR warmup steps
```
- **Rule of thumb**: 2-5% of total steps
- **1000**: For 50K total steps
- **2000**: For 100K total steps
- **4000**: For 200K total steps

### Two-Phase Training

```python
PHASE1_STEPS = 20000             # Freeze embeddings phase
PHASE2_STEPS = 80000             # Full fine-tuning phase
```
- **Standard ratio**: 20% Phase 1, 80% Phase 2
- **Conservative**: 30% Phase 1, 70% Phase 2 (more stable)
- **Aggressive**: 10% Phase 1, 90% Phase 2 (faster adaptation)
- **Must equal**: PHASE1_STEPS + PHASE2_STEPS = MAX_STEPS

```python
FREEZE_EMBEDDINGS_PHASE1 = True  # Freeze embeddings?
```
- **True**: Recommended (prevents corrupting HyenaDNA embeddings)
- **False**: Full fine-tuning from start (risky)

### Optimization

```python
USE_BFLOAT16 = True              # Use bfloat16 precision
```
- **True**: Recommended for RTX 5090 (Tensor Cores!)
- **False**: Full float32 (slower, more memory, more precise)

```python
GRADIENT_CLIP_NORM = 1.0         # Gradient clipping
```
- **0.5**: Aggressive clipping (very stable)
- **1.0**: Standard clipping (good default)
- **2.0**: Loose clipping (allows large updates)
- **If training unstable**: Reduce to 0.5
- **If too slow to converge**: Increase to 2.0

```python
WEIGHT_DECAY = 0.1               # AdamW weight decay
```
- **0.0**: No regularization
- **0.1**: Standard (good default)
- **0.2**: Strong regularization (prevent overfitting)

```python
GRADIENT_ACCUMULATION = 1        # Accumulation steps
```
- **1**: No accumulation (standard)
- **2**: Effective batch size × 2 (if OOM)
- **4**: Effective batch size × 4 (for very large batches)

### Evaluation & Checkpointing

```python
EVAL_INTERVAL = 500              # Steps between evals
SAVE_INTERVAL = 2000             # Steps between saves
```
- **Frequent eval**: Every 100-500 steps (good for monitoring)
- **Infrequent eval**: Every 1000-2000 steps (faster training)
- **Save interval**: 5-10x eval interval (save disk space)

### Data Loading

```python
NUM_WORKERS = 4                  # CPU workers
PREFETCH_BATCHES = 2             # Prefetch queue size
```
- **CPU-bound system**: Increase NUM_WORKERS to 6-8
- **Fast GPU**: Increase PREFETCH_BATCHES to 4-6
- **Low RAM**: Reduce PREFETCH_BATCHES to 1

See [DATA_LOADING.md](DATA_LOADING.md) for details.

## Common Configurations

### 1. Quick Test (5 minutes)
```python
MODEL_D_MODEL = 128
MODEL_N_LAYERS = 2
BATCH_SIZE = 16
SEQ_LEN = 1024
MAX_STEPS = 1000
```

### 2. Standard Comparison (4 hours)
```python
MODEL_D_MODEL = 512
MODEL_N_LAYERS = 6
BATCH_SIZE = 32
SEQ_LEN = 4096
MAX_STEPS = 100000
```

### 3. Max Quality (8-12 hours)
```python
MODEL_D_MODEL = 1024
MODEL_N_LAYERS = 12
BATCH_SIZE = 16
SEQ_LEN = 8192
MAX_STEPS = 200000
```

### 4. Memory-Constrained (16GB GPU)
```python
MODEL_D_MODEL = 256
MODEL_N_LAYERS = 4
BATCH_SIZE = 8
SEQ_LEN = 2048
MAX_STEPS = 50000
```

### 5. Throughput-Optimized (RTX 5090)
```python
MODEL_D_MODEL = 512
MODEL_N_LAYERS = 6
BATCH_SIZE = 64          # Large batch
SEQ_LEN = 2048           # Shorter sequence
MAX_STEPS = 200000       # More steps to compensate
NUM_WORKERS = 8          # More workers
PREFETCH_BATCHES = 4     # More prefetch
```

## Tuning Strategy

### 1. Start with Defaults
```bash
# Use preset first
python train_hyena.py --config tustin
```

### 2. Adjust for Your GPU
Check memory usage:
```bash
nvidia-smi -l 1
```

If **< 50% VRAM used**:
- Increase `BATCH_SIZE` (32 → 64)
- Increase `SEQ_LEN` (4096 → 8192)
- Increase `MODEL_D_MODEL` (512 → 768 or 1024)

If **OOM (Out of Memory)**:
- Reduce `BATCH_SIZE` (32 → 16)
- Reduce `SEQ_LEN` (4096 → 2048)
- Reduce `MODEL_D_MODEL` (512 → 256)

### 3. Adjust for Training Speed
Check GPU utilization:
```bash
nvidia-smi -l 1
```

If **< 90% GPU utilization**:
- Increase `NUM_WORKERS` (4 → 6 or 8)
- Increase `PREFETCH_BATCHES` (2 → 4)
- Check data loading (see [DATA_LOADING.md](DATA_LOADING.md))

### 4. Adjust for Stability
If **loss → NaN**:
- Lower `LEARNING_RATE` (1e-4 → 5e-5)
- Lower `GRADIENT_CLIP_NORM` (1.0 → 0.5)
- Increase `WARMUP_STEPS` (2000 → 4000)

If **training very slow**:
- Increase `LEARNING_RATE` (1e-4 → 2e-4)
- Increase `BATCH_SIZE` (32 → 64)

## Usage Examples

### Example 1: Quick Hyperparameter Sweep

Edit `train_hyena.py`:
```python
# Try different learning rates
LEARNING_RATE = 5e-5  # First run
# LEARNING_RATE = 1e-4  # Second run
# LEARNING_RATE = 2e-4  # Third run
```

Run:
```bash
python train_hyena.py --config custom
# Change LEARNING_RATE, repeat
```

### Example 2: Larger Model
```python
MODEL_D_MODEL = 768      # Increase from 512
MODEL_N_LAYERS = 8       # Increase from 6
BATCH_SIZE = 24          # Reduce slightly (memory)
SEQ_LEN = 6144           # Increase from 4096
```

### Example 3: Fast Iteration
```python
BATCH_SIZE = 64          # Max out batch size
SEQ_LEN = 2048           # Shorter sequences
NUM_WORKERS = 8          # More workers
EVAL_INTERVAL = 1000     # Less frequent eval
```

## Advanced: Command-Line Overrides

You can also add argparse options for quick overrides:

```python
# In train_hyena.py main():
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--batch-size', type=int, default=None)

if args.lr:
    LEARNING_RATE = args.lr
if args.batch_size:
    BATCH_SIZE = args.batch_size
```

Then:
```bash
python train_hyena.py --config custom --lr 5e-5 --batch-size 64
```

## Tips

1. **Always start with a preset** (quick/tustin/zoh) to verify setup
2. **Make ONE change at a time** when tuning
3. **Keep notes** of what works and what doesn't
4. **Monitor GPU utilization** (should be >90%)
5. **Check validation loss** (should decrease smoothly)
6. **Save good configs** (copy hyperparameters to a file)

## Hyperparameter Search

For systematic search, use a simple bash loop:

```bash
# Search over learning rates
for lr in 3e-5 5e-5 1e-4 2e-4; do
    # Edit LEARNING_RATE in train_hyena.py
    sed -i "s/LEARNING_RATE = .*/LEARNING_RATE = $lr/" train_hyena.py

    # Train
    python train_hyena.py --config custom

    # Results in different checkpoint dirs
done
```

---

**Happy tuning! 🚀**

For questions, check:
- [QUICKSTART.md](QUICKSTART.md) - Training basics
- [DATA_LOADING.md](DATA_LOADING.md) - Data pipeline tuning
- [README_HYENA.md](README_HYENA.md) - Full documentation
