# Stable 20K Step Training Guide

## Overview

This is a **stable, production-ready** 20K step training configuration designed to:
- ✅ **Prevent NaN/Inf** with conservative hyperparameters
- ✅ **Track all metrics**: train/val loss & accuracy
- ✅ **Save checkpoints** every 500 steps
- ✅ **Detect and report** any numerical issues
- ✅ **Resume training** if interrupted

## Quick Start

### Start Training
```bash
python train_20k_stable.py
```

That's it! The script will:
1. Auto-download HG38 dataset if needed
2. Download HyenaDNA weights
3. Initialize Tustin Mamba model
4. Train for 20K steps with full metrics
5. Save checkpoints every 500 steps

## Configuration

### Stable Hyperparameters (No NaN)

```python
Model: 256d × 4 layers (Tustin Mamba)
Learning Rate: 5e-5 (conservative)
Batch Size: 16
Sequence Length: 1024
Gradient Clip: 0.5 (aggressive)
Steps: 20,000
Checkpoints: Every 500 steps
Evaluation: Every 500 steps
```

### Why These Parameters?

**Lower Learning Rate (5e-5):**
- Prevents gradient explosion
- Stable convergence
- No NaN issues

**Aggressive Gradient Clipping (0.5):**
- Prevents large gradients
- Essential for preventing NaN
- Slightly slower but much more stable

**Smaller Batch & Sequence:**
- More stable gradients
- Fits in memory easily
- Reduces chance of numerical issues

## Monitoring Training

### Real-time Progress

During training, you'll see:

```
Training: 45%|████████████▌             | 9000/20000 [45:23<50:32, loss: 0.8234]

══════════════════════════════════════════════════════════════════════
Evaluation - Step 9000
══════════════════════════════════════════════════════════════════════
Metric               Train           Validation
──────────────────────────────────────────────────────────────────────
Loss                 0.823456        0.856234
Accuracy             0.7234          0.7123
══════════════════════════════════════════════════════════════════════

✓ Checkpoint saved
```

### View Saved Metrics

At any time, view all saved metrics:

```bash
python view_metrics.py
```

Or specify custom location:
```bash
python view_metrics.py ./checkpoints/stable_20k/metrics.csv
```

**Output:**
```
================================================================================
TRAINING METRICS SUMMARY
================================================================================

Total checkpoints: 40
Steps recorded: 0 → 20000

────────────────────────────────────────────────────────────────────────────
LATEST METRICS (Most Recent Checkpoint)
────────────────────────────────────────────────────────────────────────────
Step: 20000

Metric               Value
───────────────────────────────────
Train Loss           0.456789
Train Accuracy       0.8234
Val Loss             0.478912
Val Accuracy         0.8156

────────────────────────────────────────────────────────────────────────────
BEST METRICS (Across All Checkpoints)
────────────────────────────────────────────────────────────────────────────

Metric                    Best Value      At Step
──────────────────────────────────────────────────
Lowest Train Loss         0.445123        19500
Highest Train Acc         0.8289          19500
Lowest Val Loss           0.467234        18500
Highest Val Acc           0.8201          18000
```

### Watch Metrics in Real-time

```bash
# In one terminal: run training
python train_20k_stable.py

# In another terminal: watch metrics file
watch -n 5 "tail -20 ./checkpoints/stable_20k/metrics.csv"
```

## Checkpoints

### Location
```
./checkpoints/stable_20k/
├── metrics.csv                 # All metrics (CSV format)
├── checkpoint_000500.pkl       # Step 500
├── checkpoint_001000.pkl       # Step 1000
├── checkpoint_001500.pkl       # Step 1500
...
└── checkpoint_020000.pkl       # Step 20000 (final)
```

### What's Saved
Each checkpoint contains:
- Model parameters
- Optimizer state
- Training step
- Metrics (train/val loss & accuracy)

### Resume Training

If training is interrupted:

```python
# Edit train_20k_stable.py, change start_step:
state = train_with_metrics(state, loader, STABLE_CONFIG, start_step=10000)
```

Or implement auto-resume using `checkpoint_utils.py`.

## NaN Detection

### Built-in Protection

The training script includes:

1. **Gradient Clipping**: Prevents large gradients
2. **Learning Rate**: Conservative to prevent explosion
3. **NaN Detection**: Checks every step
4. **Emergency Checkpoints**: Saves if NaN detected

### If NaN Occurs

The script will:
```
❌ NaN/Inf detected at step 1234!
Loss value: nan
Saving emergency checkpoint...
Training failed: NaN/Inf loss at step 1234
```

**Solutions:**
1. Lower learning rate further (try 1e-5)
2. Increase gradient clipping (try 0.3)
3. Check data loader for corrupted data
4. Verify model initialization

## Expected Results

### Initial (Step 0)
```
Train Loss: ~2.5
Train Acc:  ~0.25 (random)
Val Loss:   ~2.5
Val Acc:    ~0.25 (random)
```

### After 5K Steps
```
Train Loss: ~1.2-1.5
Train Acc:  ~0.55-0.65
Val Loss:   ~1.3-1.6
Val Acc:    ~0.53-0.63
```

### After 20K Steps
```
Train Loss: ~0.4-0.6
Train Acc:  ~0.75-0.85
Val Loss:   ~0.5-0.7
Val Acc:    ~0.73-0.83
```

## Training Time

### GPU (Recommended)
- RTX 5090: ~2-3 hours
- RTX 4090: ~3-4 hours
- RTX 3090: ~4-5 hours

### CPU (Slow)
- ~24-48 hours (not recommended)

## Files Created

```
./checkpoints/stable_20k/
├── metrics.csv                 # Training metrics
├── checkpoint_*.pkl            # Model checkpoints
└── config.json                 # Configuration

./results/stable_20k/
└── (future analysis results)
```

## Troubleshooting

### Problem: NaN Loss

**Solution:**
```python
# Edit train_20k_stable.py
STABLE_CONFIG = HyenaFineTuneConfig(
    learning_rate=1e-5,           # Even lower
    gradient_clip_norm=0.3,       # More aggressive
    batch_size=8,                 # Smaller batch
)
```

### Problem: Out of Memory

**Solution:**
```python
# Edit train_20k_stable.py
STABLE_CONFIG = HyenaFineTuneConfig(
    batch_size=8,                 # Smaller batch
    seq_len=512,                  # Shorter sequences
)
```

### Problem: Slow Training

**Solution:**
- Use GPU instead of CPU
- Increase batch size (if memory allows)
- Use fewer eval_iters for faster evaluation

### Problem: Accuracy Not Improving

**Possible causes:**
- Learning rate too low (increase to 1e-4)
- Not enough training steps (run longer)
- Model too small (increase d_model or n_layers)
- Check if loss is decreasing (main indicator)

## Next Steps

After 20K training completes:

1. **View final metrics:**
   ```bash
   python view_metrics.py
   ```

2. **Analyze results:**
   - Check val_loss curve (should decrease)
   - Check val_acc curve (should increase)
   - Compare train vs val (gap = overfitting)

3. **Continue training:**
   - If val_loss still decreasing, train longer
   - If overfitting (train << val), stop here

4. **Use best checkpoint:**
   - Find step with lowest val_loss
   - Use that checkpoint for inference/deployment

## Tips for Better Results

1. **Monitor overfitting:**
   - If train_loss << val_loss, you're overfitting
   - Stop training or add regularization

2. **Learning rate tuning:**
   - If loss plateaus early, increase LR
   - If NaN/unstable, decrease LR

3. **Longer training:**
   - 20K steps is moderate
   - For production, consider 50K-100K steps

4. **Model size:**
   - Current: 256d × 4 layers
   - For better accuracy: 512d × 6 layers
   - Edit `STABLE_CONFIG` in script

## Summary

**Start training:**
```bash
python train_20k_stable.py
```

**View metrics:**
```bash
python view_metrics.py
```

**Location:**
- Checkpoints: `./checkpoints/stable_20k/`
- Metrics: `./checkpoints/stable_20k/metrics.csv`

**Configuration:** Conservative hyperparameters to prevent NaN

**Duration:** ~2-3 hours on GPU, ~24-48 hours on CPU

**Output:** Full train/val loss & accuracy tracking
