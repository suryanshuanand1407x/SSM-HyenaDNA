#!/usr/bin/env python
"""Quick test to verify training works with real HG38 data."""

import sys
sys.path.insert(0, '/workspace/hyena')

from config_hyena import QUICK_CONFIG
from hyena_data_hg38 import HG38DataLoader
from model_hybrid import create_hybrid_model
from mamba_optim import OptimizedTrainState, train_step_fused
import jax
import jax.numpy as jnp
import optax

print("=" * 70)
print("Testing Training Pipeline with Real HG38 Data")
print("=" * 70)
print()

# Initialize data loader
print("1. Loading HG38 data...")
loader = HG38DataLoader(QUICK_CONFIG)
print()

# Create model
print("2. Creating hybrid model...")
rng = jax.random.PRNGKey(42)
model, params, pretrained_weights = create_hybrid_model(QUICK_CONFIG, rng)
print(f"   ✓ Model created")
print()

# Create optimizer
print("3. Creating optimizer...")
tx = optax.adamw(QUICK_CONFIG.learning_rate, weight_decay=QUICK_CONFIG.weight_decay)
opt_state = tx.init(params)
state = OptimizedTrainState(
    step=0,
    params=params,
    opt_state=opt_state,
    tx=tx,
    apply_fn=model.apply
)
print(f"   ✓ Optimizer ready")
print()

# Test training step
print("4. Testing training step...")
x, y, mask = loader.get_batch('train')
print(f"   Batch loaded: x.shape={x.shape}, y.shape={y.shape}")

state, loss = train_step_fused(state, x, y, mask)
loss_value = float(loss)
print(f"   ✓ Training step completed")
print(f"   Loss: {loss_value:.4f}")
print()

# Test a few more steps
print("5. Running 5 training steps...")
for i in range(5):
    x, y, mask = loader.get_batch('train')
    state, loss = train_step_fused(state, x, y, mask)
    loss_value = float(loss)
    print(f"   Step {i+1}/5: loss={loss_value:.4f}")

print()
print("=" * 70)
print("✓ All tests passed! Training pipeline works with real HG38 data!")
print("=" * 70)
print()
print("You can now run full training with:")
print("  python train_hyena.py --config quick")
print("  python train_hyena.py --config tustin")
print("  python train_hyena.py --config zoh")
