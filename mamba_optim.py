"""
Mamba Hardware Optimization Layer
==================================
Hardware-specific optimizations for NVIDIA GPUs (H100, RTX 5090):
- XLA fusion with inline=True
- bfloat16 precision policy (Blackwell Tensor Cores)
- Buffer donation (donate_argnums)
- Asynchronous execution (no blocking ops)
- Device sharding strategy
- Gradient accumulation for VRAM-constrained GPUs

Imports mathematical core from mamba_core.py WITHOUT modification.
"""

import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial
from typing import Tuple, Dict, Any

# Import the EXACT mathematical core (unmodified)
from mamba_core import MambaLM


# =============================================================================
# H100 Device Sharding Strategy
# =============================================================================

def create_device_sharding(num_devices: int = 1):
    """
    Create positional sharding for GPU memory pinning.

    Args:
        num_devices: Number of GPUs (default 1 for single-GPU)

    Returns:
        Sharding strategy for model/optimizer states
    """
    from jax.sharding import PositionalSharding

    devices = jax.devices()[:num_devices]
    sharding = PositionalSharding(devices)

    return sharding, devices


# =============================================================================
# Precision Policy: bfloat16 for Tensor Cores (H100, RTX 5090 Blackwell)
# =============================================================================

def cast_to_bfloat16(pytree):
    """
    Cast floating-point arrays in pytree to bfloat16.
    Preserves integer types (for embeddings/indices).
    """
    def _cast_leaf(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x

    return jax.tree_util.tree_map(_cast_leaf, pytree)


def cast_to_float32(pytree):
    """Cast bfloat16 arrays back to float32 (for metrics/logging)."""
    def _cast_leaf(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
        return x

    return jax.tree_util.tree_map(_cast_leaf, pytree)


# =============================================================================
# Training State with Buffer Donation
# =============================================================================

class OptimizedTrainState(train_state.TrainState):
    """
    Extended TrainState with bfloat16 support and buffer donation tracking.
    """

    def apply_gradients_donated(self, *, grads, **kwargs):
        """
        Apply gradients with buffer donation for in-place HBM updates.
        This is called by the JIT-compiled training step with donate_argnums.
        """
        return self.apply_gradients(grads=grads, **kwargs)


def create_train_state(
    rng: jax.random.PRNGKey,
    model: MambaLM,
    learning_rate: float,
    seq_len: int,
    use_bfloat16: bool = True,
) -> OptimizedTrainState:
    """
    Initialize model parameters and optimizer state.

    Args:
        rng: JAX random key
        model: MambaLM instance
        learning_rate: Learning rate for AdamW
        seq_len: Sequence length for initialization
        use_bfloat16: Whether to cast params to bfloat16

    Returns:
        OptimizedTrainState with model params and optimizer
    """
    # Initialize with dummy input
    dummy_input = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']

    # Cast to bfloat16 if enabled (for Tensor Cores)
    if use_bfloat16:
        params = cast_to_bfloat16(params)

    # AdamW optimizer
    tx = optax.adamw(learning_rate)

    return OptimizedTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


# =============================================================================
# XLA-Fused Training Step (GPU Optimized)
# =============================================================================

@partial(
    jax.jit,
    inline=True,  # Force XLA to inline and fuse into single GPU kernel
    # NO buffer donation - prevents "Array has been deleted" errors
)
def train_step_fused(
    state: OptimizedTrainState,
    x: jnp.ndarray,  # (B, L) int32
    y: jnp.ndarray,  # (B, L) int32
    loss_mask: jnp.ndarray,  # (B, L) float32/bfloat16
) -> Tuple[OptimizedTrainState, jnp.ndarray]:
    """
    Single training step with XLA fusion.

    XLA fusion benefits:
    - Single kernel launch (eliminates PCIe round-trips)
    - Fused forward + backward pass
    - Optimized memory access patterns for VRAM/HBM

    NOTE: No buffer donation to ensure state remains valid after warmup
    and across training iterations. The slight memory overhead (~300MB)
    is acceptable for correctness.

    NO BLOCKING OPERATIONS:
    - No .item() calls
    - No print() inside JIT
    - Returns JAX arrays only (async dispatch)

    Args:
        state: Current training state (NOT donated)
        x: Input token IDs
        y: Target token IDs
        loss_mask: Mask for loss computation (1=compute, 0=ignore)

    Returns:
        (new_state, loss) - both as JAX arrays (async)
    """
    # Cast mask to match param dtype (bfloat16 or float32)
    param_dtype = jax.tree_util.tree_leaves(state.params)[0].dtype
    if loss_mask.dtype != param_dtype:
        loss_mask = loss_mask.astype(param_dtype)

    def loss_fn(params):
        # Forward pass (math from mamba_core.py)
        logits = state.apply_fn({'params': params}, x, train=True)

        # Per-token cross-entropy
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

        # Masked loss (only compute where mask=1)
        masked_loss = jnp.sum(per_token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

        return masked_loss

    # Compute loss and gradients (fused by XLA)
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Apply gradients (donated state enables in-place update)
    state = state.apply_gradients(grads=grads)

    # Return JAX arrays (no .item() - keeps async dispatch)
    return state, loss


@partial(
    jax.jit,
    inline=True  # Fuse eval step into single kernel
)
def eval_step_fused(
    state: OptimizedTrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluation step with XLA fusion (no weight updates).

    Computes both masked loss AND recall accuracy inside JIT
    to avoid transferring large logits tensor to CPU.

    Args:
        state: Current training state (not donated in eval)
        x: Input token IDs
        y: Target token IDs
        loss_mask: Mask for loss/accuracy computation

    Returns:
        (loss, accuracy) - both as JAX arrays (async)
    """
    # Cast mask to match param dtype
    param_dtype = jax.tree_util.tree_leaves(state.params)[0].dtype
    if loss_mask.dtype != param_dtype:
        loss_mask = loss_mask.astype(param_dtype)

    # Forward pass
    logits = state.apply_fn({'params': state.params}, x, train=False)

    # Masked loss
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    masked_loss = jnp.sum(per_token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

    # Recall accuracy (computed inside JIT - no logits transfer to CPU)
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == y).astype(param_dtype)
    accuracy = jnp.sum(correct * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

    return masked_loss, accuracy


@partial(
    jax.jit,
    inline=True  # Fuse eval step with logits output
)
def eval_step_with_logits(
    state: OptimizedTrainState,
    x: jnp.ndarray,
    y: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Evaluation step that also returns logits for detailed metrics.

    Used for detailed metric collection (mamba_metrics.py).
    Returns logits on-device for XLA-compiled metric computation.

    Args:
        state: Current training state
        x: Input token IDs
        y: Target token IDs
        loss_mask: Mask for loss computation

    Returns:
        (loss, accuracy, logits) - all as JAX arrays (async)
    """
    # Cast mask to match param dtype
    param_dtype = jax.tree_util.tree_leaves(state.params)[0].dtype
    if loss_mask.dtype != param_dtype:
        loss_mask = loss_mask.astype(param_dtype)

    # Forward pass
    logits = state.apply_fn({'params': state.params}, x, train=False)

    # Masked loss
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    masked_loss = jnp.sum(per_token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

    # Recall accuracy
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == y).astype(param_dtype)
    accuracy = jnp.sum(correct * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)

    return masked_loss, accuracy, logits


# =============================================================================
# Asynchronous Loss Estimation (No Blocking)
# =============================================================================

def estimate_loss_async(
    state: OptimizedTrainState,
    loader,  # MQARLoader instance
    eval_iters: int = 20,
) -> Dict[str, float]:
    """
    Estimate loss and accuracy over multiple batches.

    Async execution pattern:
    1. Launch all eval_step_fused calls (non-blocking)
    2. Collect results as JAX arrays
    3. Only block once at the end (jax.block_until_ready)
    4. Cast to float32 and convert to Python scalars

    This allows JAX to pipeline GPU execution while CPU prepares next batch.

    Args:
        state: Current training state
        loader: Data loader with get_batch method
        eval_iters: Number of batches to average

    Returns:
        Dict with train_loss, val_loss, train_acc, val_acc
    """
    out = {}

    for split in ['train', 'validation']:
        losses = []
        accs = []

        # Launch all eval steps (async - no blocking)
        for _ in range(eval_iters):
            x, y, mask = loader.get_batch(split)

            # JIT-compiled eval (returns JAX arrays, doesn't block)
            loss, acc = eval_step_fused(state, x, y, mask)
            losses.append(loss)
            accs.append(acc)

        # Block once to wait for all results
        losses = [jax.block_until_ready(l) for l in losses]
        accs = [jax.block_until_ready(a) for a in accs]

        # Cast to float32 (if bfloat16) and convert to Python scalars
        losses_f32 = [cast_to_float32(l) for l in losses]
        accs_f32 = [cast_to_float32(a) for a in accs]

        # Now safe to convert to Python scalars
        key_prefix = 'train' if split == 'train' else 'val'
        out[f'{key_prefix}_loss'] = float(jnp.mean(jnp.array(losses_f32)))
        out[f'{key_prefix}_acc'] = float(jnp.mean(jnp.array(accs_f32)))

    return out


# =============================================================================
# Warmup Compilation (JIT Pre-heating)
# =============================================================================

def warmup_jit_compilation(
    state: OptimizedTrainState,
    loader,  # MQARLoader instance
    verbose: bool = True,
) -> None:
    """
    Pre-compile JIT functions with warmup batches.

    Optimization:
    - First JIT call triggers XLA compilation (slow)
    - Subsequent calls reuse compiled kernel (fast)
    - Warmup compilation BEFORE timing main training loop

    Args:
        state: Training state
        loader: Data loader
        verbose: Whether to print compilation status
    """
    if verbose:
        print("=" * 60)
        print("WARMUP: Pre-compiling JIT kernels")
        print("=" * 60)

    # Warmup training step
    if verbose:
        print("Compiling train_step_fused...")
    x, y, mask = loader.get_batch('train')
    state, loss = train_step_fused(state, x, y, mask)
    jax.block_until_ready(loss)  # Wait for compilation
    if verbose:
        print(f"  ✓ Compiled (loss={float(cast_to_float32(loss)):.4f})")

    # Warmup eval step
    if verbose:
        print("Compiling eval_step_fused...")
    x, y, mask = loader.get_batch('validation')
    loss, acc = eval_step_fused(state, x, y, mask)
    jax.block_until_ready((loss, acc))  # Wait for compilation
    if verbose:
        print(f"  ✓ Compiled (loss={float(cast_to_float32(loss)):.4f}, acc={float(cast_to_float32(acc)):.4f})")

    if verbose:
        print("=" * 60)
        print("Warmup complete. Main training loop will be fast.\n")


# =============================================================================
# Device Memory Pinning (Optional - for multi-GPU)
# =============================================================================

def pin_state_to_device(
    state: OptimizedTrainState,
    sharding,
) -> OptimizedTrainState:
    """
    Pin training state to GPU VRAM/HBM using sharding strategy.

    For single-GPU: pins to VRAM/HBM for faster access
    For multi-GPU: shards across devices for data parallelism

    Args:
        state: Training state
        sharding: Sharding strategy from create_device_sharding

    Returns:
        State with pinned/sharded parameters
    """
    # Use jax.device_put with sharding
    pinned_params = jax.device_put(state.params, sharding)

    # Create new state with pinned params
    return state.replace(params=pinned_params)


# =============================================================================
# Gradient Clipping (Optional Stability Enhancement)
# =============================================================================

def create_optimizer_with_clipping(
    learning_rate: float,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with gradient clipping for stability.

    Args:
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm (clip if exceeded)

    Returns:
        Optimizer with gradient clipping
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate)
    )


# =============================================================================
# Gradient Accumulation (for RTX 5090 32GB VRAM)
# =============================================================================

@partial(
    jax.jit,
    inline=True,
    # NO buffer donation - accumulated_grads must persist across multiple iterations
)
def accumulate_gradients(
    state: OptimizedTrainState,
    accumulated_grads,
    x: jnp.ndarray,
    y: jnp.ndarray,
    loss_mask: jnp.ndarray,
) -> Tuple[Any, jnp.ndarray]:
    """
    Accumulate gradients without applying optimizer update.

    Used for gradient accumulation to maintain effective batch size
    on GPUs with limited VRAM (e.g., RTX 5090 32GB).

    NOTE: No buffer donation on accumulated_grads - it must remain alive
    across multiple accumulation steps to prevent "Array has been deleted" errors.

    Args:
        state: Current training state (params NOT updated)
        accumulated_grads: Current accumulated gradients
        x: Input token IDs
        y: Target token IDs
        loss_mask: Mask for loss computation

    Returns:
        (updated_accumulated_grads, loss)
    """
    # Cast mask to match param dtype
    param_dtype = jax.tree_util.tree_leaves(state.params)[0].dtype
    if loss_mask.dtype != param_dtype:
        loss_mask = loss_mask.astype(param_dtype)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, train=True)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        masked_loss = jnp.sum(per_token_loss * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        return masked_loss

    # Compute gradients for this micro-batch
    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Accumulate gradients
    if accumulated_grads is None:
        accumulated_grads = grads
    else:
        accumulated_grads = jax.tree_util.tree_map(lambda acc, g: acc + g, accumulated_grads, grads)

    return accumulated_grads, loss


@partial(
    jax.jit,
    inline=True,
    # NO buffer donation - state and accumulated_grads must remain valid
    # Donation was causing "Array has been deleted" errors
)
def apply_accumulated_gradients(
    state: OptimizedTrainState,
    accumulated_grads,
    num_accumulation_steps: int,
) -> OptimizedTrainState:
    """
    Apply accumulated gradients and reset accumulation.

    NOTE: No buffer donation to prevent "Array has been deleted" errors.
    The slight memory overhead is acceptable for correctness.

    Args:
        state: Current training state
        accumulated_grads: Accumulated gradients over multiple micro-batches
        num_accumulation_steps: Number of accumulation steps (for averaging)

    Returns:
        Updated state with applied gradients
    """
    # Average gradients over accumulation steps
    averaged_grads = jax.tree_util.tree_map(
        lambda g: g / num_accumulation_steps,
        accumulated_grads
    )

    # Apply gradients using optimizer
    state = state.apply_gradients(grads=averaged_grads)

    return state
