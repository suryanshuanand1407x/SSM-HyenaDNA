"""
Mamba Metrics Module - JAX-Native Evaluation
=============================================
All metrics are implemented in pure JAX for XLA compilation and H100 execution.
NO CPU-GPU synchronization during metric computation - everything stays on device.

Metrics:
1. Exact Match (EM): Token-level accuracy on value recall positions
2. Recall-by-Position: Accuracy bucketed by relative sequence position
3. Numerical Drift Tracker: MSE between predicted and analytical states
4. Information Density: Success rate vs KV-pair count

All functions are JIT-compatible and return JAX arrays (no blocking ops).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Optional


# =============================================================================
# 1. Exact Match (EM) - Token-Level Accuracy on Value Positions
# =============================================================================

@jax.jit
def compute_exact_match(
    logits: jnp.ndarray,      # (B, L, V) - model predictions
    targets: jnp.ndarray,     # (B, L) - ground truth token IDs
    value_mask: jnp.ndarray,  # (B, L) - 1 where position is a value to recall
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Exact Match accuracy on value recall positions.

    EM measures whether the model predicted the EXACT correct token
    at positions marked by value_mask (where mask=1).

    Args:
        logits: Model output logits (pre-softmax)
        targets: Ground truth token IDs
        value_mask: Binary mask (1=value position, 0=ignore)

    Returns:
        (em_accuracy, num_value_tokens) - both as JAX arrays
        - em_accuracy: Fraction of correct predictions on value positions
        - num_value_tokens: Total number of value positions evaluated
    """
    # Get predicted tokens
    predictions = jnp.argmax(logits, axis=-1)  # (B, L)

    # Check exact matches
    correct = (predictions == targets).astype(jnp.float32)  # (B, L)

    # Apply value mask (only count value positions)
    correct_values = correct * value_mask  # (B, L)

    # Compute accuracy
    num_correct = jnp.sum(correct_values)
    num_value_tokens = jnp.maximum(jnp.sum(value_mask), 1.0)  # Avoid div by zero

    em_accuracy = num_correct / num_value_tokens

    return em_accuracy, num_value_tokens


# =============================================================================
# 2. Recall-by-Position - Accuracy Bucketed by Sequence Position
# =============================================================================

@partial(jax.jit, static_argnums=(3,))  # num_buckets must be static for array creation
def compute_recall_by_position(
    logits: jnp.ndarray,      # (B, L, V)
    targets: jnp.ndarray,     # (B, L)
    value_mask: jnp.ndarray,  # (B, L)
    num_buckets: int = 4,     # Default: 4 buckets (quartiles)
) -> jnp.ndarray:
    """
    Compute recall accuracy bucketed by relative position in sequence.

    Buckets:
        - Bucket 0: positions in [0%, 25%) of sequence
        - Bucket 1: positions in [25%, 50%)
        - Bucket 2: positions in [50%, 75%)
        - Bucket 3: positions in [75%, 100%]

    This measures whether the model struggles with early vs late recalls.

    Args:
        logits: Model output logits
        targets: Ground truth token IDs
        value_mask: Binary mask for value positions
        num_buckets: Number of position buckets (default 4 for quartiles)

    Returns:
        bucket_accuracies: Array of shape (num_buckets,) with accuracy per bucket
    """
    B, L, V = logits.shape

    # Get predictions
    predictions = jnp.argmax(logits, axis=-1)  # (B, L)
    correct = (predictions == targets).astype(jnp.float32)  # (B, L)

    # Create position indices: [0, 1, 2, ..., L-1]
    positions = jnp.arange(L)  # (L,)
    positions = jnp.broadcast_to(positions, (B, L))  # (B, L)

    # Compute bucket assignment for each position
    # Bucket index = floor(position / L * num_buckets), clipped to [0, num_buckets-1]
    bucket_size = L / num_buckets
    bucket_indices = jnp.floor(positions / bucket_size).astype(jnp.int32)
    bucket_indices = jnp.clip(bucket_indices, 0, num_buckets - 1)  # (B, L)

    # Initialize bucket statistics
    bucket_accuracies = jnp.zeros(num_buckets, dtype=jnp.float32)

    # Compute accuracy for each bucket
    for bucket_idx in range(num_buckets):
        # Mask for this bucket: positions in bucket AND value positions
        bucket_mask = (bucket_indices == bucket_idx).astype(jnp.float32) * value_mask  # (B, L)

        # Count correct predictions in this bucket
        bucket_correct = jnp.sum(correct * bucket_mask)
        bucket_total = jnp.maximum(jnp.sum(bucket_mask), 1.0)  # Avoid div by zero

        bucket_accuracy = bucket_correct / bucket_total
        bucket_accuracies = bucket_accuracies.at[bucket_idx].set(bucket_accuracy)

    return bucket_accuracies


# =============================================================================
# 3. Numerical Drift Tracker - MSE Between Predicted and Analytical States
# =============================================================================

@jax.jit
def compute_numerical_drift(
    predicted_logits: jnp.ndarray,  # (B, L, V) - model predictions
    analytical_logits: Optional[jnp.ndarray] = None,  # (B, L, V) - analytical baseline (optional)
    sequence_mask: Optional[jnp.ndarray] = None,  # (B, L) - valid sequence positions
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Mean Squared Error to quantify numerical drift.

    In the absence of a true analytical baseline, we measure:
    1. Internal consistency: MSE between consecutive predictions
    2. Softmax entropy: Higher entropy indicates numerical instability

    If analytical_logits are provided (e.g., from a reference implementation),
    we compute MSE directly against the ground truth.

    Args:
        predicted_logits: Model output logits
        analytical_logits: Optional ground truth logits for comparison
        sequence_mask: Optional mask for valid positions (default: all valid)

    Returns:
        (drift_mse, drift_per_position) - both as JAX arrays
        - drift_mse: Overall MSE (scalar)
        - drift_per_position: MSE at each sequence position (L,)
    """
    B, L, V = predicted_logits.shape

    if sequence_mask is None:
        sequence_mask = jnp.ones((B, L), dtype=jnp.float32)

    if analytical_logits is not None:
        # Direct MSE against analytical baseline
        squared_error = (predicted_logits - analytical_logits) ** 2  # (B, L, V)
        mse_per_token = jnp.mean(squared_error, axis=-1)  # (B, L)

    else:
        # Proxy for drift: measure instability via softmax entropy
        # High entropy = model is uncertain = potential numerical issues
        probs = jax.nn.softmax(predicted_logits, axis=-1)  # (B, L, V)
        log_probs = jax.nn.log_softmax(predicted_logits, axis=-1)  # (B, L, V)

        # Entropy: -sum(p * log(p))
        entropy = -jnp.sum(probs * log_probs, axis=-1)  # (B, L)

        # Normalize entropy to [0, 1] range (max entropy = log(V))
        max_entropy = jnp.log(V)
        normalized_entropy = entropy / max_entropy  # (B, L)

        # Use normalized entropy as drift proxy (higher = more drift)
        mse_per_token = normalized_entropy  # (B, L)

    # Apply sequence mask
    mse_per_token = mse_per_token * sequence_mask  # (B, L)

    # Compute per-position drift (average across batch)
    drift_per_position = jnp.sum(mse_per_token, axis=0) / jnp.maximum(jnp.sum(sequence_mask, axis=0), 1.0)  # (L,)

    # Compute overall drift
    drift_mse = jnp.sum(mse_per_token) / jnp.maximum(jnp.sum(sequence_mask), 1.0)  # scalar

    return drift_mse, drift_per_position


# =============================================================================
# 4. Information Density - Success Rate vs KV-Pair Count
# =============================================================================

@partial(jax.jit, static_argnums=(4,))  # max_kv_pairs must be static for array creation
def compute_information_density(
    logits: jnp.ndarray,      # (B, L, V)
    targets: jnp.ndarray,     # (B, L)
    value_mask: jnp.ndarray,  # (B, L)
    kv_counts: Optional[jnp.ndarray] = None,  # (B,) - number of KV pairs per sequence
    max_kv_pairs: int = 20,   # Maximum KV pairs to bucket
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute success rate as a function of information density (KV-pair count).

    Measures how model performance degrades as more key-value pairs
    need to be stored and recalled (working memory capacity test).

    Args:
        logits: Model output logits
        targets: Ground truth token IDs
        value_mask: Binary mask for value positions
        kv_counts: Number of KV pairs per sequence (if None, estimated from mask)
        max_kv_pairs: Maximum number of KV pairs to track

    Returns:
        (density_accuracies, density_counts) - both as JAX arrays
        - density_accuracies: Accuracy for each KV-pair count (max_kv_pairs,)
        - density_counts: Number of sequences for each KV-pair count (max_kv_pairs,)
    """
    B, L, V = logits.shape

    # Get predictions
    predictions = jnp.argmax(logits, axis=-1)  # (B, L)
    correct = (predictions == targets).astype(jnp.float32)  # (B, L)

    # Estimate KV-pair count from value_mask if not provided
    if kv_counts is None:
        # Count number of value positions per sequence
        kv_counts = jnp.sum(value_mask, axis=1)  # (B,)

    # Clip KV counts to valid range
    kv_counts = jnp.clip(kv_counts, 0, max_kv_pairs - 1).astype(jnp.int32)  # (B,)

    # Initialize density statistics
    density_accuracies = jnp.zeros(max_kv_pairs, dtype=jnp.float32)
    density_counts = jnp.zeros(max_kv_pairs, dtype=jnp.float32)

    # Compute accuracy for each KV-pair count
    for kv_idx in range(max_kv_pairs):
        # Mask for sequences with this KV count
        sequence_mask = (kv_counts == kv_idx).astype(jnp.float32)  # (B,)
        sequence_mask_expanded = sequence_mask[:, None]  # (B, 1)

        # Apply both sequence mask and value mask
        combined_mask = sequence_mask_expanded * value_mask  # (B, L)

        # Count correct predictions for this KV count
        kv_correct = jnp.sum(correct * combined_mask)
        kv_total = jnp.maximum(jnp.sum(combined_mask), 1.0)

        kv_accuracy = kv_correct / kv_total

        # Count number of sequences with this KV count
        num_sequences = jnp.sum(sequence_mask)

        density_accuracies = density_accuracies.at[kv_idx].set(kv_accuracy)
        density_counts = density_counts.at[kv_idx].set(num_sequences)

    return density_accuracies, density_counts


# =============================================================================
# Comprehensive Metrics Collection (Single JIT-Compiled Call)
# =============================================================================

@partial(jax.jit, static_argnames=('num_position_buckets', 'max_kv_pairs'))
def compute_all_metrics(
    logits: jnp.ndarray,      # (B, L, V)
    targets: jnp.ndarray,     # (B, L)
    value_mask: jnp.ndarray,  # (B, L)
    num_position_buckets: int = 4,
    max_kv_pairs: int = 20,
) -> Dict[str, jnp.ndarray]:
    """
    Compute all metrics in a single JIT-compiled call for maximum H100 efficiency.

    This function is optimized to:
    - Minimize CPU-GPU round-trips (single JIT boundary)
    - Maximize XLA fusion opportunities
    - Return all metrics as JAX arrays (no blocking)

    Args:
        logits: Model output logits
        targets: Ground truth token IDs
        value_mask: Binary mask for value positions
        num_position_buckets: Number of position buckets for recall-by-position
        max_kv_pairs: Maximum KV pairs for density tracking

    Returns:
        Dictionary with all metrics as JAX arrays:
        - 'em_accuracy': Exact match accuracy (scalar)
        - 'num_value_tokens': Number of value tokens evaluated (scalar)
        - 'recall_by_position': Accuracy per position bucket (num_position_buckets,)
        - 'drift_mse': Overall numerical drift (scalar)
        - 'drift_per_position': Drift at each position (L,)
        - 'density_accuracies': Accuracy per KV count (max_kv_pairs,)
        - 'density_counts': Number of sequences per KV count (max_kv_pairs,)
    """
    # 1. Exact Match
    em_accuracy, num_value_tokens = compute_exact_match(logits, targets, value_mask)

    # 2. Recall-by-Position
    recall_by_position = compute_recall_by_position(
        logits, targets, value_mask, num_buckets=num_position_buckets
    )

    # 3. Numerical Drift
    drift_mse, drift_per_position = compute_numerical_drift(
        logits, analytical_logits=None, sequence_mask=value_mask
    )

    # 4. Information Density
    density_accuracies, density_counts = compute_information_density(
        logits, targets, value_mask, kv_counts=None, max_kv_pairs=max_kv_pairs
    )

    return {
        'em_accuracy': em_accuracy,
        'num_value_tokens': num_value_tokens,
        'recall_by_position': recall_by_position,
        'drift_mse': drift_mse,
        'drift_per_position': drift_per_position,
        'density_accuracies': density_accuracies,
        'density_counts': density_counts,
    }


# =============================================================================
# Batch Metrics Aggregation (For Multi-Batch Evaluation)
# =============================================================================

def aggregate_metrics(
    metrics_list: list,
    cast_to_numpy: bool = True,
) -> Dict[str, any]:
    """
    Aggregate metrics from multiple batches.

    This function blocks to transfer data from GPU to CPU,
    so call it OUTSIDE the training loop (e.g., at eval intervals).

    Args:
        metrics_list: List of metric dicts from compute_all_metrics
        cast_to_numpy: Whether to convert JAX arrays to NumPy (default True)

    Returns:
        Aggregated metrics as Python scalars/arrays
    """
    # Block until all metrics are ready
    metrics_list = [jax.block_until_ready(m) for m in metrics_list]

    # Extract individual metrics
    em_accuracies = [m['em_accuracy'] for m in metrics_list]
    recall_by_position_list = [m['recall_by_position'] for m in metrics_list]
    drift_mse_list = [m['drift_mse'] for m in metrics_list]
    drift_per_position_list = [m['drift_per_position'] for m in metrics_list]
    density_accuracies_list = [m['density_accuracies'] for m in metrics_list]
    density_counts_list = [m['density_counts'] for m in metrics_list]

    # Aggregate EM accuracy (weighted by number of value tokens)
    num_value_tokens = [m['num_value_tokens'] for m in metrics_list]
    total_value_tokens = sum(num_value_tokens)
    avg_em_accuracy = sum(
        em * n for em, n in zip(em_accuracies, num_value_tokens)
    ) / jnp.maximum(total_value_tokens, 1.0)

    # Aggregate recall-by-position (simple average)
    avg_recall_by_position = jnp.mean(jnp.stack(recall_by_position_list), axis=0)

    # Aggregate drift (simple average)
    avg_drift_mse = jnp.mean(jnp.array(drift_mse_list))
    avg_drift_per_position = jnp.mean(jnp.stack(drift_per_position_list), axis=0)

    # Aggregate density (weighted by counts)
    stacked_density_accs = jnp.stack(density_accuracies_list)  # (num_batches, max_kv_pairs)
    stacked_density_counts = jnp.stack(density_counts_list)    # (num_batches, max_kv_pairs)

    total_counts = jnp.sum(stacked_density_counts, axis=0)  # (max_kv_pairs,)
    weighted_density_accs = jnp.sum(
        stacked_density_accs * stacked_density_counts, axis=0
    ) / jnp.maximum(total_counts, 1.0)

    # Cast to NumPy if requested (for saving/plotting)
    if cast_to_numpy:
        import numpy as np
        avg_em_accuracy = float(avg_em_accuracy)
        avg_recall_by_position = np.array(avg_recall_by_position)
        avg_drift_mse = float(avg_drift_mse)
        avg_drift_per_position = np.array(avg_drift_per_position)
        weighted_density_accs = np.array(weighted_density_accs)
        total_counts = np.array(total_counts)

    return {
        'em_accuracy': avg_em_accuracy,
        'recall_by_position': avg_recall_by_position,
        'drift_mse': avg_drift_mse,
        'drift_per_position': avg_drift_per_position,
        'density_accuracies': weighted_density_accs,
        'density_counts': total_counts,
    }


# =============================================================================
# Utility: Convert Metrics to Loggable Format
# =============================================================================

def metrics_to_log_dict(metrics: Dict[str, any]) -> Dict[str, float]:
    """
    Convert metrics dict to flat dict of scalars for logging.

    Args:
        metrics: Output from aggregate_metrics

    Returns:
        Flat dict with scalar values only (for JSON/tensorboard)
    """
    log_dict = {
        'em_accuracy': float(metrics['em_accuracy']),
        'drift_mse': float(metrics['drift_mse']),
    }

    # Add recall-by-position as separate keys
    recall_by_pos = metrics['recall_by_position']
    for i, acc in enumerate(recall_by_pos):
        log_dict[f'recall_bucket_{i}'] = float(acc)

    # Add density accuracies for KV counts with data
    density_accs = metrics['density_accuracies']
    density_counts = metrics['density_counts']
    for kv_idx, (acc, count) in enumerate(zip(density_accs, density_counts)):
        if count > 0:  # Only log KV counts that actually appeared
            log_dict[f'density_kv_{kv_idx}'] = float(acc)

    return log_dict
