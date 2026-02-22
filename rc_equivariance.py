"""
Reverse Complement (RC) Equivariance for Genomic Sequences
===========================================================
Implements RC equivariance so DNA sequences and their reverse complements
are treated identically by the model.

Biological Context:
- DNA is double-stranded: 5'-ATCG-3' pairs with 3'-TAGC-5'
- Functionally equivalent, should produce same embeddings
- Critical for genomics, not needed for general sequence modeling
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


# =============================================================================
# DNA Complement Mapping
# =============================================================================

# DNA complement table: A<->T, C<->G, N->N
DNA_COMPLEMENT = {
    0: 3,  # A -> T
    1: 2,  # C -> G
    2: 1,  # G -> C
    3: 0,  # T -> A
    4: 4,  # N -> N (unknown stays unknown)
    5: 5,  # PAD -> PAD
}


def reverse_complement_tokens(seq: jnp.ndarray) -> jnp.ndarray:
    """
    Compute reverse complement of tokenized DNA sequence.

    Args:
        seq: (B, L) or (L,) tokenized DNA sequence
            Tokens: A=0, C=1, G=2, T=3, N=4, PAD=5

    Returns:
        rc_seq: Reverse complement sequence (same shape)

    Example:
        Input:  [0, 3, 1, 2]  # ATCG
        Output: [2, 1, 0, 3]  # CGAT (reverse of TAGC)
    """
    # Create lookup table for complement
    complement_map = jnp.array([3, 2, 1, 0, 4, 5], dtype=jnp.int32)

    # Apply complement
    complement_seq = complement_map[seq]

    # Reverse
    rc_seq = jnp.flip(complement_seq, axis=-1)

    return rc_seq


# =============================================================================
# RC Equivariant Data Augmentation
# =============================================================================

@jax.jit
def rc_augment_batch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Augment batch with reverse complement sequences.

    For each sequence, randomly choose original or RC with 50% probability.

    Args:
        x: (B, L) input sequences
        y: (B, L) target sequences
        mask: (B, L) loss mask
        key: Random key

    Returns:
        x_aug: Augmented input (same shape)
        y_aug: Augmented targets (same shape)
        mask_aug: Augmented mask (same shape)
    """
    B, L = x.shape

    # Random choice: 0 = original, 1 = RC
    use_rc = jax.random.bernoulli(key, 0.5, shape=(B,))

    def apply_rc_if_selected(i):
        """Apply RC to sequence i if selected."""
        seq_x = x[i]
        seq_y = y[i]
        seq_mask = mask[i]

        # Compute RC
        rc_x = reverse_complement_tokens(seq_x)
        rc_y = reverse_complement_tokens(seq_y)
        rc_mask = jnp.flip(seq_mask, axis=-1)

        # Select original or RC
        return jax.lax.cond(
            use_rc[i],
            lambda: (rc_x, rc_y, rc_mask),  # Use RC
            lambda: (seq_x, seq_y, seq_mask)  # Use original
        )

    # Apply to all sequences in batch
    x_aug, y_aug, mask_aug = jax.vmap(apply_rc_if_selected)(jnp.arange(B))

    return x_aug, y_aug, mask_aug


def rc_augment_batch_double(
    x: jnp.ndarray,
    y: jnp.ndarray,
    mask: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Augment batch by ADDING reverse complement sequences.

    Doubles the batch size by concatenating original + RC.

    Args:
        x: (B, L) input sequences
        y: (B, L) target sequences
        mask: (B, L) loss mask

    Returns:
        x_aug: (2B, L) original + RC
        y_aug: (2B, L) targets
        mask_aug: (2B, L) masks
    """
    # Compute RC
    rc_x = jax.vmap(reverse_complement_tokens)(x)
    rc_y = jax.vmap(reverse_complement_tokens)(y)
    rc_mask = jnp.flip(mask, axis=-1)

    # Concatenate
    x_aug = jnp.concatenate([x, rc_x], axis=0)
    y_aug = jnp.concatenate([y, rc_y], axis=0)
    mask_aug = jnp.concatenate([mask, rc_mask], axis=0)

    return x_aug, y_aug, mask_aug


# =============================================================================
# RC Equivariant Embedding Layer
# =============================================================================

def rc_equivariant_embed(
    seq: jnp.ndarray,
    embed_fn,
    mode: str = "symmetric"
) -> jnp.ndarray:
    """
    RC-equivariant embedding layer.

    Ensures embed(seq) ≈ reverse(embed(rc(seq)))

    Args:
        seq: (B, L) tokenized sequence
        embed_fn: Embedding function (seq -> embeddings)
        mode: "symmetric" or "learned"

    Returns:
        embeddings: (B, L, D) RC-equivariant embeddings
    """
    # Forward embedding
    fwd_embed = embed_fn(seq)

    if mode == "symmetric":
        # Compute RC embedding
        rc_seq = jax.vmap(reverse_complement_tokens)(seq)
        rc_embed = embed_fn(rc_seq)

        # Symmetric combination: average forward and reversed-RC
        rc_embed_reversed = jnp.flip(rc_embed, axis=1)
        embeddings = (fwd_embed + rc_embed_reversed) / 2.0

    elif mode == "learned":
        # Just use forward (model learns RC invariance from data)
        embeddings = fwd_embed

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return embeddings


# =============================================================================
# RC Equivariant Loss
# =============================================================================

@jax.jit
def rc_consistency_loss(
    model_fn,
    params,
    x: jnp.ndarray,
    train: bool = True
) -> jnp.ndarray:
    """
    RC consistency loss: ||f(x) - reverse(f(rc(x)))||²

    Encourages model to produce consistent outputs for sequence and RC.

    Args:
        model_fn: Model forward function
        params: Model parameters
        x: (B, L) input sequence
        train: Training mode

    Returns:
        loss: Scalar consistency loss
    """
    # Forward pass
    fwd_logits = model_fn({'params': params}, x, train=train)

    # RC forward pass
    rc_x = jax.vmap(reverse_complement_tokens)(x)
    rc_logits = model_fn({'params': params}, rc_x, train=train)

    # Reverse the RC logits (to align with forward)
    rc_logits_reversed = jnp.flip(rc_logits, axis=1)

    # Consistency loss (MSE between forward and reversed-RC)
    diff = fwd_logits - rc_logits_reversed
    loss = jnp.mean(diff ** 2)

    return loss


# =============================================================================
# Integration with Training
# =============================================================================

def create_rc_aware_loss(
    base_loss_fn,
    rc_weight: float = 0.1
):
    """
    Create loss function with RC consistency term.

    Args:
        base_loss_fn: Standard cross-entropy loss function
        rc_weight: Weight for RC consistency term (default 0.1)

    Returns:
        combined_loss_fn: Loss function with RC consistency
    """
    def combined_loss(model_fn, params, x, y, mask, train=True):
        # Standard CE loss
        ce_loss = base_loss_fn(model_fn, params, x, y, mask, train)

        # RC consistency loss
        rc_loss = rc_consistency_loss(model_fn, params, x, train)

        # Combine
        total_loss = ce_loss + rc_weight * rc_loss

        return total_loss

    return combined_loss


# =============================================================================
# Reverse Complement Data Loader Wrapper
# =============================================================================

class RCDataLoader:
    """
    Wrapper for data loader that adds RC augmentation.

    Usage:
        base_loader = HG38DataLoader(config)
        rc_loader = RCDataLoader(base_loader, mode='random')

        x, y, mask = rc_loader.get_batch('train')
        # x now includes random RC augmentation
    """

    def __init__(
        self,
        base_loader,
        mode: str = 'random',
        seed: int = 42
    ):
        """
        Initialize RC data loader.

        Args:
            base_loader: Base data loader (e.g., HG38DataLoader)
            mode: Augmentation mode:
                - 'none': No augmentation (pass-through)
                - 'random': Randomly flip sequences to RC (50%)
                - 'double': Add RC as extra samples (2x batch size)
            seed: Random seed
        """
        self.base_loader = base_loader
        self.mode = mode
        self.key = jax.random.PRNGKey(seed)

    def get_batch(self, split: str = 'train'):
        """
        Get batch with RC augmentation.

        Args:
            split: 'train', 'validation', or 'test'

        Returns:
            x, y, mask: Augmented batch
        """
        # Get base batch
        x, y, mask = self.base_loader.get_batch(split)

        # Apply RC augmentation
        if self.mode == 'none':
            return x, y, mask

        elif self.mode == 'random':
            # Random RC flip (50% probability)
            self.key, subkey = jax.random.split(self.key)
            x_aug, y_aug, mask_aug = rc_augment_batch(x, y, mask, subkey)
            return x_aug, y_aug, mask_aug

        elif self.mode == 'double':
            # Double batch size with RC
            x_aug, y_aug, mask_aug = rc_augment_batch_double(x, y, mask)
            return x_aug, y_aug, mask_aug

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# Testing & Validation
# =============================================================================

def test_rc_equivariance():
    """Test RC transformations."""
    print("Testing RC Equivariance...")

    # Test sequence: ATCGN
    seq = jnp.array([[0, 3, 1, 2, 4]])  # (1, 5)

    # Compute RC
    rc_seq = reverse_complement_tokens(seq)

    print(f"Original:  {seq}")
    print(f"RC:        {rc_seq}")

    # Expected: ATCGN -> NCGAT
    # Step 1 (complement): ATCGN -> TAGCN = [3,0,2,1,4]
    # Step 2 (reverse): TAGCN -> NCGAT = [4,1,2,0,3]
    expected = jnp.array([[4, 1, 2, 0, 3]])

    assert jnp.array_equal(rc_seq, expected), "RC transformation failed!"
    print("✓ RC transformation correct")

    # Test double RC = identity
    rc_rc_seq = reverse_complement_tokens(rc_seq)
    assert jnp.array_equal(rc_rc_seq, seq), "RC(RC) != identity!"
    print("✓ RC(RC) = identity")

    print("\n✓ All RC tests passed!")


if __name__ == "__main__":
    test_rc_equivariance()
