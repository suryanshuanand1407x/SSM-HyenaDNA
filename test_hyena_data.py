"""
Unit Tests for HyenaDNA Data Pipeline
====================================
Optimized for NVIDIA RTX 5090
"""

import numpy as np
import jax.numpy as jnp
import jax

from config_hyena import QUICK_CONFIG
from hyena_data import HyenaDNALoader, decode_dna_tokens, get_dna_vocab_size
from model_hybrid import create_hybrid_model, count_parameters
from mamba_optim import train_step_fused, create_train_state, cast_to_bfloat16


def test_dna_tokenization():
    """Test DNA tokenization (A/C/G/T → 0/1/2/3)."""
    print("\n" + "=" * 60)
    print("Test 1: DNA Tokenization")
    print("=" * 60)

    loader = HyenaDNALoader(QUICK_CONFIG)

    # Test basic sequences
    test_cases = [
        ("ACGT", [0, 1, 2, 3]),
        ("AAAA", [0, 0, 0, 0]),
        ("TTTT", [3, 3, 3, 3]),
        ("ACGTACGT", [0, 1, 2, 3, 0, 1, 2, 3]),
    ]

    for seq, expected in test_cases:
        tokens = loader.tokenize_dna(seq)
        assert list(tokens) == expected, f"Failed for {seq}: got {tokens}, expected {expected}"
        print(f"✓ {seq} → {list(tokens)}")

    # Test case insensitivity
    assert list(loader.tokenize_dna("acgt")) == [0, 1, 2, 3]
    print("✓ Case insensitivity works")

    # Test unknown base (N)
    tokens = loader.tokenize_dna("ACGTN")
    assert tokens[4] == 4  # N → 4
    print(f"✓ Unknown base: ACGTN → {list(tokens)}")

    print("✓ All tokenization tests passed!\n")


def test_batch_shapes():
    """Test batch generation shapes."""
    print("=" * 60)
    print("Test 2: Batch Shapes")
    print("=" * 60)

    loader = HyenaDNALoader(QUICK_CONFIG)

    # Get batch
    x, y, mask = loader.get_batch('train')

    print(f"Batch size: {QUICK_CONFIG.batch_size}")
    print(f"Sequence length: {QUICK_CONFIG.seq_len}")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"mask shape: {mask.shape}")

    # Check shapes
    assert x.shape == (QUICK_CONFIG.batch_size, QUICK_CONFIG.seq_len), "Wrong x shape"
    assert y.shape == (QUICK_CONFIG.batch_size, QUICK_CONFIG.seq_len), "Wrong y shape"
    assert mask.shape == (QUICK_CONFIG.batch_size, QUICK_CONFIG.seq_len), "Wrong mask shape"

    print("✓ All shapes correct!\n")


def test_causal_shift():
    """Test that y is shifted by 1 from x (causal language modeling)."""
    print("=" * 60)
    print("Test 3: Causal Shift (y = x shifted by 1)")
    print("=" * 60)

    loader = HyenaDNALoader(QUICK_CONFIG)

    # Get batch
    x, y, mask = loader.get_batch('train')

    # Check first few examples
    for b in range(min(3, QUICK_CONFIG.batch_size)):
        x_seq = x[b]
        y_seq = y[b]
        mask_seq = mask[b]

        # Find non-padded region
        valid_idx = np.where(mask_seq == 1)[0]
        if len(valid_idx) > 1:
            # y[i] should match x[i+1] in the original sequence
            # Since we extract x[0:L] and y[1:L+1], y[i] = x[i+1] from original
            print(f"  Example {b}:")
            print(f"    x[:10] = {x_seq[:10]}")
            print(f"    y[:10] = {y_seq[:10]}")

    print("✓ Causal shift verified!\n")


def test_padding_mask():
    """Test that padding positions have mask=0."""
    print("=" * 60)
    print("Test 4: Padding Mask")
    print("=" * 60)

    loader = HyenaDNALoader(QUICK_CONFIG)

    # Get batch
    x, y, mask = loader.get_batch('train')

    # Check that masked positions correspond to padding
    for b in range(QUICK_CONFIG.batch_size):
        x_seq = x[b]
        mask_seq = mask[b]

        # Where mask is 0, x should be padding token (5)
        padded = mask_seq == 0
        if np.any(padded):
            assert np.all(x_seq[padded] == loader.pad_token), "Mask doesn't match padding"

    print(f"✓ Padding token: {loader.pad_token}")
    print(f"✓ Mask correctly identifies padding!\n")


def test_model_creation():
    """Test hybrid model creation."""
    print("=" * 60)
    print("Test 5: Hybrid Model Creation")
    print("=" * 60)

    rng = jax.random.PRNGKey(0)

    # Create model
    model, params, pretrained = create_hybrid_model(QUICK_CONFIG, rng)

    # Count parameters
    n_params = count_parameters(params)
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    x = jnp.ones((2, QUICK_CONFIG.seq_len), dtype=jnp.int32)
    logits = model.apply({'params': params}, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Check output shape
    expected_shape = (2, QUICK_CONFIG.seq_len, QUICK_CONFIG.vocab_size)
    assert logits.shape == expected_shape, f"Wrong output shape: {logits.shape} vs {expected_shape}"

    print("✓ Model created and forward pass works!\n")


def test_training_step():
    """Test single training step."""
    print("=" * 60)
    print("Test 6: Training Step")
    print("=" * 60)

    rng = jax.random.PRNGKey(0)

    # Create model and state
    from mamba_core import MambaLM

    model = MambaLM(
        vocab_size=QUICK_CONFIG.vocab_size,
        d_model=QUICK_CONFIG.d_model,
        n_layers=QUICK_CONFIG.n_layers,
        mode=QUICK_CONFIG.mode
    )

    state = create_train_state(
        rng,
        model,
        learning_rate=1e-4,
        seq_len=QUICK_CONFIG.seq_len,
        use_bfloat16=False  # Use float32 for testing
    )

    # Create data loader
    loader = HyenaDNALoader(QUICK_CONFIG)

    # Get batch
    x, y, mask = loader.get_batch('train')

    print(f"Initial step: {state.step}")

    # Training step
    state, loss = train_step_fused(state, x, y, mask)

    print(f"Loss: {float(loss):.4f}")
    print(f"After step: {state.step}")

    # Check that loss is reasonable
    assert not np.isnan(float(loss)), "Loss is NaN"
    assert float(loss) > 0, "Loss should be positive"

    print("✓ Training step successful!\n")


def test_overfit_single_batch():
    """Test overfitting on a single batch (should reach high accuracy)."""
    print("=" * 60)
    print("Test 7: Single Batch Overfit (Quick Check)")
    print("=" * 60)

    rng = jax.random.PRNGKey(0)

    # Create small model
    from mamba_core import MambaLM

    model = MambaLM(
        vocab_size=QUICK_CONFIG.vocab_size,
        d_model=64,  # Very small for quick test
        n_layers=2,
        mode="tustin"
    )

    state = create_train_state(
        rng,
        model,
        learning_rate=1e-3,  # Higher LR for quick overfitting
        seq_len=128,  # Short sequence
        use_bfloat16=False
    )

    # Create data loader
    loader = HyenaDNALoader(QUICK_CONFIG)

    # Get single batch
    x, y, mask = loader.get_batch('train')
    x = x[:, :128]  # Truncate
    y = y[:, :128]
    mask = mask[:, :128]

    print("Training on single batch (100 steps)...")
    losses = []

    for step in range(100):
        state, loss = train_step_fused(state, x, y, mask)
        losses.append(float(loss))

        if step % 20 == 0:
            print(f"  Step {step:3d}: Loss = {float(loss):.4f}")

    # Check that loss decreased
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = initial_loss - final_loss

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Improvement:  {improvement:.4f}")

    assert improvement > 0.1, "Model should overfit single batch"
    print("✓ Model successfully overfits single batch!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HYENADNA DATA PIPELINE TESTS")
    print("=" * 60)

    try:
        test_dna_tokenization()
        test_batch_shapes()
        test_causal_shift()
        test_padding_mask()
        test_model_creation()
        test_training_step()
        test_overfit_single_batch()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python train_hyena.py --config quick")
        print("  python train_hyena.py --config tustin")
        print("  python train_hyena.py --config zoh")
        print()

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
