"""
Hybrid Model: HyenaDNA Embeddings + Mamba Blocks
==============================================
Load pre-trained HyenaDNA weights and replace Hyena operators with Mamba blocks.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, Optional
import numpy as np

from mamba_core import MambaLM
from config_hyena import HyenaFineTuneConfig


class HybridHyenaMamba(nn.Module):
    """
    Hybrid model: HyenaDNA embeddings + Mamba blocks + HyenaDNA output head.

    Architecture:
        1. Token Embedding (from HyenaDNA pre-trained)
        2. Mamba Blocks (randomly initialized)
        3. Output Layer (from HyenaDNA or random)
    """
    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mode: str = "tustin"  # "tustin" or "zoh"

    @nn.compact
    def __call__(self, x, use_parallel=True, train=True):
        """
        Forward pass.

        Args:
            x: (B, L) token IDs

        Returns:
            logits: (B, L, vocab_size)
        """
        # Use MambaLM directly (it has embedding + mamba stack + output)
        # We'll load HyenaDNA weights into the embedding layer separately
        model = MambaLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            mode=self.mode
        )
        return model(x, use_parallel=use_parallel, train=train)


def load_hyenadna_pretrained(config: HyenaFineTuneConfig):
    """
    Load HyenaDNA pre-trained model from HuggingFace.

    Args:
        config: Fine-tuning configuration

    Returns:
        pretrained_weights: Dictionary with embeddings and layer norms
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading HyenaDNA model: {config.pretrained_model}")

        # Load HyenaDNA model from HuggingFace
        model = AutoModel.from_pretrained(
            config.pretrained_model,
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model,
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )

        print(f"HyenaDNA model loaded successfully")
        print(f"Model vocab size: {len(tokenizer)}")
        print(f"Model hidden size: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'unknown'}")

        # Extract embeddings (convert PyTorch → NumPy → JAX)
        pretrained_weights = {}

        # Token embeddings
        if hasattr(model, 'embeddings'):
            embeddings = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
            pretrained_weights['embeddings'] = embeddings
            print(f"Extracted embeddings: shape {embeddings.shape}")

        # Layer norms (if available)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            layer_norms = []
            for layer in model.encoder.layer:
                if hasattr(layer, 'LayerNorm'):
                    ln_weight = layer.LayerNorm.weight.detach().cpu().numpy()
                    ln_bias = layer.LayerNorm.bias.detach().cpu().numpy()
                    layer_norms.append({'weight': ln_weight, 'bias': ln_bias})
            if layer_norms:
                pretrained_weights['layer_norms'] = layer_norms
                print(f"Extracted {len(layer_norms)} layer norms")

        return pretrained_weights

    except Exception as e:
        print(f"WARNING: Could not load HyenaDNA weights: {e}")
        print("Will use random initialization for all layers")
        return {}


def inject_hyenadna_weights(
    params: Dict[str, Any],
    pretrained_weights: Dict[str, Any],
    config: HyenaFineTuneConfig
) -> Dict[str, Any]:
    """
    Inject HyenaDNA pre-trained weights into Mamba model parameters.

    Args:
        params: Mamba model parameters (randomly initialized)
        pretrained_weights: HyenaDNA pre-trained weights
        config: Configuration

    Returns:
        params: Updated parameters with HyenaDNA embeddings
    """
    if not pretrained_weights:
        print("No pre-trained weights to inject")
        return params

    # Inject token embeddings
    if config.load_embeddings and 'embeddings' in pretrained_weights:
        embeddings = pretrained_weights['embeddings']

        # Find embedding layer in params
        # Structure: params['Embed_0']['embedding']
        if 'Embed_0' in params and 'embedding' in params['Embed_0']:
            current_shape = params['Embed_0']['embedding'].shape
            pretrained_shape = embeddings.shape

            print(f"Injecting embeddings:")
            print(f"  Current shape: {current_shape}")
            print(f"  Pretrained shape: {pretrained_shape}")

            # Handle shape mismatch
            if current_shape == pretrained_shape:
                params['Embed_0']['embedding'] = jnp.array(embeddings)
                print("  ✓ Embeddings injected successfully")
            else:
                print(f"  WARNING: Shape mismatch, using initialization")
                # Try to copy what we can
                min_vocab = min(current_shape[0], pretrained_shape[0])
                min_dim = min(current_shape[1], pretrained_shape[1])
                params['Embed_0']['embedding'] = params['Embed_0']['embedding'].at[:min_vocab, :min_dim].set(
                    jnp.array(embeddings[:min_vocab, :min_dim])
                )
                print(f"  Partial injection: copied [{min_vocab}, {min_dim}]")

    # Inject layer norms (if available and requested)
    if config.load_layer_norms and 'layer_norms' in pretrained_weights:
        layer_norms = pretrained_weights['layer_norms']
        print(f"Layer norm injection: {len(layer_norms)} available")
        # TODO: Map to Mamba's RMSNorm if compatible

    return params


def create_hybrid_model(config: HyenaFineTuneConfig, rng: jax.random.PRNGKey):
    """
    Create hybrid model with HyenaDNA embeddings + Mamba blocks.

    Args:
        config: Configuration
        rng: Random key

    Returns:
        model: HybridHyenaMamba instance
        params: Model parameters with injected HyenaDNA weights
        pretrained_weights: Original HyenaDNA weights (for reference)
    """
    # Create model
    model = HybridHyenaMamba(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        mode=config.mode
    )

    # Initialize parameters (random)
    dummy_input = jnp.ones((1, config.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)['params']
    print(f"Model initialized with {config.n_layers} Mamba layers")

    # Load HyenaDNA pre-trained weights
    pretrained_weights = load_hyenadna_pretrained(config)

    # Inject pre-trained weights
    params = inject_hyenadna_weights(params, pretrained_weights, config)

    return model, params, pretrained_weights


def freeze_embeddings(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark embeddings as frozen by creating a frozen copy.

    Note: In JAX/Flax, we handle freezing through optax's masking.
    This function returns a mask indicating which parameters to freeze.

    Args:
        params: Model parameters

    Returns:
        freeze_mask: Dictionary with same structure, True=freeze, False=train
    """
    import jax.tree_util as tree_util

    # Create mask: freeze embeddings, train everything else
    def create_mask(path, value):
        # Freeze if this is the embedding layer
        if 'Embed_0' in path:
            return True  # Freeze
        return False  # Train

    # Traverse parameter tree
    def mask_tree(params, path=''):
        if isinstance(params, dict):
            return {k: mask_tree(v, f"{path}/{k}") for k, v in params.items()}
        else:
            return create_mask(path, params)

    freeze_mask = mask_tree(params)
    return freeze_mask


def count_parameters(params: Dict[str, Any]) -> int:
    """Count total parameters in model."""
    import jax.tree_util as tree_util
    return sum(x.size for x in tree_util.tree_leaves(params))
