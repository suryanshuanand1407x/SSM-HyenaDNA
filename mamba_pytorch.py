"""
Mamba Model - PyTorch Implementation
For RTX 5090 GPU Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    """
    Mamba block with SSM (State Space Model).
    PyTorch version for GPU compatibility.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, mode="tustin"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.mode = mode

        # Linear projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM initialization
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        """
        Args:
            x: (B, L, D) tensor
        Returns:
            (B, L, D) tensor
        """
        B, L, D = x.shape

        # Input projection: split into x and z
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)

        # Convolution (requires channel-first)
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Trim padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # SSM computation
        y = self.ssm(x_conv)

        # Gating
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """
        Selective State Space Model.

        Args:
            x: (B, L, d_inner)
        Returns:
            (B, L, d_inner)
        """
        B, L, D = x.shape

        # Get dt, B, C parameters
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)

        BC = self.x_proj(x)  # (B, L, 2*d_state)
        B_ssm, C_ssm = BC.chunk(2, dim=-1)  # Each (B, L, d_state)

        # Discretize A
        A = -torch.exp(self.A_log.float())  # (d_state,)
        A = repeat(A, 'n -> d n', d=self.d_inner)  # (d_inner, d_state)

        # Discretization: Tustin or ZOH
        if self.mode == "tustin":
            # Tustin bilinear transform
            A_discrete = (2 + dt.unsqueeze(-1) * A) / (2 - dt.unsqueeze(-1) * A)
            B_discrete = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, d_inner, d_state)
        else:  # zoh
            # Zero-order hold
            A_discrete = torch.exp(dt.unsqueeze(-1) * A)
            B_discrete = (A_discrete - 1) / A * B_ssm.unsqueeze(2)

        # Scan (parallel version using cumsum)
        # Simplified version - full scan would be more complex
        y = torch.einsum('bld,bldn->bln', x, B_discrete)
        y = y * torch.einsum('bln,n->bln', C_ssm, torch.ones(self.d_state, device=x.device))
        y = y.sum(dim=-1)  # (B, L, d_inner)

        # Add skip connection (D parameter)
        y = y + x * self.D

        return y


class MambaLM(nn.Module):
    """
    Mamba Language Model - PyTorch version.
    """

    def __init__(self, vocab_size, d_model, n_layers, d_state=16, d_conv=4, expand=2, mode="tustin"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, mode)
            for _ in range(n_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, x):
        """
        Args:
            x: (B, L) token IDs
        Returns:
            (B, L, vocab_size) logits
        """
        # Embed
        h = self.embedding(x)  # (B, L, D)

        # Apply Mamba layers
        for layer in self.layers:
            h = h + layer(h)  # Residual connection

        # Normalize
        h = self.norm(h)

        # Project to vocab
        logits = self.lm_head(h)

        return logits


def create_mamba_model(config):
    """Create Mamba model from config."""
    model = MambaLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        mode=config.mode
    )
    return model
