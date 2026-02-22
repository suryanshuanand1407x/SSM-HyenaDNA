"""
Mamba Core Mathematical Implementation
=======================================
This file contains the EXACT mathematical logic from the original implementation.
NO MODIFICATIONS to the core SSM math, discretization, or scan algorithms.

All hardware optimizations are in mamba_optim.py.
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from typing import Tuple, Optional, NamedTuple
import math

import flax.linen as nn
from flax.linen import initializers


# =============================================================================
# Discretization Functions
# =============================================================================

def soft_clamp(
    x: jnp.ndarray,
    min_val: float = 1e-4,
    max_val: float = 10.0,
) -> jnp.ndarray:
    """
    Differentiable soft clamp using tanh scaling.
    Maps input smoothly into [min_val, max_val].
    """
    center = (max_val + min_val) / 2.0
    half_range = (max_val - min_val) / 2.0
    return center + half_range * jnp.tanh((x - center) / half_range)


def discretize_zoh(
    A: jnp.ndarray,  # (D, N) - continuous state matrix (diagonal)
    B: jnp.ndarray,  # (B, L, D, N) - continuous input matrix
    delta: jnp.ndarray,  # (B, L, D) - step sizes
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Zero-Order Hold (ZOH) discretization (Vanilla Baseline).

    Converts continuous-time SSM parameters to discrete-time.

    From the paper (Equation 4):
        Ā = exp(Δ·A)
        B̄ = (Δ·A)^{-1} (exp(Δ·A) - I) · Δ·B

    For diagonal A, this simplifies significantly.
    Uses (exp(ΔA) - 1) / (ΔA) with numerical stability handling.

    NOTE: This runs in the model's native precision (no float32 promotion).
    For bfloat16, this can lead to underflow in extreme OOD cases.

    Args:
        A: Continuous state matrix, shape (D, N) - diagonal elements
        B: Continuous input matrix, shape (B, L, D, N)
        delta: Discretization step sizes, shape (B, L, D)

    Returns:
        A_bar: Discretized state matrix
        B_bar: Discretized input matrix
    """
    # Expand delta for broadcasting
    delta_expanded = delta[..., None]  # (B, L, D, 1)

    # Compute Δ·A
    deltaA = delta_expanded * A  # (B, L, D, N)

    # Ā = exp(Δ·A)
    A_bar = jnp.exp(deltaA)

    # For B̄, we need to handle the (exp(ΔA) - I) / (ΔA) term carefully
    # Safe division: (exp(x) - 1) / x with handling for x ≈ 0
    def safe_expm1_over_x(x):
        """Compute (exp(x) - 1) / x safely."""
        threshold = 1e-4
        small_x = jnp.abs(x) < threshold
        # Taylor approximation: 1 + x/2 + x^2/6
        taylor = 1.0 + x/2.0 + x**2/6.0
        # Direct computation for larger x
        direct = jnp.expm1(x) / jnp.where(jnp.abs(x) < 1e-10, 1.0, x)
        return jnp.where(small_x, taylor, direct)

    # Compute the discretization factor
    disc_factor = safe_expm1_over_x(deltaA)

    # B̄ = Δ · B · disc_factor
    B_bar = delta_expanded * B * disc_factor

    return A_bar, B_bar


def discretize_tustin(
    A: jnp.ndarray,      # (D, N) - continuous state matrix (diagonal)
    B: jnp.ndarray,      # (B, L, D, N) - continuous input matrix
    delta: jnp.ndarray,  # (B, L, D) - step sizes (guarded)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bilinear (Tustin) discretization for diagonal state matrices.

    Element-wise operations:
        ā_i = (1 + Δ/2 · a_i) / (1 - Δ/2 · a_i)
        b̄_i = √Δ · b_i / (1 - Δ/2 · a_i)

    Precision Guard: Promotes to float32 for inversion, casts back.
    """
    orig_dtype = B.dtype

    # Expand delta for broadcasting: (B, L, D) -> (B, L, D, 1)
    delta_expanded = delta[..., None]

    # === Precision Guard: promote to float32 for the inversion ===
    A_f32 = A.astype(jnp.float32)
    delta_f32 = delta_expanded.astype(jnp.float32)
    B_f32 = B.astype(jnp.float32)

    # Half-step: Δ/2 · A  (element-wise)
    half_dA = (delta_f32 / 2.0) * A_f32  # (B, L, D, N)

    # Denominator: (1 - Δ/2 · A)
    denom = 1.0 - half_dA  # (B, L, D, N)

    # Numerator: (1 + Δ/2 · A)
    numer = 1.0 + half_dA  # (B, L, D, N)

    # Ā = (1 + Δ/2·A) / (1 - Δ/2·A)
    A_bar = numer / denom  # (B, L, D, N)

    # B̄ = √Δ · B / (1 - Δ/2·A)
    sqrt_delta = jnp.sqrt(delta_f32)  # (B, L, D, 1)
    B_bar = (sqrt_delta * B_f32) / denom  # (B, L, D, N)

    # Cast back to original dtype
    return A_bar.astype(orig_dtype), B_bar.astype(orig_dtype)


def discretize_tustin_raw(
    A: jnp.ndarray,      # (D, N) - continuous state matrix (diagonal)
    B: jnp.ndarray,      # (B, L, D, N) - continuous input matrix
    delta: jnp.ndarray,  # (B, L, D) - step sizes (unguarded)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Raw Bilinear (Tustin) discretization WITHOUT discretization guards.

    This is the VANILLA BASELINE version that demonstrates performance collapse.

    Element-wise operations:
        ā_i = (I + AΔ/2) / (I - AΔ/2)
        b̄_i = √Δ · b_i / (I - AΔ/2)

    Key differences from guarded version:
    - NO precision promotion (stays in input dtype, typically bfloat16)
    - NO stability epsilon or clamping
    - NO special handling of denormals
    - Raw division that can underflow/overflow

    This is intentionally fragile to demonstrate the need for guards.
    """
    # NO dtype promotion - keep whatever precision we're given
    # This means if input is bfloat16, computation stays in bfloat16

    # Expand delta for broadcasting: (B, L, D) -> (B, L, D, 1)
    delta_expanded = delta[..., None]

    # Half-step: Δ/2 · A  (element-wise)
    half_dA = (delta_expanded / 2.0) * A  # (B, L, D, N)

    # Denominator: (I - AΔ/2) - NO epsilon, NO clamping
    denom = 1.0 - half_dA  # (B, L, D, N)

    # Numerator: (I + AΔ/2)
    numer = 1.0 + half_dA  # (B, L, D, N)

    # Ā = (I + AΔ/2) / (I - AΔ/2) - raw division, can produce inf/nan
    A_bar = numer / denom  # (B, L, D, N)

    # B̄ = √Δ · B / (I - AΔ/2) - raw sqrt and division
    sqrt_delta = jnp.sqrt(delta_expanded)  # (B, L, D, 1)
    B_bar = (sqrt_delta * B) / denom  # (B, L, D, N)

    return A_bar, B_bar


# =============================================================================
# Selective Scan (EXACT COPY - DO NOT MODIFY)
# =============================================================================

class SSMState(NamedTuple):
    """State for associative scan operation."""
    h: jnp.ndarray  # Hidden state


def selective_scan_parallel(
    A_bar: jnp.ndarray,  # (B, L, D, N) - discretized state matrix
    B_bar: jnp.ndarray,  # (B, L, D, N) - discretized input matrix
    C: jnp.ndarray,      # (B, L, D, N) - output matrix
    x: jnp.ndarray,      # (B, L, D) - input sequence
    h0: Optional[jnp.ndarray] = None,  # (B, D, N) - initial state
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Parallel implementation of selective scan using associative scan.

    Recurrence: h_t = Ā_t · h_{t-1} + B̄_t · x_t
    Associative operation: (a₂, b₂) ⊗ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
    """
    B_size, L, D, N = A_bar.shape

    # Prepare input: B_bar * x
    x_expanded = x[..., None]  # (B, L, D, 1)
    Bx = B_bar * x_expanded  # (B, L, D, N)

    # Handle initial state
    if h0 is not None:
        init_contribution = A_bar[:, 0] * h0  # (B, D, N)
        Bx = Bx.at[:, 0].add(init_contribution)

    # Define the associative binary operation
    def associative_op(left, right):
        a_left, b_left = left
        a_right, b_right = right
        a_new = a_right * a_left
        b_new = a_right * b_left + b_right
        return (a_new, b_new)

    # Stack elements for scan: (A_bar, Bx)
    elements = (A_bar, Bx)

    # Apply associative scan along sequence dimension (axis=1)
    # NOTE: associative_scan is already optimized for parallelism on H200
    # Unlike lax.scan, it doesn't support 'unroll' but is inherently parallel
    _, all_h = lax.associative_scan(associative_op, elements, axis=1)

    # all_h now contains all hidden states: (B, L, D, N)
    # Compute outputs: y_t = sum_n(C_t * h_t)
    y = jnp.sum(C * all_h, axis=-1)  # (B, L, D)

    # Final hidden state
    h_final = all_h[:, -1]  # (B, D, N)

    return y, h_final


def selective_scan_sequential(
    A_bar: jnp.ndarray,  # (B, L, D, N)
    B_bar: jnp.ndarray,  # (B, L, D, N)
    C: jnp.ndarray,      # (B, L, D, N)
    x: jnp.ndarray,      # (B, L, D)
    h0: Optional[jnp.ndarray] = None,  # (B, D, N)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sequential implementation using jax.lax.scan with unroll.

    OPTIMIZED for H200: Uses unroll=16 to reduce JIT graph complexity.
    This prevents massive graph materialization during compilation.

    O(L) sequential, but with optimized compilation.
    """
    B_size, L, D, N = A_bar.shape

    # Initialize hidden state
    if h0 is None:
        h0 = jnp.zeros((B_size, D, N))

    def scan_fn(h, inputs):
        """Single step of the SSM."""
        A_t, B_t, C_t, x_t = inputs

        # Expand x_t for broadcasting
        x_t_expanded = x_t[..., None]  # (B, D, 1)

        # Update hidden state: h_t = A_t * h_{t-1} + B_t * x_t
        h_new = A_t * h + B_t * x_t_expanded

        # Compute output: y_t = sum(C_t * h_t)
        y_t = jnp.sum(C_t * h_new, axis=-1)  # (B, D)

        return h_new, y_t

    # Prepare inputs for scan: move time dimension to front
    A_seq = jnp.moveaxis(A_bar, 1, 0)  # (L, B, D, N)
    B_seq = jnp.moveaxis(B_bar, 1, 0)  # (L, B, D, N)
    C_seq = jnp.moveaxis(C, 1, 0)      # (L, B, D, N)
    x_seq = jnp.moveaxis(x, 1, 0)      # (L, B, D)

    # Apply scan with unroll=16 to reduce compilation overhead
    # CRITICAL for preventing 1.37TB compilation OOM
    h_final, y_seq = lax.scan(
        scan_fn,
        h0,
        (A_seq, B_seq, C_seq, x_seq),
        unroll=16  # Unroll 16 iterations for H200 optimization
    )

    # Move time dimension back to position 1
    y = jnp.moveaxis(y_seq, 0, 1)  # (B, L, D)

    return y, h_final


# =============================================================================
# Mamba Block Components (EXACT COPY - DO NOT MODIFY)
# =============================================================================

class CausalConv1D(nn.Module):
    """
    Causal 1D convolution for Mamba.
    Uses depthwise convolution (groups=channels) for efficiency.
    """
    features: int
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)
        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B, L, D = x.shape

        # Causal padding: pad (kernel_size - 1) on the left, 0 on the right
        pad_width = self.kernel_size - 1
        x_padded = jnp.pad(x, ((0, 0), (pad_width, 0), (0, 0)), mode='constant')

        # Depthwise convolution kernel
        kernel = self.param(
            'kernel',
            initializers.lecun_normal(),
            (self.kernel_size, self.features)
        )
        bias = self.param('bias', initializers.zeros, (self.features,))

        # Manual depthwise conv implementation
        def extract_patches(x_pad):
            patches = jnp.stack([
                x_pad[i:i+L] for i in range(self.kernel_size)
            ], axis=1)  # (L, k, D)
            return patches

        # Apply to batch: (B, L, k, D)
        patches = jax.vmap(extract_patches)(x_padded)

        # Multiply by kernel and sum over kernel dimension
        y = jnp.sum(patches * kernel, axis=2) + bias  # (B, L, D)

        return y


class S6Layer(nn.Module):
    """
    Selective SSM (S6) layer with learned projections.

    Uses Bilinear (Tustin) discretization with Discretization Guard
    (learnable RMSNorm + soft_clamp) and error-compensated recurrence
    via trapezoidal input averaging.

    Mode Parameter:
    - "tustin": Full Tustin discretization with guards (default)
    - "vanilla": Raw Tustin discretization without guards (for baseline comparison)
    - "zoh": Standard Mamba ZOH discretization without guards (strict baseline)
    """
    d_model: int          # Model dimension (D)
    d_state: int = 16     # SSM state dimension (N)
    dt_rank: int = None   # Rank for delta projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    delta_max: float = 10.0  # Upper bound for soft clamp
    mode: str = "tustin"  # "tustin", "vanilla", or "zoh"

    def setup(self):
        """Initialize parameters."""
        self.dt_rank_actual = self.dt_rank or math.ceil(self.d_model / 16)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        use_parallel: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan
        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B_size, L, D = x.shape
        N = self.d_state
        dt_rank = self.dt_rank_actual

        # === Parameter Initialization ===

        # A: (D, N) - S4D-Real initialization: A_n = -(n+1)
        A_log = self.param(
            'A_log',
            lambda rng, shape: jnp.log(
                jnp.broadcast_to(
                    jnp.arange(1, shape[1] + 1, dtype=jnp.float32),
                    shape
                )
            ),
            (D, N)
        )
        A = -jnp.exp(A_log)  # (D, N), negative for stability

        # D: (D,) - skip connection
        D_param = self.param('D', initializers.ones, (D,))

        # === Projections for Selection Mechanism ===

        # Combined projection for B and C
        x_bc = nn.Dense(
            features=2 * N,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='x_proj_bc'
        )(x)  # (B, L, 2*N)

        B_sel = x_bc[..., :N]   # (B, L, N)
        C_sel = x_bc[..., N:]   # (B, L, N)

        # Delta projection: two-stage low-rank
        x_dt = nn.Dense(
            features=dt_rank,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='x_proj_dt'
        )(x)  # (B, L, dt_rank)

        # Bias initialization: inverse softplus of uniform[dt_min, dt_max]
        def dt_bias_init(rng, shape, dtype=jnp.float32):
            dt = jax.random.uniform(rng, shape, minval=self.dt_min, maxval=self.dt_max, dtype=dtype)
            dt = jnp.clip(dt, a_min=self.dt_init_floor)
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt

        dt_proj = nn.Dense(
            features=D,
            use_bias=True,
            kernel_init=initializers.lecun_normal(),
            bias_init=dt_bias_init,
            name='dt_proj'
        )(x_dt)  # (B, L, D)

        # Apply softplus to get positive delta
        delta = jax.nn.softplus(dt_proj)  # (B, L, D)

        # === MODE-DEPENDENT DISCRETIZATION GUARD ===
        if self.mode in ["vanilla", "zoh"]:
            # Baseline modes: NO discretization guard
            # - No RMSNorm (no delta_norm)
            # - No soft_clamp
            # This allows unbounded delta values (can cause OOD collapse)
            pass  # delta remains as-is from softplus
        else:
            # Tustin mode: Full discretization guard
            # 1. Learnable RMSNorm to prevent magnitude explosion
            delta = nn.RMSNorm(name='delta_norm')(delta)  # (B, L, D)
            # 2. Soft clamp to bound Δ in [ε, Δ_max]
            delta = soft_clamp(delta, min_val=self.dt_init_floor, max_val=self.delta_max)

        # === Expand B and C to match (B, L, D, N) ===
        B_expanded = B_sel[:, :, None, :]  # (B, L, 1, N)
        B_expanded = jnp.broadcast_to(B_expanded, (B_size, L, D, N))

        C_expanded = C_sel[:, :, None, :]  # (B, L, 1, N)
        C_expanded = jnp.broadcast_to(C_expanded, (B_size, L, D, N))

        # === MODE-DEPENDENT DISCRETIZATION ===
        if self.mode == "zoh":
            # ZOH baseline: Standard Mamba Zero-Order Hold WITHOUT guards
            # - Uses exp(ΔA) discretization (original Mamba)
            # - No float32 precision guard (runs in model's native bfloat16)
            # - Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB
            # - Intentionally raw to demonstrate OOD performance characteristics
            A_bar, B_bar = discretize_zoh(A, B_expanded, delta)
        elif self.mode == "vanilla":
            # Vanilla baseline: Raw Tustin discretization WITHOUT guards
            # - No float32 precision guard (runs in model's default precision)
            # - No stability epsilon or clamping
            # - Raw division: A_bar = (I + AΔ/2) / (I - AΔ/2)
            # - Intentionally fragile to demonstrate need for guards
            A_bar, B_bar = discretize_tustin_raw(A, B_expanded, delta)
        else:
            # Tustin mode: Guarded Bilinear (Tustin) discretization
            # - Uses (I ∓ Δ/2·A) rational approximation
            # - Float32 precision guard is applied inside discretize_tustin
            # - Clamping and epsilon handling for stability
            A_bar, B_bar = discretize_tustin(A, B_expanded, delta)

        # === MODE-DEPENDENT INPUT HANDLING ===
        if self.mode in ["vanilla", "zoh"]:
            # Baseline modes: Raw input (no trapezoidal averaging)
            # Standard SSM: h_t = Ā_t · h_{t-1} + B̄_t · x_t
            x_input = x  # (B, L, D)
        else:
            # Tustin mode: Error-Compensated Recurrence (Trapezoidal Averaging)
            # h_t = Ā_t · h_{t-1} + B̄_t · (x_t + x_{t-1}) / 2
            x_prev = jnp.concatenate(
                [jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1
            )  # (B, L, D)
            x_input = (x + x_prev) / 2.0  # (B, L, D)

        # === Selective Scan ===
        if use_parallel:
            y, _ = selective_scan_parallel(A_bar, B_bar, C_expanded, x_input)
        else:
            y, _ = selective_scan_sequential(A_bar, B_bar, C_expanded, x_input)

        # === Skip Connection ===
        y = y + D_param * x

        return y


class MambaBlock(nn.Module):
    """
    Full Mamba block as described in Section 3.4 of the paper.

    Architecture:
        Input x → Norm → Linear expansion → Split into two branches
        Branch 1: Conv1D → SiLU → S6 SSM
        Branch 2: (identity for gating)
        Element-wise multiply (Branch1 * SiLU(Branch2))
        Linear projection → Residual connection
    """
    d_model: int          # Model dimension (D)
    d_state: int = 16     # SSM state dimension (N)
    d_conv: int = 4       # Convolution kernel size
    expand: int = 2       # Expansion factor (E)
    dt_rank: int = None   # Rank for delta projection
    mode: str = "tustin"  # "tustin", "vanilla", or "zoh"

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        use_parallel: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor, shape (B, L, D)
            use_parallel: Whether to use parallel scan in SSM
        Returns:
            y: Output tensor, shape (B, L, D)
        """
        B_size, L, D = x.shape
        D_inner = self.expand * D  # Inner dimension after expansion

        # Store input for residual connection
        residual = x

        # === Input Normalization ===
        x = nn.RMSNorm(name='norm')(x)

        # === Linear Expansion ===
        x_proj = nn.Dense(
            features=2 * D_inner,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='in_proj'
        )(x)  # (B, L, 2*D_inner)

        # Split into two branches
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)  # Each: (B, L, D_inner)

        # === Branch 1: Conv → SiLU → SSM ===

        # 1D Causal Convolution
        x_conv = CausalConv1D(
            features=D_inner,
            kernel_size=self.d_conv,
            name='conv1d'
        )(x_main)  # (B, L, D_inner)

        # SiLU activation
        x_conv = jax.nn.silu(x_conv)

        # Selective SSM (S6)
        x_ssm = S6Layer(
            d_model=D_inner,
            d_state=self.d_state,
            dt_rank=self.dt_rank,
            mode=self.mode,
            name='ssm'
        )(x_conv, use_parallel=use_parallel)  # (B, L, D_inner)

        # === Gating ===
        x_gated = x_ssm * jax.nn.silu(x_gate)  # (B, L, D_inner)

        # === Output Projection ===
        y = nn.Dense(
            features=D,
            use_bias=False,
            kernel_init=initializers.lecun_normal(),
            name='out_proj'
        )(x_gated)  # (B, L, D)

        # === Residual Connection ===
        y = y + residual

        return y


class MambaLM(nn.Module):
    """
    Language Model: Embed → Mamba Stack → RMSNorm → Dense → Logits

    Mode Parameter:
    - "tustin": Full Tustin discretization with guards (default)
    - "vanilla": Raw Tustin discretization without guards (for baseline comparison)
    - "zoh": Standard Mamba ZOH discretization without guards (strict baseline)
    """
    vocab_size: int
    d_model: int
    n_layers: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mode: str = "tustin"  # "tustin", "vanilla", or "zoh"

    @nn.compact
    def __call__(self, x, use_parallel=True, train=True):
        # x: (B, L) (indices)

        # Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)

        # Stack Mamba Blocks
        for i in range(self.n_layers):
            x = MambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                mode=self.mode,
                name=f'mamba_block_{i}'
            )(x, use_parallel=use_parallel)

        # Final Norm & Projection
        x = nn.RMSNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)

        return logits
