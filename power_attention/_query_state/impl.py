## QUERY STATE CUSTOM OP
# Computes the query state forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import Optional
from power_attention._query_state.fwd import query_state_fwd
from power_attention._query_state.bwd import query_state_bwd
from power_attention._utils import compute_expanded_dim

# Define the primary query_state entrypoint
@torch.library.custom_op("power_attention::query_state", mutates_args=())
def query_state(Q : torch.Tensor, S : torch.Tensor, Y : Optional[torch.Tensor],
                rowmax : Optional[torch.Tensor],
                deg : int, scale : Optional[float], zero_initial_state : bool) -> torch.Tensor:
    r"""Compute query interaction with expanded state vectors.

    This function implements the query-state interaction from [1]. It computes how queries
    interact with the expanded state vectors using symmetric power embeddings, which provide
    an efficient way to compute higher-order attention patterns.

    For each query $Q_t$, we compute:

    $$Y_t = \phi(Q_t) \cdot S_{<t}$$

    where $\phi$ maps $Q_t$ to its symmetric power embedding of degree $deg$. This is equivalent to
    computing attention scores with keys and values from previous chunks, but using the
    compressed state representation for efficiency.

    The computation can be numerically unstable in fp16, so we provide two stabilization
    mechanisms:

    1. A scaling factor that scales the symmetric power embedding
    2. Optional output scaling using Y and rowmax from the attention computation

    Args:
        Q: Query tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        S: State tensor of shape `(batch_size, num_chunks, num_heads, expanded_dim, head_dim)`.
           Contains expanded state vectors from update_state.
        Y: Optional output tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
           Used for output scaling when provided.
        rowmax: Optional scaling tensor of shape `(batch_size, num_chunks, chunk_size, num_heads)`.
           Used for matching the scale of the attention output and that of the query state output.
        deg: Power attention degree. Must be even for symmetric power formulation.
        scale: Optional stabilization factor. Defaults to state_dim for fp16, 1.0 otherwise.
            Helps prevent overflow in symmetric power computation.
        zero_initial_state: Whether the initial state should be treated as zero.

    Returns:
        O: Output tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`. 
           Note that the output is normalized by the minimum of the scale and the rowmax.

    Note:
        - Q must be contiguous along the last dimension
        - Feature dimension must be 32 or 64
        - Inputs must be fp16 or bf16
        - chunk_size must be at least 128 and a multiple of 16
        - Scaling is particularly important for deg > 2

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    b, n, c, h, d = Q.shape
    if scale is None:
        scale = 1.
    O = query_state_fwd(Q, S, Y, rowmax, deg, scale, zero_initial_state)
    return O
@query_state.register_fake
def query_state_fake(Q, S, Y, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    return torch.empty(b, n, c, h, d, device=Q.device, dtype=Q.dtype)
# Autograd setup
def query_state_setup(ctx, inputs, output):
    Q, S, Y, rowmax, deg, scale, zero_initial_state = inputs
    b, n, c, h, d = Q.shape
    if scale is None:
        scale = 1.
    ctx.save_for_backward(Q, S, rowmax)
    ctx.deg = deg
    ctx.scale = scale
    ctx.fused = Y is not None
    ctx.zero_initial_state = zero_initial_state
def query_state_backward(ctx, dO):
    Q, S, rowmax = ctx.saved_tensors
    dQ, dS, dY_attn = query_state_bwd(Q, S, dO, rowmax, ctx.deg, ctx.scale, ctx.zero_initial_state)
    if ctx.fused:
        dY = dY_attn
    else:
        dY = None
    return dQ, dS, dY, None, None, None, None
# Register autograd
torch.library.register_autograd(
    "power_attention::query_state", query_state_backward, setup_context=query_state_setup
)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, fused=False, device='cuda', requires_grad=False, seed=42, zero_initial_state=False, scale=None, q_std=1.0, S_std=1.0, Y_std=1.0, rowmax_std=1.0):
    torch.manual_seed(seed)
    deg = 2
    D = compute_expanded_dim(d, deg)
    Q = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) * q_std / d**.25
    S = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device) * S_std / d**.25
    if fused:
        Y = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) * Y_std / d**.25
        rowmax = F.logsigmoid(torch.randn(size=(b, n, c, h), dtype=torch.float32, device=device) * rowmax_std)
    else:
        Y = None
        rowmax = None
    if zero_initial_state:
        S[:, 0] = 0
    if requires_grad:
        Q, S, Y = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, S, Y))
    return dict(Q=Q, S=S, Y=Y, rowmax=rowmax, deg=deg, scale=scale, zero_initial_state=zero_initial_state)

## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_fwd_bwd

    # Hyperparameters
    b, n, c, h, d = (8, 8, 128, 16, 64)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda', requires_grad=True)

    # Benchmark
    print(f"Benchmarking query state \n {b=} {n=} {c=} {h=} {d=} {dtype}")
    print_fwd_bwd(query_state, *inputs)

