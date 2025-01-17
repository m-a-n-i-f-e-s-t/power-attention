## QUERY STATE CUSTOM OP
# Computes the query state forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import Optional, Tuple
from power_attention._query_state.fwd import query_state_fwd
from power_attention._query_state.bwd import ExpandedDim as compute_expanded_dim, query_state_bwd

# Define the primary query_state entrypoint
@torch.library.custom_op("power_attention::query_state", mutates_args=())
def query_state(Q : torch.Tensor, S : torch.Tensor,
                Y : Optional[torch.Tensor],
                rowmax : Optional[torch.Tensor],
                deg : int, stabilizer : Optional[float], zero_initial_state : bool,
                eps : float, deterministic : bool) -> torch.Tensor:
    """Compute query state forward pass.
    
    Input shapes:
        Q: [batch, num_chunks, chunk_size, head, dim] - Query tensor
        S: [batch, num_chunks, head, state_dim, dim] - State tensor
        Y: Optional[batch, num_chunks, chunk_size, head, dim] - Optional output tensor
        rowmax: Optional[batch, num_chunks, chunk_size, head] - Optional rowmax tensor
        deg: int - Power attention degree
        stabilizer: Optional[float] - Stabilization factor, defaults to state_dim for fp16 and 1.0 otherwise
        zero_initial_state: bool - Whether the initial state is zero
        eps: float - Small constant for numerical stability
        deterministic: bool - Whether to accumulate dQ deterministically

    Output shapes:
        O: [batch, num_chunks, chunk_size, head, dim] - Output tensor

    Input restrictions:
        - Q contiguous along the last dimension
        - Q feature dimension must be 32 or 64
        - fp16 or bf16 only
        - chunk_size must be at least 128 and a multiple of 16
    """
    b, n, c, h, d = Q.shape
    if stabilizer is None:
        stabilizer = 1.
    O = query_state_fwd(Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps)
    return O
@query_state.register_fake
def query_state_fake(Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic):
    b, n, c, h, d = Q.shape
    return torch.empty(b, n, c, h, d, device=Q.device, dtype=Q.dtype)
# Autograd setup
def query_state_setup(ctx, inputs, output):
    Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic = inputs
    b, n, c, h, d = Q.shape
    if stabilizer is None:
        stabilizer = 1.
    ctx.save_for_backward(Q, S, rowmax)
    ctx.deg = deg
    ctx.stabilizer = stabilizer
    ctx.fused = Y is not None
    ctx.deterministic = deterministic
    ctx.zero_initial_state = zero_initial_state
def query_state_backward(ctx, dO):
    Q, S, rowmax = ctx.saved_tensors
    dQ, dS, dY_attn = query_state_bwd(Q, S, dO, rowmax, ctx.deg, ctx.stabilizer, ctx.zero_initial_state, ctx.deterministic)
    if ctx.fused:
        dY = dY_attn
    else:
        dY = None
    return dQ, dS, dY, None, None, None, None, None, None
# Register autograd
torch.library.register_autograd(
    "power_attention::query_state", query_state_backward, setup_context=query_state_setup
)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, fused=False, device='cuda', requires_grad=False, seed=42, deterministic=True, zero_initial_state=False, stabilizer=None, q_std=1.0, S_std=1.0, Y_std=1.0, rowmax_std=1.0):
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
    eps = 1e-7
    if zero_initial_state:
        S[:, 0] = 0
    if requires_grad:
        Q, S, Y = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, S, Y))
    return dict(Q=Q, S=S, Y=Y, rowmax=rowmax, deg=deg, stabilizer=stabilizer, zero_initial_state=zero_initial_state, eps=eps, deterministic=deterministic)

## TUTORIAL ##
if __name__ == '__main__':
    from tests_and_benchmarks._timing import get_compiled_versions, estimate_runtime
    from tests_and_benchmarks._timing import report_fwd_bwd

    # Hyperparameters
    b, n, c, h, d = (8, 8, 128, 16, 64)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda', requires_grad=True, deterministic=False)

    # Benchmark
    print(f"Benchmarking query state \n {b=} {n=} {c=} {h=} {d=} {dtype}")
    report_fwd_bwd(query_state, *inputs)

