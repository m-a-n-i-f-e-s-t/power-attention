## QUERY STATE BACKWARD PASS

## IMPLEMENTATION ##
import torch
from typing import Optional, Tuple
from power_attention_cuda import query_states_bwd as query_states_bwd_cuda

from power_attention._utils import compute_expanded_dim

# Define a traceable inner backward pass
@torch.library.custom_op("power_attention::query_state_bwd", mutates_args=(), device_types='cuda')
def query_state_bwd(Q : torch.Tensor, S : torch.Tensor,
                               dO : torch.Tensor, rowmax : Optional[torch.Tensor],
                               deg : int, scale : float, zero_initial_state : bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute query state backward.

    Computes gradients with respect to inputs Q, S, s for the query state operation.
    
    Input shapes:
        Q: [batch, seq, chunk, head, dim] - Query tensor
        S: [batch, seq, head, expanded_dim, dim] - State tensor
        s: [batch, seq, head, expanded_dim] - State scaling tensor
        dO: [batch, seq, chunk, head, dim] - Gradient of loss with respect to output O
        do: [batch, seq, chunk, head] - Gradient of loss with respect to output scaling o
        rowmax: [batch, seq, head, expanded_dim] - Rowmax tensor
        deg: int - Power attention degree
        scale: float - Scaling factor
        zero_initial_state: bool - Whether the initial state is zero

    Output shapes:
        dQ: [batch, seq, chunk, head, dim] - Gradient with respect to Q
        dS: [batch, seq, head, expanded_dim, dim] - Gradient with respect to S
        dY_attn: [batch, seq, chunk, head, dim] - Gradient with respect to Y_attn

    Input restrictions:
        - Q, S contiguous along the last dimension
        - Q, S have same feature dimension, and it must be 32 or 64
        - Q, S have same dtype
        - fp16 or bf16 only
        - do, s must be float32
    """
    dQ, dS, _, dY_attn = query_states_bwd_cuda(
        Q, S, dO, rowmax, None, deg, scale, zero_initial_state, False, False)
    return dQ, dS, dY_attn

@query_state_bwd.register_fake
def query_state_bwd_fake(Q, S, dO, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    return (torch.empty_like(Q), 
            torch.empty_like(S), 
            torch.empty_like(dO) if rowmax is not None else torch.empty([0], device=dO.device, dtype=dO.dtype))

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42, fused=False, zero_initial_state=False, scale=None):
    torch.manual_seed(seed)
    deg = 2
    D = compute_expanded_dim(d, deg)
    Q = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    S = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device) 
    dO = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device)
    rowmax = torch.randn(size=(b, n, c, h), dtype=torch.float32, device=device) if fused else None
    if zero_initial_state:
        S[:, 0] = 0
    return dict(Q=Q, S=S, dO=dO, rowmax=rowmax, deg=deg, scale=scale, zero_initial_state=zero_initial_state)

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, c, h, d = (1, 1, 128, 1, 32)
    dtype = torch.float16
    scale = 1.0
    # Create inputs
    inputs = Q, S, dO, rowmax, deg, scale, zero_initial_state = create_inputs(b, n, c, h, d, dtype, 'cuda', fused=True, scale=scale)
    # Run functions
    with torch.no_grad():
        dQ, dS, dY_attn = query_state_bwd(Q, S, dO, rowmax, deg, scale, zero_initial_state)
    # Compile functions
    compiled_bwd = torch.compile(query_state_bwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(2):
            outputs = compiled_bwd(*inputs)

