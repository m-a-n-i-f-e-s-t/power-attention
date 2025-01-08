## QUERY STATE BACKWARD PASS

## IMPLEMENTATION ##
import torch
from typing import Optional, Tuple
from state_kernel_cuda import (
    query_states_bwd as query_states_bwd_cuda,
    InnerBlock_TD as InnerBlock,
    OuterBlock_TD as OuterBlock
)

def ExpandedDim(head_size, deg):
    return ((InnerBlock // OuterBlock + head_size // OuterBlock) * (head_size // InnerBlock) // 2) * (InnerBlock * OuterBlock)

# Define a traceable inner backward pass
@torch.library.custom_op("state_kernel::query_state_bwd", mutates_args=(), device_types='cuda')
def query_state_bwd(Q : torch.Tensor, S : torch.Tensor,
                               dO : torch.Tensor, rowmax : Optional[torch.Tensor],
                               deg : int, stabilizer : float, zero_initial_state : bool,deterministic : bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        stabilizer: float - Stabilization factor
        zero_initial_state: bool - Whether the initial state is zero
        deterministic: bool - Whether to accumulate dQ deterministically
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
        Q, S, dO, rowmax, None, deg, stabilizer, zero_initial_state, False, deterministic)
    return dQ, dS, dY_attn

@query_state_bwd.register_fake
def query_state_bwd_fake(Q, S, dO, rowmax, deg, stabilizer, zero_initial_state, deterministic):
    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    return (torch.empty_like(Q), 
            torch.empty_like(S), 
            torch.empty_like(dO))

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42, deterministic=True, fused=False, zero_initial_state=False, stabilizer=None):
    torch.manual_seed(seed)
    deg = 2
    D = ExpandedDim(d, deg)
    Q = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    S = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device) 
    dO = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device)
    rowmax = torch.randn(size=(b, n, c, h), dtype=torch.float32, device=device) if fused else None
    if zero_initial_state:
        S[:, 0] = 0
    return Q, S, dO, rowmax, deg, stabilizer, zero_initial_state, deterministic

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, c, h, d = (1, 1, 128, 1, 32)
    dtype = torch.float16
    stabilizer = 1.0
    # Create inputs
    inputs = Q, S, dO, rowmax, deg, stabilizer, zero_initial_state, deterministic = create_inputs(b, n, c, h, d, dtype, 'cuda', fused=True, stabilizer=stabilizer)
    # Run functions
    with torch.no_grad():
        dQ, dS, dY_attn = query_state_bwd(Q, S, dO, rowmax, deg, stabilizer, zero_initial_state, deterministic)
    # Compile functions
    compiled_bwd = torch.compile(query_state_bwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(2):
            outputs = compiled_bwd(*inputs)

