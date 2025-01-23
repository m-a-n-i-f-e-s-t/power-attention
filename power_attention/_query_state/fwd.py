## QUERY STATE FORWARD PASS
# This is a custom op that computes the query state forward pass using CUDA.
# This file contains the implementation of the forward pass as well as a fake
# implementation for tracing and testing.

## IMPLEMENTATION ##
import torch
from power_attention_cuda import compute_query_states as compute_query_states_cuda
from typing import Optional

from power_attention._utils import compute_expanded_dim
from power_attention._config import eps

@torch.library.custom_op('power_attention::query_state_forward', mutates_args=('Y',), device_types='cuda')
def query_state_fwd(Q : torch.Tensor, S : torch.Tensor,
                    Y : Optional[torch.Tensor],
                    rowmax : Optional[torch.Tensor], deg : int,
                    scale : float, zero_initial_state : bool) -> torch.Tensor:
    """Compute query state forward pass.
    
    Input shapes:
        Q: [batch, num_chunks, chunk_size, head, dim] - Query tensor
        S: [batch, num_chunks, head, state_dim, dim] - State tensor
        Y: Optional[batch, num_chunks, chunk_size, head, dim] - Optional output tensor
        rowmax: Optional[batch, num_chunks, chunk_size, head] - Optional rowmax tensor
        deg: int - Power attention degree
        scale: float - Scaling factor
        zero_initial_state: bool - Whether the initial state is zero
        
    Output shapes:
        O: [batch, num_chunks, chunk_size, head, dim] - Output tensor

    Input restrictions:
        - Q contiguous along the last dimension
        - Q feature dimension must be 32 or 64
        - fp16 or bf16 only
        - chunk_size must be a multiple of 16, and at least 128
    """
    # TODO(jbuckman): Figure out how to make this traceable despite buffer reuse, and remove clone()
    Y = Y.clone() if Y is not None else None
    O, _ = compute_query_states_cuda(Q, S, Y, rowmax,
                                        None, deg, scale, zero_initial_state, eps,
                                        False, True)
    return O

# Fake implementation for tracing and testing
@query_state_fwd.register_fake
def query_state_fwd_fake(Q, S, Y, rowmax, deg, scale, zero_initial_state):
    b, n, c, h, d = Q.shape
    return torch.empty(b, n, c, h, d, device=Q.device, dtype=Q.dtype)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, fused=False, device='cuda', seed=42, zero_initial_state=False, scale=None):
    torch.manual_seed(seed)
    deg = 2
    D = compute_expanded_dim(d, deg)
    Q = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    S = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device)
    if fused:
        Y = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device)
        rowmax = torch.randn(size=(b, n, c, h), dtype=torch.float32, device=device)
    else:
        Y = None
        rowmax = None
    if zero_initial_state:
        S[:, 0] = 0
    return dict(
        Q=Q, 
        S=S, 
        Y=Y, 
        rowmax=rowmax, 
        deg=deg, 
        scale=scale, 
        zero_initial_state=zero_initial_state, 
    )

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, c, h, d = (2, 4, 128, 8, 32)
    dtype = torch.float16
    scale = 1.0
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda', scale=scale)
    # Run function
    with torch.no_grad():
        O = query_state_fwd(**inputs)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_query_state_fwd = torch.compile(query_state_fwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            O = compiled_query_state_fwd(*inputs)