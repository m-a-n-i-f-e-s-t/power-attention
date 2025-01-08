## CHUNK STATE BACKWARD PASS

## IMPLEMENTATION ##
import torch
from state_kernel_cuda import (
    chunk_states_bwd,
    InnerBlock_DT,
    OuterBlock_DT,
)
from typing import Tuple

def ExpandedDim(head_size, deg):
    return ((InnerBlock_DT // OuterBlock_DT + head_size // OuterBlock_DT) * (head_size // InnerBlock_DT) // 2) * (InnerBlock_DT * OuterBlock_DT)

@torch.library.custom_op("state_kernel::chunk_state_bwd", mutates_args=(), device_types='cuda')
def chunk_state_bwd(K : torch.Tensor, V : torch.Tensor, dS : torch.Tensor, deg : int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for chunk state operation.

    Computes gradients with respect to inputs K and V for the chunk state operation.
    
    Input shapes:
        K: [batch, chunk_n, chunk_size, heads, head_size] - Key tensor from forward pass
        V: [batch, chunk_n, chunk_size, heads, head_size] - Value tensor from forward pass
        dS: [batch, chunk_n, heads, expanded_dim, head_size] - Gradient of loss w.r.t. forward S output
        deg: int - Degree parameter used in forward pass

    Output shapes:
        dK: [batch, chunk_n, chunk_size, heads, head_size] - Gradient with respect to input K
        dV: [batch, chunk_n, chunk_size, heads, head_size] - Gradient with respect to input V

    Restrictions:
        - head_size must be a multiple of InnerBlock_DT
        - expanded_dim is computed based on head_size and deg
        - fp16 or bf16
    """
    dK, dV = chunk_states_bwd(K, V, dS, deg)
    return dK, dV

@chunk_state_bwd.register_fake
def chunk_state_bwd_fake(K, V, dS, deg):
    return torch.empty_like(K), torch.empty_like(V)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42):
    torch.manual_seed(seed)
    D = ExpandedDim(d, deg=2)  # Expanded dimension
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    V = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    dS = torch.randn(size=(b, n, h, D, d), dtype=dtype, device=device)
    deg = 2
    return K, V, dS, deg

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, c, h, d = (2, 4, 128, 8, 32)
    dtype = torch.float16
    # Create inputs
    K, V, dS, deg = create_inputs(b, n, c, h, d, dtype=dtype, device='cuda')
    # Run function
    with torch.no_grad():
        dK, dV = chunk_state_bwd(K, V, dS, deg)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_fn = torch.compile(chunk_state_bwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            dK, dV = compiled_fn(K, V, dS, deg)

