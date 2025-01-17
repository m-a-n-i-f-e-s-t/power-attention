## UPDATE STATE CUSTOM OP
# Computes the update state forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.utils._pytree import tree_map
from types import NoneType
from typing import Tuple
from power_attention._update_state.fwd import ExpandedDim as compute_expanded_dim, update_state_fwd
from power_attention._update_state.bwd import update_state_bwd


# Define the primary update_state entrypoint
@torch.library.custom_op("power_attention::update_state", mutates_args=())
def update_state(K : torch.Tensor, V : torch.Tensor, deg : int) -> torch.Tensor:
    """Compute update state forward pass.
    
    Computes the update state operation by accumulating key-value pairs into expanded state vectors.
    
    Input shapes:
        K: [batch, chunk_n, chunk_size, head, dim] - Key tensor
        V: [batch, chunk_n, chunk_size, head, dim] - Value tensor
        deg: int - Degree of expansion for state vectors
        
    Output shapes:
        S: [batch, chunk_n, head, expanded_dim, dim] - Expanded state vectors
        
    Input restrictions:
        - K, V must have same shape
        - K, V must be contiguous along last dimension
        - K, V must have same dtype (fp16 or bf16)
        - chunk_size must be at least 128
    """
    S = update_state_fwd(K, V, deg)
    return S
# Make it traceable
@torch.library.register_fake("power_attention::update_state")
def update_state_fake(K, V, deg):
    b, n, c, h, d = K.shape
    D = compute_expanded_dim(d, deg)
    return torch.empty(b, n, h, D, d, device=K.device, dtype=K.dtype)
# Autograd setup
def update_state_setup(ctx, inputs, output):
    K, V, deg = inputs
    ctx.save_for_backward(K, V)
    ctx.deg = deg
def update_state_backward(ctx, dS):
    K, V = ctx.saved_tensors
    dS = dS.contiguous()
    dK, dV = update_state_bwd(K, V, dS, ctx.deg)
    return dK, dV, None
# Register autograd
torch.library.register_autograd(
    "power_attention::update_state", update_state_backward, setup_context=update_state_setup
)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42, requires_grad=False):
    torch.manual_seed(seed)
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    V = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    if requires_grad:
        K, V = tree_map(lambda x: x.requires_grad_(True), (K, V))
    return K, V, 2

## TUTORIAL ##
if __name__ == '__main__':
    from tests_and_benchmarks._timing import report_fwd_bwd

    # Hyperparameters
    b, n, c, h, d = (8, 8, 128, 16, 64)
    dtype = torch.float16
    # Create inputs
    K, V, deg = create_inputs(b, n, c, h, d, dtype, 'cuda', requires_grad=True)
    
    print(f"Benchmarking chunk state \n {b=} {n=} {c=} {h=} {d=} {dtype=}")

    # benchmark
    report_fwd_bwd(update_state, K, V, deg)
