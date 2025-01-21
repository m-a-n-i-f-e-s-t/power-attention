## UPDATE STATE FORWARD PASS
# This is a custom op that computes the states for the symmetric power attention kernel.
# This file contains the implementation of the forward pass as well as a fake
# implementation for tracing and testing.

## IMPLEMENTATION ##
import torch
from power_attention_cuda import compute_update_states as update_states_fwd_cuda
from power_attention._utils import compute_expanded_dim

@torch.library.custom_op("power_attention::update_state_fwd", mutates_args=(), device_types='cuda')
def update_state_fwd(K : torch.Tensor, V : torch.Tensor, deg : int) -> torch.Tensor:
    """Compute state update for symmetric power attention kernel.
    
    This operation computes the states needed for the symmetric power attention kernel.
    It processes chunks of the input sequence to compute intermediate states efficiently.

    Args:
        K: [batch, seq_len, chunks, heads, head_dim] - Key tensor
        V: [batch, seq_len, chunks, heads, head_dim] - Value tensor  
        deg: int - Degree of the power series expansion (right now must be 2)

    Returns:
        S: [batch, seq_len, heads, expanded_dim, head_dim] - Main state

    Shape Requirements:
        - head_dim must be 32 or 64
        - chunk_size must be a multiple of 16 and of Block_DT
    """
    S, _ = update_states_fwd_cuda(K, V, deg, False, True)
    return S

@update_state_fwd.register_fake
def update_state_fwd_fake(K, V, deg):
    b, n, c, h, d = K.shape
    D = compute_expanded_dim(d, deg)
    return (torch.empty(b, n, h, D, d, device=K.device, dtype=K.dtype))

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42):
    torch.manual_seed(seed)
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    V = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device)
    return dict(K=K, V=V, deg=2)

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, c, h, d = (2, 4, 128, 8, 32)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda')
    # Run function
    with torch.no_grad():
        S = update_state_fwd(**inputs)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_update_state_forward = torch.compile(update_state_fwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            S = compiled_update_state_forward(**inputs)
