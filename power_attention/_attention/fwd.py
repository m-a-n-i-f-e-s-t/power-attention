## POWER ATTENTION FORWARD PASS
# This is a custom op that computes the power attention forward pass using CUDA.
# This file contains the implementation of the forward pass as well as a fake
# implementation for tracing and testing.

## IMPLEMENTATION ##
import torch
from power_attention_cuda import attention_fwd as attention_fwd_cuda
from power_attention._config import normal_space, flash_equivalent, eps
from typing import Optional, Tuple

@torch.library.custom_op('power::attention_forward', mutates_args=(), device_types='cuda')
def attention_fwd(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, 
                  log_G_Q : Optional[torch.Tensor], log_G_K : Optional[torch.Tensor],
                  deg : int, scale : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute power attention forward pass.
    
    Input shapes:
        Q, K, V: [batch, seq, head, dim] - Query, Key and Value tensors
        log_G_Q, log_G_K: Optional[batch, seq, head] - Optional log gating factors
        deg: int - Power attention degree
        scale: float - Scale factor for query-key inner product
    Output shapes:
        Y: [batch, seq, head, dim] - Output tensor
        y: [batch, seq, head] - Output scaling factors
        rowmax: [batch, seq, head] - Rowmax tensor

    Input restrictions:
        - Q, K, V contiguous along the last dimension
        - Q, K, V have same feature dimension, and it must be 32 or 64
        - Q, K, V have same dtype
        - fp16 or bf16 only
        - log_G_Q and log_G_K are both present or both None
        
    """
    Y, y, rowmax, _ = attention_fwd_cuda(Q, K, V, log_G_Q, log_G_K, None, deg, False, 1 / scale, eps, flash_equivalent, normal_space)
    return Y, y, rowmax

# Fake implementation for tracing and testing
@attention_fwd.register_fake
def attention_fwd_fake(Q, K, V, log_G_Q, log_G_K, deg, scale):
    b, t, h, d = Q.shape
    return (torch.empty(b, t, h, d, device=Q.device, dtype=Q.dtype), 
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32),
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32))

# Useful function to create sample inputs
def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, scale=1.0, deg=2):
    generator = torch.Generator(device=device).manual_seed(42)
    Q = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) / d**.25
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) / d**.25
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator)
    if gating:
        log_G_Q = torch.zeros(size=(b, t, h), dtype=torch.float32, device=device) - .01
        log_G_K = torch.zeros(size=(b, t, h), dtype=torch.float32, device=device) - .01
    else:
        log_G_Q = None
        log_G_K = None
    return dict(Q=Q, K=K, V=V, log_G_Q=log_G_Q, log_G_K=log_G_K, deg=deg, scale=scale)

## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, t, h, d = (2, 32, 8, 32)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, t, h, d, dtype, 'cuda', gating=True, scale=1.0, deg=2)
    # Run function
    with torch.no_grad():
        Y, y, rowmax = attention_fwd(*inputs)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_attention_fwd = torch.compile(attention_fwd, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            Y, y, rowmax = compiled_attention_fwd(*inputs)
