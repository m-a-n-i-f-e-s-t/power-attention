## POWER ATTENTION BACKWARD PASS
# This is a custom op that computes the power attention backward pass using CUDA.
# This file contains the implementation of the backward pass as well as a fake
# implementation for tracing and testing.
# There are two versions of the backward pass: one that does not include gating
# and one that does. This is due to restrictions around tracing custom ops.

## IMPLEMENTATION ##
import torch
from typing import Tuple
from power_attention_cuda import attention_bwd as attention_bwd_cuda


@torch.library.custom_op("power::attention_bwd_gatingless", mutates_args=(), device_types='cuda')
def attention_bwd_gatingless(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, 
                             dY : torch.Tensor, dy : torch.Tensor, rowmax : torch.Tensor,
                             deg : int, scale : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute power attention backward pass without gating.

    Computes gradients with respect to inputs Q, K, V for the power attention operation.
    
    Input shapes:
        Q, K, V: [batch, seq, head, dim] - Query, Key and Value tensors
        dY: [batch, seq, head, dim] - Gradient of loss with respect to output Y
        dy: [batch, seq, head] - Gradient of loss with respect to output scaling y
        rowmax: [batch, seq, head] - Rowmax tensor for stablizing attention
        deg: int - Power attention degree
        scale: float - Scale factor for key-query inner product

    Output shapes:
        dQ: [batch, seq, head, dim] - Gradient with respect to Q
        dK: [batch, seq, head, dim] - Gradient with respect to K  
        dV: [batch, seq, head, dim] - Gradient with respect to V

    Input restrictions:
        - Q, K, V contiguous along the last dimension
        - Q, K, V have same feature dimension, and it must be 32 or 64
        - Q, K, V have same dtype
        - fp16 or bf16 only
    """
    dQ, dK, dV, _, _ = attention_bwd_cuda(Q, K, V, None, None, dY, dy, rowmax, None, None, None, deg, 1 / scale, 1e-6, False, False, False)
    return dQ, dK, dV

# Fake implementation for tracing and testing
@attention_bwd_gatingless.register_fake
def attention_bwd_gatingless_fake(Q, K, V, dY, dy, rowmax, deg, scale):
    return torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)

@torch.library.custom_op("power::attention_bwd_gating", mutates_args=(), device_types='cuda')
def attention_bwd_gating(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, 
                         log_G_Q : torch.Tensor, log_G_K : torch.Tensor, 
                         dY : torch.Tensor, dy : torch.Tensor, rowmax : torch.Tensor,
                         deg : int, scale : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute power attention backward pass with gating.

    Computes gradients with respect to inputs Q, K, V and gating factors for the power attention operation.
    
    Input shapes:
        Q, K, V: [batch, seq, head, dim] - Query, Key and Value tensors
        log_G_Q, log_G_K: [batch, seq, head] - Log gating factors for Q and K
        dY: [batch, seq, head, dim] - Gradient of loss with respect to output Y
        dy: [batch, seq, head] - Gradient of loss with respect to output scaling y
        rowmax: [batch, seq, head] - Rowmax tensor for stablizing attention
        deg: int - Power attention degree
        scale: float - Stabilization factor

    Output shapes:
        dQ: [batch, seq, head, dim] - Gradient with respect to Q
        dK: [batch, seq, head, dim] - Gradient with respect to K
        dV: [batch, seq, head, dim] - Gradient with respect to V
        dlog_G: [batch, seq, head] - Gradient with respect to log gating factors

    Input restrictions:
        - Q, K, V contiguous along the last dimension
        - Q, K, V have same feature dimension, and it must be 32 or 64
        - Q, K, V have same dtype
        - fp16 or bf16 only
        - log_G_Q and log_G_K must be float32
    """
    dQ, dK, dV, dlog_G_Q, dlog_G_K = attention_bwd_cuda(
        Q, K, V, log_G_Q, log_G_K, dY, dy, rowmax, None, None, None, deg, 1 / scale, 1e-6, False, False, False
    )
    dlog_G = dlog_G_Q + dlog_G_K
    return dQ, dK, dV, dlog_G

# Fake implementation for tracing and testing
@attention_bwd_gating.register_fake
def attention_bwd_gating_fake(Q, K, V, log_G_Q, log_G_K, dY, dy, rowmax, deg, scale):
    return torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V), torch.empty_like(log_G_Q)

# Useful function to create sample inputs
def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, scale=1.0, deg=2):
    generator = torch.Generator(device=device).manual_seed(42)
    Q = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) / d**.25 * scale
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) / d**.25 * scale
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) * scale
    dY = torch.randn(size=(b, t, h, d), dtype=dtype, device=device, generator=generator) * scale
    dy = torch.randn(size=(b, t, h), dtype=torch.float32, device=device, generator=generator) * scale
    rowmax = torch.randn(size=(b, t, h), dtype=torch.float32, device=device, generator=generator) * scale
    if gating:
        log_G_Q = torch.zeros(size=(b, t, h), dtype=torch.float32, device=device) - .01
        log_G_K = torch.zeros(size=(b, t, h), dtype=torch.float32, device=device) - .01
        return dict(Q=Q, K=K, V=V, log_G_Q=log_G_Q, log_G_K=log_G_K, dY=dY, dy=dy, rowmax=rowmax, deg=deg, scale=scale)
    else:
        return dict(Q=Q, K=K, V=V, dY=dY, dy=dy, rowmax=rowmax, deg=deg, scale=scale)


## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, t, h, d = (2, 4, 8, 32)
    dtype = torch.float16
    gating = True
    scale = 1.0
    deg = 2
    # Create inputs
    inputs = create_inputs(b, t, h, d, dtype, 'cuda', gating, scale, deg)
    # Run function
    with torch.no_grad():
        dQ, dK, dV, dlog_G = attention_bwd_gating(inputs['Q'], inputs['K'], inputs['V'], inputs['log_G_Q'], inputs['log_G_K'], inputs['dY'], inputs['dy'], inputs['rowmax'], inputs['deg'], inputs['scale'])
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_attention_bwd = torch.compile(attention_bwd_gating, fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            dQ, dK, dV, dlog_G = compiled_attention_bwd(inputs['Q'], inputs['K'], inputs['V'], inputs['log_G_Q'], inputs['log_G_K'], inputs['dY'], inputs['dy'], inputs['rowmax'], inputs['deg'], inputs['scale'])

