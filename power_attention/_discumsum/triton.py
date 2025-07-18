import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def prune_configs_by_size(configs, nargs, **kwargs):
    """Prune autotune configs based on input size"""
    D = nargs.get('D', 1)
    pruned_configs = []
    for config in configs:
        # For small feature dimensions, use smaller block sizes
        block_d = config.kwargs['BLOCK_D']
        if D <= 64 and block_d <= 128:
            pruned_configs.append(config)
        elif D <= 256 and block_d <= 512:
            pruned_configs.append(config)
        elif D > 256:
            pruned_configs.append(config)
    return pruned_configs if pruned_configs else configs


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
    ],
    key=['D'],
    prune_configs_by={'early_config_prune': prune_configs_by_size},
)
@triton.jit
def discumsum_fwd_kernel(
    X_ptr, log_G_ptr, Y_ptr,
    B, T, H, D,
    stride_X_b, stride_X_t, stride_X_h, stride_X_d,
    stride_G_b, stride_G_t, stride_G_h,
    stride_Y_b, stride_Y_t, stride_Y_h, stride_Y_d,
    BLOCK_D: tl.constexpr,
):
    """
    Forward kernel for discounted cumulative sum.
    Y[0] = 0
    Y[t+1] = Y[t] * exp(log_G[t]) + X[t] for t = 0, 1, ..., T-1
    """
    # Get program IDs
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_d = tl.program_id(axis=2)
    
    # Compute offsets
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < D
    
    # Base pointers for this batch and head
    X_base = X_ptr + pid_b * stride_X_b + pid_h * stride_X_h
    G_base = log_G_ptr + pid_b * stride_G_b + pid_h * stride_G_h
    Y_base = Y_ptr + pid_b * stride_Y_b + pid_h * stride_Y_h
    
    # Initialize accumulator (state) with zeros
    state = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # Set Y[0] = 0
    Y_ptrs = Y_base + d_offset * stride_Y_d
    tl.store(Y_ptrs, tl.zeros((BLOCK_D,), dtype=Y_ptr.dtype.element_ty), mask=d_mask)
    
    # Main loop over time steps
    for t in range(T):
        # Load log_G[t] and compute exp(log_G[t])
        G_ptr = G_base + t * stride_G_t
        log_g = tl.load(G_ptr)
        g = tl.exp(log_g)
        
        # Load X[t]
        X_ptrs = X_base + t * stride_X_t + d_offset * stride_X_d
        x = tl.load(X_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        
        # Update state: state = state * g + x
        state = state * g + x
        
        # Store Y[t+1] = state
        Y_ptrs = Y_base + (t + 1) * stride_Y_t + d_offset * stride_Y_d
        tl.store(Y_ptrs, state.to(Y_ptr.dtype.element_ty), mask=d_mask)


@triton.jit
def discumsum_bwd_kernel(
    dY_ptr, Y_ptr, log_G_ptr, dX_ptr, dG_ptr,
    B, T, H, D,
    stride_dY_b, stride_dY_t, stride_dY_h, stride_dY_d,
    stride_Y_b, stride_Y_t, stride_Y_h, stride_Y_d,
    stride_G_b, stride_G_t, stride_G_h,
    stride_dX_b, stride_dX_t, stride_dX_h, stride_dX_d,
    stride_dG_b, stride_dG_t, stride_dG_h,
    BLOCK_D: tl.constexpr,
):
    """
    Backward kernel for discounted cumulative sum.
    Process in reverse time order:
    dstate = dstate * exp(log_G[t]) + dY[t+1]
    dX[t] = dstate
    dlog_G[t] = sum_over_d(Y[t] * dstate * exp(log_G[t]))
    """
    # Get program IDs - process all features in one block per (batch, head)
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    
    # Process all features in this kernel
    d_offset = tl.arange(0, BLOCK_D)
    d_mask = d_offset < D
    
    # Base pointers for this batch and head
    dY_base = dY_ptr + pid_b * stride_dY_b + pid_h * stride_dY_h
    Y_base = Y_ptr + pid_b * stride_Y_b + pid_h * stride_Y_h
    G_base = log_G_ptr + pid_b * stride_G_b + pid_h * stride_G_h
    dX_base = dX_ptr + pid_b * stride_dX_b + pid_h * stride_dX_h
    dG_base = dG_ptr + pid_b * stride_dG_b + pid_h * stride_dG_h
    
    # Initialize gradient accumulator (dstate)
    dstate = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # Process in reverse time order
    for t in range(T - 1, -1, -1):
        # Load dY[t+1]
        dY_ptrs = dY_base + (t + 1) * stride_dY_t + d_offset * stride_dY_d
        dy = tl.load(dY_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        
        # Load log_G[t] and compute exp(log_G[t])
        G_ptr = G_base + t * stride_G_t
        log_g = tl.load(G_ptr)
        g = tl.exp(log_g)
        
        # Update dstate: dstate = dstate * g + dY[t+1]
        dstate = dstate * g + dy
        
        # Store dX[t] = dstate
        dX_ptrs = dX_base + t * stride_dX_t + d_offset * stride_dX_d
        tl.store(dX_ptrs, dstate.to(dX_ptr.dtype.element_ty), mask=d_mask)
        
        # Compute gradient w.r.t. log_G
        # Load Y[t] for gradient computation
        Y_ptrs = Y_base + t * stride_Y_t + d_offset * stride_Y_d
        y = tl.load(Y_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        
        # Compute dlog_G contribution: Y[t] * dstate * exp(log_G[t])
        dg_local = y * dstate * g
        dg_sum = tl.sum(dg_local, axis=0)
        
        # Store directly (no atomic needed since only one block per (b,h))
        dG_ptr = dG_base + t * stride_dG_t
        tl.store(dG_ptr, dg_sum)


def discumsum_fwd_triton(X: torch.Tensor, log_G: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of discounted cumulative sum forward pass.
    
    Args:
        X: Input tensor [B, T, H, D]
        log_G: Log discount factors [B, T, H]
    
    Returns:
        Y: Output tensor [B, T+1, H, D]
    """
    B, T, H, D = X.shape
    assert log_G.shape == (B, T, H), f"Expected log_G shape ({B}, {T}, {H}), got {log_G.shape}"
    
    # Create output tensor with +1 time dimension
    Y = torch.empty(B, T + 1, H, D, dtype=X.dtype, device=X.device)
    
    # Grid configuration: one block per (batch, head, feature_block)
    num_D_blocks = triton.cdiv(D, 64)  # Start with 64 as minimum block size
    grid = (B, H, num_D_blocks)
    
    # Launch kernel
    discumsum_fwd_kernel[grid](
        X, log_G, Y,
        B, T, H, D,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        log_G.stride(0), log_G.stride(1), log_G.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
    )
    
    return Y


def discumsum_bwd_triton(
    dY: torch.Tensor, 
    Y: torch.Tensor, 
    log_G: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of discounted cumulative sum backward pass.
    
    Args:
        dY: Gradient w.r.t. output [B, T+1, H, D]
        Y: Forward output [B, T+1, H, D]
        log_G: Log discount factors [B, T, H]
    
    Returns:
        dX: Gradient w.r.t. input [B, T, H, D]
        dlog_G: Gradient w.r.t. log discount factors [B, T, H]
    """
    B, T_plus_1, H, D = dY.shape
    T = T_plus_1 - 1
    assert Y.shape == (B, T + 1, H, D), f"Expected Y shape ({B}, {T+1}, {H}, {D}), got {Y.shape}"
    assert log_G.shape == (B, T, H), f"Expected log_G shape ({B}, {T}, {H}), got {log_G.shape}"
    
    # Create output tensors
    dX = torch.empty(B, T, H, D, dtype=dY.dtype, device=dY.device)
    dlog_G = torch.zeros(B, T, H, dtype=torch.float32, device=dY.device)

    # Use power-of-2 block size that covers all features
    BLOCK_D = triton.next_power_of_2(D)
    
    # Grid: one block per (batch, head) - no feature blocking
    grid = (B, H)
    
    # Launch kernel
    discumsum_bwd_kernel[grid](
        dY, Y, log_G, dX, dlog_G,
        B, T, H, D,
        dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        log_G.stride(0), log_G.stride(1), log_G.stride(2),
        dX.stride(0), dX.stride(1), dX.stride(2), dX.stride(3),
        dlog_G.stride(0), dlog_G.stride(1), dlog_G.stride(2),
        BLOCK_D=BLOCK_D,
    )
    
    return dX, dlog_G


def discumsum_triton(X: torch.Tensor, log_G: torch.Tensor) -> torch.Tensor:
    """
    Main Triton interface for discounted cumulative sum, matching the CUDA API.
    
    Args:
        X: Input tensor [B, T, H, D] or [B, T, H, *feature_dims]
        log_G: Log discount factors [B, T, H]
    
    Returns:
        Y: Output tensor [B, T+1, H, D] or [B, T+1, H, *feature_dims]
    """
    original_shape = X.shape
    B, T, H = original_shape[:3]
    
    # Flatten feature dimensions if needed
    if len(original_shape) > 4:
        X = X.view(B, T, H, -1)
    
    # Run forward pass
    Y = discumsum_fwd_triton(X, log_G)
    
    # Restore original feature dimensions
    if len(original_shape) > 4:
        new_shape = (B, T + 1, H) + original_shape[3:]
        Y = Y.view(new_shape)
    
    return Y


# Autograd function to connect forward and backward
class DiscumsumTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, log_G):
        # Flatten feature dimensions for computation
        original_shape = X.shape
        B, T, H = original_shape[:3]
        
        if len(original_shape) > 4:
            X_flat = X.view(B, T, H, -1)
        else:
            X_flat = X
        
        Y_flat = discumsum_fwd_triton(X_flat, log_G)
        
        # Restore original shape
        if len(original_shape) > 4:
            new_shape = (B, T + 1, H) + original_shape[3:]
            Y = Y_flat.view(new_shape)
        else:
            Y = Y_flat
        
        ctx.save_for_backward(log_G, Y_flat)
        ctx.original_shape = original_shape
        
        return Y
    
    @staticmethod
    def backward(ctx, dY):
        log_G, Y_flat = ctx.saved_tensors
        original_shape = ctx.original_shape
        B, T, H = original_shape[:3]
        
        # Flatten gradient for computation
        if len(dY.shape) > 4:
            dY_flat = dY.view(B, T + 1, H, -1)
        else:
            dY_flat = dY
        
        # Compute gradients
        dX_flat, dlog_G = discumsum_bwd_triton(dY_flat, Y_flat, log_G)
        
        # Restore original shape for dX
        if len(original_shape) > 4:
            dX = dX_flat.view(original_shape)
        else:
            dX = dX_flat
        
        return dX, dlog_G


# Main user-facing function
@torch.compiler.disable
def discumsum(X: torch.Tensor, log_G: torch.Tensor) -> torch.Tensor:
    """
    Triton-based discounted cumulative sum with autograd support.
    
    This is a drop-in replacement for the CUDA version.
    """
    return DiscumsumTritonFunction.apply(X, log_G)


if __name__ == '__main__':
    # Comprehensive test
    from power_attention._discumsum.create_inputs import create_inputs
    from power_attention._discumsum.reference import discumsum_reference
    from power_attention._utils import diff
    
    # Test configurations: (B, T, H, D, d, dtype)
    test_configs = [
        (2, 16, 4, 128, 1, torch.float32, "FP32"),
        (2, 16, 4, 128, 1, torch.float16, "FP16"),
        (2, 16, 4, 128, 1, torch.bfloat16, "BF16"),
        (2, 2, 4, 128, 1, torch.bfloat16, "Short BF16"),
        (1, 32, 2, 256, 1, torch.float32, "Long FP32"),
    ]
    
    
    for i, (B, T, H, D, d, dtype, desc) in enumerate(test_configs):
        print(f'\nTest {i+1}: {desc} - B={B}, T={T}, H={H}, D={D}')
        atol = {
            torch.float32: 1e-5,
            torch.float16: 5e-2,
            torch.bfloat16: 3e-1,
        }[dtype]
        
        # Create test inputs
        inputs = create_inputs(B, T, H, D, d, dtype, 'cuda', requires_grad=True)
        X = (inputs['X'].view(B, T, H, -1) * 1e-1).detach().clone().requires_grad_(True)  # Flatten to 4D
        X_ref = X.detach().clone().requires_grad_(True)
        log_G = (-inputs['log_G'].abs()).detach().clone().requires_grad_(True)
        log_G_ref = log_G.detach().clone().requires_grad_(True)
        
        # Test forward pass
        Y_triton = discumsum(X, log_G)
        Y_ref = discumsum_reference(X_ref, log_G_ref)
        
        diff(Y_ref, Y_triton, verbose=True, title=f'{desc} Forward', atol=atol, assert_close=False)
        
        # Test backward pass
        Y_triton.sum().backward()
        Y_ref.sum().backward()
        
        diff(X_ref.grad, X.grad, verbose=True, title=f'{desc} Backward', atol=atol, assert_close=False)
        diff(log_G_ref.grad, log_G.grad, verbose=True, title=f'{desc} Backward', atol=atol, assert_close=False)
        
    