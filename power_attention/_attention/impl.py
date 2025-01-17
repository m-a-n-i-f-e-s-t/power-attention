## POWER ATTENTION CUSTOM OP
# Computes the power attention forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import Optional, Tuple
from power_attention._attention.fwd import ExpandedDim as compute_expanded_dim, attention_fwd
from power_attention._attention.bwd import attention_bwd_gatingless, attention_bwd_gating

@torch.library.custom_op("power::attention", mutates_args=())
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, log_G: Optional[torch.Tensor],
              deg: int, scale: Optional[float], eps: float, deterministic: bool, normalize_output: bool, flash_equivalent: bool, normal_space: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute power attention with optional gating.
    
    Computes the power attention operation Y = softmax(Q·K^T/sqrt(d))^deg · V with optional gating factors.
    
    Input shapes:
        Q, K, V: [batch, seq, head, dim] - Query, Key and Value tensors
        log_G: [batch, seq, head] or None - Optional log gating factors
        deg: int - Power attention degree
        scale: float or None - Optional scale of key-query inner product, defaults to 1.0
        eps: float - Small constant for numerical stability
        deterministic: bool - Whether to deterministically accumulate gradients in the backward pass, might lower throughput with small batch sizes
        normalize_output: bool - Whether to normalize the output by the sum of the attention weights, defaults to False
        flash_equivalent: bool - Whether to use flash_equivalent for the attention operation, defaults to False. If True, this is equivalent to flash attention (with optional gating)
        normal_space: bool - Whether to do computation in normal space instead of log space, this helps speed up the kernel but potentially become less stable.
    Output shapes:
        Y: [batch, chunk_n, chunk_size, head, dim] - Scaled output tensor
        y: [batch, chunk_n, chunk_size, head] - Scaled normalization factors
        rowmax: [batch, chunk_n, chunk_size, head] - Rowmax scaling factors

    Input restrictions:
        - Q, K, V must have same shape
        - Q, K, V must be contiguous along last dimension
        - Q, K, V must have same dtype (fp16 or bf16)
        - log_G must be float32 if provided
    """
    #  batch, seq, head, features
    b, t, h, d = Q.shape
    if scale is None:
        scale = 1.0

    if log_G is not None:
        log_G_Q, log_G_K = log_G, log_G.clone()
    else:
        log_G_Q = log_G_K = None

    # TODO (sean): implement normalize_output in kernel if needed
    Y, y, rowmax = attention_fwd(Q, K, V, log_G_Q, log_G_K, deg, scale, eps, flash_equivalent, normal_space)

    rowmax = rowmax.detach()
    return Y, y, rowmax

# Make it traceable
@attention.register_fake
def attention_fake(Q, K, V, log_G, deg, scale, eps, deterministic, normalize_output, flash_equivalent, normal_space):
    b, t, h, d = Q.shape
    return (torch.empty(b, t, h, d, device=Q.device, dtype=Q.dtype), 
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32),
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32))

# Autograd setup
def attention_setup(ctx, inputs, output):
    Q, K, V, log_G, deg, scale, eps, deterministic, normalize_output, flash_equivalent, normal_space = inputs
    _, _, rowmax = output

    b, t, h, d = Q.shape
    D = compute_expanded_dim(d, deg)
    if scale is None:
        scale = torch.tensor(1., dtype=torch.float32, device=Q.device)

    if log_G is not None:
        log_G_Q, log_G_K = log_G, log_G
    else:
        log_G_Q = log_G_K = None

    ctx.save_for_backward(Q, K, V, log_G_Q, log_G_K, rowmax)
    ctx.scale = scale
    ctx.eps = eps
    ctx.deg = deg
    ctx.deterministic = deterministic
    ctx.normalize_output = normalize_output
    ctx.flash_equivalent = flash_equivalent
    ctx.normal_space = normal_space

def attention_backward(ctx, dY, dy, _):
    Q, K, V, log_G_Q, log_G_K, rowmax = ctx.saved_tensors
    if log_G_Q is None:
        dQ, dK, dV = attention_bwd_gatingless(Q, K, V, dY, dy, rowmax, ctx.deg, ctx.scale, ctx.eps, ctx.deterministic, ctx.flash_equivalent, ctx.normal_space)
        dlog_G = None
    else:
        dQ, dK, dV, dlog_G = attention_bwd_gating(Q, K, V, log_G_Q, log_G_K, dY, dy, rowmax, ctx.deg, ctx.scale, ctx.eps, ctx.deterministic, ctx.flash_equivalent, ctx.normal_space)
    assert dQ.is_contiguous(), 'dQ must be contiguous'
    assert dK.is_contiguous(), 'dK must be contiguous'
    assert dV.is_contiguous(), 'dV must be contiguous'
    if dlog_G is not None:
        assert dlog_G.is_contiguous(), 'dlog_G must be contiguous'
    return dQ, dK, dV, dlog_G, None, None, None, None, None, None, None
# Register autograd
torch.library.register_autograd(
    "power::attention", attention_backward, setup_context=attention_setup
)

## Useful function to create sample inputs
def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, requires_grad=False, deterministic=False, seed=42, scale=1.0, p=2, normalize_output=False, std=1.0, flash_equivalent=False, normal_space=False):
    torch.manual_seed(seed)
    D = compute_expanded_dim(d, p)
    Q = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    if gating:
        log_G = F.logsigmoid(torch.rand(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    deg = p
    eps = 1e-5
    if requires_grad:
        Q, K, V, log_G = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return Q, K, V, log_G, deg, scale, eps, deterministic, normalize_output, flash_equivalent, normal_space

## TUTORIAL ##
if __name__ == '__main__':
    from tests_and_benchmarks._timing import report_fwd_bwd

    # Hyperparameters
    b, t, h, d = (8, 1024, 16, 64)
    dtype = torch.float16
    gating = True
    deterministic = False
    flash_equivalent = False
    normal_space = True
    # Create inputs
    inputs = create_inputs(b, t, h, d, dtype, 'cuda', gating, requires_grad=True, deterministic=deterministic, normalize_output=False, flash_equivalent=flash_equivalent, normal_space=normal_space)

    print(f"Benchmarking power attention {b=} {t=} {h=} {d=} {gating=} {dtype=} {deterministic=} {flash_equivalent=} {normal_space=}")

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        O, o, rowmax = attention(*inputs)
        torch.autograd.backward((O, o), grad_tensors=(O, o))

        Q, K, V = inputs[:3]
        from flash_attn import flash_attn_func
        O_flash = flash_attn_func(Q, K, V, dropout_p=0.0, softmax_scale=None, causal=True)
        torch.autograd.backward((O_flash,), grad_tensors=(O_flash,))

    else:
        # benchmark 
        report_fwd_bwd(attention, *inputs)

        # Run flash attention
        Q, K, V = inputs[:3]
        from flash_attn import flash_attn_func
        print("benchmarking flash attention")
        report_fwd_bwd(flash_attn_func, Q, K, V, dropout_p=0.0, softmax_scale=None, causal=True)


