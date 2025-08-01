## POWER ATTENTION CUSTOM OP
# Computes the power attention forward pass and backward pass using CUDA.
import torch
from typing import Optional, Tuple
from power_attention_cuda import attention_fwd as attention_fwd_cuda, attention_bwd as attention_bwd_cuda
from power_attention._config import normal_space, flash_equivalent, eps

## FWD ##

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


## BWD ##

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


## IMPLEMENTATION ##


@torch.library.custom_op("power::attention", mutates_args=())
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
              log_G: Optional[torch.Tensor], deg: int, scale: Optional[float],
              causal:bool=True, head_first:bool=False, norm:bool=True, use_log2:bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Computes the power attention operation Y = ((Q·K^T) * scale)^deg · V with optional gating factors.
    
    This function implements the core symmetric power attention mechanism from [1]. It replaces
    softmax attention with an even power `deg`, which is equivalent to a linear transformer
    with symmetric power embeddings. This formulation allows for more focused attention while
    maintaining efficient computation through the linear transformer framework.

    For a sequence of queries Q, keys K, and values V, the attention mechanism computes:

    $$Y_i = \sum_{j=1}^i A_{ij} V_j$$

    where the attention weights without gating are:

    $$A_{ij} = \frac{\phi(Q_i)^\top \phi(K_j)}{\sum_{k=1}^i \phi(Q_i)^\top \phi(K_k)}$$

    Here ϕ is the symmetric power embedding that maps vectors to their deg-th symmetric power.
    With gating (if log_G is provided):

    $$A_{ij} = \frac{\phi(Q_i)^\top \phi(K_j) \exp(\log G_j)}{\sum_{k=1}^i \phi(Q_i)^\top \phi(K_k) \exp(\log G_k)}$$

    Args:
        Q: Query tensor of shape `(batch_size, seq_len, num_heads, head_dim)`.
        K: Key tensor of shape `(batch_size, seq_len, num_heads, head_dim)`.
        V: Value tensor of shape `(batch_size, seq_len, num_heads, head_dim)`.
        log_G: Optional log gating factors of shape `(batch_size, seq_len, num_heads)`.
            When provided, applies multiplicative gating to attention weights.
        deg: Power attention degree. Must be even. Higher values make attention more "focused".
        scale: Optional scale factor for Q·K^T. Usually 1/sqrt(head_dim).

    Returns:
        Tuple containing:
            - Y: Output tensor of shape `(batch_size, seq_len, num_heads, head_dim)`.
            - y: Normalization factors of shape `(batch_size, seq_len, num_heads)`.
            - rowmax: Row-wise maximum values of shape `(batch_size, seq_len, num_heads)`, always in log space.

    Note:
        - Input tensors must have matching dtypes (fp16 or bf16)
        - If provided, log_G must be float32
        - Q, K, V must be contiguous along the last dimension
        - deg must be even for the symmetric power formulation

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    assert causal and not head_first and not use_log2, "Only the default arguments are supported for the CUDA implementation"

    #  batch, seq, head, features
    b, t, h, d = Q.shape
    if scale is None:
        scale = 1.0

    if log_G is not None:
        log_G_Q, log_G_K = log_G, log_G.clone()
    else:
        log_G_Q = log_G_K = None

    # TODO (sean): implement normalize_output in kernel if needed
    Y, y, rowmax = attention_fwd(Q, K, V, log_G_Q, log_G_K, deg, scale)

    rowmax = rowmax.detach()
    if normal_space:
        rowmax = torch.log(rowmax ** deg)
    if norm:
        return (Y / y.unsqueeze(-1)).to(Y.dtype), rowmax
    else:
        return Y, rowmax

# Make it traceable
@attention.register_fake
def attention_fake(Q, K, V, log_G, deg, scale, causal=True, head_first=False, norm=True, use_log2=False):
    b, t, h, d = Q.shape
    return (torch.empty(b, t, h, d, device=Q.device, dtype=Q.dtype), 
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32))

# Autograd setup
def attention_setup(ctx, inputs, output):
    Q, K, V, log_G, deg, scale, *_ = inputs
    _, rowmax = output

    b, t, h, d = Q.shape
    if scale is None:
        scale = torch.tensor(1., dtype=torch.float32, device=Q.device)

    if log_G is not None:
        log_G_Q, log_G_K = log_G, log_G
    else:
        log_G_Q = log_G_K = None

    ctx.save_for_backward(Q, K, V, log_G_Q, log_G_K, rowmax)
    ctx.scale = scale
    ctx.deg = deg

def attention_backward(ctx, dY, _):
    Q, K, V, log_G_Q, log_G_K, rowmax = ctx.saved_tensors
    b, t, h, d = Q.shape
    dy = torch.zeros(b, t, h, device=Q.device, dtype=torch.float32)
    if log_G_Q is None:
        dQ, dK, dV = attention_bwd_gatingless(Q, K, V, dY, dy, rowmax, ctx.deg, ctx.scale)
        dlog_G = None
    else:
        dQ, dK, dV, dlog_G = attention_bwd_gating(Q, K, V, log_G_Q, log_G_K, dY, dy, rowmax, ctx.deg, ctx.scale)
    assert dQ.is_contiguous(), 'dQ must be contiguous'
    assert dK.is_contiguous(), 'dK must be contiguous'
    assert dV.is_contiguous(), 'dV must be contiguous'
    if dlog_G is not None:
        assert dlog_G.is_contiguous(), 'dlog_G must be contiguous'
    return dQ, dK, dV, dlog_G, None, None, None, None, None, None, None, None
# Register autograd
torch.library.register_autograd(
    "power::attention", attention_backward, setup_context=attention_setup
)


## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_runtime
    from power_attention._attention.create_inputs import create_inputs

    # Hyperparameters
    b, t, h, d = (8, 1024, 16, 64)
    dtype = torch.float16
    deg = 2
    device = 'cuda'
    gating = True
    # Create inputs
    inputs = create_inputs(b, t, h, d, dtype, device, gating, requires_grad=True, scale=1.0, deg=deg)

    print(f"Benchmarking power attention {b=} {t=} {h=} {d=} {gating=} {dtype=}")

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
        print_runtime(attention, *inputs)

        # Run flash attention
        Q, K, V = inputs[:3]
        from flash_attn import flash_attn_func
        print("benchmarking flash attention")
        print_runtime(flash_attn_func, Q, K, V, dropout_p=0.0, softmax_scale=None, causal=True)


