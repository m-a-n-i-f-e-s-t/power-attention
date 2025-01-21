## POWER ATTENTION CUSTOM OP
# Computes the power attention forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from typing import Optional, Tuple
from power_attention._attention.fwd import attention_fwd
from power_attention._attention.bwd import attention_bwd_gatingless, attention_bwd_gating
from power_attention._config import normal_space
@torch.library.custom_op("power::attention", mutates_args=())
def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
              log_G: Optional[torch.Tensor], deg: int, scale: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            - rowmax: Row-wise maximum values of shape `(batch_size, seq_len, num_heads)`.

    Note:
        - Input tensors must have matching dtypes (fp16 or bf16)
        - If provided, log_G must be float32
        - Q, K, V must be contiguous along the last dimension
        - deg must be even for the symmetric power formulation

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
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
    Y, y, rowmax = attention_fwd(Q, K, V, log_G_Q, log_G_K, deg, scale)

    rowmax = rowmax.detach()
    if normal_space:
        rowmax = torch.log(rowmax ** deg)
    return Y, y, rowmax

# Make it traceable
@attention.register_fake
def attention_fake(Q, K, V, log_G, deg, scale):
    b, t, h, d = Q.shape
    return (torch.empty(b, t, h, d, device=Q.device, dtype=Q.dtype), 
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32),
            torch.empty(b, t, h, device=Q.device, dtype=torch.float32))

# Autograd setup
def attention_setup(ctx, inputs, output):
    Q, K, V, log_G, deg, scale = inputs
    _, _, rowmax = output

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

def attention_backward(ctx, dY, dy, _):
    Q, K, V, log_G_Q, log_G_K, rowmax = ctx.saved_tensors
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
    return dQ, dK, dV, dlog_G, None, None
# Register autograd
torch.library.register_autograd(
    "power::attention", attention_backward, setup_context=attention_setup
)

## Useful function to create sample inputs
def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, requires_grad=False, seed=42, scale=1.0, deg=2, std=1.0):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    if gating:
        log_G = F.logsigmoid(torch.rand(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    if requires_grad:
        Q, K, V, log_G = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, scale=scale)

## TUTORIAL ##
if __name__ == '__main__':
    from perf._timing import report_fwd_bwd

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
        report_fwd_bwd(attention, *inputs)

        # Run flash attention
        Q, K, V = inputs[:3]
        from flash_attn import flash_attn_func
        print("benchmarking flash attention")
        report_fwd_bwd(flash_attn_func, Q, K, V, dropout_p=0.0, softmax_scale=None, causal=True)


