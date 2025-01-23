## POWER FULL KERNEL ##
# Implements the power self-attention algorithm using CUDA kernels.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from enum import Enum
from torch.utils._pytree import tree_map
from power_attention._attention import attention, attention_reference
from power_attention._update_state import update_state, update_state_reference
from power_attention._discumsum import discumsum, discumsum_reference
from power_attention._query_state import query_state, query_state_reference
from power_attention._utils import compute_expanded_dim, layernorm
import math


class UpdateStateImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    MATMUL = 2

class QueryStateImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    MATMUL = 2

class DiscumsumImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1

class AttentionImpl(Enum):
    CUTLASS = 0
    REFERENCE = 1
    TRITON = 2

def update_state_matmul(K, V, *args):
    # K: [b,n,c,h,D]
    # V: [b,n,c,h,d]
    # Output: [b,n,h,D,d]
    return torch.matmul(K.permute(0, 1, 3, 4, 2), V.transpose(2, 3))  # [b,n,h,D,d]

def post_query_state(Y, rowmax, scale, zero_initial_state):
    """Post-process the query state output with layernorm.
    """
    scale_tensor = torch.tensor(1 / scale, device=rowmax.device, dtype=rowmax.dtype)
    scale_attn = torch.exp(-rowmax)
    min_scale = torch.min(scale_tensor, scale_attn)
    if zero_initial_state:
        min_scale[:, 0:1] = scale_attn.narrow(1, 0, 1)
    return layernorm(Y, eps=min_scale*1e-5)

def query_state_matmul(Q, S, attn_Y, rowmax, deg, scale, zero_initial_state):
    """Query state implementation using matmul. This implementation is used when deg == 1.
    Note that unlike the cutlass implementation, the final scaling doesn't discriminate between
    the first chunk and the rest, because torch.compile doesn't work well with in-place operations.

    Args:
        Q: [b,n,c,h,D]
        S: [b,n,h,D,d]
        attn_Y: [b,n,c,h,d]
        rowmax: [b,n,c,h]
        deg: int
        scale: float
        zero_initial_state: bool
    """
    b, n, c, h, d = Q.shape
    Y = torch.matmul(Q.to(Q.dtype).transpose(2, 3), S / scale).transpose(2, 3)  # [b,n,c,h,d]
    scale_qs = torch.tensor(1 / scale, dtype=Q.dtype, device=Q.device)
    scale_attn = torch.exp(-rowmax)
    min_scale = torch.min(scale_qs, scale_attn)
    qs_factor = min_scale / scale_qs
    attn_factor = min_scale / scale_attn
    Y = attn_Y * attn_factor.unsqueeze(-1) + Y * qs_factor.unsqueeze(-1)
    return layernorm(Y, eps=min_scale*1e-5)

IMPL_MAP = {
    UpdateStateImpl.CUTLASS: update_state,
    UpdateStateImpl.REFERENCE: update_state_reference,
    UpdateStateImpl.MATMUL: update_state_matmul,
    QueryStateImpl.CUTLASS: query_state,
    QueryStateImpl.REFERENCE: query_state_reference,
    QueryStateImpl.MATMUL: query_state_matmul,
    DiscumsumImpl.CUTLASS: discumsum,
    DiscumsumImpl.REFERENCE: discumsum_reference,
    AttentionImpl.CUTLASS: attention,
    AttentionImpl.REFERENCE: attention_reference,
    # AttentionImpl.TRITON: attention_triton,
}

POWER_FULL_DOC = r"""
Compute symmetric power attention with optional chunking.

This function implements the symmetric power attention mechanism from [1]. It generalizes
linear transformers by using symmetric power embeddings, which provide better expressivity
while maintaining tractable state sizes.

For a sequence of queries $Q_i$, keys $K_i$, and values $V_i ∈ ℝ^d$, the attention mechanism
computes outputs $Y_i ∈ ℝ^d$ as:

$$Y_i = Norm(\sum_{j=1}^i A_{ij} V_j)$$

where $Norm$ is a parameter-free layer normalization as follows:

$$Norm(x) = \frac{x - \mu(x)}{\sigma(x)}$$

where $\mu(x)$ and $\sigma(x)$ are the mean and standard deviation of $x$ along the feature dimension.

The attention weights are computed as follows:

$$A_{ij} = \frac{\phi(Q_i)^\top \phi(K_j)}{\sum_{k=1}^i \phi(Q_i)^\top \phi(K_k)}$$

Here $\phi$ is the symmetric power embedding that maps vectors to their deg-th symmetric power.
For long sequences, we use an equivalent RNN formulation with states $S_i$ and $Z_i$:

$$Y_{i} = \frac{S_i \phi(Q_i)}{Z_i \phi(Q_i)} \qquad Z_i = Z_{i-1} + \phi(K_i)^T \qquad S_i = S_{i-1} + V_i \phi(K_i)^T$$

The state size for each head is $D(d+1)$ where $D = \binom{d+deg-1}{deg}$, providing massive
savings over full tensor products (e.g., 96% reduction for deg=4).

Args:
    Q: Query tensor of shape `(batch_size, seq_len, num_q_heads, head_dim)`.
    K: Key tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
    V: Value tensor of shape `(batch_size, seq_len, num_kv_heads, head_dim)`.
    log_G: Optional log gating factors of shape `(batch_size, seq_len, num_kv_heads)`.
        When provided, applies multiplicative gating to attention weights.
    initial_state: Optional initial state for recurrent processing. Not implemented yet.
    deg: Power attention degree. Must be even. Higher values make attention more "focused".
        Common values are:
        * deg=2: 49% state size reduction, slightly worse than baseline
        * deg=4: 96% reduction, outperforms baseline
        * deg=6: 99.8% reduction, best performance but large state
    scale: Scale factor for attention weights. Defaults to 1.0.
    chunk_size: Size of chunks for processing long sequences.
        If None, uses O(n²) attention formulation.
        If set, uses O(n) RNN formulation with chunked computation.

Returns:
    torch.Tensor: Output tensor of shape `(batch_size, seq_len, num_q_heads, head_dim)`.

Note:
    - Input tensors must have matching dtypes (fp16, bf16, or fp32)
    - If provided, log_G must be float32
    - Sequence length must be evenly divisible by chunk size
    - num_q_heads must be a multiple of num_kv_heads (for multi-query attention)
    - deg must be even for the symmetric power formulation
    - State size per head is $D(d+1)$ where $D = \binom{d+deg-1}{deg}$

References:
    [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
        Manifest AI, Aug. 15, 2024.
"""


def _make_power_full(update_state_impl: UpdateStateImpl, query_state_impl: QueryStateImpl, discumsum_impl: DiscumsumImpl, attention_impl: AttentionImpl):
    """ Create a power_full function with the given implementations.
    """
    def _power_full(Q, K, V, log_G=None, initial_state=None,
                deg=2, scale=None, chunk_size=None): # noqa: C901
        if initial_state is not None:
            raise NotImplementedError('Initial state not implemented')

        if deg == 1: # when deg == 1, update_state and query_state are essentially matmuls
            _update_state = IMPL_MAP[UpdateStateImpl.MATMUL]
            _query_state = IMPL_MAP[QueryStateImpl.MATMUL]
        else:
            _update_state = IMPL_MAP[update_state_impl]
            _query_state = IMPL_MAP[query_state_impl]
        _discumsum = IMPL_MAP[discumsum_impl]
        _attention = IMPL_MAP[attention_impl]
        
        # Establish shapes and dtypes
        assert Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
        dtype = Q.dtype
        b, t, hq, d = Q.shape
        _, _, h, _ = K.shape
        assert hq % h == 0, f"Q heads must be a multiple of KV heads: {hq=} {h=}"
        qhead_ratio = hq // h
        if chunk_size is not None:
            c = chunk_size
            assert t % chunk_size == 0, f'{t=} not evenly divisible by {chunk_size=}'
            n = t // chunk_size
        else:
            c = t
            n = 1
        gating = log_G is not None
        if gating:
            log_G = log_G.to(torch.float32)

        if not scale:
            scale = 1.0 / d**0.5

        # Quick return for simple quadratic attention
        if t <= c:
            if qhead_ratio > 1:
                K = K.repeat_interleave(qhead_ratio, dim=2)
                V = V.repeat_interleave(qhead_ratio, dim=2)
                if gating:
                    log_G = log_G.repeat_interleave(qhead_ratio, dim=2)
            log_G_accum = log_G.cumsum(1) if log_G is not None else None
            Y, _, rowmax = _attention(Q, K, V, log_G_accum, deg, scale)
            assert Y.is_contiguous(), 'Y must be contiguous'
            rowmax = rowmax - math.log(scale)
            out = layernorm(Y, eps=torch.exp(-rowmax)*1e-5)
            return out

        # Reshape into chunks
        Q = Q.view(b, n, c, hq, d)
        K = K.view(b, n, c, h, d)
        V = V.view(b, n, c, h, d)    
        if gating:
            log_G = log_G.view(b, n, c, h).to(torch.float32)
            log_G_intrachunk_accum = log_G.cumsum(2)

        # Compute chunk states
        if gating:
            log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
            cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
        else:
            cs_K = K
        S = _update_state(cs_K.contiguous(), V.contiguous(), deg)

        # Accumulate
        if gating:
            log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
        else:
            log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
        S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
        S = S.narrow(1, 0, n)

        # Restrict to non-initial chunks for recurrent outputs
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        if gating:
            log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous()

        # Compute attention
        Q_flatbatch = Q.view(b*n, c, hq, d)
        K_flatbatch = K.view(b*n, c, h, d)
        V_flatbatch = V.view(b*n, c, h, d)    
        log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum.view(b*n, c, h) if gating else None
        if qhead_ratio > 1:
            K_flatbatch = K_flatbatch.repeat_interleave(qhead_ratio, dim=2)
            V_flatbatch = V_flatbatch.repeat_interleave(qhead_ratio, dim=2)
            if gating:
                log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum_flatbatch.repeat_interleave(qhead_ratio, dim=2)
        attn_Y, _, rowmax = _attention(Q_flatbatch, K_flatbatch, V_flatbatch, log_G_intrachunk_accum_flatbatch, deg, scale)
        attn_Y = attn_Y.view(b, n, c, hq, d)
        rowmax = rowmax.view(b, n, c, hq).detach()
        rowmax = rowmax - math.log(scale)

        if gating:
            if qhead_ratio > 1:
                log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
            Q = Q * torch.exp(log_G_intrachunk_accum / deg).unsqueeze(-1).to(Q.dtype)
        if qhead_ratio > 1:
            S = S.repeat_interleave(qhead_ratio, dim=2)
        D = float(compute_expanded_dim(d, deg))
        Y = _query_state(Q.contiguous(), S.contiguous(), attn_Y.contiguous(), rowmax.contiguous(), deg, D, initial_state is None)
        if deg > 1: # TODO(sean) remove this when generical-p kernel is ready
            Y = post_query_state(Y, rowmax, D, initial_state is None)

        # Epilogue
        out = Y.contiguous().view(b, t, hq, d).to(dtype)
        return out

    _power_full.__doc__ = POWER_FULL_DOC
    return _power_full

power_full_reference = _make_power_full(UpdateStateImpl.REFERENCE, QueryStateImpl.REFERENCE, DiscumsumImpl.REFERENCE, AttentionImpl.REFERENCE)

power_full = _make_power_full(UpdateStateImpl.CUTLASS, QueryStateImpl.CUTLASS, DiscumsumImpl.CUTLASS, AttentionImpl.CUTLASS)

## Useful function to create sample inputs
def create_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    if gating:
        log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    initial_state = None
    scale = 1.0
    if requires_grad:
        Q, K, V, log_G, initial_state = tree_map(
            lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state,
                deg=deg, scale=scale,
                chunk_size=chunk_size)


## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_fwd_bwd

    # Create inputs
    t = 1024
    chunk_size=128
    b = 8
    h = 16
    d = 64
    deg = 2
    gating = True
    dtype = torch.float16
    inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)
    
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        O = power_full(**inputs)
        torch.autograd.backward((O,), grad_tensors=(O,))
    else:
        # Benchmark
        print(f"Benchmarking power_full {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

        print_fwd_bwd(power_full, **inputs)
