## POWER FULL KERNEL ##
# Implements the power self-attention algorithm using CUDA kernels.

## IMPLEMENTATION ##
import torch

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
    temporal: Whether to use temporal normalization. Disabling this can hurt learning performance but may run a bit faster.

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


def make_power_full(update_state_impl, query_state_impl, discumsum_impl, attention_impl):
    """ Create a power_full function with the given implementations.
    """
    def _power_full(Q, K, V, log_G=None, initial_state=None, return_final_state=False,
                    deg=2, scale=None, chunk_size=None): # noqa: C901
        assert deg % 2 == 0, f'deg must be even: {deg=}'
        
        _update_state = update_state_impl
        _query_state = query_state_impl
        _discumsum = discumsum_impl
        _attention = attention_impl
        
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

        # --- Simple quadratic attention ---
        if t <= c:
            if qhead_ratio > 1:
                K = K.repeat_interleave(qhead_ratio, dim=2)
                V = V.repeat_interleave(qhead_ratio, dim=2)
                if gating:
                    log_G = log_G.repeat_interleave(qhead_ratio, dim=2)
            log_G_accum = log_G.cumsum(1) if log_G is not None else None
            out = _attention(Q, K, V, log_G_accum, deg, scale=scale, norm=True)
            if not return_final_state:
                return out
            else:
                if gating:
                    log_discount_weights = (log_G_accum.narrow(1, c-1, 1) - log_G_accum) / deg
                    cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
                S_final = _update_state(cs_K.unsqueeze(1).contiguous(), V.unsqueeze(1).contiguous(), deg)[:,-1]
                if initial_state is not None:
                    S_final += initial_state
                return out, S_final

        # --- Reshape into chunks ---
        Q = Q.view(b, n, c, hq, d)
        K = K.view(b, n, c, h, d)
        V = V.view(b, n, c, h, d)    
        if gating:
            log_G = log_G.view(b, n, c, h)
            log_G_intrachunk_accum = log_G.cumsum(2)

        # --- Update State ---
        if gating:
            log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / deg
            cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
        else:
            cs_K = K
        S, s = _update_state(cs_K.contiguous(), V.contiguous(), deg)

        # TODO(sean): properly handle initial state gating
        if initial_state is not None:
            S = torch.cat([initial_state.unsqueeze(1), S], dim=1) # n + 1 chunks

        # --- Accumulate ---
        if gating:
            log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
        else:
            log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
        S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence
        S = S.narrow(1, 0 if initial_state is None else 1, n)
        s = _discumsum(s, log_G_chunk_sum)
        s = s.narrow(1, 0, n)
        if return_final_state:
            final_state = S[:,-1]

        # --- Merge chunks for attention ---
        Q, K, V = map(lambda x: x.contiguous(), (Q, K, V))
        if gating:
            log_G_intrachunk_accum = log_G_intrachunk_accum.contiguous()

        # TODO(sean): fuse the qhead ratio into attention kernel, it already supports it
        Q_flatbatch = Q.view(b*n, c, hq, d)
        K_flatbatch = K.view(b*n, c, h, d)
        V_flatbatch = V.view(b*n, c, h, d)
        log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum.view(b*n, c, h) if gating else None
        if qhead_ratio > 1:
            K_flatbatch = K_flatbatch.repeat_interleave(qhead_ratio, dim=2)
            V_flatbatch = V_flatbatch.repeat_interleave(qhead_ratio, dim=2)
            if gating:
                log_G_intrachunk_accum_flatbatch = log_G_intrachunk_accum_flatbatch.repeat_interleave(qhead_ratio, dim=2)

        # --- Compute attention ---
        attn_Y, l_attn, rowmax = _attention(Q_flatbatch, K_flatbatch, V_flatbatch, log_G_intrachunk_accum_flatbatch, deg, scale=scale, norm=False)
        attn_Y, l_attn, rowmax = map(lambda x: x.view(b, n, *x.shape[1:]), (attn_Y, l_attn, rowmax)) # [b, n, c, hq ...]
        # --- Gate Query for Query State ---
        if gating:
            if qhead_ratio > 1:
                log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
            Q = Q * torch.exp(log_G_intrachunk_accum / deg).unsqueeze(-1).to(Q.dtype)
        if qhead_ratio > 1:
            S = S.repeat_interleave(qhead_ratio, dim=2)

        # --- Compute Query State ---
        Q, S, s, attn_Y, l_attn, rowmax = map(lambda x: x.contiguous(), (Q, S, s, attn_Y, l_attn, rowmax))
        Y = _query_state(Q, S, s, attn_Y, l_attn, rowmax, deg, scale, initial_state is None)

        # Epilogue
        out = Y.contiguous().view(b, t, hq, d).to(dtype)
        if return_final_state:
            return out, final_state
        else:
            return out

    _power_full.__doc__ = POWER_FULL_DOC
    return _power_full

