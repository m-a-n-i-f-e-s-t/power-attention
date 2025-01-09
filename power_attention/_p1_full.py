"""
Since there is no expansion, p1 can be implemented mostly using ordiary torch 
matmuls (except for the attention & discumsum parts).
This is a clean, minimal implementation that is easy to understand and debug.
It is useful for parity checking and speed benchmarking against our kernel implementation.
"""

import torch
from torch.utils._pytree import tree_map
import math
from power_attention._discumsum import discumsum, discumsum_reference
from power_attention._attention import attention, attention_reference

def p1_full(Q, K, V, log_G=None, initial_state=None,
               stabilizer=None, ε=1e-5, chunk_size=None,
               deterministic=False,
               use_reference=False,
               normal_space=False):
    if use_reference:
        _attention = attention_reference
        _discumsum = discumsum_reference
    else:
        _attention = attention
        _discumsum = discumsum
    if initial_state is not None:
        raise NotImplementedError('Initial state not implemented')

    # Establish shapes and dtypes
    assert Q.dtype == K.dtype == V.dtype, 'dtypes of inputs must match'
    dtype = Q.dtype
    b, t, hq, d = Q.shape
    _, _,  h, _ = K.shape
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

    # Quick return for simple quadratic attention
    if t <= c:
        if qhead_ratio > 1:
            K = K.repeat_interleave(qhead_ratio, dim=2)
            V = V.repeat_interleave(qhead_ratio, dim=2)
            if gating:
                log_G = log_G.repeat_interleave(qhead_ratio, dim=2)
        log_G_accum = log_G.cumsum(1) if log_G is not None else None
        Y, _, _ = _attention(Q, K, V, log_G_accum, 1, stabilizer, ε, deterministic, False, False, False)
        assert Y.is_contiguous(), 'Y must be contiguous'
        return Y

    # Reshape into chunks
    Q = Q.view(b, n, c, hq, d)
    K = K.view(b, n, c, h, d)
    V = V.view(b, n, c, h, d)    
    if gating:
        log_G = log_G.view(b, n, c, h).to(torch.float32)
        log_G_intrachunk_accum = log_G.cumsum(2)

    # Compute chunk states
    if gating:
        log_discount_weights = (log_G_intrachunk_accum.narrow(2, c-1, 1) - log_G_intrachunk_accum) / 1
        cs_K = K * torch.exp(log_discount_weights).unsqueeze(-1).to(K.dtype)
    else:
        cs_K = K
    S = _update_state(cs_K.contiguous(), V.contiguous())

    # Accumulate
    if gating:
        log_G_chunk_sum = log_G_intrachunk_accum[:,:,-1].contiguous()
    else:
        log_G_chunk_sum = torch.zeros(size=(b, n, h), device=Q.device, dtype=torch.float32)
    S = _discumsum(S, log_G_chunk_sum) # Note that this adds an empty chunk to the start of the sequence

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
    attn_Y, _, rowmax = _attention(Q_flatbatch, K_flatbatch, V_flatbatch, log_G_intrachunk_accum_flatbatch, 1, stabilizer, ε, deterministic, False, False, normal_space)
    if not normal_space: rowmax = torch.exp(rowmax)
    attn_Y = attn_Y.view(b, n, c, hq, d)
    rowmax = rowmax.view(b, n, c, hq, 1).detach()

    if gating:
        if qhead_ratio > 1:
            log_G_intrachunk_accum = log_G_intrachunk_accum.repeat_interleave(qhead_ratio, dim=3)
        Q = Q * torch.exp(log_G_intrachunk_accum / 1).unsqueeze(-1).to(Q.dtype)
    if qhead_ratio > 1:
        S = S.repeat_interleave(qhead_ratio, dim=2)

    qs_Y = (_query_state(Q, S[:,:-1]) * stabilizer / rowmax).to(dtype)
    return (attn_Y + qs_Y).reshape(b, t, hq, d)


def _update_state(K, V):
    return torch.einsum('bnchD,bnchd->bnhDd', K, V)

def _query_state(Q, S):
    return torch.einsum('bnchD,bnhDd->bnchd', Q, S)
