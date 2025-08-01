import torch
import triton
import triton.language as tl
import math
import os
from torch.utils._pytree import tree_map
import torch.nn.functional as F

from power_attention._utils import diff


fwd_configs = [
    triton.Config({'BM': BM, 'BN': BN}, num_stages=s, num_warps=w) \
    for BM in [16, 32, 64, 128]\
    for BN in [32, 64]\
    for s in [1, 2, 3]\
    for w in [4, 8]
]

def keep(conf):
    BM = conf.kwargs["BM"]
    BN = conf.kwargs["BN"]
    if BM * BN < 128 * 128 and conf.num_warps == 8:
        return False
    return True

def prune_fwd_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BM"] % kwargs["r"] == 0 and config.kwargs["BN"] % kwargs["w"] == 0:
            pruned_configs.append(config)
    return pruned_configs

@triton.jit
def _power(a, deg: tl.constexpr):
    if deg == 1:
        return a
    elif deg == 2:
        return a + a
    elif deg == 4:
        a = a + a
        return a + a
    elif deg == 8:
        a = a + a
        a = a + a
        return a + a
    else:
        return deg * a

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                     start_m, range_m, range_n, r: tl.constexpr, w: tl.constexpr, #
                     deg: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, #
                     BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                     M_CTX: tl.constexpr, N_CTX: tl.constexpr, STAGE: tl.constexpr, use_log2: tl.constexpr):
    offset = N_CTX - M_CTX * w // r
    if STAGE == 1: # causal, non-masking part
        lo, hi = 0, ((start_m * BM) * w // r + offset) // BN * BN
    elif STAGE == 2: # causal, masking part
        lo, hi = ((start_m * BM) * w // r + offset) // BN * BN, ((((start_m + 1) * BM) * w // r + offset) + (BN - 1)) // BN * BN
    else: # non-causal
        lo, hi = 0, N_CTX

    p_k = tl.advance(p_k, (0, lo))
    p_v = tl.advance(p_v, (lo, 0))
    if gating:
        p_gk = tl.advance(p_gk, (lo,))

    for start_n in range(lo, hi, BN):
        start_n = tl.multiple_of(start_n, BN)
        # -- compute qk ----
        k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        s = tl.dot(q, k) * scale
        signs = tl.where(s > 0, 1, -1)
        if gating:
            gk = tl.load(p_gk, boundary_check=(0,), padding_option="zero")
        else:
            gk = None
        if use_log2:
            s = _power(tl.log2(s.abs() + 1e-7), deg)
        else:
            s = _power(tl.log(s.abs() + 1e-7), deg)
        if gating:
            s = s + gq[:, None] - gk[None, :]
        if STAGE == 2:
            mask = (range_m[:, None] // r + N_CTX // w - M_CTX // r) >= ((start_n + range_n[None, :]) // w)
            s = s + tl.where(mask, 0., -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        if deg % 2 == 0:
            if use_log2:
                p = tl.exp2(s - m_ij[:, None])
            else:
                p = tl.exp(s - m_ij[:, None])
        else:
            if use_log2:
                p = tl.exp2(s - m_ij[:, None]) * signs
            else:
                p = tl.exp(s - m_ij[:, None]) * signs
        l_ij = tl.sum(p, 1)
        # -- scale acc --
        if use_log2:
            alpha = tl.exp2(m_i - m_ij)
        else:
            alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(p.to(v.dtype), v, acc)
        # -- update m_i
        m_i = m_ij
        p_k = tl.advance(p_k, (0, BN))
        p_v = tl.advance(p_v, (BN, 0))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return acc, l_i, m_i


@triton.autotune(list(filter(keep, fwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "deg", "norm"], prune_configs_by={'early_config_prune': prune_fwd_configs})
@triton.jit
def _attn_fwd(Q, K, V, LOG_GQ, LOG_GK, L, M, Out,  #
              stride_qb, stride_qh, stride_qm, stride_qd,  #
              stride_kb, stride_kh, stride_kn, stride_kd,  #
              stride_vb, stride_vh, stride_vn, stride_ve,  #
              stride_mb, stride_mh, stride_mm, #
              stride_gqb, stride_gqh, stride_gqd,  #
              stride_gkb, stride_gkh, stride_gkd,  #
              stride_ob, stride_oh, stride_om, stride_oe,  #
              stride_lb, stride_lh, stride_lm, #
              H, M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
              deg: tl.constexpr, scale: tl.constexpr, gating: tl.constexpr,  #
              DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, STAGE: tl.constexpr,  #
              norm: tl.constexpr, use_log2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    q_offset = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_offset = off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    v_offset = off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    gq_offset = off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
    gk_offset = off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    p_q = tl.make_block_ptr(Q+q_offset, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m*BM, 0), (BM, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V+v_offset, (N_CTX, DIM_VO), (stride_vn, stride_ve), (0, 0), (BN, DIM_VO), (1, 0))
    p_k = tl.make_block_ptr(K+k_offset, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, 0), (DIM_QK, BN), (0, 1))
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ+gq_offset, (M_CTX,), (stride_gqd,), (start_m*BM,), (BM,), (0,))
    else:
        p_gq = None
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK+gk_offset, (N_CTX,), (stride_gkd,), (0,), (BN,), (0,))
    else:
        p_gk = None

    range_m = start_m * BM + tl.arange(0, BM)
    range_n = tl.arange(0, BN)

    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BM], dtype=tl.float32) + 1.0
    acc = tl.zeros([BM, DIM_VO], dtype=tl.float32)

    q = tl.load(p_q, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    if gating:
        gq = tl.load(p_gq, cache_modifier=".cg", boundary_check=(0,), padding_option="zero")
    else:
        gq = None

    if STAGE & 1: # non-masking part
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, scale, gating, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 4 - STAGE, use_log2)

    if STAGE & 2: # masking part
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, scale, gating, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 2, use_log2)
        
    # epilogue
    l_i = l_i + 1e-6
    if norm: # normalize by temporal norm and fuse norm into rowmax
        if use_log2:
            m_i += tl.log2(l_i)
        else:
            m_i += tl.log(l_i)
        acc = acc / l_i[:, None]
    o_offset = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
    m_offset = off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    l_offset = off_b.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
    p_o = tl.make_block_ptr(Out+o_offset, (M_CTX, DIM_VO), (stride_om, stride_oe), (start_m*BM, 0), (BM, DIM_VO), (1, 0))
    p_l = L + l_offset + range_m * stride_lm
    p_m = M + m_offset + range_m * stride_mm
    tl.store(p_m, m_i, mask=range_m < M_CTX)
    if not norm: # store temporal norm
        tl.store(p_l, l_i, mask=range_m < M_CTX)
    tl.store(p_o, acc.to(Out.type.element_ty), boundary_check=(0, 1))


bwd_configs_dkdv = [
    triton.Config({'BN1': BN1, 'BM1': BM1, 'BLK_SLICE_FACTOR': BLK_SLICE_FACTOR}, num_stages=s, num_warps=w) \
    for BN1 in [64, 128]\
    for BM1 in [16, 32]\
    for s in [1, 3]\
    for w in [4, 8]\
    for BLK_SLICE_FACTOR in [1, 2]\
]

def prune_bwd_dkdv_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BM1"] % kwargs["r"] == 0 and config.kwargs["BN1"] % kwargs["w"] == 0:
            pruned_configs.append(config)
    return pruned_configs

bwd_configs_dq = [
    triton.Config({'BN2': BN2, 'BM2': BM2, 'BLK_SLICE_FACTOR': BLK_SLICE_FACTOR}, num_stages=s, num_warps=w) \
    for BM2 in [64, 128]\
    for BN2 in [16, 32]\
    for s in [1, 3]\
    for w in [4, 8]\
    for BLK_SLICE_FACTOR in [1]\
]

def prune_bwd_dq_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BM2"] % kwargs["r"] == 0 and config.kwargs["BN2"] % kwargs["w"] == 0:
            pruned_configs.append(config)
    return pruned_configs

def keep_bwd(conf):
    FACTOR = conf.kwargs["BLK_SLICE_FACTOR"]
    if 'BM1' in conf.kwargs:
        return conf.kwargs["BM1"] // FACTOR >= 16
    elif 'BN2' in conf.kwargs:
        return conf.kwargs["BN2"] // FACTOR >= 16
    raise ValueError(f"Invalid config: {conf.kwargs}")

preprocess_configs = [
    triton.Config({'BM': BM})
    for BM in [64, 128, 256]
]

@triton.autotune(preprocess_configs, key=["M_CTX", "HEAD_DIM"])
@triton.jit
def _attn_bwd_preprocess(O, DO, Delta,  #
                         stride_ob, stride_oh, stride_om, stride_oe, #
                         stride_dob, stride_doh, stride_dom, stride_doe, #
                         stride_db, stride_dh, stride_dm, #
                         BM: tl.constexpr, HEAD_DIM: tl.constexpr, M_CTX: tl.constexpr  #
                         ):
    range_m = tl.program_id(0) * BM + tl.arange(0, BM)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    mask_m = range_m < M_CTX
    # load
    o = tl.load(O + off_b * stride_ob + off_h * stride_oh + range_m[:, None] * stride_om + off_n[None, :] * stride_oe, cache_modifier=".cg", mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + range_m[:, None] * stride_dom + off_n[None, :] * stride_doe, cache_modifier=".cg", mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + range_m * stride_dm, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                    Q, LOG_GQ, DO, M, Delta, DL, #
                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                    M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
                    deg: tl.constexpr, log_scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                    start_n, start_m, num_steps: tl.constexpr, #
                    MASK: tl.constexpr, norm: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_n = start_n + tl.arange(0, BN)

    p_qT = tl.make_block_ptr(Q, (DIM_QK, M_CTX), (stride_qd, stride_qm), (0, start_m), (DIM_QK, BM), (0, 1))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM, DIM_VO), (1, 0))
    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM,), (0,))
    else:
        p_gq = None

    curr_m = start_m
    for _ in range(num_steps):
        range_m = curr_m + tl.arange(0, BM)
        # --- re-compute p ---
        qT = tl.load(p_qT, boundary_check=(0, 1), padding_option="zero")
        if gating:
            gq = tl.load(p_gq, boundary_check=(0,), padding_option="zero")
        else:
            gq = None
        sT = tl.dot(k, qT) # (N, M)
        s_signs = tl.where(sT > 0, 1, -1)
        if use_log2:
            zT = _power(tl.log2(sT.abs() + 1e-7) + log_scale, deg)
        else:
            zT = _power(tl.log(sT.abs() + 1e-7) + log_scale, deg)
        if gating:
            zT = zT + gq[None, :] - gk[:, None]
        p_m = M + range_m * stride_mm
        m = tl.load(p_m, mask=range_m < M_CTX, other=float("inf"))
        if MASK:
            mask = (range_m[None, :] // r + N_CTX // w - M_CTX // r) >= (range_n[:, None] // w)
            zT = tl.where(mask, zT, -float("inf"))
        if use_log2:
            pT = tl.exp2(zT - m[None, :])
        else:
            pT = tl.exp(zT - m[None, :])
        if deg % 2 == 1:
            pT = pT * s_signs

        # --- compute dv ---
        do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        dv = tl.dot(pT.to(Q.type.element_ty), do, dv)

        # --- compute dp ---
        if norm:
            dl_or_delta = - tl.load(Delta + range_m * stride_dm, mask=range_m < M_CTX, other=0.0)
        else:
            dl_or_delta = tl.load(DL + range_m * stride_dlm, mask=range_m < M_CTX, other=0.0)
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32) # (BN, BM)
        dsT = pT * (dpT + dl_or_delta[None, :])
        if gating:
            dgk += -tl.sum(dsT, 1, keep_dims=False)
        signs = tl.where(sT > 0, 1, -1) # recreate signs to allow compiler to free s_signs, hopefully
        if use_log2:
            dsT = _power(dsT * signs * 1.44269504 / (tl.abs(sT) + 1e-7), deg)
        else:
            dsT = _power(dsT * signs / (tl.abs(sT) + 1e-7), deg)
        dk = tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT), dk)
        # increment pointers
        curr_m += BM
        p_qT = tl.advance(p_qT, (0, BM))
        p_do = tl.advance(p_do, (BM, 0))
        if gating:
            p_gq = tl.advance(p_gq, (BM,))

    return dk, dv, dgk


@triton.jit
def _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                  K, V, LOG_GK, #
                  stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                  M_CTX, N_CTX, r, w, #
                  deg: tl.constexpr, log_scale: tl.constexpr, gating: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                  start_m, start_n, num_steps: tl.constexpr, #
                  MASK: tl.constexpr, use_log2: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_m = start_m + tl.arange(0, BM)

    p_kT = tl.make_block_ptr(K, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, start_n), (DIM_QK, BN), (0, 1))
    p_vT = tl.make_block_ptr(V, (DIM_VO, N_CTX), (stride_ve, stride_vn), (0, start_n), (DIM_VO, BN), (0, 1))
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN,), (0,))
    else:
        p_gk = None

    curr_n = start_n
    for _ in range(num_steps):
        range_n = curr_n + tl.arange(0, BN)
        # --- re-compute p ---
        kT = tl.load(p_kT, boundary_check=(0, 1), padding_option="zero")
        vT = tl.load(p_vT, boundary_check=(0, 1), padding_option="zero")
        if gating:
            gk = tl.load(p_gk, boundary_check=(0,), padding_option="zero")
        else:
            gk = None
        s = tl.dot(q, kT) # (M, N)
        s_signs = tl.where(s > 0, 1, -1)
        if use_log2:
            z = _power(tl.log2(s.abs() + 1e-7) + log_scale, deg)
        else:
            z = _power(tl.log(s.abs() + 1e-7) + log_scale, deg)
        if gating:
            z = z + gq[:, None] - gk[None, :]
        if MASK:
            mask = (range_m[:, None] // r + N_CTX // w - M_CTX // r) >= (range_n[None, :] // w)
            z = tl.where(mask, z, -float("inf"))
        
        if use_log2:
            p = tl.exp2(z - m[:, None])
        else:
            p = tl.exp(z - m[:, None])
        if deg % 2 == 1:
            p = p * s_signs

        # --- compute dQ ---
        dp = tl.dot(do, vT, out_dtype=tl.float32)
        ds = p * (dp + dl_or_delta[:, None])
        if gating:
            dgq += tl.sum(ds, 1, keep_dims=False)
        signs = tl.where(s > 0, 1, -1) # recreate signs to allow compiler to free s_signs, hopefully
        if use_log2:
            ds = _power(ds * signs * 1.44269504 / (tl.abs(s) + 1e-7), deg)
        else:
            ds = _power(ds * signs / (tl.abs(s) + 1e-7), deg)
        dq = tl.dot(ds.to(kT.type.element_ty), tl.trans(kT), dq)
        # increment pointers
        curr_n += BN
        p_kT = tl.advance(p_kT, (0, BN))
        p_vT = tl.advance(p_vT, (0, BN))
        if gating:
            p_gk = tl.advance(p_gk, (BN,))

    return dq, dgq


@triton.autotune(list(filter(keep_bwd, bwd_configs_dkdv)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "deg"], prune_configs_by={'early_config_prune': prune_bwd_dkdv_configs})
@triton.jit
def attn_bwd_dkdv(Q, K, V, LOG_GQ, LOG_GK, M, Delta, DO, DL, DQ, DK, DV, DLOG_GQ, DLOG_GK, #
              stride_qb, stride_qh, stride_qm, stride_qd, #
              stride_kb, stride_kh, stride_kn, stride_kd, #
              stride_vb, stride_vh, stride_vn, stride_ve, #
              stride_mb, stride_mh, stride_mm, #
              stride_db, stride_dh, stride_dm, #
              stride_dob, stride_doh, stride_dom, stride_doe, #
              stride_dlb, stride_dlh, stride_dlm, #
              stride_dqb, stride_dqh, stride_dqm, stride_dqd, #
              stride_dkb, stride_dkh, stride_dkn, stride_dkd, #
              stride_dvb, stride_dvh, stride_dvn, stride_dve, #
              stride_gqb, stride_gqh, stride_gqm, #
              stride_gkb, stride_gkh, stride_gkn, #
              H, M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
              deg: tl.constexpr,  #
              scale: tl.constexpr,  #
              gating: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM1: tl.constexpr,  #
              BN1: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              norm: tl.constexpr,  #
              use_log2: tl.constexpr,  #
              ):
    if STAGE == 3:
        tl.static_assert((BM1 // BLK_SLICE_FACTOR) % r == 0, "Sliced BM1 must be divisible by w")
    else:
        tl.static_assert(BM1 % r == 0, "BM1 must be divisible by r")
    tl.static_assert(BN1 % w == 0, "BN1 must be divisible by w")

    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    start_n = tl.program_id(0)*BN1
    log_scale = tl.log2(scale) if use_log2 else tl.log(scale)


    Q += off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    M += off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    Delta += off_b.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
    DO += off_b.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
    DL += off_b.to(tl.int64) * stride_dlb + off_h.to(tl.int64) * stride_dlh
    DQ += off_b.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
    DK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
    DV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
    if gating:
        LOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        LOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh
        DLOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        DLOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    # -- First part: compute dk, dv
    MASK_BLOCK_M1: tl.constexpr = BM1 // BLK_SLICE_FACTOR
    range_n = start_n + tl.arange(0, BN1)

    dv = tl.zeros([BN1, DIM_VO], dtype=tl.float32)
    dk = tl.zeros([BN1, DIM_QK], dtype=tl.float32)
    dgk = tl.zeros([BN1,], dtype=tl.float32)

    # load k, v, gk
    p_k = tl.make_block_ptr(K, (N_CTX, DIM_QK), (stride_kn, stride_kd), (start_n, 0), (BN1, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V, (N_CTX, DIM_VO), (stride_vn, stride_ve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    k = tl.load(p_k, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    v = tl.load(p_v, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    if gating:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN1,), (0,))
        gk = tl.load(p_gk, cache_modifier=".cg", boundary_check=(0,), padding_option="zero")
    else:
        gk = None

    offset = (N_CTX - M_CTX * w // r)
    start_m = (max(0, start_n - offset)) * r // w // MASK_BLOCK_M1 * MASK_BLOCK_M1 if STAGE == 3 else 0
    if STAGE & 2: # masked blocks
        end_m = tl.cdiv((max(0, start_n - offset) + BN1) * r // w, MASK_BLOCK_M1) * MASK_BLOCK_M1
        num_steps = (end_m - start_m) // MASK_BLOCK_M1
        dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                    Q, LOG_GQ, DO, M, Delta, DL, #
                                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                                    M_CTX, N_CTX, r, w, #
                                    deg, log_scale, gating, MASK_BLOCK_M1, BN1, DIM_QK, DIM_VO, #
                                    start_n, start_m, num_steps, #
                                    MASK=True, norm=norm, use_log2=use_log2)
        start_m += num_steps * MASK_BLOCK_M1
        
    # unmasked blocks
    num_steps = (M_CTX - start_m) // BM1
    dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                Q, LOG_GQ, DO, M, Delta, DL, #
                                stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, stride_mm, stride_dm, stride_dlm, #
                                M_CTX, N_CTX, r, w, #
                                deg, log_scale, gating, BM1, BN1, DIM_QK, DIM_VO, #
                                start_n, start_m, num_steps, #
                                MASK=False, norm=norm, use_log2=use_log2)

    p_dv = tl.make_block_ptr(DV, (N_CTX, DIM_VO), (stride_dvn, stride_dve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    p_dk = tl.make_block_ptr(DK, (N_CTX, DIM_QK), (stride_dkn, stride_dkd), (start_n, 0), (BN1, DIM_QK), (1, 0))

    mask_dkv = range_n < N_CTX
    tl.store(p_dv, dv.to(DV.type.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, dk.to(DK.type.element_ty), boundary_check=(0, 1))
    if gating:
        p_dgk = DLOG_GK + range_n * stride_gkn
        tl.store(p_dgk, dgk, mask=mask_dkv)


@triton.autotune(list(filter(keep_bwd, bwd_configs_dq)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w", "gating", "deg"], prune_configs_by={'early_config_prune': prune_bwd_dq_configs})
@triton.jit
def attn_bwd_dq(Q, K, V, LOG_GQ, LOG_GK, M, Delta, DO, DL, DQ, DK, DV, DLOG_GQ, DLOG_GK, #
              stride_qb, stride_qh, stride_qm, stride_qd, #
              stride_kb, stride_kh, stride_kn, stride_kd, #
              stride_vb, stride_vh, stride_vn, stride_ve, #
              stride_mb, stride_mh, stride_mm, #
              stride_db, stride_dh, stride_dm, #
              stride_dob, stride_doh, stride_dom, stride_doe, #
              stride_dlb, stride_dlh, stride_dlm, #
              stride_dqb, stride_dqh, stride_dqm, stride_dqd, #
              stride_dkb, stride_dkh, stride_dkn, stride_dkd, #
              stride_dvb, stride_dvh, stride_dvn, stride_dve, #
              stride_gqb, stride_gqh, stride_gqm, #
              stride_gkb, stride_gkh, stride_gkn, #
              H, M_CTX, N_CTX, r: tl.constexpr, w: tl.constexpr, #
              deg: tl.constexpr,  #
              scale: tl.constexpr,  #
              gating: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM2: tl.constexpr,  #
              BN2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              norm: tl.constexpr,  #
              use_log2: tl.constexpr,  #
              ):
    if STAGE == 3:
        tl.static_assert((BN2 // BLK_SLICE_FACTOR) % w == 0, "Sliced BN2 must be divisible by w")
    else:
        tl.static_assert(BN2 % w == 0, "BN2 must be divisible by w")
    tl.static_assert(BM2 % r == 0, "BM2 must be divisible by r")

    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    log_scale = tl.log2(scale) if use_log2 else tl.log(scale)

    Q += off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    M += off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    Delta += off_b.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
    DO += off_b.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
    DL += off_b.to(tl.int64) * stride_dlb + off_h.to(tl.int64) * stride_dlh
    DQ += off_b.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
    DK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
    DV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
    if gating:
        LOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        LOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh
        DLOG_GQ += off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
        DLOG_GK += off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    # -- Second part: compute dq
    start_m = tl.program_id(0) * BM2

    MASK_BLOCK_N2: tl.constexpr = BN2 // BLK_SLICE_FACTOR

    # load q, gq
    p_q = tl.make_block_ptr(Q, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM2, DIM_VO), (1, 0))

    if gating:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
    else:
        p_gq = None
    q = tl.load(p_q, cache_modifier=".cg", boundary_check=(0, 1), padding_option="zero")
    if gating:
        gq = tl.load(p_gq, cache_modifier=".cg", boundary_check=(0,), padding_option="zero")
    else:
        gq = None
    range_m = start_m + tl.arange(0, BM2)
    m = tl.load(M + range_m * stride_mm, mask=range_m < M_CTX, other=float("inf"))
    if norm:
        dl_or_delta = - tl.load(Delta + range_m * stride_dm, cache_modifier=".cg", mask=range_m < M_CTX, other=0.0)
    else:
        dl_or_delta = tl.load(DL + range_m * stride_dlm, cache_modifier=".cg", mask=range_m < M_CTX, other=0.0)
    do = tl.load(p_do, cache_modifier=".cg", boundary_check=(0,1), padding_option="zero")

    dq = tl.zeros([BM2, DIM_QK], dtype=tl.float32)
    dgq = tl.zeros([BM2,], dtype=tl.float32)

    offset = N_CTX - M_CTX * w // r
    end_n = tl.cdiv((start_m + BM2) * w // r + offset, MASK_BLOCK_N2) * MASK_BLOCK_N2 if STAGE & 2 else N_CTX
    if STAGE & 2: # masked blocks
        start_n = (start_m * w // r + offset) // MASK_BLOCK_N2 * MASK_BLOCK_N2
        num_steps = (end_n - start_n) // MASK_BLOCK_N2
        dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                               K, V, LOG_GK, #
                               stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                               M_CTX, N_CTX, r, w, #
                               deg, log_scale, gating, BM2, MASK_BLOCK_N2, DIM_QK, DIM_VO, #
                               start_m, start_n, num_steps, #
                               MASK=True, use_log2=use_log2)
        end_n -= num_steps * MASK_BLOCK_N2
    
    # unmasked blocks
    num_steps = end_n // BN2
    dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, dl_or_delta, #
                           K, V, LOG_GK, #
                           stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                           M_CTX, N_CTX, r, w, #
                           deg, log_scale, gating, BM2, BN2, DIM_QK, DIM_VO, #
                           start_m, end_n - num_steps * BN2, num_steps, #
                           MASK=False, use_log2=use_log2)

    # store dq, dgq
    p_dq = tl.make_block_ptr(DQ, (M_CTX, DIM_QK), (stride_dqm, stride_dqd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    tl.store(p_dq, dq.to(DQ.type.element_ty), boundary_check=(0, 1))
    if gating:
        p_dgq = tl.make_block_ptr(DLOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
        tl.store(p_dgq, dgq, boundary_check=(0,))


class _power_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, log_G, deg, r, w, causal, head_first, scale, norm, use_log2):
        """ 
            Attention formulation of power attention.
            
            When norm is True, the output is normalized by temporal norm (l), and the returned
            temporal norm is uninitialized.
            When norm is False, the output is not normalized, and l is the temporal norm (sum of exponentials of the powered attention scores)


            Args:
            Q: (B, H_Q, CTX, D)
            K: (B, H_K, CTX, D)
            V: (B, H_K, CTX, E)
            deg: int
            log_G: (B, H_Q // R, CTX) or (B, CTX, H_Q // R)
            r: int, number of heads in q to form a group
            w: int, number of heads in k to form a group
            causal: bool
            head_first: bool
            norm: bool
            use_log2: bool

            Returns:
                o: (B, H_Q // R, CTX, E) if head_first else (B, CTX, H_Q // R, E)
                l: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
                rowmax: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
        """
        if head_first:
            b, hq, tq, d, hk, tk, e = *Q.shape, K.shape[1], K.shape[2], V.shape[-1]
        else:
            b, tq, hq, d, tk, hk, e = *Q.shape, K.shape[1], K.shape[2], V.shape[-1]
        assert r & (r - 1) == 0, "r must be a power of 2"
        assert w == 1, "w must be 1"
        assert hq % r == 0, "hq must be divisible by r"
        assert hk % w == 0, "hk must be divisible by w"
        assert hq // r == hk // w, f"hq // r must be equal to hk // w, {hq=} {r=} {hk=} {w=}"
        assert isinstance(deg, int) and deg > 0, "deg must be a positive integer"
        assert d in {16, 32, 64, 128, 256}, "d must be 16, 32, 64, 128, or 256"
        assert e in {16, 32, 64, 128, 256}, "e must be 16, 32, 64, 128, or 256"
        assert w == 1, "w>1 not well supported yet"

        h = hq // r
        gating = log_G is not None
        if use_log2:
            log_G = log_G * math.log2(math.e)
        if head_first:
            o = torch.empty((b, h, tq * r, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, hk, tk)
                log_GQ = log_G.narrow(2, -tq, tq).repeat_interleave(r, dim=2)
                log_GK = log_G.repeat_interleave(w, dim=2)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(1), log_GQ.stride(2))
                gk_strides = (log_GK.stride(0), log_GK.stride(1), log_GK.stride(2))
            else:
                log_GQ = None
                log_GK = None
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, h, tq * r, d)
            K = K.view(b, h, tk * w, d)
            V = V.view(b, h, tk * w, e)
            rowmax = torch.empty((b, h, tq * r), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, h, tq * r), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3))
            k_strides = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
            v_strides = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
            l_strides = (l.stride(0), l.stride(1), l.stride(2))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(1), rowmax.stride(2))
            o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        else:
            o = torch.empty((b, tq * r, h, e), device=Q.device, dtype=Q.dtype)
            if gating:
                assert log_G.shape == (b, tk, h)
                log_GQ = log_G.narrow(1, -tq, tq).repeat_interleave(r, dim=1)
                log_GK = log_G.repeat_interleave(w, dim=1)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(2), log_GQ.stride(1))
                gk_strides = (log_GK.stride(0), log_GK.stride(2), log_GK.stride(1))
            else:
                log_GQ = None
                log_GK = None
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, tq * r, h, d)
            K = K.view(b, tk * w, h, d)
            V = V.view(b, tk * w, h, e)
            rowmax = torch.empty((b, tq * r, h), device=Q.device, dtype=torch.float32)
            l = torch.empty((b, tq * r, h), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(2), Q.stride(1), Q.stride(3))
            k_strides = (K.stride(0), K.stride(2), K.stride(1), K.stride(3))
            v_strides = (V.stride(0), V.stride(2), V.stride(1), V.stride(3))
            l_strides = (l.stride(0), l.stride(2), l.stride(1))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(2), rowmax.stride(1))
            o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(r*tq, args["BM"]), b * h)
        _attn_fwd[grid](
            Q, K, V, log_GQ, log_GK, l, rowmax, o, *q_strides, *k_strides, *v_strides, *rowmax_strides, *gq_strides, *gk_strides, *o_strides, *l_strides,
            H=h, M_CTX=tq*r, N_CTX=tk*w, r=r, w=w, deg=deg, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        o = o.view(b, hq, tq, e) if head_first else o.view(b, tq, hq, e)
        l = l.view(b, hq, tq) if head_first else l.view(b, tq, hq)
        rowmax = rowmax.view(b, hq, tq) if head_first else rowmax.view(b, tq, hq)
        ctx.save_for_backward(Q, K, V, l, rowmax, o, log_GQ, log_GK)
        ctx.b = b
        ctx.h = h
        ctx.hq = hq
        ctx.hk = hk
        ctx.tq = tq
        ctx.tk = tk
        ctx.t = tk
        ctx.r = r
        ctx.w = w
        ctx.grid = grid
        ctx.d = d
        ctx.e = e
        ctx.deg = deg
        ctx.gating = gating
        ctx.scale = scale
        ctx.norm = norm
        ctx.q_strides = q_strides
        ctx.k_strides = k_strides
        ctx.v_strides = v_strides
        ctx.rowmax_strides = rowmax_strides
        ctx.gq_strides = gq_strides
        ctx.gk_strides = gk_strides
        ctx.o_strides = o_strides
        ctx.head_first = head_first
        ctx.stage = stage
        ctx.use_log2 = use_log2
        return o, l, rowmax

    @staticmethod
    def backward(ctx, do, dl, drowmax):
        Q, K, V, l, rowmax, o, log_GQ, log_GK = ctx.saved_tensors
        if log_GQ is not None:
            log_GQ = log_GQ.contiguous() # needed for reuse log_GQ's strides for dlog_GQ
            log_GK = log_GK.contiguous() # needed for reuse log_GK's strides for dlog_GK
        do = do.contiguous()
        b, h, hq, hk, tq, tk, stage, norm, gating, use_log2 = ctx.b, ctx.h, ctx.hq, ctx.hk, ctx.tq, ctx.tk, ctx.stage, ctx.norm, ctx.gating, ctx.use_log2
        q_strides, k_strides, v_strides, rowmax_strides, gq_strides, gk_strides = ctx.q_strides, ctx.k_strides, ctx.v_strides, ctx.rowmax_strides, ctx.gq_strides, ctx.gk_strides
        r, w, d, e, deg, scale = ctx.r, ctx.w, ctx.d, ctx.e, ctx.deg, ctx.scale

        dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)
        dlog_GQ = torch.empty_like(log_GQ) if gating else None
        dlog_GK = torch.empty_like(log_GK) if gating else None
        delta = torch.empty_like(rowmax) if norm else torch.empty((0, 0, 0))

        if ctx.head_first:
            do = do.reshape(b, h, tq*r, e)
            o = o.reshape(b, h, tq*r, e)
            dl = dl.reshape(b, h, tq*r)
            delta = delta.reshape(b, h, tq*r) if norm else delta
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2), x.stride(3)), (do, dQ, dK, dV, o))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(1), x.stride(2)), (dl, delta))
        else:
            do = do.reshape(b, tq*r, h, e)
            o = o.reshape(b, tq*r, h, e)
            dl = dl.reshape(b, tq*r, h)
            delta = delta.reshape(b, tq*r, h) if norm else delta
            do_strides, dQ_strides, dK_strides, dV_strides, o_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1), x.stride(3)), (do, dQ, dK, dV, o))
            dl_strides, delta_strides = map(lambda x: (x.stride(0), x.stride(2), x.stride(1)), (dl, delta))

        if norm:
            _attn_bwd_preprocess[lambda args: (triton.cdiv(tq*r, args["BM"]), b, h)](
                o, do, delta,
                *o_strides, *do_strides, *delta_strides,
                HEAD_DIM=e, M_CTX=tq*r
            )

        attn_bwd_dkdv[lambda args: (triton.cdiv(w*tk, args["BN1"]), b * h)](
            Q, K, V, log_GQ, log_GK, rowmax, delta, do, dl, dQ, dK, dV, dlog_GQ, dlog_GK,
            *q_strides, *k_strides, *v_strides,
            *rowmax_strides, *delta_strides, *do_strides, *dl_strides, 
            *dQ_strides, *dK_strides, *dV_strides,
            *gq_strides, *gk_strides,
            H=h, M_CTX=tq*r, N_CTX=tk*w, r=r, w=w, deg=deg, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        attn_bwd_dq[lambda args: (triton.cdiv(r*tq, args["BM2"]), b * h)](
            Q, K, V, log_GQ, log_GK, rowmax, delta, do, dl, dQ, dK, dV, dlog_GQ, dlog_GK,
            *q_strides, *k_strides, *v_strides,
            *rowmax_strides, *delta_strides, *do_strides, *dl_strides, 
            *dQ_strides, *dK_strides, *dV_strides,
            *gq_strides, *gk_strides,
            H=h, M_CTX=tq*r, N_CTX=tk*w, r=r, w=w, deg=deg, scale=scale, gating=gating, DIM_QK=d, DIM_VO=e, STAGE=stage, norm=norm, use_log2=use_log2)
        if gating:
            # TODO(sean): when w>1, the following is incorrect, but we also need to change the way gating works 
            assert hk == h, "assuming hk == h"
            if ctx.head_first:
                dlog_G = dlog_GK.view(b, h, tk, w).transpose(2, 3).view(b, hk, tk).contiguous()
                dlog_G[:, :, -tq:] += dlog_GQ.view(b, h, tq, r).sum(dim=-1)
            else:
                dlog_G = dlog_GK.view(b, tk, w, h).view(b, tk, hk)
                dlog_G[:, -tq:, :] += dlog_GQ.view(b, tq, r, h).sum(dim=-2)
        else:
            dlog_G = None
        dQ = dQ.view(b, hq, tq, d) if ctx.head_first else dQ.view(b, tq, hq, d)
        dK = dK.view(b, hk, tk, d) if ctx.head_first else dK.view(b, tk, hk, d)
        dV = dV.view(b, hk, tk, e) if ctx.head_first else dV.view(b, tk, hk, e)
        return dQ, dK, dV, dlog_G, None, None, None, None, None, None, None, None

def _attention_fn(Q, K, V, log_G, deg, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):
    r = Q.shape[2] // K.shape[2]
    w = 1
    Y, l, rowmax = _power_attention.apply(Q, K, V, log_G, deg, r, w, causal, head_first, scale, norm, use_log2) # type: ignore
    rowmax = rowmax.detach()
    if norm:
        return Y
    return Y, l, rowmax

attention = torch.compiler.disable(_attention_fn)

if __name__ == "__main__":
    from power_attention._attention.create_inputs import create_inputs
    from perf._timing import benchmark_speed

    kw = dict(b=1, h=6, d=64, dtype=torch.bfloat16, device='cuda', scale=1.0, deg=2, seed=42, std=1/8.0, gating=True, norm=False)
    benchmark_speed('fwd', attention, create_inputs, kw, compile=False)