import torch
import triton
import triton.language as tl
from torch.utils._pytree import tree_map
import torch.nn.functional as F

def attention_reference(Q, K, V, deg, log_G, r=1, w=1, causal=True, head_first=False, norm=False):
    if head_first:
        b, hq, ctx, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
    else:
        b, ctx, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]
    assert hq % r == 0, "hq must be divisible by r"
    assert hk % w == 0, "hk must be divisible by w"
    assert hq // r == hk // w, "hq // r must be equal to hk // w"
    assert isinstance(deg, int) and deg > 0, "deg must be a positive integer"
    h = hq // r
    if log_G is not None:
        if head_first:
            assert log_G.shape == (b, h, ctx)
        else:
            assert log_G.shape == (b, ctx, h)
            log_G = log_G.transpose(1, 2) # (b, h, ctx)
    if head_first:
        Q = Q.view(b, h, ctx * r, d)
        K = K.view(b, h, ctx * w, d)
        V = V.view(b, h, ctx * w, e)
    else:
        Q = Q.view(b, ctx * r, h, d).transpose(1, 2)
        K = K.view(b, ctx * w, h, d).transpose(1, 2)
        V = V.view(b, ctx * w, h, e).transpose(1, 2)
    
    _qidx = torch.arange(ctx*r, device=Q.device).unsqueeze(1)
    _kidx = torch.arange(ctx*w, device=K.device).unsqueeze(0)
    m = (_qidx // r) >= (_kidx // w)
    s = torch.matmul(Q, K.transpose(2,3))
    s = float(deg) * torch.where(m, torch.log(s.abs() + 1e-7), -float("inf"))
    if log_G is not None:
        s = s + (log_G.repeat_interleave(r, dim=2)[..., :, None] - log_G.repeat_interleave(w, dim=2)[..., None, :])
    rowmax = torch.max(s, dim=-1, keepdim=True).values.detach()
    p = torch.exp(s - rowmax).to(V.dtype)
    o = torch.matmul(p, V)
    if norm:
        o = o - (o.sum(dim=-1, keepdim=True) / d)
        o = o / torch.sqrt(o.sum(dim=-1, keepdim=True)**2 + 1e-7)
    if not head_first:
        o = o.transpose(1, 2)
        rowmax = rowmax.transpose(1, 2)
    return o, rowmax.squeeze(-1)


fwd_configs = [
    triton.Config({'BM': BM, 'BN': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128, 256]\
    for BN in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
]

def keep(conf):
    BM = conf.kwargs["BM"]
    BN = conf.kwargs["BN"]
    if BM * BN < 128 * 128 and conf.num_warps == 8:
        return False
    return True

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        M_CTX = nargs["M_CTX"] if "M_CTX" in nargs else kwargs["M_CTX"]
        N_CTX = nargs["N_CTX"] if "N_CTX" in nargs else kwargs["N_CTX"]
        if config.kwargs["BM"] <= M_CTX and config.kwargs["BN"] <= N_CTX:
            pruned_configs.append(config)
    return pruned_configs

@triton.jit
def _attn_fwd_inner(acc, m_i, q, gq, p_k, p_gk, p_v, #
                     start_m, range_m, range_n, r, w, #
                     deg, BM, BN, DIM_QK, DIM_VO, #
                     M_CTX, N_CTX, STAGE):
    if STAGE == 1: # causal, non-masking part
        lo, hi = 0, start_m * BM
    elif STAGE == 2: # causal, masking part
        lo, hi = start_m * BM, (start_m + 1) * BM
        lo = tl.multiple_of(lo, BM)
        hi = tl.multiple_of(hi, BM)
    else: # non-causal
        lo, hi = 0, N_CTX

    p_k = tl.advance(p_k, (0, lo))
    p_v = tl.advance(p_v, (lo, 0))
    if p_gk is not None:
        p_gk = tl.advance(p_gk, (lo,))

    for start_n in range(lo, hi, BN):
        start_n = tl.multiple_of(start_n, BN)
        # -- compute qk ----
        k = tl.load(p_k)
        s = tl.dot(q, k)
        signs = tl.where(s > 0, 1, -1)
        if p_gk is not None:
            gk = tl.load(p_gk)
        else:
            gk = None
        s = deg * tl.log(s.abs() + 1e-7)
        if gq is not None and gk is not None:
            s = s + gq[:, None] - gk[None, :]
        if STAGE == 2:
            mask = (range_m[:, None] // r) >= ((start_n + range_n[None, :]) // w)
            s = s + tl.where(mask, 0., -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(s, 1))
        # -- scale acc --
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        v = tl.load(p_v)
        if deg % 2 == 0:
            p = tl.exp(s - m_ij[:, None])
        else:
            p = tl.exp(s - m_ij[:, None]) * signs
        acc = tl.dot(p.to(v.dtype), v, acc)
        # -- update m_i
        m_i = m_ij
        p_k = tl.advance(p_k, (0, BN))
        p_v = tl.advance(p_v, (BN, 0))
        if p_gk is not None:
            p_gk = tl.advance(p_gk, (BN,))

    return acc, m_i


@triton.autotune(list(filter(keep, fwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
def _attn_fwd(Q, K, V, LOG_GQ, LOG_GK, M, Out,  #
              stride_qb, stride_qh, stride_qm, stride_qd,  #
              stride_kb, stride_kh, stride_kn, stride_kd,  #
              stride_vb, stride_vh, stride_vn, stride_ve,  #
              stride_mb, stride_mh, stride_mm, #
              stride_gqb, stride_gqh, stride_gqd,  #
              stride_gkb, stride_gkh, stride_gkd,  #
              stride_ob, stride_oh, stride_om, stride_oe,  #
              H, M_CTX, N_CTX, r, w, #
              deg: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM: tl.constexpr,  #
              BN: tl.constexpr,  #
              NORM: tl.constexpr,  #
              ):
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
    if LOG_GQ is not None:
        p_gq = tl.make_block_ptr(LOG_GQ+gq_offset, (M_CTX,), (stride_gqd,), (start_m*BM,), (BM,), (0,))
    else:
        p_gq = None
    if LOG_GK is not None:
        p_gk = tl.make_block_ptr(LOG_GK+gk_offset, (N_CTX,), (stride_gkd,), (0,), (BN,), (0,))
    else:
        p_gk = None

    range_m = start_m * BM + tl.arange(0, BM)
    range_n = tl.arange(0, BN)

    m_i = tl.zeros([BM], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BM, DIM_VO], dtype=tl.float32)

    q = tl.load(p_q)
    if p_gq is not None:
        gq = tl.load(p_gq)
    else:
        gq = None

    if STAGE & 1: # non-masking part
        acc, m_i = _attn_fwd_inner(acc, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 4 - STAGE)
        
    if STAGE & 2: # masking part
        acc, m_i = _attn_fwd_inner(acc, m_i, q, gq, p_k, p_gk, p_v, #
                                   start_m, range_m, range_n, r, w, #
                                   deg, BM, BN, DIM_QK, DIM_VO, #
                                   M_CTX, N_CTX, 2)
        
    if NORM:
        acc = acc - (tl.sum(acc, axis=-1, keep_dims=True) / DIM_VO)
        acc = acc / tl.sqrt(tl.sum(acc*acc, axis=-1, keep_dims=True) / DIM_VO + 1e-7)

    o_offset = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
    m_offset = off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    p_o = tl.make_block_ptr(Out+o_offset, (M_CTX, DIM_VO), (stride_om, stride_oe), (start_m*BM, 0), (BM, DIM_VO), (1, 0))
    tl.store(M+m_offset+range_m*stride_mm, m_i)
    tl.store(p_o, acc.to(Out.type.element_ty))


bwd_configs = [
    triton.Config({'BN1': BN1, 'BM1': BM1, 'BN2': BN2, 'BM2': BM2, 'BLK_SLICE_FACTOR': BLK_SLICE_FACTOR}, num_stages=s, num_warps=w) \
    for BN1 in [64, 128, 256]\
    for BM1 in [32, 64]\
    for BM2 in [64, 128, 256]\
    for BN2 in [32, 64]\
    for s in [3, 4, 7]\
    for w in [4, 8]\
    for BLK_SLICE_FACTOR in [2,]\
]

def keep_bwd(conf):
    BN1 = conf.kwargs["BN1"]
    BM2 = conf.kwargs["BM2"]
    if BN1 != BM2:
        return False
    return True


@triton.jit
def _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                    Q, LOG_GQ, DO, M, #
                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, #
                    M_CTX, N_CTX, r, w, #
                    deg: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                    start_n, start_m, num_steps: tl.constexpr, #
                    MASK: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_m = start_m + tl.arange(0, BM)
    range_n = start_n + tl.arange(0, BN)

    p_qT = tl.make_block_ptr(Q, (DIM_QK, M_CTX), (stride_qd, stride_qm), (0, start_m), (DIM_QK, BM), (0, 1))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_doe), (start_m, 0), (BM, DIM_VO), (1, 0))
    if LOG_GQ is not None:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM,), (0,))
    else:
        p_gq = None

    curr_m = start_m
    for _ in range(num_steps):
        qT = tl.load(p_qT)
        if p_gq is not None:
            gq = tl.load(p_gq)
        else:
            gq = None
        p_m = M + (curr_m + tl.arange(0, BM))
        sT = tl.dot(k, qT) # (N, M)
        signs = sT > 0
        sT = deg * tl.log(sT.abs() + 1e-7)
        if gq is not None and gk is not None:
            sT = sT + gq[None, :] - gk[:, None]
        m = tl.load(p_m)
        if MASK:
            mask = (range_m[None, :] // r) >= (range_n[:, None] // w)
            sT = tl.where(mask, sT, -float("inf"))
        pT = tl.exp(sT - m[None, :])
        do = tl.load(p_do)
        # compute dV
        dv = tl.dot(pT.to(Q.type.element_ty), do, dv)
        # compute dK and dgk
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        dsT = pT * dpT
        if gk is not None:
            dgk += -tl.sum(dsT, 1, keep_dims=False)
        dsT = dsT * signs * deg / (tl.abs(sT) + 1e-7)
        dk = tl.dot(dsT, tl.trans(qT), dk)
        # increment pointers
        curr_m += BM
        p_qT = tl.advance(p_qT, (0, BM))
        p_do = tl.advance(p_do, (BM, 0))
        if p_gq is not None:
            p_gq = tl.advance(p_gq, (BM,))

    return dk, dv, dgk


@triton.jit
def _attn_bwd_dq(dq, dgq, q, gq, do, m, #
                  K, V, LOG_GK, #
                  stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                  M_CTX, N_CTX, r, w, #
                  deg: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DIM_QK: tl.constexpr, DIM_VO: tl.constexpr, #
                  start_m, start_n, num_steps: tl.constexpr, #
                  MASK: tl.constexpr):
    tl.static_assert(BM % r == 0, "BM must be divisible by r")
    tl.static_assert(BN % w == 0, "BN must be divisible by w")
    range_m = start_m + tl.arange(0, BM)
    range_n = start_n + tl.arange(0, BN)

    p_kT = tl.make_block_ptr(K, (DIM_QK, N_CTX), (stride_kd, stride_kn), (0, start_n), (DIM_QK, BN), (0, 1))
    p_vT = tl.make_block_ptr(V, (DIM_VO, N_CTX), (stride_ve, stride_vn), (0, start_n), (DIM_VO, BN), (0, 1))
    if LOG_GK is not None:
        p_gk = tl.make_block_ptr(LOG_GK, (N_CTX,), (stride_gkn,), (start_n,), (BN,), (0,))
    else:
        p_gk = None

    curr_n = start_n
    for _ in range(num_steps):
        kT = tl.load(p_kT)
        vT = tl.load(p_vT)
        if p_gk is not None:
            gk = tl.load(p_gk)
        else:
            gk = None
        s = tl.dot(q, kT) # (M, N)
        signs = s > 0
        s = deg * tl.log(s.abs() + 1e-7)
        if gq is not None and gk is not None:
            s = s + gq[:, None] - gk[None, :]
        if MASK:
            mask = (range_m[:, None] // r) >= (range_n[None, :] // w)
            s = tl.where(mask, s, -float("inf"))
        
        p = tl.exp(s - m[:, None])
        # compute dQ
        dp = tl.dot(do, vT).to(tl.float32)
        ds = dp * p
        if gq is not None:
            dgq += tl.sum(ds, 1, keep_dims=False)
        ds = ds * signs * deg / (tl.abs(s) + 1e-7)
        dq = tl.dot(ds, tl.trans(kT), dq)
        # increment pointers
        curr_n += BN
        p_kT = tl.advance(p_kT, (0, BN))
        p_vT = tl.advance(p_vT, (0, BN))
        if p_gk is not None:
            p_gk = tl.advance(p_gk, (BN,))

    return dq, dgq


@triton.autotune(list(filter(keep_bwd, bwd_configs)), key=["M_CTX", "N_CTX", "DIM_QK", "DIM_VO", "r", "w"])
@triton.jit
def _attn_bwd(Q, K, V, LOG_GQ, LOG_GK, M, DO, DQ, DK, DV, DLOG_GQ, DLOG_GK, #
              stride_qb, stride_qh, stride_qm, stride_qd, #
              stride_kb, stride_kh, stride_kn, stride_kd, #
              stride_vb, stride_vh, stride_vn, stride_ve, #
              stride_mb, stride_mh, stride_mm, #
              stride_gqb, stride_gqh, stride_gqm, #
              stride_gkb, stride_gkh, stride_gkn, #
              stride_dob, stride_doh, stride_dom, stride_doe, #
              stride_dqb, stride_dqh, stride_dqm, stride_dqd, #
              stride_dkb, stride_dkh, stride_dkn, stride_dkd, #
              stride_dvb, stride_dvh, stride_dvn, stride_dve, #
              H, M_CTX, N_CTX, r, w, #
              deg: tl.constexpr,  #
              DIM_QK: tl.constexpr,  #
              DIM_VO: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              BM1: tl.constexpr,  #
              BN1: tl.constexpr,  #
              BM2: tl.constexpr,  #
              BN2: tl.constexpr,  #
              BLK_SLICE_FACTOR: tl.constexpr,  #
              ):
    if STAGE == 3:
        tl.static_assert((BM1 // BLK_SLICE_FACTOR) % r == 0, "Sliced BM1 must be divisible by w")
        tl.static_assert((BN2 // BLK_SLICE_FACTOR) % w == 0, "Sliced BN2 must be divisible by w")
    else:
        tl.static_assert(BM1 % r == 0, "BM1 must be divisible by r")
        tl.static_assert(BN2 % w == 0, "BN2 must be divisible by w")
    tl.static_assert(BN1 % w == 0, "BN1 must be divisible by w")
    tl.static_assert(BM2 % r == 0, "BM2 must be divisible by r")

    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    start_n = tl.program_id(0)*BN1

    offset_q = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    offset_k = off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
    offset_v = off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
    offset_m = off_b.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    offset_do = off_b.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
    offset_gq = off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh
    offset_gk = off_b.to(tl.int64) * stride_gkb + off_h.to(tl.int64) * stride_gkh

    Q += offset_q
    V += offset_v
    M += offset_m
    DO += offset_do
    if LOG_GQ is not None:
        LOG_GQ += offset_gq
    if LOG_GK is not None:
        LOG_GK += offset_gk
    if DLOG_GQ is not None:
        DLOG_GQ += offset_gq
    if DLOG_GK is not None:
        DLOG_GK += offset_gk

    # -- First part: compute dk, dv
    MASK_BLOCK_M1: tl.constexpr = BM1 // BLK_SLICE_FACTOR
    range_n = start_n + tl.arange(0, BN1)

    dv = tl.zeros([BN1, DIM_VO], dtype=tl.float32)
    dk = tl.zeros([BN1, DIM_QK], dtype=tl.float32)
    dgk = tl.zeros([BN1,], dtype=tl.float32)

    # load k, v, gk
    p_k = tl.make_block_ptr(K+offset_k, (N_CTX, DIM_QK), (stride_kn, stride_kd), (start_n, 0), (BN1, DIM_QK), (1, 0))
    p_v = tl.make_block_ptr(V+offset_v, (N_CTX, DIM_VO), (stride_vn, stride_ve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    k = tl.load(p_k)
    v = tl.load(p_v)
    gk = tl.load(LOG_GK + range_n * stride_gkn)

    start_m = start_n if STAGE == 3 else 0
    if STAGE & 2: # masked blocks
        num_steps = BN1 // MASK_BLOCK_M1
        dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                    Q, LOG_GQ, DO, M, #
                                    stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, #
                                    M_CTX, N_CTX, r, w, #
                                    deg, MASK_BLOCK_M1, BN1, DIM_QK, DIM_VO, #
                                    start_n, start_m, num_steps, #
                                    MASK=True)
        start_m += num_steps * MASK_BLOCK_M1
        
    # unmasked blocks
    num_steps = (M_CTX - start_m) // BM1
    dk, dv, dgk = _attn_bwd_dkdv(dk, dv, dgk, k, v, gk, #
                                Q, LOG_GQ, DO, M, #
                                stride_qm, stride_qd, stride_dom, stride_doe, stride_gqm, #
                                M_CTX, N_CTX, r, w, #
                                deg, BM1, BN1, DIM_QK, DIM_VO, #
                                start_n, start_m, num_steps, #
                                MASK=False)

    offset_dk = off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
    offset_dv = off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
    p_dv = tl.make_block_ptr(DV+offset_dv, (N_CTX, DIM_VO), (stride_dvn, stride_dve), (start_n, 0), (BN1, DIM_VO), (1, 0))
    p_dk = tl.make_block_ptr(DK+offset_dk, (N_CTX, DIM_QK), (stride_dkn, stride_dkd), (start_n, 0), (BN1, DIM_QK), (1, 0))
    tl.store(p_dv, dv)
    tl.store(p_dk, dk)
    if DLOG_GK is not None:
        p_dgk = DLOG_GK + range_n * 1
        tl.store(p_dgk, dgk)

    # -- Second part: compute dq
    start_m = tl.program_id(0) * BM2

    MASK_BLOCK_N2: tl.constexpr = BN2 // BLK_SLICE_FACTOR

    dq = tl.zeros([BM2, DIM_QK], dtype=tl.float32)
    if LOG_GQ is not None:
        dgq = tl.zeros([BM2,], dtype=tl.float32)
    else:
        dgq = None

    # load q, gq
    p_q = tl.make_block_ptr(Q, (M_CTX, DIM_QK), (stride_qm, stride_qd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    p_do = tl.make_block_ptr(DO, (M_CTX, DIM_VO), (stride_dom, stride_dom), (start_m, 0), (BM2, DIM_VO), (1, 0))
    if LOG_GQ is not None:
        p_gq = tl.make_block_ptr(LOG_GQ, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
    else:
        p_gq = None
    q = tl.load(p_q)
    if LOG_GQ is not None:
        gq = tl.load(p_gq)
    else:
        gq = None
    m = tl.load(M + start_m + tl.arange(0, BM2))
    do = tl.load(p_do)

    end_n = start_m + BM2 if STAGE & 2 else M_CTX
    if STAGE & 2: # masked blocks
        num_steps = BM2 // MASK_BLOCK_N2
        dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, #
                               K, V, LOG_GK, #
                               stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                               M_CTX, N_CTX, r, w, #
                               deg, BM2, MASK_BLOCK_N2, DIM_QK, DIM_VO, #
                               start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, #
                               MASK=True)
        end_n -= num_steps * MASK_BLOCK_N2
    
    # unmasked blocks
    num_steps = end_n // BN2
    dq, dgq = _attn_bwd_dq(dq, dgq, q, gq, do, m, #
                           K, V, LOG_GK, #
                           stride_kn, stride_kd, stride_vn, stride_ve, stride_gkn, #
                           M_CTX, N_CTX, r, w, #
                           deg, BM2, BN2, DIM_QK, DIM_VO, #
                           start_m, end_n - num_steps * BN2, num_steps, #
                           MASK=False)

    # store dq, dgq
    offset_dq = off_b.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
    offset_dgq = off_b.to(tl.int64) * stride_gqb + off_h.to(tl.int64) * stride_gqh

    p_dq = tl.make_block_ptr(DQ+offset_dq, (M_CTX, DIM_QK), (stride_dqm, stride_dqd), (start_m, 0), (BM2, DIM_QK), (1, 0))
    tl.store(p_dq, dq)
    if LOG_GQ is not None:
        p_dgq = tl.make_block_ptr(DLOG_GQ+offset_dgq, (M_CTX,), (stride_gqm,), (start_m,), (BM2,), (0,))
        tl.store(p_dgq, dgq)


class _power_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, deg, log_G, r, w, causal, head_first, norm):
        """ Args:
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

            Returns:
                o: (B, H_Q // R, CTX, E) if head_first else (B, CTX, H_Q // R, E)
                rowmax: (B, H_Q // R, CTX) if head_first else (B, CTX, H_Q // R)
        """
        if head_first:
            b, hq, t, d, hk, e = *Q.shape, K.shape[1], V.shape[-1]
        else:
            b, t, hq, d, hk, e = *Q.shape, K.shape[2], V.shape[-1]
        assert r in {1, 2, 4, 8, 16}, "r must be 1, 2, 4, 8, or 16"
        assert w in {1, 2, 4, 8, 16}, "w must be 1, 2, 4, 8, or 16"
        assert hq % r == 0, "hq must be divisible by r"
        assert hk % w == 0, "hk must be divisible by w"
        assert hq // r == hk // w, "hq // r must be equal to hk // w"
        assert isinstance(deg, int) and deg > 0, "deg must be a positive integer"
        assert d in {16, 32, 64, 128, 256}, "d must be 16, 32, 64, 128, or 256"
        assert e in {16, 32, 64, 128, 256}, "e must be 16, 32, 64, 128, or 256"

        h = hq // r
        o = torch.empty_like(Q)

        if head_first:
            assert log_G.shape == (b, h, t)
            log_GQ = log_G.repeat_interleave(r, dim=2)
            log_GK = log_G.repeat_interleave(w, dim=2)
            gq_strides = (log_GQ.stride(0), log_GQ.stride(1), log_GQ.stride(2))
            gk_strides = (log_GK.stride(0), log_GK.stride(1), log_GK.stride(2))
            Q = Q.view(b, h, t * r, d)
            K = K.view(b, h, t * w, d)
            V = V.view(b, h, t * w, e)
            rowmax = torch.empty((b, h, t), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3))
            k_strides = (K.stride(0), K.stride(1), K.stride(2), K.stride(3))
            v_strides = (V.stride(0), V.stride(1), V.stride(2), V.stride(3))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(1), rowmax.stride(2))
            o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
        else:
            if log_G is not None:
                assert log_G.shape == (b, t, h)
                log_GQ = log_G.repeat_interleave(r, dim=1)
                log_GK = log_G.repeat_interleave(w, dim=1)
                gq_strides = (log_GQ.stride(0), log_GQ.stride(2), log_GQ.stride(1))
                gk_strides = (log_GK.stride(0), log_GK.stride(2), log_GK.stride(1))
            else:
                log_GQ = None
                log_GK = None
                gq_strides = (0, 0, 0)
                gk_strides = (0, 0, 0)
            Q = Q.view(b, t * r, h, d)
            K = K.view(b, t * w, h, d)
            V = V.view(b, t * w, h, e)
            rowmax = torch.empty((b, t, h), device=Q.device, dtype=torch.float32)
            q_strides = (Q.stride(0), Q.stride(2), Q.stride(1), Q.stride(3))
            k_strides = (K.stride(0), K.stride(2), K.stride(1), K.stride(3))
            v_strides = (V.stride(0), V.stride(2), V.stride(1), V.stride(3))
            rowmax_strides = (rowmax.stride(0), rowmax.stride(2), rowmax.stride(1))
            o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

        stage = 3 if causal else 1

        grid = lambda args: (triton.cdiv(r*t, args["BM"]), b * h)
        _attn_fwd[grid](
            Q, K, V, log_GQ, log_GK, rowmax, o, *q_strides, *k_strides, *v_strides, *rowmax_strides, *gq_strides, *gk_strides, *o_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, deg=deg, DIM_QK=d, DIM_VO=e, STAGE=stage, NORM=norm)
        
        ctx.save_for_backward(Q, K, V, rowmax, log_GQ, log_GK)
        ctx.b = b
        ctx.h = h
        ctx.t = t
        ctx.r = r
        ctx.w = w
        ctx.grid = grid
        ctx.d = d
        ctx.e = e
        ctx.deg = deg
        ctx.q_strides = q_strides
        ctx.k_strides = k_strides
        ctx.v_strides = v_strides
        ctx.rowmax_strides = rowmax_strides
        ctx.gq_strides = gq_strides
        ctx.gk_strides = gk_strides
        ctx.o_strides = o_strides
        ctx.head_first = head_first
        ctx.norm = norm
        ctx.stage = stage
        return o, rowmax

    @staticmethod
    def backward(ctx, do, drowmax=None):
        Q, K, V, rowmax, log_GQ, log_GK = ctx.saved_tensors
        do = do.contiguous() # needed for reuse o's strides for do
        assert log_GQ.is_contiguous() # needed for reuse log_GQ's strides for dlog_GQ
        assert log_GK.is_contiguous() # needed for reuse log_GK's strides for dlog_GK
        assert do.is_contiguous()
        b, h, t, norm, stage = ctx.b, ctx.h, ctx.t, ctx.norm, ctx.stage
        assert not norm, "normalized backward not implemented yet"
        q_strides, k_strides, v_strides, rowmax_strides, gq_strides, gk_strides, o_strides = ctx.q_strides, ctx.k_strides, ctx.v_strides, ctx.rowmax_strides, ctx.gq_strides, ctx.gk_strides, ctx.o_strides
        r, w, d, e, deg = ctx.r, ctx.w, ctx.d, ctx.e, ctx.deg
        do = do.contiguous()
        rowmax = rowmax.contiguous()

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dlog_GQ = torch.empty_like(log_GQ)
        dlog_GK = torch.empty_like(log_GK)

        if ctx.head_first:
            dQ_strides = (dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3))
            dK_strides = (dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3))
            dV_strides = (dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3))
        else:
            dQ_strides = (dQ.stride(0), dQ.stride(2), dQ.stride(1), dQ.stride(3))
            dK_strides = (dK.stride(0), dK.stride(2), dK.stride(1), dK.stride(3))
            dV_strides = (dV.stride(0), dV.stride(2), dV.stride(1), dV.stride(3))
        
        grid = lambda args: (triton.cdiv(w*t, args["BN1"]), b * h)

        _attn_bwd[grid](
            Q, K, V, log_GQ, log_GK, rowmax, do, dQ, dK, dV, dlog_GQ, dlog_GK,
            *q_strides, *k_strides, *v_strides, *rowmax_strides, *gq_strides, *gk_strides, *o_strides,
            *dQ_strides, *dK_strides, *dV_strides,
            H=h, M_CTX=t*r, N_CTX=t*w, r=r, w=w, deg=deg, DIM_QK=d, DIM_VO=e, STAGE=stage)
        if ctx.head_first:
            dlog_G = dlog_GQ.view(b, h, t, r).sum(dim=-1) + dlog_GK.view(b, h, t, w).sum(dim=-1)
        else:
            dlog_G = dlog_GQ.view(b, t, r, h).sum(dim=-2) + dlog_GK.view(b, t, w, h).sum(dim=-2)
        return dQ, dK, dV, None, dlog_G, None, None, None, None, None


def attention(Q, K, V, deg, log_G, r=1, w=1, causal=True, head_first=False, norm=False):
    return _power_attention.apply(Q, K, V, deg, log_G, r, w, causal, head_first, norm)


def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', scale=1.0, deg=2, r=1, w=1, causal=True, gating=False, head_first=False, norm=False, requires_grad=False, seed=42, std=1.0):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) * std
    if gating:
        log_G = F.logsigmoid(torch.rand(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    if requires_grad:
        Q, K, V, log_G = tree_map(lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, r=r, w=w, causal=causal, head_first=head_first, norm=norm)


if __name__ == "__main__":
    from power_attention._attention.impl import attention as attention_cutlass, create_inputs as create_inputs_cutlass
    from power_attention._attention.reference import attention_reference as attention_reference_old
    from perf._timing import benchmark_speed

    # Hyperparameters
    kw = dict(b=1, t=128, h=1, d=64, dtype=torch.float16, device='cuda', scale=1.0, deg=2, seed=42)

    # Check correctness
    inputs_triton = create_inputs(**(kw | {'requires_grad': True, 'r': 1, 'w': 1, 'causal': True, 'head_first': False, 'norm': False}))
    inputs_ref = create_inputs(**(kw | {'requires_grad': True, 'r': 1, 'w': 1, 'causal': True, 'head_first': False, 'norm': False}))
    inputs_cutlass = create_inputs_cutlass(**(kw | {'requires_grad': True}))

    for key in ['Q', 'K', 'V', 'log_G']:
        torch.testing.assert_close(inputs_triton[key], inputs_cutlass[key])
        torch.testing.assert_close(inputs_triton[key], inputs_ref[key])

    o_triton, rowmax_triton = attention(**inputs_triton)
    o_cutlass, _, rowmax_cutlass = attention_cutlass(**inputs_cutlass)
    o_ref1, _, rowmax_ref1 = attention_reference_old(**inputs_cutlass)
    o_ref, rowmax_ref = attention_reference(**inputs_triton)

    torch.testing.assert_close(rowmax_cutlass, rowmax_ref.to(rowmax_cutlass.dtype), atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(rowmax_triton, rowmax_cutlass, atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(o_triton, o_ref, atol=1.5e-2, rtol=1e-2)
    torch.testing.assert_close(o_cutlass, o_ref, atol=1.5e-2, rtol=1e-2)
    torch.testing.assert_close(o_triton, o_cutlass, atol=2e-3, rtol=1e-2)
    print("Correctness check passed")
    
    