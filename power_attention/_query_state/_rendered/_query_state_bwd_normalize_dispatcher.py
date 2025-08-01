import triton
import triton.language as tl

def _query_state_bwd_normalize(dO, Delta, M, L, dY_qs, dL_qs, dY_attn, dL_attn,
                        stride_dob, stride_dot, stride_doh, stride_doe,
                        stride_db, stride_dt, stride_dh,
                        stride_mb, stride_mt, stride_mh,
                        stride_lb, stride_lt, stride_lh,
                        stride_dYqs_b, stride_dYqs_t, stride_dYqs_h, stride_dYqs_e,
                        stride_dLqs_b, stride_dLqs_t, stride_dLqs_h,
                        stride_dYattn_b, stride_dYattn_t, stride_dYattn_h, stride_dYattn_e,
                        stride_dLattn_b, stride_dLattn_t, stride_dLattn_h,
                        n, h, c, e: tl.constexpr, D,
                        zero_initial_state: tl.constexpr,
                        deg: tl.constexpr,
                        scale_p,
                        BLOCK_T: tl.constexpr):
    if ((BLOCK_T == 128)) or (((BLOCK_T == 256)) or (((BLOCK_T == 32)) or ((BLOCK_T == 64)))):
        
        off_bnh = tl.program_id(0)
        off_bn = off_bnh // h
        off_h = off_bnh % h
        off_t = tl.program_id(1)
        
        dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
        Delta += off_bn.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
        M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
        L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
        dY_qs += off_bn.to(tl.int64) * stride_dYqs_b + off_h.to(tl.int64) * stride_dYqs_h
        dL_qs += off_bn.to(tl.int64) * stride_dLqs_b + off_h.to(tl.int64) * stride_dLqs_h
        dY_attn += off_bn.to(tl.int64) * stride_dYattn_b + off_h.to(tl.int64) * stride_dYattn_h
        dL_attn += off_bn.to(tl.int64) * stride_dLattn_b + off_h.to(tl.int64) * stride_dLattn_h
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
        range_e = tl.arange(0, e).to(tl.int64)
        p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x e]
        p_m = M + range_t * stride_mt
        p_dY_attn = dY_attn + range_t[:, None] * stride_dYattn_t + range_e[None, :] * stride_dYattn_e # [BLOCK_T x e]
        p_l = L + range_t * stride_lt
        p_dL_attn = dL_attn + range_t * stride_dLattn_t
        p_dY_qs = dY_qs + range_t[:, None] * stride_dYqs_t + range_e[None, :] * stride_dYqs_e # [BLOCK_T x e]
        p_dL_qs = dL_qs + range_t * stride_dLqs_t
        
        # --- compute dL_attn ---
        rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
        l = tl.load(p_l, mask=range_t < c, other=float('inf'))
        p_delta = Delta + range_t * stride_dt
        delta = tl.load(p_delta, mask=range_t < c, other=0.)
        
        gamma = tl.sqrt(D).to(tl.float32)
        m = tl.exp(rowmax)
        alpha = tl.maximum(gamma, m) # BLOCK_T
        attn_factor = m / alpha / l # BLOCK_T
        dL_attn = -attn_factor * delta # BLOCK_T
        tl.store(p_dL_attn, dL_attn, mask=range_t < c)
        
        # --- compute dL_qs ---
        qs_factor = scale_p * gamma / alpha / l # BLOCK_T
        dL_qs = -qs_factor * delta # BLOCK_T
        tl.store(p_dL_qs, dL_qs, mask=range_t < c)
        
        # --- compute dY_attn ---
        do = tl.load(p_do, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x e]
        dy_attn = (do * attn_factor[:, None]).to(dO.dtype.element_ty) # BLOCK_T x e
        tl.store(p_dY_attn, dy_attn, mask=(range_t < c)[:, None])
        
        # --- compute dY_qs ---
        dy_qs = (do * qs_factor[:, None]).to(dO.dtype.element_ty) # BLOCK_T x e
        tl.store(p_dY_qs, dy_qs, mask=(range_t < c)[:, None])
            
    else:
        tl.static_assert(False, "No matching config found")