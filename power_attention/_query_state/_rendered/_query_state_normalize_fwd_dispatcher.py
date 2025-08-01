import triton
import triton.language as tl

def _query_state_normalize_fwd(Y_qs, L_qs, Y_attn, L_attn, M, O, L,
                     stride_Yqs_b, stride_Yqs_t, stride_Yqs_h, stride_Yqs_e,
                     stride_Lqs_b, stride_Lqs_t, stride_Lqs_h,
                     stride_Yattn_b, stride_Yattn_t, stride_Yattn_h, stride_Yattn_e,
                     stride_Lattn_b, stride_Lattn_t, stride_Lattn_h,
                     stride_mb, stride_mt, stride_mh,
                     stride_ob, stride_ot, stride_oh, stride_oe,
                     stride_lb, stride_lt, stride_lh,
                     n, h, c, e: tl.constexpr, D,
                     zero_initial_state,
                     deg: tl.constexpr,
                     scale_p,
                     BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
    if ((BLOCK_E == 32) and ((BLOCK_T == 128))) or (((BLOCK_E == 32) and ((BLOCK_T == 256))) or (((BLOCK_E == 32) and ((BLOCK_T == 32))) or (((BLOCK_E == 32) and ((BLOCK_T == 64))) or (((BLOCK_E == 64) and ((BLOCK_T == 128))) or (((BLOCK_E == 64) and ((BLOCK_T == 256))) or (((BLOCK_E == 64) and ((BLOCK_T == 32))) or ((BLOCK_E == 64) and ((BLOCK_T == 64))))))))):
        
        off_bnh = tl.program_id(0)
        off_bn = off_bnh // h
        off_h = off_bnh % h
        off_n = off_bn % n
        off_t = tl.program_id(1)
        off_e = tl.program_id(2)
        
        Y_qs += off_bn.to(tl.int64) * stride_Yqs_b + off_h.to(tl.int64) * stride_Yqs_h
        L_qs += off_bn.to(tl.int64) * stride_Lqs_b + off_h.to(tl.int64) * stride_Lqs_h
        O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
        L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
        Y_attn += off_bn.to(tl.int64) * stride_Yattn_b + off_h.to(tl.int64) * stride_Yattn_h
        L_attn += off_bn.to(tl.int64) * stride_Lattn_b + off_h.to(tl.int64) * stride_Lattn_h
        M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
        range_e = tl.arange(0, BLOCK_E_VALID).to(tl.int64) + off_e * BLOCK_E_VALID
        
        y = tl.zeros((BLOCK_T, BLOCK_E_VALID), dtype=tl.float32)
        l = tl.zeros((BLOCK_T,), dtype=tl.float32)
        gamma = tl.sqrt(D).to(tl.float32)
        
        mask_T = range_t < c
        
        p_y_attn = Y_attn + range_t[:, None] * stride_Yattn_t + range_e[None, :] * stride_Yattn_e
        p_m = M + range_t * stride_mt
        p_l_attn = L_attn + range_t * stride_Lattn_t
        p_y_qs = Y_qs + range_t[:, None] * stride_Yqs_t + range_e[None, :] * stride_Yqs_e
        p_l_qs = L_qs + range_t * stride_Lqs_t
        rowmax = tl.load(p_m, mask=mask_T, other=-float('inf')) # BLOCK_T
        y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32) # BLOCK_T x BLOCK_E_VALID
        l_attn = tl.load(p_l_attn, mask=mask_T, other=0.).to(tl.float32) # BLOCK_T
        y_qs = tl.load(p_y_qs, mask=mask_T[:, None], other=0.).to(tl.float32) # BLOCK_T x BLOCK_E_VALID
        l_qs = tl.load(p_l_qs, mask=mask_T, other=0.).to(tl.float32) # BLOCK_T
        m = tl.exp(rowmax)
        alpha = tl.maximum(gamma, m) # BLOCK_T
        qs_factor = scale_p * gamma / alpha # BLOCK_T
        attn_factor = m / alpha # BLOCK_T
        
        o = y_attn * attn_factor[:, None] + y_qs * qs_factor[:, None] # BLOCK_T x BLOCK_E_VALID
        l = l_attn * attn_factor + l_qs * qs_factor # BLOCK_T
        o = o / l[:, None] # BLOCK_T x BLOCK_E_VALID
        
        # store y back to O
        p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
        p_l = L + range_t * stride_lt
        tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
        tl.store(p_l, l, mask=mask_T)
            
    else:
        tl.static_assert(False, "No matching config found")