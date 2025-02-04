import triton
import triton.language as tl

def _query_state_fwd(Q, S, Y, M, O, deg: tl.constexpr, scale, zero_initial_state: tl.constexpr,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_yb, stride_yt, stride_yh, stride_ye,
                     stride_mb, stride_mt, stride_mh,
                     stride_ob, stride_ot, stride_oh, stride_oe,
                     n, h, c, d: tl.constexpr, D: tl.constexpr, e: tl.constexpr,
                     block1: tl.constexpr,
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    if ((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
        tl.static_assert(block1 >= block2 and block1 % block2 == 0)
        off_bnh = tl.program_id(0)
        off_bn = off_bnh // h
        off_h = off_bnh % h
        off_t = tl.program_id(1)
        off_e = tl.program_id(2)
        
        Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
        S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
        O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
        if Y is not None:
            Y += off_bn.to(tl.int64) * stride_yb + off_h.to(tl.int64) * stride_yh
            M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
        range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
        range_d1 = tl.arange(0, block1).to(tl.int64)
        p_s = S + tl.arange(0, BLOCK_D)[:, None] * stride_sD + range_e[None, :] * stride_se
        
        y = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
        mask_T = range_t < c
        
        m, n = 0, 0
        for m in range(0, d//block1):
            p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
            q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1
        
            for n in range(0, (m+1)*block1//block2):
                off_d2 = n*block2
                off_d2 = tl.multiple_of(off_d2, block2)
                off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # BLOCK_T
                p_s_0 = S + (range_d1[:, None] + off_D + 0 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_T, other=0.) # BLOCK_T
                phik_0 = q_d1 * (q_d2_0[:, None]) # BLOCK_T x block1
                s_0 = tl.load(p_s_0) # block1 x BLOCK_E
                if scale != 1.0:
                    s_0 = (s_0 * scale).to(Q.dtype.element_ty) # block1 x BLOCK_E
                y = tl.dot(phik_0.to(Q.dtype.element_ty), s_0, y) # BLOCK_T x BLOCK_E
                
        
        chunk_id = off_bn % n
        if Y is not None:
            p_y_attn = Y + range_t[:, None] * stride_yt + range_e[None, :] * stride_ye
            p_m = M + range_t * stride_mt
            rowmax = tl.load(p_m, mask=mask_T, other=float(0.0))
            y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32)
            exp_neg_m = tl.exp(-rowmax)
            min_scale = tl.minimum(scale, exp_neg_m)
            y_attn = y_attn * (min_scale / exp_neg_m)[:, None]
            if (chunk_id > 0 or (not zero_initial_state)):
                o = y * ((min_scale / scale)[:, None]) + y_attn
            else:
                o = y
        else:
            o = y
        
        # store y back to O
        p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
        tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
            
    elif ((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
        tl.static_assert(block1 >= block2 and block1 % block2 == 0)
        off_bnh = tl.program_id(0)
        off_bn = off_bnh // h
        off_h = off_bnh % h
        off_t = tl.program_id(1)
        off_e = tl.program_id(2)
        
        Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
        S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
        O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
        if Y is not None:
            Y += off_bn.to(tl.int64) * stride_yb + off_h.to(tl.int64) * stride_yh
            M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
        range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
        range_d1 = tl.arange(0, block1).to(tl.int64)
        p_s = S + tl.arange(0, BLOCK_D)[:, None] * stride_sD + range_e[None, :] * stride_se
        
        y = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
        mask_T = range_t < c
        
        m, n = 0, 0
        for m in range(0, d//block1):
            p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
            q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1
        
            for n in range(0, (m+1)*block1//block2):
                off_d2 = n*block2
                off_d2 = tl.multiple_of(off_d2, block2)
                off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # BLOCK_T
                p_s_0 = S + (range_d1[:, None] + off_D + 0 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E
                p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # BLOCK_T
                p_s_1 = S + (range_d1[:, None] + off_D + 1 * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E
                q_d2_0 = tl.load(p_q_d2_0, mask=mask_T, other=0.) # BLOCK_T
                q_d2_1 = tl.load(p_q_d2_1, mask=mask_T, other=0.) # BLOCK_T
                phik_0 = q_d1 * (q_d2_0[:, None]) # BLOCK_T x block1
                s_0 = tl.load(p_s_0) # block1 x BLOCK_E
                if scale != 1.0:
                    s_0 = (s_0 * scale).to(Q.dtype.element_ty) # block1 x BLOCK_E
                phik_1 = q_d1 * (q_d2_1[:, None]) # BLOCK_T x block1
                s_1 = tl.load(p_s_1) # block1 x BLOCK_E
                if scale != 1.0:
                    s_1 = (s_1 * scale).to(Q.dtype.element_ty) # block1 x BLOCK_E
                y = tl.dot(phik_0.to(Q.dtype.element_ty), s_0, y) # BLOCK_T x BLOCK_E
                y = tl.dot(phik_1.to(Q.dtype.element_ty), s_1, y) # BLOCK_T x BLOCK_E
                
        
        chunk_id = off_bn % n
        if Y is not None:
            p_y_attn = Y + range_t[:, None] * stride_yt + range_e[None, :] * stride_ye
            p_m = M + range_t * stride_mt
            rowmax = tl.load(p_m, mask=mask_T, other=float(0.0))
            y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32)
            exp_neg_m = tl.exp(-rowmax)
            min_scale = tl.minimum(scale, exp_neg_m)
            y_attn = y_attn * (min_scale / exp_neg_m)[:, None]
            if (chunk_id > 0 or (not zero_initial_state)):
                o = y * ((min_scale / scale)[:, None]) + y_attn
            else:
                o = y
        else:
            o = y
        
        # store y back to O
        p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
        tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
            
    else:
        tl.static_assert(False, "No matching config found")