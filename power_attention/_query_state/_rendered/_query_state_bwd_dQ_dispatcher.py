import triton
import triton.language as tl

def _query_state_bwd_dQ(Q, S, M, dO, dQ, dY_attn, deg: tl.constexpr, scale, zero_initial_state,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_mb, stride_mt, stride_mh,
                     stride_dob, stride_dot, stride_doh, stride_doe,
                     stride_dyb, stride_dyt, stride_dyh, stride_dye,
                     stride_dqb, stride_dqt, stride_dqh, stride_dqd,
                     n, h, c, d: tl.constexpr, D: tl.constexpr, e: tl.constexpr,
                     block1: tl.constexpr, BLOCK_T: tl.constexpr, 
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    if ((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
        if (d == 32):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            
        
        elif (d == 64):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    elif m == 1:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    elif m == 2:
                        dq_2 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_3 += dpq_0 * q_d2_0[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 0)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_0[:, None].broadcast_to(dq_2.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 0)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_0[:, None].broadcast_to(dq_3.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            p_dq_2 = dQ + range_t[:, None] * stride_dqt + (2 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_2, dq_2, mask=(range_t < c)[:, None])
            p_dq_3 = dQ + range_t[:, None] * stride_dqt + (3 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_3, dq_3, mask=(range_t < c)[:, None])
            
        
        elif (d == 128):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_4 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_5 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_6 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_7 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    elif m == 1:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    elif m == 2:
                        dq_2 += dpq_0 * q_d2_0[:, None]
                    elif m == 3:
                        dq_3 += dpq_0 * q_d2_0[:, None]
                    elif m == 4:
                        dq_4 += dpq_0 * q_d2_0[:, None]
                    elif m == 5:
                        dq_5 += dpq_0 * q_d2_0[:, None]
                    elif m == 6:
                        dq_6 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_7 += dpq_0 * q_d2_0[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 0)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_0[:, None].broadcast_to(dq_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 0)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_0[:, None].broadcast_to(dq_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = (tl.arange(0, block1) + 4 * block1) == (off_d2 + 0)
                        dq_4 += tl.where(mask[None, :].broadcast_to(dq_4.shape), dq_d2_0[:, None].broadcast_to(dq_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = (tl.arange(0, block1) + 5 * block1) == (off_d2 + 0)
                        dq_5 += tl.where(mask[None, :].broadcast_to(dq_5.shape), dq_d2_0[:, None].broadcast_to(dq_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = (tl.arange(0, block1) + 6 * block1) == (off_d2 + 0)
                        dq_6 += tl.where(mask[None, :].broadcast_to(dq_6.shape), dq_d2_0[:, None].broadcast_to(dq_6.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 7 * block1) == (off_d2 + 0)
                        dq_7 += tl.where(mask[None, :].broadcast_to(dq_7.shape), dq_d2_0[:, None].broadcast_to(dq_7.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            p_dq_2 = dQ + range_t[:, None] * stride_dqt + (2 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_2, dq_2, mask=(range_t < c)[:, None])
            p_dq_3 = dQ + range_t[:, None] * stride_dqt + (3 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_3, dq_3, mask=(range_t < c)[:, None])
            p_dq_4 = dQ + range_t[:, None] * stride_dqt + (4 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_4, dq_4, mask=(range_t < c)[:, None])
            p_dq_5 = dQ + range_t[:, None] * stride_dqt + (5 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_5, dq_5, mask=(range_t < c)[:, None])
            p_dq_6 = dQ + range_t[:, None] * stride_dqt + (6 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_6, dq_6, mask=(range_t < c)[:, None])
            p_dq_7 = dQ + range_t[:, None] * stride_dqt + (7 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_7, dq_7, mask=(range_t < c)[:, None])
            
    elif ((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 256) and ((block1 == 16))))) or (((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((block1 == 16))))) or ((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 256) and ((block1 == 16))))))):
        
        if (d == 32):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    
                    p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
                    p_sT_1 = S + (range_d1[None, :] + off_D + 1 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    q_d2_1 = tl.load(p_q_d2_1, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_1 = tl.load(p_sT_1) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    
                    dpq_1 = tl.dot(do, sT_1) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_1 * q_d2_1[:, None]
                    else:
                        dq_1 += dpq_1 * q_d2_1[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    
                    dq_d2_1 = tl.sum(dpq_1 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 1)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_1[:, None].broadcast_to(dq_0.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 1)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_1[:, None].broadcast_to(dq_1.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            
        
        elif (d == 64):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    
                    p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
                    p_sT_1 = S + (range_d1[None, :] + off_D + 1 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    q_d2_1 = tl.load(p_q_d2_1, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_1 = tl.load(p_sT_1) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    elif m == 1:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    elif m == 2:
                        dq_2 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_3 += dpq_0 * q_d2_0[:, None]
                    
                    dpq_1 = tl.dot(do, sT_1) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_1 * q_d2_1[:, None]
                    elif m == 1:
                        dq_1 += dpq_1 * q_d2_1[:, None]
                    elif m == 2:
                        dq_2 += dpq_1 * q_d2_1[:, None]
                    else:
                        dq_3 += dpq_1 * q_d2_1[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 0)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_0[:, None].broadcast_to(dq_2.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 0)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_0[:, None].broadcast_to(dq_3.shape), 0.)
                    
                    dq_d2_1 = tl.sum(dpq_1 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 1)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_1[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 1)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_1[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 1)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_1[:, None].broadcast_to(dq_2.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 1)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_1[:, None].broadcast_to(dq_3.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            p_dq_2 = dQ + range_t[:, None] * stride_dqt + (2 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_2, dq_2, mask=(range_t < c)[:, None])
            p_dq_3 = dQ + range_t[:, None] * stride_dqt + (3 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_3, dq_3, mask=(range_t < c)[:, None])
            
        
        elif (d == 128):     
            off_bnh = tl.program_id(0)
            off_bn = off_bnh // h
            off_h = off_bnh % h
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
            S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
            dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
            dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
            do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
            if M is not None:
                M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
                p_m = M + range_t * stride_mt
                rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
                qs_factor = tl.minimum(tl.exp(-rowmax), scale)
                do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
            dq_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_4 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_5 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_6 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dq_7 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            if dY_attn is not None:
                chunk_id = off_bn % n
                dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
                p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
                if chunk_id > 0 or (not zero_initial_state):
                    attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
                    dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
                    tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
                else:
                    tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])
            
            for m in range(0, d//block1):
                p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
                q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
                dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    
                    p_q_d2_0 = Q + range_t[:] * stride_qt + (off_d2 + 0) * stride_qd # [BLOCK_T]
                    p_sT_0 = S + (range_d1[None, :] + off_D + 0 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    
                    p_q_d2_1 = Q + range_t[:] * stride_qt + (off_d2 + 1) * stride_qd # [BLOCK_T]
                    p_sT_1 = S + (range_d1[None, :] + off_D + 1 * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
                    q_d2_0 = tl.load(p_q_d2_0, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_0 = tl.load(p_sT_0) # [BLOCK_E x block1]
                    q_d2_1 = tl.load(p_q_d2_1, mask=(range_t < c), other=0.) # [BLOCK_T]
                    sT_1 = tl.load(p_sT_1) # [BLOCK_E x block1]
                    
                    dpq_0 = tl.dot(do, sT_0) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_0 * q_d2_0[:, None]
                    elif m == 1:
                        dq_1 += dpq_0 * q_d2_0[:, None]
                    elif m == 2:
                        dq_2 += dpq_0 * q_d2_0[:, None]
                    elif m == 3:
                        dq_3 += dpq_0 * q_d2_0[:, None]
                    elif m == 4:
                        dq_4 += dpq_0 * q_d2_0[:, None]
                    elif m == 5:
                        dq_5 += dpq_0 * q_d2_0[:, None]
                    elif m == 6:
                        dq_6 += dpq_0 * q_d2_0[:, None]
                    else:
                        dq_7 += dpq_0 * q_d2_0[:, None]
                    
                    dpq_1 = tl.dot(do, sT_1) # [BLOCK_T x block1]
                    if m == 0:
                        dq_0 += dpq_1 * q_d2_1[:, None]
                    elif m == 1:
                        dq_1 += dpq_1 * q_d2_1[:, None]
                    elif m == 2:
                        dq_2 += dpq_1 * q_d2_1[:, None]
                    elif m == 3:
                        dq_3 += dpq_1 * q_d2_1[:, None]
                    elif m == 4:
                        dq_4 += dpq_1 * q_d2_1[:, None]
                    elif m == 5:
                        dq_5 += dpq_1 * q_d2_1[:, None]
                    elif m == 6:
                        dq_6 += dpq_1 * q_d2_1[:, None]
                    else:
                        dq_7 += dpq_1 * q_d2_1[:, None]
                    
                    dq_d2_0 = tl.sum(dpq_0 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 0)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_0[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 0)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_0[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 0)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_0[:, None].broadcast_to(dq_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 0)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_0[:, None].broadcast_to(dq_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = (tl.arange(0, block1) + 4 * block1) == (off_d2 + 0)
                        dq_4 += tl.where(mask[None, :].broadcast_to(dq_4.shape), dq_d2_0[:, None].broadcast_to(dq_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = (tl.arange(0, block1) + 5 * block1) == (off_d2 + 0)
                        dq_5 += tl.where(mask[None, :].broadcast_to(dq_5.shape), dq_d2_0[:, None].broadcast_to(dq_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = (tl.arange(0, block1) + 6 * block1) == (off_d2 + 0)
                        dq_6 += tl.where(mask[None, :].broadcast_to(dq_6.shape), dq_d2_0[:, None].broadcast_to(dq_6.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 7 * block1) == (off_d2 + 0)
                        dq_7 += tl.where(mask[None, :].broadcast_to(dq_7.shape), dq_d2_0[:, None].broadcast_to(dq_7.shape), 0.)
                    
                    dq_d2_1 = tl.sum(dpq_1 * q_d1, 1) # [BLOCK_T]
                    if off_d2//block1 == 0:
                        mask = (tl.arange(0, block1) + 0 * block1) == (off_d2 + 1)
                        dq_0 += tl.where(mask[None, :].broadcast_to(dq_0.shape), dq_d2_1[:, None].broadcast_to(dq_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = (tl.arange(0, block1) + 1 * block1) == (off_d2 + 1)
                        dq_1 += tl.where(mask[None, :].broadcast_to(dq_1.shape), dq_d2_1[:, None].broadcast_to(dq_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = (tl.arange(0, block1) + 2 * block1) == (off_d2 + 1)
                        dq_2 += tl.where(mask[None, :].broadcast_to(dq_2.shape), dq_d2_1[:, None].broadcast_to(dq_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = (tl.arange(0, block1) + 3 * block1) == (off_d2 + 1)
                        dq_3 += tl.where(mask[None, :].broadcast_to(dq_3.shape), dq_d2_1[:, None].broadcast_to(dq_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = (tl.arange(0, block1) + 4 * block1) == (off_d2 + 1)
                        dq_4 += tl.where(mask[None, :].broadcast_to(dq_4.shape), dq_d2_1[:, None].broadcast_to(dq_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = (tl.arange(0, block1) + 5 * block1) == (off_d2 + 1)
                        dq_5 += tl.where(mask[None, :].broadcast_to(dq_5.shape), dq_d2_1[:, None].broadcast_to(dq_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = (tl.arange(0, block1) + 6 * block1) == (off_d2 + 1)
                        dq_6 += tl.where(mask[None, :].broadcast_to(dq_6.shape), dq_d2_1[:, None].broadcast_to(dq_6.shape), 0.)
                    else:
                        mask = (tl.arange(0, block1) + 7 * block1) == (off_d2 + 1)
                        dq_7 += tl.where(mask[None, :].broadcast_to(dq_7.shape), dq_d2_1[:, None].broadcast_to(dq_7.shape), 0.)
                    # save dq
            p_dq_0 = dQ + range_t[:, None] * stride_dqt + (0 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_0, dq_0, mask=(range_t < c)[:, None])
            p_dq_1 = dQ + range_t[:, None] * stride_dqt + (1 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_1, dq_1, mask=(range_t < c)[:, None])
            p_dq_2 = dQ + range_t[:, None] * stride_dqt + (2 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_2, dq_2, mask=(range_t < c)[:, None])
            p_dq_3 = dQ + range_t[:, None] * stride_dqt + (3 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_3, dq_3, mask=(range_t < c)[:, None])
            p_dq_4 = dQ + range_t[:, None] * stride_dqt + (4 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_4, dq_4, mask=(range_t < c)[:, None])
            p_dq_5 = dQ + range_t[:, None] * stride_dqt + (5 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_5, dq_5, mask=(range_t < c)[:, None])
            p_dq_6 = dQ + range_t[:, None] * stride_dqt + (6 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_6, dq_6, mask=(range_t < c)[:, None])
            p_dq_7 = dQ + range_t[:, None] * stride_dqt + (7 * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
            tl.store(p_dq_7, dq_7, mask=(range_t < c)[:, None])
            
    else:
        tl.static_assert(False, "No matching config found")