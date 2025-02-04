import triton
import triton.language as tl

def _update_state_bwd(K, V, dS, dK, dV, deg: tl.constexpr,
                      stride_kb, stride_kt, stride_kh, stride_kd,
                      stride_vb, stride_vt, stride_vh, stride_ve,
                      stride_dsb, stride_dsh, stride_dsD, stride_dse,
                      stride_dkb, stride_dkt, stride_dkh, stride_dkd,
                      stride_dvb, stride_dvt, stride_dvh, stride_dve,
                      T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                      block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr, V_IN_REGS: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    if ((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((V_IN_REGS == False) and ((block1 == 16)))))) or (((BLOCK_D == 16) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((V_IN_REGS == True) and ((block1 == 16)))))) or (((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((V_IN_REGS == False) and ((block1 == 16)))))) or ((BLOCK_D == 16) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((V_IN_REGS == True) and ((block1 == 16)))))))):
        
        if (d == 32):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    else:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
        
        elif (d == 64):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_3 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 0))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_0[:, None].broadcast_to(dk_2.shape), 0.)
                    else:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 0))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_0[:, None].broadcast_to(dk_3.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dk_2 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (2 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_2, dk_2, mask=mask_T[:, None])
            p_dk_3 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (3 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_3, dk_3, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
        
        elif (d == 128):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_4 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_5 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_6 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_7 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 3:
                        dk_3 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 4:
                        dk_4 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 5:
                        dk_5 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 6:
                        dk_6 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_7 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 0))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_0[:, None].broadcast_to(dk_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 0))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_0[:, None].broadcast_to(dk_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = ((range_d1 + 4 * block1) == (off_d2 + 0))
                        dk_4 += tl.where(mask[None, :].broadcast_to(dk_4.shape), dk_d2_0[:, None].broadcast_to(dk_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = ((range_d1 + 5 * block1) == (off_d2 + 0))
                        dk_5 += tl.where(mask[None, :].broadcast_to(dk_5.shape), dk_d2_0[:, None].broadcast_to(dk_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = ((range_d1 + 6 * block1) == (off_d2 + 0))
                        dk_6 += tl.where(mask[None, :].broadcast_to(dk_6.shape), dk_d2_0[:, None].broadcast_to(dk_6.shape), 0.)
                    else:
                        mask = ((range_d1 + 7 * block1) == (off_d2 + 0))
                        dk_7 += tl.where(mask[None, :].broadcast_to(dk_7.shape), dk_d2_0[:, None].broadcast_to(dk_7.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dk_2 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (2 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_2, dk_2, mask=mask_T[:, None])
            p_dk_3 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (3 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_3, dk_3, mask=mask_T[:, None])
            p_dk_4 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (4 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_4, dk_4, mask=mask_T[:, None])
            p_dk_5 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (5 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_5, dk_5, mask=mask_T[:, None])
            p_dk_6 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (6 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_6, dk_6, mask=mask_T[:, None])
            p_dk_7 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (7 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_7, dk_7, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
    elif ((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((V_IN_REGS == False) and ((block1 == 16)))))) or (((BLOCK_D == 32) and ((BLOCK_E == 32) and ((BLOCK_T == 128) and ((V_IN_REGS == True) and ((block1 == 16)))))) or (((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((V_IN_REGS == False) and ((block1 == 16)))))) or ((BLOCK_D == 32) and ((BLOCK_E == 64) and ((BLOCK_T == 128) and ((V_IN_REGS == True) and ((block1 == 16)))))))):
        
        if (d == 32):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd # BLOCK_T
                    p_ds_1 = dS + (range_d1[:, None] + off_D + 1 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    k_d2_1 = tl.load(p_k_d2_1, mask=mask_T, other=0.) # BLOCK_T
                    ds_1 = (tl.load(p_ds_1) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    phik_1 = k_d1 * (k_d2_1[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_1.to(K.dtype.element_ty), ds_1, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    
                    dphik_1 = tl.dot(v, tl.trans(ds_1)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    else:
                        dk_1 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    else:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    dk_d2_1 = tl.sum(dphik_1 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 1))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_1[:, None].broadcast_to(dk_0.shape), 0.)
                    else:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 1))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_1[:, None].broadcast_to(dk_1.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
        
        elif (d == 64):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd # BLOCK_T
                    p_ds_1 = dS + (range_d1[:, None] + off_D + 1 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    k_d2_1 = tl.load(p_k_d2_1, mask=mask_T, other=0.) # BLOCK_T
                    ds_1 = (tl.load(p_ds_1) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    phik_1 = k_d1 * (k_d2_1[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_1.to(K.dtype.element_ty), ds_1, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_3 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    
                    dphik_1 = tl.dot(v, tl.trans(ds_1)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    else:
                        dk_3 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 0))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_0[:, None].broadcast_to(dk_2.shape), 0.)
                    else:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 0))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_0[:, None].broadcast_to(dk_3.shape), 0.)
                    dk_d2_1 = tl.sum(dphik_1 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 1))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_1[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 1))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_1[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 1))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_1[:, None].broadcast_to(dk_2.shape), 0.)
                    else:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 1))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_1[:, None].broadcast_to(dk_3.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dk_2 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (2 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_2, dk_2, mask=mask_T[:, None])
            p_dk_3 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (3 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_3, dk_3, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
        
        elif (d == 128):     
            tl.static_assert(block1 >= block2 and block1 % block2 == 0)
            off_bh = tl.program_id(0)
            off_b = off_bh // H
            off_h = off_bh % H
            off_t = tl.program_id(1)
            off_e = tl.program_id(2)
            
            K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
            V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
            dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
            dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
            dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh
            
            range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
            range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
            range_d1 = tl.arange(0, block1)
            p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
            dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
            dk_0 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_2 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_3 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_4 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_5 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_6 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            dk_7 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
            mask_T = range_t < T
            if V_IN_REGS:
                v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
            for m in range(0, d//block1):
                p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
                k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)
            
                for n in range(0, (m+1)*block1//block2):
                    off_d2 = n*block2
                    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
                    off_d2 = tl.multiple_of(off_d2, block2)
                    off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
                    p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd # BLOCK_T
                    p_ds_0 = dS + (range_d1[:, None] + off_D + 0 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd # BLOCK_T
                    p_ds_1 = dS + (range_d1[:, None] + off_D + 1 * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
                    k_d2_0 = tl.load(p_k_d2_0, mask=mask_T, other=0.) # BLOCK_T
                    ds_0 = (tl.load(p_ds_0) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    k_d2_1 = tl.load(p_k_d2_1, mask=mask_T, other=0.) # BLOCK_T
                    ds_1 = (tl.load(p_ds_1) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
                    phik_0 = k_d1 * (k_d2_0[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_0.to(K.dtype.element_ty), ds_0, dv) # BLOCK_T x BLOCK_E
                    phik_1 = k_d1 * (k_d2_1[:, None]) # BLOCK_T x block1
                    dv = tl.dot(phik_1.to(K.dtype.element_ty), ds_1, dv) # BLOCK_T x BLOCK_E
                    
                    if not V_IN_REGS:
                        v = tl.load(p_v, mask=mask_T[:, None], other=0.)
            
                    
                    dphik_0 = tl.dot(v, tl.trans(ds_0)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 3:
                        dk_3 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 4:
                        dk_4 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 5:
                        dk_5 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    elif m == 6:
                        dk_6 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    else:
                        dk_7 += dphik_0 * k_d2_0[:, None] # BLOCK_T x block1
                    
                    dphik_1 = tl.dot(v, tl.trans(ds_1)).to(tl.float32) # BLOCK_T x block1
                    if m == 0:
                        dk_0 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 1:
                        dk_1 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 2:
                        dk_2 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 3:
                        dk_3 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 4:
                        dk_4 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 5:
                        dk_5 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    elif m == 6:
                        dk_6 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    else:
                        dk_7 += dphik_1 * k_d2_1[:, None] # BLOCK_T x block1
                    dk_d2_0 = tl.sum(dphik_0 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 0))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_0[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 0))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_0[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 0))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_0[:, None].broadcast_to(dk_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 0))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_0[:, None].broadcast_to(dk_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = ((range_d1 + 4 * block1) == (off_d2 + 0))
                        dk_4 += tl.where(mask[None, :].broadcast_to(dk_4.shape), dk_d2_0[:, None].broadcast_to(dk_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = ((range_d1 + 5 * block1) == (off_d2 + 0))
                        dk_5 += tl.where(mask[None, :].broadcast_to(dk_5.shape), dk_d2_0[:, None].broadcast_to(dk_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = ((range_d1 + 6 * block1) == (off_d2 + 0))
                        dk_6 += tl.where(mask[None, :].broadcast_to(dk_6.shape), dk_d2_0[:, None].broadcast_to(dk_6.shape), 0.)
                    else:
                        mask = ((range_d1 + 7 * block1) == (off_d2 + 0))
                        dk_7 += tl.where(mask[None, :].broadcast_to(dk_7.shape), dk_d2_0[:, None].broadcast_to(dk_7.shape), 0.)
                    dk_d2_1 = tl.sum(dphik_1 * k_d1, 1) # BLOCK_T
                    if off_d2//block1 == 0:
                        mask = ((range_d1 + 0 * block1) == (off_d2 + 1))
                        dk_0 += tl.where(mask[None, :].broadcast_to(dk_0.shape), dk_d2_1[:, None].broadcast_to(dk_0.shape), 0.)
                    elif off_d2//block1 == 1:
                        mask = ((range_d1 + 1 * block1) == (off_d2 + 1))
                        dk_1 += tl.where(mask[None, :].broadcast_to(dk_1.shape), dk_d2_1[:, None].broadcast_to(dk_1.shape), 0.)
                    elif off_d2//block1 == 2:
                        mask = ((range_d1 + 2 * block1) == (off_d2 + 1))
                        dk_2 += tl.where(mask[None, :].broadcast_to(dk_2.shape), dk_d2_1[:, None].broadcast_to(dk_2.shape), 0.)
                    elif off_d2//block1 == 3:
                        mask = ((range_d1 + 3 * block1) == (off_d2 + 1))
                        dk_3 += tl.where(mask[None, :].broadcast_to(dk_3.shape), dk_d2_1[:, None].broadcast_to(dk_3.shape), 0.)
                    elif off_d2//block1 == 4:
                        mask = ((range_d1 + 4 * block1) == (off_d2 + 1))
                        dk_4 += tl.where(mask[None, :].broadcast_to(dk_4.shape), dk_d2_1[:, None].broadcast_to(dk_4.shape), 0.)
                    elif off_d2//block1 == 5:
                        mask = ((range_d1 + 5 * block1) == (off_d2 + 1))
                        dk_5 += tl.where(mask[None, :].broadcast_to(dk_5.shape), dk_d2_1[:, None].broadcast_to(dk_5.shape), 0.)
                    elif off_d2//block1 == 6:
                        mask = ((range_d1 + 6 * block1) == (off_d2 + 1))
                        dk_6 += tl.where(mask[None, :].broadcast_to(dk_6.shape), dk_d2_1[:, None].broadcast_to(dk_6.shape), 0.)
                    else:
                        mask = ((range_d1 + 7 * block1) == (off_d2 + 1))
                        dk_7 += tl.where(mask[None, :].broadcast_to(dk_7.shape), dk_d2_1[:, None].broadcast_to(dk_7.shape), 0.)
                    
            
            
            # save dk, dv
            mask_T = range_t < T
            p_dk_0 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (0 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_0, dk_0, mask=mask_T[:, None])
            p_dk_1 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (1 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_1, dk_1, mask=mask_T[:, None])
            p_dk_2 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (2 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_2, dk_2, mask=mask_T[:, None])
            p_dk_3 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (3 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_3, dk_3, mask=mask_T[:, None])
            p_dk_4 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (4 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_4, dk_4, mask=mask_T[:, None])
            p_dk_5 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (5 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_5, dk_5, mask=mask_T[:, None])
            p_dk_6 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (6 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_6, dk_6, mask=mask_T[:, None])
            p_dk_7 = dK + range_t[:, None].to(tl.int64) * stride_dkt + (7 * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
            tl.store(p_dk_7, dk_7, mask=mask_T[:, None])
            p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
            tl.store(p_dv, dv, mask=mask_T[:, None])
                
                
    else:
        tl.static_assert(False, "No matching config found")