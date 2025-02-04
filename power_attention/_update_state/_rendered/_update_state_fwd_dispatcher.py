import triton
import triton.language as tl

def _update_state_fwd(K, V, S, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_vb, stride_vt, stride_vh, stride_ve,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr):
    block2: tl.constexpr = BLOCK_D // block1
    if ((BLOCK_D == 128) and ((BLOCK_E == 32) and ((BLOCK_T == 16) and ((block1 == 16))))) or (((BLOCK_D == 128) and ((BLOCK_E == 32) and ((BLOCK_T == 32) and ((block1 == 16))))) or (((BLOCK_D == 128) and ((BLOCK_E == 64) and ((BLOCK_T == 16) and ((block1 == 16))))) or ((BLOCK_D == 128) and ((BLOCK_E == 64) and ((BLOCK_T == 32) and ((block1 == 16))))))):
        
        off_bh = tl.program_id(0)
        off_b = off_bh // H
        off_h = off_bh % H
        off_D = tl.program_id(1)
        off_e = tl.program_id(2)
        off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)
        off_d1 = tl.multiple_of(off_d1, block1)
        off_d2 = tl.multiple_of(off_d2, block2)
        
        K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
        V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
        S += off_b.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh + off_D.to(tl.int64) * BLOCK_D * stride_sD
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64)
        range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
        range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
        p_k_d1 = K + range_d1[:, None] * stride_kd + range_t[None, :] * stride_kt # [block1 x BLOCK_T]
        p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve # [BLOCK_T x BLOCK_E]
        
        p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd
        s_0 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd
        s_1 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_2 = K + range_t[:] * stride_kt + (off_d2 + 2) * stride_kd
        s_2 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_3 = K + range_t[:] * stride_kt + (off_d2 + 3) * stride_kd
        s_3 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_4 = K + range_t[:] * stride_kt + (off_d2 + 4) * stride_kd
        s_4 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_5 = K + range_t[:] * stride_kt + (off_d2 + 5) * stride_kd
        s_5 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_6 = K + range_t[:] * stride_kt + (off_d2 + 6) * stride_kd
        s_6 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_7 = K + range_t[:] * stride_kt + (off_d2 + 7) * stride_kd
        s_7 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        for tid in range(0, tl.cdiv(T, BLOCK_T)):
            k_d1 = tl.load(p_k_d1) # block1 x BLOCK_T
            v = tl.load(p_v)
            k_d2_0 = tl.load(p_k_d2_0) * multiplier # BLOCK_T
            phik_0 = k_d1 * k_d2_0
            k_d2_1 = tl.load(p_k_d2_1) * multiplier # BLOCK_T
            phik_1 = k_d1 * k_d2_1
            k_d2_2 = tl.load(p_k_d2_2) * multiplier # BLOCK_T
            phik_2 = k_d1 * k_d2_2
            k_d2_3 = tl.load(p_k_d2_3) * multiplier # BLOCK_T
            phik_3 = k_d1 * k_d2_3
            k_d2_4 = tl.load(p_k_d2_4) * multiplier # BLOCK_T
            phik_4 = k_d1 * k_d2_4
            k_d2_5 = tl.load(p_k_d2_5) * multiplier # BLOCK_T
            phik_5 = k_d1 * k_d2_5
            k_d2_6 = tl.load(p_k_d2_6) * multiplier # BLOCK_T
            phik_6 = k_d1 * k_d2_6
            k_d2_7 = tl.load(p_k_d2_7) * multiplier # BLOCK_T
            phik_7 = k_d1 * k_d2_7
            s_0 = tl.dot(phik_0.to(K.dtype.element_ty), v, s_0)
            s_1 = tl.dot(phik_1.to(K.dtype.element_ty), v, s_1)
            s_2 = tl.dot(phik_2.to(K.dtype.element_ty), v, s_2)
            s_3 = tl.dot(phik_3.to(K.dtype.element_ty), v, s_3)
            s_4 = tl.dot(phik_4.to(K.dtype.element_ty), v, s_4)
            s_5 = tl.dot(phik_5.to(K.dtype.element_ty), v, s_5)
            s_6 = tl.dot(phik_6.to(K.dtype.element_ty), v, s_6)
            s_7 = tl.dot(phik_7.to(K.dtype.element_ty), v, s_7)
            p_v += BLOCK_T * stride_vt
            p_k_d1 += BLOCK_T * stride_kt
            p_k_d2_0 += BLOCK_T * stride_kt
            p_k_d2_1 += BLOCK_T * stride_kt
            p_k_d2_2 += BLOCK_T * stride_kt
            p_k_d2_3 += BLOCK_T * stride_kt
            p_k_d2_4 += BLOCK_T * stride_kt
            p_k_d2_5 += BLOCK_T * stride_kt
            p_k_d2_6 += BLOCK_T * stride_kt
            p_k_d2_7 += BLOCK_T * stride_kt
            
        
        range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
        p_s_0 = S + range_d2_0[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_0, s_0)
        range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
        p_s_1 = S + range_d2_1[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_1, s_1)
        range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
        p_s_2 = S + range_d2_2[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_2, s_2)
        range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
        p_s_3 = S + range_d2_3[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_3, s_3)
        range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
        p_s_4 = S + range_d2_4[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_4, s_4)
        range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
        p_s_5 = S + range_d2_5[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_5, s_5)
        range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
        p_s_6 = S + range_d2_6[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_6, s_6)
        range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
        p_s_7 = S + range_d2_7[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_7, s_7)
        
    elif ((BLOCK_D == 256) and ((BLOCK_E == 32) and ((BLOCK_T == 16) and ((block1 == 16))))) or (((BLOCK_D == 256) and ((BLOCK_E == 32) and ((BLOCK_T == 32) and ((block1 == 16))))) or (((BLOCK_D == 256) and ((BLOCK_E == 64) and ((BLOCK_T == 16) and ((block1 == 16))))) or ((BLOCK_D == 256) and ((BLOCK_E == 64) and ((BLOCK_T == 32) and ((block1 == 16))))))):
        
        off_bh = tl.program_id(0)
        off_b = off_bh // H
        off_h = off_bh % H
        off_D = tl.program_id(1)
        off_e = tl.program_id(2)
        off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)
        off_d1 = tl.multiple_of(off_d1, block1)
        off_d2 = tl.multiple_of(off_d2, block2)
        
        K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
        V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
        S += off_b.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh + off_D.to(tl.int64) * BLOCK_D * stride_sD
        
        range_t = tl.arange(0, BLOCK_T).to(tl.int64)
        range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
        range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
        p_k_d1 = K + range_d1[:, None] * stride_kd + range_t[None, :] * stride_kt # [block1 x BLOCK_T]
        p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve # [BLOCK_T x BLOCK_E]
        
        p_k_d2_0 = K + range_t[:] * stride_kt + (off_d2 + 0) * stride_kd
        s_0 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_1 = K + range_t[:] * stride_kt + (off_d2 + 1) * stride_kd
        s_1 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_2 = K + range_t[:] * stride_kt + (off_d2 + 2) * stride_kd
        s_2 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_3 = K + range_t[:] * stride_kt + (off_d2 + 3) * stride_kd
        s_3 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_4 = K + range_t[:] * stride_kt + (off_d2 + 4) * stride_kd
        s_4 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_5 = K + range_t[:] * stride_kt + (off_d2 + 5) * stride_kd
        s_5 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_6 = K + range_t[:] * stride_kt + (off_d2 + 6) * stride_kd
        s_6 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_7 = K + range_t[:] * stride_kt + (off_d2 + 7) * stride_kd
        s_7 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_8 = K + range_t[:] * stride_kt + (off_d2 + 8) * stride_kd
        s_8 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_9 = K + range_t[:] * stride_kt + (off_d2 + 9) * stride_kd
        s_9 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_10 = K + range_t[:] * stride_kt + (off_d2 + 10) * stride_kd
        s_10 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_11 = K + range_t[:] * stride_kt + (off_d2 + 11) * stride_kd
        s_11 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_12 = K + range_t[:] * stride_kt + (off_d2 + 12) * stride_kd
        s_12 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_13 = K + range_t[:] * stride_kt + (off_d2 + 13) * stride_kd
        s_13 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_14 = K + range_t[:] * stride_kt + (off_d2 + 14) * stride_kd
        s_14 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        p_k_d2_15 = K + range_t[:] * stride_kt + (off_d2 + 15) * stride_kd
        s_15 = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
        for tid in range(0, tl.cdiv(T, BLOCK_T)):
            k_d1 = tl.load(p_k_d1) # block1 x BLOCK_T
            v = tl.load(p_v)
            k_d2_0 = tl.load(p_k_d2_0) * multiplier # BLOCK_T
            phik_0 = k_d1 * k_d2_0
            k_d2_1 = tl.load(p_k_d2_1) * multiplier # BLOCK_T
            phik_1 = k_d1 * k_d2_1
            k_d2_2 = tl.load(p_k_d2_2) * multiplier # BLOCK_T
            phik_2 = k_d1 * k_d2_2
            k_d2_3 = tl.load(p_k_d2_3) * multiplier # BLOCK_T
            phik_3 = k_d1 * k_d2_3
            k_d2_4 = tl.load(p_k_d2_4) * multiplier # BLOCK_T
            phik_4 = k_d1 * k_d2_4
            k_d2_5 = tl.load(p_k_d2_5) * multiplier # BLOCK_T
            phik_5 = k_d1 * k_d2_5
            k_d2_6 = tl.load(p_k_d2_6) * multiplier # BLOCK_T
            phik_6 = k_d1 * k_d2_6
            k_d2_7 = tl.load(p_k_d2_7) * multiplier # BLOCK_T
            phik_7 = k_d1 * k_d2_7
            k_d2_8 = tl.load(p_k_d2_8) * multiplier # BLOCK_T
            phik_8 = k_d1 * k_d2_8
            k_d2_9 = tl.load(p_k_d2_9) * multiplier # BLOCK_T
            phik_9 = k_d1 * k_d2_9
            k_d2_10 = tl.load(p_k_d2_10) * multiplier # BLOCK_T
            phik_10 = k_d1 * k_d2_10
            k_d2_11 = tl.load(p_k_d2_11) * multiplier # BLOCK_T
            phik_11 = k_d1 * k_d2_11
            k_d2_12 = tl.load(p_k_d2_12) * multiplier # BLOCK_T
            phik_12 = k_d1 * k_d2_12
            k_d2_13 = tl.load(p_k_d2_13) * multiplier # BLOCK_T
            phik_13 = k_d1 * k_d2_13
            k_d2_14 = tl.load(p_k_d2_14) * multiplier # BLOCK_T
            phik_14 = k_d1 * k_d2_14
            k_d2_15 = tl.load(p_k_d2_15) * multiplier # BLOCK_T
            phik_15 = k_d1 * k_d2_15
            s_0 = tl.dot(phik_0.to(K.dtype.element_ty), v, s_0)
            s_1 = tl.dot(phik_1.to(K.dtype.element_ty), v, s_1)
            s_2 = tl.dot(phik_2.to(K.dtype.element_ty), v, s_2)
            s_3 = tl.dot(phik_3.to(K.dtype.element_ty), v, s_3)
            s_4 = tl.dot(phik_4.to(K.dtype.element_ty), v, s_4)
            s_5 = tl.dot(phik_5.to(K.dtype.element_ty), v, s_5)
            s_6 = tl.dot(phik_6.to(K.dtype.element_ty), v, s_6)
            s_7 = tl.dot(phik_7.to(K.dtype.element_ty), v, s_7)
            s_8 = tl.dot(phik_8.to(K.dtype.element_ty), v, s_8)
            s_9 = tl.dot(phik_9.to(K.dtype.element_ty), v, s_9)
            s_10 = tl.dot(phik_10.to(K.dtype.element_ty), v, s_10)
            s_11 = tl.dot(phik_11.to(K.dtype.element_ty), v, s_11)
            s_12 = tl.dot(phik_12.to(K.dtype.element_ty), v, s_12)
            s_13 = tl.dot(phik_13.to(K.dtype.element_ty), v, s_13)
            s_14 = tl.dot(phik_14.to(K.dtype.element_ty), v, s_14)
            s_15 = tl.dot(phik_15.to(K.dtype.element_ty), v, s_15)
            p_v += BLOCK_T * stride_vt
            p_k_d1 += BLOCK_T * stride_kt
            p_k_d2_0 += BLOCK_T * stride_kt
            p_k_d2_1 += BLOCK_T * stride_kt
            p_k_d2_2 += BLOCK_T * stride_kt
            p_k_d2_3 += BLOCK_T * stride_kt
            p_k_d2_4 += BLOCK_T * stride_kt
            p_k_d2_5 += BLOCK_T * stride_kt
            p_k_d2_6 += BLOCK_T * stride_kt
            p_k_d2_7 += BLOCK_T * stride_kt
            p_k_d2_8 += BLOCK_T * stride_kt
            p_k_d2_9 += BLOCK_T * stride_kt
            p_k_d2_10 += BLOCK_T * stride_kt
            p_k_d2_11 += BLOCK_T * stride_kt
            p_k_d2_12 += BLOCK_T * stride_kt
            p_k_d2_13 += BLOCK_T * stride_kt
            p_k_d2_14 += BLOCK_T * stride_kt
            p_k_d2_15 += BLOCK_T * stride_kt
            
        
        range_d2_0 = tl.arange(0, block1).to(tl.int64) + 0 * block1
        p_s_0 = S + range_d2_0[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_0, s_0)
        range_d2_1 = tl.arange(0, block1).to(tl.int64) + 1 * block1
        p_s_1 = S + range_d2_1[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_1, s_1)
        range_d2_2 = tl.arange(0, block1).to(tl.int64) + 2 * block1
        p_s_2 = S + range_d2_2[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_2, s_2)
        range_d2_3 = tl.arange(0, block1).to(tl.int64) + 3 * block1
        p_s_3 = S + range_d2_3[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_3, s_3)
        range_d2_4 = tl.arange(0, block1).to(tl.int64) + 4 * block1
        p_s_4 = S + range_d2_4[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_4, s_4)
        range_d2_5 = tl.arange(0, block1).to(tl.int64) + 5 * block1
        p_s_5 = S + range_d2_5[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_5, s_5)
        range_d2_6 = tl.arange(0, block1).to(tl.int64) + 6 * block1
        p_s_6 = S + range_d2_6[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_6, s_6)
        range_d2_7 = tl.arange(0, block1).to(tl.int64) + 7 * block1
        p_s_7 = S + range_d2_7[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_7, s_7)
        range_d2_8 = tl.arange(0, block1).to(tl.int64) + 8 * block1
        p_s_8 = S + range_d2_8[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_8, s_8)
        range_d2_9 = tl.arange(0, block1).to(tl.int64) + 9 * block1
        p_s_9 = S + range_d2_9[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_9, s_9)
        range_d2_10 = tl.arange(0, block1).to(tl.int64) + 10 * block1
        p_s_10 = S + range_d2_10[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_10, s_10)
        range_d2_11 = tl.arange(0, block1).to(tl.int64) + 11 * block1
        p_s_11 = S + range_d2_11[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_11, s_11)
        range_d2_12 = tl.arange(0, block1).to(tl.int64) + 12 * block1
        p_s_12 = S + range_d2_12[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_12, s_12)
        range_d2_13 = tl.arange(0, block1).to(tl.int64) + 13 * block1
        p_s_13 = S + range_d2_13[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_13, s_13)
        range_d2_14 = tl.arange(0, block1).to(tl.int64) + 14 * block1
        p_s_14 = S + range_d2_14[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_14, s_14)
        range_d2_15 = tl.arange(0, block1).to(tl.int64) + 15 * block1
        p_s_15 = S + range_d2_15[:, None] * stride_sD + range_e[None, :] * stride_se
        tl.store(p_s_15, s_15)
        
    else:
        tl.static_assert(False, "No matching config found")