import torch
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.kernels.sympow_mma.op import op as sympow_mma
import math

def query_state(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile=None):
    if len(Q.shape) == 4: # inference call
        Q = Q.unsqueeze(1) # [b, 1, c, hq, d]
        S = S.unsqueeze(1) # [b, 1, hk, D, e]
        s = s.unsqueeze(1) # [b, 1, hk, D]
        Y_attn = Y_attn.unsqueeze(1) # [b, 1, c, hq, e]
        l_attn = l_attn.unsqueeze(1) # [b, 1, c, hq]
        rowmax = rowmax.unsqueeze(1) # [b, 1, c, hq]
        Y = query_state(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile)
        return Y.squeeze(1)

    b, n, c, hq, d = Q.shape
    _, _, hk, D, e = S.shape
    assert hq == hk or hq % (8 * hk) == 0, f'hq must be equal to hk or a multiple of 8 * hk: {hq=}, {hk=}'
    scale_p = scale**deg
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile

    γ = torch.ones(n, device=Q.device, dtype=torch.float32) * math.sqrt(float(D))
    # special case for zero initial state, no need to scale by γ
    if zero_initial_state:
        γ[0] = 1.0
    γ = γ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, n, 1, 1]

    Q = Q.transpose(2, 3) # [b, n, hq, c, d]
    S = (S / γ.unsqueeze(-1)).to(S.dtype) # [b, n, hk, D, d]
    s = s / γ # [b, n, hk, D]

    B = b * n * hk
    group_ratio = hq // hk
    Q = Q.reshape(B, c * group_ratio, d)
    S = S.reshape(B, D, d)
    s = s.reshape(B, D)

    Y_qs = sympow_mma(Q, S, power=deg, d_tile=d_tile, expand_dim=-1) # B c*group_ratio e
    l_qs = sympow_mma(Q.to(torch.float32), s.unsqueeze(-1).to(torch.float32), power=deg, d_tile=d_tile, expand_dim=-1).squeeze(-1) # B c*group_ratio 

    Y_qs = Y_qs.view(b, n, hq, c, d).transpose(2, 3) # b n c hq d
    l_qs = l_qs.view(b, n, hq, c).transpose(2, 3) # b n c hq

    alpha = torch.maximum(γ, torch.exp(rowmax)) # [b, n, c, hq]
    attn_factor = torch.exp(rowmax) / alpha # [b, n, c, hq]
    qs_factor = γ / alpha # [b, n, c, hq]

    O = Y_attn * attn_factor.unsqueeze(-1) + Y_qs * scale_p * qs_factor.unsqueeze(-1) # [b, n, c, hq, e]
    l = l_attn * attn_factor + l_qs * scale_p * qs_factor # [b, n, c, hq]
    O = (O.to(torch.float32) / (l.unsqueeze(-1))).to(Y_qs.dtype)
    return O.contiguous()

