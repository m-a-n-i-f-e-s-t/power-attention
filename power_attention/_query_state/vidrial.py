import torch
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.kernels.sympow_mma.op import op as sympow_mma
import math

def query_state(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile=None):
    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    scale_p = scale**deg
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile

    γ = torch.ones(n, device=Q.device, dtype=torch.float32) * math.sqrt(float(D))
    # special case for zero initial state, no need to scale by γ
    if zero_initial_state:
        γ[0] = 1.0
    γ = γ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, n, 1, 1]

    Q = Q.transpose(2, 3) # [b, n, h, c, d]
    S = (S / γ.unsqueeze(-1)).to(S.dtype) # [b, n, h, D, d]
    s = s / γ # [b, n, h, D]

    B = b * n * h
    Q = Q.reshape(B, c, d)
    S = S.reshape(B, D, d)
    s = s.reshape(B, D)

    Y_qs = sympow_mma(Q, S, power=deg, d_tile=d_tile, expand_dim=-1)
    l_qs = sympow_mma(Q.to(torch.float32), s.unsqueeze(-1).to(torch.float32), power=deg, d_tile=d_tile, expand_dim=-1).squeeze(-1)

    Y_qs = Y_qs.view(b, n, h, c, d).transpose(2, 3)# b n c h d
    l_qs = l_qs.view(b, n, h, c).transpose(2, 3) # b n c h

    alpha = torch.maximum(γ, torch.exp(rowmax)) # [b, n, c, h]
    attn_factor = torch.exp(rowmax) / alpha # [b, n, c, h]
    qs_factor = γ / alpha # [b, n, c, h]

    O = Y_attn * attn_factor.unsqueeze(-1) + Y_qs * scale_p * qs_factor.unsqueeze(-1) # [b, n, c, h, d]
    l = l_attn * attn_factor + l_qs * scale_p * qs_factor # [b, n, c, h]
    O = (O.to(torch.float32) / (l.unsqueeze(-1))).to(Y_qs.dtype)
    return O.contiguous()

