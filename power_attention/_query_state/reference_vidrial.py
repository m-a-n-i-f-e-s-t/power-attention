import torch
import math
from einops import rearrange
from power_attention._utils import dummify, compute_expanded_dim, diff
from vidrial.kernels.sympow.interface import interface_reference as sympow_reference
from vidrial.py_utils.common import default_d_tile

def query_state(Q, S, s, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile=None):
    """Compute query state output. It computes the following equation:

            ( (Q @ (S/γ)) * (scale^p) * γ/α + Y_attn * exp(rowmax)/α )
        O = -------------------------------------------------------------
            ( (Q @ (s/γ)) * (scale^p) * γ/α + l_attn * exp(rowmax)/α )
        where α = max(γ, exp(rowmax))
                γ = 1.0 if S == 0 else sqrt(D)

    args:
        Q: [b, n, c, h, d] - query
        S: [b, n, h, D, d] - state
        s: [b, n, h, D] - sum of keys
        Y_attn: [b, n, c, h, d] - attention output
        l_attn: [b, n, c, h] - sum of powered attention scores, in normal space
        rowmax: [b, n, c, h] - max of powered attention scores, in log space
        deg: int - degree of power
        scale: float - sm_scale used in attention
        zero_initial_state: bool - whether the initial state is zero
    returns:
        O: [b, n, c, h, d] - temporal-normalized output
    """

    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    scale_p = scale**deg
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile

    γ = torch.ones(n, device=Q.device, dtype=torch.float32) * math.sqrt(float(D))
    # special case for zero initial state, no need to scale by γ
    if zero_initial_state:
        γ[0] = 1.0
    γ = γ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, n, 1, 1]

    Q = rearrange(Q, 'b n c h d -> b n h c d') # [b, n, h, c, d]
    phi_Q = sympow_reference(Q, deg, d_tile=d_tile, dim=-1, duplicate_correction=True) # [b, n, h, c, D]
    S = (S / γ.unsqueeze(-1)).to(S.dtype) # [b, n, h, D, d]
    s = s / γ # [b, n, h, D]

    Y_qs = torch.matmul(phi_Q.to(Q.dtype), S.to(Q.dtype)).to(Q.dtype)  # [b n h c d]
    l_qs = torch.matmul(phi_Q.to(torch.float32), s.unsqueeze(-1).to(torch.float32)).squeeze(-1) # [b n h c]
    Y_qs = rearrange(Y_qs, 'b n h c d -> b n c h d')
    l_qs = rearrange(l_qs, 'b n h c -> b n c h')

    alpha = torch.maximum(γ, torch.exp(rowmax)) # [b, n, c, h]
    attn_factor = torch.exp(rowmax) / alpha # [b, n, c, h]
    qs_factor = γ / alpha # [b, n, c, h]

    O = Y_attn * attn_factor.unsqueeze(-1) + Y_qs * scale_p * qs_factor.unsqueeze(-1) # [b, n, c, h, d]
    l = l_attn * attn_factor + l_qs * scale_p * qs_factor # [b, n, c, h]
    O = (O.to(torch.float32) / (l.unsqueeze(-1))).to(Y_qs.dtype)

    return O.contiguous()
