# A query state implementation that fuses the normalizer into a dimension of the output tensor, using a custom kernel.
import torch
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma_normalized.interface import sympow_mma_normalized
import math

def _query_state(Q, S, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile=None):
    b, n, c, h, d = Q.shape
    _, _, _, D, _ = S.shape
    scale_p = scale**deg
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile

    γ = torch.ones(n, device=Q.device, dtype=torch.float32) * math.sqrt(float(D))
    # special case for zero initial state, no need to scale by γ
    if zero_initial_state:
        γ[0] = 1.0
    γ = γ.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # [1, n, 1, 1]
    S = (S / γ.unsqueeze(-1)).to(S.dtype) # [b, n, h, D, d]

    max_scale = torch.maximum(γ, torch.exp(rowmax)) # [b, n, c, h]
    alpha = scale_p * γ / max_scale
    beta = torch.exp(rowmax) / max_scale
    transpose_head = lambda x: x.transpose(2,3)
    group_batch = lambda x: x.reshape(b*n*h, *x.shape[3:])
    Q, Y_attn, alpha, beta, l_attn = map(transpose_head, [Q, Y_attn, alpha, beta, l_attn])
    Q, S, Y_attn, alpha, beta, l_attn = map(group_batch, [Q, S, Y_attn, alpha, beta, l_attn])

    return sympow_mma_normalized(Q, S, Y_attn, alpha, beta, l_attn, expand_dim=-1, power=deg, d_tile=d_tile)

query_state = torch.compiler.disable(_query_state, recursive=True)