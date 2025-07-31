import torch
import sys
from pathlib import Path

from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.kernels.sympow_mma.op import op as sympow_mma

def update_state(K, V, deg, d_tile=None):
    b, n, c, h, d = K.shape
    B = b * n * h
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    K = K.transpose(2, 3).contiguous().view(B, c, d)
    V = V.transpose(2, 3).contiguous().view(B, c, d)
    K = K.transpose(1, 2)
    D = sympow_dim(d, deg, d_tile=d_tile)

    S = sympow_mma(K, V, power=deg, expand_dim=-2, d_tile=d_tile)
    s = sympow_mma(K.to(torch.float32), torch.ones((B, c, 1), device=K.device, dtype=torch.float32), power=deg, expand_dim=-2, d_tile=d_tile)

    return S.view(b, n, h, D, d), s.view(b, n, h, D)


def default_D(d, deg, d_tile=None):
    return sympow_dim(d, deg, d_tile=default_d_tile(d, deg))