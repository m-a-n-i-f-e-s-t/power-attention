import torch
from einops import rearrange
from power_attention._utils import dummify
from vidrial.kernels.sympow.interface import interface_reference as sympow_reference
from vidrial.py_utils.common import default_d_tile

def update_state(K, V, deg, d_tile=None):
    """Reference implementation of the chunk state forward pass, with fused normalization
    args:

        K, V: [b, n, c, h, d]
    returns:
        S: [b, n, h, D, d]
        N: [b, n, h, D]
    """
    if len(K.shape) == 4: # inference call
        K = K.unsqueeze(1) # [b, 1, c, h, d]
        V = V.unsqueeze(1) # [b, 1, c, h, d]
        S = update_state(K, V, deg, d_tile)
        return S.squeeze(1)

    b, n, c, h, d = K.shape
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    K, V = K.transpose(2, 3), V.transpose(2, 3)  # [b, n, h, c, d]

    phi_K = sympow_reference(K, deg, d_tile=d_tile, dim=-1, duplicate_correction=True)
    phi_K_T = phi_K.transpose(-1, -2) # [b, n, h, D, c]
    S = torch.matmul(phi_K_T, V)  # [b, n, h, D, d]

    return S
