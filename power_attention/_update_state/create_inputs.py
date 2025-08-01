import torch
from power_attention._utils import compute_expanded_dim
from torch.utils._pytree import tree_map
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim

def create_inputs(b=2, n=4, c=128, h=8, d=32, deg=2, dtype=torch.float16, device='cuda', seed=42, d_tile=None, use_vidrial_layout=False, requires_grad=False):
    torch.manual_seed(seed)
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    V = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    if requires_grad:
        K, V = tree_map(lambda x: x.requires_grad_(True), (K, V))
    if not use_vidrial_layout:
        return dict(K=K, V=V, deg=deg)
    else:
        return dict(K=K, V=V, deg=deg, d_tile=d_tile)

def input_properties(b=2, n=4, c=128, h=8, d=32, deg=2, dtype=torch.float16, device='cuda', seed=42, requires_grad=False, d_tile=None, use_vidrial_layout=False):
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    return dict(
        K=((b, n, c, h, d), dtype, device),
        V=((b, n, c, h, d), dtype, device),
    )

def output_properties(b=2, n=4, c=128, h=8, d=32, deg=2, dtype=torch.float16, device='cuda', seed=42, requires_grad=False, d_tile=None, use_vidrial_layout=False, fused=False):
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    if not use_vidrial_layout:
        D = compute_expanded_dim(d, deg=deg)
    else:
        D = sympow_dim(d, deg, d_tile=d_tile)
    if not fused:
        return dict(
            S=((b, n, h, D, d), dtype, device),
            s=((b, n, h, D), torch.float32, device),
        )
    else:
        return dict(
            S=((b, n, h, D, d), dtype, device),
        )