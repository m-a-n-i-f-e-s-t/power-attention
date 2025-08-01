import torch
from torch.utils._pytree import tree_map
import math
import torch.nn.functional as F
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim

## Useful function to create sample inputs
def create_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    if gating:
        log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    initial_state = None
    if requires_grad:
        Q, K, V, log_G, initial_state = tree_map(
            lambda x: x.clone().detach().requires_grad_(True) if x is not None else None, (Q, K, V, log_G, initial_state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                initial_state=initial_state,
                return_final_state=False,
                deg=deg, scale=scale,
                chunk_size=chunk_size)

def create_inference_inputs(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False):
    torch.manual_seed(seed)
    Q = torch.randn(size=(b, 1, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
    K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
    if gating:
        log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
    else:
        log_G = None
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    state = None if not initial_state else torch.randn(size=(b, h, D, d), dtype=dtype, device=device)
    if requires_grad:
        Q, K, V, log_G, state = tree_map(
            lambda x: x.clone().detach().requires_grad_(True) if x is not None else None, (Q, K, V, log_G, state))
    return dict(Q=Q, K=K, V=V, log_G=log_G, 
                state=state,
                deg=deg, scale=scale,
                chunk_size=chunk_size)

def input_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0):
    return dict(
        Q=((b, t, h * qhead_ratio, d), dtype, device),
        K=((b, t, h, d), dtype, device),
        V=((b, t, h, d), dtype, device)) | (dict(log_G=((b, t, h), torch.float32, device)) if gating else {})

def inference_input_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    return dict(
        Q=((b, 1, h * qhead_ratio, d), dtype, device),
        K=((b, t, h, d), dtype, device),
        V=((b, t, h, d), dtype, device)) | (dict(log_G=((b, t, h), torch.float32, device)) if gating else {}) | (dict(state=((b, h, D, d), dtype, device)) if initial_state else {})

def output_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.float16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0):
    return dict(
        Y=((b, t, h * qhead_ratio, d), dtype, device)
    )

def inference_output_properties(b=2, t=1024, h=8, d=32, qhead_ratio=1, dtype=torch.bfloat16, device='cuda', gating=False,
                  chunk_size=None, deg=2, requires_grad=False, seed=42, scale=1.0, initial_state=False):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    return dict(
        Y=((b, 1, h * qhead_ratio, d), dtype, device),
        state=((b, h, D, d), dtype, device) if (initial_state or t == chunk_size) else None
    )