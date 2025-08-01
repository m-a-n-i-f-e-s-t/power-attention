import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

def create_inputs(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', scale=1.0, deg=2, causal=True, gating=False, head_first=False, requires_grad=False, seed=42, std=1.0, norm=True, use_log2=False, inference=False, qhead_ratio=1):
    torch.manual_seed(seed)
    hq = h * qhead_ratio
    hk = h
    tq = 1 if inference else t
    q_shape_base = (b, hq, tq) if head_first else (b, tq, hq)
    k_shape_base = (b, hk, t) if head_first else (b, t, hk)
    Q = torch.randn(size=(*q_shape_base, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(*k_shape_base, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(*k_shape_base, d), dtype=dtype, device=device) * std
    log_G = F.logsigmoid(torch.ones(size=k_shape_base, dtype=torch.float32, device=device)) if gating else None
    if requires_grad:
        Q, K, V, log_G = tree_map(lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, causal=causal, head_first=head_first, scale=scale, norm=norm, use_log2=use_log2)

def create_inputs_cuda(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', scale=1.0, deg=2, causal=True, gating=False, requires_grad=False, seed=42, std=1.0, norm=True, use_log2=False):
    torch.manual_seed(seed)
    shape_base = (b, t, h)
    Q = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    K = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    V = torch.randn(size=(*shape_base, d), dtype=dtype, device=device) * std
    log_G = F.logsigmoid(torch.rand(size=shape_base, dtype=torch.float32, device=device)) if gating else None
    if requires_grad:
        Q, K, V, log_G = tree_map(lambda x: x.requires_grad_(True) if x is not None else None, (Q, K, V, log_G))
    return dict(Q=Q, K=K, V=V, log_G=log_G, deg=deg, causal=causal, scale=scale, head_first=False, norm=norm, use_log2=use_log2)

def input_properties(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', gating=False, head_first=False, inference=False, qhead_ratio=1, **kwargs):
    hq = h * qhead_ratio
    tq = 1 if inference else t
    q_shape_base = (b, hq, tq) if head_first else (b, tq, hq)
    k_shape_base = (b, h, t) if head_first else (b, t, h)
    return dict(
        Q=((*q_shape_base, d), dtype, device), 
        K=((*k_shape_base, d), dtype, device), 
        V=((*k_shape_base, d), dtype, device), 
    ) | ({'log_G': (k_shape_base, torch.float32, device)} if gating else {})

def output_properties(b=2, t=32, h=8, d=32, dtype=torch.float16, device='cuda', head_first=False, inference=False, qhead_ratio=1, norm=True, **kwargs):
    hq = h * qhead_ratio
    tq = 1 if inference else t
    q_shape_base = (b, hq, tq) if head_first else (b, tq, hq)
    if norm:
        return dict(
            Y=((*q_shape_base, d), dtype, device),
        )
    else:
        return dict(
            Y=((*q_shape_base, d), dtype, device),
            l=(q_shape_base, torch.float32, device),
            rowmax=(q_shape_base, torch.float32, device)
        )