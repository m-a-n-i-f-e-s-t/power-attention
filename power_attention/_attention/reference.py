import torch
import torch.nn.functional as F
from power_attention._utils import dummify
import math
flash_equivalent = False
normal_space = False
normalize_output = False
ε = 1e-6

def attention(Q, K, V, log_G, deg, causal=True, head_first=False, scale=1.0, norm=False, use_log2=False):
    r = Q.shape[2] // K.shape[2]
    w = 1
    if head_first:
        b, hq, ctx_q, d, hk, ctx_k, e = *Q.shape, K.shape[1], K.shape[2], V.shape[-1]
    else:
        b, ctx_q, hq, d, ctx_k, hk, e = *Q.shape, K.shape[1], K.shape[2], V.shape[-1]
    assert hq % r == 0, "hq must be divisible by r"
    assert hk % w == 0, "hk must be divisible by w"
    assert hq // r == hk // w, "hq // r must be equal to hk // w"
    assert isinstance(deg, int) and deg % 2 == 0, "deg must be a positive even integer"
    h = hq // r
    log_GK = log_G
    if log_GK is not None:
        if head_first:
            assert log_GK.shape == (b, h, ctx_k)
        else:
            assert log_GK.shape == (b, ctx_k, h)
            log_GK = log_GK.transpose(1, 2) # (b, h, ctx_k)
        log_GQ = log_GK[..., -ctx_q:]
    if head_first:
        Q = Q.view(b, h, r *ctx_q, d)
        K = K.view(b, h, w * ctx_k, d)
        V = V.view(b, h, w * ctx_k, e)
    else:
        Q = Q.view(b, ctx_q * r, h, d).transpose(1, 2)
        K = K.view(b, ctx_k * w, h, d).transpose(1, 2)
        V = V.view(b, ctx_k * w, h, e).transpose(1, 2)
    
    exp = torch.exp if not use_log2 else torch.exp2
    log = torch.log if not use_log2 else torch.log2

    _qidx = torch.arange(ctx_q * r, device=Q.device).unsqueeze(1)
    _kidx = torch.arange(ctx_k * w, device=K.device).unsqueeze(0)
    s = torch.matmul(Q, K.transpose(2,3)) * scale
    m = (_qidx // r + ctx_k - ctx_q) >= (_kidx // w) if causal else torch.ones_like(s, dtype=torch.bool)
    signs = torch.sign(s)
    s = float(deg) * torch.where(m, log(s.abs() + 1e-7), -float("inf"))
    if log_GK is not None:
        s = s + (log_GQ.repeat_interleave(r, dim=2)[..., :, None] - log_GK.repeat_interleave(w, dim=2)[..., None, :])
    rowmax = torch.max(s, dim=-1, keepdim=True).values.detach()
    if deg % 2 == 0:
        p = exp(s - rowmax).to(V.dtype)
    else:
        p = exp(s - rowmax).to(V.dtype) * signs
    l = (torch.sum(p, dim=-1).to(torch.float32) + 1e-6) # [b, h, ctx_q]
    o = torch.matmul(p, V) # [b, h, ctx_q, e]
    rowmax = rowmax # [b, h, ctx_q]

    if head_first:
        l = l.view(b, hq, ctx_q)
        o = o.view(b, hq, ctx_q, e)
        rowmax = rowmax.view(b, hq, ctx_q)
    else:
        l = l.view(b, h, ctx_q, r).permute(0, 2, 3, 1).reshape(b, ctx_q, hq)
        o = o.view(b, h, ctx_q, r, e).permute(0, 2, 3, 1, 4).reshape(b, ctx_q, hq, e)
        rowmax = rowmax.view(b, h, ctx_q, r).permute(0, 2, 3, 1).reshape(b, ctx_q, hq)
    if norm:
        return (o / l[..., None]).to(V.dtype)
    return o, l, rowmax.to(torch.float32)

attention_fwd = attention





## This is useful sometimes but not used at the moment

class FlashAttentionReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale=0.0):
        """Reference implementation of the attention forward pass
        args:
            Q, K, V: [b, t, h, d]
            ε: float
        returns:
            Y: [b, t, h, d]
        """
        device = Q.device
        b, t, h, d = Q.shape

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # [b, h, t, d]
        S = torch.matmul(Q, K.transpose(-1, -2)) * torch.tensor(softmax_scale, dtype=torch.float32, device=device)
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=device))
        S = torch.where(mask, S, -torch.inf)
        R = S.max(dim=-1, keepdim=True)[0]
        P = F.softmax(S - R, dim=-1)
        Y = P.to(Q.dtype) @ V

        ctx.save_for_backward(Q, K, V)
        ctx.softmax_scale = softmax_scale
        return Y.transpose(1, 2).contiguous() # [b, t, h, d]

    @staticmethod
    def backward(ctx, dY):
        """Reference implementation of the attention backward pass
        args:
            Q, K, V: [b, t, h, d]
            dY: [b, t, h, d]
        returns:
            dQ: [b, t, h, d]
            dK: [b, t, h, d]
            dV: [b, t, h, d]
            dS: [b, h, t, t]
        """
        Q, K, V = ctx.saved_tensors
        dY = dY.transpose(1, 2) # [b, h, t, d]

        # recompute forward
        S = torch.matmul(Q, K.transpose(-1, -2)) * torch.tensor(ctx.softmax_scale, dtype=Q.dtype, device=Q.device)
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=Q.device))
        S = torch.where(mask, S, -torch.inf)
        R = S.max(dim=-1, keepdim=True)[0]
        P = torch.exp(S - R)
        y = P.sum(-1, dtype=torch.float32) # [b, h, t]

        # compute backward
        dP_adj = dY.to(Q.dtype) @ V.transpose(-1, -2)
        dy = - (dP_adj * P).sum(-1) / y**2 
        dV = (P / y.unsqueeze(-1)).to(Q.dtype).transpose(-1, -2) @ dY # [b, h, t, d]
        dP = dP_adj / y.unsqueeze(-1) + dy.unsqueeze(-1)
        dS = dP * P * ctx.softmax_scale

        dQ = dS.to(Q.dtype) @ K  # [b, h, t, d]
        dK = dS.transpose(-1, -2).to(Q.dtype) @ Q  # [b, h, t, d]
        dQ, dK, dV = dQ.transpose(1, 2), dK.transpose(1, 2), dV.transpose(1, 2) # [b, t, h, d]
        return dQ, dK, dV, None
        


def flash_attention_reference(*args, **kwargs):
    if args and kwargs:
        raise ValueError("Cannot pass both args and kwargs")
    if kwargs:
        args = (kwargs['Q'], kwargs['K'], kwargs['V'], kwargs['softmax_scale'])
    return FlashAttentionReference.apply(*args)
flash_attention_reference_fwd = dummify(FlashAttentionReference.forward)


def create_inputs_flash(b, t, h, d, dtype, device, softmax_scale=0.0, requires_grad=False, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    K = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    V = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    return Q, K, V, softmax_scale