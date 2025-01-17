import code
from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange
from power_attention_cuda import attention_bwd, attention_fwd
from torch.utils._pytree import tree_map
from types import NoneType
from typing import Optional, Tuple
from power_attention._update_state import ExpandedDim
from power_attention._utils import dummify
from power_attention._attention.impl import attention


class AttentionReference(torch.autograd.Function):
    @staticmethod
    def _softmax(S, scale, deg, log_G, ε, flash_equivalent, normal_space, dtype, device):
        """Calculate different flavors of "softmax" based on QK product.
        """
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=device))
        if flash_equivalent or normal_space:
            T = S
            if scale is not None:
                T *= scale if flash_equivalent else scale ** (1/deg)
            if normal_space:
                if log_G is not None:
                    Z = T * torch.exp((log_G[..., None] - log_G[..., None, :]) / deg)
                else:
                    Z = T
                Z_masked = torch.where(mask, Z, 0.0)
                Z_rowmax = torch.max(torch.abs(Z_masked), dim=-1, keepdim=True)[0].detach() # [b, h, t, 1]
                Z_rowmax.requires_grad = True
                Z_scaled = Z_masked / Z_rowmax
                P = (Z_scaled ** deg).to(dtype)
        else:
            signs = torch.sign(S)
            T = deg * torch.log(torch.abs(S) + ε)
            if scale is not None:
                T += torch.log(scale)
            if log_G is not None:
                Z = T + (log_G[..., None] - log_G[..., None, :])
            else:
                Z = T
            Z_masked = torch.where(mask, Z, -torch.inf)
            Z_rowmax = torch.max(Z_masked, dim=-1, keepdim=True)[0].detach() # [b, h, t, 1]
            Z_rowmax.requires_grad = True
            Z_scaled = Z_masked - Z_rowmax
            if deg % 2 == 1:
                P = (torch.exp(Z_scaled) * signs).to(dtype) # [b, h, t, t]
            else:
                P = torch.exp(Z_scaled).to(dtype)
        return P, Z_rowmax

    @staticmethod
    def forward(ctx, Q, K, V, log_G, deg, scale, ε, deterministic, normalize_output=False, flash_equivalent=False, normal_space=False):
        """Reference implementation of the attention forward pass
        args:
            Q, K, V: [b, t, h, d], query, key, value
            log_G: [b, t, h] or None, log of cumulative gating factor
            deg: int, degree of power
            scale: float or None, scale of key-query inner product
            ε: float
            deterministic: bool
            normalize_output: bool, whether to normalize the output by the sum of the attention weights, defaults to False
            flash_equivalent: bool, whether to use flash_equivalent for the attention operation, defaults to False
            normal_space: bool, whether to do computation in normal space instead of log space, this helps speed up the kernel but potentially become less stable.
        returns:
            Y: [b, t, h, d]
            y: [b, t, h]
        """
        device = Q.device
        b, t, h, d = Q.shape
        D = ExpandedDim(d, deg)
        if isinstance(scale, float):
            scale = torch.tensor(scale, dtype=torch.float32, device=device)

        assert not (normal_space and flash_equivalent), "Normal space and flash equivalent cannot be both True"

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2) # [b, h, t, d]
        log_G = log_G.transpose(1, 2) if log_G is not None else None # [b, h, t]
        S = torch.matmul(Q, K.transpose(-1, -2))
        mask = torch.tril(torch.full(S.shape[-2:], True, dtype=torch.bool, device=device))
        P, Z_rowmax = AttentionReference._softmax(S, scale, deg, log_G, ε, flash_equivalent, normal_space, Q.dtype, Q.device)

        y = P.sum(-1, dtype=torch.float32) # [b, h, t]

        if normalize_output:
            O = (P / y.unsqueeze(-1)).to(Q.dtype) @ V # [b, h, t, d]
            y = y.detach()
            y.requires_grad = True
        else:
            O = torch.matmul(P, V) # [b, h, t, d]

        ctx.save_for_backward(Q, K, V, log_G, O)

        O = O.transpose(1, 2).contiguous() # [b, t, h, d]
        y = y.transpose(1, 2).contiguous() # [b, t, h]
        ctx.scale = scale
        ctx.ε = ε
        ctx.deg = deg
        ctx.b = b
        ctx.gating = log_G is not None
        ctx.normalize_output = normalize_output
        ctx.flash_equivalent = flash_equivalent
        ctx.normal_space = normal_space
        return O, y, Z_rowmax.squeeze(-1).transpose(1, 2).contiguous()  # [b, t, h]

    @staticmethod
    def backward_impl(ctx, dO, dy, dZ_rowmax):
        """Reference implementation of the attention backward pass
        args:
            Q, K, V: [b, t, h, d]
            dO: [b, t, h, d]
            dy: [b, t, h]
            dZ_rowmax: [b, t, h]
        returns:
            dQ: [b, t, h, d]
            dK: [b, t, h, d]
            dV: [b, t, h, d]
            dS: [b, h, t, t]
        """
        dO, dy = dO.transpose(1, 2), dy.transpose(1, 2) # [b, h, t, d] and [b, h, t]
        Q, K, V, log_G, O = ctx.saved_tensors # [b, h, t, d]

        # recompute forward
        assert not (ctx.normal_space and ctx.flash_equivalent), "Normal space and flash equivalent cannot be both True"

        S = torch.matmul(Q, K.transpose(-1, -2))
        P, _ = AttentionReference._softmax(S.clone(), ctx.scale, ctx.deg, log_G, ctx.ε, ctx.flash_equivalent, ctx.normal_space, Q.dtype, Q.device)

        # compute backward

        if ctx.normalize_output:
            # Y = torch.matmul(P, V) # [b, t, h, d]
            y = P.sum(-1, dtype=torch.float32) # [b, h, t]
            dP_adj = dO.to(Q.dtype) @ V.transpose(-1, -2)
            dy = - (dP_adj * P).sum(-1) / y**2 
            dV = (P / y.unsqueeze(-1)).to(Q.dtype).transpose(-1, -2) @ dO # [b, h, t, d]
            dP = dP_adj / y.unsqueeze(-1) + dy.unsqueeze(-1)

        else:
            dV = P.transpose(-1, -2).to(Q.dtype) @ dO  # [b, h, t, d]
            dP = torch.matmul(dO, V.transpose(-1, -2))
            dP += dy[..., None]

        dZ = P * dP  # [b, h, t, t]
        if not ctx.flash_equivalent:
            dS = (dZ * ctx.deg / (torch.abs(S) + ctx.ε) * torch.sign(S)).to(Q.dtype)
        else:
            dS = dZ * ctx.scale
        dQ = dS @ K  # [b, h, t, d]
        dK = dS.transpose(-1, -2) @ Q  # [b, h, t, d]

        if ctx.gating:
            dlog_G_Q = dZ.sum(-1)
            dlog_G_K = -dZ.sum(-2)
            dlog_G = dlog_G_Q + dlog_G_K
            dlog_G = dlog_G.transpose(1, 2) # [b, t, h]
        else:
            dlog_G = None

        dQ, dK, dV = dQ.transpose(1, 2), dK.transpose(1, 2), dV.transpose(1, 2) # [b, t, h, d]
        # return dQ, dK, dV, dlog_G, None, None, None, None, None, dP, dZ, dS, dy
        return dQ, dK, dV, dlog_G, None, None, None, None, None
    
    @staticmethod
    def backward(ctx, *args):
        dQ, dK, dV, dlog_G, _, _, _, _, _ = AttentionReference.backward_impl(ctx, *args)
        return dQ, dK, dV, dlog_G, None, None, None, None, None, None, None

attention_reference = AttentionReference.apply
attention_reference_fwd = dummify(AttentionReference.forward)
attention_reference_fwd_impl = AttentionReference.forward
attention_reference_bwd_impl = AttentionReference.backward_impl


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
        


flash_attention_reference = FlashAttentionReference.apply
flash_attention_reference_fwd = dummify(FlashAttentionReference.forward)


class JacobAttentionLayer(torch.nn.Module):
    def __init__(self, n_head, n_embd, gating, dtype, device='cuda'):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.gating = gating
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd + n_head, bias=False, device=device, dtype=dtype)

    def create_inputs(self, B, T, dtype, device='cuda', seed=42):
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((B, T, self.n_embd), dtype=dtype, device=device, requires_grad=True, generator=generator)
        return (x,)

    def forward(self, x, attention_kernel='p2'):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        nh = self.n_head
        hs = C // nh

        self.bias = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkvgr = self.c_attn(x)
        qkv = qkvgr[...,:3*self.n_embd]
        if self.gating:
            log_g = torch.nn.functional.logsigmoid(qkvgr[...,3*self.n_embd:3*self.n_embd+self.n_head].to(dtype=torch.float32)).contiguous()
        else:
            log_g = None
        q, k, v  = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, nh, hs).contiguous()
        k = k.view(B, T, nh, hs).contiguous()
        v = v.view(B, T, nh, hs).contiguous()

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if attention_kernel == 'sdpa': pass
        elif attention_kernel == 'flash': pass
        elif attention_kernel == 'p2':
            if self.gating: log_g = torch.cumsum(log_g, dim=1)
            y_unnormalized, norm, _ = attention(q, k, v, log_g,
                                2,
                                1.0 / hs**0.5,
                                1e-7,
                                False,
                                False,
                                False)
            y = (y_unnormalized / norm.unsqueeze(-1)).to(dtype=v.dtype)
        elif attention_kernel == 'powerref':
            if self.gating: log_g = torch.cumsum(log_g, dim=1)
            y_unnormalized, norm, _ = attention_reference(q, k, v, log_g,
                                                            2,
                                                            1.0 / hs**0.5,
                                                            1e-7,
                                                            False,
                                                            False)
            y = (y_unnormalized / norm.unsqueeze(-1)).to(dtype=v.dtype)
        elif attention_kernel in {'expref', 'p2ref', 'p4ref'}:
            q = q.transpose(1, 2) # (B, nh, T, hs)
            k = k.transpose(1, 2) # (B, nh, T, hs)
            v = v.transpose(1, 2) # (B, nh, T, hs)
            if self.gating: log_g = log_g.cumsum(dim=1).transpose(1, 2) # (B, nh, T)
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / hs**0.5)
            if attention_kernel == 'p2ref': att = 2. * torch.log(torch.abs(att) + 1e-7)
            if attention_kernel == 'p4ref': att = 4. * torch.log(torch.abs(att) + 1e-7)
            if self.gating: att += log_g[..., None] - log_g[..., None, :]
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            # att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2) # (B, T, nh, hs)
        else:
            raise NotImplementedError(f'Unknown attention kernel: {attention_kernel}')
        y = y.contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        return y


def create_inputs_flash(b, t, h, d, dtype, device, softmax_scale=0.0, requires_grad=False, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    Q = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    K = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    V = torch.randn(b, t, h, d, dtype=dtype, device=device, requires_grad=requires_grad, generator=generator)
    return Q, K, V, softmax_scale