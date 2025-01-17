from functools import partial
import os
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from power_attention._utils import dummify, print_tensor

InnerBlock = 16
OuterBlock = 1

class QueryStateReference(torch.autograd.Function):
    @staticmethod
    def expand(Q, deg):
        """ Reference implementation of key expansion
        args:
            Q: [b, n, h, c d]
            deg: int
        returns:
            phi_Q: [b, n, h, c D]
        """
        b, n, h, c, d = Q.shape
        Q_outer = rearrange(Q, 'b n h c (x o) -> b n h c x o', o=OuterBlock)
        Q_inner = rearrange(Q, 'b n h c (y i) -> b n h c y i', i=InnerBlock)
        phi_Q_unmasked = torch.einsum('bnhcxo,bnhcyi->bnhcxyoi', Q_outer, Q_inner).to(Q.dtype)
        _, _, _, _, x, y, o, i = phi_Q_unmasked.shape
        phi_Q_shape = (b, n, h, c, int((InnerBlock // OuterBlock + x) * y // 2), o, i)
        phi_Q = torch.empty(phi_Q_shape, device=Q.device, dtype=Q.dtype)
        idx = 0
        for y_idx in range(y):
            for x_idx in range(x):
                if (x_idx * OuterBlock) < (y_idx + 1) * InnerBlock:
                    phi_Q[:, :, :, :, idx, :, :] = phi_Q_unmasked[:, :, :, :, x_idx, y_idx, :, :]
                    idx += 1
        phi_Q = rearrange(phi_Q, 'b n h c k o i -> b n h c (k o i)') # [b, n, h, c, D]
        return phi_Q.to(Q.dtype)

    @staticmethod
    def scale_combine_outputs(Y, att_Y, stabilizer, rowmax, zero_initial_state, eps):
        """ Scale the outputs from attention and query state and combine them to prevent overflow
        """
        b, n, c, h, d = Y.shape
        dtype = Y.dtype
        scale_qs = torch.tensor(1, dtype=Y.dtype, device=Y.device) if stabilizer is None else 1 / stabilizer
        scale_attn = torch.exp(-rowmax)
        qs_factor = scale_attn / scale_qs
        if not zero_initial_state:
            Y = att_Y + Y * qs_factor.unsqueeze(-1)
        else:
            Y[:, 0:1] = att_Y.narrow(1, 0, 1)
            Y[:, 1:] = att_Y.narrow(1, 1, n - 1) + Y.narrow(1, 1, n - 1) * qs_factor.narrow(1, 1, n - 1).unsqueeze(-1)
        return Y.to(dtype)
    
    @staticmethod
    def scale_combine_gradients(dY, stabilizer, rowmax, zero_initial_state):
        """ Scale the gradient from output to gradients for attention
        """
        b, n, c, h, d = dY.shape
        scale_qs = torch.tensor(1, dtype=dY.dtype, device=dY.device) if stabilizer is None else 1 / stabilizer
        scale_attn = torch.exp(-rowmax)
        qs_factor = scale_attn / scale_qs
        if not zero_initial_state:
            dY_attn = dY.clone()
            dY_qs = (dY * qs_factor.unsqueeze(-1)).to(dY.dtype)
        else:
            dY_attn = dY.clone()
            dY_qs = torch.empty_like(dY)
            dY_qs[:, 0] = 0
            dY_qs[:, 1:] = (dY[:, 1:].clone() * qs_factor.narrow(1, 1, n - 1).unsqueeze(-1))
        return dY_attn, dY_qs

    @staticmethod
    def forward(ctx, Q, S, Y, rowmax, deg, stabilizer, zero_initial_state, eps, deterministic):
        """Compute query state output
        args:
            Q: [b, n, c, h, d]
            S: [b, n, h, D, d]
            Y: [b, n, c, h, d] or None
            rowmax: [b, n, c, h] or None
            deg: int
            stabilizer: float or None
            zero_initial_state: bool
            eps: float
        returns:
            Y: [b, n, c, h, d]
        """

        b, n, c, h, d = Q.shape
        _, _, _, _, D = S.shape
        if isinstance(stabilizer, float):
            stabilizer = torch.tensor(stabilizer, dtype=torch.float32, device=Q.device)
        att_Y = Y

        Q = rearrange(Q, 'b n c h d -> b n h c d')
        
        phi_Q = QueryStateReference.expand(Q, deg)
        if stabilizer is not None:
            phi_Q = phi_Q / torch.sqrt(stabilizer)
            S = S / torch.sqrt(stabilizer)

        Y = torch.matmul(phi_Q.to(Q.dtype), S).to(Q.dtype)  # [b n h c d]

        Y = rearrange(Y, 'b n h c d -> b n c h d')

        if att_Y is not None:
            assert rowmax is not None, "rowmax must be provided when fused is true"
            Y = QueryStateReference.scale_combine_outputs(Y, att_Y, stabilizer, rowmax, zero_initial_state, eps)

        ctx.save_for_backward(Q, S, rowmax)
        ctx.stabilizer = stabilizer
        ctx.zero_initial_state = zero_initial_state
        ctx.deg = deg
        ctx.d = d
        ctx.fused = att_Y is not None
        return Y.contiguous()

    @staticmethod
    def backward(ctx, dY):
        """
        args:
            dY: [b, n, c, h, d]
        returns:
            dQ: [b, n, c, h, d]
            dS: [b, n, h, D, d]
        """
        Q, S, rowmax = ctx.saved_tensors
        b, n, c, h, d = Q.shape
        divisor = torch.sqrt(ctx.stabilizer) if ctx.stabilizer is not None else None

        phi_Q = QueryStateReference.expand(Q, ctx.deg)
        if ctx.fused:
            dY_attn, dY_qs = QueryStateReference.scale_combine_gradients(dY, ctx.stabilizer, rowmax, ctx.zero_initial_state)
        else:
            dY_attn, dY_qs = None, dY
        dY_qs = dY_qs.transpose(2, 3).to(Q.dtype) # [b, n, h, c, d]

        S_t = S.transpose(-1, -2)  # [b, n, h, d, D]
        if divisor is not None:
            dY_qs = (dY_qs / divisor).to(Q.dtype)
            phi_Q = (phi_Q / divisor).to(Q.dtype)

        dS = torch.matmul(phi_Q.transpose(-1, -2), dY_qs).to(dY_qs.dtype)  # [b, n, h, D, d]
        dphi_Q = dY_qs @ S_t  # [b, n, h, c, D]

        dQ = torch.zeros(Q.shape, dtype=Q.dtype, device=Q.device) # [b, n, h, c, d]
        dphi_Q = rearrange(dphi_Q, 'b n h c (z i o) -> b n h c z i o', i=InnerBlock, o=OuterBlock).to(Q.dtype)
        Q_inner = rearrange(Q, 'b n h c (y i) -> b n h c y i', i=InnerBlock)
        Q_outer = rearrange(Q, 'b n h c (x o) -> b n h c x o', o=OuterBlock)
        y, x = ctx.d // InnerBlock, ctx.d // OuterBlock
        BlockD = InnerBlock * OuterBlock

        for j in range(ctx.d // OuterBlock):
            for i in range((j * OuterBlock) // InnerBlock, y):
                z = (i * (i + 1) // 2 * InnerBlock // OuterBlock) + j
                dPQ_block = dphi_Q[..., z, :, :] # [b, n, h, c, InnerBlock, OuterBlock]
                dPQ_block = rearrange(dPQ_block, 'b n h c i o -> b n h c (i o)')
                Q_i = Q_inner[..., i, :] # [b, n, h, c, InnerBlock]
                Q_o = Q_outer[..., j, :] # [b, n, h, c, OuterBlock]
                dQ_i = (dPQ_block * Q_o)
                dQ_o = (dPQ_block * Q_i).sum(dim=-1, keepdim=True)
                dQ[..., i * InnerBlock:(i + 1) * InnerBlock] += dQ_i
                dQ[..., j * OuterBlock:(j + 1) * OuterBlock] += dQ_o

        dQ = dQ.transpose(2, 3)
        return dQ, dS, dY_attn, None, None, None, None, None, None

def query_state_reference(*args, **kwargs):
    if args and kwargs:
        raise ValueError("Cannot pass both args and kwargs")
    if kwargs:
        args = (kwargs['Q'], kwargs['S'], kwargs['Y'], kwargs['rowmax'], kwargs['deg'], 
                kwargs['stabilizer'], kwargs['zero_initial_state'], kwargs['eps'], kwargs['deterministic'])
    return QueryStateReference.apply(*args)
query_state_reference_fwd = dummify(QueryStateReference.forward)