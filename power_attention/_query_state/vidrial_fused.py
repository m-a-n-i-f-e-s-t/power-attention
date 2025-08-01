# A query state implementation that fuses the normalizer into a dimension of the output tensor, using torch.
import torch
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.op import op as sympow_mma
import math

import torch
import triton
import triton.language as tl
import os
from math import comb, log, sqrt
from power_attention.kernelgen import kernelgen
from power_attention._utils import compute_expanded_dim, diff

def prune_configs(configs, nargs, **kwargs):
    smallest_block_T = min(c.kwargs["BLOCK_T"] for c in configs)
    pruned_configs = [c for c in configs if c.kwargs.get("BLOCK_E", 0) <= nargs["e"] and (c.kwargs["BLOCK_T"] <= nargs["c"] or c.kwargs["BLOCK_T"] == smallest_block_T)]
    return pruned_configs

fwd_configs = [
    triton.Config({'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BE in [32, 64]
    for BT in [32, 64, 128, 256]
    for nw in [4, 8]
    for ns in [1, 3]
]


@triton.autotune(fwd_configs, key=["deg", "e"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(fwd_configs)
def _query_state_normalize_fwd(Y_qs, L_qs, Y_attn, L_attn, M, O, L,
                     stride_Yqs_b, stride_Yqs_t, stride_Yqs_h, stride_Yqs_e,
                     stride_Lqs_b, stride_Lqs_t, stride_Lqs_h,
                     stride_Yattn_b, stride_Yattn_t, stride_Yattn_h, stride_Yattn_e,
                     stride_Lattn_b, stride_Lattn_t, stride_Lattn_h,
                     stride_mb, stride_mt, stride_mh,
                     stride_ob, stride_ot, stride_oh, stride_oe,
                     stride_lb, stride_lt, stride_lh,
                     n, h, c, e: tl.constexpr, D,
                     zero_initial_state,
                     deg: tl.constexpr,
                     scale_p,
                     BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    """
Compute query state output. It computes the following equation:

        ( Y_qs * (scale^p) * γ/α + Y_attn * exp(rowmax)/α )
    O = -------------------------------------------------------------
        ( l_qs * (scale^p) * γ/α + l_attn * exp(rowmax)/α )
    where α = max(γ, exp(rowmax))
          γ = 1.0 if S == 0 else sqrt(D)


    <kernelgen>
BLOCK_E_VALID: tl.constexpr = e if e < BLOCK_E else BLOCK_E
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_n = off_bn % n
off_t = tl.program_id(1)
off_e = tl.program_id(2)

Y_qs += off_bn.to(tl.int64) * stride_Yqs_b + off_h.to(tl.int64) * stride_Yqs_h
L_qs += off_bn.to(tl.int64) * stride_Lqs_b + off_h.to(tl.int64) * stride_Lqs_h
O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
Y_attn += off_bn.to(tl.int64) * stride_Yattn_b + off_h.to(tl.int64) * stride_Yattn_h
L_attn += off_bn.to(tl.int64) * stride_Lattn_b + off_h.to(tl.int64) * stride_Lattn_h
M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, BLOCK_E_VALID).to(tl.int64) + off_e * BLOCK_E_VALID

y = tl.zeros((BLOCK_T, BLOCK_E_VALID), dtype=tl.float32)
l = tl.zeros((BLOCK_T,), dtype=tl.float32)
gamma = tl.sqrt(D).to(tl.float32)

mask_T = range_t < c

p_y_attn = Y_attn + range_t[:, None] * stride_Yattn_t + range_e[None, :] * stride_Yattn_e
p_m = M + range_t * stride_mt
p_l_attn = L_attn + range_t * stride_Lattn_t
p_y_qs = Y_qs + range_t[:, None] * stride_Yqs_t + range_e[None, :] * stride_Yqs_e
p_l_qs = L_qs + range_t * stride_Lqs_t
rowmax = tl.load(p_m, mask=mask_T, other=-float('inf')) # BLOCK_T
y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32) # BLOCK_T x BLOCK_E_VALID
l_attn = tl.load(p_l_attn, mask=mask_T, other=0.).to(tl.float32) # BLOCK_T
y_qs = tl.load(p_y_qs, mask=mask_T[:, None], other=0.).to(tl.float32) # BLOCK_T x BLOCK_E_VALID
l_qs = tl.load(p_l_qs, mask=mask_T, other=0.).to(tl.float32) # BLOCK_T
m = tl.exp(rowmax)
alpha = tl.maximum(gamma, m) # BLOCK_T
qs_factor = scale_p * gamma / alpha # BLOCK_T
attn_factor = m / alpha # BLOCK_T

o = y_attn * attn_factor[:, None] + y_qs * qs_factor[:, None] # BLOCK_T x BLOCK_E_VALID
l = l_attn * attn_factor + l_qs * qs_factor # BLOCK_T
o = o / l[:, None] # BLOCK_T x BLOCK_E_VALID

# store y back to O
p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
p_l = L + range_t * stride_lt
tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
tl.store(p_l, l, mask=mask_T)
    </kernelgen>
    """
    pass


dQ_bwd_configs = [
    triton.Config({'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BT in [32, 64, 128, 256]
    for nw in [4, 8]
    for ns in [1, 3]
]

def prune_configs_bdw(configs, nargs, **kwargs):
    pruned_configs = [c for c in configs if c.kwargs.get("BLOCK_T", 0) <= nargs["c"]]
    return pruned_configs

@triton.autotune(dQ_bwd_configs, key=["deg", "e"], prune_configs_by={'early_config_prune': prune_configs_bdw})
@triton.jit
@kernelgen(dQ_bwd_configs)
def _query_state_bwd_normalize(dO, Delta, M, L, dY_qs, dL_qs, dY_attn, dL_attn,
                        stride_dob, stride_dot, stride_doh, stride_doe,
                        stride_db, stride_dt, stride_dh,
                        stride_mb, stride_mt, stride_mh,
                        stride_lb, stride_lt, stride_lh,
                        stride_dYqs_b, stride_dYqs_t, stride_dYqs_h, stride_dYqs_e,
                        stride_dLqs_b, stride_dLqs_t, stride_dLqs_h,
                        stride_dYattn_b, stride_dYattn_t, stride_dYattn_h, stride_dYattn_e,
                        stride_dLattn_b, stride_dLattn_t, stride_dLattn_h,
                        n, h, c, e: tl.constexpr, D,
                        zero_initial_state: tl.constexpr,
                        deg: tl.constexpr,
                        scale_p,
                        BLOCK_T: tl.constexpr):
    """
    This kernel will compute dQ and dY_attn and dL_attn.
        dL_attn = -exp(rowmax) / alpha / l * delta
        dY_attn = exp(rowmax) / alpha / l * dO
        dL_qs = -(scale^p) * gamma / alpha / l * delta
        dY_qs = (scale^p) * gamma / alpha / l * dO
        where alpha = max(gamma, exp(rowmax))
              gamma = 1.0 if S == 0 else sqrt(D)

    <kernelgen>
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_t = tl.program_id(1)

dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
Delta += off_bn.to(tl.int64) * stride_db + off_h.to(tl.int64) * stride_dh
M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
L += off_bn.to(tl.int64) * stride_lb + off_h.to(tl.int64) * stride_lh
dY_qs += off_bn.to(tl.int64) * stride_dYqs_b + off_h.to(tl.int64) * stride_dYqs_h
dL_qs += off_bn.to(tl.int64) * stride_dLqs_b + off_h.to(tl.int64) * stride_dLqs_h
dY_attn += off_bn.to(tl.int64) * stride_dYattn_b + off_h.to(tl.int64) * stride_dYattn_h
dL_attn += off_bn.to(tl.int64) * stride_dLattn_b + off_h.to(tl.int64) * stride_dLattn_h

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, e).to(tl.int64)
p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x e]
p_m = M + range_t * stride_mt
p_dY_attn = dY_attn + range_t[:, None] * stride_dYattn_t + range_e[None, :] * stride_dYattn_e # [BLOCK_T x e]
p_l = L + range_t * stride_lt
p_dL_attn = dL_attn + range_t * stride_dLattn_t
p_dY_qs = dY_qs + range_t[:, None] * stride_dYqs_t + range_e[None, :] * stride_dYqs_e # [BLOCK_T x e]
p_dL_qs = dL_qs + range_t * stride_dLqs_t

# --- compute dL_attn ---
rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
l = tl.load(p_l, mask=range_t < c, other=float('inf'))
p_delta = Delta + range_t * stride_dt
delta = tl.load(p_delta, mask=range_t < c, other=0.)

gamma = tl.sqrt(D).to(tl.float32)
m = tl.exp(rowmax)
alpha = tl.maximum(gamma, m) # BLOCK_T
attn_factor = m / alpha / l # BLOCK_T
dL_attn = -attn_factor * delta # BLOCK_T
tl.store(p_dL_attn, dL_attn, mask=range_t < c)

# --- compute dL_qs ---
qs_factor = scale_p * gamma / alpha / l # BLOCK_T
dL_qs = -qs_factor * delta # BLOCK_T
tl.store(p_dL_qs, dL_qs, mask=range_t < c)

# --- compute dY_attn ---
do = tl.load(p_do, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x e]
dy_attn = (do * attn_factor[:, None]).to(dO.dtype.element_ty) # BLOCK_T x e
tl.store(p_dY_attn, dy_attn, mask=(range_t < c)[:, None])

# --- compute dY_qs ---
dy_qs = (do * qs_factor[:, None]).to(dO.dtype.element_ty) # BLOCK_T x e
tl.store(p_dY_qs, dy_qs, mask=(range_t < c)[:, None])
    </kernelgen>
    """
    pass


preprocess_configs = [
    triton.Config({'BM': BM})
    for BM in [64, 128, 256]
]

@triton.autotune(preprocess_configs, key=["HEAD_DIM"])
@triton.jit
def _query_state_normalize_bwd_preprocess(O, DO, Delta,  #
                         stride_ob, stride_om, stride_oh, stride_oe, #
                         stride_dob, stride_dom, stride_doh, stride_doe, #
                         stride_db, stride_dm, stride_dh, #
                         BM: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    range_m = tl.program_id(0) * BM + tl.arange(0, BM)
    off_b = tl.program_id(1)
    off_h = tl.program_id(2)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(O + off_b * stride_ob + off_h * stride_oh + range_m[:, None] * stride_om + off_n[None, :] * stride_oe, cache_modifier=".cg").to(tl.float32)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + range_m[:, None] * stride_dom + off_n[None, :] * stride_doe, cache_modifier=".cg").to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + range_m * stride_dm, delta)



class _query_state_normalize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_qs, l_qs, Y_attn, l_attn, rowmax, deg, scale, D, zero_initial_state):
        """Compute query state output. It computes the following equation:

                ( (Y_qs * (scale^p) * γ/α + Y_attn * exp(rowmax)/α )
            O = -------------------------------------------------------------
                ( (l_qs * (scale^p) * γ/α + l_attn * exp(rowmax)/α )
            where α = max(γ, exp(rowmax))
                  γ = sqrt(D)

        args:
            Y_qs: [b, n, c, h, e] - query state output
            l_qs: [b, n, c, h] - sum of normalization factor from query state
            Y_attn: [b, n, c, h, e] - attention output
            l_attn: [b, n, c, h] - sum of powered attention scores, in normal space
            rowmax: [b, n, c, h] - max of powered attention scores, in log space
            deg: int - degree of power
            scale: float - sm_scale used in attention
            D: int - expanded dimension
            zero_initial_state: bool - whether the initial state is zero
        returns:
            O: [b, n, c, h, e] - temporal-normalized output
        """
        
        b, n, c, h, e = Y_qs.shape
        assert Y_attn.shape == (b, n, c, h, e), f"Y_attn.shape: {Y_attn.shape}, expected ({b}, {n}, {c}, {h}, {e})"
        assert l_qs.shape == (b, n, c, h), f"l_qs.shape: {l_qs.shape}, expected ({b}, {n}, {c}, {h})"
        assert l_attn.shape == (b, n, c, h), f"l_attn.shape: {l_attn.shape}, expected ({b}, {n}, {c}, {h})"
        assert rowmax.shape == (b, n, c, h), f"rowmax.shape: {rowmax.shape}, expected ({b}, {n}, {c}, {h})"
        O = torch.empty((b*n, c, h, e), device=Y_qs.device, dtype=Y_qs.dtype) # [b*n, c, h, e]
        l = torch.empty((b*n, h, c), device=Y_qs.device, dtype=torch.float32).transpose(-1, -2) # [b*n, c, h]
        # put batch dims together
        Y_qs, l_qs, Y_attn, l_attn, rowmax = map(lambda x: x.view(b*n, *x.shape[2:]), (Y_qs, l_qs, Y_attn, l_attn, rowmax))

        grid = lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]), triton.cdiv(e, args["BLOCK_E"]))
        _query_state_normalize_fwd[grid](
            Y_qs, l_qs, Y_attn, l_attn, rowmax, O, l,
            *Y_qs.stride(),
            *l_qs.stride(),
            *Y_attn.stride(),
            *l_attn.stride(),
            *rowmax.stride(),
            *O.stride(),
            *l.stride(),
            n, h, c, e, float(D), zero_initial_state, deg, scale**deg
        )

        O = O.view(b, n, c, h, e)
        O[..., 0] = 1.0 # First feature is reserved for normalization
        ctx.save_for_backward(rowmax, O, l)
        ctx.deg = deg
        ctx.zero_initial_state = zero_initial_state
        ctx.b = b
        ctx.n = n
        ctx.c = c
        ctx.h = h
        ctx.e = e
        ctx.D = D
        ctx.scale_p = scale**deg
        return O
        
    @staticmethod
    def backward(ctx, dO):
        """ Computes the following equation for the backward pass:

            delta = (dO * O) @ 1
            dl_attn = -exp(rowmax) / α / l * delta
            dY_attn = exp(rowmax) / α / l * dO
            dl_qs = -(scale^p) * γ / α / l * delta
            dY_qs = (scale^p) * γ / α / l * dO

        args:
            dO: [b, n, c, h, e]
        returns:
            dY_qs: [b, n, c, h, e]
            dl_qs: [b, n, c, h]
            dY_attn: [b, n, c, h, e]
            dl_attn: [b, n, c, h]
        """
        rowmax, O, l = ctx.saved_tensors
        b, n, c, h, e = O.shape
        deg = ctx.deg
        zero_initial_state = ctx.zero_initial_state
        scale_p = ctx.scale_p
        D = ctx.D

        dY_qs = torch.empty_like(O) # [b*n, c, h, e]
        dO.narrow(-1, 0, 1).fill_(0.0)
        dl_qs = torch.empty((b*n, h, c), device=dY_qs.device, dtype=torch.float32).transpose(-1, -2)
        dY_attn = torch.empty((b*n, c, h, e), device=dY_qs.device, dtype=dY_qs.dtype)
        dl_attn = torch.empty((b*n, h, c), device=dY_qs.device, dtype=torch.float32).transpose(-1, -2)
        delta = torch.empty((b*n, h, c), device=dY_qs.device, dtype=torch.float32).transpose(-1, -2)
        O, dO, dY_qs = O.view(b*n, c, h, e), dO.view(b*n, c, h, e), dY_qs.view(b*n, c, h, e)
        
        # --- compute delta and O dO ---
        _query_state_normalize_bwd_preprocess[lambda args: (triton.cdiv(c, args["BM"]), b*n, h)](
            O, dO, delta,
            *O.stride(),
            *dO.stride(),
            *delta.stride(),
            HEAD_DIM=e
        )

        # --- compute dQ and dY_attn and dl_attn ---
        _query_state_bwd_normalize[lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]))](
            dO, delta, rowmax, l, dY_qs, dl_qs, dY_attn, dl_attn,
            *dO.stride(),
            *delta.stride(),
            *rowmax.stride(),
            *l.stride(),
            *dY_qs.stride(),
            *dl_qs.stride(),
            *dY_attn.stride(),
            *dl_attn.stride(),
            n, h, c, e, float(D),
            zero_initial_state=zero_initial_state,
            deg=deg,
            scale_p=scale_p
        )

        dY_qs, dl_qs, dY_attn, dl_attn = map(lambda x: x.view(b, n, *x.shape[1:]), (dY_qs, dl_qs, dY_attn, dl_attn))
        return dY_qs, dl_qs, dY_attn, dl_attn, None, None, None, None, None

def _query_state_normalize_fn(Y_qs, l_qs, Y_attn, l_attn, rowmax, deg, scale, D, zero_initial_state=True):
    return _query_state_normalize.apply(Y_qs, l_qs, Y_attn, l_attn, rowmax, deg, scale, D, zero_initial_state)


query_state_normalize = torch.compiler.disable(_query_state_normalize_fn)

def query_state(Q, S, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile=None):
    if len(Q.shape) == 4: # inference call
        Q = Q.unsqueeze(1) # [b, 1, c, hq, d]
        S = S.unsqueeze(1) # [b, 1, hk, D, e]
        Y_attn = Y_attn.unsqueeze(1) # [b, 1, c, hq, e]
        l_attn = l_attn.unsqueeze(1) # [b, 1, c, hq]
        rowmax = rowmax.unsqueeze(1) # [b, 1, c, hq]
        O = query_state(Q, S, Y_attn, l_attn, rowmax, deg, scale, zero_initial_state, d_tile)
        return O.squeeze(1) # type: ignore

    b, n, c, hq, d = Q.shape
    _, _, hk, D, e = S.shape
    assert hq == hk or hq % hk == 0, f'hq must be equal to hk or a multiple of hk: {hq=}, {hk=}'
    group_ratio = hq // hk
    d_tile = default_d_tile(d, deg) if d_tile is None else d_tile
    transpose_head = lambda x: x.transpose(2,3)
    Q = transpose_head(Q) # [b, n, hq, c, d]
    Q = Q.reshape(b, n, hk, c * group_ratio, d)
    Y_qs = sympow_mma(Q, S, power=deg, scale_A=1/math.sqrt(float(D)), d_tile=d_tile, expand_dim=-1).reshape(b, n, hq, c, e) # [b, n, hq, c, e]
    l_qs = Y_qs.narrow(-1, 0, 1).to(torch.float32).squeeze(-1)
    Y_qs, l_qs = map(transpose_head, (Y_qs, l_qs)) # [b, n, c, hq, ...]
    O = query_state_normalize(Y_qs, l_qs, Y_attn, l_attn, rowmax, deg, scale, D, zero_initial_state) # type: ignore
    return O
