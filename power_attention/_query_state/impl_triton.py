import torch
import triton
import triton.language as tl
from math import comb
from power_attention.kernelgen import kernelgen

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BLOCK_E"] <= nargs["e"] and config.kwargs["BLOCK_D"] <= nargs["D"] and config.kwargs["BLOCK_T"] <= nargs["c"]:
            pruned_configs.append(config)
    return pruned_configs

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BD in [16, 32]
    for BE in [32, 64]
    for BT in [128, 256]
    for nw in [4, 8]
    for ns in [1, 3, 5, 7]
]

@triton.jit
def get_offsets_p2(off_D, d, block1, block_D):
    """ Return off_d1, off_d2 for the starting offset on dimension 1 and 2, given block offset of the expanded dimension D. 

    Define block1, block2 to be the block size along the first, the second dimension in the hypercube. Define m, n to be the offset in unit of blocks along the first, the second dimension in the hypercube.

    We use the following invariant to find the offset
       
       block2 <= block1
       m*(1+m)*block1/2 <= off_D*block2 <= (m+1)*(m+2)*block1/2
       
       or, let z = = off_D*block2/block1*2
       m*(1+m) <= z <= (m+1)*(m+2)
    """
    tl.static_assert(d % block1 == 0)
    block2: tl.constexpr = block_D // block1
    tl.static_assert(block1 >= block2 and block1 % block2 == 0)
    z = off_D.to(tl.float32)/(block1//block2)*2
    m = (tl.math.floor((tl.math.sqrt(1 + 4*z) - 1) / 2)).to(tl.int32)
    n = off_D - (m*(1+m)*(block1//block2)/2).to(tl.int32)
    return m*block1, n*block2


@triton.autotune(fwd_configs, key=["deg", "d", "e", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(fwd_configs)
def _query_state_fwd(Q, S, Y, M, O, deg: tl.constexpr, scale, zero_initial_state: tl.constexpr,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_yb, stride_yt, stride_yh, stride_ye,
                     stride_mb, stride_mt, stride_mh,
                     stride_ob, stride_ot, stride_oh, stride_oe,
                     n, h, c, d: tl.constexpr, D: tl.constexpr, e: tl.constexpr,
                     block1: tl.constexpr,
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    """
    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
{% set block2 = BLOCK_D // block1 -%}
tl.static_assert(block1 >= block2 and block1 % block2 == 0)
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_t = tl.program_id(1)
off_e = tl.program_id(2)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
O += off_bn.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
if Y is not None:
    Y += off_bn.to(tl.int64) * stride_yb + off_h.to(tl.int64) * stride_yh
    M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
range_d1 = tl.arange(0, block1).to(tl.int64)
p_s = S + tl.arange(0, BLOCK_D)[:, None] * stride_sD + range_e[None, :] * stride_se

y = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
mask_T = range_t < c

m, n = 0, 0
for m in range(0, d//block1):
    p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # BLOCK_T x block1
    q_d1 = tl.load(p_q_d1, mask=mask_T[:, None], other=0.) # BLOCK_T x block1

    for n in range(0, (m+1)*block1//block2):
        off_d2 = n*block2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) -%}
        p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # BLOCK_T
        p_s_{{i}} = S + (range_d1[:, None] + off_D + {{i}} * block1) * stride_sD + range_e[None, :] * stride_se # block1 x BLOCK_E
        {% endfor -%}

        {% for i in range(block2) -%}
        q_d2_{{i}} = tl.load(p_q_d2_{{i}}, mask=mask_T, other=0.) # BLOCK_T
        {% endfor -%}
        {% for i in range(block2) -%}
        phik_{{i}} = q_d1 * (q_d2_{{i}}[:, None]) # BLOCK_T x block1
        s_{{i}} = tl.load(p_s_{{i}}) # block1 x BLOCK_E
        if scale != 1.0:
            s_{{i}} = (s_{{i}} * scale).to(Q.dtype.element_ty) # block1 x BLOCK_E
        {% endfor -%}

        {% for i in range(block2) -%}
        y = tl.dot(phik_{{i}}.to(Q.dtype.element_ty), s_{{i}}, y) # BLOCK_T x BLOCK_E
        {% endfor %}

chunk_id = off_bn % n
if Y is not None:
    p_y_attn = Y + range_t[:, None] * stride_yt + range_e[None, :] * stride_ye
    p_m = M + range_t * stride_mt
    rowmax = tl.load(p_m, mask=mask_T, other=float(0.0))
    y_attn = tl.load(p_y_attn, mask=mask_T[:, None], other=0.).to(tl.float32)
    exp_neg_m = tl.exp(-rowmax)
    min_scale = tl.minimum(scale, exp_neg_m)
    y_attn = y_attn * (min_scale / exp_neg_m)[:, None]
    if (chunk_id > 0 or (not zero_initial_state)):
        o = y * ((min_scale / scale)[:, None]) + y_attn
    else:
        o = y
else:
    o = y

# store y back to O
p_o = O + range_t[:, None] * stride_ot + range_e[None, :] * stride_oe
tl.store(p_o, o.to(O.dtype.element_ty), mask=mask_T[:, None])
    </kernelgen>
    """
    pass


dQ_bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_T': BT, 'BLOCK_D': BD, 'BLOCK_E': BE}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BT in [128, 256]
    for BD in [16, 32]
    for BE in [32, 64]
    for nw in [4, 8]
    for ns in [1, 3, 5, 7]
]

@triton.autotune(dQ_bwd_configs, key=["deg", "d", "e", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(dQ_bwd_configs)
def _query_state_bwd_dQ(Q, S, M, dO, dQ, dY_attn, deg: tl.constexpr, scale, zero_initial_state,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     stride_mb, stride_mt, stride_mh,
                     stride_dob, stride_dot, stride_doh, stride_doe,
                     stride_dyb, stride_dyt, stride_dyh, stride_dye,
                     stride_dqb, stride_dqt, stride_dqh, stride_dqd,
                     n, h, c, d: tl.constexpr, D: tl.constexpr, e: tl.constexpr,
                     block1: tl.constexpr, BLOCK_T: tl.constexpr, 
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr):
    """
    This kernel will compute dQ.
    qs_factor = tl.minimum(tl.exp(-m)/scale, 1.0)
    attn_factor = tl.minimum(1.0, scale/tl.exp(-m))
    O = Y_attn * attn_factor + (phi(Q) @ S * scale) * qs_factor
    dY_attn = dO * attn_factor
    dphi(Q) = (dO * qs_factor * scale) @ (S.T)

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
{% set block2 = BLOCK_D // block1 -%}
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_t = tl.program_id(1)
off_e = tl.program_id(2)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
S += off_bn.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh
dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
dQ += off_bn.to(tl.int64) * stride_dqb + off_h.to(tl.int64) * stride_dqh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
range_d1 = tl.arange(0, block1)
p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]
do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
if M is not None:
    M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh
    p_m = M + range_t * stride_mt
    rowmax = tl.load(p_m, mask=range_t < c, other=-float('inf'))
    qs_factor = tl.minimum(tl.exp(-rowmax), scale)
    do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)
{% for j in range(d//block1) -%}
dq_{{j}} = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
{% endfor -%}

if dY_attn is not None:
    chunk_id = off_bn % n
    dY_attn += off_bn.to(tl.int64) * stride_dyb + off_h.to(tl.int64) * stride_dyh
    p_dy_attn = dY_attn + range_t[:, None] * stride_dyt + range_e[None, :] * stride_dye # [BLOCK_T x BLOCK_E]
    if chunk_id > 0 or (not zero_initial_state):
        attn_factor = tl.minimum(scale/tl.exp(-rowmax), 1.0)
        dy_attn = (do * attn_factor[:, None]).to(Q.dtype.element_ty)
        tl.store(p_dy_attn, dy_attn, mask=(range_t < c)[:, None])
    else:
        tl.store(p_dy_attn, do, mask=(range_t < c)[:, None])

for m in range(0, d//block1):
    p_q_d1 = Q + range_t[:, None] * stride_qt + (m*block1 + range_d1[None, :]) * stride_qd # [BLOCK_T x block1]
    q_d1 = tl.load(p_q_d1, mask=(range_t < c)[:, None], other=0.) # [BLOCK_T x block1]
    dq_d1 = tl.zeros((BLOCK_T, block1), dtype=tl.float32)

    for n in range(0, (m+1)*block1//block2):
        off_d2 = n*block2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) %}
        p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # [BLOCK_T]
        p_sT_{{i}} = S + (range_d1[None, :] + off_D + {{i}} * block1) * stride_sD + range_e[:, None] * stride_se # [BLOCK_E x block1]
        {% endfor -%}

        {% for i in range(block2) -%}
        q_d2_{{i}} = tl.load(p_q_d2_{{i}}, mask=(range_t < c), other=0.) # [BLOCK_T]
        sT_{{i}} = tl.load(p_sT_{{i}}) # [BLOCK_E x block1]
        {% endfor -%}

        {% for i in range(block2) %}
        dpq_{{i}} = tl.dot(do, sT_{{i}}) # [BLOCK_T x block1]
        if m == 0:
            dq_0 += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% for j in range(1, d//block1 - 1) -%}
        elif m == {{j}}:
            dq_{{j}} += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% endfor -%}
        else:
            dq_{{d//block1 - 1}} += dpq_{{i}} * q_d2_{{i}}[:, None]
        {% endfor -%}

        {% for i in range(block2) %}
        dq_d2_{{i}} = tl.sum(dpq_{{i}} * q_d1, 1) # [BLOCK_T]
        if off_d2//block1 == 0:
            mask = (tl.arange(0, block1) + {{0}} * block1) == (off_d2 + {{i}})
            dq_{{0}} += tl.where(mask[None, :].broadcast_to(dq_{{0}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{0}}.shape), 0.)
        {% for j in range(1, d//block1 - 1) -%}
        elif off_d2//block1 == {{j}}:
            mask = (tl.arange(0, block1) + {{j}} * block1) == (off_d2 + {{i}})
            dq_{{j}} += tl.where(mask[None, :].broadcast_to(dq_{{j}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{j}}.shape), 0.)
        {% endfor -%}
        else:
            mask = (tl.arange(0, block1) + {{d//block1 - 1}} * block1) == (off_d2 + {{i}})
            dq_{{d//block1 - 1}} += tl.where(mask[None, :].broadcast_to(dq_{{d//block1 - 1}}.shape), dq_d2_{{i}}[:, None].broadcast_to(dq_{{d//block1 - 1}}.shape), 0.)
        {% endfor -%}

# save dq
{% for j in range(d//block1) -%}
p_dq_{{j}} = dQ + range_t[:, None] * stride_dqt + ({{j}} * block1 + tl.arange(0, block1)[None, :]).to(tl.int64) * stride_dqd
tl.store(p_dq_{{j}}, dq_{{j}}, mask=(range_t < c)[:, None])
{% endfor -%}
    </kernelgen>
    """
    pass

dS_bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_T': BT, 'BLOCK_D': BD, 'BLOCK_E': BE}, num_warps=nw, num_stages=ns)
    for block1 in [16]
    for BT in [16, 32]
    for BD in [128, 256]
    for BE in [32, 64]
    for nw in [4, 8]
    for ns in [1, 3, 5, 7]
]

@triton.autotune(dS_bwd_configs, key=["deg", "d", "e", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(dS_bwd_configs)
def _query_state_bwd_dS(Q, M, dO, dS, deg: tl.constexpr, scale, zero_initial_state,
                     stride_qb, stride_qt, stride_qh, stride_qd,
                     stride_mb, stride_mt, stride_mh,
                     stride_dob, stride_dot, stride_doh, stride_doe,
                     stride_dsb, stride_dsh, stride_dsD, stride_dse,
                     n, h, c, d: tl.constexpr, D: tl.constexpr, e: tl.constexpr,
                     block1: tl.constexpr, BLOCK_T: tl.constexpr, 
                     BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr):
    """
    This kernel will compute dS.
    qs_factor = tl.minimum(tl.exp(-m)/scale, 1.0)
    attn_factor = tl.minimum(1.0, scale/tl.exp(-m))
    O = Y_attn * attn_factor + (phi(Q) @ S * scale) * qs_factor
    dS = phi(Q).T @ (dO * qs_factor * scale)

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
{% set block2 = BLOCK_D // block1 -%}
off_bnh = tl.program_id(0)
off_bn = off_bnh // h
off_h = off_bnh % h
off_D = tl.program_id(1)
off_e = tl.program_id(2)
off_d1, off_d2 = get_offsets_p2(off_D, d, block1, BLOCK_D)
off_d1 = tl.multiple_of(off_d1, block1)
off_d2 = tl.multiple_of(off_d2, block2)

Q += off_bn.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
dO += off_bn.to(tl.int64) * stride_dob + off_h.to(tl.int64) * stride_doh
dS += off_bn.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh + off_D.to(tl.int64) * BLOCK_D * stride_dsD
if M is not None:
    M += off_bn.to(tl.int64) * stride_mb + off_h.to(tl.int64) * stride_mh

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
p_qT_d1 = Q + range_d1[:, None] * stride_qd + range_t[None, :] * stride_qt # [block1 x BLOCK_T]
p_do = dO + range_t[:, None] * stride_dot + range_e[None, :] * stride_doe # [BLOCK_T x BLOCK_E]

{% set block2 = BLOCK_D // block1 -%}
{% for i in range(block2) -%}
p_q_d2_{{i}} = Q + range_t[:] * stride_qt + (off_d2 + {{i}}) * stride_qd # [BLOCK_T]
{% endfor -%}

{% for i in range(block2) -%}
ds_{{i}} = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
{% endfor -%}

for tid in range(0, tl.cdiv(c, BLOCK_T)):
    if M is not None:
        p_m = M + (range_t + tid * BLOCK_T) * stride_mt
        rowmax = tl.load(p_m, mask=(range_t + tid * BLOCK_T) < c, other=float('inf'))
    qT_d1 = tl.load(p_qT_d1) # block1 x BLOCK_T
    do = tl.load(p_do) # [BLOCK_T x BLOCK_E]
    {% for i in range(block2) -%}
    q_d2_{{i}} = tl.load(p_q_d2_{{i}}) # BLOCK_T
    phiqT_{{i}} = qT_d1 * q_d2_{{i}}[None, :] # [block1 x BLOCK_T]
    {% endfor -%}

    if M is not None:
        qs_factor = tl.minimum(tl.exp(-rowmax), scale)
        do = (do * qs_factor[:, None]).to(Q.dtype.element_ty)

    {% for i in range(block2) -%}
    ds_{{i}} = tl.dot(phiqT_{{i}}, do, ds_{{i}}) # [block1 x BLOCK_E]
    {% endfor -%}
    p_do += BLOCK_T * stride_dot
    p_qT_d1 += BLOCK_T * stride_qt
    {% for i in range(block2) -%}
    p_q_d2_{{i}} += BLOCK_T * stride_qt
    {% endfor %}

{% for i in range(block2) -%}
range_d2_{{i}} = tl.arange(0, block1).to(tl.int64) + {{i}} * block1
p_ds_{{i}} = dS + range_d2_{{i}}[:, None] * stride_dsD + range_e[None, :] * stride_dse
tl.store(p_ds_{{i}}, ds_{{i}})
{% endfor -%}
    </kernelgen>
    """
    pass


def compute_expanded_dim(d, deg, d_block=16):
    """ Compute the expanded state dimension for symmetric power for any given degree.

        Args:
            d: int, feature dimension of input tensor
            deg: int, degree of symmetric power attention
            d_block: int, block size for a single dimension. The smaller d_block is, the less waste there is.

        Returns:
            int, expanded state dimension D
    """
    hyper_cube_dim = d_block ** deg
    num_blocks_per_dim = d // d_block
    D = hyper_cube_dim * comb(num_blocks_per_dim + deg - 1, deg)
    return D


class _query_state(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, S, Y, rowmax, deg, scale, zero_initial_state):
        """ Compute query state output
        args:
            Q: [b, n, c, h, d]
            S: [b, n, h, D, d]
            Y: [b, n, c, h, e] or None
            rowmax: [b, n, c, h] or None, always in log space
            deg: int
            scale: float or None
            zero_initial_state: bool
        returns:
            O: [b, n, c, h, e]
        """
        
        b, n, c, h, d, D, e = *Q.shape, S.shape[-2], S.shape[-1]
        O = torch.empty((b*n, c, h, e), device=Q.device, dtype=Q.dtype)
        Q = Q.view(b*n, c, h, d)
        S = S.view(b*n, h, D, d)
        rowmax = rowmax.view(b*n, c, h) if rowmax is not None else None
        Y = Y.view(b*n, c, h, e) if Y is not None else None
        fused = Y is not None
        assert fused == (rowmax is not None), "Y and rowmax must both be None or both be not None"

        stride_qb, stride_qt, stride_qh, stride_qd = Q.stride()
        stride_sb, stride_sh, stride_sD, stride_sd = S.stride()
        stride_yb, stride_yt, stride_yh, stride_ye = Y.stride() if Y is not None else (0, 0, 0, 0)
        stride_mb, stride_mt, stride_mh = rowmax.stride() if rowmax is not None else (0, 0, 0)
        stride_ob, stride_ot, stride_oh, stride_oe = O.stride()
        
        grid = lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]), triton.cdiv(e, args["BLOCK_E"]))
        _query_state_fwd[grid](
            Q, S, Y, rowmax, O, deg, scale or 1.0, zero_initial_state or False,
            stride_qb, stride_qt, stride_qh, stride_qd,
            stride_sb, stride_sh, stride_sD, stride_sd,
            stride_yb, stride_yt, stride_yh, stride_ye,
            stride_mb, stride_mt, stride_mh,
            stride_ob, stride_ot, stride_oh, stride_oe,
            n, h, c, d, D, e
        )

        O = O.view(b, n, c, h, e)
        ctx.save_for_backward(Q, S, rowmax)
        ctx.deg = deg
        ctx.scale = scale or 1.0
        ctx.zero_initial_state = zero_initial_state or False
        ctx.b = b
        ctx.n = n
        ctx.c = c
        ctx.h = h
        ctx.d = d
        ctx.D = D
        ctx.e = e
        return O
        
    @staticmethod
    def backward(ctx, dO):
        """
        O = Y_attn * attn_factor + (phi(Q) @ S) * qs_factor
        dY_attn = dO * attn_factor
        dphi(Q) = (dO * qs_factor) @ S.T
        dS = phi(Q).T @ (dO * qs_factor)
        """
        Q, S, rowmax = ctx.saved_tensors
        b, n, c, h, d, D, e = ctx.b, ctx.n, ctx.c, ctx.h, ctx.d, ctx.D, ctx.e
        deg = ctx.deg
        scale = ctx.scale
        zero_initial_state = ctx.zero_initial_state

        dQ = torch.empty_like(Q)
        dS = torch.empty((b*n, h, D, e), device=Q.device, dtype=Q.dtype)
        if rowmax is not None:
            dY_attn = torch.empty((b*n, c, h, e), device=Q.device, dtype=Q.dtype)
        else:
            dY_attn = None
        dO = dO.view(b*n, c, h, e)

        stride_qb, stride_qt, stride_qh, stride_qd = Q.stride()
        stride_dqb, stride_dqt, stride_dqh, stride_dqd = dQ.stride()
        stride_sb, stride_sh, stride_sD, stride_sd = S.stride()
        stride_dsb, stride_dsh, stride_dsD, stride_dse = dS.stride()
        stride_dyb, stride_dyt, stride_dyh, stride_dye = dY_attn.stride() if dY_attn is not None else (0, 0, 0, 0)
        stride_mb, stride_mt, stride_mh = rowmax.stride() if rowmax is not None else (0, 0, 0)
        stride_dob, stride_dot, stride_doh, stride_doe = dO.stride()

        grid1 = lambda args: (b*n*h, triton.cdiv(c, args["BLOCK_T"]), triton.cdiv(e, args["BLOCK_E"]))
        _query_state_bwd_dQ[grid1](
            Q, S, rowmax, dO, dQ, dY_attn, deg, scale or 1.0, zero_initial_state,
            stride_qb, stride_qt, stride_qh, stride_qd,
            stride_sb, stride_sh, stride_sD, stride_sd,
            stride_mb, stride_mt, stride_mh,
            stride_dob, stride_dot, stride_doh, stride_doe,
            stride_dyb, stride_dyt, stride_dyh, stride_dye,
            stride_dqb, stride_dqt, stride_dqh, stride_dqd,
            n, h, c, d, D, e
        )

        grid2 = lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), triton.cdiv(e, args["BLOCK_E"]))
        _query_state_bwd_dS[grid2](
            Q, rowmax, dO, dS, deg, scale or 1.0, zero_initial_state,
            stride_qb, stride_qt, stride_qh, stride_qd,
            stride_mb, stride_mt, stride_mh,
            stride_dob, stride_dot, stride_doh, stride_doe,
            stride_dsb, stride_dsh, stride_dsD, stride_dse,
            n, h, c, d, D, e
        )

        dQ = dQ.view(b, n, c, h, d)
        dS = dS.view(b, n, h, D, e)
        return dQ, dS, dY_attn.view(b, n, c, h, e) if dY_attn is not None else None, None, None, None, None


def query_state(Q, S, Y, rowmax, deg, scale, zero_initial_state):
    return _query_state.apply(Q, S, Y, rowmax, deg, scale, zero_initial_state)

if __name__ == "__main__":
    from power_attention._query_state.impl import create_inputs, query_state as query_state_cutlass
    from perf._timing import benchmark_speed

    # Hyperparameters
    kw = dict(b=4, n=16, c=128, h=12, d=64, fused=True)

    # Check correctness
    inputs_triton = create_inputs(**(kw | dict(requires_grad=True)))
    inputs_cutlass = create_inputs(**(kw | dict(requires_grad=True)))
    output_triton = query_state(inputs_triton['Q'], inputs_triton['S'], inputs_triton['Y'], inputs_triton['rowmax'], inputs_triton['deg'], inputs_triton['scale'], inputs_triton['zero_initial_state'])
    output_cutlass = query_state_cutlass(inputs_cutlass['Q'], inputs_cutlass['S'], inputs_cutlass['Y'], inputs_cutlass['rowmax'], inputs_cutlass['deg'], inputs_cutlass['scale'], inputs_cutlass['zero_initial_state'])
    torch.testing.assert_close(output_triton, output_cutlass, atol=1e-4, rtol=1e-2)
    print("Correctness check passed")

    # Check gradients
    torch.autograd.backward(output_triton, torch.ones_like(output_triton))
    torch.autograd.backward(output_cutlass, torch.ones_like(output_cutlass))
    torch.testing.assert_close(inputs_triton['Q'].grad, inputs_cutlass['Q'].grad, atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(inputs_triton['Y'].grad, inputs_cutlass['Y'].grad, atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(inputs_triton['S'].grad, inputs_cutlass['S'].grad, atol=1e-4, rtol=1e-2)
    print("Gradient check passed")

    # Thorough benchmarking
    def print_rowstr(rowstr):
        print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))

    ctx = 16384
    for mode in ['fwd', 'bwd', 'fwd+bwd']:
        print(f"triton-vs-cutlass-batch{kw['b']}-ctx{ctx}-head{kw['h']}-dim{kw['d']}-{mode}")
        print_rowstr("chunk_size,triton,cutlass,triton speedup")
        for chunk_size in [2**i for i in range(7, 14)]:
            kw['c'] = chunk_size
            kw['n'] = ctx // chunk_size
            triton_time = benchmark_speed(mode, query_state, create_inputs, kw, compile=False)
            cutlass_time = benchmark_speed(mode, query_state_cutlass, create_inputs, kw, compile=False)
            speedup = cutlass_time / triton_time
            print_rowstr(f"{chunk_size}, {triton_time:.2f}, {cutlass_time:.2f}, {speedup:.2f}")
