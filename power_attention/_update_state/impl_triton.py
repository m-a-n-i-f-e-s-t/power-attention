import torch
import triton
import triton.language as tl
from math import comb
from power_attention.kernelgen import kernelgen

def prune_configs(configs, nargs, **kwargs):
    pruned_configs = []
    for config in configs:
        if config.kwargs["BLOCK_E"] <= nargs["e"] and config.kwargs["BLOCK_D"] <= nargs["D"] and config.kwargs["BLOCK_T"] <= nargs["T"]:
            pruned_configs.append(config)
    return pruned_configs

fwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT}, num_warps=nw, num_stages=ns)
    for BD in [128, 256]
    for BE in [32, 64]
    for BT in [16, 32]
    for block1 in [16]
    for nw in [4, 8]
    for ns in [1, 3]
]

def keep(config):
    block1 = config.kwargs["block1"]
    block2 = config.kwargs["BLOCK_D"] // block1
    if block1 >= block2 and block1 % block2 == 0:
        return True
    return False

@triton.jit
def get_offsets_p2(off_D, d, block1, block_D):
    """ Return off_d1, off_d2, and the multiplier for the starting offset on dimension 1 and 2, given block offset of the expanded dimension D. 

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
    multiplier = 1 if (n + 1) * block2 > m * block1 else 2
    return m*block1, n*block2, multiplier

@triton.autotune(list(filter(keep, fwd_configs)), key=["deg", "d", "e", "D"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(list(filter(keep, fwd_configs)))
def _update_state_fwd(K, V, S, deg: tl.constexpr, 
                     stride_kb, stride_kt, stride_kh, stride_kd,
                     stride_vb, stride_vt, stride_vh, stride_ve,
                     stride_sb, stride_sh, stride_sD, stride_se,
                     T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                     block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr):
    """ 
    This is a templated kernel, which, when called, will render the embedded
    template into a triton kernel in ./_rendered/_update_state_fwd_dispatcher.py, 
    and call the rendered kernel.

    Note that the rendered kernel is only for inspection purpose, modifying it will
    have no effect on runtime.

    <kernelgen>
block2: tl.constexpr = BLOCK_D // block1
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_D = tl.program_id(1)
off_e = tl.program_id(2)
off_d1, off_d2, multiplier = get_offsets_p2(off_D, d, block1, BLOCK_D)
off_d1 = tl.multiple_of(off_d1, block1)
off_d2 = tl.multiple_of(off_d2, block2)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
S += off_b.to(tl.int64) * stride_sb + off_h.to(tl.int64) * stride_sh + off_D.to(tl.int64) * BLOCK_D * stride_sD

range_t = tl.arange(0, BLOCK_T).to(tl.int64)
range_d1 = tl.arange(0, block1).to(tl.int64) + off_d1
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
p_k_d1 = K + range_d1[:, None] * stride_kd + range_t[None, :] * stride_kt # [block1 x BLOCK_T]
p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve # [BLOCK_T x BLOCK_E]

{% set block2 = BLOCK_D // block1 -%}
{% for i in range(block2) -%}
p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd
s_{{i}} = tl.zeros((block1, BLOCK_E), dtype=tl.float32)
{% endfor -%}

for tid in range(0, tl.cdiv(T, BLOCK_T)):
    k_d1 = tl.load(p_k_d1) # block1 x BLOCK_T
    v = tl.load(p_v)
    {% for i in range(block2) -%}
    k_d2_{{i}} = tl.load(p_k_d2_{{i}}) * multiplier # BLOCK_T
    phik_{{i}} = k_d1 * k_d2_{{i}}
    {% endfor -%}
    {% for i in range(block2) -%}
    s_{{i}} = tl.dot(phik_{{i}}.to(K.dtype.element_ty), v, s_{{i}})
    {% endfor -%}
    p_v += BLOCK_T * stride_vt
    p_k_d1 += BLOCK_T * stride_kt
    {% for i in range(block2) -%}
    p_k_d2_{{i}} += BLOCK_T * stride_kt
    {% endfor %}

{% for i in range(block2) -%}
range_d2_{{i}} = tl.arange(0, block1).to(tl.int64) + {{i}} * block1
p_s_{{i}} = S + range_d2_{{i}}[:, None] * stride_sD + range_e[None, :] * stride_se
tl.store(p_s_{{i}}, s_{{i}})
{% endfor -%}
    </kernelgen>
    """
    pass

bwd_configs = [
    triton.Config({'block1': block1, 'BLOCK_D': BD, 'BLOCK_E': BE, 'BLOCK_T': BT, 'V_IN_REGS': V_IN_REGS}, num_warps=nw, num_stages=ns)
    for BD in [16, 32]
    for BE in [32, 64]
    for BT in [128]
    for block1 in [16]
    for nw in [4]
    for ns in [1, 3]
    for V_IN_REGS in [True, False]
]

@triton.autotune(list(filter(keep, bwd_configs)), key=["deg", "d", "e"], prune_configs_by={'early_config_prune': prune_configs})
@triton.jit
@kernelgen(list(filter(keep, bwd_configs)))
def _update_state_bwd(K, V, dS, dK, dV, deg: tl.constexpr,
                      stride_kb, stride_kt, stride_kh, stride_kd,
                      stride_vb, stride_vt, stride_vh, stride_ve,
                      stride_dsb, stride_dsh, stride_dsD, stride_dse,
                      stride_dkb, stride_dkt, stride_dkh, stride_dkd,
                      stride_dvb, stride_dvt, stride_dvh, stride_dve,
                      T, H, d: tl.constexpr, e: tl.constexpr, D: tl.constexpr,
                      block1: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_T: tl.constexpr, V_IN_REGS: tl.constexpr):
    """
    In this case, the kernel template is given a list of possible values that some
    constexpr variables might take. This information is needed to be provided 
    manually because the autotune config doesn't have it.

    <kernelgen d=(32, 64, 128)>
block2: tl.constexpr = BLOCK_D // block1
{% set block1 = block1 -%}
{% set block2 = BLOCK_D // block1 -%}
tl.static_assert(block1 >= block2 and block1 % block2 == 0)
off_bh = tl.program_id(0)
off_b = off_bh // H
off_h = off_bh % H
off_t = tl.program_id(1)
off_e = tl.program_id(2)

K += off_b.to(tl.int64) * stride_kb + off_h.to(tl.int64) * stride_kh
V += off_b.to(tl.int64) * stride_vb + off_h.to(tl.int64) * stride_vh
dS += off_b.to(tl.int64) * stride_dsb + off_h.to(tl.int64) * stride_dsh
dK += off_b.to(tl.int64) * stride_dkb + off_h.to(tl.int64) * stride_dkh
dV += off_b.to(tl.int64) * stride_dvb + off_h.to(tl.int64) * stride_dvh

range_t = tl.arange(0, BLOCK_T).to(tl.int64) + off_t * BLOCK_T
range_e = tl.arange(0, BLOCK_E).to(tl.int64) + off_e * BLOCK_E
range_d1 = tl.arange(0, block1)
p_v = V + range_t[:, None] * stride_vt + range_e[None, :] * stride_ve
dv = tl.zeros((BLOCK_T, BLOCK_E), dtype=tl.float32)
{% for j in range(d//block1) -%}
dk_{{j}} = tl.zeros((BLOCK_T, block1), dtype=tl.float32)
{% endfor -%}

mask_T = range_t < T
if V_IN_REGS:
    v = tl.load(p_v, mask=mask_T[:, None], other=0.)

for m in range(0, d//block1):
    p_k_d1 = K + range_t[:, None] * stride_kt + (m*block1 + range_d1[None, :]) * stride_kd # BLOCK_T x block1
    k_d1 = tl.load(p_k_d1, mask=mask_T[:, None], other=0.)

    for n in range(0, (m+1)*block1//block2):
        off_d2 = n*block2
        multiplier = 1 if (n + 1) * block2 > m * block1 else 2
        off_d2 = tl.multiple_of(off_d2, block2)
        off_D = (m*(1+m)//2)*block1*block1 + off_d2*block1
        {% for i in range(block2) -%}
        p_k_d2_{{i}} = K + range_t[:] * stride_kt + (off_d2 + {{i}}) * stride_kd # BLOCK_T
        p_ds_{{i}} = dS + (range_d1[:, None] + off_D + {{i}} * block1) * stride_dsD + range_e[None, :] * stride_dse # block1 x BLOCK_E
        {% endfor -%}

        {% for i in range(block2) -%}
        k_d2_{{i}} = tl.load(p_k_d2_{{i}}, mask=mask_T, other=0.) # BLOCK_T
        ds_{{i}} = (tl.load(p_ds_{{i}}) * multiplier).to(K.dtype.element_ty) # block1 x BLOCK_E
        {% endfor -%}
        {% for i in range(block2) -%}
        phik_{{i}} = k_d1 * (k_d2_{{i}}[:, None]) # BLOCK_T x block1
        dv = tl.dot(phik_{{i}}.to(K.dtype.element_ty), ds_{{i}}, dv) # BLOCK_T x BLOCK_E
        {% endfor %}
        if not V_IN_REGS:
            v = tl.load(p_v, mask=mask_T[:, None], other=0.)

        {% for i in range(block2) %}
        dphik_{{i}} = tl.dot(v, tl.trans(ds_{{i}})).to(tl.float32) # BLOCK_T x block1
        if m == 0:
            dk_0 += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% for j in range(1, d//block1 - 1) -%}
        elif m == {{j}}:
            dk_{{j}} += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% endfor -%}
        else:
            dk_{{d//block1 - 1}} += dphik_{{i}} * k_d2_{{i}}[:, None] # BLOCK_T x block1
        {% endfor -%}
        
        {% for i in range(block2) -%}
        dk_d2_{{i}} = tl.sum(dphik_{{i}} * k_d1, 1) # BLOCK_T
        if off_d2//block1 == 0:
            mask = ((range_d1 + {{0}} * block1) == (off_d2 + {{i}}))
            dk_{{0}} += tl.where(mask[None, :].broadcast_to(dk_{{0}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{0}}.shape), 0.)
        {% for j in range(1, d//block1 - 1) -%}
        elif off_d2//block1 == {{j}}:
            mask = ((range_d1 + {{j}} * block1) == (off_d2 + {{i}}))
            dk_{{j}} += tl.where(mask[None, :].broadcast_to(dk_{{j}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{j}}.shape), 0.)
        {% endfor -%}
        else:
            mask = ((range_d1 + {{d//block1 - 1}} * block1) == (off_d2 + {{i}}))
            dk_{{d//block1 - 1}} += tl.where(mask[None, :].broadcast_to(dk_{{d//block1 - 1}}.shape), dk_d2_{{i}}[:, None].broadcast_to(dk_{{d//block1 - 1}}.shape), 0.)
        {% endfor %}


# save dk, dv
mask_T = range_t < T
{% for j in range(d//block1) -%}
p_dk_{{j}} = dK + range_t[:, None].to(tl.int64) * stride_dkt + ({{j}} * block1 + range_d1[None, :].to(tl.int64)) * stride_dkd
tl.store(p_dk_{{j}}, dk_{{j}}, mask=mask_T[:, None])
{% endfor -%}
p_dv = dV + range_t[:, None].to(tl.int64) * stride_dvt + range_e[None, :].to(tl.int64) * stride_dve
tl.store(p_dv, dv, mask=mask_T[:, None])
    
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


class _update_state(torch.autograd.Function):

    @staticmethod
    def forward(ctx, k, v, deg):
        """ Args: 
            K: (B, N, T, H, d)
            V: (B, N, T, H, e)
            deg: int

            Returns:
            S: (B, H, D, e) where D > comb(d + deg - 1, deg) with padding
        """
        assert k.shape == v.shape
        assert k.shape[1] == v.shape[1]
        assert k.shape[2] == v.shape[2]
        assert k.shape[3] == v.shape[3]
        assert k.shape[4] == v.shape[4]

        b, n, t, h, d, e = *k.shape, v.shape[-1]
        k = k.view(b*n, t, h, d)
        v = v.view(b*n, t, h, e)
        stride_kb, stride_kt, stride_kh, stride_kd = k.stride()
        stride_vb, stride_vt, stride_vh, stride_ve = v.stride()

        D = compute_expanded_dim(d, deg, 16)

        s = torch.empty((b*n, h, D, e), device=k.device, dtype=k.dtype)
        stride_sb, stride_sh, stride_sD, stride_se = s.stride()
        grid = lambda args: (b*n*h, triton.cdiv(D, args["BLOCK_D"]), triton.cdiv(e, args["BLOCK_E"]))

        if deg != 2:
            raise NotImplementedError("Only deg = 2 is supported for now")

        _update_state_fwd[grid](
            k, v, s, deg,
            stride_kb, stride_kt, stride_kh, stride_kd,
            stride_vb, stride_vt, stride_vh, stride_ve,
            stride_sb, stride_sh, stride_sD, stride_se,
            t, h, d, e, D)

        ctx.save_for_backward(k, v)
        ctx.deg = deg
        ctx.d = d
        ctx.e = e
        ctx.D = D

        return s.view(b, n, h, D, e)

    @staticmethod
    def backward(ctx, ds):
        k, v = ctx.saved_tensors
        deg, d = ctx.deg, ctx.d

        b, n, h, D, e, t = *ds.shape, k.shape[1]
        assert D == ctx.D

        ds = ds.view(b*n, h, D, e)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        stride_kb, stride_kt, stride_kh, stride_kd = k.stride()
        stride_vb, stride_vt, stride_vh, stride_ve = v.stride()
        stride_dsb, stride_dsh, stride_dsD, stride_dse = ds.stride()
        stride_dkb, stride_dkt, stride_dkh, stride_dkd = dk.stride()
        stride_dvb, stride_dvt, stride_dvh, stride_dve = dv.stride()

        grid = lambda args: (b*n*h, triton.cdiv(t, args["BLOCK_T"]), triton.cdiv(e, args["BLOCK_E"]))
        _update_state_bwd[grid](
            k, v, ds, dk, dv, deg,
            stride_kb, stride_kt, stride_kh, stride_kd,
            stride_vb, stride_vt, stride_vh, stride_ve,
            stride_dsb, stride_dsh, stride_dsD, stride_dse,
            stride_dkb, stride_dkt, stride_dkh, stride_dkd,
            stride_dvb, stride_dvt, stride_dvh, stride_dve,
            t, h, d, e, D)
        
        dk = dk.view(b, n, t, h, d)
        dv = dv.view(b, n, t, h, e)
        return dk, dv, None


def update_state(K, V, deg):
    return _update_state.apply(K, V, deg)


if __name__ == "__main__":
    from power_attention._update_state.impl import create_inputs, update_state as update_state_cutlass
    from perf._timing import benchmark_speed


    # Hyperparameters
    kw = dict(b=8, n=8, c=256, h=16, d=64, dtype=torch.bfloat16, device='cuda', seed=42)

    # Check correctness
    inputs_triton = create_inputs(**(kw | dict(requires_grad=True)))
    inputs_cutlass = create_inputs(**(kw | dict(requires_grad=True)))
    s_triton = update_state(inputs_triton['K'], inputs_triton['V'], inputs_triton['deg'])
    s_cutlass = update_state_cutlass(inputs_cutlass['K'], inputs_cutlass['V'], inputs_cutlass['deg'])
    torch.testing.assert_close(s_triton, s_cutlass, atol=1e-4, rtol=1e-2)
    print("Fwd correctness check passed")

    # Check gradients
    torch.autograd.backward(s_triton, torch.ones_like(s_triton))
    torch.autograd.backward(s_cutlass, torch.ones_like(s_cutlass))
    torch.testing.assert_close(inputs_triton['K'].grad, inputs_cutlass['K'].grad, atol=1e-4, rtol=1e-2)
    torch.testing.assert_close(inputs_triton['V'].grad, inputs_cutlass['V'].grad, atol=1e-4, rtol=1e-2)
    print("Bwd correctness check passed")

    # Thorough benchmarking
    def print_rowstr(rowstr):
        print(" | ".join([f"{r.upper():<10}" for r in rowstr.split(",")]))

    ctx = 16384
    for mode in ['fwd', 'bwd', 'fwd+bwd']:
        print(f"triton-vs-cutlass-batch{kw['b']}-ctx{ctx}-head{kw['h']}-dim{kw['d']}-{mode}")
        print_rowstr("chunk_size,triton,cutlass,triton speedup")
        for chunk_size in [2**i for i in range(6, 14)]:
            kw['c'] = chunk_size
            kw['n'] = ctx // chunk_size
            triton_time = benchmark_speed(mode, update_state, create_inputs, kw, compile=False)
            cutlass_time = benchmark_speed(mode, update_state_cutlass, create_inputs, kw, compile=False)
            speedup = cutlass_time / triton_time
            print_rowstr(f"{chunk_size}, {triton_time:.2f}, {cutlass_time:.2f}, {speedup:.2f}")




