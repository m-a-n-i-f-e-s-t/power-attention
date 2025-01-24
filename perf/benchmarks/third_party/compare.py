import torch
import triton
import math
from itertools import product
from perf._benchmark import Measurement
from perf._registration import register_benchmark
from perf._timing import benchmark_speed
from perf._utils import describe_gpu

from power_attention.power_full import power_full
from power_attention._utils import compute_expanded_dim

providers = ["sdpa"]
params = {
    "dtype": [torch.bfloat16],
    "device": ["cuda"],
    "gating": [False, True],
    "chunk_size": [128, None],
    "deg": [1, 2],
    "direction": ["fwd", "bwd", "fwd+bwd"],
    "b": [6],
    "hq": [12],
    "hk": [12],
    "d": [64],
    "causal": [True]
}
colors = {
    "sdpa": "black",
    "power-p1-chunk128": "#E6B3FF",  # light purple
    "power-p2-chunk128": "#9933FF",  # darker purple
    "power-p1": "#FFB366",  # light orange
    "power-p2": "#FF8000",  # dark orange
    "flash": "#FFB3B3",     # light red
    "fla-chunk": "#B3FFB3", # light green
    "fla-fused": "#33CC33", # darker green
    "rwkv-chunk": "#B3E6FF", # light blue
    "rwkv-fused": "#0099FF"  # darker blue
}

for deg, chunk_size in product(params['deg'], params['chunk_size']):
    if chunk_size is None:
        providers.append(f"power-p{deg}")
    else:
        providers.append(f"power-p{deg}-chunk{chunk_size}")

try:
    from flash_attn.flash_attn_interface import flash_attn_func
    providers.append("flash")
except BaseException:
    pass

try:
    from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
    # TODO: add rwkv for comparison
    # from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
    providers += ["fla-chunk", "fla-fused"]
except BaseException:
    pass


configs = [
    triton.testing.Benchmark(
        x_names=["ctx"],
        x_vals=[2**i for i in range(10, 16)],
        line_arg="provider",
        line_vals=providers,
        line_names=[provider.upper() for provider in providers],
        styles=[(colors[provider], "-") for provider in providers],
        ylabel="TFLOPS",
        plot_name=f"power-attention-compare-batch{batch}-headQ{head_q}-headK{head_k}-d{head_dim}-gating{gating}-causal{causal}-mode{mode}",
        args={
            "batch": batch,
            "head_q": head_q,
            "head_k": head_k,
            "head_dim": head_dim,
            "mode": mode,
            "dtype": dtype,
            "device": device,
            "gating": gating,
            "causal": causal
        }
    )
    for batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal in product(params['b'], params['hq'], params['hk'], params['d'], params['direction'], params['dtype'], params['device'], params['gating'], params['causal'])
]


def calculate_flops(ctx, batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider):
    """ calculate theoretical flops

    Returns:
        fwd_flops: FLOPs for forward pass
        bwd_flops: FLOPs for backward pass
    """
    def _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False):
        if mode == "fwd":
            return batch * head_q * (2 * ctx * ctx * head_dim * 2 + (ctx * ctx if gating else 0) + (ctx * ctx * 3 if power else 0)) * (0.5 if causal else 1.0)
        else:
            return batch * head_q * (ctx * ctx * head_dim * 2 # QK^T
                    + (ctx * ctx if gating else 0) # gating
                    + (ctx * ctx * 3 if power else 0) # power 
                    + ctx * head_dim * ctx * 2 # dV
                    + ctx * ctx * head_dim * 2 # dP
                    + ctx * ctx # dS
                    + ctx * head_dim * ctx * 2 # dQ
                    + ctx * head_dim * ctx * 2 # dK
                    ) * (0.5 if causal else 1.0)
    
    def _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, D, gating, causal, power=False):
        if mode == "fwd":
            return batch * head_q * (ctx/chunk_size) * (
                    + chunk_size * D * 2 # state expansion
                    + D * head_dim * chunk_size * 2 # update state
                    + chunk_size * head_dim * D * 2) # query state
        else:
            return batch * head_q * (ctx/chunk_size) * (
                + D * head_dim * chunk_size * 2 # dS
                + chunk_size * head_dim * D * 2 # dQ
                + chunk_size * head_dim * D * 2 # dK
                + chunk_size * head_dim * D * 2 # dV
                )
        

    if "power" in provider:
        deg = int(provider.split("-")[1][1:])
        chunk_size = int(provider.split("-")[2][5:]) if len(provider.split("-")) > 2 else None
        D = compute_expanded_dim(head_dim, deg)
        if chunk_size is None:
            return _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=True)
        else:
            attn_flops = _attention_flops(batch * ctx // chunk_size, chunk_size, head_q, head_k, head_dim, mode, gating, causal, power=True)
            chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, D, gating, causal, power=True)
            return attn_flops + chunk_flops
    elif "flash" in provider or "sdpa" in provider:
        return _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False)
    elif "fla" in provider:
        chunk_size = min(64, max(16, triton.next_power_of_2(ctx)))
        attn_flops = _attention_flops(batch, ctx, head_q, head_k, head_dim, mode, gating, causal, power=False)
        chunk_flops = _chunk_flops(batch, ctx, chunk_size, head_q, head_k, head_dim, mode, head_dim, gating, causal, power=False)
        return attn_flops + chunk_flops
    else:
        raise ValueError(f"Unknown provider: {provider}")


@triton.testing.perf_report(configs)
def bench_compare(ctx, batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider, measure):
    assert head_q % head_k == 0, "head_q must be divisible by head_k"
    q = torch.randn((batch, ctx, head_q, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))
    k = torch.randn((batch, ctx, head_k, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))
    v = torch.randn((batch, ctx, head_k, head_dim), device=device, dtype=dtype, requires_grad=("bwd" in mode))

    if "power" in provider:
        deg = int(provider.split("-")[1][1:])
        chunk_size = int(provider.split("-")[2][5:]) if len(provider.split("-")) > 2 else None
        if gating:
            log_g = torch.randn((batch, ctx, head_q), device=device, dtype=torch.float32)
        else:
            log_g = None
        def run_power():
            return torch.compile(power_full)(q, k, v, log_G=log_g, deg=deg, scale=1.0 / head_dim**0.5, chunk_size=chunk_size)
        fn = run_power
    elif "flash" in provider:
        def run_flash():
            return torch.compile(flash_attn_func)(q, k, v, causal=causal, softmax_scale=1.0 / head_dim**0.5)
        fn = run_flash
    elif "sdpa" in provider:
        def run_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                 attn_mask=None,
                                                                 dropout_p=0,
                                                                 is_causal=True,
                                                                 scale=1.0 / head_dim**0.5,
                                                                 enable_gqa=head_q > head_k)
        fn = run_sdpa
    elif "fla" in provider:
        def run_fla():
            is_chunked = "chunk" in provider
            if is_chunked:
                o, s = chunk_linear_attn(q, k, v, scale=1.0 / head_dim**0.5, initial_state=None, output_final_state=False, head_first=False, normalize=True)
                return o
            else:
                o, s = fused_chunk_linear_attn(q, k, v, scale=1.0 / head_dim**0.5, initial_state=None, output_final_state=False, head_first=False, normalize=True)
                return o
        fn = run_fla
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if mode == "bwd":
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
    elif mode == "fwd+bwd":
        fwd_fn = fn
        def fwd_bwd():
            o = fwd_fn()
            do = torch.randn_like(o)
            return o.backward(do, retain_graph=True)
        fn = fwd_bwd
    else:
        fn = fn

    ms = triton.testing.do_bench(fn)
    flops = calculate_flops(ctx, batch, head_q, head_k, head_dim, mode, dtype, device, gating, causal, provider)
    if measure == "throughput":
        return flops * 1e-12 / (ms * 1e-3)
    elif measure == "time":
        return ms
    elif measure == "flops":
        return flops
    else:
        raise ValueError(f"Unknown measure: {measure}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", type=str, default="throughput", choices=["throughput", "time", "flops"])
    args = parser.parse_args()
    bench_compare.run(save_path=".", print_data=True, measure=args.measure)
