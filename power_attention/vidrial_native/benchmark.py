import torch as th
import logging

from power_attention.vidrial_native.impl import *
from perf._timing import estimate_runtime, get_compiled_version
logging.basicConfig(level=logging.INFO)

def benchmark(algo, inputs):
    print(f"Benchmarking {algo}")
    fwd = get_compiled_version(algo.interface, inputs, 'fwd')
    bwd = get_compiled_version(algo.interface, inputs, 'bwd')
    fwd_time = estimate_runtime(fwd)
    bwd_time = estimate_runtime(bwd)
    print(f"  fwd average time: {fwd_time:.4f} ms")
    print(f"  bwd average time: {bwd_time:.4f} ms")

def long_ctx_benchmark():
    Algorithm = ChunkedPowerAttention
    t, c = 65536, 1024
    b, n, h, d = 1, t//c, 8, 64
    power, d_tile = 2, 8 
    dtype = th.bfloat16

    algo = Algorithm(b, n, c, h, d, power, d_tile, False, dtype)
    benchmark(algo, algo.make_inputs(requires_grad=True))

    # Compare against flash attention (very slow)
    # attn_algo = Algorithm(b, 1, t, h, d, power, d_tile, False, dtype)
    # benchmark(attn_algo, attn_algo.make_inputs(requires_grad=True))


if __name__ == "__main__":
    long_ctx_benchmark()