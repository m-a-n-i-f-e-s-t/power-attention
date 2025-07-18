import torch
from power_attention.power_full_maker import make_power_full
from power_attention._attention.triton import attention
from power_attention._update_state.triton import update_state
from power_attention._discumsum.triton import discumsum
from power_attention._query_state.triton import query_state

power_full = make_power_full(update_state, query_state, discumsum, attention)


## TUTORIAL ##
if __name__ == '__main__':
    from perf._inspect import print_runtime
    from power_attention.create_inputs import create_inputs

    # Create inputs
    t = 1024
    chunk_size=128
    b = 8
    h = 16
    d = 64
    deg = 2
    gating = True
    dtype = torch.float16
    inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)
    
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'profile':
        O = power_full(**inputs)
        torch.autograd.backward((O,), grad_tensors=(O,))
    else:
        # Benchmark
        print(f"Benchmarking power_full {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

        print_runtime(power_full, **inputs)
