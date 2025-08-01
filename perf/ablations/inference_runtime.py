import torch
from power_attention.vidrial_fused import power_full_inference
from power_attention.create_inputs import create_inference_inputs
from power_attention.vidrial_fused import update_state, query_state, attention
from perf._timing import estimate_runtime, get_compiled_version, sanitize_kwargs
from vidrial.py_utils.common import default_d_tile
from vidrial.kernels.sympow_mma.dimensions import sympow_dim
from vidrial.jit.decorator import set_settings, PickBest
import pandas as pd
import logging

def flops_estimate(b, t, h, d, qhead_ratio, deg, chunk_size):
    D = sympow_dim(d, deg, d_tile=default_d_tile(d, deg))
    attention_flops = (qhead_ratio * t * d * 2 + qhead_ratio * d * t * 2) * b * h
    query_state_flops = (qhead_ratio * D * d * 2) * b * h
    update_state_flops = 0 if t < chunk_size else (D * d * 2 *t)
    return attention_flops + query_state_flops + update_state_flops

def measure_attention_time(**kwargs):
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': True, 'device': 'cuda'})
    Q, K, V, log_G, state = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    if kwargs.get('profile', False):
        time = estimate_runtime(get_compiled_version(attention, {**inputs, 'log_G_accum': log_G_accum, 'r': r, 'w': w, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False}, direction='fwd', compile=False))
        return time
    else:
        sanitize_kwargs(attention)(**{**inputs, 'log_G_accum': log_G_accum, 'r': r, 'w': w, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False})
        return 0

def measure_query_state_time(**kwargs):
    d, deg = kwargs['d'], kwargs['deg']
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': True, 'device': 'cuda'})
    Q, K, V, log_G, state, scale = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state'], inputs['scale']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    attn_Y, l_attn, rowmax = attention(Q, K, V, log_G_accum, deg, r=r, w=w, scale=scale, norm=False) # type: ignore
    if kwargs.get('profile', False):
        time = estimate_runtime(get_compiled_version(query_state, {**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state}, direction='fwd', compile=False))
        return time
    else:
        sanitize_kwargs(query_state)(**{**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state})
        return 0

def measure_update_state_time(**kwargs):
    t, chunk_size = kwargs['t'], kwargs['chunk_size']
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': True, 'device': 'cuda'})
    if t < chunk_size:
        return 0
    if kwargs.get('profile', False):
        time = estimate_runtime(get_compiled_version(update_state, {**inputs}, direction='fwd', compile=False)) / chunk_size
        return time
    else:
        sanitize_kwargs(update_state)(**inputs)
        return 0

def measure_total_time(**kwargs):
    chunk_size = kwargs['chunk_size']
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': True, 'device': 'cuda'})
    if kwargs.get('profile', False):
        time = estimate_runtime(get_compiled_version(power_full_inference, inputs, direction='fwd', compile=False)) - measure_update_state_time(**kwargs) * (chunk_size - 1)
        return time
    else:
        sanitize_kwargs(power_full_inference)(**inputs)
        return 0

def measure_time(**kwargs):
    return {
        'total_time': measure_total_time(**kwargs),
        'attention_time': measure_attention_time(**kwargs),
        'query_state_time': measure_query_state_time(**kwargs),
        'update_state_time': measure_update_state_time(**kwargs),
    }


def main(profile=False):
    df = []
    b, t, h, d, chunk_size, deg, gating, dtype = 32, 64, 8, 64, 64, 2, True, torch.bfloat16
    print(f"Measuring runtime for {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")

    # logging.basicConfig(level=logging.DEBUG)
    with set_settings(policy=PickBest):
        for qhead_ratio in [1, 8, 16, 32]:
            print(f"========== {qhead_ratio=} ==========")
            df.append({**measure_time(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating, profile=profile), 'group': qhead_ratio})

    df = pd.DataFrame(df)
    print(df)


    import matplotlib.pyplot as plt

    # Create stack plot
    plt.figure(figsize=(12, 8))

    # Prepare data for stack plot
    x = df['group']
    attention_times = df['attention_time']
    query_state_times = df['query_state_time'] 
    update_state_times = df['update_state_time']

    # Create the stack plot
    plt.stackplot(x, attention_times, query_state_times, update_state_times,
                labels=['Attention', 'Query State', 'Update State'],
                alpha=0.8)

    # Also plot the total time as a line for reference
    plt.plot(x, df['total_time'], 'k--', linewidth=2, marker='o', label='Total Time', markersize=6)

    plt.xlabel('Group Size (qhead_ratio)')
    plt.ylabel('Time (ms)')
    plt.title(f'Inference Time Breakdown vs Group Size\n{b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'power_inference_time_breakdown_{b}_{t}_{h}_{d}_{chunk_size}_{deg}_{gating}_{dtype}.png', dpi=150, bbox_inches='tight')
    plt.show()


def torch_profile():
    from torch.profiler import profile, ProfilerActivity, record_function
    b, t, h, d, chunk_size, deg, gating, dtype = 32, 64, 8, 64, 64, 2, True, torch.bfloat16
    qhead_ratio = 16
    print(f"Profiling runtime for {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=} {qhead_ratio=}")
    inputs = create_inference_inputs(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, initial_state=True)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof:
        power_full_inference(inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state'], deg=deg, scale=1.0 / d**0.5, chunk_size=chunk_size)
    prof.export_chrome_trace(f'power_inference_time_breakdown_{b}_{t}_{h}_{d}_{chunk_size}_{deg}_{gating}_{dtype}_{qhead_ratio}.json')

    print(prof.key_averages(group_by_stack_n=2).table(sort_by='cuda_time_total', row_limit=10))

if __name__ == '__main__':
    main()
