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
    t = kwargs['t']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    Q, K, V, log_G, state = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    if kwargs.get('profile', False):
        sanitize_kwargs(attention)(**{**inputs, 'log_G_accum': log_G_accum, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False})
        return 0
    else:
        time = estimate_runtime(get_compiled_version(attention, {**inputs, 'log_G_accum': log_G_accum, 'scale': 1.0 / kwargs['d']**0.5, 'norm': False}, direction='fwd', compile=False))
        return time

def measure_query_state_time(**kwargs):
    t, d, deg = kwargs['t'], kwargs['d'], kwargs['deg']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if inputs['state'] is None:
        return 0
    Q, K, V, log_G, state, scale = inputs['Q'], inputs['K'], inputs['V'], inputs['log_G'], inputs['state'], inputs['scale']
    hq, hk = Q.shape[2], K.shape[2]
    log_G_accum = log_G.cumsum(1) if log_G is not None else None
    r, w = hq // hk, 1
    attn_Y, l_attn, rowmax = attention(Q, K, V, log_G_accum, deg, scale=scale, norm=False) # type: ignore
    if kwargs.get('profile', False):
        sanitize_kwargs(query_state)(**{**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state})
        return 0
    else:
        time = estimate_runtime(get_compiled_version(query_state, {**inputs, 'Y_attn': attn_Y, 'l_attn': l_attn, 'rowmax': rowmax, 'zero_initial_state': False, 'S': state}, direction='fwd', compile=False))
        return time

def measure_update_state_time(**kwargs):
    t, chunk_size = kwargs['t'], kwargs['chunk_size']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if kwargs.get('profile', False):
        sanitize_kwargs(update_state)(**inputs)
        return 0
    else:
        time = estimate_runtime(get_compiled_version(update_state, {**inputs}, direction='fwd', compile=False)) / chunk_size
        return time

def measure_total_time(**kwargs):
    t, chunk_size = kwargs['t'], kwargs['chunk_size']
    kwargs['t'] = kwargs['t'] % (kwargs['switch_over_seq_len'] + 1) if kwargs['t'] < (kwargs['switch_over_seq_len'] + 1) else (kwargs['t'] % (kwargs['chunk_size'] + 1))
    inputs = create_inference_inputs(**{**kwargs, 'initial_state': t > kwargs['switch_over_seq_len'], 'device': 'cuda'})
    if kwargs.get('profile', False):
        sanitize_kwargs(power_full_inference)(**inputs)
        return 0
    else:
        time = estimate_runtime(get_compiled_version(power_full_inference, inputs, direction='fwd', compile=False))
        if kwargs['t'] > kwargs['switch_over_seq_len'] and kwargs['t'] % kwargs['chunk_size'] == 0:
            time -= measure_update_state_time(**kwargs) * (chunk_size - 1)
        return time

def measure_time(**kwargs):
    return {
        'total_time': measure_total_time(**kwargs),
        'attention_time': measure_attention_time(**kwargs),
        'query_state_time': measure_query_state_time(**kwargs),
        'update_state_time': measure_update_state_time(**kwargs),
    }


def measure_flashinfer_time(b, t, h, qhead_ratio, d, chunk_size, deg, gating, dtype, page_size=16, workspace_buffer=1024 * 1024 * 1024 * 4, **kwargs):
    import flashinfer
    workspace_buffer = torch.empty(workspace_buffer, dtype=torch.uint8, device="cuda:0")
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    pages_per_batch = t // page_size
    max_num_pages = b * pages_per_batch
    kv_page_indptr = torch.tensor(
        list(range(0, b * pages_per_batch, pages_per_batch)), dtype=torch.int32, device="cuda:0"
    )
    kv_page_indices = torch.arange(b * pages_per_batch).int().to("cuda:0")
    kv_last_page_len = torch.tensor(
        [page_size - 1] * b, dtype=torch.int32, device="cuda:0"
    )
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads=h * qhead_ratio,
        num_kv_heads=h,
        head_dim=d,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    q = torch.randn(b, h * qhead_ratio, d, device="cuda:0", dtype=torch.bfloat16)
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, h, d, device="cuda:0", dtype=torch.bfloat16
    )
    time = estimate_runtime(get_compiled_version(decode_wrapper.run, {'q': q, 'paged_kv_cache': kv_cache}, direction='fwd', compile=False))
    return time


def inference_time_breakdown(profile=False):
    df = []
    b, t, h, d, chunk_size, deg, gating, dtype = 32, 64, 8, 64, 64, 2, True, torch.bfloat16
    print(f"Measuring runtime for {b=} {t=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")
    logging.basicConfig(level=logging.ERROR)

    with set_settings(policy=PickBest):
        for qhead_ratio in [1, 8]:
            print(f"========== {qhead_ratio=} ==========")
            df.append({
                **measure_time(b=b, t=t%chunk_size, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating),
                'group': qhead_ratio,
            })

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


def compare_with_flashinfer():
    # install from https://github.com/flashinfer-ai/flashinfer
    df = []
    b, h, d, chunk_size, deg, gating, dtype, switch_over_seq_len = 32, 8, 64, 64, 2, True, torch.bfloat16, 512
    print(f"Measuring runtime for {b=} {h=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}")
    logging.basicConfig(level=logging.ERROR)
    qhead_ratio = 8

    with set_settings(policy=PickBest):
        for t in [512, 2048]:
            print(f"========== {t=} ==========")
            measurement = measure_time(b=b, t=t, h=h, d=d, qhead_ratio=qhead_ratio, deg=deg, chunk_size=chunk_size, dtype=dtype, gating=gating, switch_over_seq_len=switch_over_seq_len)
            flash_measurement = measure_flashinfer_time(b=b, t=t, h=h, qhead_ratio=qhead_ratio, d=d, chunk_size=chunk_size, deg=deg, gating=gating, dtype=dtype)
            df.append({
                **measurement,
                'group': qhead_ratio,
                'flashinfer_time': flash_measurement,
                'speedup': flash_measurement / measurement['total_time'],
                'context_size': t,
            })

    df = pd.DataFrame(df)
    print(df)


    import matplotlib.pyplot as plt

    # Create stack plot
    plt.figure(figsize=(12, 8))

    # Prepare data for stack plot
    x = df['context_size']
    power_inference_time = df['total_time']
    flashinfer_time = df['flashinfer_time']
    attention_time = df['attention_time']
    # query_state_time = df['query_state_time']
    # update_state_time = df['update_state_time']


    # Create the stack plot
    plt.plot(x, power_inference_time, 'k--', color='red', linewidth=2, marker='o', label='Power Attention', markersize=6)
    plt.plot(x, flashinfer_time, 'k-', color='blue', linewidth=2, marker='o', label='FlashInfer', markersize=6)
    plt.plot(x, attention_time, 'k-', color='green', linewidth=2, marker='o', label='Attention', markersize=6)
    # plt.plot(x, query_state_time, 'k-', color='yellow', linewidth=2, marker='o', label='Query State', markersize=6)
    # plt.plot(x, update_state_time, 'k-', color='purple', linewidth=2, marker='o', label='Update State', markersize=6)

    plt.xlabel('Context Size (tokens)')
    plt.ylabel('Time (ms)')
    plt.title(f'Power Infer vs FlashInfer\n{b=} {h=} {qhead_ratio=} {d=} {chunk_size=} {deg=} {gating=} {dtype=}')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'power_infer_vs_flashinfer_{b}_{h}_{qhead_ratio}_{d}_{chunk_size}_{deg}_{gating}_{dtype}.png', dpi=150, bbox_inches='tight')
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
    compare_with_flashinfer()
