import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from tests_and_benchmarks._benchmark import Measurement
from tests_and_benchmarks._registration import register_benchmark
from tests_and_benchmarks._timing import get_timing_functions

from power_attention.power_full import (
    power_full,
    create_inputs,
)

shapes = [
    {'b': 64, 't': 512,  'h': 12, 'd': 64, 'qhead_ratio': 1},
    {'b': 64, 't': 512,  'h':  6, 'd': 64, 'qhead_ratio': 2},
    {'b': 2, 't': 16384, 'h': 12, 'd': 64, 'qhead_ratio': 1}, 
    {'b': 2, 't': 16384, 'h':  6, 'd': 64, 'qhead_ratio': 2}
]
other_param_ranges = {
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 128],
    'deg': [1, 2],
    'log_space': [False, True],
}
TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

@register_benchmark(param_configs=TEST_CASES)
def power_full_speed(**kw):
    inputs = create_inputs(**kw, requires_grad=True)
    fwd_timing_fn, bwd_timing_fn, fwd_bwd_timing_fn = get_timing_functions(power_full, inputs)
    fwd_time = fwd_timing_fn()
    bwd_time = bwd_timing_fn()
    fwd_bwd_time = fwd_bwd_timing_fn()
    return [
        Measurement(attrs=dict(direction='fwd'), value=fwd_time),
        Measurement(attrs=dict(direction='bwd'), value=bwd_time),
        Measurement(attrs=dict(direction='fwd+bwd'), value=fwd_bwd_time)
    ]