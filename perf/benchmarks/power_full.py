import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from perf._benchmark import Measurement
from perf._registration import register_benchmark
from perf._timing import benchmark_speed
from perf._utils import describe_gpu
from perf._precision import benchmark_precision

from power_attention.power_full import (
    power_full,
    power_full_reference,
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
    'direction': ['fwd', 'bwd', 'fwd+bwd'],
}
SPEED_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

@register_benchmark(param_configs=SPEED_TEST_CASES, groups=['speed', 'power_full'])
@register_benchmark(param_configs=[{'b': 2, 't': 512, 'h': 6, 'd': 64, 'qhead_ratio': 2, 'dtype': torch.bfloat16, 'device': 'cuda', 'gating': True, 'chunk_size': 128, 'deg': 2, 'direction': 'fwd+bwd'}], groups=['speed', 'power_full'], label='mini')
def power_full_speed(direction=None, **kw):
    """Measure speed of power attention implementation.
    
    Args:
        direction: str. One of 'fwd', 'bwd', or 'fwd+bwd' to measure forward pass,
            backward pass, or combined forward+backward pass timing
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns a Measurement object with the timing in milliseconds and GPU info.
    """
    time = benchmark_speed(direction, power_full, create_inputs, kw)
    gpu_info = describe_gpu()
    return Measurement(attrs=dict(gpus=gpu_info), value=time)



shapes = [
    {'t': 512, 'chunk_size': None},
    {'t': 512, 'chunk_size': 128},
    {'t': 16384, 'chunk_size': None},
    {'t': 16384, 'chunk_size': 128},
    {'t': 16384, 'chunk_size': 1024},
]
other_param_ranges = {
    'b': [1],
    'h': [1],
    'd': [64],
    'qhead_ratio': [1],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'deg': [1, 2],
    'direction': ['fwd', 'bwd'],
    'relative': [False],
}
PRECISION_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

MINI_SETTINGS = {'b': 1, 't': 512, 'h': 1, 'd': 64, 'qhead_ratio': 1, 'dtype': torch.bfloat16, 'device': 'cuda', 'gating': True, 'chunk_size': 128, 'deg': 2}

@register_benchmark(param_configs=PRECISION_TEST_CASES, groups=['precision', 'power_full'])
@register_benchmark(param_configs=[{**MINI_SETTINGS, 'direction': 'fwd'}, {**MINI_SETTINGS, 'direction': 'bwd'}], groups=['precision', 'power_full'], label='mini')
def power_full_precision(direction=None, relative=False, **kw):
    """Measure precision of power attention implementation compared to fp32 reference.
    
    Args:
        direction: str. One of 'fwd' or 'bwd' to measure forward pass or backward pass precision
        relative: bool. If True, return the relative error instead of the absolute error
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns:
        For forward pass (direction='fwd'): Measurement of error of Y.
        For backward pass (direction='bwd'): A list of Measurement objects, one for each input
            gradient (Q, K, V and optionally log_G), containing the maximum absolute difference
            between test and reference gradients
    """
    error = benchmark_precision(direction, relative, power_full_reference, power_full, create_inputs, 
                                kw | {'dtype': torch.float32, 'chunk_size': None}, # reference is fp32 and no chunking
                                kw)
    if direction == 'fwd':
        return Measurement(value=error)
    elif direction == 'bwd':
        return [
            Measurement(attrs=dict(which=f'{inp}_grad'), value=error[inp])
            for inp in ['Q', 'K', 'V'] + (['log_G'] if kw['gating'] else [])
        ]
    else:
        raise ValueError(f"Invalid direction: {direction}")
