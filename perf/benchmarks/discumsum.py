import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from perf._benchmark import Measurement
from perf._registration import register_benchmark
from perf._timing import benchmark_speed
from perf._utils import describe_gpu
from perf._precision import benchmark_precision

from power_attention._discumsum.impl import (
    discumsum,
    create_inputs,
)
from power_attention._discumsum.reference import (
    discumsum_reference,
)




# Speed test configurations
shapes = [
    {'b': 64, 'n': 4, 'h': 6, 'D': 64, 'd': 64},
    {'b': 64, 'n': 4, 'h': 6, 'D': 2040, 'd': 64},
    {'b': 2, 'n': 16, 'h': 6, 'D': 64, 'd': 64},
    {'b': 2, 'n': 16, 'h': 6, 'D': 2040, 'd': 64}
]
other_param_ranges = {
    'X_dtype': [torch.bfloat16],
    'device': ['cuda'],
    'direction': ['fwd', 'bwd', 'fwd+bwd'],
}
SPEED_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

@register_benchmark(param_configs=SPEED_TEST_CASES, groups=['speed', 'discumsum'])
def discumsum_speed(direction=None, **kw):
    """Measure speed of discumsum implementation.
    
    Args:
        direction: str. One of 'fwd', 'bwd', or 'fwd+bwd' to measure forward pass,
            backward pass, or combined forward+backward pass timing
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns a Measurement object with the timing in milliseconds and GPU info.
    """
    time = benchmark_speed(direction, discumsum, create_inputs, kw)
    gpu_info = describe_gpu()
    return Measurement(attrs=dict(gpus=gpu_info), value=time)




# Precision test configurations
shapes = [
    {'b': 1, 'n': 8, 'h': 1, 'D': 64, 'd': 64},
    {'b': 1, 'n': 8, 'h': 1, 'D': 2048, 'd': 64},
    {'b': 1, 'n': 64, 'h': 1, 'D': 64, 'd': 64},
    {'b': 1, 'n': 64, 'h': 1, 'D': 2048, 'd': 64},
]
other_param_ranges = {
    'X_dtype': [torch.bfloat16],
    'device': ['cuda'],
    'direction': ['fwd', 'bwd'],
    'relative': [False],
}
PRECISION_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

@register_benchmark(param_configs=PRECISION_TEST_CASES, groups=['precision', 'discumsum'])
def discumsum_precision(direction=None, relative=False, **kw):
    """Measure precision of discumsum implementation compared to fp32 reference.
    
    Args:
        direction: str. One of 'fwd' or 'bwd' to measure forward pass or backward pass precision
        relative: bool. If True, return the relative error instead of the absolute error
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns:
        For forward pass (direction='fwd'): Measurement of error of output.
        For backward pass (direction='bwd'): A list of Measurement objects, one for each input
            gradient (X and log_G), containing the maximum absolute difference between test
            and reference gradients.
    """
    error = benchmark_precision(direction, relative, discumsum_reference, discumsum, create_inputs, 
                                kw | {'X_dtype': torch.float32}, # reference is fp32
                                kw)
    if direction == 'fwd':
        return Measurement(value=error)
    elif direction == 'bwd':
        return [
            Measurement(attrs=dict(which=f'{inp}_grad'), value=error[inp])
            for inp in ['X', 'log_G']
        ]
    else:
        raise ValueError(f"Invalid direction: {direction}")
