import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from perf._benchmark import Measurement
from perf._registration import register_benchmark
from perf._timing import benchmark_speed
from perf._utils import describe_gpu
from perf._precision import benchmark_precision

from power_attention._query_state.impl import (
    query_state,
    create_inputs,
)
from power_attention._query_state.reference import (
    query_state_reference,
)



# Speed test configurations
shapes = [
    {'b': 64, 'n': 4, 'c': 128, 'h': 6, 'd': 64},
    {'b': 64, 'n': 1, 'c': 1024, 'h': 6, 'd': 64},
    {'b': 2, 'n': 128, 'c': 128, 'h': 6, 'd': 64},
    {'b': 2, 'n': 16, 'c': 1024, 'h': 6, 'd': 64}
]
other_param_ranges = {
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'fused': [True, False],
    'scale': [1.0],
    'direction': ['fwd', 'bwd', 'fwd+bwd'],
}
SPEED_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]
@register_benchmark(param_configs=SPEED_TEST_CASES, groups=['speed', 'query_state'])
def query_state_speed(direction=None, **kw):
    time = benchmark_speed(direction, query_state, create_inputs, kw)
    gpu_info = describe_gpu()
    return Measurement(attrs=dict(gpus=gpu_info), value=time)




# Precision test configurations
shapes = [
    {'b': 1, 'n': 1, 'c': 128, 'h': 1, 'd': 64},
    {'b': 1, 'n': 1, 'c': 2048, 'h': 1, 'd': 64},
]
other_param_ranges = {
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'fused': [True, False],
    'scale': [1.0],
    'direction': ['fwd', 'bwd'],
    'relative': [False],
}
PRECISION_TEST_CASES = [
    {**shape, **dict(zip(other_param_ranges.keys(), values))}
    for shape in shapes
    for values in product(*other_param_ranges.values())
]

@register_benchmark(param_configs=PRECISION_TEST_CASES, groups=['precision', 'query_state'])
def query_state_precision(direction=None, relative=False, **kw):
    """Measure precision of query_state implementation compared to fp32 reference.
    
    Args:
        direction: str. One of 'fwd' or 'bwd' to measure forward pass or backward pass precision
        relative: bool. If True, return the relative error instead of the absolute error
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns:
        For forward pass (direction='fwd'): Measurement of error of output.
        For backward pass (direction='bwd'): A list of Measurement objects, one for each input
            gradient (Q, S, and when fused=True, Y), containing the maximum absolute 
            difference between test and reference gradients.
    """
    def fn_with_layernorm(fn):
        def wrapper(**inputs):
            o = fn(**inputs).float()
            return (o - o.mean(-1, keepdim=True)) / o.std(-1, keepdim=True, correction=False)
        return wrapper

    # We wrap query_state inside a layernorm because the scale of the output for each
    # token is different, and the scale of the first token is usually very large, which 
    # has to do with the rowmax of the first token is usually very small, so it's scaled
    # by exp(-rowmax), which is large. This caused the scale of numerical error to be
    # different for each token, and thus hard to compare. The layernorm is purely here to
    # normalize the scale of each token, without it, query_state should still match its 
    # reference implementation.
    error = benchmark_precision(direction, relative, fn_with_layernorm(query_state_reference), fn_with_layernorm(query_state), create_inputs, 
                                kw | {'dtype': torch.float32}, # reference is fp32
                                kw)
    if direction == 'fwd':
        return Measurement(value=error)
    elif direction == 'bwd':
        return [
            Measurement(attrs=dict(which=f'{inp}_grad'), value=error[inp])
            for inp in ['Q', 'S'] + (['Y'] if kw['fused'] else [])
        ]
    else:
        raise ValueError(f"Invalid direction: {direction}")