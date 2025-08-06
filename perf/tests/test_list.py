from itertools import product
from functools import partial
import torch
import pytest

def and_extra_params(fns, params):
    if 'extra_params' in fns:
        return params | fns['extra_params']
    return params

def fn_set_and_param_range_to_test_cases(fn_sets, param_ranges):
    return [
        pytest.param((fns, params), id=id_fn((fns, params)), marks=[getattr(pytest.mark, fns['name']),
                                                                    getattr(pytest.mark, fns['extends']),
                                                                    getattr(pytest.mark, fns['impl'])])
        for fns in fn_sets
        for params in [
            dict(zip(and_extra_params(fns, param_ranges).keys(), values))
            for values in product(*and_extra_params(fns, param_ranges).values())
        ]
    ]

def id_fn(fns_params):
    fns, params = fns_params
    return f"{fns['name']}-" + '-'.join([f"{k}_{params[k]}" for k in params])



## attention ##

from power_attention._attention import (
    create_inputs as attention_create_inputs,
    input_properties as attention_input_properties,
    output_properties as attention_output_properties,
    attention_reference,
    attention_reference_fwd,
    # attention_cuda,
    attention_triton,
)
attention_input_output = {'create_inputs': attention_create_inputs, 'input_properties': attention_input_properties, 'output_properties': attention_output_properties}
attention_fn_sets = [
    {'name': 'attention_reference', 'extends': 'attention', 'impl': 'reference',
        'fn': attention_reference, 'fwd': attention_reference_fwd, **attention_input_output},
    # TODO: fix
    # {'name': 'attention_cuda', 
    #     'fn': attention_cuda, 'ref': attention_reference, **attention_input_output},
    {'name': 'attention_triton', 'extends': 'attention', 'impl': 'triton',
        'fn': attention_triton, 'ref': attention_reference, **attention_input_output},
]
attention_param_ranges = {
    'b': [1],
    't': [128, 1024],
    'h': [2],
    'd': [32, 64],
    'deg': [2, 4],
    'scale': [1.0, 1/8.0],
    'gating': [False, True],
    'norm': [True, False],
    'causal': [True],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'qhead_ratio': [1, 8],
}
ATTENTION_TEST_CASES = fn_set_and_param_range_to_test_cases(attention_fn_sets, attention_param_ranges)
attention_inference_input_output = {'create_inputs': partial(attention_create_inputs, inference=True), 'input_properties': partial(attention_input_properties, inference=True), 'output_properties': partial(attention_output_properties, inference=True)}
attention_inference_fn_sets = [
    {'name': 'attention_triton_inference', 'extends': 'attention', 'impl': 'triton',
        'fn': attention_triton, 'ref': attention_reference, **attention_inference_input_output, 'fwd_only': True},
]
attention_inference_param_ranges = {
    'b': [1],
    't': [64],
    'h': [2],
    'd': [64],
    'deg': [2, 4],
    'scale': [1/8.0],
    'gating': [True],
    'norm': [True, False],
    'causal': [True],
    'dtype': [torch.bfloat16],
    'qhead_ratio': [8],
    'device': ['cuda'],
}
ATTENTION_TEST_CASES += fn_set_and_param_range_to_test_cases(attention_inference_fn_sets, attention_inference_param_ranges)




## update_state ##
from power_attention._update_state import (
    create_inputs as update_state_create_inputs,
    input_properties as update_state_input_properties,
    output_properties as update_state_output_properties,
    update_state_reference,
    update_state_reference_fwd,
    # update_state_cuda,
    update_state_triton,
    update_state_vidrial_reference,
    update_state_vidrial,
    update_state_vidrial_fused,
    update_state_reference_vidrial_fused
)
update_state_input_output = {'create_inputs': update_state_create_inputs, 'input_properties': update_state_input_properties, 'output_properties': update_state_output_properties}
update_state_vidrial_input_output = {'create_inputs': partial(update_state_create_inputs, use_vidrial_layout=True), 'input_properties': partial(update_state_input_properties, use_vidrial_layout=True), 'output_properties': partial(update_state_output_properties, use_vidrial_layout=True)}
update_state_vidrial_fused_input_output = {'create_inputs': partial(update_state_create_inputs, use_vidrial_layout=True), 'input_properties': partial(update_state_input_properties, use_vidrial_layout=True), 'output_properties': partial(update_state_output_properties, use_vidrial_layout=True, fused=True)}
update_state_fn_sets = [
    {'name': 'update_state_reference', 'extends': 'update_state', 'impl': 'reference',
        'fn': update_state_reference, 'fwd': update_state_reference_fwd, **update_state_input_output},
    {'name': 'update_state_vidrial_reference', 'extends': 'update_state', 'impl': 'vidrial_reference',
        'fn': update_state_vidrial_reference, **update_state_vidrial_input_output, 'extra_params': {'d_tile': [8]}},
    {'name': 'update_state_triton', 'extends': 'update_state', 'impl': 'triton',
        'fn': update_state_triton, 'ref': update_state_reference, **update_state_input_output},
    {'name': 'update_state_vidrial_fused', 'extends': 'update_state', 'impl': 'vidrial_fused',
     'fn': update_state_vidrial_fused, 'ref': update_state_reference_vidrial_fused, **update_state_vidrial_fused_input_output},
]
update_state_param_ranges = {
    'b': [1],
    'n': [2, 4], 
    'c': [128, 1024],
    'h': [2],
    'd': [32, 64],
    'deg': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
UPDATE_STATE_TEST_CASES = fn_set_and_param_range_to_test_cases(update_state_fn_sets, update_state_param_ranges)



## discumsum ##
from power_attention._discumsum import (
    create_inputs as discumsum_create_inputs,
    input_properties as discumsum_input_properties,
    output_properties as discumsum_output_properties,
    discumsum_reference,
    # discumsum as discumsum_cuda,
    discumsum_triton
)
discumsum_input_output = {'create_inputs': discumsum_create_inputs, 'input_properties': discumsum_input_properties, 'output_properties': discumsum_output_properties}
discumsum_fn_sets = [
    # {'name': 'discumsum_cuda', 'extends': 'discumsum', 'impl': 'cuda',
    #     'fn': discumsum_cuda, 'ref': discumsum_reference, **discumsum_input_output},
    {'name': 'discumsum_triton', 'extends': 'discumsum', 'impl': 'triton',
        'fn': discumsum_triton, 'ref': discumsum_reference, **discumsum_input_output},
]
discumsum_param_ranges = {
    'b': [2],
    'n': [1, 8],
    'h': [1, 2],
    'D': [4],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'device': ['cuda']
}
DISCUMSUM_TEST_CASES = fn_set_and_param_range_to_test_cases(discumsum_fn_sets, discumsum_param_ranges)






## query_state ##
from power_attention._query_state import (
    create_inputs as query_state_create_inputs,
    input_properties as query_state_input_properties,
    output_properties as query_state_output_properties,
    query_state_reference,
    query_state_reference_fwd,
    # query_state_cuda,
    query_state_triton,
    query_state_vidrial_reference,
    query_state_vidrial,
    query_state_vidrial_fused,
    query_state_vidrial_fused_reference,
)
query_state_input_output = {'create_inputs': query_state_create_inputs, 'input_properties': query_state_input_properties, 'output_properties': query_state_output_properties}
query_state_vidrial_input_output = {'create_inputs': partial(query_state_create_inputs, use_vidrial_layout=True), 'input_properties': partial(query_state_input_properties, use_vidrial_layout=True), 'output_properties': partial(query_state_output_properties, use_vidrial_layout=True)}
query_state_fused_input_output = {'create_inputs': partial(query_state_create_inputs, fused_norm=True, use_vidrial_layout=True), 'input_properties': partial(query_state_input_properties, fused_norm=True, use_vidrial_layout=True), 'output_properties': partial(query_state_output_properties)}
query_state_fn_sets = [
    {'name': 'query_state_reference', 'extends': 'query_state', 'impl': 'reference',
        'fn': query_state_reference, 'fwd': query_state_reference_fwd, **query_state_input_output},
    {'name': 'query_state_vidrial_reference', 'extends': 'query_state', 'impl': 'vidrial_reference',
        'fn': query_state_vidrial_reference, **query_state_vidrial_input_output},
    {'name': 'query_state_triton', 'extends': 'query_state', 'impl': 'triton',
        'fn': query_state_triton, 'ref': query_state_reference, **query_state_input_output},
    {'name': 'query_state_vidrial_fused', 'extends': 'query_state', 'impl': 'vidrial_fused',
        'fn': query_state_vidrial_fused, 'ref': query_state_vidrial_fused_reference, **query_state_fused_input_output},
]
query_state_param_ranges = {
    'b': [1],
    'n': [2, 4], 
    'c': [128, 1024],
    'h': [2],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'zero_initial_state': [True]
}
QUERY_STATE_TEST_CASES = fn_set_and_param_range_to_test_cases(query_state_fn_sets, query_state_param_ranges)

query_state_inference_input_output = {'create_inputs': partial(query_state_create_inputs, inference=True), 'input_properties': partial(query_state_input_properties, inference=True), 'output_properties': partial(query_state_output_properties, inference=True)}
query_state_inference_fn_sets = [
    {'name': 'query_state_vidrial_fused_inference', 'extends': 'query_state', 'impl': 'vidrial_fused',
        'fn': query_state_vidrial_fused, 'ref': query_state_vidrial_fused_reference, **query_state_fused_input_output, 'fwd_only': True},
]
query_state_inference_param_ranges = {
    'b': [32],
    'n': [1], 
    'c': [128],
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [1, 16],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
QUERY_STATE_TEST_CASES += fn_set_and_param_range_to_test_cases(query_state_inference_fn_sets, query_state_inference_param_ranges)



## power full

from power_attention import (
    create_inputs as power_full_create_inputs,
    create_inference_inputs as power_full_create_inputs_inference,
    input_properties as power_full_input_properties,
    output_properties as power_full_output_properties,
    inference_input_properties as power_full_inference_input_properties,
    inference_output_properties as power_full_inference_output_properties,
)
from power_attention.reference import power_full as power_full_reference
from power_attention.vidrial_fused import power_full as power_full_vidrial_fused, power_full_inference as power_full_vidrial_fused_inference
from power_attention.vidrial_fused_reference import power_full as power_full_vidrial_fused_reference, power_full_inference as power_full_vidrial_fused_inference_reference
from power_attention.triton import power_full
power_full_input_output = {'create_inputs': power_full_create_inputs, 'input_properties': power_full_input_properties, 'output_properties': power_full_output_properties}
power_full_fn_sets = [
    {'name': 'power_full_reference', 'extends': 'power_full', 'impl': 'reference',
        'fn': power_full_reference, **power_full_input_output},
    {'name': 'power_full', 'extends': 'power_full', 'impl': 'full',
        'fn': power_full, 'ref': power_full_reference, **power_full_input_output},
    {'name': 'power_full_vidrial_fused_reference', 'extends': 'power_full', 'impl': 'vidrial_fused_reference',
        'fn': power_full_vidrial_fused_reference, **power_full_input_output},
    {'name': 'power_full_vidrial_fused', 'extends': 'power_full', 'impl': 'vidrial_fused',
        'fn': power_full_vidrial_fused, **power_full_input_output},
]
# Define parameter ranges
power_full_param_ranges = {
    'b': [1],
    't': [128, 1024], 
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [1],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 128],
    'deg': [2]
}
POWER_FULL_TEST_CASES = fn_set_and_param_range_to_test_cases(power_full_fn_sets, power_full_param_ranges)

power_full_inference_input_output = {'create_inputs': power_full_create_inputs_inference, 'input_properties': power_full_inference_input_properties, 'output_properties': power_full_inference_output_properties}
power_full_inference_fn_sets = [
    {'name': 'power_full_vidrial_fused_inference_reference', 'extends': 'power_full', 'impl': 'vidrial_fused_reference', 
        'fn': power_full_vidrial_fused_inference_reference, **power_full_inference_input_output, 'fwd_only': True},
    {'name': 'power_full_vidrial_fused_inference', 'extends': 'power_full', 'impl': 'vidrial_fused', 
        'fn': power_full_vidrial_fused_inference, 'ref': power_full_vidrial_fused_inference_reference, **power_full_inference_input_output, 'fwd_only': True},
]
power_full_inference_param_ranges = {
    'b': [1],
    't': [1, 64, 128],
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [1, 8],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [True],
    'chunk_size': [128],
    'deg': [2],
    'initial_state': [True, False],
}
power_full_inference_param_ranges_no_cache = {
    'b': [1],
    't': [0],
    'h': [2],
    'd': [32, 64],
    'qhead_ratio': [1, 8],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [True],
    'chunk_size': [128],
    'deg': [2],
    'initial_state': [True],
}
POWER_FULL_TEST_CASES += fn_set_and_param_range_to_test_cases(power_full_inference_fn_sets, power_full_inference_param_ranges)
POWER_FULL_TEST_CASES += fn_set_and_param_range_to_test_cases(power_full_inference_fn_sets, power_full_inference_param_ranges_no_cache)
