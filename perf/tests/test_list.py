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
    'b': [4],
    't': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'deg': [2, 4],
    'scale': [1.0, 1/8.0],
    'gating': [False, True],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
}
ATTENTION_TEST_CASES = fn_set_and_param_range_to_test_cases(attention_fn_sets, attention_param_ranges)





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
    update_state_vidrial
)
update_state_input_output = {'create_inputs': update_state_create_inputs, 'input_properties': update_state_input_properties, 'output_properties': update_state_output_properties}
update_state_vidrial_input_output = {'create_inputs': partial(update_state_create_inputs, use_vidrial_layout=True), 'input_properties': partial(update_state_input_properties, use_vidrial_layout=True), 'output_properties': partial(update_state_output_properties, use_vidrial_layout=True)}
update_state_fn_sets = [
    {'name': 'update_state_reference', 'extends': 'update_state', 'impl': 'reference',
        'fn': update_state_reference, 'fwd': update_state_reference_fwd, **update_state_input_output},
    {'name': 'update_state_vidrial_reference', 'extends': 'update_state', 'impl': 'vidrial_reference',
        'fn': update_state_vidrial_reference, **update_state_vidrial_input_output, 'extra_params': {'d_tile': [8]}},
    # TODO: fix
    # {'name': 'update_state_cuda', 
    #     'fn': update_state_cuda, 'ref': update_state_reference, **update_state_input_output},
    {'name': 'update_state_triton', 'extends': 'update_state', 'impl': 'triton',
        'fn': update_state_triton, 'ref': update_state_reference, **update_state_input_output}
]
update_state_param_ranges = {
    'b': [1],
    'n': [4, 8], 
    'c': [1024, 4096],
    'h': [4],
    'd': [32, 64],
    'deg': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
UPDATE_STATE_TEST_CASES = fn_set_and_param_range_to_test_cases(update_state_fn_sets, update_state_param_ranges)

update_state_vidrial_fn_sets = [
    {'name': 'update_state_vidrial', 'extends': 'update_state', 'impl': 'vidrial',
        'fn': update_state_vidrial, 'ref': update_state_vidrial_reference, **update_state_vidrial_input_output, 'extra_params': {'d_tile': [8]}},
]
update_state_vidrial_param_ranges = {
    'b': [1],
    'n': [4, 4], 
    'c': [1024, 4096],
    'h': [4],
    'd': [64],
    'deg': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
UPDATE_STATE_TEST_CASES += fn_set_and_param_range_to_test_cases(update_state_vidrial_fn_sets, update_state_vidrial_param_ranges)



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
    'b': [1, 2],
    'n': [8, 32],
    'h': [4, 8],
    'D': [4],
    'd': [32, 64],
    'dtype': [torch.float16, torch.bfloat16],
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
)
query_state_input_output = {'create_inputs': query_state_create_inputs, 'input_properties': query_state_input_properties, 'output_properties': query_state_output_properties}
query_state_vidrial_input_output = {'create_inputs': partial(query_state_create_inputs, use_vidrial_layout=True), 'input_properties': partial(query_state_input_properties, use_vidrial_layout=True), 'output_properties': partial(query_state_output_properties, use_vidrial_layout=True)}
query_state_fn_sets = [
    {'name': 'query_state_reference', 'extends': 'query_state', 'impl': 'reference',
        'fn': query_state_reference, 'fwd': query_state_reference_fwd, **query_state_input_output},
    {'name': 'query_state_vidrial_reference', 'extends': 'query_state', 'impl': 'vidrial_reference',
        'fn': query_state_vidrial_reference, **query_state_vidrial_input_output},
    # TODO: fix
    # {'name': 'query_state_cuda', 
    #     'fn': query_state_cuda, 'ref': query_state_reference, **query_state_input_output},
    {'name': 'query_state_triton', 'extends': 'query_state', 'impl': 'triton',
        'fn': query_state_triton, 'ref': query_state_reference, **query_state_input_output},
]
query_state_param_ranges = {
    'b': [1],
    'n': [4, 8], 
    'c': [1024, 4096],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'zero_initial_state': [True]
}
QUERY_STATE_TEST_CASES = fn_set_and_param_range_to_test_cases(query_state_fn_sets, query_state_param_ranges)
query_state_vidrial_fn_sets = [
    {'name': 'query_state_vidrial', 'extends': 'query_state', 'impl': 'vidrial',
        'fn': query_state_vidrial, 'ref': query_state_vidrial_reference, **query_state_vidrial_input_output},
]
query_state_vidrial_param_ranges = {
    'b': [1],
    'n': [4, 8], 
    'c': [1024, 4096],
    'h': [4],
    'd': [64],
    'deg': [2],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
QUERY_STATE_TEST_CASES += fn_set_and_param_range_to_test_cases(query_state_vidrial_fn_sets, query_state_vidrial_param_ranges)





## power full

from power_attention import (
    create_inputs as power_full_create_inputs,
    input_properties as power_full_input_properties,
    output_properties as power_full_output_properties,
)
from power_attention.reference import power_full as power_full_reference
from power_attention.vidrial_reference import power_full as power_full_vidrial_reference
from power_attention.triton import power_full
power_full_input_output = {'create_inputs': power_full_create_inputs, 'input_properties': power_full_input_properties, 'output_properties': power_full_output_properties}
power_full_fn_sets = [
    {'name': 'power_full_reference', 'extends': 'power_full', 'impl': 'reference',
        'fn': power_full_reference, **power_full_input_output},
    {'name': 'power_full', 'extends': 'power_full', 'impl': 'full',
        'fn': power_full, 'ref': power_full_reference, **power_full_input_output},
    {'name': 'power_full_vidrial_reference', 'extends': 'power_full', 'impl': 'vidrial_reference',
        'fn': power_full_vidrial_reference, **power_full_input_output},
]
# Define parameter ranges
power_full_param_ranges = {
    'b': [1],
    't': [1024, 4096], 
    'h': [1],
    'd': [32, 64],
    'qhead_ratio': [1],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 1024],
    'deg': [2],
    'scale': [1/8.0],
}
POWER_FULL_TEST_CASES = fn_set_and_param_range_to_test_cases(power_full_fn_sets, power_full_param_ranges)