import torch
import pytest
from perf._checks import (
    check_inputs_forwards_match,
    check_inputs_backwards_match,
)
from perf.tests.test_list import power_full_param_ranges, power_full_input_output, fn_set_and_param_range_to_test_cases
from power_attention.vidrial_reference import power_full as power_full_vidrial_reference
from power_attention.vidrial_fused_reference import power_full as power_full_vidrial_fused_reference
from power_attention.reference import power_full as power_full_reference
power_full_fn_sets = [
    {'name': 'power_full_reference', 'extends': 'power_full', 'impl': 'reference',
        'fn': power_full_reference, **power_full_input_output},
    {'name': 'power_full_vidrial_reference', 'extends': 'power_full', 'impl': 'vidrial_reference',
        'fn': power_full_vidrial_reference, **power_full_input_output},
    {'name': 'power_full_vidrial_fused_reference', 'extends': 'power_full', 'impl': 'vidrial_fused_reference',
        'fn': power_full_vidrial_fused_reference, **power_full_input_output},
]
TEST_CASES = fn_set_and_param_range_to_test_cases(power_full_fn_sets, power_full_param_ranges)

## CHUNK CONSISTENCY TESTS ##
# These tests confirm that the reference implementation is invariant to chunking

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_consistency_wrt_chunk_size(fns_params):
    fns, params = fns_params
    if params['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = fns['create_inputs'](**(params | {'chunk_size': None, 'dtype': torch.float32}))
    inputs_recurrent = fns['create_inputs'](**(params | {'dtype': torch.float32}))
    check_inputs_forwards_match(
        fn=fns['fn'],
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-1,
    )

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_grad_consistency_wrt_chunk_size(fns_params):
    fns, params = fns_params
    if params['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = fns['create_inputs'](**(params | {'chunk_size': None, 'dtype': torch.float32}), requires_grad=True)
    inputs_recurrent = fns['create_inputs'](**(params | {'dtype': torch.float32}), requires_grad=True)
    check_inputs_backwards_match(
        fn=fns['fn'],
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-2,
    )


# TODO(sean): find a better place for this test
# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_full_reference_log_space_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}))
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}))

#     check_inputs_forwards_match(
#         fn=power_full_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-1,
#     )

# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_full_reference_log_space_grad_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}), requires_grad=True)
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}), requires_grad=True)
#     check_inputs_backwards_match(
#         fn=power_full_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-3,
#     )
