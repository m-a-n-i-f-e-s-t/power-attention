import torch
import pytest
import gc
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fn_forwards_match,
    check_fn_backwards_match,
)
from perf.tests.test_list import ATTENTION_TEST_CASES, UPDATE_STATE_TEST_CASES, DISCUMSUM_TEST_CASES, QUERY_STATE_TEST_CASES, POWER_FULL_TEST_CASES

TEST_CASES = ATTENTION_TEST_CASES + UPDATE_STATE_TEST_CASES + DISCUMSUM_TEST_CASES + QUERY_STATE_TEST_CASES + POWER_FULL_TEST_CASES


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_input_properties(fns_params):
    fns, params = fns_params
    inputs = fns['create_inputs'](**params)
    desired_input_properties = fns['input_properties'](**params)
    check_tensor_property_pairs(
        *[(inputs[k], desired_input_properties[k]) for k in desired_input_properties]
    )
    check_inputs_created_determinstically(fns['create_inputs'], params)
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_output_properties(fns_params):
    fns, params = fns_params
    inputs = fns['create_inputs'](**params)
    with torch.no_grad():
        outputs = fns['fn'](**inputs)
    desired_output_properties = fns['output_properties'](**params)
    if len(desired_output_properties) == 1: outputs = [outputs]
    check_tensor_property_pairs(
        *zip(outputs, desired_output_properties.values())
    )
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_backward_properties(fns_params):
    fns, params = fns_params
    inputs = fns['create_inputs'](**params, requires_grad=True)
    with torch.autograd.enable_grad():
        outputs = fns['fn'](**inputs)
        if isinstance(outputs, torch.Tensor): outputs = [outputs]
        outputs = [o for o in outputs if o is not None and o.requires_grad]
        torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])
    desired_input_properties = fns['input_properties'](**params)
    check_tensor_property_pairs(
        *[(inputs[k].grad, desired_input_properties[k]) for k in desired_input_properties if inputs[k].grad is not None]
    )
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_compiles_fwd(fns_params):
    fns, params = fns_params
    check_fn_compiles(fns['fn'], fns['create_inputs'](**params))
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_compiles_with_backward(fns_params):
    fns, params = fns_params
    check_fn_compiles_with_backward(fns['fn'], fns['create_inputs'](**params, requires_grad=True))
    cleanup()

@pytest.mark.skip(reason='Have not yet figured out how to make opcheck pass')
@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_opcheck(fns_params):
    fns, params = fns_params
    torch.library.opcheck(fns['fn'], fns['create_inputs'](**params, requires_grad=True))
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_bwd_matches_autograd(fns_params):
    fns, params = fns_params
    if 'fwd' not in fns: pytest.skip('No autograd-compatible forward implementation for this function')
    check_fn_backwards_match(
        ref_fn=fns['fwd'],
        gold_inputs=fns['create_inputs'](**(params | {'dtype': torch.float32}), requires_grad=True),
        test_fn=fns['fn'],
        test_inputs=fns['create_inputs'](**params, requires_grad=True),
        rtol=4., # TODO(sean): this is pretty high, double check correctness
    )
    cleanup()


@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_fwd_matches_reference(fns_params):
    fns, params = fns_params
    if 'ref' not in fns: pytest.skip('No reference implementation for this function (typically because it is itself a reference)')
    torch.compiler.reset() # TODO(sean): figure out why this is needed to make triton pass
    gold_inputs = fns['create_inputs'](**(params | {'dtype': torch.float32}))
    test_inputs = fns['create_inputs'](**params)
    check_fn_forwards_match(
        ref_fn=fns['ref'],
        gold_inputs=gold_inputs,
        test_fn=fns['fn'],
        test_inputs=test_inputs,
        rtol=4.0,
        atol=1e-3,
    )
    cleanup()

@pytest.mark.parametrize("fns_params", TEST_CASES)
def test_bwd_matches_reference(fns_params):
    fns, params = fns_params
    if 'ref' not in fns: pytest.skip('No reference implementation for this function (typically because it is itself a reference)')
    if 'fwd_only' in fns: pytest.skip('Fwd-only implementation, skipping backward test')
    torch.compiler.reset() # TODO(sean): figure out why this is needed to make triton pass
    gold_inputs = fns['create_inputs'](**(params | {'dtype': torch.float32}), requires_grad=True)
    test_inputs = fns['create_inputs'](**params, requires_grad=True)
    check_fn_backwards_match(
        ref_fn=fns['ref'],
        gold_inputs=gold_inputs,
        test_fn=fns['fn'],
        test_inputs=test_inputs,
        rtol=4.0,
        atol=1e-3,
    )
    cleanup()
