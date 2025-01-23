import torch
import pytest
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_fn_forwards_match,
    check_fn_backwards_match,
)


## OP IMPL TESTS ##
from power_attention._update_state.impl import (
    update_state,
    update_state_fake,
    create_inputs as create_inputs_impl
)
from power_attention._utils import compute_expanded_dim

param_ranges_impl = {
    'b': [4],
    'n': [4, 8], 
    'c': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
}
IMPL_TEST_CASES = [
    dict(zip(param_ranges_impl.keys(), values))
    for values in product(*param_ranges_impl.values())
]
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['c']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_create_inputs(kw):
    inputs = create_inputs_impl(**kw)
    check_tensor_property_pairs(
        (inputs['K'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_output(kw):
    inputs = create_inputs_impl(**kw)
    with torch.no_grad():
        S = update_state(**inputs)
    D = compute_expanded_dim(kw['d'], inputs['deg'])
    check_tensor_property_pairs(
        (S, ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_backward(kw):
    inputs = create_inputs_impl(**kw, requires_grad=True)
    with torch.autograd.enable_grad():
        outputs = update_state(**inputs)
        torch.autograd.backward(outputs, torch.ones_like(outputs))
    check_tensor_property_pairs(
        (inputs['K'].grad, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'].grad, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_backward_determinism(kw):
    check_inputs_created_determinstically(create_inputs_impl, kw)

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_compiles_fwd(kw):
    check_fn_compiles(update_state, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_compiles_with_backward(kw):
    check_fn_compiles_with_backward(update_state, create_inputs_impl(**kw, requires_grad=True))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_update_state_fake_fn_implementation_matches(kw):
    check_fake_fn_implementation_matches(update_state, update_state_fake, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
@pytest.mark.skip(reason='Have not yet figured out how to make opcheck pass')
def test_update_state_opcheck(kw):
    torch.library.opcheck(update_state, create_inputs_impl(**kw, requires_grad=True))

## REFERENCE TESTS ##
from power_attention._update_state.reference import (
    update_state_reference,
    update_state_reference_fwd,
)

param_ranges_ref = {
    'b': [2],
    'n': [8, 32], 
    'c': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda'],
}

REF_TEST_CASES = [
    dict(zip(param_ranges_ref.keys(), values))
    for values in product(*param_ranges_ref.values())
]

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_update_state_reference_matches_autograd(kw):
    check_fn_backwards_match(
        ref_fn=update_state_reference_fwd,
        gold_inputs=create_inputs_impl(**(kw | {'dtype': torch.float32}), requires_grad=True),
        test_fn=update_state_reference,
        test_inputs=create_inputs_impl(**kw, requires_grad=True),
        rtol=4., # TODO(sean): this is pretty high, double check correctness
    )

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_update_state_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}))
    test_inputs = create_inputs_impl(**kw)

    check_fn_forwards_match(
        ref_fn=update_state_reference,
        gold_inputs=gold_inputs,
        test_fn=update_state,
        test_inputs=test_inputs,
        rtol=2.
    )

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_update_state_grad_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}), requires_grad=True)
    test_inputs = create_inputs_impl(**kw, requires_grad=True)
    check_fn_backwards_match(
        ref_fn=update_state_reference,
        gold_inputs=gold_inputs,
        test_fn=update_state,
        test_inputs=test_inputs,
        rtol=2.
    )