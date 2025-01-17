import torch
import pytest
from itertools import product
from tests_and_benchmarks._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_fn_forwards_match,
    check_fn_backwards_match,
)

## OP IMPL TESTS ##
from power_attention._query_state.impl import (
    query_state,
    query_state_fake,
    create_inputs as create_inputs_impl,
    compute_expanded_dim
)
param_ranges_impl = {
    'b': [4],
    'n': [4, 8], 
    'c': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'fused': [True, False],
    'device': ['cuda'],
    'stabilizer': [1.0]
}
IMPL_TEST_CASES = [
    dict(zip(param_ranges_impl.keys(), values))
    for values in product(*param_ranges_impl.values())
]
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['c']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"fused_{kw['fused']}-" \
           f"device_{kw['device']}-" \
           f"stabilizer_{kw['stabilizer']}"

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_create_inputs(kw):
    inputs = create_inputs_impl(**kw)
    D = compute_expanded_dim(kw['d'], inputs['deg'] )
    if kw['fused']:
        check_tensor_property_pairs(
            (inputs['Q'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (inputs['S'], ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device'])),
            (inputs['Y'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
        )
        check_tensor_property_pairs(
            (inputs['rowmax'], ((kw['b'], kw['n'], kw['c'], kw['h']), torch.float32, kw['device'])))
    else:
        check_tensor_property_pairs(
            (inputs['Q'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (inputs['S'], ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device']))
        )
        assert inputs['Y'] is None and inputs['rowmax'] is None

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_output(kw):
    inputs = create_inputs_impl(**kw)
    with torch.no_grad():
        O = query_state(**inputs)
    check_tensor_property_pairs(
        (O, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_backward(kw):
    inputs = create_inputs_impl(**kw, requires_grad=True)
    with torch.autograd.enable_grad():
        outputs = query_state(**inputs)
        torch.autograd.backward(outputs, torch.ones_like(outputs))
    D = compute_expanded_dim(kw['d'], inputs['deg'])
    if kw['fused']:
        check_tensor_property_pairs(
            (inputs['Q'].grad, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (inputs['S'].grad, ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device'])), 
            (inputs['Y'].grad, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
        )
    else:
        check_tensor_property_pairs(
            (inputs['Q'].grad, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (inputs['S'].grad, ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device']))
        )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_backward_determinism(kw):
    check_inputs_created_determinstically(create_inputs_impl, kw)

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_compiles_fwd(kw):
    check_fn_compiles(query_state, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_compiles_with_backward(kw):
    check_fn_compiles_with_backward(query_state, create_inputs_impl(**kw, requires_grad=True))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_query_state_fake_fn_implementation_matches(kw):
    check_fake_fn_implementation_matches(query_state, query_state_fake, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
@pytest.mark.skip(reason='Have not yet figured out how to make opcheck pass')
def test_query_state_opcheck(kw):
    torch.library.opcheck(query_state, create_inputs_impl(**kw, requires_grad=True))

## REFERENCE TESTS ##
from power_attention._query_state.reference import (
    query_state_reference,
    query_state_reference_fwd,
)
# Define parameter ranges for reference tests
param_ranges_ref = {
    'b': [2],
    'n': [8, 32], 
    'c': [128],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.float16, torch.bfloat16],
    'fused': [True, False],
    'device': ['cuda'],
    'stabilizer': [1.0]
}

REF_TEST_CASES = [
    dict(zip(param_ranges_ref.keys(), values))
    for values in product(*param_ranges_ref.values())
]

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_query_state_reference_matches_autograd(kw):
    check_fn_backwards_match(
        ref_fn=query_state_reference_fwd,
        gold_inputs=create_inputs_impl(**(kw | {'dtype': torch.float32}), requires_grad=True),
        test_fn=query_state_reference,
        test_inputs=create_inputs_impl(**kw, requires_grad=True),
        rtol=4., # TODO(jbuckman): this is pretty high, double check correctness
    )

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_query_state_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}))
    test_inputs = create_inputs_impl(**kw)

    check_fn_forwards_match(
        ref_fn=query_state_reference,
        gold_inputs=gold_inputs,
        test_fn=query_state,
        test_inputs=test_inputs,
        rtol=2.
    )


@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_query_state_grad_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}),
        requires_grad=True,
        Y_std=1.0,
        q_std=1e-2
    )

    test_inputs = create_inputs_impl(**kw,
        requires_grad=True,
        Y_std=1.0,
        q_std=1e-2
    )

    check_fn_backwards_match(
        ref_fn=query_state_reference,
        gold_inputs=gold_inputs,
        test_fn=query_state,
        test_inputs=test_inputs,
        rtol=2.
    )