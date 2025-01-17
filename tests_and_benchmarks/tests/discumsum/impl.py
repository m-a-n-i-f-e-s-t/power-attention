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
    check_fn_backwards_match
)

from power_attention._discumsum.impl import (
    discumsum,
    discumsum_fake,
    create_inputs as create_inputs_impl
)

param_ranges_impl = {
    'b': [1, 2],
    'n': [8, 32],
    'h': [4, 8],
    'D': [4],
    'd': [32, 64],
    'X_dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda']
}

TEST_CASES_IMPL = [
    dict(zip(param_ranges_impl.keys(), values))
    for values in product(*param_ranges_impl.values())
]

# Human-readable id string
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['X_dtype']}-" \
           f"device_{kw['device']}"

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_create_inputs(kw):
    inputs = create_inputs_impl(**kw)
    check_tensor_property_pairs(
        (inputs['X'], ((kw['b'], kw['n'], kw['h'], kw['D'], kw['d']), kw['X_dtype'], kw['device'])),
        (inputs['log_G'], ((kw['b'], kw['n'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_output(kw):
    inputs = create_inputs_impl(**kw)
    with torch.no_grad():
        cum_X = discumsum(**inputs)
    check_tensor_property_pairs(
        (cum_X, ((kw['b'], kw['n']+1, kw['h'], kw['D'], kw['d']), kw['X_dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_backward(kw):
    inputs = create_inputs_impl(**kw, requires_grad=True)
    output = discumsum(**inputs)
    torch.autograd.backward(output, output)
    check_tensor_property_pairs(
        (inputs['X'].grad, ((kw['b'], kw['n'], kw['h'], kw['D'], kw['d']), kw['X_dtype'], kw['device'])),
        (inputs['log_G'].grad, ((kw['b'], kw['n'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_create_inputs_determinism(kw):
    check_inputs_created_determinstically(create_inputs_impl, kw)

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_compiles(kw):
    check_fn_compiles(discumsum, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_compiles_with_backward(kw):
    check_fn_compiles_with_backward(discumsum, create_inputs_impl(**kw, requires_grad=True))

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
def test_discumsum_fake_fn_implementation_matches(kw):
    check_fake_fn_implementation_matches(discumsum, discumsum_fake, create_inputs_impl(**kw))

## REFERENCE TESTS ##
from power_attention._discumsum.reference import (
    discumsum_reference
)

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
@pytest.mark.parametrize("compile", [False, True])
def test_discumsum_matches_reference(kw, compile):
    check_fn_forwards_match(
        ref_fn=discumsum_reference,
        gold_inputs=create_inputs_impl(**(kw | {'X_dtype': torch.float32})),
        test_fn=torch.compile(discumsum) if compile else discumsum,
        test_inputs=create_inputs_impl(**kw),
        rtol=1.01,
        atol=1e-3
    )

@pytest.mark.parametrize("kw", TEST_CASES_IMPL, ids=id_fn)
@pytest.mark.parametrize("compile", [False, True])
def test_discumsum_grad_matches_reference(kw, compile):
    # TODO (sean): the atomicAdd for dlog_G makes the gradient slightly worse than reference, hence the high tol. If this becomes a problem, fix it.
    check_fn_backwards_match(
        ref_fn=discumsum_reference,
        gold_inputs=create_inputs_impl(**(kw | {'X_dtype': torch.float32}), requires_grad=True),
        test_fn=torch.compile(discumsum) if compile else discumsum,
        test_inputs=create_inputs_impl(**kw, requires_grad=True),
        rtol=2.,
    )