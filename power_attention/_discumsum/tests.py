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

## FORWARD TESTS ##
from power_attention._discumsum.fwd import (
    create_inputs as create_inputs_fwd,
    discumsum_fwd,
    discumsum_fwd_fake
)

# Define parameter ranges
param_ranges = {
    'b': [1, 2, 4],
    'n': [4, 8, 32],
    'h': [1, 4, 8], 
    'd': [16, 32, 64],
    'X_dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda']
}

# Human-readable id string
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['X_dtype']}-" \
           f"device_{kw['device']}"

# Generate all combinations
FWD_TEST_CASES = [
    dict(zip(param_ranges.keys(), values))
    for values in product(*param_ranges.values())
]

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_discumsum_fwd_create_inputs(kw):
    inputs = create_inputs_fwd(kw['b'], kw['n'], kw['h'], kw['d'], kw['X_dtype'], kw['device'])
    check_tensor_property_pairs(
        (inputs['X'], ((kw['b'], kw['n'], kw['h'], kw['d']), kw['X_dtype'], kw['device'])),
        (inputs['log_G'], ((kw['b'], kw['n'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn) 
def test_discumsum_fwd_output(kw):
    inputs = create_inputs_fwd(kw['b'], kw['n'], kw['h'], kw['d'], kw['X_dtype'], kw['device'])
    with torch.no_grad():
        cum_X = discumsum_fwd(**inputs)
    check_tensor_property_pairs(
        (cum_X, ((kw['b'], kw['n']+1, kw['h'], kw['d']), kw['X_dtype'], kw['device']))
    )
    # Check first element is zero
    assert torch.all(cum_X[:,0] == 0)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_discumsum_fwd_create_inputs_determinism(kw):
    check_inputs_created_determinstically(create_inputs_fwd, kw)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_discumsum_fwd_compiles(kw):
    check_fn_compiles(
        discumsum_fwd,
        create_inputs_fwd(**kw)
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_discumsum_fwd_fake_implementation(kw):
    check_fake_fn_implementation_matches(
        discumsum_fwd,
        discumsum_fwd_fake,
        create_inputs_fwd(**kw)
    )

## BACKWARD TESTS ##
from power_attention._discumsum.bwd import (
    create_inputs as create_inputs_bwd,
    discumsum_bwd,
    discumsum_bwd_fake
)
param_ranges_bwd = {
    'b': [1, 2, 4],
    'n': [4, 8, 32],
    'h': [1, 4, 8],
    'd': [16, 32, 64],
    'X_dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda']
}
BWD_TEST_CASES = [
    dict(zip(param_ranges_bwd.keys(), values))
    for values in product(*param_ranges_bwd.values())
]

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_discumsum_bwd_create_inputs(kw):
    inputs = create_inputs_bwd(**kw)    
    check_tensor_property_pairs(
        (inputs['dout'], ((kw['b'], kw['n']+1, kw['h'], kw['d']), kw['X_dtype'], kw['device'])),
        (inputs['out'], ((kw['b'], kw['n']+1, kw['h'], kw['d']), kw['X_dtype'], kw['device'])),
        (inputs['log_G'], ((kw['b'], kw['n'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_discumsum_bwd_output(kw):
    inputs = create_inputs_bwd(**kw)
    with torch.no_grad():
        dX, dlog_G = discumsum_bwd(**inputs)
    check_tensor_property_pairs(
        (dX, ((kw['b'], kw['n'], kw['h'], kw['d']), kw['X_dtype'], kw['device'])),
        (dlog_G, ((kw['b'], kw['n'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_discumsum_bwd_create_inputs_determinism(kw):
    check_inputs_created_determinstically(create_inputs_bwd, kw)

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_discumsum_bwd_compiles(kw):
    check_fn_compiles(discumsum_bwd, create_inputs_bwd(**kw))

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_discumsum_bwd_fake_implementation(kw):
    check_fake_fn_implementation_matches(discumsum_bwd, discumsum_bwd_fake, create_inputs_bwd(**kw))

## OP IMPL TESTS ##
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