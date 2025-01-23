import torch
import pytest
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fake_fn_implementation_matches,
)

## BACKWARD TESTS ##
from power_attention._utils import compute_expanded_dim
from power_attention._query_state.bwd import (
    query_state_bwd,
    query_state_bwd_fake,
    create_inputs as create_inputs_bwd
)

param_ranges_bwd = {
    'b': [2],
    'n': [4, 8], 
    'c': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'dtype': [torch.bfloat16],
    'device': ['cuda'],
    'fused': [True, False],
    'scale': [1.0]
}
BWD_TEST_CASES = [
    dict(zip(param_ranges_bwd.keys(), values))
    for values in product(*param_ranges_bwd.values())
]
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['c']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"fused_{kw['fused']}-" \
           f"device_{kw['device']}-" \
           f"scale_{kw['scale']}"


@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_query_state_bwd_create_inputs(kw):
    inputs = create_inputs_bwd(**kw)
    D = compute_expanded_dim(kw['d'], 2)
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['S'], ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device'])),
        (inputs['dO'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_query_state_bwd_output(kw):
    inputs = create_inputs_bwd(**kw)
    D = compute_expanded_dim(kw['d'], 2)
    with torch.no_grad():
        dQ, dS, ds = query_state_bwd(**inputs)
        check_tensor_property_pairs(
            (dQ, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dS, ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device']))
        )

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_query_state_bwd_determinism(kw):
    check_inputs_created_determinstically(create_inputs_bwd, kw)

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_query_state_bwd_compiles(kw):
    torch._dynamo.config.cache_size_limit = 32
    inputs = create_inputs_bwd(**kw)
    check_fn_compiles(query_state_bwd, inputs, rtol=1e-2, atol=1e-2)

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_query_state_bwd_fake_implementation(kw):
    inputs = create_inputs_bwd(**kw)
    check_fake_fn_implementation_matches(query_state_bwd, query_state_bwd_fake, inputs)
