import torch
import pytest
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fake_fn_implementation_matches,
)

## FORWARD TESTS ##
from power_attention._query_state.fwd import (
    query_state_fwd,
    query_state_fwd_fake,
    create_inputs as create_inputs_fwd,
    compute_expanded_dim_size
)

# Define parameter ranges
param_ranges_fwd = {
    'b': [1, 2],
    'n': [8, 32],
    'c': [128],
    'h': [1, 4],
    'd': [32, 64],
    'dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda'], 
    'fused': [True, False],
    'stabilizer': [1.0, 100.0]
}
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['c']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"fused_{kw['fused']}-" \
           f"device_{kw['device']}-" \
           f"stabilizer_{kw['stabilizer']}"
FWD_TEST_CASES = [
    dict(zip(param_ranges_fwd.keys(), values))
    for values in product(*param_ranges_fwd.values())
]

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_query_state_fwd_create_inputs(kw):
    inputs = create_inputs_fwd(
        kw['b'], kw['n'], kw['c'], kw['h'], kw['d'], 
        kw['dtype'], kw['device']
    )
    D = compute_expanded_dim_size(kw['d'], inputs['deg'])
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['S'], ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device'])),
        (inputs['Y'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_query_state_fwd_output(kw):
    inputs = create_inputs_fwd(**kw)
    with torch.no_grad():
        O = query_state_fwd(**inputs)
    check_tensor_property_pairs(
        (O, ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_query_state_fwd_determinism(kw):
    check_inputs_created_determinstically(create_inputs_fwd, kw)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_query_state_fwd_compiles(kw):
    inputs = create_inputs_fwd(**kw)
    check_fn_compiles(query_state_fwd, inputs)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_query_state_fwd_fake_implementation(kw):
    inputs = create_inputs_fwd(**kw)
    check_fake_fn_implementation_matches(query_state_fwd, query_state_fwd_fake, inputs)

