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
from power_attention._update_state.fwd import (
    update_state_fwd,
    update_state_fwd_fake,
    create_inputs as create_inputs_fwd
)
from power_attention._utils import compute_expanded_dim
# Define parameter ranges
param_ranges_fwd = {
    'b': [1, 2],
    'n': [8, 32],
    'c': [128],
    'h': [1, 4],
    'd': [32, 64],
    'dtype': [torch.float16, torch.bfloat16],
    'device': ['cuda'], 
}
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['c']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"

FWD_TEST_CASES = [
    dict(zip(param_ranges_fwd.keys(), values))
    for values in product(*param_ranges_fwd.values())
]

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_update_state_fwd_create_inputs(kw):
    inputs = create_inputs_fwd(
        kw['b'], kw['n'], kw['c'], kw['h'], kw['d'], 
        kw['dtype'], kw['device']
    )
    check_tensor_property_pairs(
        (inputs['K'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['n'], kw['c'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_update_state_fwd_output(kw):
    inputs = create_inputs_fwd(**kw)
    with torch.no_grad():
        S = update_state_fwd(**inputs)
    D = compute_expanded_dim(kw['d'], inputs['deg'])
    check_tensor_property_pairs(
        (S, ((kw['b'], kw['n'], kw['h'], D, kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_update_state_fwd_determinism(kw):
    check_inputs_created_determinstically(create_inputs_fwd, kw)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_update_state_fwd_compiles(kw):
    inputs = create_inputs_fwd(**kw)
    check_fn_compiles(update_state_fwd, inputs)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_update_state_fwd_fake_implementation(kw):
    inputs = create_inputs_fwd(**kw)
    check_fake_fn_implementation_matches(update_state_fwd, update_state_fwd_fake, inputs)