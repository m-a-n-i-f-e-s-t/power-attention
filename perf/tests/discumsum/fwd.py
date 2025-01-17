import torch
import pytest
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fake_fn_implementation_matches
)
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
