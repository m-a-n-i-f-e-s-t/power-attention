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
# Human-readable id string
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['n']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['X_dtype']}-" \
           f"device_{kw['device']}"

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

