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
from power_attention._attention.fwd import (
    attention_fwd,
    attention_fwd_fake,
    create_inputs as create_inputs_fwd,
)
# Define parameter ranges
param_ranges_fwd = {
    'b': [2],
    't': [32, 1024],
    'h': [8, 16],
    'd': [32, 64],
    'dtype': [torch.float16],
    'device': ['cuda'],
    'gating': [True, False],
    'scale': [1.0],
    'deg': [2],
}
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['t']}_{kw['h']}_{kw['d']}-" \
           f"gating_{kw['gating']}-" \
           f"scale_{kw['scale']}-" \
           f"deg_{kw['deg']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"

FWD_TEST_CASES = [
    dict(zip(param_ranges_fwd.keys(), values))
    for values in product(*param_ranges_fwd.values())
]

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_attention_fwd_create_inputs(kw):
    inputs = create_inputs_fwd(**kw)
    if kw['gating']:
        check_tensor_property_pairs(
            (inputs['log_G_Q'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
            (inputs['log_G_K'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
        )
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['K'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_attention_fwd_output(kw):
    inputs = create_inputs_fwd(**kw)
    with torch.no_grad():
        Y, y, rowmax = attention_fwd(**inputs)
    check_tensor_property_pairs(
        (Y, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (y, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
        (rowmax, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_attention_fwd_determinism(kw):
    check_inputs_created_determinstically(create_inputs_fwd, kw)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_attention_fwd_compiles(kw):
    inputs = create_inputs_fwd(**kw)
    check_fn_compiles(attention_fwd, inputs)

@pytest.mark.parametrize("kw", FWD_TEST_CASES, ids=id_fn)
def test_attention_fwd_fake_implementation(kw):
    inputs = create_inputs_fwd(**kw)
    check_fake_fn_implementation_matches(attention_fwd, attention_fwd_fake, inputs)