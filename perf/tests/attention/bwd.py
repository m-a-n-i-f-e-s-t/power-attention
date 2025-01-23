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
from power_attention._attention.bwd import (
    attention_bwd_gatingless,
    attention_bwd_gating,
    attention_bwd_gatingless_fake,
    attention_bwd_gating_fake,
    create_inputs as create_inputs_bwd,
)

param_ranges_bwd = {
    'b': [2],
    't': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'deg': [1, 2, 3, 4],
    'scale': [1.0, 1/8.0],
    'gating': [False, True],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
}
BWD_TEST_CASES = [
    dict(zip(param_ranges_bwd.keys(), values))
    for values in product(*param_ranges_bwd.values())
]
for kw in BWD_TEST_CASES:
    if kw['dtype'] == torch.float16 and kw['scale'] == 1.0:
        kw['scale'] = 1 / kw['d']**.5

def id_fn(kw):
    return f"shape_{kw['b']}_{kw['t']}_{kw['h']}_{kw['d']}-" \
           f"deg_{kw['deg']}-" \
           f"scale_{kw['scale']}-" \
           f"gating_{kw['gating']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_attention_bwd_create_inputs(kw):
    inputs = create_inputs_bwd(**kw)
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['K'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['dY'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['dy'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
        (inputs['rowmax'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
    )
    if kw['gating']:
        check_tensor_property_pairs(
            (inputs['log_G_Q'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
            (inputs['log_G_K'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
        )

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_attention_bwd_output(kw):
    inputs = create_inputs_bwd(**kw)
    if kw['gating']:
        with torch.no_grad():
            dQ, dK, dV, dlog_G = attention_bwd_gating(**inputs)
        check_tensor_property_pairs(
            (dQ, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dK, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dV, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dlog_G, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
        )
    else:
        with torch.no_grad():
            dQ, dK, dV = attention_bwd_gatingless(**inputs)
        check_tensor_property_pairs(
            (dQ, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dK, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
            (dV, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device']))
        )


@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_attention_bwd_create_inputs_determinism(kw):
    check_inputs_created_determinstically(create_inputs_bwd, kw)

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_attention_bwd_compiles(kw):
    inputs = create_inputs_bwd(**kw)
    if kw['gating']:
        check_fn_compiles(attention_bwd_gating, inputs)
    else:
        check_fn_compiles(attention_bwd_gatingless, inputs)

@pytest.mark.parametrize("kw", BWD_TEST_CASES, ids=id_fn)
def test_attention_bwd_fake_implementation(kw):
    inputs = create_inputs_bwd(**kw)
    if kw['gating']:
        check_fake_fn_implementation_matches(attention_bwd_gating, attention_bwd_gating_fake, inputs)
    else:
        check_fake_fn_implementation_matches(attention_bwd_gatingless, attention_bwd_gatingless_fake, inputs)


