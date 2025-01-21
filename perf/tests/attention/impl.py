import torch
import pytest
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fake_fn_implementation_matches,
    check_fn_forwards_match,
    check_fn_backwards_match,
)


## OP IMPL TESTS ##
from power_attention._attention.impl import (
    attention,
    attention_fake,
    create_inputs as create_inputs_impl
)

param_ranges_impl = {
    'b': [4],
    't': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'deg': [1, 2, 3, 4],
    'scale': [1.0, 1/8.0],
    'gating': [False, True],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
}
IMPL_TEST_CASES = [
    dict(zip(param_ranges_impl.keys(), values))
    for values in product(*param_ranges_impl.values())
]
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['t']}_{kw['h']}_{kw['d']}-" \
           f"deg_{kw['deg']}-" \
           f"scale_{kw['scale']}-" \
           f"gating_{kw['gating']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_create_inputs(kw):
    inputs = create_inputs_impl(**kw)
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['K'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_output(kw):
    inputs = create_inputs_impl(**kw)
    with torch.no_grad():
        Y, y, rowmax = attention(**inputs)
    check_tensor_property_pairs(
        (Y, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (y, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device'])),
        (rowmax, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_backward(kw):
    inputs = create_inputs_impl(**(kw | {'requires_grad': True}))
    with torch.autograd.enable_grad():
        Y, y, rowmax = attention(**inputs)
        torch.autograd.backward((Y, y), (torch.ones_like(Y), torch.ones_like(y)))
    if kw['gating']:
        check_tensor_property_pairs(
            (inputs['log_G'].grad, ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
        )
    check_tensor_property_pairs(
        (inputs['Q'].grad, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['K'].grad, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'].grad, ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_backward_determinism(kw):
    check_inputs_created_determinstically(create_inputs_impl, kw)

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_compiles_fwd(kw):
    check_fn_compiles(attention, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_compiles_with_backward(kw):
    check_fn_compiles_with_backward(attention, create_inputs_impl(**(kw | {'requires_grad': True})))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
def test_attention_fake_fn_implementation_matches(kw):
    check_fake_fn_implementation_matches(attention, attention_fake, create_inputs_impl(**kw))

@pytest.mark.parametrize("kw", IMPL_TEST_CASES, ids=id_fn)
@pytest.mark.skip(reason='Have not yet figured out how to make opcheck pass')
def test_attention_opcheck(kw):
    torch.library.opcheck(attention, create_inputs_impl(**kw, requires_grad=True))

## REFERENCE TESTS ##
from power_attention._attention.reference import (
    attention_reference,
    attention_reference_fwd,
)

param_ranges_ref = {
    'b': [4],
    't': [128, 1024],
    'h': [4],
    'd': [32, 64],
    'deg': [1, 2, 3, 4],
    'scale': [1.0, 1/8.0],
    'gating': [False, True],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
}
REF_TEST_CASES = [
    dict(zip(param_ranges_ref.keys(), values))
    for values in product(*param_ranges_ref.values())
]

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_attention_reference_matches_autograd(kw):
    check_fn_backwards_match(
        ref_fn=attention_reference_fwd,
        gold_inputs=create_inputs_impl(**(kw | {'dtype': torch.float32}), requires_grad=True),
        test_fn=attention_reference,
        test_inputs=create_inputs_impl(**kw, requires_grad=True),
        rtol=2.,
    )

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_attention_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}))
    ref_inputs = create_inputs_impl(**kw)
    
    check_fn_forwards_match(
        ref_fn=attention_reference,
        gold_inputs=gold_inputs,
        test_fn=attention,
        test_inputs=ref_inputs,
        rtol=2.
    )

@pytest.mark.parametrize("kw", REF_TEST_CASES, ids=id_fn)
def test_attention_grad_matches_reference(kw):
    gold_inputs = create_inputs_impl(**(kw | {'dtype': torch.float32}), requires_grad=True)
    test_inputs = create_inputs_impl(**kw, requires_grad=True)
    check_fn_backwards_match(
        ref_fn=attention_reference,
        gold_inputs=gold_inputs,
        test_fn=attention,
        test_inputs=test_inputs,
        rtol=2.
    )