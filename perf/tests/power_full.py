import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
from perf._checks import (
    check_tensor_property_pairs,
    check_inputs_created_determinstically,
    check_fn_compiles,
    check_fn_compiles_with_backward,
    check_fn_forwards_match,
    check_fn_backwards_match,
    check_inputs_forwards_match,
    check_inputs_backwards_match,
)

## POWER FULL TESTS ##
from power_attention.power_full import (
    power_full,
    power_full_reference,
    create_inputs,
)

# Define parameter ranges
param_ranges = {
    'b': [2],
    't': [512, 1024], 
    'h': [4],
    'd': [32, 64],
    'qhead_ratio': [1, 2],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
    'gating': [False, True],
    'chunk_size': [None, 128],
    'deg': [1, 2],
}
# Human-readable id string
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['t']}_{kw['h']}_{kw['d']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}-" \
           f"qhead_ratio_{kw['qhead_ratio']}-" \
           f"gating_{kw['gating']}-" \
           f"chunk_size_{kw['chunk_size']}-" \
           f"deg_{kw['deg']}"
# Generate all combinations
TEST_CASES = [
    dict(zip(param_ranges.keys(), values))
    for values in product(*param_ranges.values())
]

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_create_inputs(kw):
    inputs = create_inputs(**kw)
    check_tensor_property_pairs(
        (inputs['Q'], ((kw['b'], kw['t'], kw['h'] * kw['qhead_ratio'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['K'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device'])),
        (inputs['V'], ((kw['b'], kw['t'], kw['h'], kw['d']), kw['dtype'], kw['device']))
    )
    if kw['gating']:
        check_tensor_property_pairs(
            (inputs['log_G'], ((kw['b'], kw['t'], kw['h']), torch.float32, kw['device']))
        )

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_output(kw):
    inputs = create_inputs(**kw)
    with torch.no_grad():
        Y = power_full(**inputs)
    check_tensor_property_pairs(
        (Y, ((kw['b'], kw['t'], kw['h'] * kw['qhead_ratio'], kw['d']), kw['dtype'], kw['device']))
    )

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_create_inputs_determinism(kw):
    check_inputs_created_determinstically(create_inputs, kw)

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_compiles(kw):
    check_fn_compiles(
        power_full, 
        create_inputs(**kw) 
    )

@pytest.mark.skip(reason='Skipping test for now, compile changes the numerics')
@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_compiled_matches_eager(kw):
    check_fn_compiles(
        power_full,
        create_inputs(**kw), 
        # The high atol and rtol here is because torch.compile makes the computation numerically different than eager mode. See https://github.com/pytorch/pytorch/issues/141436
        rtol=.1, atol=.1
    )

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_backward_compiles(kw):
    check_fn_compiles_with_backward(
        power_full, 
        create_inputs(**kw, requires_grad=True)
    )

@pytest.mark.skip(reason='Skipping test for now, compile changes the numerics')
@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_backward_compiled_matches_eager(kw):
    check_fn_compiles_with_backward(
        power_full, 
        create_inputs(**kw, requires_grad=True), 
        rtol=4e-2, atol=5e-3, grad_scale=1e-5
    )

## CONSISTENCY TESTS ##
# These tests confirm that the reference implementation is invariant to chunking, modulo layernorm.


# TODO(sean): find a better place for this test
# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_full_reference_log_space_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}))
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}))

#     check_inputs_forwards_match(
#         fn=power_full_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-1,
#     )

# @pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
# def test_power_full_reference_log_space_grad_consistency(kw):
#     if kw['log_space'] is True: pytest.skip("Skipping test for log_space=True")
#     inputs_log_space = create_inputs(**(kw | {'log_space': True, 'dtype': torch.float32}), requires_grad=True)
#     inputs_normal_space = create_inputs(**(kw | {'log_space': False, 'dtype': torch.float32}), requires_grad=True)
#     check_inputs_backwards_match(
#         fn=power_full_reference,
#         inputs1=inputs_log_space,
#         inputs2=inputs_normal_space,
#         atol=1e-3,
#     )


@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_reference_chunk_size_consistency(kw):
    if kw['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = create_inputs(**(kw | {'chunk_size': None, 'dtype': torch.float32}))
    inputs_recurrent = create_inputs(**(kw | {'dtype': torch.float32}))
    check_inputs_forwards_match(
        fn=power_full_reference,
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-1,
    )

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
def test_power_full_reference_chunk_size_grad_consistency(kw):
    if kw['chunk_size'] is None: pytest.skip("Skipping test for chunk_size=None, because it is vacuously true")
    inputs_attention = create_inputs(**(kw | {'chunk_size': None, 'dtype': torch.float32}), requires_grad=True)
    inputs_recurrent = create_inputs(**(kw | {'dtype': torch.float32}), requires_grad=True)
    check_inputs_backwards_match(
        fn=power_full_reference,
        inputs1=inputs_attention,
        inputs2=inputs_recurrent,
        atol=1e-3,
    )

# TODO(jbuckman): find the right place for this test
# from state_kernel._chunk_state.reference import (
#     SymmetricPowerChunkStateReference
# )
# from state_kernel._chunk_state.impl import create_inputs as create_inputs_impl_cs
# from state_kernel._query_state.reference import (
#     QueryStateReference
# )
# 
# @pytest.mark.parametrize("b", [1, 2, 4])
# @pytest.mark.parametrize("n", [1, 4])
# @pytest.mark.parametrize("c", [16])
# @pytest.mark.parametrize("h", [1, 4])
# @pytest.mark.parametrize("d", [32, 64])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("device", ['cuda'])    
# @pytest.mark.parametrize("deg", [1, 2])
# def test_expansion(b, n, c, h, d, dtype, device, deg):
#     torch.manual_seed(42)
#     K, V, deg = create_inputs_impl_cs(b, n, c, h, d, dtype, device, deg)
#     phi_K1 = SymmetricPowerChunkStateReference.expand(K, deg) # [b, n, h, c, D]
#     phi_K2 = QueryStateReference.expand(K, deg) # [b, n, h, c, D]
#     prod = phi_K1 @ phi_K2.transpose(-1, -2) # [b, n, h, c, c]
#     prod_ref = (K @ K.transpose(-1, -2))**2 # [b, n, h, c, c]
#     torch.testing.assert_close(prod, prod_ref, rtol=1e-2, atol=1e-4)



## REFERENCE TESTS ##
# These tests are for comparing the outputs of the kernel to the reference implementation.

@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
@pytest.mark.parametrize("compile", [False, True])
def test_power_full_kernel_matches_reference(kw, compile):
    gold_inputs = create_inputs(**(kw | {'dtype': torch.float32}))
    test_inputs = create_inputs(**kw)
    check_fn_forwards_match(
        ref_fn=power_full_reference,
        gold_inputs=gold_inputs,
        test_fn=torch.compile(power_full) if compile else power_full,
        test_inputs=test_inputs,
        rtol=3., # if test error is more than 3x reference error, then it is probably a real failure
    )


@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
@pytest.mark.parametrize("compile", [False, True])
def test_power_full_kernel_grad_matches_reference(kw, compile):
    gold_inputs = create_inputs(requires_grad=True, **(kw | {'dtype': torch.float32}))
    test_inputs = create_inputs(requires_grad=True, **kw)
    check_fn_backwards_match(
        ref_fn=power_full_reference,
        gold_inputs=gold_inputs,
        test_fn=torch.compile(power_full) if compile else power_full,
        test_inputs=test_inputs,
        rtol=2, # if test error is more than 2x reference error, then it is probably a real failure
        atol=3e-4
    )
