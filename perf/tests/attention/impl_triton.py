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

param_ranges = {
    'b': [1],
    't': [256, 1024],
    'h': [1],
    'd': [32, 64, 128],
    'deg': [1, 2, 3, 4, 5],
    'dtype': [torch.bfloat16, torch.float16],
    'device': ['cuda'],
}
TEST_CASES = [
    dict(zip(param_ranges.keys(), values))
    for values in product(*param_ranges.values())
]
def id_fn(kw):
    return f"shape_{kw['b']}_{kw['t']}_{kw['h']}_{kw['d']}-" \
           f"deg_{kw['deg']}-" \
           f"dtype_{kw['dtype']}-" \
           f"device_{kw['device']}"


from power_attention._attention.impl_triton import (
    create_inputs as create_inputs_impl,
    attention,
    attention_reference
)


@pytest.mark.parametrize("kw", TEST_CASES, ids=id_fn)
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
