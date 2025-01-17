import torch
from typing import Any

def compare_tensors(ref: torch.Tensor, test: torch.Tensor, relative: bool = False) -> float:
    """Calculate maximum absolute difference between two tensors."""
    if ref.numel() == 0 and test.numel() == 0:
        return 0.
    if relative:
        diffs = (ref - test).abs() / (ref.abs() + 1e-8)
    else:
        diffs = (ref - test).abs()
    return diffs.max().item()
    
def compare_numbers(a: float, b: float, relative: bool = False) -> float:
    """Calculate maximum absolute or relative difference between two numbers."""
    if relative:
        return abs(a - b) / (abs(b) + 1e-8)
    else:
        return abs(a - b)

def compare(a: Any, b: Any, relative: bool = False) -> float:
    """Calculate maximum absolute or relative difference between tensors or pairs of tensors."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return compare_tensors(a, b, relative)
    elif a is None and b is None:
        return 0.
    elif isinstance(a, (float, int, bool)) and isinstance(b, (float, int, bool)):
        return compare_numbers(a, b, relative)
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f"Dict keys don't match: {a.keys()} vs {b.keys()}")
        return max([compare(a[k], b[k], relative) for k in a.keys()])

    try:
        return max([compare(_a, _b, relative) for _a, _b in zip(a, b, strict=True)])
    except TypeError as e:
        if "zip" not in str(e):
            raise
        raise TypeError(f"Inputs must both be tensors, dicts, or iterables of tensors. Got types: {type(a)} and {type(b)}")
