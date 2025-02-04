import torch

from perf._timing import get_compiled_version, estimate_runtime

def top_k(tensor, k):
    """Return the top k values and indices of a whole tensor"""
    strides = tensor.stride()
    vals, indices = torch.topk(tensor.flatten(), k)
    # reconstruct indices to match original tensor shape
    indices_o = []
    for idx in indices:
        idx = idx.item()
        idx_o = [0] * len(strides)
        for i, stride in enumerate(strides):
            idx_o[i] = idx // stride
            idx = idx % stride
        indices_o.append(tuple(idx_o))
    return vals, indices_o

def inspect_diff_details(gold, ref, test, tol, atol=0, num_vals=10):
    """Print details about the difference between gold, ref, and test.

    Args:
        gold: Gold tensor or iterable of tensors
        ref: Reference tensor or iterable of tensors
        test: Test tensor or iterable of tensors
        tol: float. Tolerance for difference
        num_vals: int. Number of values to print.

    Returns:
        str: Details about the difference between gold, ref, and test
    """
    if isinstance(gold, torch.Tensor):
        assert isinstance(ref, torch.Tensor) and isinstance(test, torch.Tensor), "If gold is tensor, ref and test must also be tensors"
        gold_ref_diff = torch.abs(gold - ref)
        gold_test_diff = torch.abs(gold - test)
        abs_error = (gold_test_diff - ((1 + tol) * gold_ref_diff + atol))
        violations = (abs_error > 0)
        num_violations = violations.sum().item()
        violation_pct = num_violations / gold.numel()
        if num_violations > 0:
            top_abs_errors, abs_error_indices = top_k(abs_error, k=num_vals)
            gold_vals = [gold[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            ref_vals = [ref[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            test_vals = [test[*abs_error_indices[i]].item() for i in range(len(abs_error_indices))]
            rel_error = gold_test_diff / torch.abs(gold)
            return f"\tViolations: {num_violations}, {violation_pct * 100:.2f}%\nIndices: {abs_error_indices}\nGold: {gold_vals}...\nRef: {ref_vals}...\nTest: {test_vals}... \nMax rel error: {rel_error.max().item():.4f}"
        return ""
    elif hasattr(gold, '__iter__'):
        if isinstance(gold, dict):
            keys = sorted(gold.keys())
            gold = [gold[k] for k in keys]
            ref = [ref[k] for k in keys]
            test = [test[k] for k in keys]
        msgs = [inspect_diff_details(g, r, t, tol, atol) for g, r, t in zip(gold, ref, test)]
        msg = """\nDiff details:\n"""
        for i, imsg in enumerate(msgs):
            msg += f"\n\tElement {i}/{len(msgs)}:\n\t\t{imsg}"
        return msg
    else:
        raise TypeError(f"Expected tensor or iterable of tensors, got {type(gold)}")


def print_runtime(fn, **inputs):
    fwd_fn = get_compiled_version(fn, inputs, direction='fwd')
    bwd_fn = get_compiled_version(fn, inputs, direction='bwd')
    fwdbwd_fn = get_compiled_version(fn, inputs, direction='fwd+bwd')
    fwd_time = estimate_runtime(fwd_fn)
    bwd_time = estimate_runtime(bwd_fn)
    fwdbwd_time = estimate_runtime(fwdbwd_fn)
    print(f"fwd_time: {fwd_time:.2f}ms, bwd_time: {bwd_time:.2f}ms, fwdbwd_time: {fwdbwd_time:.2f}ms")