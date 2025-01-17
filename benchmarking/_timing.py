from collections.abc import Iterable

import torch


def get_compiled_versions(fn, grads, *args, warmup=3, **kwargs):
    """Takes a function and args andreturns compiled versions for fwd, bwd, and fwd+bwd passes.

    Args:
        fn: Function to compile

    Returns:
        Tuple of (fwd_fn, bwd_fn, fwd_bwd_fn)
    """
    # Run forward pass to identify output shapes and check for mutations
    original_args = args
    original_grads = grads
    args = tuple(arg.clone().detach().requires_grad_() if isinstance(arg, torch.Tensor) else arg for arg in args)
    grads = clone_or_none(grads)
    out = fn(*args, **kwargs)
    pruned_out, pruned_grads = prune_non_tensors(out, grads)
    torch.autograd.backward(pruned_out, grad_tensors=pruned_grads, retain_graph=True)
    # Check that args match original args
    check_tensors_unchanged(args, original_args, f'({fn}) ')
    check_tensors_unchanged(grads, original_grads, f'({fn}) Gradient ')
    # Define functions
    def fwd():
        with torch.no_grad():
            return fn(*args, **kwargs)
    torch._dynamo.config.compiled_autograd = True
    def bwd():
        torch.autograd.backward(pruned_out, grad_tensors=pruned_grads, retain_graph=True)
    torch._dynamo.config.compiled_autograd = False
    def fwd_bwd():
        out = fn(*args, **kwargs)
        pruned_out, pruned_grads = prune_non_tensors(out, grads)
        torch.autograd.backward(pruned_out, grad_tensors=pruned_grads)
    # Compile functions
    compiled_fwd = torch.compile(fwd, dynamic=False)
    compiled_bwd = torch.compile(bwd, dynamic=False)
    compiled_fwd_bwd = torch.compile(fwd_bwd, dynamic=False)
    # Warmup passes
    for _ in range(warmup):
        compiled_fwd()
        compiled_bwd()
        compiled_fwd_bwd()
    # Return compiled functions
    return (compiled_fwd, compiled_bwd, compiled_fwd_bwd)

def clone_or_none(x):
    """Helper function to clone a tensor or iterable of tensors if they exist, otherwise return None."""
    if x is None:
        return None
    elif isinstance(x, torch.Tensor):
        return x.clone()
    else:
        return tuple(clone_or_none(item) for item in x)

def prune_non_tensors(out, grads=None):
    if grads is None:
        if not isinstance(out, torch.Tensor):
            out = tuple(o for o in out if isinstance(o, torch.Tensor))
            return None
        return None
    else:
        if not isinstance(out, torch.Tensor):
            out, grads = zip(*tuple((o, g) for o, g in zip(out, grads, strict=False) if isinstance(o, torch.Tensor)), strict=False)
        return out, grads


def check_tensors_unchanged(tensor1, tensor2, prefix=''):
    assert type(tensor1) == type(tensor2), f'Mismatch in inputs: {type(tensor1)=}, {type(tensor2)=}'
    if isinstance(tensor1, torch.Tensor):
        assert tensor1.shape == tensor2.shape, f'{prefix}Functions must not mutate their inputs: Tensor shapes were modified'
        assert tensor1.dtype == tensor2.dtype, f'{prefix}Functions must not mutate their inputs: Tensor dtypes were modified'
        assert tensor1.device == tensor2.device, f'{prefix}Functions must not mutate their inputs: Tensor devices were modified'
        assert tensor1.stride() == tensor2.stride(), f'{prefix}Functions must not mutate their inputs: Tensor strides were modified'
    elif isinstance(tensor1, Iterable):
        for t, o in zip(tensor1, tensor2, strict=False):
            check_tensors_unchanged(t, o, prefix)


def wrap_with_timer(fn, n=10):
    """Takes a function and returns a function that calls it n times and returns the total time."""
    def timed_fn(*args, **kwargs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(n):
            out = fn(*args, **kwargs)
        end_event.record()
        end_event.synchronize()
        return out, start_event.elapsed_time(end_event)
    return timed_fn

def estimate_runtime(fn, *args, num1=10, num2=30, **kwargs):
    """Takes a function and returns a an estimate of time per iteration."""
    timed_fn_1 = wrap_with_timer(fn, num1)
    timed_fn_2 = wrap_with_timer(fn, num2)

    _, t1 = timed_fn_1(*args, **kwargs)
    _, t2 = timed_fn_2(*args, **kwargs)

    return (t2 - t1) / (num2 - num1)

def get_timing_functions(fn, grads, *args, num1=10, num2=30, warmup=3, **kwargs):
    """Returns three functions that estimate timings for forward, backward and forward+backward passes.

    Args:
        fn: Function to time
        grads: Gradients to pass to fn
        *args: Arguments to pass to fn
        num1: First number of iterations for timing estimate
        num2: Second number of iterations for timing estimate
        warmup: Number of warmup iterations
        **kwargs: Keyword arguments to pass to fn

    Returns:
        Tuple of (fwd_timing_fn, bwd_timing_fn, fwd_bwd_timing_fn) that each return estimated ms per iteration
    """
    # Get compiled versions
    fwd, bwd, fwd_bwd = get_compiled_versions(fn, grads, *args, warmup=warmup, **kwargs)

    # Create timing functions that return estimates
    def get_fwd_time():
        return estimate_runtime(fwd, num1=num1, num2=num2)

    def get_bwd_time():
        return estimate_runtime(bwd, num1=num1, num2=num2)

    def get_fwd_bwd_time():
        return estimate_runtime(fwd_bwd, num1=num1, num2=num2)

    return get_fwd_time, get_bwd_time, get_fwd_bwd_time
