from collections.abc import Iterable

import torch
from perf._utils import clone_or_none, prune_non_tensors, tensors_to_ones_like


def get_compiled_version(fn, inputs, direction, warmup=3, compile=True):
    """Takes a function and args and returns compiled versions for fwd, bwd, and fwd+bwd passes.

    Args:
        fn: Function to compile
        inputs: A dict, keyword arguments to pass to fn

    Returns:
        Tuple of (fwd_fn, bwd_fn, fwd_bwd_fn)
    """
    # Run forward pass to identify output shapes and check for mutations
    # Define functions
    def fwd():
        with torch.no_grad():
            return fn(**inputs)
    torch._dynamo.config.compiled_autograd = True
    outputs = fn(**inputs)
    grads = tensors_to_ones_like(outputs)
    def bwd():
        torch.autograd.backward(outputs, grad_tensors=grads, retain_graph=True)
    torch._dynamo.config.compiled_autograd = False
    def fwd_bwd():
        outputs = fn(**inputs)
        torch.autograd.backward(outputs, grad_tensors=grads)
    # Compile functions
    if direction == 'fwd':
        compiled_fn = torch.compile(fwd, dynamic=False) if compile else fwd
    elif direction == 'bwd':
        compiled_fn = torch.compile(bwd, dynamic=False) if compile else bwd
    elif direction == 'fwd+bwd':
        compiled_fn = torch.compile(fwd_bwd, dynamic=False) if compile else fwd_bwd
    # Warmup passes
    for _ in range(warmup):
        compiled_fn()
    # Return compiled functions
    return compiled_fn



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


def wrap_with_timer(fn, n=10, warmup=3):
    """Takes a function and returns a function that calls it n times and returns the total time."""
    def timed_fn(*args, **kwargs):
        for _ in range(warmup):
            fn(*args, **kwargs)

        x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')
        def flush_cache():
            x.zero_()

        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            flush_cache()
            torch.cuda._sleep(1_000_000)
            start_events[i].record()
            out = fn(*args, **kwargs)
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        total_time = sum(times)
        return out, total_time
    return timed_fn

def estimate_runtime(fn, *args, num1=10, num2=50, **kwargs):
    """Takes a function and returns a an estimate of time per iteration."""
    timed_fn_1 = wrap_with_timer(fn, num1)
    timed_fn_2 = wrap_with_timer(fn, num2)

    _, t1 = timed_fn_1(*args, **kwargs)
    _, t2 = timed_fn_2(*args, **kwargs)

    return (t2 - t1) / (num2 - num1)

def get_timing_functions(fn, inputs, num1=10, num2=30, warmup=3, compile=True):
    """Returns three functions that estimate timings for forward, backward and forward+backward passes.

    Args:
        fn: Function to time
        inputs: A dict, keyword arguments to pass to fn
        num1: First number of iterations for timing estimate
        num2: Second number of iterations for timing estimate
        warmup: Number of warmup iterations

    Returns:
        Tuple of (fwd_timing_fn, bwd_timing_fn, fwd_bwd_timing_fn) that each return estimated ms per iteration
    """
    # Get compiled versions
    fwd, bwd, fwd_bwd = get_compiled_versions(fn, inputs, warmup=warmup, compile=compile)

    # Create timing functions that return estimates
    def get_fwd_time():
        return estimate_runtime(fwd, num1=num1, num2=num2)

    def get_bwd_time():
        return estimate_runtime(bwd, num1=num1, num2=num2)

    def get_fwd_bwd_time():
        return estimate_runtime(fwd_bwd, num1=num1, num2=num2)

    return get_fwd_time, get_bwd_time, get_fwd_bwd_time

def benchmark_speed(direction, fn, create_inputs, create_inputs_kwargs, num1=10, num2=30, warmup=3, compile=True):
    """Measure speed of a function implementation.
    
    Args:
        direction: str. One of 'fwd', 'bwd', or 'fwd+bwd' to measure forward pass,
            backward pass, or combined forward+backward pass timing
        fn: Function to benchmark
        create_inputs: Function that creates input tensors for fn
        **kw: Keyword arguments passed to create_inputs() to configure the test case
            
    Returns:
        float: Time in milliseconds per iteration
    """
    inputs = create_inputs(**create_inputs_kwargs, requires_grad=True)
    fn = get_compiled_version(fn, inputs, direction=direction, warmup=warmup, compile=compile)
    return estimate_runtime(fn, num1=num1, num2=num2)
