from collections.abc import Iterable
from functools import wraps
import inspect
import torch
from perf._utils import clone_or_none, prune_non_tensors, tensors_to_ones_like


def sanitize_kwargs(fn):
    """
    Sanitizes kwargs by removing any that are not in the function signature.
    """
    @wraps(fn)
    def wrapper(**kwargs):
        sig = inspect.signature(fn)
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return fn(**valid_kwargs)
    return wrapper


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
            return sanitize_kwargs(fn)(**inputs)
    torch._dynamo.config.compiled_autograd = True
    outputs = sanitize_kwargs(fn)(**inputs)
    grads = tensors_to_ones_like(outputs)
    def bwd():
        nonlocal grads, outputs
        tuples = [(o, g) for o, g in zip(outputs, grads) if o.requires_grad]
        outputs, grads = [t[0] for t in tuples], [t[1] for t in tuples]
        torch.autograd.backward(outputs, grad_tensors=grads, retain_graph=True)
    torch._dynamo.config.compiled_autograd = False
    def fwd_bwd():
        nonlocal grads
        outputs = sanitize_kwargs(fn)(**inputs)
        tuples = [(o, g) for o, g in zip(outputs, grads) if o.requires_grad]
        outputs, grads = [t[0] for t in tuples], [t[1] for t in tuples]
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
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn(*args, **kwargs)

        cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
        def flush_cache():
            cache.zero_()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            flush_cache()
            torch.cuda._sleep(5_000_000_0)
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
