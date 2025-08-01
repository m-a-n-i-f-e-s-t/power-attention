from typing import Any, Callable
import torch

from perf._precision import compare
from perf._utils import clone_grads, clear_grads, same_device
from perf._inspect import inspect_diff_details
from perf._precision import get_violation_pct

from power_attention._utils import diff

# General checks

def sanity_check_tensor(tensor):
    """ Perform a sanity check on a tensor."""
    if tensor is None:
        return
    if not torch.isfinite(tensor).all():
        non_finite_indices = torch.argwhere(~torch.isfinite(tensor))
        torch.set_printoptions(edgeitems=10)
        raise AssertionError(f"Tensor contains non-finite values at indices: {non_finite_indices}")
    

def sanity_check_tensors(tensors):
    """ Perform a sanity check on a list of tensors."""
    for i, tensor_or_tensors in enumerate(tensors):
        try:
            if isinstance(tensor_or_tensors, torch.Tensor):
                sanity_check_tensor(tensor_or_tensors)
            elif isinstance(tensor_or_tensors, dict):
                sanity_check_tensors(tensor_or_tensors.values())
            elif isinstance(tensor_or_tensors, (list, tuple)):
                sanity_check_tensors(tensor_or_tensors)
            elif isinstance(tensor_or_tensors, (int, float, bool)) or tensor_or_tensors is None:
                pass # Simple types are always sane
            else:
                raise ValueError(f"Unsupported type: {type(tensor_or_tensors)}")
        except AssertionError as e:
            raise AssertionError(f"Element {i+1}/{len(tensors)} failed: {e}")

def check_tensor_properties(tensor: torch.Tensor, properties: Any = None):
    """Check that a tensor is well-behaved and matches expected properties.
    
    Properties can be a tuple of (shape, dtype, device), a dict with keys 'shape', 'dtype', and 'device',
    or an object with attributes 'shape', 'dtype', and 'device' (like another tensor).
    """
    # Handle tuple unpacking
    if properties is None:
        pass
    elif isinstance(properties, (list, tuple)):
        shape, dtype, device = properties
    # Handle dict
    elif isinstance(properties, dict):
        shape = properties['shape']
        dtype = properties['dtype'] 
        device = properties['device']
    # Handle object with attributes
    else:
        shape = properties.shape
        dtype = properties.dtype
        device = properties.device
    # Check properties
    if properties is not None:
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
        assert same_device(tensor.device, torch.device(device)), f"Expected device {device}, got {tensor.device}"        
    sanity_check_tensor(tensor)


def check_tensors_properties(tensors_or_tensor, properties_or_property):
    """Check that tensor(s) match expected properties."""
    if isinstance(tensors_or_tensor, torch.Tensor):
        check_tensor_properties(tensors_or_tensor, properties_or_property)
    elif isinstance(tensors_or_tensor, dict):
        if not isinstance(properties_or_property, dict):
            raise ValueError("Properties must be a dict when tensors are a dict")
        if tensors_or_tensor.keys() != properties_or_property.keys():
            raise ValueError("Dict keys don't match between tensors and properties")
        for key in tensors_or_tensor:
            try:
                check_tensor_properties(tensors_or_tensor[key], properties_or_property[key])
            except AssertionError as e:
                raise AssertionError(f"Key {key} failed: {e}")
    else:
        if len(tensors_or_tensor) != len(properties_or_property):
            raise ValueError("Number of tensors must match number of properties")
        for i, (tensor, props) in enumerate(zip(tensors_or_tensor, properties_or_property)):
            try:
                check_tensor_properties(tensor, props)
            except AssertionError as e:
                raise AssertionError(f"Element {i+1}/{len(tensors_or_tensor)} failed: {e}")

def check_tensor_property_pairs(*tensor_property_pairs):
    """Check that a list of tensor-property pairs match expected properties."""
    return check_tensors_properties(*zip(*tensor_property_pairs))


def check_allclose(a, b, rtol=None, atol=None):
    """Check if two objects or iterables of objects are equivalent.
    
    Args:
        a: First object or iterable of objects
        b: Second object or iterable of objects
        atol: Absolute tolerance threshold
        rtol: Relative tolerance threshold, relative to magnitude of b
    """
    sanity_check_tensors([a, b])
    abs_error = compare(a, b, relative=False)
    rel_error = compare(a, b, relative=True)
    if (atol is not None and abs_error > atol) or (rtol is not None and rel_error > rtol):
        if isinstance(a, torch.Tensor):
            diff(a, b, rtol=rtol, atol=atol, assert_close=False, verbose=True)
        elif isinstance(a, (list, tuple)):
            for i, (a_i, b_i) in enumerate(zip(a, b)):
                diff(a_i, b_i, rtol=rtol, atol=atol, assert_close=False, verbose=True, title=f"Tensor {i+1}/{len(a)}")
        elif isinstance(a, dict):
            for key in a:
                diff(a[key], b[key], rtol=rtol, atol=atol, assert_close=False, verbose=True, title=f"Tensor {key}")
        else:
            raise ValueError(f"Unsupported type: {type(a)}")
            
        raise AssertionError(f"Values don't match within tolerance rtol={rtol}, atol={atol}. Max absolute error: {abs_error:.4e}, max relative error: {rel_error:.4e}")


def check_error_within_tolerance(test_error, *, atol: float = None, rtol: float = None, ref_error: float = None):
    """Check if test error is within tolerance.

    atol is an absolute tolerance threshold, rtol is a relative tolerance threshold as compared to a reference error.
    Rtol does not apply if ref_error is not provided.
    
    Args:
        test_error: Test error level (float) or iterable of test errors to compare
        atol: Absolute tolerance threshold
        rtol: Relative tolerance threshold as compared to reference error, eg .2 means errors up to 20% larger than reference are allowed
        ref_error: Reference error level (float) or iterable of reference errors
        
    Raises:
        AssertionError: If test error exceeds reference by more than tolerance,
            with details about which errors failed and by how much
        TypeError: If inputs are not both floats or both iterables
    """
    def _get_error_message(test_error, ref_error, max_allowed, rtol, atol, index=None):
        """Helper to format error message for tolerance check failures."""
        prefix = f"Element {index}/{len(ref_error)} " if index is not None else ""
        if ref_error is not None:
            rtol_str = f"{rtol*100:.0f}%" if rtol is not None else "N/A"
            atol_str = f"{atol:.0e}" if atol is not None else "N/A" 
            return f"{prefix}error ({test_error:.4f}) exceeds reference ({ref_error:.4f}) by more than {max_allowed:.4f} = max({rtol_str}, {atol_str})"
        else:
            atol_str = f"{atol:.0e}" if atol is not None else "N/A"
            return f"{prefix}error ({test_error:.4f}) exceeds {atol_str}"

    # Handle single error case
    if isinstance(test_error, float):
        max_allowed = max(
            (ref_error * rtol) if ref_error is not None and rtol is not None else 0.,
            atol if atol is not None else 0.
        )
        if test_error > max_allowed:
            raise AssertionError("Precision failure: " + _get_error_message(test_error, ref_error, max_allowed, rtol, atol))
        return
        
    # Handle iterable case
    try:
        failures = []
        for i, (ref, test) in enumerate(zip(ref_error if ref_error is not None else [None] * len(test_error), test_error)):
            max_allowed = max((ref * rtol) if ref is not None and rtol is not None else 0., 
                            atol if atol is not None else 0.)
            if test > max_allowed:
                failures.append(_get_error_message(test, ref, max_allowed, rtol, atol, i))
        if failures:
            raise AssertionError("Precision failure(s): " + ", ".join(failures))
            
    except TypeError:
        raise TypeError("Inputs must both be floats or both be iterables")

# Specific checks

def check_inputs_created_determinstically(create_inputs, kwargs, rtol=None, atol=None):
    """Check that the create_inputs function produces the same inputs for the same kwargs every time."""
    first_inputs = create_inputs(**kwargs)
    second_inputs = create_inputs(**kwargs)
    check_allclose(first_inputs, second_inputs, rtol=rtol, atol=atol)


def check_fn_compiles(fn, inputs, rtol=None, atol=None):
    """Check that the fn compiles and its output is unchanged."""
    with torch.no_grad():
        output_or_outputs = fn(*inputs.values())    
    compiled_fn = torch.compile(fn)
    with torch.no_grad():
        for _ in range(3):
            output_or_outputs_from_compiled = compiled_fn(*inputs.values()) # TODO(jbuckman): horrible hack to get torch.compile to work
    check_allclose(output_or_outputs, output_or_outputs_from_compiled, rtol=rtol, atol=atol)
    torch._dynamo.reset()


def check_fn_compiles_with_backward(fn, inputs, rtol=None, atol=None):
    """Check that the fn compiles and its gradients are unchanged, including the backward pass."""
    def fwd_bwd():
        output_or_outputs = fn(**inputs)
        if not isinstance(output_or_outputs, torch.Tensor):
            output_or_outputs = [o for o in output_or_outputs if o is not None and o.requires_grad]
        output_grads = torch.ones_like(output_or_outputs) if isinstance(output_or_outputs, torch.Tensor) else [torch.ones_like(o) for o in output_or_outputs]
        torch.autograd.backward(output_or_outputs, grad_tensors=output_grads, retain_graph=True)
        return clone_grads(inputs)
    grads = fwd_bwd()
    compiled_fn = torch.compile(fwd_bwd)
    for _ in range(2):
        clear_grads(inputs)
        grads_from_compiled = compiled_fn()
    check_tensors_properties(grads, grads_from_compiled)
    if rtol is not None or atol is not None:
        check_allclose(grads, grads_from_compiled, rtol=rtol, atol=atol)
    torch._dynamo.reset()


def check_fake_fn_implementation_matches(fn, fake_fn, inputs):
    """Check that the fake_fn implementation matches the real fn implementation, meaning they 
    produce outputs that have the same shape, dtype, and device."""
    real_output = fn(**inputs)
    fake_output = fake_fn(**inputs)
    check_tensors_properties(real_output, fake_output)


def check_inputs_forwards_match(*, fn, inputs1, inputs2, atol=None, rtol=None):
    """Given a function, check that it produces the same output for two sets of inputs.
    
    Args:
        fn: Function to check
        inputs1: First set of inputs for function
        inputs2: Second set of inputs for function
        atol: float. Absolute tolerance for difference
        rtol: float. Relative tolerance for difference
    """
    with torch.no_grad():
        output1 = fn(**inputs1)
        output2 = fn(**inputs2)
    check_allclose(output1, output2, atol=atol, rtol=rtol)

def check_inputs_backwards_match(*, fn, inputs1, inputs2, atol=None, rtol=None):
    """Given a function, check that it produces the same gradients for two sets of inputs.
    
    Args:
        fn: Function to check
        inputs1: First set of inputs for function
        inputs2: Second set of inputs for function
        atol: float. Absolute tolerance for difference
        rtol: float. Relative tolerance for difference
    """
    output1 = fn(**inputs1)
    torch.autograd.backward(output1, grad_tensors=torch.ones_like(output1))
    grads1 = clone_grads(inputs1)

    output2 = fn(**inputs2)
    torch.autograd.backward(output2, grad_tensors=torch.ones_like(output2))
    grads2 = clone_grads(inputs2)

    sanity_check_tensors([grads1, grads2])
    check_allclose(grads1, grads2, atol=atol, rtol=rtol)

def check_fn_forwards_match(*, ref_fn, gold_inputs, test_fn, test_inputs, rtol=None, atol=0., diff_tol=None):
    """Given two functions, check that they produce the same output for the same inputs.

    gold_inputs are high-precision inputs, the reference function is run with these inputs
    to produce a gold output. Then both the reference and test functions are run with
    the low-precision test_inputs to produce a reference and test output.
    
    Args:
        ref_fn: Reference function
        gold_inputs: High-precision inputs for reference function
        test_fn: Test function
        test_inputs: Low-precision inputs for reference & test function
        tol: float. Tolerance for difference
        atol: float. Absolute tolerance for difference
        diff_tol: float. Tolerance for percentage of elements that differ
    """
    def filter_none(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, (tuple, list)):
            return [o for o in outputs if o is not None]
        else:
            raise ValueError(f"Unsupported type: {type(outputs)}")
    ref_inputs = test_inputs
    with torch.no_grad():
        gold_output = filter_none(ref_fn(**gold_inputs))
        ref_output = filter_none(ref_fn(**ref_inputs))
        test_output = filter_none(test_fn(**test_inputs))

    sanity_check_tensors([gold_output, ref_output, test_output])
    ref_err = compare(gold_output, ref_output)
    test_err = compare(gold_output, test_output)
    try:
        check_error_within_tolerance(test_err, atol=atol, rtol=rtol, ref_error=ref_err)
    except AssertionError as e:
        violation_pct = get_violation_pct(gold_output, ref_output, test_output, tol=rtol, atol=atol)
        if diff_tol is not None and violation_pct < diff_tol:
            return
            
        msg = inspect_diff_details(gold_output, ref_output, test_output, tol=rtol, atol=atol)
        if isinstance(gold_output, torch.Tensor):
            gold_output = [gold_output]
            ref_output = [ref_output]
            test_output = [test_output]
        for i, (gold, ref, test) in enumerate(zip(gold_output, ref_output, test_output)):
            diff(gold, ref, rtol=rtol, atol=atol, assert_close=False, verbose=True, title=f"gold_output[{i}] vs ref_output[{i}]")
            diff(gold, test, rtol=rtol, atol=atol, assert_close=False, verbose=True, title=f"gold_output[{i}] vs test_output[{i}]")
        raise AssertionError(f"Precision failure: {e}\n{msg}\nViolation percentage: {violation_pct * 100:.2f}%")

def check_fn_backwards_match(*, ref_fn, gold_inputs, test_fn, test_inputs, rtol=None, atol=0., diff_tol=None):
    """Given two functions, check that they produce the same gradients for the same inputs.

    gold_inputs are high-precision inputs, the reference function is run with these inputs
    to produce a gold output. Then both the reference and test functions are run with
    the low-precision test_inputs to produce a reference and test output.
    
    Args:
        ref_fn: Reference function
        gold_inputs: High-precision inputs for reference function
        test_fn: Test function
        test_inputs: Low-precision inputs for reference & test function
        tol: float. Tolerance for difference
        atol: float. Absolute tolerance for difference
        diff_tol: float. Tolerance for percentage of elements that differ
    """

    def _create_grad_tensors(example):
        if isinstance(example, torch.Tensor):
            return torch.ones_like(example)
        elif isinstance(example, (tuple, list)):
            return [torch.ones_like(e) for e in example]
        else:
            raise ValueError(f"Unsupported type: {type(example)}")
        
    def get_outputs_with_grad(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, (tuple, list)):
            return [o for o in outputs if o.requires_grad]
        else:
            raise ValueError(f"Unsupported type: {type(outputs)}")

    gold_output = get_outputs_with_grad(ref_fn(**gold_inputs))
    torch.autograd.backward(gold_output, grad_tensors=_create_grad_tensors(gold_output))
    gold_grads = clone_grads(gold_inputs)

    ref_output = get_outputs_with_grad(ref_fn(**test_inputs))
    torch.autograd.backward(ref_output, grad_tensors=_create_grad_tensors(ref_output))
    ref_grads = clone_grads(test_inputs)
    
    clear_grads(test_inputs)
    test_output = get_outputs_with_grad(test_fn(**test_inputs))
    torch.autograd.backward(test_output, grad_tensors=_create_grad_tensors(test_output))
    test_grads = clone_grads(test_inputs)

    sanity_check_tensors([gold_grads, ref_grads, test_grads])
    ref_err = compare(gold_grads, ref_grads)
    test_err = compare(gold_grads, test_grads)
    try:
        check_error_within_tolerance(test_err, atol=atol, rtol=rtol, ref_error=ref_err)
    except AssertionError as e:
        violation_pct = get_violation_pct(gold_grads, ref_grads, test_grads, tol=rtol, atol=atol)
        if diff_tol is not None and violation_pct < diff_tol:
            return
        msg = inspect_diff_details(gold_grads, ref_grads, test_grads, tol=rtol, atol=atol)
        raise AssertionError(f"Precision failure: {e}\n{msg}\nViolation percentage: {violation_pct * 100:.2f}%")
    torch.cuda.empty_cache()
