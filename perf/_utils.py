import torch

def same_device(device1: torch.device, device2: torch.device) -> bool:
    """Return True if two devices are effectively the same."""
    if device1.type != device2.type:
        return False
    return (device1.index == device2.index) or (device1.index is None or device2.index is None)


def clone_grads(tensor_or_tensors):
    """Clone gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor, dict, or iterable of items that may be tensors
        
    Returns:
        List of cloned gradients for any tensors that required grad
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad and tensor_or_tensors.grad is not None:
            return tensor_or_tensors.grad.detach().clone()
        return None
        
    # Handle dict case
    if isinstance(tensor_or_tensors, dict):
        grads = {}
        for key, item in tensor_or_tensors.items():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                if item.grad is not None:
                    grads[key] = item.grad.detach().clone()
                else:
                    grads[key] = None
        return grads
        
    # Handle iterable case
    grads = []
    for item in tensor_or_tensors:
        if isinstance(item, torch.Tensor) and item.requires_grad:
            if item.grad is not None:
                grads.append(item.grad.detach().clone())
            else:
                grads.append(None)
    return grads


def clear_grads(tensor_or_tensors):
    """Clear gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor or iterable of items that may be tensors
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad:
            tensor_or_tensors.grad = None
        return
    # Handle iterable/dict case
    if isinstance(tensor_or_tensors, dict):
        for item in tensor_or_tensors.values():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None
    else:
        for item in tensor_or_tensors:
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None

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

def tensors_to_ones_like(items):
    """Helper function to convert a tensor or iterable/dict of tensors to ones of the same shape."""
    if isinstance(items, torch.Tensor):
        return torch.ones_like(items)
    elif isinstance(items, dict):
        return {k: tensors_to_ones_like(v) for k, v in items.items() if isinstance(v, torch.Tensor)}
    else:
        return tuple(tensors_to_ones_like(item) for item in items if isinstance(item, torch.Tensor))

def try_convert_and_compare(thing, string):
    """Helper function that attempts to convert a string to the type of 'thing' and checks equality.
    
    Args:
        thing: The object to compare against
        string: String to try converting to thing's type
        
    Returns:
        bool: True if conversion succeeded and values are equal, False otherwise
    """
    if thing is None:
        return string == "None"
    try:
        converted = type(thing)(string)
        return converted == thing
    except (ValueError, TypeError):
        return False

def check_filter_matches(filter_strings, attrs):
    """Helper function that checks if a dict matches a list of key=value filter strings.
    
    Args:
        filter_strings: List of strings in format "key=value"
        attrs: Dict to check against
        
    Returns:
        bool: True if all filter key/value pairs match corresponding entries in attrs
        
    Raises:
        ValueError: If any filter string is not in the format "key=value"
    """
    for f in filter_strings:
        try:
            key, value = f.split('=')
        except ValueError:
            raise ValueError(f"Filter string '{f}' is not in the format 'key=value'")
            
        if key not in attrs:
            return False
        if not try_convert_and_compare(attrs[key], value):
            return False
            
    return True

def filter_measurements(measurements, filters):
    """Filter measurements based on key=value filters."""
    # Convert filters to dict
    return [m for m in measurements if check_filter_matches(filters, m.attrs)]

def describe_gpu():
    """Returns a string describing the GPU type and count, e.g. '2xA100'
    
    Returns:
        str: Description of GPU type and count, or 'CPU' if no GPUs available
    """
    if not torch.cuda.is_available():
        return "None"
        
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0).replace('NVIDIA ', '').replace('RTX ', '')
            
    return f"{gpu_count}x{gpu_name}"

