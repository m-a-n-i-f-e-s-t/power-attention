import torch
from math import floor, ceil, log
from functools import partial

InnerBlock_DT = 16
OuterBlock_DT = 8
InnerBlock_TD = 16
OuterBlock_TD = 1

DEFAULT_SEEDS = [40 + i for i in range(2)]
class DummyCtx:
    def save_for_backward(self, *args):
        self.saved_tensors = args

def dummify(fn):
    return partial(fn, DummyCtx())

def compute_expanded_dim(head_size, deg):
    return ((InnerBlock_DT // OuterBlock_DT + head_size // OuterBlock_DT) * (head_size // InnerBlock_DT) // 2) * (InnerBlock_DT * OuterBlock_DT)

def layernorm(x, eps=None):
    """Custom layernorm that supports eps as a tensor.

    Args:
        x: Input tensor
        eps: Epsilon value for layernorm. If a tensor, it must be broadcastable to the last dimension of x.

    Returns:
        Tensor: Layernormed tensor
    """
    o = x.float()
    if isinstance(eps, torch.Tensor):
        eps = eps.unsqueeze(-1)
    elif eps is None:
        eps = 0.0
    return ((o - o.mean(-1, keepdim=True)) / (o.std(-1, keepdim=True, correction=False) + eps)).to(x.dtype)

def unscale_ballnorm(x, log_scale, radius=1.0):
    """Ballnorm along the last dimension.
    Anything outside the radius of the 1-ball gets projected onto its surface. Inside is left unchanged.
    The x is assumed to be pre-scaled for stability by the exp(-log_scale) (meaning the "true" value is x * exp(log_scale)),
    so we take this into account when deciding whether or not to project.
    """

    assert len(log_scale.shape) == len(x.shape), f"log_scale.shape: {log_scale.shape}, x.shape: {x.shape}"
    assert log_scale.shape[:-1] == x.shape[:-1], f"log_scale.shape: {log_scale.shape}, x.shape: {x.shape}"
    assert log_scale.shape[-1] == 1, f"log_scale.shape: {log_scale.shape}"

    o = x.float()
    mean = o.mean(-1, keepdim=True)
    std = torch.clamp(o.std(-1, keepdim=True, correction=False), min=1e-9)
    y = torch.where(torch.log(std) + log_scale > log(radius),
                    (o - mean) / std,              # outside the ball, project onto its surface
                    o * torch.exp(log_scale))      # inside the ball, leave unchanged but undo effect of log_scale
    return y.to(x.dtype)

# Credit: https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451
def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def print_tensor(tensor, indent=0, multi_idx=None):
    """Prints a tensor in a readable format.
    
    For 1D/2D tensors, prints in CSV-like format.
    For higher dimensions, prints recursively with headers for each slice.
    For scalar tensors, prints the value directly.
    
    Args:
        tensor: torch.Tensor to print
        indent: Number of spaces to indent (used in recursive calls)
        multi_idx: List of indices for higher dimensional tensors (used in recursive calls)
    """
    import pandas as pd

    if multi_idx is None:
        multi_idx = []
    
    # Handle scalar tensors (0-dimensional)
    if tensor.dim() == 0:
        print(' ' * indent + str(tensor.item()))
        return
        
    if tensor.dim() <= 2:
        # Convert to numpy for prettier printing
        array = tensor.detach().to(torch.float32).cpu().numpy()
        # Create pandas DataFrame for nice formatting
        if tensor.dim() == 1:
            array = array.reshape(1, -1)
        df = pd.DataFrame(array)
        # Print with proper indentation
        for line in df.to_string(header=False, index=False).split('\n'):
            print(' ' * indent + line)
            
    else:
        # Handle higher dimensional tensors recursively
        for i in range(tensor.shape[0]):
            current_idx = multi_idx + [i]
            idx_str = f"[{','.join(map(str, current_idx))}]"
            print(' ' * indent + f"Index {idx_str}:")
            print_tensor(tensor[i], indent + 2, current_idx)
            if i < tensor.shape[0] - 1:
                print()
    

def diff(a, b, rtol=None, atol=None, assert_close=True, verbose=True, title=None):
    """ A diff function that helps debug numerical issues

    Args:
        a: torch.Tensor
        b: torch.Tensor
        rtol: float
        atol: float
        assert_close: bool
        verbose: bool
    Returns:
        bool: True if a and b are close, False otherwise
    """
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if rtol is None: rtol = 1e-3
    if atol is None: atol = 1e-5
    equal = torch.allclose(a, b, rtol=rtol, atol=atol)
    error_max = torch.max(torch.abs(a - b))
    error_hist = torch.histc(torch.abs(a - b), bins=100, min=0, max=1)
    
    # Calculate absolute error
    abs_diff = torch.abs(a - b)
    total_elements = a.numel()
    
    # Calculate relative error where b is non-zero
    b_nonzero = b != 0
    rel_diff = torch.zeros_like(abs_diff)
    rel_diff[b_nonzero] = abs_diff[b_nonzero] / torch.abs(b[b_nonzero])
    
    if verbose:
        print('\n' * 3)
        print('=' * 10 + f" {title} " + '=' * 10)
        print(f"Max absolute error: {error_max.item()}")
        print(f"Tensors are {'close' if equal else 'different'} according to torch.allclose")
        
        # Calculate thresholds for relative error table
        rel_thresholds = torch.logspace(
            torch.log10(torch.tensor(rtol)), 
            0.0, 
            steps=10
        )
        
        # Calculate thresholds for absolute error table
        abs_thresholds = torch.logspace(
            torch.log10(torch.tensor(atol)), 
            0.0, 
            steps=10
        )
        
        # Print relative error table
        print("\nRelative Error Table:")
        print("---------------------")
        print(f"{'Threshold':<12} {'% matched':<12} {'Element Count':<12}")
        print("-" * 36)
        for threshold in rel_thresholds:
            count = (rel_diff <= threshold).sum().item()
            percentage = 100.0 * count / total_elements
            print(f"{threshold.item():<12.6f} {percentage:<12.2f} {count:<12}")
        
        # Print absolute error table
        print("\nAbsolute Error Table:")
        print("---------------------")
        print(f"{'Threshold':<12} {'% matched':<12} {'Element Count':<12}")
        print("-" * 36)
        for threshold in abs_thresholds:
            count = (abs_diff <= threshold).sum().item()
            percentage = 100.0 * count / total_elements
            print(f"{threshold.item():<12.6f} {percentage:<12.2f} {count:<12}")
        
        # Print some examples of largest errors
        if not equal:
            n_samples = min(5, total_elements)
            print("\nLargest Errors:")
            flat_indices = torch.argsort(abs_diff.flatten(), descending=True)[:n_samples]
            for i in range(n_samples):
                idx = flat_indices[i]
                multi_idx = torch.unravel_index(idx, a.shape)
                multi_idx_str = ', '.join(map(str, [idx.item() for idx in multi_idx]))
                print(f"Index [{multi_idx_str}]: a={a[multi_idx].item()}, b={b[multi_idx].item()}, "
                      f"abs_diff={abs_diff[multi_idx].item()}, rel_diff={rel_diff[multi_idx].item()}")
    
    if assert_close:
        assert equal, f"Tensors are not close! Max absolute error: {error_max.item()}"
    
    return equal