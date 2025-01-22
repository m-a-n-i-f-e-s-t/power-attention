import torch
from math import floor, ceil
from functools import partial
from power_attention_cuda import InnerBlock_DT, OuterBlock_DT

DEFAULT_SEEDS = [40 + i for i in range(2)]
class DummyCtx:
    def save_for_backward(self, *args):
        self.saved_tensors = args

def dummify(fn):
    return partial(fn, DummyCtx())

def compute_expanded_dim(head_size, deg):
    return ((InnerBlock_DT // OuterBlock_DT + head_size // OuterBlock_DT) * (head_size // InnerBlock_DT) // 2) * (InnerBlock_DT * OuterBlock_DT)

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
    