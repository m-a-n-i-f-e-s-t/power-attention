## DISCOUNTED CUMSUM CUSTOM OP
# Computes the discounted cumsum forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torch.utils._pytree import tree_map
from types import NoneType
from typing import Optional, Tuple
from power_attention._discumsum.fwd import discumsum_fwd
from power_attention._discumsum.bwd import discumsum_bwd

@torch.library.custom_op("power_attention::discumsum", mutates_args=())
def discumsum(X : torch.Tensor, log_G : torch.Tensor) -> torch.Tensor:
    r"""Compute discounted cumulative sum for recurrent processing.

    This function implements the discounted cumulative sum operation from [1]. It is a key
    component in transforming symmetric power attention into a linear-cost RNN, allowing
    efficient processing of long sequences through chunked computation.

    The operation implements the state update equation of the RNN:

    $$Y_t = X_t + \exp(\log G_t) \cdot Y_{t-1}$$

    with initial condition $Y_0 = 0$

    In the context of symmetric power attention:

    - $X_t$ contains expanded state vectors from the current chunk
    - $Y_t$ accumulates state information across chunks
    - $exp(log\ G_t)$ controls how much past information influences the current computation
    - The +1 in the output time dimension allows for proper causality in the RNN

    This formulation enables O(n) complexity instead of O(n²) for long sequences, while
    maintaining the expressivity of power attention through the expanded state representation.

    Args:
        X: Input tensor of shape `(batch_size, time, num_heads, *feature_dims)`.
           The tensor to be accumulated along the time dimension.
        log_G: Log discount factors of shape `(batch_size, time, num_heads)`.
           Natural logarithm of the discount/gating factors.
           These are broadcasted along the feature dimensions.

    Returns:
        Y: Accumulated tensor of shape `(batch_size, time+1, num_heads, *feature_dims)`.
           Note that the output has one more timestep than the input, with zeros at t=0.

    Note:
        - Time dimension must be a multiple of 4
        - Product of feature dimensions must be a multiple of 8
        - The batch and heads dimensions are treated as independent batch dimensions
        - Initial state support is planned but not yet implemented
        - The RNN formulation maintains O(1) memory per layer regardless of sequence length

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    b, n, h, *ds = X.shape
    if len(X.shape) > 4:
        X = X.view(*X.shape[:3], -1)
    cum_X = discumsum_fwd(X, log_G)
    return cum_X.view(b, n+1, h, *ds)

# Make it traceable
@discumsum.register_fake
def discumsum_fake(X, log_G):
    b, n, h, *ds = X.shape
    return torch.empty(b, n+1, h, *ds, device=X.device, dtype=X.dtype)

# Autograd setup
def discumsum_setup(ctx, inputs, output):
    X, log_G = inputs
    ctx.save_for_backward(log_G, output)
    ctx.original_shape = X.shape

def discumsum_backward(ctx, dcum_X):
    log_G, cum_X = ctx.saved_tensors
    b, n, h, *ds = ctx.original_shape
    dcum_X = dcum_X.view(b, n+1, h, -1)
    cum_X = cum_X.view(b, n+1, h, -1)
    dX, dlog_G = discumsum_bwd(dcum_X, cum_X, log_G)
    return dX.view(b, n, h, *ds), dlog_G

# Register autograd
torch.library.register_autograd(
    "power_attention::discumsum", discumsum_backward, setup_context=discumsum_setup
)

# Useful function to create sample inputs   
def create_inputs(b=2, n=4, h=8, D=64, d=16, X_dtype=torch.float16, device='cuda', requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(42)
    X = torch.randn(size=(b, n, h, D, d), dtype=X_dtype, device=device, generator=generator)
    log_G = torch.zeros(size=(b, n, h), dtype=torch.float32, device=device) - 0.01
    if requires_grad:
        X, log_G = tree_map(
            lambda x: x.requires_grad_(True), (X, log_G)
        )
    return dict(
        X=X,
        log_G=log_G
    )


## TUTORIAL ##
if __name__ == '__main__':
    # Hyperparameters
    b, n, h, D, d = (2, 4, 8, 64, 16)
    dtype = torch.float16
    # Create inputs
    X, log_G = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
    # Run forward
    cum_X = discumsum(X, log_G)
    # Run backward
    torch.autograd.backward(cum_X, cum_X)
    # Compile function, fullgraph=True confirms no graph breaks
    compiled_discumsum = torch.compile(discumsum, fullgraph=True)
    for _ in range(3):
        inputs = create_inputs(b, n, h, D, d, dtype, 'cuda', requires_grad=True)
        outputs = compiled_discumsum(*inputs)
        torch.autograd.backward(outputs, outputs)

