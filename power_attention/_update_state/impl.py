## UPDATE STATE CUSTOM OP
# Computes the update state forward pass and backward pass using CUDA.

## IMPLEMENTATION ##
import torch
from torch.utils._pytree import tree_map
from power_attention._update_state.fwd import update_state_fwd
from power_attention._update_state.bwd import update_state_bwd
from power_attention._utils import compute_expanded_dim

# Define the primary update_state entrypoint
@torch.library.custom_op("power_attention::update_state", mutates_args=())
def update_state(K : torch.Tensor, V : torch.Tensor, deg : int) -> torch.Tensor:
    r"""Compute expanded state vectors for symmetric power attention.
    
    This function implements the state expansion mechanism from [1]. It uses symmetric tensors
    to efficiently capture higher-order interactions between keys and values, achieving massive
    dimensionality reduction compared to full tensor products (e.g., 96% reduction for deg=4).

    For each chunk i, the state tensor accumulates symmetric outer products:

    $$
    S_{ij} = \sum_{t \in \text{chunk}_i} \phi(K_t)_i \phi(V_t)_j \quad \text{for } i+j < deg
    $$

    where ϕ maps vectors to their symmetric power embedding, i,j are power indices, and t
    iterates over positions in the chunk. By using symmetric tensors instead of full tensor
    products, we achieve both better performance and much smaller state sizes.

    The expanded dimension is computed as:

    $$
    \text{expanded_dim} = \binom{\text{head_dim} + deg - 1}{deg}
    $$

    Args:
        K: Key tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        V: Value tensor of shape `(batch_size, num_chunks, chunk_size, num_heads, head_dim)`.
        deg: Power attention degree. Controls the size of expanded state vectors.
            Must be even for symmetric power formulation.

    Returns:
        S: Expanded state tensor of shape `(batch_size, num_chunks, num_heads, expanded_dim, head_dim)`,
           where expanded_dim is computed using the binomial formula above.

    Note:
        - K and V must have matching shapes and dtypes (fp16 or bf16)
        - K and V must be contiguous along the last dimension
        - chunk_size must be at least 128
        - The symmetric formulation provides massive memory savings:
          * deg=2: 49% reduction
          * deg=4: 96% reduction
          * deg=6: 99.8% reduction

    References:
        [1] J. Buckman, C. Gelada, and S. Zhang, "Symmetric Power Transformers." 
            Manifest AI, Aug. 15, 2024.
    """
    S = update_state_fwd(K, V, deg)
    return S
# Make it traceable
@torch.library.register_fake("power_attention::update_state")
def update_state_fake(K, V, deg):
    b, n, c, h, d = K.shape
    D = compute_expanded_dim(d, deg)
    return torch.empty(b, n, h, D, d, device=K.device, dtype=K.dtype)
# Autograd setup
def update_state_setup(ctx, inputs, output):
    K, V, deg = inputs
    ctx.save_for_backward(K, V)
    ctx.deg = deg
def update_state_backward(ctx, dS):
    K, V = ctx.saved_tensors
    dS = dS.contiguous()
    dK, dV = update_state_bwd(K, V, dS, ctx.deg)
    return dK, dV, None
# Register autograd
torch.library.register_autograd(
    "power_attention::update_state", update_state_backward, setup_context=update_state_setup
)

# Useful function to create sample inputs
def create_inputs(b=2, n=4, c=128, h=8, d=32, dtype=torch.float16, device='cuda', seed=42, requires_grad=False):
    torch.manual_seed(seed)
    K = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    V = torch.randn(size=(b, n, c, h, d), dtype=dtype, device=device) / d**.25
    if requires_grad:
        K, V = tree_map(lambda x: x.requires_grad_(True), (K, V))
    return dict(K=K, V=V, deg=2)

## TUTORIAL ##
if __name__ == '__main__':
    from perf._timing import report_fwd_bwd

    # Hyperparameters
    b, n, c, h, d = (8, 8, 128, 16, 64)
    dtype = torch.float16
    # Create inputs
    inputs = create_inputs(b, n, c, h, d, dtype, 'cuda', requires_grad=True)
    
    print(f"Benchmarking chunk state \n {b=} {n=} {c=} {h=} {d=} {dtype=}")

    # benchmark
    report_fwd_bwd(update_state, inputs['K'], inputs['V'], inputs['deg'])
