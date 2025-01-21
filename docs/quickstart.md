# Quickstart

This guide covers the core concepts and usage of symmetric power attention, implementing the architecture described in [Buckman et al. (2024)](https://manifestai.com/articles/symmetric-power-transformers/).

## Basic Usage

```python linenums="1"
import math
import torch
import torch.nn.functional as F
from power_attention.power_full import power_full

# Parameters

b = 2 # batch size
t = 1024 # sequence length
h = 8 # number of key/value attention heads
d = 64 # model dim
qhead_ratio = 1 # query heads to key/value heads for multi-query attention
dtype = torch.bfloat16 # data type for attention tensors
device = 'cuda' # device to create tensors on
gating = True

# Create test inputs

Q = torch.randn(size=(b, t, h * qhead_ratio, d), dtype=dtype, device=device) / math.sqrt(d)
K = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)
V = torch.randn(size=(b, t, h, d), dtype=dtype, device=device) / math.sqrt(d)

if gating:
    log_G = F.logsigmoid(torch.randn(size=(b, t, h), dtype=torch.float32, device=device))
else:
    log_G = None

initial_state = None
scale = 1./d**0.5
normal_space = not log_space

# Run power attention
output = power_full(
    Q=Q, K=K, V=V, log_G=log_G, 
    initial_state=initial_state,
    deg=2,
    scale=scale,
    chunk_size=128
)

print("Ran power attention, output shape:", output.shape)
```

<div class="admonition warning">
<p class="admonition-title">Important</p>
<p>Currently the kernel only supports degree 1, 2 or 4.</p>
</div>

## Training Example

For a complete training setup, see the [training script](https://github.com/manifestai/power-attention/blob/main/train/train.py). Here's a minimal example showing how to use symmetric power attention in a transformer block:

```python linenums="1"
import torch
import torch.nn as nn
from power_attention import power_full
from contextlib import nullcontext

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.qhead_ratio = config.qhead_ratio
        self.degree = config.degree
        self.chunk_size = config.chunk_size
        self.log_space = config.log_space
        self.gating = not config.disable_gating
        
        # key, query, value projections for all heads, but in a batch
        self.qkv_size = (config.qhead_ratio + 2) * self.n_head * self.head_size
        self.gating_size = config.n_head if self.gating else 0
        self.c_attn = nn.Linear(config.n_embd, self.qkv_size + self.gating_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.qhead_ratio * self.n_head * self.head_size, config.n_embd, bias=config.bias)
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        h = self.n_head
        hq = self.qhead_ratio * h
        d = self.head_size

        # calculate query, key, values and gating for all heads in batch
        qkvg = self.c_attn(x)
        qkv = qkvg[...,:self.qkv_size]
        q, k, v = qkv.split([hq*d, h*d, h*d], dim=2)
        q = q.view(B, T, hq, d)
        k = k.view(B, T, h, d)
        v = v.view(B, T, h, d)

        # compute gating tensor if enabled
        if self.gating:
            log_g = torch.nn.functional.logsigmoid(qkvg[...,self.qkv_size:].to(dtype=torch.float32))
        else:
            log_g = None

        # apply power attention
        y = power_full(q, k, v, log_g,
                      deg=self.degree,
                      scale=1.0 / d**0.5,
                      chunk_size=self.chunk_size)

        # output projection
        y = y.contiguous().view(B, T, hq * d)
        return self.c_proj(y)

# Example usage
class Config:
    n_head = 12
    head_size = 64
    qhead_ratio = 1
    degree = 2
    chunk_size = 128
    log_space = False
    disable_gating = False
    n_embd = 768
    bias = False

config = Config()
attention = CausalSelfAttention(config).to('cuda')

# Setup autocast
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ctx = nullcontext() if dtype == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=getattr(torch, dtype))

# Forward pass with autocast
x = torch.randn(2, 1024, config.n_embd, device='cuda')  # (batch_size, seq_len, n_embd)
with ctx:
    output = attention(x)  # Shape: (batch_size, seq_len, n_embd)
    print(output.shape)
```

<div class="admonition warning">
<p class="admonition-title">Data Type Warning</p>
<p>Power Attention only supports fp16 and bf16 data types. BF16 is strongly recommended over FP16, especially for degree > 2.</p>
</div>

<div class="admonition note">
<p class="admonition-title">Training Setup</p>
<p>
For the complete training setup including model architecture, data loading, learning rate scheduling, and evaluation functions, see the full <a href="https://github.com/manifestai/power-attention/blob/main/train/train.py">training script</a>.
</p>
</div>

## Core Concepts

### Power Parameter (p)

The power parameter p controls the dimensionality of the symmetric tensor space. The attention scores are computed as:

$$
A_{ij} = \frac{(\phi_p(Q_i)^T \phi_p(K_j))}{(\sum_{k=1}^i \phi_p(Q_i)^T \phi_p(K_k))}
$$

```python linenums="1"
# Minimal state size, not recommended
output_p2 = power_attention(q, k, v, degree=2)

# Preliminary experiments suggest good results
output_p4 = power_attention(q, k, v, degree=4)
```

### Gating Mechanism

The gating mechanism modulates the contribution of each position through a multiplicative factor:

$$
Y_i = \sum_{j=1}^i A_{ij} V_j G_j
$$

<div class="admonition tip">
<p class="admonition-title">Best Practice</p>
<p>Always use gating for optimal performance. The gating mechanism is crucial for achieving transformer-level perplexity.</p>
</div>

```python linenums="1"
import torch.nn.functional as F

# Create gating tensor (recommended)
log_g = F.logsigmoid(
    torch.randn(batch_size, num_heads, seq_len,
                dtype=torch.float32, device='cuda')
)

# With gating (recommended)
output_gated = power_attention(q, k, v, log_g=log_g)

# Without gating (not recommended)
output_ungated = power_attention(q, k, v, log_g=None)
```

### Memory Management

For long sequences, use chunking to manage memory usage. The chunk size determines the tradeoff between memory and computation:

<div class="admonition note">
<p class="admonition-title">Chunk Size Selection</p>
<p>Smaller chunks use less memory but require more computation.
</div>

### Numerical Precision

<div class="admonition warning">
<p class="admonition-title">Precision Warning</p>
<p>BF16 is strongly recommended over FP16, especially for degree > 2, due to the power operations involved in the attention computation.</p>
</div>

## Next Steps

- Study the [benchmarking guide](benchmarking.md) for performance optimization
- Read the [mathematical background](https://manifestai.com/articles/symmetric-power-transformers/#4-1-mathematical-background)
- Check the [experimental results](https://manifestai.com/articles/symmetric-power-transformers/#2-experiments)
