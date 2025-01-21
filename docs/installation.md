# Installation

Power Attention can be installed either via pre-built wheels or built from source.

## Installing Pre-built Wheels

### Prerequisites
- Python 3.11 or 3.12 (3.13 compatibility coming soon)
- PyTorch >=2.5
- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)
- CUDA driver supporting your PyTorch installation

```bash
pip install power-attention
```

## Building from Source

### Extra Prerequisites
- CUDA Toolkit 12.0+
- C++ compiler compatible with CUDA

1. Clone the repository:
```bash
git clone https://github.com/manifestai/power-attention.git
cd power-attention
```

2. Install with development dependencies:
```bash
pip install -e .[dev]
```

## Build Configuration

When building from source, the build process can be customized using environment variables:

### Compilation Settings
- `NVCC_THREADS`: Number of threads for NVCC compilation (default: 4)
- `MAX_JOBS`: Maximum number of parallel compilation jobs (auto-configured based on CPU cores and memory)
- `NVCC_VERBOSE`: Enable verbose NVCC output and keep temporary files

### Fast Build Mode
For development, you can enable fast build mode which compiles only a subset of kernels:

```bash
export FAST_BUILD=1
export FAST_HEAD_DIM=64    # Head dimension [32, 64]
export FAST_DEG=4          # Power parameter p [1-4]
export FAST_STATE_DEG=2    # State power parameter [1-4]
export FAST_IS_FP16=false  # Use FP16 vs BF16 [true/false]
export FAST_IS_CAUSAL=true # Enable causal attention [true/false]
export FAST_GATING=true    # Enable gating mechanism [true/false]

pip install -e .[dev]
```

## Verifying Installation

```python
import torch
from power_attention.power_full import create_inputs, power_full

t = 1024
chunk_size=128
b = 8
h = 16
d = 64
deg = 2
gating = True
dtype = torch.float16
inputs = create_inputs(b=b, t=t, h=h, d=d, dtype=dtype, device='cuda', gating=gating, chunk_size=chunk_size, deg=deg, requires_grad=True)

output = power_full(**inputs)
torch.autograd.backward((output,), grad_tensors=(output,))

print("Ran power attention forwards & backwards, output shape:", output.shape)
```

## Troubleshooting

### CUDA/GPU Issues
1. Check CUDA toolkit version:
```bash
nvcc --version
```

2. Check GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

3. Verify PyTorch CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Memory Issues
- Reduce batch size or sequence length
- Use chunking with smaller chunk sizes
- Try fast build with lower degree parameters

For more help, check [GitHub issues](https://github.com/manifestai/power-attention/issues).
