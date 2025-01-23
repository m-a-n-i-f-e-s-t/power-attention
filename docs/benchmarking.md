# Benchmarking

This guide explains how to run benchmarks for symmetric power attention and visualize the results.

<div class="admonition note">
<p class="admonition-title">Prerequisites</p>
<p>All benchmarks require an NVIDIA GPU with compute capability 8.0+ (Ampere or newer).</p>
</div>

## Quick Start

Install benchmarking dependencies:

```bash
pip install -e .[dev]  # Includes benchmarking tools
```

## Running Benchmarks

The benchmarking system allows you to run various benchmarks and track performance over time. Results are saved as YAML files that can be later visualized.

### Basic Usage

```bash
# Run all available benchmarks
python -m perf.create_report

# Run specific benchmarks
python -m perf.create_report -b speed -b precision

# Filter benchmark configurations
python -m perf.create_report -f t=512 -f chunk_size=128

# Specify custom output file
python -m perf.create_report -o my_results.yaml
```

By default, when running without specifying an output file:
- Results are saved in the `reports/` directory
- Filenames follow the format `YYYYMMDDHHMM_<commit-hash>.yaml`
- If a report for the current commit exists, results are added to it
- Reports are only saved if the git working directory is clean

### Benchmark Reports

Each benchmark report contains detailed measurements with their configurations. Example report entry:

```yaml
- attrs:
    batch_size: 32
    seq_len: 2048
    num_heads: 12
    head_dim: 64
    dtype: torch.bfloat16
    device: cuda
  value: 123.45    # The measured value (e.g., milliseconds, memory usage)
```

## Visualizing Results

The benchmarking suite includes a plotting tool to visualize results across time interactively:

```bash
python -m perf.plot_reports -o my_plots.html
```