import os
import re
import subprocess
from pathlib import Path
import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

this_dir = os.path.dirname(os.path.abspath(__file__))

def get_version():
    """Get version from pyproject.toml."""
    with open(os.path.join(this_dir, "pyproject.toml")) as f:
        content = f.read()
    version_match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not version_match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return version_match.group(1)

# Define all possible CUDA sources
ALL_CUDA_SOURCES = [
    'csrc/power_attention/api.cpp',
    'csrc/power_attention/src/update_state_fwd_fp16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_fwd_bf16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_fwd_fp16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_fwd_bf16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_bwd_bf16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_bwd_fp16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_bwd_bf16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/update_state_bwd_fp16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_fwd_bf16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_fwd_fp16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_fwd_bf16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_fwd_fp16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_bwd_bf16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_bwd_fp16_kq_hdim32_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_bwd_bf16_kq_hdim64_deg2_sm80.cu',
    'csrc/power_attention/src/query_state_bwd_fp16_kq_hdim64_deg2_sm80.cu',
    # attention fwd related
    'csrc/power_attention/src/attention/power_fwd_hdim32_fp16_deg1_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_fp16_deg2_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_fp16_deg3_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_fp16_deg4_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_bf16_deg1_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_bf16_deg2_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_bf16_deg3_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim32_bf16_deg4_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_fp16_deg1_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_fp16_deg2_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_fp16_deg3_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_fp16_deg4_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_bf16_deg1_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_bf16_deg2_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_bf16_deg3_causal_sm80.cu',
    'csrc/power_attention/src/attention/power_fwd_hdim64_bf16_deg4_causal_sm80.cu',
    # Backward kernels
    'csrc/power_attention/src/attention/power_bwd_hdim32_fp16_deg1_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_fp16_deg2_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_fp16_deg3_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_fp16_deg4_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_bf16_deg1_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_bf16_deg2_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_bf16_deg3_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim32_bf16_deg4_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_fp16_deg1_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_fp16_deg2_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_fp16_deg3_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_fp16_deg4_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_bf16_deg1_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_bf16_deg2_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_bf16_deg3_sm80.cu',
    'csrc/power_attention/src/attention/power_bwd_hdim64_bf16_deg4_sm80.cu',
    # discumsum
    'csrc/power_attention/src/discumsum_fwd_bf16_sm80.cu',
    'csrc/power_attention/src/discumsum_fwd_fp16_sm80.cu',
    'csrc/power_attention/src/discumsum_fwd_fp32_sm80.cu',
    'csrc/power_attention/src/discumsum_bwd_bf16_sm80.cu',
    'csrc/power_attention/src/discumsum_bwd_fp16_sm80.cu',
    'csrc/power_attention/src/discumsum_bwd_fp32_sm80.cu',
]

def get_cuda_sources():
    """Get CUDA sources based on environment variables."""
    if os.environ.get('FAST_BUILD'):
        # Get configured values
        head_dim = int(os.environ.get('FAST_HEAD_DIM', '64'))
        is_fp16 = os.environ.get('FAST_IS_FP16', 'true').lower() == 'true'
        deg = int(os.environ.get('FAST_DEG', '2'))
        state_deg = int(os.environ.get('FAST_STATE_DEG', '2'))
        
        # Build patterns for matching
        dim_pattern = f'hdim{head_dim}'
        fp_pattern = 'fp16' if is_fp16 else 'bf16'
        deg_pattern = f'deg{deg}'
        state_deg_pattern = f'deg{state_deg}'
        
        # Filter function
        def should_include(file):
            if not file.endswith('.cu'):
                return True
            
            # Always include discumsum files
            if 'discumsum' in file:
                return True
                
            # Handle attention kernels differently
            if 'attention/power' in file:
                # For attention files, we only check hdim and fp type
                # Also need to consider causal files when FAST_IS_CAUSAL is true
                is_causal = os.environ.get('FAST_IS_CAUSAL', 'true').lower() == 'true'
                # Exclude files that don't match our dimension or fp type
                if f'hdim{head_dim}' not in file or fp_pattern not in file:
                    return False
                if is_causal:
                    return 'causal' in file or 'bwd' in file  # Include both causal and backward files
                else:
                    return 'causal' not in file
            
            # For state files (update_state and query_state)
            # More explicit filtering
            if f'hdim{head_dim}' not in file or fp_pattern not in file:
                return False
            if 'state' in file and state_deg_pattern not in file:
                return False
            return True
        
        sources = [
            src for src in ALL_CUDA_SOURCES 
            if should_include(src)
        ]
        print("\nSelected CUDA sources for fast build:")
        for src in sources:
            print(f"  {src}")
        print()
        return sources
    
    return ALL_CUDA_SOURCES

ext_modules = []

PACKAGE_NAME = 'power-attention'

debug_flags = []
if os.environ.get('DEBUG_POWER_BWD_DKDV'):
    debug_flags = ['-DDEBUG_POWER_BWD_DKDV']
if os.environ.get('DEBUG_POWER_BWD_DQ'):
    debug_flags = ['-DDEBUG_POWER_BWD_DQ']

fast_flags = []
if os.environ.get('FAST_BUILD'):
    fast_flags = [
        f'-DFAST_IS_EVEN_MN={os.environ.get("FAST_IS_EVEN_MN", "true")}',
        f'-DFAST_IS_EVEN_K={os.environ.get("FAST_IS_EVEN_K", "true")}',
        f'-DFAST_DEG={os.environ.get("FAST_DEG", "2")}',
        f'-DFAST_STATE_DEG={os.environ.get("FAST_STATE_DEG", "2")}',
        f'-DFAST_GATING={os.environ.get("FAST_GATING", "true")}',
        f'-DFAST_FLASH_EQUIVALENT={os.environ.get("FAST_FLASH_EQUIVALENT", "false")}',
        f'-DFAST_NORMAL_SPACE={os.environ.get("FAST_NORMAL_SPACE", "false")}',
        f'-DFAST_IS_CAUSAL={os.environ.get("FAST_IS_CAUSAL", "true")}',
        f'-DFAST_IS_FP16={os.environ.get("FAST_IS_FP16", "true")}',
        f'-DFAST_HEAD_DIM={os.environ.get("FAST_HEAD_DIM", "64")}'
    ]

    print("\nFast build configuration:")
    print(f"  Head dimension: {os.environ.get('FAST_HEAD_DIM')}")
    print(f"  Using FP16: {os.environ.get('FAST_IS_FP16')}")
    print(f"  Degree: {os.environ.get('FAST_DEG')}")
    print()

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get('MAX_JOBS'):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() - 1)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9,
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to
            # minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ['MAX_JOBS'] = str(max_jobs)

        print(f'MAX_JOBS: {os.environ["MAX_JOBS"]}')
        super().__init__(*args, **kwargs)


class CustomBdistWheel(_bdist_wheel):
    def get_tag(self):
        tag = super().get_tag()
        if debug_flags:
            tag = (tag[0], tag[1], tag[2] + '.debug')
        return tag


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    diagnostic_flags = ["--display-error-number", "--resource-usage"]
    if os.environ.get('NVCC_VERBOSE'):
        diagnostic_flags.extend(["--verbose", "--keep", "--keep-dir", "nvcc_temp"])
    return nvcc_extra_args + ["--threads", nvcc_threads] + diagnostic_flags


subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])

torch._C._GLIBCXX_USE_CXX11_ABI = False  # noqa: SLF001
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

ext_modules.append(
    CUDAExtension(
        name='power_attention_cuda',
        sources=get_cuda_sources(),
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'] + generator_flag + debug_flags + fast_flags,
            'nvcc': append_nvcc_threads(
                [
                    '-O3',
                    '-std=c++17',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '--use_fast_math',
                    '-maxrregcount=255',
                    '-gencode',
                    'arch=compute_80,code=sm_80',
                    '-gencode',
                    'arch=compute_80,code=compute_80',
                    '-Xcompiler',
                    '-rdynamic',
                    '-lineinfo',
                    '--ptxas-options=-v', 
                    '-Xptxas',
                    '-warn-lmem-usage',
                ]
                + generator_flag
                + fast_flags
            ),
        },
        include_dirs=[
            Path(this_dir) / 'csrc' / 'power_attention',
            Path(this_dir) / 'csrc' / 'power_attention' / 'src',
            Path(this_dir) / 'csrc' / 'power_attention' / 'src' / 'attention',
            Path(this_dir) / 'csrc' / 'cutlass' / 'include',
            Path(this_dir) / 'csrc' / 'cutlass' / 'tools' / 'include',
        ],
    ),
)

# Development dependencies
dev_requires = [
    'numpy>=2.2',
    'torch>=2.5',
    'triton>=3.1',
    'setuptools==69.5.1',
    'einops',
    'ninja==1.11.1.3',
    'pybind11==2.13.6',
    'psutil',
    'tiktoken',
    'datasets',
    'httpx',
    'tqdm',
    'pytest>=8.3.4',
    'pytest-xdist',
    'wheel',
    'build',
    'cibuildwheel',
    'auditwheel',
    'twine',
    'click',
    'pyyaml',
    'bokeh',
    'matplotlib',
    'pymdown-extensions',
    'mkdocs',
    'mkdocstrings-python',
]

setup(
    name=PACKAGE_NAME,
    version=get_version(),
    packages=find_packages(
        exclude=('build', 'csrc', 'include', 'tests', 'dist', 'benchmarks'),
    ),
    ext_modules=ext_modules,
    cmdclass={
        'bdist_wheel': CustomBdistWheel,
        'build_ext': NinjaBuildExtension,
    },
    extras_require={
        'dev': dev_requires,
    },
)
