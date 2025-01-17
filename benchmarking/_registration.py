from typing import Callable, Dict, List, Optional, Any
from collections import defaultdict
from benchmarking._benchmark import Benchmark

# Global registry state
_benchmarks: Dict[str, Benchmark] = {}
_groups: Dict[str, List[str]] = defaultdict(list)

def register(
    param_configs: List[Dict[str, Any]] = [{}],
    groups: Optional[List[str]] = None,
    label: Optional[str] = None,
) -> Callable:
    """Decorator to register a benchmark function."""
    def decorator(func: Callable) -> Callable:
        benchmark = Benchmark(
            func=func,
            param_configs=param_configs,
            groups=set(groups or []),
            label=label
        )       

        if benchmark.name in _benchmarks:
            raise ValueError(f"Benchmark {benchmark.name} already registered")
        
        _benchmarks[benchmark.name] = benchmark
        if benchmark.groups:
            for group in benchmark.groups:
                _groups[group].append(benchmark.name)
        
        return func
    return decorator

def get_benchmark(name: str) -> Benchmark:
    """Get a benchmark by name."""
    if name not in _benchmarks:
        raise KeyError(f"No benchmark named {name}")
    return _benchmarks[name]

def get_group(group: str) -> List[Benchmark]:
    """Get all benchmarks in a group."""
    if group not in _groups:
        raise KeyError(f"No benchmark group named {group}")
    return [_benchmarks[name] for name in _groups[group]]

def list_benchmarks() -> List[str]:
    """List all registered benchmark names."""
    return list(_benchmarks.keys())

def list_groups() -> List[str]:
    """List all registered group names."""
    return list(_groups.keys())
