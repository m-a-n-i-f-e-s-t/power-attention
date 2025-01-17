from typing import Callable, Dict, List, Optional, Any
from collections import defaultdict
from perf._benchmark import Benchmark

# Global registry state
_benchmarks: Dict[str, Benchmark] = {}
_groups: Dict[str, List[str]] = defaultdict(list)

def register_benchmark(
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

def lookup(*names: str) -> List[Benchmark]:
    """Get benchmarks by name or group name.
    
    Args:
        *names: Names of benchmarks or benchmark groups
        
    Returns:
        List of matching benchmarks, with duplicates removed
        
    Raises:
        KeyError: If any name is not found as either a benchmark or group
    """
    result = set()
    for name in names:
        try:
            # Try as individual benchmark first
            benchmark = get_benchmark(name)
            result.add(benchmark)
        except KeyError:
            try:
                # Try as group name
                result.update(get_group(name))
            except KeyError:
                raise KeyError(f"'{name}' not found as benchmark or group name")
    return sorted(result, key=lambda x: x.name)

def list_benchmarks() -> List[str]:
    """List all registered benchmark names."""
    return list(_benchmarks.keys())

def list_groups() -> List[str]:
    """List all registered group names."""
    return list(_groups.keys())
