from typing import Callable, Dict, List, Set, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
from tests_and_benchmarks._utils import check_filter_matches

@dataclass
class Measurement:
    """A result of calling a benchmark function."""
    attrs: Dict[str, Any]
    value: float
    name: Optional[str] = None

@dataclass
class Benchmark:
    """A benchmark function and its associated parameter configurations."""
    func: Callable
    param_configs: List[Dict[str, Any]]
    groups: Set[str]
    label: Optional[str] = None

    @property
    def fn_name(self) -> str:
        """Name of the benchmark function."""
        return self.func.__name__

    @property
    def name(self) -> str:
        """Full name of the benchmark, including lablel as suffix if present."""
        if self.label:
            return f"{self.fn_name}_{self.label}"
        return self.fn_name
    
    def __str__(self) -> str:
        """String representation of the benchmark."""
        return f'<Benchmark {self.name} ({len(self.param_configs)})>'
    
    def __repr__(self) -> str:
        """String representation of the benchmark."""
        return str(self)
    
    def __hash__(self) -> int:
        """Make benchmark hashable by its name."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Define equality based on name for hash consistency."""
        if not isinstance(other, Benchmark):
            return NotImplemented
        return self.name == other.name
    
    def filter(self, filter: Dict[str, Any]) -> 'Benchmark':
        """Filter the benchmark's parameter configurations and return a new Benchmark with filtered configs.
        
        Args:
            filter: List of filter strings in "key=value" format
            
        Returns:
            New Benchmark object with filtered parameter configurations and updated label
        """
        filtered_configs = [params for params in self.param_configs if check_filter_matches(filter, params)]
        filter_str = "filtered_" + ",".join(filter)
        new_label = f"{self.label}_{filter_str}" if self.label else filter_str
        return Benchmark(
            func=self.func,
            param_configs=filtered_configs, 
            groups=self.groups,
            label=new_label
        )

    def __call__(self) -> List[Measurement]:
        """Run the benchmark function with all parameter configurations."""
        measurements = []
        for params in self.param_configs:

            result = self.func(**params)
            
            # Handle different return types
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                # List of results
                for r in result:
                    assert isinstance(r, Measurement)
                    # Create new Measurement with name and merged attrs
                    measurements.append(Measurement(
                        attrs={**params, **r.attrs},
                        value=r.value,
                        name=self.fn_name  # Use base_name for measurements
                    ))
            elif isinstance(result, Measurement):
                # Single Measurement
                measurements.append(Measurement(
                    attrs={**params, **result.attrs},
                    value=result.value,
                    name=self.fn_name  # Use base_name for measurements
                ))
            else:
                # Single value - must be float
                if not isinstance(result, float):
                    raise TypeError(f"Benchmark {self.name} returned {type(result)}, expected float, Measurement, or iterable of Measurements")
                measurements.append(Measurement(
                    attrs=params,
                    value=result,
                    name=self.fn_name  # Use base_name for measurements
                ))
        return measurements
