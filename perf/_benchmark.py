from typing import Callable, Dict, List, Set, Optional, Any, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
from perf._utils import check_filter_matches
import torch
import click
import numpy as np

@dataclass
class Measurement:
    """A result of calling a benchmark function."""
    value: float
    attrs: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.attrs is None:
            self.attrs = {}
        self.attrs = {k: make_serializable(v) for k, v in self.attrs.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the measurement."""
        return {
            'attrs': self.attrs,
            'value': self.value
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Measurement':
        """Create a Measurement from a dictionary."""
        return cls(attrs=d['attrs'], value=d['value'])
    
    def hashable_attrs(self):
        return tuple(sorted(self.attrs.items()))
    

def make_serializable(obj):
    """Convert non-serializable objects to strings."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return tuple(make_serializable(x) for x in obj)
    return obj


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

    def __call__(self, show_progress: bool = False) -> List[Measurement]:
        """Run the benchmark function with all parameter configurations.
        
        Args:
            show_progress: Whether to show a progress bar for parameter configurations
        """
        measurements = []
        configs = self.param_configs
        
        # Create iterator based on whether we want progress bar
        if show_progress:
            with click.progressbar(
                [configs[i] for i in np.random.permutation(len(configs))], # shuffling the configs gives a more even distribution of progress bar
                label=f'Running {self.name}',
                length=len(configs),
                show_pos=True
            ) as bar:
                for params in bar:
                    measurements.extend(self._run_single_config(params))
        else:
            for params in configs:
                measurements.extend(self._run_single_config(params))
                
        return measurements
    
    def _run_single_config(self, params) -> List[Measurement]:
        """Run benchmark with a single parameter configuration."""
        try:
            result = self.func(**params)
        except Exception as e:
            raise RuntimeError(f"\nError running {self.name} with params {params}: {e}") from e
        
        # Handle different return types
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
            return [
                Measurement(
                    attrs={**params, **r.attrs, 'benchmark': self.name, 'fn': self.fn_name},
                    value=r.value) 
                for r in result
            ]
        elif isinstance(result, Measurement):
            # Single Measurement
            return [Measurement(
                attrs={**params, **result.attrs, 'benchmark': self.name, 'fn': self.fn_name},
                value=result.value
            )]
        else:
            # Single value - must be float
            if not isinstance(result, float):
                raise TypeError(f"Benchmark {self.name} returned {type(result)}, expected float, Measurement, or iterable of Measurements")
            return [Measurement(
                attrs={**params, 'benchmark': self.name, 'fn': self.fn_name},
                value=result
            )]