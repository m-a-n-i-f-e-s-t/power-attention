# Benchmarking either precision or timing of target implementations, to
# detect regressions over time.
# The idea is to run this every commit, which will produce a YAML file with the results,
# in which we can see the error and time for each test case. (Error and times are nothing special,
# a benchmark could produce any scalar value.)

import torch
import click
import yaml
from pathlib import Path
import pprint
from tests_and_benchmarks.benchmarks import *
from tests_and_benchmarks._registration import lookup, list_benchmarks

torch._dynamo.config.cache_size_limit = 64 # Increased from a default of 8 to prevent warnings

def make_serializable(obj):
    """Convert non-serializable objects to strings."""
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return obj


def load_existing_results(output_path):
    """Load existing results from YAML file if it exists."""
    try:
        with open(output_path) as f:
            return yaml.safe_load(f) or []
    except FileNotFoundError:
        return []


@click.command()
@click.option('--output', '-o', 
              type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help='Output YAML file path (defaults to benchmark_results.yaml)')
@click.option('--benchmarks', '-b', 
              multiple=True,
              help='Benchmarks or groups of benchmarks to run.')
@click.option('--reset', is_flag=True,
              help='Start with fresh results, ignoring any existing ones in the output file.')
@click.option('--filter', '-f',
              type=str,
              multiple=True,
              help='Filter measurements by key=value pairs. Can be specified multiple times.')
def main(output, benchmarks, reset, filter):
    """Run benchmarks and save results to YAML.
    
    If no benchmarks specified, runs all discovered benchmarks.
    If no output specified, saves to benchmark_results.yaml
    By default, updates existing results in the output file unless --reset is specified.
    """

    # Filter benchmarks if specified
    if not benchmarks:
        benchmarks = list_benchmarks() # all benchmarks
    benchmarks = lookup(*benchmarks)
    if filter:
        benchmarks = [benchmark.filter(filter) for benchmark in benchmarks]

    benchmark_str = '\n\t'.join(str(b) for b in benchmarks)
    click.echo(f"Running {len(benchmarks)} benchmarks ({sum(len(b.param_configs) for b in benchmarks)} configs): \n\t{benchmark_str}")

    # Load existing results unless reset is specified
    results = [] if (reset or not output) else load_existing_results(output)
    
    # Run selected benchmarks and collect results
    with click.progressbar(benchmarks, label='Running benchmarks') as bar:
        for benchmark in bar:
            measurements = benchmark()
            for measurement in measurements:
                results.append({
                    'name': measurement.name,
                    'attrs': measurement.attrs,
                    'value': measurement.value
                })
    
    # Make results serializable before saving
    serializable_results = make_serializable(results)
    
    if output:
        # Save results to YAML
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            yaml.safe_dump(serializable_results, f, default_flow_style=False, sort_keys=False)        
        click.echo(f"Benchmark results {'saved to' if reset else 'updated in'} {output}")
    else:
        click.echo(pprint.pformat(serializable_results))

if __name__ == '__main__':
    main()