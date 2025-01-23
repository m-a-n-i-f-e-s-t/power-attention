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
from perf.benchmarks import *
from perf._registration import lookup, list_benchmarks, list_groups, get_group
import datetime
import subprocess
import os
from perf._benchmark import Measurement

torch._dynamo.config.cache_size_limit = 128 # Increased from a default of 8 to prevent warnings


def load_results(file_path):
    """Load results from a YAML file and convert to a dict keyed by hashable attributes.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary mapping hashable attributes to Measurement objects
    """
    try:
        with open(file_path) as f:
            raw_results = yaml.safe_load(f) or []
            measurements = [Measurement.from_dict(m) for m in raw_results]
            return {m.hashable_attrs(): m for m in measurements}
    except FileNotFoundError:
        return {}

def get_git_info():
    """Get current git commit hash and check if working directory is clean."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        is_clean = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip() == ''
        return commit_hash, is_clean
    except subprocess.CalledProcessError:
        return None, False

def find_and_load_report(reports_dir, commit_hash):
    """Find and load report file for the given commit hash from a directory.
    
    Args:
        reports_dir: Directory containing report files
        commit_hash: Git commit hash to search for
        
    Returns:
        Tuple of (results_dict, file_path) where results_dict maps hashable attributes 
        to Measurement objects and file_path is the Path of the found file or None
    """
    if not reports_dir.exists():
        return {}, None
    
    for file in reports_dir.glob(f'*_{commit_hash}.yaml'):
        try:
            return load_results(file), file
        except FileNotFoundError:
            continue
    return {}, None

@click.command()
@click.option('--output', '-o', 
              type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help='Output YAML file path (defaults to reports/<datetime>_<commit>.yaml for a clean commit)')
@click.option('--benchmarks', '-b', 
              multiple=True,
              help='Benchmarks or groups of benchmarks to run.')
@click.option('--filter', '-f',
              type=str,
              multiple=True,
              help='Filter measurements by key=value pairs. Can be specified multiple times.')
@click.option('--list', '-l', 
              is_flag=True,
              help='List all available benchmarks and groups')
def main(output, benchmarks, filter, list):
    """Run benchmarks and save results to YAML.
    """    
    if list:
        click.echo("Available benchmarks:")
        for name in sorted(list_benchmarks()):
            benchmark = lookup(name)[0]
            groups_str = f" (groups: {', '.join(sorted(benchmark.groups))})" if benchmark.groups else ""
            click.echo(f"  {name}{groups_str}")
            
        groups = list_groups()
        if groups:
            click.echo("\nAvailable groups:")
            for group in sorted(groups):
                members = get_group(group)
                click.echo(f"  {group}: {', '.join(b.name for b in members)}")
        return

    results = {}
    file_to_be_deleted = None
    if output is None:
        # Find repository root (assumes this file is in perf/ directory)
        repo_root = Path(__file__).parent.parent
        reports_dir = repo_root / 'reports'
        
        commit_hash, is_clean = get_git_info()
        
        if not is_clean:
            click.secho("Warning: Git branch is not clean. Results will not be saved.", fg='red')
        else:
            now = datetime.datetime.now().strftime('%Y%m%d%H%M')
            
            # Load existing report if one exists
            results, file_to_be_deleted = find_and_load_report(reports_dir, commit_hash)
            if file_to_be_deleted:
                click.echo(f"Adding to existing report for commit {commit_hash}")
            
            output = reports_dir / f'{now}_{commit_hash}.yaml'

    else:
        results = load_results(output)

    # Filter benchmarks if specified
    if not benchmarks:
        benchmarks = list_benchmarks() # all benchmarks
    benchmarks = lookup(*benchmarks)
    if filter:
        benchmarks = [benchmark.filter(filter) for benchmark in benchmarks]

    
    # Run selected benchmarks and collect results
    benchmark_str = '\n\t'.join(str(b) for b in benchmarks)
    click.echo(f"Running {len(benchmarks)} benchmarks ({sum(len(b.param_configs) for b in benchmarks)} configs): \n\t{benchmark_str}")
    for benchmark in benchmarks:
        measurements = benchmark(show_progress=True)
        for m in measurements:
            results[m.hashable_attrs()] = m

    # This makes the yaml more readable
    nice_results = [results[k].to_dict() for k in sorted(results.keys(), key=lambda k: str(k))]

    if output:
        # Save results to YAML
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output, 'w') as f:
                yaml.safe_dump(nice_results, f, default_flow_style=False, sort_keys=False)
            if file_to_be_deleted:
                file_to_be_deleted.unlink()  # Delete old report only after successful write
            click.echo(f"Benchmark report written to: {output}")
        except Exception as e:
            click.secho(f"Error writing report: {e}", fg='red')
            if output.exists():
                output.unlink()  # Clean up partial write
            raise
    else:
        click.echo(pprint.pformat(nice_results))

if __name__ == '__main__':
    main()