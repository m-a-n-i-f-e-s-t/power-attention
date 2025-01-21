import click
import yaml
from pathlib import Path
import pprint
from typing import Dict, Any, List
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import save, output_file
from perf._registration import lookup, list_benchmarks
from perf._benchmark import Measurement, Benchmark
from perf.benchmarks import *


def load_all_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    """Load and combine all YAML reports from the reports directory.
    
    Returns:
        List of dicts with keys 'commit', 'datetime', and 'measurements'
        The datetime is parsed from the filename format YYMMDDHHMM_hash.yaml
    """
    results = []
    
    for yaml_path in reports_dir.glob('*.yaml'):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or []
                # Extract commit hash and datetime from filename YYMMDDHHMM_hash.yaml
                datetimestr, commit = yaml_path.stem.split('_')
                
                # Convert raw dicts to Measurement objects
                measurements = [Measurement.from_dict(m) for m in data]
                
                results.append(dict(commit=commit, datetime=datetimestr, measurements=measurements))
                
        except Exception as e:
            click.echo(f"Warning: Could not load {yaml_path}: {e}", err=True)
    
    return results

def filter_measurements(measurements: List[Measurement], benchmarks: List[Benchmark], filters: List[str]) -> List[Measurement]:
    """Filter measurements based on benchmark names and key=value filters."""
    # Convert filters to dict
    filter_dict = {}
    for f in filters:
        try:
            key, value = f.split('=')
            filter_dict[key] = value
        except ValueError:
            click.echo(f"Warning: Ignoring invalid filter '{f}'", err=True)
    
    filtered = []
    benchmark_names = set(b.name for b in benchmarks)
    for m in measurements:
        # Check if measurement belongs to selected benchmarks
        if m.attrs['benchmark'] not in benchmark_names:
            continue
            
        # Check if measurement matches all filters
        matches = True
        for key, value in filter_dict.items():
            m_attr = m.get(key)
            if m_attr is None:
                matches = False
                break
            try:
                # Try to convert value to same type as m_attr
                converted_value = type(m_attr)(value)
                if m_attr != converted_value:
                    matches = False
                    break
            except (ValueError, TypeError):
                # If conversion fails, no match
                matches = False
                break
        if matches:
            filtered.append(m)
    
    return filtered

def create_plot(reports: List[Dict[str, Any]]):
    """Create an interactive plot of the measurements."""
    all_unique_measurements = {
        m.hashable_attrs()
        for report in reports
        for m in report['measurements']
    }
    reports = sorted(
        [
            {
                'commit': report['commit'],
                'datetime': report['datetime'],
                'measurements': {m.hashable_attrs(): m for m in report['measurements']}
        }
        for report in reports
    ], key=lambda r: r['datetime'])

    lines = {}
    for m_attrs in all_unique_measurements:
        commit_idxs = []
        commits = []
        values = []
        datetimes = []
        # Create separate lists for each attribute
        attr_lists = {}
        
        for idx, report in enumerate(reports):
            if m_attrs in report['measurements']:
                measurement = report['measurements'][m_attrs]
                commit_idxs.append(idx)
                commits.append(report['commit'])
                dt = report['datetime']
                formatted_dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[8:10]}:{dt[10:12]}"
                datetimes.append(formatted_dt)
                values.append(measurement.value)
                
                # Store each attribute separately
                for key, value in measurement.attrs.items():
                    if key not in attr_lists:
                        attr_lists[key] = []
                    attr_lists[key].append(str(value))
                
        data_dict = {
            'commit_idxs': commit_idxs,
            'commits': commits,
            'values': values,
            'datetimes': datetimes,
        }
        # Add each attribute list to the data dictionary
        data_dict.update(attr_lists)
        lines[m_attrs] = data_dict

    # Create the figure
    p = figure(width=800, height=600, tools='pan,box_zoom,reset,save')
    p.title.text = "Benchmark Results Over Time"
    p.xaxis.axis_label = "Commit"
    p.yaxis.axis_label = "Value"

    # Plot each configuration as a separate line
    for info_dict in lines.values():
        source = ColumnDataSource(info_dict)
        p.line('commit_idxs', 'values', line_width=1, line_alpha=0.5, source=source)
        p.scatter('commit_idxs', 'values', size=4, alpha=0.5, source=source)
    
    # Create hover tool with separate line for each attribute
    tooltips = [
        ('Datetime', '@datetimes'),
        ('Commit', '@commits'),
        ('Value', '@values{0.000}'),
    ]
    # Add all possible attributes as tooltips
    all_attrs = set().union(*(d.keys() for d in lines.values()))
    attr_keys = sorted(k for k in all_attrs if k not in {'commit_idxs', 'commits', 'values', 'datetimes'})
    tooltips.extend((f'[{key}]', f'@{key}') for key in attr_keys)
    
    hover = HoverTool(tooltips=tooltips)
    p.add_tools(hover)
        
    # Style the plot
    p.grid.grid_line_alpha = 0.3
    p.xaxis.major_label_orientation = 0.7
    
    # Set x-axis ticks to be evenly spaced commits
    p.xaxis.major_label_overrides = dict(zip(range(len(reports)), [r['datetime'] for r in reports]))
    
    return p

@click.command()
@click.option('--benchmarks', '-b', 
              multiple=True,
              help='Benchmarks or groups of benchmarks to plot.')
@click.option('--filter', '-f',
              type=str,
              multiple=True,
              help='Filter measurements by key=value pairs. Can be specified multiple times.')
@click.option('--output', '-o',
              type=click.Path(dir_okay=False),
              help='Save plot to the specified file instead of displaying it.')
def main(benchmarks: List[str], filter: List[str], output: str):
    """Plot benchmark results from all report files."""
    # Find repository root and reports directory
    repo_root = Path(__file__).parent.parent
    reports_dir = repo_root / 'reports'
    
    if not reports_dir.exists():
        click.echo("No reports directory found!")
        return
    reports = load_all_reports(reports_dir)
    if not reports:
        click.echo("No measurements found in reports!")
        return
    
    # Get selected benchmarks
    if not benchmarks:
        benchmarks = list_benchmarks()  # all benchmarks
    benchmarks = lookup(*benchmarks)
    if filter:
        benchmarks = [benchmark.filter(filter) for benchmark in benchmarks]
    benchmark_str = '\n\t'.join(str(b) for b in benchmarks)
    click.echo(f"Plotting {len(benchmarks)} benchmarks ({sum(len(b.param_configs) for b in benchmarks)} configs): \n\t{benchmark_str}")

    # Filter measurements
    filtered_reports = [dict(commit=report['commit'], 
                             datetime=report['datetime'], 
                             measurements=filter_measurements(report['measurements'], benchmarks, filter)) 
                        for report in reports]
    
    # Create and show/save the plot
    p = create_plot(filtered_reports)
    if output:
        output_file(output)
        save(p, output)
    else:
        show(p)

if __name__ == '__main__':
    main()
