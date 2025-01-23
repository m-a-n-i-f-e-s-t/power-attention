import click
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set, Hashable
from bokeh.plotting import figure, show
from bokeh.models import (
    ColumnDataSource, 
    HoverTool, 
    Select, 
    Column, 
    Row,
    CustomJS,
    Div,
    CDSView,
    IndexFilter,
    BooleanFilter,
    Tabs,
    TabPanel
)
from bokeh.layouts import layout
from bokeh.io import save, output_file, curdoc
from perf._registration import lookup, list_benchmarks
from perf._benchmark import Measurement, Benchmark
from perf._utils import filter_measurements
from perf.benchmarks import *

def load_all_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    """Load and combine all YAML reports from the reports directory."""
    results = []
    
    for yaml_path in reports_dir.glob('*.yaml'):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f) or []
                datetimestr, commit = yaml_path.stem.split('_')
                measurements = [Measurement.from_dict(m) for m in data]
                results.append(dict(commit=commit, datetime=datetimestr, measurements=measurements))
        except Exception as e:
            click.echo(f"Warning: Could not load {yaml_path}: {e}", err=True)
    
    return results

def get_unique_values(reports: List[Dict[str, Any]], attr: str) -> Set[Any]:
    """Get all unique values for a given attribute across all measurements."""
    values = set()
    for report in reports:
        for m in report['measurements']:
            if attr in m.attrs:
                values.add(str(m.attrs[attr]))
    return values

def create_plot_for_benchmark(reports: List[Dict[str, Any]], benchmark_name: str):
    """Create an interactive plot for a specific benchmark type."""
    # Sort reports by datetime
    reports = sorted(reports, key=lambda r: r['datetime'])
    
    # Filter measurements for this benchmark type
    filtered_reports = []
    for report in reports:
        measurements = [m for m in report['measurements'] 
                       if m.attrs.get('benchmark', '') == benchmark_name]
        if measurements:
            filtered_reports.append({
                'commit': report['commit'],
                'datetime': report['datetime'],
                'measurements': measurements
            })
    
    if not filtered_reports:
        return None
    
    # Get all unique attributes and their values
    all_attrs = set()
    for report in filtered_reports:
        for m in report['measurements']:
            all_attrs.update(m.attrs.keys())
    
    # Create initial data source with all measurements
    data = {
        'commit_idxs': [],
        'commits': [],
        'datetimes': [],
        'values': [],
    }
    # Add columns for all attributes
    for attr in all_attrs:
        data[attr] = []
    
    # Populate data
    for idx, report in enumerate(filtered_reports):
        for m in report['measurements']:
            data['commit_idxs'].append(idx)
            data['commits'].append(report['commit'])
            dt = report['datetime']
            formatted_dt = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]} {dt[8:10]}:{dt[10:12]}"
            data['datetimes'].append(formatted_dt)
            data['values'].append(m.value)
            
            # Add all attributes
            for attr in all_attrs:
                data[attr].append(str(m.attrs.get(attr, '')))
    
    # Verify all columns have the same length
    lengths = {k: len(v) for k, v in data.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Inconsistent column lengths: {lengths}")
    
    source = ColumnDataSource(data)
    
    # Create dropdowns for filtering with styling
    excluded_attrs = {'commit_idxs', 'commits', 'values', 'datetimes', 'gpus', 'fn', 'benchmark'}
    key_attrs = sorted([attr for attr in all_attrs if attr not in excluded_attrs])
    dropdowns = []
    
    # Create initial filter based on first values
    initial_indices = list(range(len(data['values'])))  # Start with all indices
    for attr in key_attrs:
        values = sorted(get_unique_values(filtered_reports, attr))
        if values:
            # Create dropdown
            select = Select(
                title=attr,
                value=values[0] if values else 'all',  # Default to first value
                options=['all'] + list(values),
                width=200,
                styles={'margin': '4px'}
            )
            dropdowns.append(select)
            
            # Apply initial filter
            if values:
                initial_indices = [i for i in initial_indices 
                                 if data[attr][i] == values[0]]
    
    # Create filtered data with initial selection
    filtered_data = {k: [v[i] for i in initial_indices] for k, v in data.items()}
    filtered_source = ColumnDataSource(filtered_data)
    
    # Create figure with styling
    p = figure(width=1000, height=600, tools='pan,box_zoom,reset,save')
    p.title.text = f"{benchmark_name} Results Over Time"
    p.title.text_font_size = '16pt'
    p.xaxis.axis_label = "Commit"
    p.yaxis.axis_label = "Value"
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    
    # Plot points and lines - draw lines first so points appear on top
    p.line('commit_idxs', 'values', line_width=2, line_alpha=0.8, source=filtered_source)
    p.scatter('commit_idxs', 'values', size=8, alpha=1.0, source=filtered_source)
    
    # Style the plot
    p.grid.grid_line_alpha = 0.3
    p.xaxis.major_label_orientation = 0.7
    
    # Set x-axis to show commit hashes and only at data points
    p.xaxis.ticker = list(range(len(filtered_reports)))
    p.xaxis.major_label_overrides = {i: r['commit'] for i, r in enumerate(filtered_reports)}
    
    # Create hover tool with styling
    tooltips = """
        <div style="padding: 10px;">
            <div>Datetime: @datetimes</div>
            <div>Commit: @commits</div>
            <div>Value: @values{0.000}</div>
    """
    
    for attr in sorted(all_attrs):
        if attr not in {'commit_idxs', 'commits', 'values', 'datetimes'}:
            tooltips += f"""
            <div>{attr}: @{attr}</div>"""
    
    tooltips += "</div>"
    
    hover = HoverTool(tooltips=tooltips)
    p.add_tools(hover)
    
    # Create JavaScript callback for filtering
    js_code = """
    const data = source.data;
    const indices = [];
    
    for (let i = 0; i < data.values.length; i++) {
        let match = true;
        """
    
    for i, attr in enumerate(key_attrs):
        if attr in all_attrs:
            js_code += f"""
        if (dropdown{i}.value !== 'all' && data['{attr}'][i] !== dropdown{i}.value) {{
            match = false;
        }}
            """
    
    js_code += """
        if (match) {
            indices.push(i);
        }
    }
    
    // Update filtered source with matching data
    const filtered_data = {};
    Object.keys(data).forEach(key => {
        filtered_data[key] = indices.map(i => data[key][i]);
    });
    filtered_source.data = filtered_data;
    """
    
    callback = CustomJS(
        args=dict(
            source=source,
            filtered_source=filtered_source,
            **{f'dropdown{i}': dropdown for i, dropdown in enumerate(dropdowns)}
        ),
        code=js_code
    )
    
    for dropdown in dropdowns:
        dropdown.js_on_change('value', callback)
    
    # Create layout with styling
    controls = Column(children=dropdowns, spacing=10, styles={'margin': '20px'})
    plot_layout = Row(controls, p, spacing=20)
    
    return plot_layout

def create_interactive_plot(reports: List[Dict[str, Any]]):
    """Create tabs with interactive plots for each benchmark type."""
    # Get all unique benchmark types
    benchmark_types = set()
    for report in reports:
        for m in report['measurements']:
            if 'benchmark' in m.attrs:
                benchmark_types.add(m.attrs['benchmark'])
    
    # Create a tab for each benchmark type
    tabs = []
    for benchmark in sorted(benchmark_types):
        plot = create_plot_for_benchmark(reports, benchmark)
        if plot is not None:
            panel = TabPanel(child=plot, title=benchmark)
            tabs.append(panel)
    
    return Tabs(tabs=tabs)

@click.command()
@click.option('--output', '-o',
              type=click.Path(dir_okay=False),
              help='Save plot to the specified file instead of displaying it.')
@click.option('--number_of_reports', '-n',
              type=int,
              default=None,
              help='Number of most recent reports to include.')
def main(output: str, number_of_reports: int):
    """Plot benchmark results with interactive filtering."""
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
    
    # Sort reports by datetime and limit if specified
    reports = sorted(reports, key=lambda r: r['datetime'])
    if number_of_reports is not None:
        reports = reports[-number_of_reports:]
    
    # Apply dark theme
    curdoc().theme = 'night_sky'
    
    # Create interactive plot
    plot = create_interactive_plot(reports)
    
    # Save or show
    if output:
        output_file(output, title="Benchmark Results")
        save(plot)
        click.echo(f"Plot saved to {output}")
    else:
        show(plot)

if __name__ == '__main__':
    main() 