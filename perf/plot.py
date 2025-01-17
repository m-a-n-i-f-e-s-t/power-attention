import click
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Spectral11, Category20
from bokeh.models import (
    ColumnDataSource, HoverTool, Legend, LegendItem,
    Div, Range1d
)

def load_benchmark_data(yaml_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load benchmark data from a YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-serializable types to strings."""
    df = df.copy()
    for col in df.columns:
        # Convert any range objects to strings
        if any(isinstance(x, range) for x in df[col].dropna()):
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, range) else x)
        # Convert any other problematic types to strings
        if any(isinstance(x, (np.bool_, np.integer, np.floating)) for x in df[col].dropna()):
            df[col] = df[col].astype(str)
    return df

def create_hover_tool():
    """Create a HoverTool with common tooltips."""
    tooltips = [
        ('Value', '@value{0.000}'),
        ('Sample', '@sample'),
    ]
    return HoverTool(tooltips=tooltips)

def style_figure(p, title):
    """Apply common styling to a figure."""
    p.title.text = title
    p.title.text_font_size = '14pt'
    p.title.text_font = 'helvetica'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_dash = [6, 4]
    p.background_fill_color = "#fafafa"
    p.border_fill_color = "white"
    p.outline_line_color = None
    p.min_border_left = 100   # More space for legend
    return p

def create_plot_for_group(group_data: pd.DataFrame, title: str) -> figure:
    """Create a single plot for a group of data."""
    p = figure(width=400, height=300, tools='pan,box_zoom,reset,save')
    p = style_figure(p, title)
    p.xaxis.axis_label = 'Sample'
    p.yaxis.axis_label = 'Value'
    
    # Plot the data
    source = ColumnDataSource({
        'sample': list(range(len(group_data))),
        'value': group_data['value'].values
    })
    
    line = p.line('sample', 'value', line_color='blue',
                  line_width=2, source=source)
    circle = p.circle('sample', 'value', size=6,
                     color='blue', source=source)
    
    hover = create_hover_tool()
    p.add_tools(hover)
    
    return p

def plot_benchmark_results(data: Dict[str, List[Dict[str, Any]]], output_path: Path):
    """Create a single HTML file with all benchmark results."""
    output_file(output_path, title="Benchmark Results")
    
    # Add page title
    title = Div(
        text='<h1>Benchmark Results</h1>',
        width=800,
        styles={
            'font-family': 'helvetica',
            'text-align': 'center',
            'margin': '20px 0',
            'color': '#333'
        }
    )
    
    all_plots = [title]
    
    # Process each benchmark type
    for benchmark_name, results in data.items():
        # Convert to DataFrame and preprocess
        df = pd.DataFrame(results)
        df = preprocess_data(df)
        
        # Create section title
        section_title = Div(
            text=f'<h2>{benchmark_name}</h2>',
            width=800,
            styles={
                'font-family': 'helvetica',
                'border-bottom': '2px solid #666',
                'margin': '20px 0'
            }
        )
        all_plots.append(section_title)
        
        # Group by all columns except 'value' and 'sample'
        group_cols = [col for col in df.columns if col != 'value']
        grouped = df.groupby(group_cols)
        
        # Create plots for each unique combination
        plots_in_section = []
        row_of_plots = []
        
        for name, group in grouped:
            # Create title from group parameters
            param_strs = []
            for col, val in zip(group_cols, name):
                param_strs.append(f"{col}={val}")
            plot_title = " | ".join(param_strs)
            
            # Create plot
            p = create_plot_for_group(group, plot_title)
            row_of_plots.append(p)
            
            # Create rows of 3 plots
            if len(row_of_plots) == 3:
                plots_in_section.append(row(*row_of_plots))
                row_of_plots = []
        
        # Add any remaining plots
        if row_of_plots:
            plots_in_section.append(row(*row_of_plots))
        
        # Add all plots for this section
        all_plots.append(column(plots_in_section))
        
        # Add separator
        separator = Div(
            text='<hr style="margin: 40px 0; border: none; border-top: 1px solid #ccc;">',
            width=800
        )
        all_plots.append(separator)
    
    # Save the final layout
    save(column(all_plots))

@click.command()
@click.argument('yaml_files', type=click.Path(exists=True, path_type=Path), nargs=-1)
@click.option('--output', '-o', 
              type=click.Path(dir_okay=False, path_type=Path),
              default='benchmark_results.html',
              help='Output HTML file (default: benchmark_results.html)')
def main(yaml_files: List[Path], output: Path):
    """Plot benchmark results from YAML files."""
    if not yaml_files:
        click.echo("No YAML files provided!")
        return
    
    # For now, just use the first file
    yaml_path = yaml_files[0]
    data = load_benchmark_data(yaml_path)
    plot_benchmark_results(data, output)
    
    click.echo(f"Interactive plots saved to {output}")

if __name__ == '__main__':
    main()
