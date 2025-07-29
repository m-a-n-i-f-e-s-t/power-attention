import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
import numpy as np

def parse_csv(csv_string):
    # Split the input into lines and strip whitespace
    lines = csv_string.strip().split("\n")
    
    # Extract the column headers (ignore the first "ctx" column)
    headers = lines[0].split(",")[1:]
    
    # Initialize a dictionary to hold the lists
    data = {header: [] for header in headers}
    
    # Process each data row
    for line in lines[1:]:
        values = line.split(",")
        ctx = float(values[0])  # The first value is "ctx"
        for header, value in zip(headers, values[1:]):
            # Skip empty values
            if value.strip():
                data[header].append((ctx, float(value)))
    
    return data

TIME = parse_csv("""ctx,sdpa,p1_att,p2_att,p1_chunk,p2_chunk
1024,18.181974,37.414570,21.654699,28.036261,
2048,24.330917,68.275370,33.866069,30.796459,
4096,42.051071,126.923770,56.256683,29.481472,115.035989
8192,77.539330,246.072662,100.871862,29.374123,115.712341
16384,145.849002,483.192327,193.776314,29.150208,114.998955
32768,284.431697,955.695465,377.833288,28.655615,115.007147
65536,554.540059,1907.619161,749.117452,30.756026,115.844266""")

FLOPS = parse_csv("""ctx,sdpa,p1_att,p2_att,p1_chunk,p2_chunk
1024,2.062e+11,2.094e+11,2.094e+11,5.214e+10,
2048,4.123e+11,4.188e+11,4.188e+11,5.214e+10,
4096,8.246e+11,8.375e+11,8.375e+11,5.214e+10,1.053e+12
8192,1.649e+12,1.675e+12,1.675e+12,5.214e+10,1.053e+12
16384,3.299e+12,3.350e+12,3.350e+12,5.214e+10,1.053e+12
32768,6.597e+12,6.700e+12,6.700e+12,5.214e+10,1.053e+12
65536,1.319e+13,1.340e+13,1.340e+13,5.214e+10,1.053e+12""")

def get_better(data, prefix):
    att_as_dict = {t[0]: t[1] for t in data[f'{prefix}_att']}   
    chunk_as_dict = {t[0]: t[1] for t in data[f'{prefix}_chunk']}
    return [(ctx, min(att_as_dict.get(ctx, float('inf')), chunk_as_dict.get(ctx, float('inf')))) 
            for ctx in att_as_dict]

def relative_improvement(data, prefix):
    better = get_better(data, prefix)
    return [(ctx, sdpa / better) for (ctx, sdpa), (_, better) in zip(data['sdpa'], better)]

if __name__ == "__main__":
    # Set style for a modern, professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
        'font.size': 16,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'axes.labelweight': 'bold',
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.formatter.useoffset': False,
    })

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), dpi=300)

    # Get data
    p1_speedup = relative_improvement(TIME, 'p1')
    p2_speedup = relative_improvement(TIME, 'p2')
    p1_theory = relative_improvement(FLOPS, 'p1')
    p2_theory = relative_improvement(FLOPS, 'p2')

    # Plot data for p1
    x1, y1 = zip(*p1_speedup)
    x1t, y1t = zip(*p1_theory)
    ax1.plot(x1, y1, '-', color='black', linewidth=5.5, label='Current performance')
    ax1.plot(x1t, y1t, '-', color='#8aeb9e', linewidth=2, label='Theoretically achievable')

    # Plot data for p2
    x2, y2 = zip(*p2_speedup)
    x2t, y2t = zip(*p2_theory)
    ax2.plot(x2, y2, '-', color='black', linewidth=5.5, label='Current performance')
    ax2.plot(x2t, y2t, '-', color='#8aeb9e', linewidth=2, label='Theoretically achievable')

    # Configure both axes
    for ax, title in [(ax1, 'Degree=1'), (ax2, 'Degree=2')]:
        ax.set_xscale('log')
        ax.set_xlabel('Context length', labelpad=20)
        ax.set_ylabel('Power Attention Speedup (vs Flash)', labelpad=20)
        
        # Format y-axis with integer labels and 'x' suffix
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}x'))
        
        ax.set_yticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_ylim(bottom=-1, top=30)
        
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_xticks([1000, 5000, 10000, 50000])
        ax.xaxis.set_ticklabels(['1k', '5k', '10k', '50k'])
        ax.set_title(title, pad=20)

        ax.legend(frameon=True, 
                 facecolor='white', 
                 edgecolor='none', 
                 framealpha=0.9,
                 loc='upper left',
                 bbox_to_anchor=(0.02, 0.98),
                 fontsize=22,
                 markerscale=3,
                 handlelength=4,
                 handletextpad=0.8)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        ax.grid(True, which='major', linestyle='-', alpha=0.2)
        ax.grid(True, which='minor', linestyle='-', alpha=0.1)

    plt.tight_layout(pad=1.5)

    # Save plot
    plt.savefig('kernel_speed_plot.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

