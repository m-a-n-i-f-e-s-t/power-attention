import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Hardcode data from experiments
SDPA_DATA = [(0,  11.077120780944824), (1168.1999999999998,  6.161754608154297), (1932.84,  5.4999284744262695), (2846.16,  5.177915096282959), (3950.64,  4.807796955108643), (5267.5199999999995,  4.48173713684082), (6855.209999999999,  4.224358081817627), (8756.189999999999,  3.884178400039673), (11039.49,  3.5315630435943604), (13779.449999999999,  3.1201751232147217), (17071.649999999998,  3.0646958351135254), (21016.98,  2.9195477962493896), (25748.19,  2.815972089767456), (31429.89,  2.6381447315216064), (38247.93,  2.54599666595459), (46430.64,  2.405158758163452), (56248.829999999994,  2.474220037460327), (68026.40999999999,  2.3414483070373535), (82161.62999999999,  2.2270126342773438), (99127.07999999999,  2.2719244956970215), (119485.62,  2.2093939781188965), (143916.93,  2.137006998062134), (173228.12999999998,  2.08113956451416), (208406.87999999998,  2.079277753829956), (250621.37999999998,  2.022472620010376), (301273.47,  1.9832582473754883), (362062.35,  2.095027208328247), (435005.81999999995,  1.9579265117645264), (522535.86,  2.083192825317383), (627578.2799999999,  1.986913800239563), (753621.75,  1.9355891942977905), (904877.1,  1.9600214958190918)]
P1_DATA = [(0,  11.11016845703125), (399.3,  6.028470516204834), (660.66,  5.356439590454102), (972.8399999999999,  5.023510932922363), (1350.36,  4.701613426208496), (1800.48,  4.434966564178467), (2343.165,  4.238950252532959), (2992.935,  4.0261454582214355), (3773.3849999999998,  3.7999067306518555), (4709.925,  3.6521005630493164), (5835.224999999999,  3.3549726009368896), (7183.7699999999995,  3.1061220169067383), (8800.935,  3.055795192718506), (10742.985,  2.9036874771118164), (13073.445,  2.9006383419036865), (15870.359999999999,  2.8424882888793945), (19226.295,  2.7522618770599365), (23251.965,  2.719210624694824), (28083.495,  2.4548439979553223), (33882.42,  2.5807619094848633), (40841.13,  2.4127728939056396), (49191.945,  2.465137004852295), (59210.744999999995,  2.4511187076568604), (71235.12,  2.4021670818328857)]
P2_DATA = [(0,  11.18627643585205), (518.1,  6.209327220916748), (857.22,  5.513784408569336), (1262.28,  5.268224716186523), (1752.12,  4.939364433288574), (2336.16,  4.651852607727051), (3040.305,  4.445021152496338), (3883.395,  4.205632209777832), (4896.045,  3.981386661529541), (6111.225,  3.779463529586792), (7571.325,  3.4195854663848877), (9321.09,  3.1018481254577637), (11419.395,  3.003368854522705), (13939.245,  2.8134326934814453), (16963.065,  2.794123888015747), (20592.12,  2.7038397789001465), (24946.515,  2.592689275741577), (30169.905,  2.5497069358825684), (36438.915,  2.2575321197509766), (43963.14,  2.372074604034424), (52992.21,  2.1967389583587646), (63827.565,  2.2653846740722656), (76827.165,  2.243931770324707), (92429.04,  2.1789915561676025)]

# Convert data to numpy arrays and transform x values to hours
def process_data(data):
    x, y = zip(*data)
    x = np.array(x) / 3600  # Convert seconds to hours
    y = np.array(y)
    return x, y

if __name__ == "__main__":
    # Set style for a modern, professional look
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],  # Modern, clean fonts
        'font.size': 16,                    # Increased base font size
        'axes.labelsize': 24,               # Much larger axis labels
        'axes.titlesize': 24,
        'axes.labelweight': 'bold',
        'grid.alpha': 0.3,
        'grid.color': '#cccccc',
        'xtick.labelsize': 20,              # Much larger tick labels
        'ytick.labelsize': 20,
        'axes.formatter.useoffset': False,  # This prevents offset notation
    })

    # Larger figure size
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Process and plot data
    sdpa_x, sdpa_y = process_data(SDPA_DATA)
    p1_x, p1_y = process_data(P1_DATA)
    p2_x, p2_y = process_data(P2_DATA)

    # Plot with thicker lines
    ax.plot(sdpa_x, sdpa_y, color='#E65D2F', linewidth=5.5, label='SDPA (flash attention)')  # Warm orange
    ax.plot(p1_x, p1_y, color='#89C2F5', linewidth=5.5, label='Power attention, degree 1')   # Lighter blue
    ax.plot(p2_x, p2_y, color='#2F7AB9', linewidth=5.5, label='Power attention, degree 2')   # Darker blue

    # Set scales to logarithmic
    ax.set_xscale('log')

    # Set x-axis limits (30 minutes = 30/60 = 0.5 hours)
    ax.set_xlim(0.5, max(max(sdpa_x), max(p1_x), max(p2_x)))

    # Format x-axis to show whole numbers only
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(round(x))}'))
    
    # Automatically set y-axis limits based on data after 30 minutes
    mask = sdpa_x >= 0.5
    y_min = min(
        min(sdpa_y[mask]),
        min(p1_y[p1_x >= 0.5]),
        min(p2_y[p2_x >= 0.5])
    )
    y_max = max(
        max(sdpa_y[mask]),
        max(p1_y[p1_x >= 0.5]),
        max(p2_y[p2_x >= 0.5])
    )
    ax.set_ylim(y_min * 0.95, y_max * 1.05)

    # Customize axis labels with more padding
    ax.set_xlabel('FLOP-estimated A100 hours', labelpad=20)  # More padding
    ax.set_ylabel('Loss', labelpad=20)

    # Extra large legend
    ax.legend(frameon=True, 
             facecolor='white', 
             edgecolor='none', 
             framealpha=0.9,
             loc='upper right',
             bbox_to_anchor=(0.98, 0.98),
             fontsize=22,          # Even larger legend font
             markerscale=3,
             handlelength=4,
             handletextpad=0.8)

    # Adjust layout and style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_linewidth(1.5)    # Thicker spines
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add subtle grid for readability
    ax.grid(True, which='major', linestyle='-', alpha=0.2)
    ax.grid(True, which='minor', linestyle='-', alpha=0.1)

    # Adjust margins and layout
    plt.tight_layout(pad=1.5)  # More padding around the plot

    # Save with high quality
    plt.savefig('xl_performance_plot.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

