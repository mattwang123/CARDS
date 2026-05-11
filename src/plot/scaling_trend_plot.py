import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit

# --- Nature Style Formatting ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'legend.frameon': False,
    'figure.dpi': 300
})

def get_model_size(model_name):
    """Extract model parameter size from string."""
    match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    if match:
        return float(match.group(1))
    return None

def log_func(x, a, b):
    """Logarithmic scaling law function."""
    return a * np.log10(x) + b

def plot_scaling_dynamics(dataset_name, csv_path):
    print(f"Processing: {dataset_name.upper()}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}. Please check your path.")
        return

    # Extract sizes and filter invalid
    df['Model_Size'] = df['Model'].apply(get_model_size)
    df = df.dropna(subset=['Model_Size'])

    # The baseline is always 0% (Initial clear state)
    df_0 = df[df['Percentage'] == '0%'][['Model', 'Model_Size', 'Separate_Test_F1']]
    target_percentages = ['20%', '40%', '60%', '80%', '100%']
    
    # --- Calculate Global Y-Limits for Best Fit Scale ---
    all_diffs = []
    for pct in target_percentages:
        df_target = df[df['Percentage'] == pct][['Model', 'Model_Size', 'Separate_Test_F1']]
        merged = pd.merge(df_0, df_target, on=['Model', 'Model_Size'], suffixes=('_0', '_target'))
        diffs = (merged['Separate_Test_F1_target'] - merged['Separate_Test_F1_0']).dropna().values
        all_diffs.extend(diffs)
        
    # Ensure the scale encapsulates the data and the zero-line cleanly
    y_min, y_max = min(all_diffs), max(all_diffs)
    y_min_plot, y_max_plot = min(y_min, 0.0), max(y_max, 0.0)
    y_pad = (y_max_plot - y_min_plot) * 0.15  # 15% padding for breathing room
    
    # --- Prepare Subplots ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    ticks = [1.5, 3, 7, 14, 32, 70]
    tick_labels = [f"{t}B" for t in ticks]

    for i, target_pct in enumerate(target_percentages):
        ax = axes[i]
        
        # Merge and calculate difference
        df_target = df[df['Percentage'] == target_pct][['Model', 'Model_Size', 'Separate_Test_F1']]
        merged = pd.merge(df_0, df_target, on=['Model', 'Model_Size'], suffixes=('_0', '_target'))
        merged['F1_Diff'] = merged['Separate_Test_F1_target'] - merged['Separate_Test_F1_0']
        merged = merged.dropna(subset=['Model_Size', 'F1_Diff'])
        
        x = merged['Model_Size'].values
        y = merged['F1_Diff'].values
        
        # Curve fitting
        try:
            popt, _ = curve_fit(log_func, x, y)
            a, b = popt
            x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 100)
            y_fit = log_func(x_fit, a, b)
            
            # Clean mathematical formatting for the legend
            sign = "+" if b > 0 else "-"
            fit_label = f'Fit: {a:.3f} $\cdot$ $\log_{{10}}$(x) {sign} {abs(b):.3f}'
        except Exception:
            x_fit, y_fit, fit_label = [], [], "Fit Failed"
            
        # 1. Plot Baseline Zero-Line (Added to legend)
        ax.axhline(0, color='#999999', linestyle='--', linewidth=1.5, zorder=1, label='0% Baseline')

        # 2. Plot Scatter (No legend label to reduce clutter)
        ax.scatter(x, y, color='#2c7bb6', alpha=0.85, edgecolors='white', s=80, linewidth=0.5, zorder=2)
        
        # 3. Plot Fit Line
        if len(x_fit) > 0:
            ax.plot(x_fit, y_fit, color='#d7191c', linestyle='-', linewidth=2.5, zorder=3, label=fit_label)
        
        # Formatting & Cleanup
        ax.set_title(f'{target_pct} Reasoning Progress', fontweight='bold')
        ax.set_xscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        
        # Apply the pre-calculated dynamic global y-limits
        ax.set_ylim(y_min_plot - y_pad, y_max_plot + y_pad)
        
        if i >= 2: 
            ax.set_xlabel('Model Parameters (Log Scale)')
        if i % 3 == 0: 
            ax.set_ylabel('$\Delta$ F1 (Drop from 0%)')
            
        ax.legend(loc='lower right', handlelength=1.5)

    # Remove the 6th unused subplot
    fig.delaxes(axes[5])

    # Clean and punchy overarching title
    title_context = "Semantic Tasks" if dataset_name == 'umwp' else "Algebraic Tracking"
    plt.suptitle(
        f"Scaling Law of Epistemic Retention: {title_context} ({dataset_name.upper()})", 
        fontsize=16, y=1.02, fontweight='bold'
    )
    
    plt.tight_layout()
    # Changed extension to .png and format to 'png'
    out_file = f'Fig_Nature_Scaling_Law_{dataset_name}.png'
    plt.savefig(out_file, format='png', bbox_inches='tight')
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    # Adjust directory as needed based on execution location
    base_dir = 'experiment_result/exp_temporal_new/results/'
    
    datasets = ['umwp', 'treecut']
    for ds in datasets:
        file_path = os.path.join(base_dir, f'exp10_ultimate_proportional_{ds}.csv')
        if not os.path.exists(file_path):
            file_path = f'exp10_ultimate_proportional_{ds}.csv'
            
        plot_scaling_dynamics(ds, file_path)