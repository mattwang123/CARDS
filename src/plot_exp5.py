import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# =============================================================================
# NATURE-STYLE PLOT CONFIGURATION
# =============================================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,   
    'axes.spines.right': False, 
    'figure.dpi': 300,          
    'axes.edgecolor': '#333333',
    'text.color': '#333333',
    'ytick.color': '#333333',
    'xtick.color': '#333333'
})

def parse_exp5_json_to_df(json_path, dataset):
    """Flattens the Exp5 nested JSON into a Pandas DataFrame for easy plotting."""
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}")
        return None
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    rows = []
    for model_full, metrics in data.items():
        model_slug = model_full.split('/')[-1]
        
        # Grab the EOS ceiling. Prefers the unified_f1_eos (added in Exp 6), falls back to eos_f1
        eos_ceiling = metrics.get('unified_f1_eos', metrics.get('eos_f1', None))
        
        for key, val in metrics.items():
            if key.startswith('unified_f1_t') and key != 'unified_f1_t_eos':
                t_str = key.replace('unified_f1_t', '')
                if t_str.isdigit():
                    rows.append({
                        'Dataset': dataset,
                        'Model': model_slug,
                        'Timestep': int(t_str),
                        'Unified_F1': val,
                        'EOS_Ceiling': eos_ceiling
                    })
    return pd.DataFrame(rows)

def create_nature_plots_exp5(json_path, dataset):
    output_dir = "paper_plots/exp5_unified"
    os.makedirs(output_dir, exist_ok=True)
    
    df = parse_exp5_json_to_df(json_path, dataset)
    if df is None or df.empty:
        print(f"No valid data found for {dataset}.")
        return
        
    timesteps = sorted(df['Timestep'].unique())
    
    # -------------------------------------------------------------------------
    # FIGURE A: The Geometric Stationarity Proof (Flagship Model)
    # -------------------------------------------------------------------------
    flagship_model = "Qwen2.5-72B-Instruct"
    df_flag = df[df['Model'] == flagship_model].sort_values('Timestep')
    
    if not df_flag.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Plot Unified F1 Trend (The Wandering)
        ax.plot(df_flag['Timestep'], df_flag['Unified_F1'], 
                marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, label='Unified Probe F1 (Active Generation)')
        
        # EOS Ceiling (The Stationarity Anchor)
        eos_val = df_flag['EOS_Ceiling'].iloc[0]
        if pd.notna(eos_val):
            ax.axhline(y=eos_val, color='#d62728', linestyle=':', linewidth=2.5, label='Unified Probe at EOS (Global Aggregation)')
        
        ax.set_xscale('symlog', base=2)
        ax.set_xticks(timesteps)
        ax.set_xticklabels(timesteps)
        
        ax.set_xlabel('Tokens Generated')
        ax.set_ylabel('Unified Probe F1 Score')
        ax.set_title(f'Geometric Stationarity vs. Generative Momentum\n({flagship_model})', pad=15)
        
        ax.legend(frameon=False, loc='lower left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"FigA_Geometric_Stationarity_{dataset}.pdf"), format='pdf', bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # FIGURE B: The Universal Trend (Colorful Lines + Bold Average)
    # -------------------------------------------------------------------------
    # Make the figure slightly wider to accommodate the external legend
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['Model'].unique()
    
    # Generate a visually distinct color palette for the models
    colors = plt.get_cmap('tab20', len(models))
    
    # Plot each individual model
    for idx, model in enumerate(models):
        df_m = df[df['Model'] == model].sort_values('Timestep')
        df_m = df_m.dropna(subset=['Unified_F1']) 
        
        ax.plot(df_m['Timestep'], df_m['Unified_F1'], 
                color=colors(idx), alpha=0.7, linewidth=1.5, marker='.', markersize=6, label=model)
                
    # Calculate the global average
    avg_df = df.groupby('Timestep')['Unified_F1'].mean().reset_index()
    
    # Plot the Average Line (Thick, Solid Black, High Z-Order to sit on top)
    ax.plot(avg_df['Timestep'], avg_df['Unified_F1'], 
            marker='D', color='#000000', linewidth=4.0, markersize=8, 
            zorder=10, label='AVERAGE TREND')

    ax.set_xscale('symlog', base=2)
    ax.set_xticks(timesteps)
    ax.set_xticklabels(timesteps)
    
    ax.set_xlabel('Tokens Generated')
    ax.set_ylabel('Unified Probe F1 Score')
    ax.set_title('Universal Latent Wandering (Single Frozen Hyperplane)', pad=15)
    
    # Place the legend OUTSIDE the plot to the right, so it doesn't cover the data
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"FigB_Unified_Universal_Trend_{dataset}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Saved Exp 5 Nature-style plots to ./{output_dir}/ directory for {dataset.upper()}")

if __name__ == '__main__':
    # Ensure this points to the directory where your exp5 json files are saved
    base_dir = "/export/fs06/hwang302/CARDS/exp_temporal_new/results"
    
    umwp_path = os.path.join(base_dir, "exp5_global_dynamics_umwp.json")
    create_nature_plots_exp5(json_path=umwp_path, dataset="umwp")
    
    treecut_path = os.path.join(base_dir, "exp5_global_dynamics_treecut.json")
    create_nature_plots_exp5(json_path=treecut_path, dataset="treecut")