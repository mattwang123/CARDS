import os
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

def create_nature_plots(csv_path="results/exp3_master_summary.csv", dataset="umwp"):
    output_dir = "paper_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df['Dataset'] == dataset]
    timesteps = sorted(df['Timestep'].unique())
    
    # -------------------------------------------------------------------------
    # FIGURE A: The Measurement Flaw (Global vs Active)
    # -------------------------------------------------------------------------
    flagship_model = "Qwen2.5-72B-Instruct"
    df_flag = df[df['Model'] == flagship_model].sort_values('Timestep')
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(df_flag['Timestep'], df_flag['Global_Test_F1'], 
            marker='o', linestyle='--', color='#999999', linewidth=2.0, label='Standard Accuracy (Includes Terminated)')
    
    ax.plot(df_flag['Timestep'], df_flag['Active_Only_F1'], 
            marker='o', linestyle='-', color='#d62728', linewidth=2.5, label='Active-Only Accuracy (True Trend)')
    
    eos_val = df_flag['EOS_F1'].iloc[0]
    ax.axhline(y=eos_val, color='#333333', linestyle=':', linewidth=2.0, label='Final Answer Accuracy (EOS)')
    
    ax.set_xscale('symlog', base=2)
    ax.set_xticks(timesteps)
    ax.set_xticklabels(timesteps)
    
    ax.set_xlabel('Tokens Generated')
    ax.set_ylabel('Model Self-Awareness (F1 Score)')
    ax.set_title(f'Correcting the Measurement Artifact\n({flagship_model})', pad=15)
    
    ax.legend(frameon=False, loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"FigA_Measurement_Correction_{dataset}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE B: The Universal Trend (Colorful Lines + Bold Average)
    # -------------------------------------------------------------------------
    # Make the figure slightly wider to accommodate the external legend
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['Model'].unique()
    
    # Generate a visually distinct color palette for the 20 models
    colors = plt.get_cmap('tab20', len(models))
    
    # Plot each individual model
    for idx, model in enumerate(models):
        df_m = df[df['Model'] == model].sort_values('Timestep')
        df_m = df_m.dropna(subset=['Active_Only_F1']) 
        
        ax.plot(df_m['Timestep'], df_m['Active_Only_F1'], 
                color=colors(idx), alpha=0.7, linewidth=1.5, marker='.', markersize=6, label=model)
                
    # Calculate the global average
    avg_df = df.groupby('Timestep')['Active_Only_F1'].mean().reset_index()
    
    # Plot the Average Line (Thick, Solid Black, High Z-Order to sit on top)
    ax.plot(avg_df['Timestep'], avg_df['Active_Only_F1'], 
            marker='D', color='#000000', linewidth=4.0, markersize=8, 
            zorder=10, label='AVERAGE TREND')

    ax.set_xscale('symlog', base=2)
    ax.set_xticks(timesteps)
    ax.set_xticklabels(timesteps)
    
    ax.set_xlabel('Tokens Generated')
    ax.set_ylabel('Model Self-Awareness (F1 Score)')
    ax.set_title('Decline of Model Awareness During Reasoning', pad=15)
    
    # Place the legend OUTSIDE the plot to the right, so it doesn't cover the data
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"FigB_Universal_Trend_{dataset}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Saved Nature-style plots to ./{output_dir}/ directory for {dataset.upper()}")

if __name__ == '__main__':
    base_csv = "/export/fs06/hwang302/CARDS/exp_temporal_new/results/exp3_master_summary.csv"
    create_nature_plots(csv_path=base_csv, dataset="umwp")
    create_nature_plots(csv_path=base_csv, dataset="treecut")