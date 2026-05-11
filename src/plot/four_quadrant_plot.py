import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def plot_epistemic_trajectories(target_models=None):
    """
    Plots epistemic trajectories.
    
    Parameters:
    - target_models: 
        - None: Plots the average across all models.
        - str: Plots for a single specific model.
        - list of str: Batch generates a separate plot for EACH model in the list.
    """
    # 1. Ensure the data file exists
    csv_path = '/home/hwang302/.local/nlp/CARDS/experiment_result/exp_temporal_new/results/exp11_average_trajectories.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 2. Load the average trajectories data
    df = pd.read_csv(csv_path)
    available_models = df['Model'].unique()

    # 3. Handle the target_models parameter logic and clean model slugs
    if target_models is None:
        models_to_plot = [None] # None signifies "All Models Average"
    else:
        if isinstance(target_models, str):
            target_models = [target_models]
            
        # Clean input models to match the dataset format (strip "org/" prefix if present)
        cleaned_targets = [m.split('/')[-1] for m in target_models]
            
        # Validate models against the dataset
        models_to_plot = [m for m in cleaned_targets if m in available_models]
        missing_models = [m for m in cleaned_targets if m not in available_models]
        
        if missing_models:
            print(f"Warning: These models were not found in the CSV and will be skipped: {missing_models}")
            
        if not models_to_plot:
            print("Error: No valid models left to plot.")
            print("Available models:", ", ".join(available_models))
            return

    # 4. Convert dataframe from Wide format to Long format for Seaborn plotting
    prob_cols = [f'Prob_{i}%' for i in range(0, 110, 10)]
    id_cols = ['Dataset', 'Model', 'Quadrant', 'Sample_Count']
    df_long = pd.melt(df, id_vars=id_cols, value_vars=prob_cols, 
                      var_name='Percentage_Str', value_name='Probability')

    # 5. Extract numeric percentage values for the X-axis mapping
    df_long['Percentage'] = df_long['Percentage_Str'].apply(lambda x: int(re.search(r'\d+', x).group()))

    # 6. Set global plot styling parameters (academic paper friendly)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 7. Iterate through both datasets and requested models
    datasets = ['umwp', 'treecut']
    for ds in datasets:
        df_ds = df_long[df_long['Dataset'] == ds]
        
        if df_ds.empty:
            print(f"Skipping {ds} - no data found in general.")
            continue

        for current_model in models_to_plot:
            # Filter for specific model if requested
            if current_model is not None:
                df_filtered = df_ds[df_ds['Model'] == current_model]
                title_desc = f"{ds.upper()} - {current_model}"
                safe_model_name = current_model.replace('/', '_')
                filename = f'epistemic_trajectories_{ds}_{safe_model_name}.png'
            else:
                df_filtered = df_ds
                title_desc = f"{ds.upper()} - All Models Average"
                filename = f'epistemic_trajectories_{ds}_all_models.png'
                
            if df_filtered.empty:
                print(f"Skipping {title_desc} - no data found.")
                continue

            plt.figure(figsize=(10, 6))

            # 8. Plot the trajectory lines aggregated by Quadrant
            ax = sns.lineplot(
                data=df_filtered, 
                x='Percentage', 
                y='Probability', 
                hue='Quadrant',
                style='Quadrant',
                markers=['o', 's', '^', 'D'], # Distinct marker for each quadrant
                dashes=False,
                errorbar=None, # Show absolute mean
                linewidth=2.5,
                markersize=8,
                palette="Set1" # High contrast palette
            )

            # 9. Enhance labels, titles, and axis ticks
            plt.title(f'Epistemic Trajectories During Test-Time Compute\n({title_desc})', 
                      fontsize=15, fontweight='bold', pad=15)
            plt.xlabel('Chain-of-Thought Generation Progress (%)', fontsize=14)
            plt.ylabel('Latent Probability P(Insufficient)', fontsize=14)
            plt.xticks(range(0, 110, 10))
            plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            plt.ylim(-0.05, 1.05)
            
            # Move legend outside the main plot area
            plt.legend(title='Epistemic Quadrant', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
            plt.tight_layout()

            # 10. Save the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close() # Close figure to avoid memory leaks
            
            print(f"✅ Saved: {filename}")

if __name__ == '__main__':
    # --- Option 1: Plot average of ALL models ---
    # plot_epistemic_trajectories()
    
    # --- Option 2: Plot a SINGLE model ---
    # plot_epistemic_trajectories('DeepSeek-R1-Distill-Llama-70B')
    
    # --- Option 3: Plot a LIST of specific models ---
    # Note: You can include the huggingface prefix, the script will clean it.
    models_to_compare = [
        'allenai/Olmo-3-7B-Think',
        'google/gemma-3-27b-it',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'Qwen/Qwen2.5-3B'
    ]
    print(f"Generating batch visualizations for {len(models_to_compare)} specific models...")
    plot_epistemic_trajectories(models_to_compare)