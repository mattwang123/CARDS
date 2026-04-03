"""
================================================================================
EXPERIMENT 1: Representation-Language Gap Analysis
================================================================================
This script merges the probing results (Latent awareness) with the verbalization 
results (Verbal awareness) to compute the Representation-Language Gap. 
It categorizes models into Base, Instruct, and Think to directly test the 
"Sycophancy Hypothesis" (how post-training impacts calibration).
"""

import json
import os
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
PROBE_RESULTS_PATH = 'experiments/probing/best_layers_linear.json'
VERBAL_RESULTS_DIR = 'experiments/verbalization/math'
DATASETS = ['umwp', 'treecut']

def get_model_type(model_name):
    """Categorize model based on its name."""
    name_lower = model_name.lower()
    if 'think' in name_lower:
        return 'Think' # For OLMo reasoning models
    elif 'instruct' in name_lower or 'chat' in name_lower:
        return 'Instruct'
    else:
        return 'Base'

def get_model_size_group(model_name):
    """Extract rough parameter count for sorting."""
    name = model_name.lower()
    if '1.5b' in name or '1b' in name: return '1.5B'
    if '3b' in name: return '3B'
    if '7b' in name or '8b' in name: return '7B-8B'
    if '14b' in name: return '14B'
    if '32b' in name: return '32B'
    return 'Other'

def get_base_family(model_name):
    """Extract family name to match Base and Instruct models."""
    return model_name.replace('-Instruct', '').replace('-Think-SFT', '').replace('-Think-DPO', '').replace('-Think', '').split('/')[-1]

def load_verbal_accuracy(model_name, dataset):
    """Load verbalization accuracy from the evaluation JSON."""
    model_slug = model_name.split('/')[-1]
    path = Path(VERBAL_RESULTS_DIR) / model_slug / f"{dataset}_results.json"
    
    if not path.exists():
        return None
        
    with open(path, 'r') as f:
        data = json.load(f)
        return data.get("accuracy", None)

def main():
    if not os.path.exists(PROBE_RESULTS_PATH):
        print(f"Error: Probe results not found at {PROBE_RESULTS_PATH}")
        return

    with open(PROBE_RESULTS_PATH, 'r') as f:
        probe_data = json.load(f)

    records = []

    for model_name, p_data in probe_data.items():
        m_type = get_model_type(model_name)
        m_family = get_base_family(model_name)
        m_size = get_model_size_group(model_name)

        for ds in DATASETS:
            # 1. Get Latent Performance (Train on X, Test on X)
            train_key = f"train_on_{ds}"
            test_key = f"test_on_{ds}"
            
            if train_key not in p_data:
                continue
                
            latent_metrics = p_data[train_key]["results"][test_key]
            latent_acc = latent_metrics["accuracy"]
            latent_f1 = latent_metrics["f1"]
            
            # 2. Get Verbal Performance
            verbal_acc = load_verbal_accuracy(model_name, ds)
            
            if verbal_acc is None:
                # Skip if verbal evaluation hasn't finished for this model/dataset
                continue
                
            # 3. Calculate Gap
            # Note: We use accuracy for the gap to ensure an apples-to-apples metric,
            # though F1 is better for overall latent evaluation due to class imbalances.
            gap = latent_acc - verbal_acc

            records.append({
                "Family": m_family,
                "Model": model_name.split('/')[-1],
                "Size": m_size,
                "Type": m_type,
                "Dataset": ds.upper(),
                "Latent_F1": latent_f1,
                "Latent_Acc": latent_acc,
                "Verbal_Acc": verbal_acc,
                "Gap (Latent-Verbal)": gap
            })

    if not records:
        print("No paired records found. Have you run both generation and evaluation?")
        return

    df = pd.DataFrame(records)
    
    # Sort for logical reading
    df = df.sort_values(by=["Dataset", "Size", "Family", "Type"])

    # Format percentages
    format_cols = ["Latent_F1", "Latent_Acc", "Verbal_Acc", "Gap (Latent-Verbal)"]
    df_display = df.copy()
    for col in format_cols:
        df_display[col] = (df_display[col] * 100).map("{:.1f}%".format)

    # --- OUTPUT 1: Global View ---
    print("\n" + "="*80)
    print("GLOBAL REPRESENTATION-LANGUAGE GAP")
    print("="*80)
    print(df_display.to_string(index=False))

    # --- OUTPUT 2: The Sycophancy Hypothesis (Base vs Instruct) ---
    print("\n" + "="*80)
    print("THE SYCOPHANCY HYPOTHESIS: IMPACT OF POST-TRAINING")
    print("Does alignment training increase the gap?")
    print("="*80)
    
    for ds in df['Dataset'].unique():
        print(f"\n--- DATASET: {ds} ---")
        ds_df = df[df['Dataset'] == ds]
        
        # Pivot to put Base and Instruct side by side
        pivot = ds_df.pivot(index='Family', columns='Type', values=['Verbal_Acc', 'Gap (Latent-Verbal)'])
        
        # Clean up column names for display
        if 'Base' in pivot['Gap (Latent-Verbal)'] and 'Instruct' in pivot['Gap (Latent-Verbal)']:
            base_gaps = pivot['Gap (Latent-Verbal)']['Base']
            inst_gaps = pivot['Gap (Latent-Verbal)']['Instruct']
            
            comparison_df = pd.DataFrame({
                'Base Verbal': pivot['Verbal_Acc']['Base'],
                'Instruct Verbal': pivot['Verbal_Acc']['Instruct'],
                'Base Gap': base_gaps,
                'Instruct Gap': inst_gaps,
                'Alignment Tax (Gap Increase)': inst_gaps - base_gaps
            }).dropna()
            
            # Format as percentages
            for col in comparison_df.columns:
                comparison_df[col] = (comparison_df[col] * 100).map("{:+.1f}%".format)
                
            print(comparison_df.to_string())
        else:
            print("  Not enough Base/Instruct pairs to generate comparison for this dataset yet.")

    # Save to CSV
    os.makedirs("experiments/analysis", exist_ok=True)
    df.to_csv("experiments/analysis/gap_analysis.csv", index=False)
    print(f"\n✓ Saved raw data to experiments/analysis/gap_analysis.csv")

if __name__ == "__main__":
    main()