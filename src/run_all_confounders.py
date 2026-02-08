"""
AUTO-RUNNER: Comprehensive Confounder Analysis
----------------------------------------------
1. Checks for artifacts (embeddings/metrics); auto-extracts if missing.
2. Runs the Confounder Matrix:
   - Trains Probe on Dataset X (and ALL).
   - Tests on Dataset Y.
   - Compares vs. "Smart Baseline" (Length + PPL + Counts).
3. Prints Correlations to diagnose transfer behavior.
"""
import os
import sys
import json
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, pointbiserialr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.extractor import HiddenStateExtractor
# Reuse logic from your analysis script to ensure consistency
from scripts.analyze_confounders import load_data, get_confounder_features

# --- CONFIGURATION (Based on your JSON results) ---
# We use the best layer from 'train_on_gsm8k' as the fixed representation layer.
MODEL_CONFIGS = {
    'qwen2.5-math-1.5b':     {'layer': 18},
    # 'qwen2.5-1.5b':          {'layer': 18},
    # 'qwen2.5-math-7b':       {'layer': 21}, 
    # 'llama-3.2-3b-instruct': {'layer': 20}, 
}

DATASET_PATHS = {
    'gsm8k': 'src/data/processed/gsm8k/gsm8k_{split}.json',
    'treecut': 'src/data/processed/treecut/treecut_{split}.json',
    'umwp': 'src/data/processed/insufficient_dataset_umwp/umwp_{split}.json'
}

def get_paths(dataset, split, model_name):
    """Generate standardized paths for artifacts"""
    data_path = DATASET_PATHS[dataset].format(split=split)
    base_name = os.path.basename(data_path).replace('.json', '')
    
    # Standard naming convention
    emb_path = f"src/data/embeddings/{base_name}_{model_name}_last_token.npy"
    met_path = f"src/data/embeddings/{base_name}_{model_name}_last_token_metrics.json"
    
    return data_path, emb_path, met_path

def ensure_artifacts(model_name, device):
    """Check availability of embeddings/metrics; extract if missing."""
    print(f"\n[1] Checking Artifacts for {model_name}...")
    extractor = None 
    
    for ds in DATASET_PATHS:
        for split in ['train', 'test']:
            d_p, e_p, m_p = get_paths(ds, split, model_name)
            
            if not os.path.exists(e_p) or not os.path.exists(m_p):
                print(f"   MISSING: {ds}/{split} -> Extracting...")
                if extractor is None:
                    extractor = HiddenStateExtractor(model_name, device=device)
                
                with open(d_p, 'r') as f:
                    data = json.load(f)
                
                # Extract and Save
                emb, met = extractor.extract_dataset(data, pooling='last_token')
                os.makedirs(os.path.dirname(e_p), exist_ok=True)
                np.save(e_p, emb)
                with open(m_p, 'w') as f:
                    json.dump(met, f, indent=2)
            else:
                print(f"   FOUND:   {ds}/{split}")

def print_correlations(labels, features, probe_probs, dataset_name):
    """Print diagnostic correlations: Feature vs Label and Feature vs Probe"""
    feat_names = ["Length", "PPL", "NumCount", "EntCount"]
    print(f"\n   [Diagnostics: {dataset_name.upper()} Correlations]")
    print(f"   {'FEATURE':<12} {'r(GT Label)':<12} {'r(Probe)':<12}")
    print("   " + "-" * 40)
    
    for i, name in enumerate(feat_names):
        feat_vals = features[:, i]
        # Correlation with Ground Truth (is the dataset biased?)
        r_label, _ = pointbiserialr(labels, feat_vals)
        # Correlation with Probe Confidence (is the probe using this feature?)
        r_probe, _ = pearsonr(probe_probs, feat_vals)
        
        # Highlight strong correlations
        mark = "*" if abs(r_label) > 0.3 else ""
        print(f"   {name:<12} {r_label:<12.3f}{mark} {r_probe:<12.3f}")

def run_analysis_matrix(model_name, layer):
    """Run the Train-on-X -> Test-on-Y matrix experiment"""
    datasets = list(DATASET_PATHS.keys())
    
    # 1. Load Everything to Memory
    print(f"\n[2] Loading Datasets for {model_name} (Layer {layer})...")
    memory = {}
    for ds in datasets:
        for split in ['train', 'test']:
            d_p, e_p, m_p = get_paths(ds, split, model_name)
            data, labels, emb = load_data(d_p, e_p, layer)
            features = get_confounder_features(data, m_p)
            
            memory[f"{ds}_{split}"] = {'emb': emb, 'y': labels, 'feat': features}

    # 2. Create "ALL" Training Set (Combine everything)
    all_emb = np.concatenate([memory[f"{ds}_train"]['emb'] for ds in datasets], axis=0)
    all_y = np.concatenate([memory[f"{ds}_train"]['y'] for ds in datasets], axis=0)
    all_feat = np.concatenate([memory[f"{ds}_train"]['feat'] for ds in datasets], axis=0)
    memory['all_train'] = {'emb': all_emb, 'y': all_y, 'feat': all_feat}

    # 3. Run Matrix
    print(f"\n[3] Results for {model_name}")
    print(f"{'TRAIN SET':<10} {'TEST SET':<10} | {'PROBE F1':<10} {'CONF. F1':<10} | {'GAP':<10} | {'VERDICT'}")
    print("-" * 85)
    
    results_log = []
    train_configs = datasets + ['all']

    for train_name in train_configs:
        tr = memory[f"{train_name}_train"]
        
        # A. Train Latent Probe (The "Hero")
        probe = LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)
        probe.fit(tr['emb'], tr['y'])
        
        # B. Train Confounder Baseline (The "Villain")
        base = LogisticRegression(class_weight='balanced', random_state=42)
        base.fit(tr['feat'], tr['y'])
        
        if train_name != 'all':
            print(f"\n--- Training on {train_name.upper()} ---")

        for test_name in datasets:
            te = memory[f"{test_name}_test"]
            
            # Predict
            probe_probs = probe.predict_proba(te['emb'])[:, 1]
            probe_preds = probe.predict(te['emb'])
            base_preds = base.predict(te['feat'])
            
            # Scores
            p_f1 = f1_score(te['y'], probe_preds)
            c_f1 = f1_score(te['y'], base_preds)
            gap = p_f1 - c_f1
            
            # Verdict
            if gap > 0.20: verdict = "DOMINANT"
            elif gap > 0.10: verdict = "STRONG"
            elif gap > 0.05: verdict = "WEAK"
            else: verdict = "FAIL"
            
            print(f"{train_name.upper():<10} {test_name.upper():<10} | {p_f1:.4f}      {c_f1:.4f}      | {gap:+.4f}      | {verdict}")
            
            # Print Correlations for Diagnostics
            print_correlations(te['y'], te['feat'], probe_probs, test_name)

            results_log.append({
                'train': train_name, 'test': test_name,
                'probe_f1': p_f1, 'conf_f1': c_f1, 'gap': gap
            })
    
    return results_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    full_report = {}

    for model, config in MODEL_CONFIGS.items():
        print(f"\n{'='*60}\nMODEL: {model}\n{'='*60}")
        try:
            ensure_artifacts(model, args.device)
            full_report[model] = run_analysis_matrix(model, config['layer'])
        except Exception as e:
            print(f"\n[ERROR] Failed on {model}: {e}")
            import traceback
            traceback.print_exc()

    # Save
    with open('comprehensive_confounder_results.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    print("\nSaved full results to comprehensive_confounder_results.json")

if __name__ == '__main__':
    main()