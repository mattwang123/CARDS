"""
VISUALIZATION: Confounder Analysis
----------------------------------
Generates plots to visualize the relationship between confounding features 
(Length, PPL, Counts) and the Probe's predictions.

1. Box Plots: Feature distribution by Class (Sufficient vs Insufficient)
2. Scatter Plots: Feature value vs Probe Confidence (Probability)
"""
import argparse
import os
import sys
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr

# Setup Plotting Style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# -----------------------------------------------------------------------------
# REUSED LOGIC (From analyze_confounders.py)
# -----------------------------------------------------------------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        sys.exit("[ERROR] spaCy model 'en_core_web_sm' not found.")
except ImportError:
    sys.exit("[ERROR] spaCy not installed.")

def count_numbers_robust(text):
    text = text.lower()
    text = re.sub(r'\d+\s*/\s*\d+', ' <NUM> ', text) 
    text = re.sub(r'[\$£€]?\d+(?:,\d{3})*(?:\.\d+)?', ' <NUM> ', text)
    words = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen)\b'
    text = re.sub(words, ' <NUM> ', text)
    return text.count('<NUM>')

def count_entities(text):
    doc = nlp(text)
    ents = [e for e in doc.ents if e.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']]
    return len(ents)

def load_data(data_path, embeddings_path, layer):
    print(f"Loading data: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    labels = np.array([1 if x['is_sufficient'] else 0 for x in data])
    
    print(f"Loading embeddings: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    if len(embeddings.shape) == 3:
        embeddings = embeddings[:, layer, :]
        
    return data, labels, embeddings

def get_confounder_features(data, metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    lengths = np.array([m['length'] for m in metrics])
    ppls = np.array([m['perplexity'] for m in metrics])
    print(f"Computing text features for {len(data)} examples...")
    num_counts = np.array([count_numbers_robust(x['question']) for x in data])
    ent_counts = np.array([count_entities(x['question']) for x in data])
    return np.column_stack((lengths, ppls, num_counts, ent_counts)), ["Length", "Perplexity", "NumCount", "EntCount"]

# -----------------------------------------------------------------------------
# PLOTTING LOGIC
# -----------------------------------------------------------------------------
def visualize(args):
    # 1. Load Train (to train probe)
    train_data, train_y, train_emb = load_data(args.train_data, args.train_embeddings, args.layer)
    
    # 2. Load Test (to visualize)
    test_data, test_y, test_emb = load_data(args.test_data, args.test_embeddings, args.layer)
    X_test_conf, feature_names = get_confounder_features(test_data, args.test_metrics)
    
    # 3. Train Probe
    print("Training Probe...")
    probe = LogisticRegression(class_weight='balanced', C=args.C, max_iter=1000)
    probe.fit(train_emb, train_y)
    
    # Get Probabilities (Confidence)
    test_probs = probe.predict_proba(test_emb)[:, 1]
    
    # 4. Generate Plots
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot A: Feature Distributions by Class (Box Plots)
    # Checks: "Are insufficient questions just shorter?"
    print("Generating Distribution Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        sns.boxplot(x=test_y, y=X_test_conf[:, i], ax=axes[i], palette="Set2")
        axes[i].set_xticklabels(['Insufficient', 'Sufficient'])
        axes[i].set_title(f"{name} Distribution")
        axes[i].set_ylabel(name)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confounder_distributions.png"))
    plt.close()
    
    # Plot B: Probe Bias (Scatter Plots)
    # Checks: "Does confidence increase linearly with this feature?"
    print("Generating Bias Scatter Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(feature_names):
        # Calculate correlation for title
        r, _ = pearsonr(test_probs, X_test_conf[:, i])
        
        # Scatter with Hue
        sns.scatterplot(
            x=X_test_conf[:, i], 
            y=test_probs, 
            hue=test_y, 
            palette={0: 'red', 1: 'green'},
            alpha=0.6,
            ax=axes[i]
        )
        
        # Add regression line to show trend
        sns.regplot(
            x=X_test_conf[:, i], 
            y=test_probs, 
            scatter=False, 
            color='black', 
            line_kws={'linestyle':'--'},
            ax=axes[i]
        )
        
        axes[i].set_title(f"{name} vs. Probe Confidence (r={r:.2f})")
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Probe Probability (Sufficient)")
        axes[i].legend(title='Ground Truth', labels=['Insufficient', 'Sufficient'])
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confounder_bias_scatter.png"))
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--train_embeddings', required=True)
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--test_embeddings', required=True)
    parser.add_argument('--test_metrics', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--C', type=float, default=1.0)
    
    args = parser.parse_args()
    visualize(args)