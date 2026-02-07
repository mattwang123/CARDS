"""
Core analysis logic for confounding factors.
STANDARD MODE: Train on Train Set -> Test on Full Test Set.
"""
import argparse
import re
import sys
import json
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, pointbiserialr

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# STRICT SPACY SETUP
# -----------------------------------------------------------------------------
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("\n[ERROR] spaCy model 'en_core_web_sm' not found.")
        print("To fix, run: python -m spacy download en_core_web_sm\n")
        sys.exit(1)
except ImportError:
    print("\n[ERROR] spaCy library not installed.")
    print("To fix, run: pip install spacy\n")
    sys.exit(1)


def count_numbers_robust(text):
    """Robustly count numbers (Fractions, Currency, Word-form)"""
    text = text.lower()
    text = re.sub(r'\d+\s*/\s*\d+', ' <NUM> ', text) 
    text = re.sub(r'[\$£€]?\d+(?:,\d{3})*(?:\.\d+)?', ' <NUM> ', text)
    words = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen)\b'
    text = re.sub(words, ' <NUM> ', text)
    return text.count('<NUM>')


def count_entities(text):
    """Count Named Entities using spaCy"""
    doc = nlp(text)
    ents = [e for e in doc.ents if e.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART']]
    return len(ents)


def load_data(data_path, embeddings_path, layer):
    """Helper to load data and specific layer embeddings"""
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
    """Extract matrix of [Length, PPL, NumCount, EntCount]"""
    # 1. Load Pre-computed Metrics (PPL, Length)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    lengths = np.array([m['length'] for m in metrics])
    ppls = np.array([m['perplexity'] for m in metrics])
    
    # 2. Compute Text Features (Counts)
    print(f"Computing text features for {len(data)} examples...")
    num_counts = np.array([count_numbers_robust(x['question']) for x in data])
    ent_counts = np.array([count_entities(x['question']) for x in data])
    
    # 3. Stack
    X = np.column_stack((lengths, ppls, num_counts, ent_counts))
    return X


def analyze_confounders(args):
    # ==========================================
    # 1. PREPARE DATA
    # ==========================================
    # Train Data
    train_data, train_labels, train_emb = load_data(args.train_data, args.train_embeddings, args.layer)
    X_train_conf = get_confounder_features(train_data, args.train_metrics)

    # Test Data
    test_data, test_labels, test_emb = load_data(args.test_data, args.test_embeddings, args.layer)
    X_test_conf = get_confounder_features(test_data, args.test_metrics)

    # ==========================================
    # 2. TRAIN LATENT PROBE (Upper Bound)
    # ==========================================
    print("\n" + "="*60)
    print(f"TRAINING LATENT PROBE (Layer {args.layer})")
    print("="*60)
    probe = LogisticRegression(class_weight='balanced', C=args.C, max_iter=1000)
    probe.fit(train_emb, train_labels)
    probe_preds = probe.predict(test_emb)
    probe_f1 = f1_score(test_labels, probe_preds)
    print(f"Latent Probe F1: {probe_f1:.4f}")

    # ==========================================
    # 3. TRAIN CONFOUNDER BASELINES
    # ==========================================
    print("\n" + "="*60)
    print("CONFOUNDER ABLATION STUDY")
    print("Training on Full Train Set -> Testing on Full Test Set")
    print("="*60)
    
    baselines = {
        "Length Only": (0, 1),
        "Perplexity Only": (1, 2),
        "NumCount (Robust)": (2, 3),
        "EntityCount (NER)": (3, 4),
        "Combined (All 4)": (0, 4)
    }
    
    print(f"{'BASELINE MODEL':<20} {'F1 SCORE':<10} {'GAP TO PROBE':<10}")
    print("-" * 45)
    
    for name, (start, end) in baselines.items():
        # Train on TRAIN set
        clf = LogisticRegression(class_weight='balanced', random_state=42)
        clf.fit(X_train_conf[:, start:end], train_labels)
        
        # Test on TEST set
        preds = clf.predict(X_test_conf[:, start:end])
        f1 = f1_score(test_labels, preds)
        
        gap = probe_f1 - f1
        print(f"{name:<20} {f1:.4f}     +{gap:.4f}")

    # ==========================================
    # 4. CORRELATIONS
    # ==========================================
    print("\n" + "="*60)
    print("DIAGNOSTICS: Correlations (Test Set)")
    print("="*60)
    probe_probs = probe.predict_proba(test_emb)[:, 1]
    
    features = {
        "Length": X_test_conf[:, 0],
        "Perplexity": X_test_conf[:, 1],
        "NumCount": X_test_conf[:, 2],
        "EntCount": X_test_conf[:, 3]
    }
    
    print(f"{'FEATURE':<15} {'vs LABEL (r)':<15} {'vs PROBE (r)':<15}")
    print("-" * 50)
    for name, feature in features.items():
        r_label, _ = pointbiserialr(test_labels, feature)
        r_probe, _ = pearsonr(probe_probs, feature)
        print(f"{name:<15} {r_label:<15.3f} {r_probe:<15.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--train_embeddings', required=True)
    parser.add_argument('--train_metrics', required=True, help="Must be generated by extractor.py")
    
    # Test
    parser.add_argument('--test_data', required=True)
    parser.add_argument('--test_embeddings', required=True)
    parser.add_argument('--test_metrics', required=True)
    
    # Config
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--C', type=float, default=1.0)
    
    args = parser.parse_args()
    analyze_confounders(args)