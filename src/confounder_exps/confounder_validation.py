"""
================================================================================
EXP B: Confounder Validation (Solving Prompt + t=0 + Unified Probe)
================================================================================

WHAT THIS SCRIPT DEFENDS AGAINST:
  Reviewer concern #1: "Your t=0 probe F1 is high because the probe learned
  surface features (length, lexical cues, distributional artifacts), not
  actual logical insufficiency."

  We do NOT defend U-shape or Q1-Q4 trajectories here. Those have separate
  defenses (see notes below).

THREE DEFENSES, ALL AT t=0 ON SOLVING PROMPT:
  D1 [CRITICAL] Minimal Contrastive Pairs
      Same length, same scenario, same lexical content; differs by 1-3 words.
      A probe relying on surface features cannot tell them apart.

  D2 [STRONG] Adversarial Lexical Injection
      Inject "insufficient cue words" (some/assume/several) into solvable
      questions. A keyword-matching probe will flip; a logical probe won't.

  D3 [LOAD-BEARING] Residual Probe After Orthogonalization
      Project out length/numcount/complexity/lexical confounder directions
      from the hidden state. Train fresh probe on residual. If F1 survives,
      the insufficiency signal is geometrically distinct from confounders.

WHAT WE DO NOT DEFEND HERE (handled elsewhere):
  - U-shape: defended by exp10 data itself. Separate probe (re-trained at
    each timestep) also shows U-shape, with depth comparable to unified
    probe (UMWP: 0.149 vs 0.127; Treecut: 0.243 vs 0.167). This proves
    U-shape is NOT a unified-probe artifact -- the representation is
    genuinely less linearly separable in the middle of generation.
  - Q1-Q4 trajectories: defended by the cross-dataset transfer experiment
    (separate script).

WHY SOLVING PROMPT + t=0 + UNIFIED PROBE:
  - Solving prompt: matches what the unified probe was trained on (exp2 CoT
    generation context). Verbalization prompt would mismatch the probe's
    learned representation space.
  - t=0: hidden state at the final prompt token, BEFORE any generation.
    Any insufficiency signal here can only come from understanding the
    question -- no generation artifacts can confound the result.
  - Unified probe: same probe used for trajectory analysis. If it passes
    these defenses at t=0, the entire downstream trajectory analysis
    inherits the validity.

OUTPUTS (under exp_confounder/results/):
  defense_1_minimal_pairs_{dataset}.csv      per-model pairwise metrics
  defense_2_lexical_injection_{dataset}.csv  per-model FP rates
  defense_3_residual_probe_{dataset}.csv     per-model F1 drop
  master_verdict.csv                         combined PASS/WARN/FAIL

USAGE:
  python exp_confounder_B_validate.py --dataset both --defenses 1 2 3
  python exp_confounder_B_validate.py --aggregate_only
================================================================================
"""

import argparse
import json
import os
import sys

# Add parent directory to path BEFORE the import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import gc
import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

# CRITICAL: use exp2 SOLVING prompt, not exp1 verbalization prompt.
# The unified probe was trained on hidden states under solving prompts.
from exp2_generate_cot_test import format_cot_prompt as format_prompt

# ---- CONFIG ----
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct'
]

DATASETS = {
    'umwp': {
        'train': 'src/data/processed/insufficient_dataset_umwp/umwp_train.json',
        'test':  'src/data/processed/insufficient_dataset_umwp/umwp_test.json'
    },
    'treecut': {
        'train': 'src/data/processed/treecut/treecut_train.json',
        'test':  'src/data/processed/treecut/treecut_test.json'
    }
}

EXPORT_BASE = '/export/fs06/hwang302/CARDS'
PAIRS_DIR = 'src/data/processed/minimal_pairs'
EXP_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')
OUT_DIR = os.path.join(EXPORT_BASE, 'exp_confounder', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

CONFOUNDER_SEED = ["some", "several", "few", "many", "approximately", "around",
                   "unknown", "unspecified", "various", "assume", "suppose"]


# ============================================================================
# Shared helpers
# ============================================================================

def load_unified_probe(model_slug, dataset):
    csv = os.path.join(EXP_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(csv): return None, None
    df = pd.read_csv(csv)
    rows = df[df['Model'] == model_slug]
    if rows.empty: return None, None
    layer = int(rows['Optimal_Layer'].iloc[0])
    probe_path = os.path.join(EXP_DIR, 'probes_proportional', dataset, model_slug,
                              f"unified_probe_layer{layer}.joblib")
    return (layer, joblib.load(probe_path)) if os.path.exists(probe_path) else (None, None)


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    n_gpu = torch.cuda.device_count()
    mem = {0: "65GB"}
    for i in range(1, n_gpu): mem[i] = "78GB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", max_memory=mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.eval()
    return model, tok


def extract_t0(model, tok, questions, model_name, layer):
    """
    Extract hidden state at the final prompt token (t=0) under the SOLVING prompt.
    This matches the context the unified probe was trained on in exp10.
    """
    out = []
    for q in tqdm(questions, desc="t=0 extraction (solving prompt)", leave=False):
        prompt = format_prompt(q, model_name)   # solving prompt from exp2
        inp = tok(prompt, return_tensors="pt").to(model.device)
        idx = inp['input_ids'].shape[1] - 1     # last token of prompt = t=0
        with torch.no_grad():
            h = model(**inp, output_hidden_states=True).hidden_states[layer]
            out.append(h[0, idx, :].to(torch.float32).cpu().numpy())
            torch.cuda.empty_cache()
    return np.array(out)


def fit_probe(X, y):
    if len(np.unique(y)) < 2: return None
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight='balanced',
                                         C=1.0, solver='lbfgs', n_jobs=-1))
    p.fit(X, y)
    return p


def effective_weights(probe):
    s, c = probe.named_steps['standardscaler'], probe.named_steps['logisticregression']
    return c.coef_[0] / s.scale_


def append_csv(path, row):
    rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
    rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def already_done(path, model_slug):
    if not os.path.exists(path): return False
    return model_slug in pd.read_csv(path)['Model'].values


# ============================================================================
# DEFENSE 1: Minimal Contrastive Pairs
# ============================================================================

def defense_1(model, tok, model_name, model_slug, layer, probe, dataset):
    """
    For each (q+, q-) verified pair from EXP A, extract t=0 hidden state under
    the SOLVING prompt and check P(insuff | q-) > P(insuff | q+).
    The solving prompt scaffolding is identical for q+ and q-, so any signal
    must come from the question content.
    """
    pairs_path = os.path.join(PAIRS_DIR, f"verified_pairs_{dataset}.json")
    pairs = [p for p in json.load(open(pairs_path)) if p.get('verified')]
    if not pairs:
        print(f"   [D1] No verified pairs for {dataset}, skipping.")
        return None

    qp = [p['q_plus']  for p in pairs]
    qm = [p['q_minus'] for p in pairs]
    eds = np.array([p['edit_distance'] for p in pairs])

    Xp = extract_t0(model, tok, qp, model_name, layer)
    Xm = extract_t0(model, tok, qm, model_name, layer)

    pp = probe.predict_proba(Xp)[:, 1]
    pm = probe.predict_proba(Xm)[:, 1]
    gaps = pm - pp

    return {
        'Dataset': dataset, 'Model': model_slug, 'Layer': layer, 'N_Pairs': len(pairs),
        'Pairwise_Acc':   round(float(np.mean(gaps > 0)), 4),
        'Strict_Acc':     round(float(np.mean((pm > 0.5) & (pp < 0.5))), 4),
        'Mean_P_plus':    round(float(np.mean(pp)), 4),
        'Mean_P_minus':   round(float(np.mean(pm)), 4),
        'Mean_Gap':       round(float(np.mean(gaps)), 4),
        'Median_Gap':     round(float(np.median(gaps)), 4),
        # Ideally near 0: small edits should produce as much gap as large ones.
        # High correlation = probe relies on lexical token count, not logic.
        'EditDist_Gap_r': round(float(np.corrcoef(eds, gaps)[0, 1]), 4) if eds.std() > 0 else 0.0,
        'Mean_Edit_Dist': round(float(eds.mean()), 2)
    }


# ============================================================================
# DEFENSE 2: Adversarial Lexical Injection
# ============================================================================

def mine_cue_words(test_data, top_k=20):
    suff = [d['question'] for d in test_data if d.get('is_sufficient', True)]
    insuff = [d['question'] for d in test_data if not d.get('is_sufficient', True)]
    vec = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
    vec.fit(suff + insuff)
    diff = vec.transform(insuff).mean(0).A1 - vec.transform(suff).mean(0).A1
    names = vec.get_feature_names_out()
    return [names[i] for i in np.argsort(-diff)[:top_k]]


def inject(question, word, position):
    sents = re.split(r'(?<=[.!?])\s+', question.strip())
    if position == 'start':
        return f"{word.capitalize()}, {question}"
    if position == 'end' and len(sents) >= 2:
        return ' '.join(sents[:-1]) + f" Note: {word}. " + sents[-1]
    if len(sents) >= 2:
        sents.insert(len(sents) // 2, f"This is {word} relevant.")
        return ' '.join(sents)
    words = question.split()
    words.insert(len(words) // 2, word + ",")
    return ' '.join(words)


def defense_2(model, tok, model_name, model_slug, layer, probe, dataset, n_cue=5):
    test_data = json.load(open(DATASETS[dataset]['test']))
    suff_qs = [d['question'] for d in test_data if d.get('is_sufficient', True)]

    cues = list(set(mine_cue_words(test_data, top_k=20)) | set(CONFOUNDER_SEED))[:n_cue]

    X_orig = extract_t0(model, tok, suff_qs, model_name, layer)
    p_orig = probe.predict_proba(X_orig)[:, 1]
    fp_base = float(np.mean(p_orig > 0.5))

    flips, increases = [], []
    for w in cues:
        for pos in ('start', 'middle', 'end'):
            injected = [inject(q, w, pos) for q in suff_qs]
            X_inj = extract_t0(model, tok, injected, model_name, layer)
            p_inj = probe.predict_proba(X_inj)[:, 1]
            flips.append(float(np.mean(p_inj > 0.5)))
            increases.append(float(np.mean(p_inj - p_orig)))

    return {
        'Dataset': dataset, 'Model': model_slug, 'Layer': layer,
        'Cues_Used': "|".join(cues),
        'Baseline_FP_Rate':       round(fp_base, 4),
        'Mean_Injection_FP_Rate': round(float(np.mean(flips)), 4),
        'Max_Injection_FP_Rate':  round(float(np.max(flips)), 4),
        'FP_Excess':              round(float(np.mean(flips) - fp_base), 4),
        'Mean_P_Increase':        round(float(np.mean(increases)), 4)
    }


# ============================================================================
# DEFENSE 3: Residual Probe (Orthogonalization)
# ============================================================================

def confounder_labels(data):
    qs = [d['question'] for d in data]
    lens = np.array([len(q.split()) for q in qs])
    nums = np.array([len(re.findall(r'\b\d+\.?\d*\b', q)) for q in qs])
    cmpx = np.array([len(re.split(r'[.!?]+', q)) for q in qs])
    pat = re.compile(r'\b(' + '|'.join(re.escape(w) for w in CONFOUNDER_SEED) + r')\b', re.I)
    lex = np.array([1 if pat.search(q) else 0 for q in qs])
    return {
        'length':     (lens > np.median(lens)).astype(int),
        'numcount':   (nums > np.median(nums)).astype(int),
        'complexity': (cmpx > np.median(cmpx)).astype(int),
        'lexical':    lex
    }


def orthogonalize(X, dirs):
    R = X.astype(np.float64).copy()
    for d in dirs:
        d = d / (np.linalg.norm(d) + 1e-12)
        R = R - np.outer(R @ d, d)
    return R


def defense_3(model, tok, model_name, model_slug, layer, probe, dataset, n_train=2000):
    train_data = json.load(open(DATASETS[dataset]['train']))
    test_data  = json.load(open(DATASETS[dataset]['test']))

    if len(train_data) > n_train:
        np.random.seed(42)
        train_data = [train_data[i] for i in np.random.choice(len(train_data), n_train, replace=False)]

    y_tr = np.array([1 if not d.get('is_sufficient', True) else 0 for d in train_data])
    y_te = np.array([1 if not d.get('is_sufficient', True) else 0 for d in test_data])
    cf_tr = confounder_labels(train_data)

    X_tr = extract_t0(model, tok, [d['question'] for d in train_data], model_name, layer)
    X_te = extract_t0(model, tok, [d['question'] for d in test_data],  model_name, layer)

    baseline_f1 = f1_score(y_te, probe.predict(X_te))

    cf_dirs, cosines = [], {}
    W_main = effective_weights(probe)
    for name, lbl in cf_tr.items():
        cp = fit_probe(X_tr, lbl)
        if cp is None: continue
        W_cf = effective_weights(cp)
        cf_dirs.append(W_cf)
        n1, n2 = np.linalg.norm(W_main), np.linalg.norm(W_cf)
        cosines[name] = float(W_main @ W_cf / (n1 * n2 + 1e-12)) if (n1 > 0 and n2 > 0) else 0.0

    X_tr_res = orthogonalize(X_tr, cf_dirs)
    X_te_res = orthogonalize(X_te, cf_dirs)
    res_probe = fit_probe(X_tr_res, y_tr)
    res_f1 = f1_score(y_te, res_probe.predict(X_te_res)) if res_probe else 0.0

    return {
        'Dataset': dataset, 'Model': model_slug, 'Layer': layer,
        'Baseline_F1':           round(float(baseline_f1), 4),
        'Residual_F1':           round(float(res_f1), 4),
        'F1_Drop':               round(float(baseline_f1 - res_f1), 4),
        **{f'Cos_vs_{k}': round(v, 4) for k, v in cosines.items()}
    }


# ============================================================================
# Master verdict
# ============================================================================

def verdict(d1, d2, d3):
    fails = []
    if d1 and d1['Pairwise_Acc'] < 0.70: fails.append(f"D1.pairwise={d1['Pairwise_Acc']:.2f}<0.70")
    if d1 and d1['Mean_Gap']     < 0.20: fails.append(f"D1.gap={d1['Mean_Gap']:.2f}<0.20")
    if d2 and d2['FP_Excess']    > 0.20: fails.append(f"D2.fp_excess={d2['FP_Excess']:.2f}>0.20")
    if d3 and d3['F1_Drop']      > 0.15: fails.append(f"D3.drop={d3['F1_Drop']:.2f}>0.15")
    if not fails:        return 'PASS', ""
    if len(fails) == 1:  return 'WARN', "; ".join(fails)
    return 'FAIL', "; ".join(fails)


def aggregate(datasets):
    rows = []
    for ds in datasets:
        d1_p = os.path.join(OUT_DIR, f'defense_1_minimal_pairs_{ds}.csv')
        d2_p = os.path.join(OUT_DIR, f'defense_2_lexical_injection_{ds}.csv')
        d3_p = os.path.join(OUT_DIR, f'defense_3_residual_probe_{ds}.csv')
        d1 = pd.read_csv(d1_p) if os.path.exists(d1_p) else pd.DataFrame()
        d2 = pd.read_csv(d2_p) if os.path.exists(d2_p) else pd.DataFrame()
        d3 = pd.read_csv(d3_p) if os.path.exists(d3_p) else pd.DataFrame()

        all_models = set()
        for df in (d1, d2, d3):
            if not df.empty: all_models |= set(df['Model'])

        for m in sorted(all_models):
            r1 = d1[d1['Model'] == m].iloc[0].to_dict() if not d1.empty and m in d1['Model'].values else None
            r2 = d2[d2['Model'] == m].iloc[0].to_dict() if not d2.empty and m in d2['Model'].values else None
            r3 = d3[d3['Model'] == m].iloc[0].to_dict() if not d3.empty and m in d3['Model'].values else None
            v, reason = verdict(r1, r2, r3)
            rows.append({
                'Dataset': ds, 'Model': m,
                'D1_Pairwise_Acc': r1['Pairwise_Acc'] if r1 else None,
                'D1_Mean_Gap':     r1['Mean_Gap']     if r1 else None,
                'D2_FP_Excess':    r2['FP_Excess']    if r2 else None,
                'D3_F1_Drop':      r3['F1_Drop']      if r3 else None,
                'Verdict':         v, 'Reasons': reason
            })

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(OUT_DIR, 'master_verdict.csv'), index=False)
        print("\n" + "=" * 95)
        print("MASTER VERDICT (t=0 confounder defenses)")
        print("=" * 95)
        print(out.to_string(index=False))
        print(f"\nSaved to {os.path.join(OUT_DIR, 'master_verdict.csv')}")


# ============================================================================
# Main loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['umwp', 'treecut', 'both'])
    parser.add_argument('--defenses', nargs='+', type=int, default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument('--aggregate_only', action='store_true')
    args = parser.parse_args()

    targets = ['umwp', 'treecut'] if args.dataset == 'both' else [args.dataset]

    if args.aggregate_only:
        aggregate(targets); return

    for dataset in targets:
        out_paths = {
            1: os.path.join(OUT_DIR, f'defense_1_minimal_pairs_{dataset}.csv'),
            2: os.path.join(OUT_DIR, f'defense_2_lexical_injection_{dataset}.csv'),
            3: os.path.join(OUT_DIR, f'defense_3_residual_probe_{dataset}.csv'),
        }

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            pending = [d for d in args.defenses if not already_done(out_paths[d], model_slug)]
            if not pending:
                print(f"[{dataset}] [SKIP] {model_slug} (all defenses done)")
                continue

            print(f"\n{'='*70}\n[{dataset}] {model_slug}  |  Running defenses: {pending}\n{'='*70}")

            layer, probe = load_unified_probe(model_slug, dataset)
            if probe is None:
                print(f"   ! Missing exp10 unified probe for {model_slug}, skipping.")
                continue

            try:
                model, tok = load_model(model_name)
            except Exception as e:
                print(f"   ! Failed to load model: {e}")
                continue

            try:
                if 1 in pending:
                    r = defense_1(model, tok, model_name, model_slug, layer, probe, dataset)
                    if r:
                        append_csv(out_paths[1], r)
                        print(f"   D1 -> pairwise={r['Pairwise_Acc']:.3f}, gap={r['Mean_Gap']:+.3f}, ed_corr={r['EditDist_Gap_r']:+.3f}")
                if 2 in pending:
                    r = defense_2(model, tok, model_name, model_slug, layer, probe, dataset)
                    append_csv(out_paths[2], r)
                    print(f"   D2 -> fp_base={r['Baseline_FP_Rate']:.3f}, fp_inj={r['Mean_Injection_FP_Rate']:.3f}, excess={r['FP_Excess']:+.3f}")
                if 3 in pending:
                    r = defense_3(model, tok, model_name, model_slug, layer, probe, dataset)
                    append_csv(out_paths[3], r)
                    print(f"   D3 -> base_f1={r['Baseline_F1']:.3f}, residual={r['Residual_F1']:.3f}, drop={r['F1_Drop']:+.3f}")
            finally:
                del model, tok
                gc.collect()
                torch.cuda.empty_cache()

    aggregate(targets)


if __name__ == '__main__':
    main()