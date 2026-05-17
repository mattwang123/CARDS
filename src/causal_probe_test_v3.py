"""
================================================================================
causal_probe_test_v3.py
================================================================================
F7 causal intervention test, paper-final version.

Tests: at the probe's best layer L*, does v_probe causally drive the
abstain-vs-num decision? Two independent causal probes:
  (a) Necessity / ablation: project out a K-dim subspace anchored at exp10's
      v_probe, observe behavior change.
  (b) Sufficiency / steering: add ±α·v_probe at L*, observe behavior change.
Both run across all four behavioral quadrants (Q1-Q4) so the result
distinguishes H_A (v_probe drives decision) from H_B (decoupling) on every
axis: under H_A we expect Q1↔Q2 to converge after ablation; under H_B all
four quadrants are stable.

Three direction candidates tested in parallel:
  v_probe — LR weights on exp10 cutoff-concat (paper anchor recognition dir)
  v_dim   — mean(h_insuff at t=0) − mean(h_suff at t=0) (Marks & Tegmark)
  v_rand  — Gaussian unit vector (null control)

Design (paper-final):

  D1. Subspace anchor: each direction's K-dim basis has basis[0] = that
      direction itself, basis[1..K-1] = INLP iterations on exp10's cached
      embeddings. For v_probe, K=0 retrained F1 reproduces exp10's reported
      F1 within ±0.005.

  D2. Manipulation check: at each K, retrain a probe on the residual after
      basis[:K] is projected out. F1-vs-K curve. For v_probe this is mostly
      redundant with inlp_probe_retrain.py (paper figure); for v_dim and
      v_rand it is an inline sanity that each direction's basis actually
      removes signal.

  D3. Full four-quadrant sampling: Q1 (hallucinate), Q2 (correct abstain),
      Q3 (correct solve), Q4 (over-abstain). Q1↔Q2 is the distinguishing
      test for ablation; Q1 alone (saturated at numeric) cannot distinguish
      H_A from H_B.

  D4. Bidirectional steering: α ∈ [-8, ..., 8]. Positive α tests sufficiency
      via Q1→Q2. Negative α tests sufficiency via Q2→Q1.

  D5. Layer mode: 'all' projects at every layer, 'lstar_only' projects only
      at L*. Discrepancy is itself a finding.

  D6. Ablation method: 'zero' replaces projected component with 0,
      'mean' replaces with population mean.

  D7. Bootstrap 95% CIs on every reported rate (N=1000).

  D8. Scope: all 21 models in DEFAULT_MODELS × 2 datasets.

Output: experiment_result/causal_results_v3/{slug}/{dataset}/
  meta.json
  summary.csv             # per (condition, cutoff, quadrant): rate + CI
  sample_major.json       # per-sample, resume-able
  manipulation_check.csv  # per (direction, K): retrained probe F1 + CI
  basis_probe.npy         # K × d, basis[0] = exp10 v_probe
  basis_dim.npy           # K × d, basis[0] = v_DIM
  DONE

Usage:
  python src/causal_probe_test_v3.py --test                       # smoke
  python src/causal_probe_test_v3.py --all_models --all_datasets  # full
  python src/causal_probe_test_v3.py --aggregate_only             # plots
================================================================================
"""

import argparse
import gc
import json
import os
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse v1's lower-level mechanics that don't have alignment concerns:
from causal_probe_test import (
    SubspaceAblationHook,
    CUTOFFS,
    FORCE_DECODE_SUFFIX,
    EXP10_DIR,
    GEN_BATCH_SIZE,
    MAX_NEW_TOKENS,
    SOURCE_BASE,
    batched_forced_decode,
    classify_answer,
    diff_in_means_direction,
    extract_t0_states,
    extract_boxed,
    find_cutoff,
    get_layer_modules,
    get_q1_q3_samples,
    is_verbalize_correct,
    load_existing_records,
    load_train_balanced_prompts,
    register_addition_hook,
    remove_handles,
)

from delta_u_intervention import is_coherent_forced


# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_BASE = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results_v3'
EMB_BASE = os.path.join(SOURCE_BASE, 'exp_temporal_new', 'embeddings_proportional')

ALPHAS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 8.0]
RANK_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
RANK_K_MAX = 128
ABLATION_LAYERS = ['all', 'lstar_only']
ABLATION_METHODS = ['zero', 'mean']
DIRECTIONS = ['probe', 'dim', 'rand']
N_BOOTSTRAP = 1000

N_Q1 = 200   # insufficient + hallucinate → primary
N_Q2 = 200   # insufficient + correct abstain → distinguishing test
N_Q3 = 100   # sufficient + solve → specificity control
N_Q4 = 100   # sufficient + over-abstain → symmetric necessity test
N_DIM_PER_CLASS = 300

K0_F1_TOLERANCE = 0.005

DEFAULT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B',                'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B',                       'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B',                  'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B',          'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think',               'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B',                      'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it',                 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct',
]


# ============================================================================
# SKLEARN PROBE
# ============================================================================

def make_probe(seed=42):
    """Hyperparams locked across exp10, inlp_probe_retrain, final_layer_probe,
    and this script."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           C=1.0, solver='lbfgs', n_jobs=-1, random_state=seed),
    )


def direction_from_probe(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    w = clf.coef_[0] / scaler.scale_
    n = np.linalg.norm(w)
    return (w / n).astype(np.float32) if n > 1e-12 else np.zeros_like(w, dtype=np.float32)


# ============================================================================
# GENERALIZED INLP BASIS BUILDER
# ============================================================================

def build_inlp_basis(anchor_direction, X_train, y_train, K_max, seed=42):
    """
    Anchored-INLP basis builder.

    basis[0]   = anchor_direction (unit-normed; whatever was passed in)
    basis[k>0] = INLP iteration on X_train: project out basis[:k], retrain
                 a probe, take its direction, Gram-Schmidt against existing
                 basis for numerical safety.

    Two callers:
      v_probe basis: anchor = exp10 saved v_probe (paper anchor for recognition)
      v_dim basis:   anchor = mean(h_insuff)−mean(h_suff) at t=0 (M&T)

    X_train should be exp10's cached embeddings flattened across the
    6-cutoff axis (paper-consistent with F1-F6).

    Returns: basis (K_max, d). If subspace exhausts, pads with random
    orthonormal continuations.
    """
    d = anchor_direction.shape[0]
    basis = np.zeros((K_max, d), dtype=np.float32)
    nrm0 = max(float(np.linalg.norm(anchor_direction)), 1e-12)
    basis[0] = (anchor_direction / nrm0).astype(np.float32)

    X = X_train.astype(np.float32, copy=True)
    X = X - (X @ basis[0:1].T) @ basis[0:1]

    for k in range(1, K_max):
        probe = make_probe(seed + k - 1)
        probe.fit(X, y_train)
        v = direction_from_probe(probe)
        for u in basis[:k]:
            v = v - float(np.dot(v, u)) * u
        nrm = np.linalg.norm(v)
        if nrm < 1e-10:
            rng = np.random.RandomState(seed + 9000 + k)
            for kk in range(k, K_max):
                w = rng.randn(d).astype(np.float32)
                for u in basis[:kk]:
                    w = w - float(np.dot(w, u)) * u
                w = w / max(np.linalg.norm(w), 1e-12)
                basis[kk] = w
            break
        basis[k] = (v / nrm).astype(np.float32)
        X = X - (X @ basis[k:k+1].T) @ basis[k:k+1]

    return basis


def random_orthonormal_basis(d, K, seed=123):
    rng = np.random.RandomState(seed)
    M = rng.randn(K, d).astype(np.float32)
    B = np.zeros_like(M)
    for k in range(K):
        v = M[k]
        for u in B[:k]:
            v = v - float(np.dot(v, u)) * u
        n = np.linalg.norm(v)
        if n < 1e-10:
            continue
        B[k] = v / n
    return B


def compute_basis_means(X, basis):
    return (X @ basis.T).mean(axis=0).astype(np.float32)


# ============================================================================
# EXP10 EMBEDDINGS
# ============================================================================

def load_exp10_embeddings(slug, dataset, best_layer):
    base = Path(EMB_BASE) / dataset / slug
    if not base.exists():
        return None
    xtr_p = base / f'X_train_layer{best_layer}.npy'
    xte_p = base / f'X_test_layer{best_layer}.npy'
    ytr_p = base / 'y_train.npy'
    yte_p = base / 'y_test.npy'
    if not all(p.exists() for p in [xtr_p, xte_p, ytr_p, yte_p]):
        return None
    X_tr = np.load(xtr_p); X_te = np.load(xte_p)
    y_tr = np.load(ytr_p); y_te = np.load(yte_p)
    if X_tr.ndim == 3:
        N, n_pct, D = X_tr.shape
        X_tr = X_tr.reshape(N * n_pct, D)
        y_tr = np.repeat(y_tr, n_pct)
    if X_te.ndim == 3:
        N, n_pct, D = X_te.shape
        X_te = X_te.reshape(N * n_pct, D)
        y_te = np.repeat(y_te, n_pct)
    return X_tr.astype(np.float32), y_tr.astype(int), X_te.astype(np.float32), y_te.astype(int)


# ============================================================================
# Q2 + Q4 SAMPLING
# ============================================================================

def get_q2_q4_samples(slug, dataset, n_q2, n_q4):
    """Sample Q2 (insufficient + correct abstain) and Q4 (sufficient +
    over-abstain) from the same baseline CoT generations v1 uses."""
    path = (Path(SOURCE_BASE) / 'experiments' / 'dynamic_tracking_test'
            / 'math' / slug / f'{dataset}_cot_generations.json')
    if not path.exists():
        return [], []
    with open(path) as f:
        items = json.load(f)
    q2, q4 = [], []
    for idx, item in enumerate(items):
        is_suff = item.get('is_sufficient', True)
        boxed = extract_boxed(item.get('generated_response', ''))
        kind = classify_answer(boxed) if boxed else 'other'
        if kind != 'insufficient':
            continue
        rec = {
            'sample_id': f'{slug}__{dataset}__q24__{idx}',
            'prompt': item['prompt'],
            'cot': item.get('generated_response', ''),
            'is_sufficient': is_suff,
        }
        if (not is_suff) and len(q2) < n_q2:
            rec['quadrant'] = 'Q2_Correct_Abstain'; q2.append(rec)
        elif is_suff and len(q4) < n_q4:
            rec['quadrant'] = 'Q4_Over_Abstain'; q4.append(rec)
        if len(q2) >= n_q2 and len(q4) >= n_q4:
            break
    return q2, q4


# ============================================================================
# HOOKS
# ============================================================================

class MeanReplacementAblationHook:
    """h' = h - sum_k ((h @ v_k) - mean_k) * v_k"""
    def __init__(self, V_basis, mean_vec):
        self.V = V_basis
        self.mean_vec = mean_vec.astype(np.float32)
        self._V_cached = None
        self._mean_cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if (self._V_cached is None or self._V_cached.device != hs.device
                or self._V_cached.dtype != hs.dtype):
            self._V_cached = torch.tensor(self.V, dtype=hs.dtype, device=hs.device)
            self._mean_cached = torch.tensor(self.mean_vec, dtype=hs.dtype, device=hs.device)
        proj_coef = hs @ self._V_cached.T
        subtract = (proj_coef - self._mean_cached) @ self._V_cached
        modified = hs - subtract
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


def register_ablation(layer_modules, V_basis, mean_vec, layer_mode, lstar,
                      model=None):
    if layer_mode == 'all':
        targets = layer_modules
    elif layer_mode == 'lstar_only':
        targets = [layer_modules[lstar]]
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")
    handles = []
    for lm in targets:
        if mean_vec is None:
            handles.append(lm.register_forward_hook(SubspaceAblationHook(V_basis)))
        else:
            handles.append(lm.register_forward_hook(
                MeanReplacementAblationHook(V_basis, mean_vec)))
    return handles


def _register_steering(target_module, v_t, a_raw, model=None):
    return register_addition_hook(target_module, v_t, a_raw)


def _register_baseline(model=None):
    return []


# ============================================================================
# BOOTSTRAP
# ============================================================================

def bootstrap_ci_rate(arr, n_boot=N_BOOTSTRAP, ci=0.95, seed=42):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (float(arr.mean()),
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)))


def bootstrap_ci_f1(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=0.95, seed=42):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    f1s = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        i = rng.randint(0, y_true.size, size=y_true.size)
        f1s[b] = f1_score(y_true[i], y_pred[i], zero_division=0)
    return (float(f1_score(y_true, y_pred, zero_division=0)),
            float(np.quantile(f1s, (1.0 - ci) / 2.0)),
            float(np.quantile(f1s, 1.0 - (1.0 - ci) / 2.0)))


# ============================================================================
# MANIPULATION CHECK
# ============================================================================

def manipulation_check_curve(X_train, y_train, X_test, y_test, basis, K_list,
                             exp10_f1=None, tolerance=K0_F1_TOLERANCE, label=''):
    """K=0 retrained probe F1, then K iterations of project-and-retrain.
    If exp10_f1 given and basis[0] is paper anchor, warn on K=0 mismatch."""
    rows = []
    p0 = make_probe(seed=42)
    p0.fit(X_train, y_train)
    f1, lo, hi = bootstrap_ci_f1(y_test, p0.predict(X_test))
    rows.append({'K': 0, 'f1': round(f1, 4),
                 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})

    if exp10_f1 is not None and not np.isnan(exp10_f1):
        delta = abs(f1 - exp10_f1)
        if delta > tolerance:
            print(f"  [WARN] {label}: K=0 retrained F1 ({f1:.4f}) differs from "
                  f"exp10 F1 ({exp10_f1:.4f}) by {delta:.4f} > {tolerance}")

    for K in K_list:
        if K > basis.shape[0]:
            continue
        B = basis[:K]
        X_tr = X_train - (X_train @ B.T) @ B
        X_te = X_test  - (X_test  @ B.T) @ B
        p = make_probe(seed=42 + K)
        p.fit(X_tr, y_train)
        f1, lo, hi = bootstrap_ci_f1(y_test, p.predict(X_te))
        rows.append({'K': K, 'f1': round(f1, 4),
                     'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
    return rows


# ============================================================================
# CONDITION BUILDER
# ============================================================================

def _alpha_to_str(a):
    sign = 'p' if a >= 0 else 'n'
    return f'a{sign}{("%g" % abs(a)).replace(".", "_")}'


def build_conditions(target_module, layer_modules, lstar,
                     directions_with_bases, alpha_scale, args):
    """Conditions: baseline + steering × direction × α + ablation × direction
    × K × layer_mode × method."""
    conditions = [('baseline', _register_baseline)]

    if not args.skip_steering:
        for name, v_unit, _basis, _means in directions_with_bases:
            for a in args.alphas:
                a_raw = a * alpha_scale
                v_t = torch.tensor(v_unit, dtype=torch.float32)
                conditions.append((
                    f'steer_{name}_{_alpha_to_str(a)}',
                    partial(_register_steering, target_module, v_t, a_raw),
                ))

    if not args.skip_ablation:
        for K in args.rank_K_values:
            for name, _v, basis_full, mean_full in directions_with_bases:
                K_eff = min(K, basis_full.shape[0])
                B = basis_full[:K_eff].copy()
                M = mean_full[:K_eff].copy()
                for layer_mode in args.ablation_layers:
                    lm_short = 'all' if layer_mode == 'all' else 'lstar'
                    for method in args.ablation_methods:
                        mv = None if method == 'zero' else M
                        conditions.append((
                            f'ablate_{name}_K{K}_{lm_short}_{method}',
                            partial(register_ablation, layer_modules, B, mv,
                                    layer_mode, lstar),
                        ))
    return conditions


# ============================================================================
# RESIDUAL STAT
# ============================================================================

@torch.no_grad()
def compute_layer_residual_norm(model, tokenizer, prompts, target_layer,
                                 batch_size=4):
    norms = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        seq_len = enc['input_ids'].shape[1]
        h = out.hidden_states[target_layer][:, seq_len - 1, :].to(torch.float32).cpu().numpy()
        norms.extend(np.linalg.norm(h, axis=1).tolist())
        del out
    return float(np.mean(norms)), float(np.std(norms))


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_summary(sample_records, slug, dataset, out_path,
                     n_boot=N_BOOTSTRAP):
    groups = {}
    for rec in sample_records:
        for cutoff_key, by_cond in rec.get('cutoffs', {}).items():
            for cond_name, v in by_cond.items():
                key = (cond_name, int(cutoff_key), rec.get('quadrant', ''))
                groups.setdefault(key, []).append(v)
    rows = []
    for (cond, ck, quad), vs in sorted(groups.items()):
        n = len(vs)
        if n == 0:
            continue
        ins = np.array([1.0 if v['answer_kind'] == 'insufficient' else 0.0 for v in vs])
        num = np.array([1.0 if v['answer_kind'] == 'numeric'      else 0.0 for v in vs])
        coh = np.array([1.0 if v.get('coherent', False)            else 0.0 for v in vs])
        vc  = np.array([1.0 if v.get('verbalize_correct', False)   else 0.0 for v in vs])
        ins_m, ins_lo, ins_hi = bootstrap_ci_rate(ins, n_boot=n_boot)
        num_m, num_lo, num_hi = bootstrap_ci_rate(num, n_boot=n_boot)
        coh_m, coh_lo, coh_hi = bootstrap_ci_rate(coh, n_boot=n_boot)
        vc_m,  vc_lo,  vc_hi  = bootstrap_ci_rate(vc,  n_boot=n_boot)
        rows.append({
            'model': slug, 'dataset': dataset, 'condition': cond,
            'cutoff_pct': ck, 'quadrant': quad, 'n': n,
            'insuff_rate': round(ins_m, 4),
            'insuff_ci_low': round(ins_lo, 4), 'insuff_ci_high': round(ins_hi, 4),
            'numeric_rate': round(num_m, 4),
            'numeric_ci_low': round(num_lo, 4), 'numeric_ci_high': round(num_hi, 4),
            'coherence_rate': round(coh_m, 4),
            'coherence_ci_low': round(coh_lo, 4), 'coherence_ci_high': round(coh_hi, 4),
            'verbalize_acc': round(vc_m, 4),
            'verbalize_ci_low': round(vc_lo, 4), 'verbalize_ci_high': round(vc_hi, 4),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================================
# PER-(MODEL, DATASET) RUN
# ============================================================================

def out_paths(output_base, slug, dataset):
    base = Path(output_base) / slug / dataset
    base.mkdir(parents=True, exist_ok=True)
    return {
        'base': base,
        'meta': base / 'meta.json',
        'summary': base / 'summary.csv',
        'sample_major': base / 'sample_major.json',
        'manipulation_check': base / 'manipulation_check.csv',
        'basis_probe': base / 'basis_probe.npy',
        'basis_dim':   base / 'basis_dim.npy',
    }


def record_has_all_cutoffs(rec, cond_name):
    if rec is None:
        return False
    cs = rec.get('cutoffs', {})
    for cutoff in CUTOFFS:
        ck = str(int(round(cutoff * 100)))
        if cs.get(ck, {}).get(cond_name) is None:
            return False
    return True


def run_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    paths = out_paths(args.output_dir, slug, dataset)

    done_marker = paths['base'] / 'DONE'
    if done_marker.exists() and not args.force:
        print(f"[DONE] {slug}/{dataset} — already complete, skipping")
        return

    exp10_csv = os.path.join(EXP10_DIR, 'results',
                              f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[SKIP] {slug}/{dataset}: no exp10 csv"); return
    df10 = pd.read_csv(exp10_csv)
    row10 = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row10.empty:
        print(f"[SKIP] {slug}/{dataset}: no exp10 entry"); return
    best_layer = int(row10['Optimal_Layer'].iloc[0])
    exp10_anchor_f1 = float(row10['Unified_Test_F1'].iloc[0])

    probe_path = os.path.join(EXP10_DIR, 'probes_proportional', dataset, slug,
                               f'unified_probe_layer{best_layer}.joblib')
    if not os.path.exists(probe_path):
        print(f"[SKIP] {slug}/{dataset}: no exp10 probe"); return
    exp10_probe = joblib.load(probe_path)
    v_probe = direction_from_probe(exp10_probe)
    hidden_dim = v_probe.shape[0]

    emb = load_exp10_embeddings(slug, dataset, best_layer)
    if emb is None:
        print(f"[SKIP] {slug}/{dataset}: no exp10 embeddings at L{best_layer}"); return
    X_tr_emb, y_tr_emb, X_te_emb, y_te_emb = emb

    print(f"[RUN] {slug}/{dataset}: best_layer={best_layer}, "
          f"exp10_F1={exp10_anchor_f1:.4f}")

    # ─── Q1-Q4 sampling ──────────────────────────────────────────────────
    q1_samples, q3_samples = get_q1_q3_samples(slug, dataset, args.n_q1, args.n_q3)
    q2_samples, q4_samples = get_q2_q4_samples(slug, dataset, args.n_q2, args.n_q4)
    all_samples = q1_samples + q2_samples + q3_samples + q4_samples
    print(f"  samples: Q1={len(q1_samples)} Q2={len(q2_samples)} "
          f"Q3={len(q3_samples)} Q4={len(q4_samples)}")
    if len(all_samples) == 0:
        print(f"  ! no samples — skipping"); return

    # ─── Load model ──────────────────────────────────────────────────────
    print(f"  Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    num_gpus = torch.cuda.device_count()
    max_mem = {i: '78GiB' for i in range(num_gpus)} if num_gpus else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', max_memory=max_mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()
    layer_modules = get_layer_modules(model)
    target_module = layer_modules[best_layer]
    n_layers = len(layer_modules)

    # ─── v_DIM extraction (M&T t=0 diff-of-means) ───────────────────────
    suff_pmpts, insuff_pmpts = load_train_balanced_prompts(slug, dataset, args.n_dim)
    if (not suff_pmpts or not insuff_pmpts
            or len(suff_pmpts) < 50 or len(insuff_pmpts) < 50):
        print(f"  ! insufficient train prompts for v_DIM; falling back to test prompts")
        suff_pmpts = [s['prompt'] for s in q3_samples][:200]
        insuff_pmpts = [s['prompt'] for s in q1_samples][:200]

    print(f"  Computing v_DIM (t=0 diff-of-means, "
          f"n_per_class={len(suff_pmpts)}/{len(insuff_pmpts)})...")
    X_suff_t0 = extract_t0_states(model, tok, suff_pmpts, best_layer)
    X_insuff_t0 = extract_t0_states(model, tok, insuff_pmpts, best_layer)
    v_dim_raw, dim_mag = diff_in_means_direction(X_insuff_t0, X_suff_t0)
    nrm = max(float(np.linalg.norm(v_dim_raw)), 1e-12)
    v_dim = (v_dim_raw / nrm).astype(np.float32)
    cos_probe_dim = float(np.dot(v_probe, v_dim))
    print(f"  cos(v_probe, v_DIM) = {cos_probe_dim:.4f}, "
          f"||v_DIM_raw|| = {dim_mag:.4f}")

    # ─── Build bases (probe + dim + rand) ───────────────────────────────
    K_max_used = min(args.rank_K_max, hidden_dim - 1)
    print(f"  Building INLP bases (K_max={K_max_used})...")
    probe_basis = build_inlp_basis(v_probe, X_tr_emb, y_tr_emb,
                                    K_max=K_max_used, seed=42)
    dim_basis   = build_inlp_basis(v_dim,   X_tr_emb, y_tr_emb,
                                    K_max=K_max_used, seed=43)
    rand_basis  = random_orthonormal_basis(hidden_dim, K_max_used, seed=123)
    np.save(paths['basis_probe'], probe_basis)
    np.save(paths['basis_dim'],   dim_basis)

    probe_means = compute_basis_means(X_tr_emb, probe_basis)
    dim_means   = compute_basis_means(X_tr_emb, dim_basis)
    rand_means  = compute_basis_means(X_tr_emb, rand_basis)

    # ─── Alpha scale ─────────────────────────────────────────────────────
    norm_pool = (suff_pmpts + insuff_pmpts)[:200]
    res_mean_norm, res_std_norm = compute_layer_residual_norm(
        model, tok, norm_pool, best_layer, batch_size=4)
    alpha_scale = res_mean_norm / np.sqrt(hidden_dim)

    v_rand = rand_basis[0]

    # ─── Manipulation check ─────────────────────────────────────────────
    print("  Manipulation check (probe retrain at each K)...")
    K_list_for_mc = [k for k in args.rank_K_values if k <= K_max_used]
    mc_rows = []
    for basis_full, name, anchor_check in [
        (probe_basis, 'probe', exp10_anchor_f1),
        (dim_basis,   'dim',   None),
        (rand_basis,  'rand',  None),
    ]:
        rows = manipulation_check_curve(
            X_tr_emb, y_tr_emb, X_te_emb, y_te_emb, basis_full,
            K_list_for_mc, exp10_f1=anchor_check,
            label=f'{slug}/{dataset}/{name}')
        for r in rows:
            mc_rows.append({'model': slug, 'dataset': dataset,
                            'direction': name, **r})
    pd.DataFrame(mc_rows).to_csv(paths['manipulation_check'], index=False)

    # ─── Meta ────────────────────────────────────────────────────────────
    cos_probe_rand = float(np.dot(v_probe, v_rand))
    cos_dim_rand   = float(np.dot(v_dim,   v_rand))
    meta = {
        'model': slug, 'dataset': dataset, 'version': 'v3',
        'best_layer': best_layer, 'final_layer': n_layers,
        'hidden_dim': hidden_dim,
        'exp10_anchor_f1': exp10_anchor_f1,
        'cos_probe_dim': cos_probe_dim,
        'cos_probe_rand': cos_probe_rand,
        'cos_dim_rand': cos_dim_rand,
        'v_dim_raw_magnitude': float(dim_mag),
        'residual_norm_mean_at_best_layer': res_mean_norm,
        'residual_norm_std_at_best_layer': res_std_norm,
        'alpha_units_definition': 'mean(||h||)/sqrt(d) at best_layer',
        'alpha_scale_raw': float(alpha_scale),
        'alphas_in_residual_units': args.alphas,
        'rank_K_values': K_list_for_mc,
        'K_max_used': K_max_used,
        'ablation_layers': args.ablation_layers,
        'ablation_methods': args.ablation_methods,
        'n_q1': len(q1_samples), 'n_q2': len(q2_samples),
        'n_q3': len(q3_samples), 'n_q4': len(q4_samples),
        'n_bootstrap': args.n_bootstrap,
        'basis_anchors': {
            'probe': 'exp10 saved v_probe at best_layer',
            'dim':   't=0 diff-of-means (mean(h_insuff)-mean(h_suff))',
            'rand':  'gaussian unit vector seed=123',
        },
        'k0_f1_tolerance': K0_F1_TOLERANCE,
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    # ─── Pre-tokenize forced inputs ──────────────────────────────────────
    print("  Pre-computing cutoff truncations...")
    forced_inputs = {}
    for s in all_samples:
        for cutoff in CUTOFFS:
            ck = int(round(cutoff * 100))
            if cutoff <= 0.0:
                truncated = ''
            else:
                c = find_cutoff(s['cot'], cutoff, tok)
                truncated = c.text
            forced_inputs[(s['sample_id'], ck)] = (
                s['prompt'] + truncated + FORCE_DECODE_SUFFIX
            )

    # ─── Build conditions ────────────────────────────────────────────────
    args.rank_K_values = K_list_for_mc
    args.rank_K_max = K_max_used
    directions_with_bases = [
        ('probe', v_probe, probe_basis, probe_means),
        ('dim',   v_dim,   dim_basis,   dim_means),
        ('rand',  v_rand,  rand_basis,  rand_means),
    ]
    conditions = build_conditions(
        target_module, layer_modules, best_layer,
        directions_with_bases, alpha_scale, args,
    )
    print(f"  Built {len(conditions)} conditions")

    # ─── Run with resume ─────────────────────────────────────────────────
    sample_records = load_existing_records(paths['sample_major'])
    by_idx = {r['sample_id']: r for r in sample_records}

    for cond_name, register_fn in tqdm(conditions, desc='  conditions'):
        need = [s for s in all_samples
                if not record_has_all_cutoffs(by_idx.get(s['sample_id']), cond_name)]
        if not need:
            continue

        handles = register_fn(model)
        try:
            for cutoff in CUTOFFS:
                ck = int(round(cutoff * 100))
                pending = [s for s in need
                           if (by_idx.get(s['sample_id']) is None) or
                              (by_idx[s['sample_id']].get('cutoffs', {})
                                .get(str(ck), {}).get(cond_name) is None)]
                if not pending:
                    continue
                batch_inputs = [forced_inputs[(s['sample_id'], ck)] for s in pending]
                gen_results = batched_forced_decode(
                    model, tok, batch_inputs,
                    batch_size=args.gen_batch_size,
                    max_new_tokens=MAX_NEW_TOKENS)
                for s, gr in zip(pending, gen_results):
                    boxed = extract_boxed(gr['raw_output'])
                    kind = classify_answer(boxed)
                    coh = is_coherent_forced(gr['continuation'], boxed)
                    correct = is_verbalize_correct(boxed, s.get('is_sufficient', True))
                    rec = by_idx.get(s['sample_id'])
                    if rec is None:
                        rec = {
                            'sample_id': s['sample_id'],
                            'quadrant': s.get('quadrant', ''),
                            'is_sufficient': s.get('is_sufficient'),
                            'cutoffs': {},
                        }
                        sample_records.append(rec)
                        by_idx[s['sample_id']] = rec
                    rec['cutoffs'].setdefault(str(ck), {})[cond_name] = {
                        'boxed': boxed,
                        'answer_kind': kind,
                        'coherent': coh,
                        'verbalize_correct': correct,
                    }
        finally:
            remove_handles(handles)

        with open(paths['sample_major'], 'w') as f:
            json.dump(sample_records, f, indent=2)
        aggregate_summary(sample_records, slug, dataset, paths['summary'],
                          n_boot=args.n_bootstrap)

    aggregate_summary(sample_records, slug, dataset, paths['summary'],
                      n_boot=args.n_bootstrap)
    done_marker.touch()
    print(f"  -> done. Summary at {paths['summary']}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# PLOTS
# ============================================================================

DIR_COLORS = {'probe': 'tab:blue', 'dim': 'tab:orange', 'rand': 'tab:green'}


def plot_per_pair_box(output_base, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    base = Path(output_base)

    dfs = []
    for p in base.glob('*/*/summary.csv'):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    if not dfs:
        print(f'No summary.csv files under {base}'); return
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(base / 'all_pairs_summary.csv', index=False)
    n_pairs = big[['model', 'dataset']].drop_duplicates().shape[0]
    print(f'Wrote all_pairs_summary.csv: {len(big)} rows, {n_pairs} pairs')

    sub = big[(big['quadrant'].str.startswith('Q1'))
              & (big['cutoff_pct'] == 0)
              & big['condition'].str.startswith('ablate_')].copy()
    if not sub.empty:
        sub['K'] = sub['condition'].str.extract(r'_K(\d+)_').astype(float).iloc[:, 0]
        sub['direction'] = sub['condition'].str.extract(r'ablate_(probe|dim|rand)_')[0]
        sub['layer_mode'] = sub['condition'].str.extract(r'_(all|lstar)_')[0]
        sub['method'] = sub['condition'].str.extract(r'_(zero|mean)$')[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        for ax, (lm, mth) in zip(axes.flat,
                                  [(lm, mth) for lm in ['all', 'lstar']
                                              for mth in ['zero', 'mean']]):
            cell = sub[(sub['layer_mode'] == lm) & (sub['method'] == mth)]
            if cell.empty:
                ax.set_visible(False); continue
            for dir_ in DIRECTIONS:
                d = cell[cell['direction'] == dir_]
                if d.empty:
                    continue
                Ks = sorted(d['K'].unique())
                data = [d[d['K'] == K]['insuff_rate'].values for K in Ks]
                positions = np.arange(len(Ks)) + (DIRECTIONS.index(dir_) - 1) * 0.25
                bp = ax.boxplot(data, positions=positions, widths=0.2,
                                patch_artist=True, labels=[''] * len(Ks))
                for patch in bp['boxes']:
                    patch.set_facecolor(DIR_COLORS[dir_]); patch.set_alpha(0.5)
            ax.set_xticks(np.arange(len(Ks)))
            ax.set_xticklabels([int(k) for k in Ks])
            ax.set_xlabel('K (rank of ablated subspace)')
            ax.set_ylabel('Q1 insuff_rate (per pair)')
            ax.set_title(f'Ablation: layer={lm}, method={mth}')
            ax.grid(alpha=0.3)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor=DIR_COLORS[d], alpha=0.5, label=d)
                                for d in DIRECTIONS],
                      loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_dir / 'box_ablate_K_curve_q1.png', dpi=150,
                    bbox_inches='tight')
        plt.close()

    for quad_prefix, label in [('Q1', 'q1'), ('Q2', 'q2'),
                                ('Q3', 'q3'), ('Q4', 'q4')]:
        sub = big[(big['quadrant'].str.startswith(quad_prefix))
                  & (big['cutoff_pct'] == 0)
                  & big['condition'].str.startswith('steer_')].copy()
        if sub.empty:
            continue
        sub['direction'] = sub['condition'].str.extract(r'steer_(probe|dim|rand)_')[0]
        sub['alpha_str'] = sub['condition'].str.extract(
            r'steer_(?:probe|dim|rand)_(.+)$')[0]

        def decode_alpha(a):
            s = a[1:]
            sign = -1 if s.startswith('n') else 1
            return sign * float(s[1:].replace('_', '.'))
        sub['alpha'] = sub['alpha_str'].apply(decode_alpha)

        fig, ax = plt.subplots(figsize=(12, 6))
        for dir_ in DIRECTIONS:
            d = sub[sub['direction'] == dir_]
            if d.empty:
                continue
            alphas = sorted(d['alpha'].unique())
            data = [d[d['alpha'] == a]['insuff_rate'].values for a in alphas]
            positions = np.array(alphas) + (DIRECTIONS.index(dir_) - 1) * 0.15
            bp = ax.boxplot(data, positions=positions, widths=0.12,
                            patch_artist=True, labels=[''] * len(alphas))
            for patch in bp['boxes']:
                patch.set_facecolor(DIR_COLORS[dir_]); patch.set_alpha(0.5)
        ax.set_xlabel('alpha (residual-units; negative = subtract direction)')
        ax.set_ylabel(f'{label.upper()} insuff_rate (per pair)')
        ax.set_title(f'Bidirectional steering, {quad_prefix} (cutoff=0)')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        ax.grid(alpha=0.3)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=DIR_COLORS[d], alpha=0.5, label=d)
                            for d in DIRECTIONS],
                  loc='upper right', fontsize=9)
        plt.tight_layout()
        plt.savefig(plot_dir / f'box_steer_alpha_{label}.png', dpi=150,
                    bbox_inches='tight')
        plt.close()


def plot_manipulation_check(output_base, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    base = Path(output_base)

    dfs = []
    for p in base.glob('*/*/manipulation_check.csv'):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    if not dfs:
        print(f'No manipulation_check.csv files under {base}'); return
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(base / 'all_pairs_manipulation.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, dir_ in zip(axes, DIRECTIONS):
        sub = big[big['direction'] == dir_]
        if sub.empty:
            ax.set_visible(False); continue
        for (m, d), g in sub.groupby(['model', 'dataset']):
            g = g.sort_values('K')
            ax.plot(g['K'], g['f1'], color='gray', alpha=0.4, lw=0.8)
        mean = sub.groupby('K')['f1'].mean()
        ax.plot(mean.index, mean.values, color='black', lw=2.0,
                label='mean across pairs')
        ax.set_xlabel('K (rank projected out)')
        ax.set_ylabel('retrained probe F1')
        ax.set_title(f'Manipulation check — {dir_}')
        ax.set_xscale('symlog')
        ax.axhline(0.5, color='red', ls='--', lw=0.5, label='chance')
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig(plot_dir / 'manipulation_check_K_curve.png', dpi=150,
                bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--output_dir', default=OUTPUT_BASE)
    p.add_argument('--n_q1', type=int, default=N_Q1)
    p.add_argument('--n_q2', type=int, default=N_Q2)
    p.add_argument('--n_q3', type=int, default=N_Q3)
    p.add_argument('--n_q4', type=int, default=N_Q4)
    p.add_argument('--n_dim', type=int, default=N_DIM_PER_CLASS)
    p.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    p.add_argument('--rank_K_values', type=int, nargs='+', default=RANK_K_VALUES)
    p.add_argument('--rank_K_max', type=int, default=RANK_K_MAX)
    p.add_argument('--ablation_layers', nargs='+', default=ABLATION_LAYERS,
                   choices=['all', 'lstar_only'])
    p.add_argument('--ablation_methods', nargs='+', default=ABLATION_METHODS,
                   choices=['zero', 'mean'])
    p.add_argument('--n_bootstrap', type=int, default=N_BOOTSTRAP)
    p.add_argument('--gen_batch_size', type=int, default=GEN_BATCH_SIZE)
    p.add_argument('--skip_steering', action='store_true')
    p.add_argument('--skip_ablation', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--aggregate_only', action='store_true')
    p.add_argument('--test', action='store_true',
                   help='Smoke: n=5 per quadrant, K=1,2, 2 alphas')
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        plot_per_pair_box(args.output_dir, Path(args.output_dir) / '_plots')
        plot_manipulation_check(args.output_dir, Path(args.output_dir) / '_plots')
        return

    if args.test:
        args.n_q1 = 5; args.n_q2 = 5; args.n_q3 = 5; args.n_q4 = 5
        args.n_dim = 20
        args.alphas = [1.0, -1.0]
        args.rank_K_values = [1, 2]
        args.rank_K_max = 2
        args.ablation_layers = ['all']
        args.ablation_methods = ['zero']
        args.n_bootstrap = 100

    if args.all_models:
        models = DEFAULT_MODELS
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("Specify --model or --all_models")

    if args.all_datasets:
        datasets = ['umwp', 'treecut']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Specify --dataset or --all_datasets")

    for m in models:
        for ds in datasets:
            t0 = time.time()
            try:
                run_one(m, ds, args)
                print(f"  [{m}/{ds}] completed in {(time.time()-t0)/60:.1f} min")
            except Exception as e:
                print(f"\n[ERROR] {m}/{ds}: {e}")
                import traceback; traceback.print_exc()

    plot_per_pair_box(args.output_dir, Path(args.output_dir) / '_plots')
    plot_manipulation_check(args.output_dir, Path(args.output_dir) / '_plots')


if __name__ == '__main__':
    main()