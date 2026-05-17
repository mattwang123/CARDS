"""
================================================================================
causal_probe_test.py  —  Unified Causal Test of the Insufficiency Direction
================================================================================

Replaces exp13_steering_new.py + two_track_exp/x_not_equal_y.py for the paper's
mechanism section. See experiment_design_causal.md for the full design.

What this script does in one run, per (model, dataset):
  1. Quality filter: requires non-degenerate baseline probe (F1 in [0.70, 0.95]).
  2. Compute directions: v_Y (probe), v_DIM (diff-in-means), v_rand (random).
  3. Iteratively retrain probes to build rank-K basis {v_Y^(1), ..., v_Y^(K)}.
  4. Pre-tokenize all (sample, cutoff) inputs ONCE.
  5. For each intervention condition (baseline + steering + rank-K ablation):
       a. Register hooks
       b. Probe sanity check (probe F1 with hooks active)
       c. Force-decode at every cutoff for every sample
       d. Save per-sample answers + per-condition aggregates
       e. Unregister hooks
  6. Save artifacts for future experiments:
       - residual stream caches at force-decode position (baseline only)
       - layer-wise MLP + attention outputs at force-decode position (baseline only)
       - top-50 next-token logits at force-decode position (baseline only)
       - rank-K probe iterates (joblib)
       - cosines + norm statistics
  7. Aggregate to a global summary.csv.

Usage:
  CUDA_VISIBLE_DEVICES=0 python src/causal_probe_test.py \
      --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset umwp

  # Or run all models in DEFAULT_MODELS, one dataset:
  CUDA_VISIBLE_DEVICES=0 python src/causal_probe_test.py \
      --all_models --dataset umwp

  # Smoke test (n=5, K=1):
  python src/causal_probe_test.py --model Qwen/Qwen2.5-Math-1.5B-Instruct \
      --dataset umwp --test
================================================================================
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


# ============================================================================
# CONFIG
# ============================================================================

# Storage roots
SOURCE_BASE = '/export/fs06/hwang302/CARDS'
OUTPUT_BASE = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results'
EXP10_DIR = os.path.join(SOURCE_BASE, 'exp_temporal_new')
EXP14_BASE = '/export/fs06/xwang397/CARDS/results_new'

# Cutoff schedule (matches exp14)
CUTOFFS = [0.0, 0.20, 0.40, 0.60, 1.00]
FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"

# Magnitudes for steering: alpha is in units of layer residual std
ALPHAS_STD_UNITS = [0.5, 1.0, 2.0, 4.0, 8.0]

# Ranks for rank-K ablation
RANK_K_VALUES = [1, 2, 4, 8, 16]

# Sample sizes
N_Q1 = 200    # hallucination samples (insufficient input, model gives number)
N_Q3 = 100    # correctly-solved samples (sufficient input, model gives correct number) — control
N_DIM_PER_CLASS = 300
PROBE_SANITY_N = 200

# Generation
MAX_NEW_TOKENS = 50
GEN_BATCH_SIZE = 16
EXTRACT_BATCH_SIZE = 8

# Models to run
DEFAULT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
]


# ============================================================================
# QUALITY FILTER
# ============================================================================

def passes_quality_filter(model_slug, dataset, exp10_csv_path):
    """Probe F1 at t=0 must be in [0.70, 0.95] — non-degenerate AND non-saturated."""
    if not os.path.exists(exp10_csv_path):
        return False, "exp10 csv missing"
    df = pd.read_csv(exp10_csv_path)
    row = df[(df['Model'] == model_slug) & (df['Percentage'] == '0%')]
    if row.empty:
        return False, f"no exp10 row for {model_slug}"
    f1 = float(row['Unified_Test_F1'].iloc[0])
    if f1 < 0.70:
        return False, f"baseline probe F1 = {f1:.3f} < 0.70 (degenerate)"
    if f1 > 0.95:
        return False, f"baseline probe F1 = {f1:.3f} > 0.95 (saturated)"
    return True, f"probe F1 = {f1:.3f}"


# ============================================================================
# DIRECTION COMPUTATION
# ============================================================================

def probe_normal_direction(probe):
    """Effective unit-norm probe direction on raw activations."""
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def diff_in_means_direction(X_insuff, X_suff):
    v = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
    norm = np.linalg.norm(v)
    return (v / norm).astype(np.float32), float(norm)


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)


def gram_schmidt(V):
    """Orthonormalize rows of V via Gram-Schmidt. V: (K, d)."""
    V = V.astype(np.float64).copy()
    K = V.shape[0]
    for i in range(K):
        for j in range(i):
            V[i] -= np.dot(V[i], V[j]) * V[j]
        V[i] /= np.linalg.norm(V[i])
    return V.astype(np.float32)


def build_rank_k_probe_basis(X_train, y_train, K=16):
    """
    Train probes iteratively, ablating each successive probe direction from the
    training data before retraining the next. Returns orthonormalized (K, d) basis.

    Note: probes are trained on the SAME (X_train, y_train) used for the unified probe,
    but with successive directions projected out.
    """
    X = X_train.astype(np.float32).copy()
    basis = []
    for k in range(K):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight='balanced',
                               max_iter=2000, solver='lbfgs')
        )
        probe.fit(X, y_train)
        scaler = probe.named_steps['standardscaler']
        clf = probe.named_steps['logisticregression']
        W = clf.coef_[0] / scaler.scale_
        v = (W / np.linalg.norm(W)).astype(np.float32)
        basis.append(v)
        # Project out v from training data
        proj = X @ v
        X = X - np.outer(proj, v)
    B = np.stack(basis, axis=0)
    return gram_schmidt(B)  # ensure clean orthonormality


# ============================================================================
# HOOKS
# ============================================================================

class AdditionHook:
    """
    Add alpha * v to every token position at the hooked layer. Matches
    exp13_steering_new.py exactly. The earlier position-restricted variant
    failed because HF generate() with SDPA + KV cache does not produce
    shape[1]==1 outputs at intermediate layers in recent transformers
    versions, so the restriction silently zeroed out the intervention.
    """
    def __init__(self, v, alpha):
        self.v = v
        self.alpha = float(alpha)
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = self.v.to(device=hs.device, dtype=hs.dtype)
        modified = hs + self.alpha * self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


class SubspaceAblationHook:
    """
    Project out the K-dim subspace spanned by orthonormal V (K, d) from the
    residual stream at every position.
    """
    def __init__(self, V_basis):
        self.V = V_basis  # numpy (K, d), orthonormal rows
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = torch.tensor(self.V, dtype=hs.dtype, device=hs.device)
        # hs shape: (B, T, d); V shape: (K, d)
        proj_coef = hs @ self._cached.T          # (B, T, K)
        proj = proj_coef @ self._cached          # (B, T, d)
        modified = hs - proj
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


def get_layer_modules(model):
    """
    Locate the ModuleList of decoder layers. Handles Llama / Qwen / Gemma2 /
    Gemma3 (multimodal-wrapped) / GPT-style / others via explicit paths plus a
    recursive ModuleList-search fallback.
    """
    import torch.nn as nn
    explicit_paths = [
        lambda m: m.model.layers,                           # Llama, Qwen, Gemma2
        lambda m: m.layers,
        lambda m: m.transformer.h,                          # GPT-2 family
        lambda m: m.transformer.layers,
        lambda m: m.gpt_neox.layers,
        lambda m: m.model.language_model.model.layers,      # Gemma3 multimodal
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
    ]
    for fn in explicit_paths:
        try:
            layers = fn(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    # Fallback: longest ModuleList in the module tree is almost always the decoder stack
    best, best_name = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 5:
            if best is None or len(mod) > len(best):
                best, best_name = mod, name
    if best is not None:
        print(f"  [layers] fallback located decoder stack at '{best_name}' (n={len(best)})")
        return best
    raise ValueError("Cannot locate transformer layers")


def register_addition_hook(layer_module, v_tensor, alpha):
    return [layer_module.register_forward_hook(
        AdditionHook(v_tensor, alpha))]


def register_ablation_hooks(layer_modules, V_basis):
    handles = []
    for lm in layer_modules:
        handles.append(lm.register_forward_hook(SubspaceAblationHook(V_basis)))
    return handles


def remove_handles(handles):
    for h in handles:
        h.remove()


# ============================================================================
# RESIDUAL NORM STATISTICS (for magnitude normalization)
# ============================================================================

@torch.no_grad()
def compute_layer_residual_stats(model, tokenizer, prompts, target_layer, batch_size=4):
    """
    Compute mean residual norm at target_layer over a sample of prompts.
    Used to normalize steering alpha to "units of residual std."
    """
    norms = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[target_layer]  # (B, T, d)
        # Per-token L2 norm
        per_token_norms = h.norm(dim=-1).flatten().to(torch.float32).cpu().numpy()
        norms.extend(per_token_norms.tolist())
        del out
    return float(np.mean(norms)), float(np.std(norms))


# ============================================================================
# CUTOFF UTILITIES (from exp14)
# ============================================================================

@dataclass
class CutoffResult:
    text: str
    actual_pct: float
    token_count: int
    total_tokens: int


def _token_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)['input_ids']


def _sentence_boundary_candidates(text):
    candidates = []
    for m in re.finditer(r'\n{2,}', text):
        candidates.append((m.end(), 0))
    for m in re.finditer(r'[.!?](?:\s+)(?=[A-Z]|$)', text):
        candidates.append((m.end(), 1))
    for m in re.finditer(r'[.!?]\n', text):
        candidates.append((m.end(), 2))
    for m in re.finditer(r'[.!?]', text):
        candidates.append((m.end(), 3))
    best = {}
    for pos, prio in candidates:
        if pos not in best or prio < best[pos]:
            best[pos] = prio
    return [(pos, prio) for pos, prio in best.items()]


def find_cutoff(text, target_pct, tokenizer):
    ids = _token_ids(tokenizer, text)
    total = len(ids)
    if total == 0:
        return CutoffResult('', 0.0, 0, 0)
    if target_pct <= 0.0:
        return CutoffResult('', 0.0, 0, total)
    target = max(1, min(total, int(target_pct * total)))
    if target_pct >= 1.0:
        return CutoffResult(text, 1.0, total, total)
    cands = _sentence_boundary_candidates(text)
    if not cands:
        truncated = tokenizer.decode(ids[:target], skip_special_tokens=True)
        return CutoffResult(truncated, target / total, target, total)
    scored = []
    for pos, prio in cands:
        ttext = text[:pos].rstrip()
        ttok = len(_token_ids(tokenizer, ttext))
        if ttok == 0:
            continue
        scored.append((abs(ttok - target), prio, ttok, pos))
    if not scored:
        truncated = tokenizer.decode(ids[:target], skip_special_tokens=True)
        return CutoffResult(truncated, target / total, target, total)
    _, _, actual, char_pos = min(scored, key=lambda x: (x[0], x[1]))
    return CutoffResult(text[:char_pos].rstrip(), actual / total, actual, total)


# ============================================================================
# FORCED-DECODE GENERATION
# ============================================================================

class StopOnCloseBrace(StoppingCriteria):
    def __init__(self, tokenizer, prompt_lens):
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens  # list of int, per-row prompt length
        self._done = [False] * len(prompt_lens)

    def __call__(self, input_ids, scores, **kwargs):
        for i, p_len in enumerate(self.prompt_lens):
            if self._done[i]:
                continue
            generated = input_ids[i, p_len:]
            if generated.numel() == 0:
                continue
            if '}' in self.tokenizer.decode(generated, skip_special_tokens=True):
                self._done[i] = True
        return all(self._done)


@torch.no_grad()
def batched_forced_decode(model, tokenizer, forced_inputs, batch_size=GEN_BATCH_SIZE,
                          max_new_tokens=MAX_NEW_TOKENS):
    """Greedy-decode each forced input with a } stopping criterion. Returns list of dicts."""
    results = [None] * len(forced_inputs)
    for i in range(0, len(forced_inputs), batch_size):
        batch_inputs = forced_inputs[i:i + batch_size]
        enc = tokenizer(batch_inputs, return_tensors='pt', padding=True,
                        truncation=False).to(model.device)
        prompt_lens = enc['attention_mask'].sum(dim=1).tolist()
        stop = StoppingCriteriaList([StopOnCloseBrace(tokenizer, prompt_lens)])
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=stop,
        )
        for b, p_len in enumerate(prompt_lens):
            cont = tokenizer.decode(out[b, p_len:], skip_special_tokens=True)
            close = cont.find('}')
            if close != -1:
                cont = cont[:close + 1]
            cont = cont.strip()
            raw = FORCE_DECODE_SUFFIX + cont
            results[i + b] = {'continuation': cont, 'raw_output': raw}
    return results


# ============================================================================
# ANSWER CLASSIFICATION
# ============================================================================

INSUFF_REGEX = re.compile(
    r'^\s*(insufficient|not\s+enough(?:\s+information)?|cannot\s+(?:be\s+)?determined|'
    r'undetermined|missing(?:\s+information)?|unknown|unsolvable|'
    r'no\s+unique\s+answer|cannot\s+determine|unable\s+to\s+determine|'
    r'n/?a|impossible|indeterminate)\s*$',
    re.IGNORECASE)
NUMERIC_REGEX = re.compile(r'[-+]?\d')
BOXED_REGEX = re.compile(r'\\boxed\s*\{([^{}]*)\}')


def extract_boxed(raw_output):
    m = BOXED_REGEX.search(raw_output)
    return m.group(1).strip() if m else None


def classify_answer(boxed):
    if boxed is None:
        return 'other'
    s = boxed.strip().strip('.').strip()
    if INSUFF_REGEX.match(s):
        return 'insufficient'
    if NUMERIC_REGEX.search(s):
        return 'numeric'
    return 'other'


def is_coherent(text, min_chars=10, max_repeat_ratio=0.5):
    s = text.strip()
    if len(s) < min_chars:
        return False
    words = s.split()
    if len(words) < 3:
        return False
    from collections import Counter
    c = Counter(words)
    return max(c.values()) / len(words) < max_repeat_ratio


def is_verbalize_correct(boxed, is_sufficient):
    kind = classify_answer(boxed)
    if is_sufficient:
        return kind == 'numeric'
    return kind == 'insufficient'


# ============================================================================
# T=0 HIDDEN STATE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_t0_states(model, tokenizer, prompts, target_layer, batch_size=EXTRACT_BATCH_SIZE):
    states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        attn = enc['attention_mask']
        last_idx = (attn.sum(dim=1) - 1).tolist()
        h = out.hidden_states[target_layer]
        for b, idx in enumerate(last_idx):
            states.append(h[b, idx, :].to(torch.float32).cpu().numpy())
        del out
    return np.array(states)


# ============================================================================
# PROBE SANITY CHECK
# ============================================================================

def probe_sanity_under_intervention(model, tokenizer, probe, target_layer,
                                    prompts, labels, layer_modules,
                                    register_hooks_fn=None):
    """
    Extract t=0 states with hooks active, run probe, return F1.
    register_hooks_fn: callable that takes the model and returns a list of handles.
    """
    handles = register_hooks_fn(model) if register_hooks_fn else []
    try:
        X = extract_t0_states(model, tokenizer, prompts, target_layer)
    finally:
        remove_handles(handles)
    proba = probe.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    f1 = float(f1_score(labels, pred, zero_division=0))
    return {
        'probe_f1': round(f1, 4),
        'mean_p_insuff_on_insuff': round(float(proba[labels == 1].mean()) if (labels == 1).any() else float('nan'), 4),
        'mean_p_insuff_on_suff':   round(float(proba[labels == 0].mean()) if (labels == 0).any() else float('nan'), 4),
    }


# ============================================================================
# DATA LOADING (from exp2 generations + evaluations)
# ============================================================================

def load_evaluated_traces(model_slug, dataset):
    path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f).get('data', [])


def load_generations(model_slug, dataset):
    path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_train_balanced_prompts(model_slug, dataset, n_per_class):
    path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_train/math/{model_slug}/{dataset}_cot_generations.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    suff = [g['prompt'] for g in data if g.get('is_sufficient', True)][:n_per_class]
    insuff = [g['prompt'] for g in data if not g.get('is_sufficient', True)][:n_per_class]
    return suff, insuff


def get_q1_q3_samples(model_slug, dataset, n_q1, n_q3):
    """Returns (q1_items, q3_items): each item has prompt, cot, is_sufficient, sample_id."""
    eval_data = load_evaluated_traces(model_slug, dataset)
    gen_data = load_generations(model_slug, dataset)
    if eval_data is None or gen_data is None:
        return None, None
    q1, q3 = [], []
    for idx, (e, g) in enumerate(zip(eval_data, gen_data)):
        quad = e.get('epistemic_quadrant', '')
        item = {
            'sample_id': str(e.get('question_idx', e.get('sample_id', idx))),
            'question': g.get('question'),
            'prompt': g.get('prompt'),
            'cot': g.get('generated_response', g.get('model_output', '')),
            'is_sufficient': g.get('is_sufficient', True),
            'quadrant': quad,
        }
        if quad == 'Q1_Hallucination' and len(q1) < n_q1:
            q1.append(item)
        elif quad == 'Q3_Solved_Correctly' and len(q3) < n_q3:
            q3.append(item)
        if len(q1) >= n_q1 and len(q3) >= n_q3:
            break
    return q1, q3


# ============================================================================
# ARTIFACT SAVING (baseline only — for future DLA / patching / SAE)
# ============================================================================

@torch.no_grad()
def save_baseline_artifacts(model, tokenizer, samples_q1, samples_q3, paths, n_layers):
    """
    Save per-sample force-decode-position residuals, MLP and attention outputs,
    and top-50 logits — for baseline (no intervention).
    Caches at every cutoff.
    """
    art_dir = paths['artifacts']
    art_dir.mkdir(parents=True, exist_ok=True)

    layer_modules = get_layer_modules(model)
    captured = {'residual': [None] * n_layers,
                'mlp': [None] * n_layers,
                'attn': [None] * n_layers}

    def res_hook_factory(layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured['residual'][layer_idx] = hs[:, -1, :].to(torch.float16).cpu().numpy().copy()
        return hook

    def mlp_hook_factory(layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured['mlp'][layer_idx] = hs[:, -1, :].to(torch.float16).cpu().numpy().copy()
        return hook

    def attn_hook_factory(layer_idx):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured['attn'][layer_idx] = hs[:, -1, :].to(torch.float16).cpu().numpy().copy()
        return hook

    # Build hooks
    handles = []
    for i, lm in enumerate(layer_modules):
        handles.append(lm.register_forward_hook(res_hook_factory(i)))
        if hasattr(lm, 'mlp'):
            handles.append(lm.mlp.register_forward_hook(mlp_hook_factory(i)))
        if hasattr(lm, 'self_attn'):
            handles.append(lm.self_attn.register_forward_hook(attn_hook_factory(i)))
        elif hasattr(lm, 'attention'):
            handles.append(lm.attention.register_forward_hook(attn_hook_factory(i)))

    try:
        for quad_name, samples in [('Q1', samples_q1), ('Q3', samples_q3)]:
            for cutoff in CUTOFFS:
                cutoff_key = int(round(cutoff * 100))
                fp = art_dir / f'{quad_name}_cutoff{cutoff_key}_artifacts.npz'
                if fp.exists():
                    continue
                res_arr, mlp_arr, attn_arr, logits_arr, ids_arr = [], [], [], [], []
                sample_ids = []
                for s in tqdm(samples, desc=f'  artifacts {quad_name}/cut{cutoff_key}', leave=False):
                    # Construct forced input
                    if cutoff <= 0.0:
                        truncated = ''
                    else:
                        c = find_cutoff(s['cot'], cutoff, tokenizer)
                        truncated = c.text
                    forced = s['prompt'] + truncated + FORCE_DECODE_SUFFIX
                    enc = tokenizer(forced, return_tensors='pt').to(model.device)
                    out = model(**enc)
                    last_logits = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
                    top_idx = np.argsort(-last_logits)[:50]
                    top_logits = last_logits[top_idx]
                    res_arr.append(np.stack(captured['residual'], axis=0)[:, 0, :])
                    mlp_arr.append(np.stack([x if x is not None else np.zeros_like(captured['residual'][0])
                                             for x in captured['mlp']], axis=0)[:, 0, :])
                    attn_arr.append(np.stack([x if x is not None else np.zeros_like(captured['residual'][0])
                                              for x in captured['attn']], axis=0)[:, 0, :])
                    logits_arr.append(top_logits.astype(np.float16))
                    ids_arr.append(top_idx.astype(np.int32))
                    sample_ids.append(s['sample_id'])
                np.savez_compressed(
                    fp,
                    residual=np.stack(res_arr, axis=0),
                    mlp=np.stack(mlp_arr, axis=0),
                    attn=np.stack(attn_arr, axis=0),
                    top_logits=np.stack(logits_arr, axis=0),
                    top_token_ids=np.stack(ids_arr, axis=0),
                    sample_ids=np.array(sample_ids),
                )
    finally:
        remove_handles(handles)


# ============================================================================
# PER (MODEL, DATASET) RUN
# ============================================================================

def out_paths(output_base, slug, dataset):
    base = Path(output_base) / slug / dataset
    base.mkdir(parents=True, exist_ok=True)
    return {
        'base': base,
        'meta': base / 'meta.json',
        'probe_sanity': base / 'probe_sanity.csv',
        'summary': base / 'summary.csv',
        'sample_major': base / 'sample_major.json',
        'probes_rank': base / 'probes_rank',
        'artifacts': base / 'artifacts',
    }


def run_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    paths = out_paths(args.output_dir, slug, dataset)
    paths['probes_rank'].mkdir(parents=True, exist_ok=True)

    # 0) Fast-skip if already complete
    done_marker = paths['base'] / 'DONE'
    if done_marker.exists() and not args.force:
        print(f"[DONE] {slug}/{dataset} — already complete, skipping")
        return

    # 1) Quality filter
    exp10_csv = os.path.join(EXP10_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    ok, reason = passes_quality_filter(slug, dataset, exp10_csv)
    if not ok:
        print(f"[SKIP] {slug}/{dataset}: {reason}")
        with open(paths['base'] / 'SKIPPED.txt', 'w') as f:
            f.write(reason + '\n')
        return
    print(f"[RUN]  {slug}/{dataset}: {reason}")

    # 2) Load probe + best layer
    df = pd.read_csv(exp10_csv)
    best_layer = int(df[(df['Model'] == slug) & (df['Percentage'] == '0%')]['Optimal_Layer'].iloc[0])
    probe_path = os.path.join(EXP10_DIR, 'probes_proportional', dataset, slug,
                              f'unified_probe_layer{best_layer}.joblib')
    probe = joblib.load(probe_path)
    v_probe = probe_normal_direction(probe)
    hidden_dim = v_probe.shape[0]

    # 3) Load samples
    q1_samples, q3_samples = get_q1_q3_samples(slug, dataset, args.n_q1, args.n_q3)
    if q1_samples is None or len(q1_samples) == 0:
        print(f"  ! no Q1 samples — skipping"); return

    # 4) Load model
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

    # 5) Compute residual scale at best layer (for alpha normalization)
    # We normalize alpha by mean(||h||) / sqrt(d): the typical per-dimension magnitude
    # of residual activations. alpha=1 then means "add a vector of norm equal to one
    # standard per-dimension activation magnitude," comparable across models.
    suff_pmpts, insuff_pmpts = load_train_balanced_prompts(slug, dataset, args.n_dim)
    if suff_pmpts is None or len(suff_pmpts) < 50 or len(insuff_pmpts) < 50:
        print("  ! insufficient train data for DIM — using test prompts as fallback")
        suff_pmpts = [s['prompt'] for s in q3_samples]
        insuff_pmpts = [s['prompt'] for s in q1_samples]
    norm_pool = (suff_pmpts + insuff_pmpts)[:200]
    res_mean_norm, res_std_norm = compute_layer_residual_stats(
        model, tok, norm_pool, best_layer, batch_size=EXTRACT_BATCH_SIZE)
    alpha_scale = res_mean_norm / np.sqrt(hidden_dim)  # typical per-dim magnitude

    # 6) Compute DIM, random
    print(f"  Extracting t=0 states for DIM (n={len(suff_pmpts)+len(insuff_pmpts)})...")
    X_suff = extract_t0_states(model, tok, suff_pmpts, best_layer)
    X_insuff = extract_t0_states(model, tok, insuff_pmpts, best_layer)
    v_dim, dim_mag = diff_in_means_direction(X_insuff, X_suff)
    v_rand = random_direction(hidden_dim, seed=42)

    # 7) Build rank-K probe basis
    print(f"  Training rank-K probe basis (K={args.rank_K_max})...")
    X_train = np.concatenate([X_suff, X_insuff], axis=0)
    y_train = np.concatenate([np.zeros(len(X_suff)), np.ones(len(X_insuff))]).astype(int)
    rank_basis = build_rank_k_probe_basis(X_train, y_train, K=args.rank_K_max)
    # Random K basis (orthonormalized random rows)
    rand_basis_full = np.random.RandomState(123).randn(args.rank_K_max, hidden_dim).astype(np.float32)
    rand_basis_full = gram_schmidt(rand_basis_full)
    # DIM rank-K basis (DIM as first row, then iteratively retrained DIM after ablation)
    dim_basis_full = build_dim_basis_rank_k(X_train, y_train, args.rank_K_max)

    # Save probes (rank-K)
    for k in range(args.rank_K_max):
        np.save(paths['probes_rank'] / f'probe_v{k+1}.npy', rank_basis[k])

    # Save meta
    cos_pd = float(np.dot(v_probe, v_dim))
    cos_pr = float(np.dot(v_probe, v_rand))
    cos_dr = float(np.dot(v_dim, v_rand))
    meta = {
        'model': slug, 'dataset': dataset,
        'best_layer': best_layer, 'hidden_dim': hidden_dim, 'n_layers': n_layers,
        'n_q1': len(q1_samples), 'n_q3': len(q3_samples),
        'cos_probe_dim': cos_pd, 'cos_probe_rand': cos_pr, 'cos_dim_rand': cos_dr,
        'dim_magnitude': dim_mag,
        'residual_norm_mean_at_best_layer': res_mean_norm,
        'residual_norm_std_at_best_layer': res_std_norm,
        'alpha_units': 'typical-per-dim-residual-magnitude = mean(||h||) / sqrt(d)',
        'alpha_scaling_factor_in_raw_units': float(alpha_scale),
        'alphas_std_units': args.alphas,
        'rank_K_values': args.rank_K_values,
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    # 8) Pre-tokenize forced inputs once per (sample, cutoff)
    print("  Pre-computing cutoff truncations...")
    all_samples = q1_samples + q3_samples
    forced_inputs = {}  # (sample_id, cutoff_key) -> str
    for s in all_samples:
        for cutoff in CUTOFFS:
            ck = int(round(cutoff * 100))
            if cutoff <= 0.0:
                truncated = ''
            else:
                c = find_cutoff(s['cot'], cutoff, tok)
                if c.token_count < args.min_cutoff_tokens and cutoff > 0.0:
                    truncated = c.text  # keep, will skip later if too short
                else:
                    truncated = c.text
            forced_inputs[(s['sample_id'], ck)] = s['prompt'] + truncated + FORCE_DECODE_SUFFIX

    # 9) Save baseline artifacts BEFORE running interventions
    if not args.skip_artifacts:
        print("  Saving baseline residual stream / MLP / attn / top-K logit artifacts...")
        save_baseline_artifacts(model, tok, q1_samples, q3_samples, paths, n_layers)

    # 10) Build sanity-probe prompts (used to compute probe F1 under each intervention)
    sanity_n = min(args.probe_sanity_n, len(suff_pmpts) + len(insuff_pmpts))
    sanity_prompts = (insuff_pmpts[:sanity_n // 2] + suff_pmpts[:sanity_n // 2])
    sanity_labels = np.array([1] * (sanity_n // 2) + [0] * (sanity_n // 2))

    # 11) Build the full condition list
    conditions = build_conditions(
        target_module, layer_modules,
        v_probe, v_dim, v_rand,
        rank_basis, rand_basis_full, dim_basis_full,
        alpha_scale,
        args
    )

    # 12) Run each condition
    sample_records = load_existing_records(paths['sample_major'])
    by_idx = {r['sample_id']: r for r in sample_records}
    sanity_rows = load_existing_csv(paths['probe_sanity'])

    for cond_name, register_fn in tqdm(conditions, desc='  conditions'):
        if cond_name in {r['condition'] for r in sanity_rows}:
            # already done — check if we also have sample results
            need = [s for s in all_samples
                    if not record_has_all_cutoffs(by_idx.get(s['sample_id']), cond_name)]
            if not need:
                continue

        # Probe sanity
        sanity = probe_sanity_under_intervention(
            model, tok, probe, best_layer,
            sanity_prompts, sanity_labels, layer_modules,
            register_hooks_fn=register_fn,
        )
        sanity_row = {'condition': cond_name, 'model': slug, 'dataset': dataset,
                      'n': sanity_n, **sanity}
        sanity_rows = [r for r in sanity_rows if r['condition'] != cond_name]
        sanity_rows.append(sanity_row)
        save_csv(paths['probe_sanity'], sanity_rows)

        # Generate
        handles = register_fn(model)
        try:
            # Group by cutoff so we batch sensibly
            for cutoff in CUTOFFS:
                ck = int(round(cutoff * 100))
                # Filter out samples that already have this (cond, cutoff) AND short-cutoff
                pending = []
                for s in all_samples:
                    rec = by_idx.get(s['sample_id'])
                    if rec is not None:
                        if rec.get('cutoffs', {}).get(str(ck), {}).get(cond_name) is not None:
                            continue
                    pending.append(s)

                if not pending:
                    continue

                # Batched forced decode
                batch_inputs = [forced_inputs[(s['sample_id'], ck)] for s in pending]
                gen_results = batched_forced_decode(
                    model, tok, batch_inputs,
                    batch_size=args.gen_batch_size,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                for s, gr in zip(pending, gen_results):
                    boxed = extract_boxed(gr['raw_output'])
                    kind = classify_answer(boxed)
                    coh = is_coherent(gr['continuation'])
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

        # Periodic save after each condition
        with open(paths['sample_major'], 'w') as f:
            json.dump(sample_records, f, indent=2)
        aggregate_summary(sample_records, slug, dataset, paths['summary'])

    # 13) Final aggregate + DONE marker
    aggregate_summary(sample_records, slug, dataset, paths['summary'])
    done_marker.touch()
    print(f"  -> done. Summary at {paths['summary']}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# CONDITION BUILDER
# ============================================================================

def build_conditions(target_module, layer_modules,
                     v_probe, v_dim, v_rand,
                     rank_basis, rand_basis, dim_basis,
                     res_std, args):
    """
    Returns list of (name, register_fn). register_fn takes model, returns handles.
    Condition names:
      baseline
      steer_probe_aSTD          (S in args.alphas)
      steer_dim_aSTD
      steer_rand_aSTD
      ablate_probe_K{K}
      ablate_dim_K{K}
      ablate_rand_K{K}
    """
    conditions = []

    # Baseline
    conditions.append(('baseline', lambda m: []))

    # Steering: add α·v at best layer, position-restricted (only during generation)
    if not args.skip_steering:
        for v, name in [(v_probe, 'probe'), (v_dim, 'dim'), (v_rand, 'rand')]:
            for a in args.alphas:
                a_raw = a * res_std  # alpha in std units → raw scalar
                v_t = torch.tensor(v, dtype=torch.float32)
                def make_fn(v_t=v_t, a_raw=a_raw, target=target_module):
                    return lambda m: register_addition_hook(target, v_t, a_raw)
                conditions.append((f'steer_{name}_a{a}', make_fn()))

    # Rank-K ablation: project out K-dim subspace at every layer
    if not args.skip_ablation:
        for K in args.rank_K_values:
            for basis, name in [(rank_basis[:K], 'probe'),
                                (dim_basis[:K], 'dim'),
                                (rand_basis[:K], 'rand')]:
                def make_fn(B=basis.copy(), lms=layer_modules):
                    return lambda m: register_ablation_hooks(lms, B)
                conditions.append((f'ablate_{name}_K{K}', make_fn()))

    return conditions


def build_dim_basis_rank_k(X_train, y_train, K):
    """Iteratively compute DIM directions, ablating each from training data."""
    X = X_train.astype(np.float32).copy()
    suff_mask = y_train == 0
    insuff_mask = y_train == 1
    basis = []
    for k in range(K):
        v = X[insuff_mask].mean(axis=0) - X[suff_mask].mean(axis=0)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            # degenerate — pad with random
            v = np.random.RandomState(999 + k).randn(X.shape[1]).astype(np.float32)
            norm = np.linalg.norm(v)
        v = v / norm
        basis.append(v)
        proj = X @ v
        X = X - np.outer(proj, v)
    B = np.stack(basis, axis=0)
    return gram_schmidt(B)


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_summary(sample_records, slug, dataset, out_path):
    rows = []
    # Per (condition, cutoff_pct, quadrant): rate of insufficient claims, numeric, verbalize_correct, coherent
    groups = {}
    for rec in sample_records:
        for cutoff_key, by_cond in rec.get('cutoffs', {}).items():
            for cond_name, v in by_cond.items():
                key = (cond_name, int(cutoff_key), rec.get('quadrant', ''))
                groups.setdefault(key, []).append(v)

    for (cond, ck, quad), vs in sorted(groups.items()):
        n = len(vs)
        if n == 0:
            continue
        ins = sum(1 for v in vs if v['answer_kind'] == 'insufficient') / n
        num = sum(1 for v in vs if v['answer_kind'] == 'numeric') / n
        coh = sum(1 for v in vs if v.get('coherent', False)) / n
        vc = sum(1 for v in vs if v.get('verbalize_correct', False)) / n
        rows.append({
            'model': slug, 'dataset': dataset, 'condition': cond,
            'cutoff_pct': ck, 'quadrant': quad, 'n': n,
            'insuff_rate': round(ins, 4),
            'numeric_rate': round(num, 4),
            'coherence_rate': round(coh, 4),
            'verbalize_acc': round(vc, 4),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================================
# RESUME / IO HELPERS
# ============================================================================

def load_existing_records(path):
    if not Path(path).exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def load_existing_csv(path):
    if not Path(path).exists():
        return []
    return pd.read_csv(path).to_dict('records')


def save_csv(path, rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def record_has_all_cutoffs(rec, cond_name):
    if rec is None:
        return False
    cs = rec.get('cutoffs', {})
    for cutoff in CUTOFFS:
        ck = str(int(round(cutoff * 100)))
        if cs.get(ck, {}).get(cond_name) is None:
            return False
    return True


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
    p.add_argument('--n_q3', type=int, default=N_Q3)
    p.add_argument('--n_dim', type=int, default=N_DIM_PER_CLASS)
    p.add_argument('--probe_sanity_n', type=int, default=PROBE_SANITY_N)
    p.add_argument('--alphas', type=float, nargs='+', default=ALPHAS_STD_UNITS)
    p.add_argument('--rank_K_values', type=int, nargs='+', default=RANK_K_VALUES)
    p.add_argument('--rank_K_max', type=int, default=16)
    p.add_argument('--min_cutoff_tokens', type=int, default=20)
    p.add_argument('--gen_batch_size', type=int, default=GEN_BATCH_SIZE)
    p.add_argument('--skip_steering', action='store_true')
    p.add_argument('--skip_ablation', action='store_true')
    p.add_argument('--skip_artifacts', action='store_true',
                   help='Skip saving residual / MLP / attn artifacts (saves time + disk)')
    p.add_argument('--force', action='store_true',
                   help='Recompute even if DONE marker exists')
    p.add_argument('--test', action='store_true',
                   help='Smoke test: tiny n, K=1, only baseline + 1 condition each')
    args = p.parse_args()

    if args.test:
        args.n_q1 = 5; args.n_q3 = 5; args.n_dim = 20; args.probe_sanity_n = 20
        args.alphas = [1.0]; args.rank_K_values = [1]; args.rank_K_max = 1

    # Resolve model list
    if args.all_models:
        models = DEFAULT_MODELS
    elif args.model:
        models = [args.model]
    else:
        raise SystemExit("Specify --model or --all_models")

    # Resolve dataset list
    if args.all_datasets:
        datasets = ['umwp', 'treecut']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Specify --dataset or --all_datasets")

    # Run
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for m in models:
        for ds in datasets:
            t0 = time.time()
            try:
                run_one(m, ds, args)
                print(f"  [{m}/{ds}] completed in {(time.time()-t0)/60:.1f} min")
            except Exception as e:
                print(f"\n[ERROR] {m}/{ds}: {e}")
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
