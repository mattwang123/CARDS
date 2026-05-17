"""
================================================================================
final_layer_probe.py  — EXP10-CONSISTENT VERSION
================================================================================
Identical to exp10's train protocol in every numerical-effecting way:
  - probe: StandardScaler + LogisticRegression(max_iter=1000,
    class_weight='balanced', C=1.0, solver='lbfgs', n_jobs=-1).
    No random_state (exp10 doesn't set it).
  - tokenizer: no pad_token / padding_side setup (exp10 doesn't set them;
    we never batch so they're never used anyway).
  - max_memory: GB units, not GiB (matches exp10's device_map='auto' budget).
  - data: same TRAIN_DIR / TEST_DIR, same JSON layout, same
    cot_len < 10 filter, same target_indices formula.

This script EXTENDS exp10 with two probes exp10 doesn't have, training each
with the exact same protocol exp10 uses:
  - probe_final_layer       — cutoff-concat probe at L (final layer).
  - probe_boxed_at_best     — probe at boxed position, layer L*.
  - probe_boxed_at_final    — probe at boxed position, layer L.

Plus Measurement A (Δu margin) for v_probe at each, and verbal accuracy at
the actual decision token (read from exp2's cached sampled generation).

USAGE matches exp10's per-dataset pattern:
  python src/final_layer_probe.py --dataset umwp --all_models
  python src/final_layer_probe.py --dataset treecut --all_models

  # smoke
  python src/final_layer_probe.py --smoke

  # rebuild master CSV from existing per-pair outputs
  python src/final_layer_probe.py --aggregate_only

  # if boxed-position outputs were generated with the previous buggy
  # make_boxed_input, use:
  python src/final_layer_probe.py --boxed_only --all_models --dataset umwp
================================================================================
"""

import argparse
import gc
import json
import os
import re
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


# ============================================================================
# CONFIG — match exp10 exactly
# ============================================================================

PERCENTAGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

SOURCE_BASE = '/export/fs06/hwang302/CARDS'
TRAIN_DIR = os.path.join(SOURCE_BASE, 'experiments/dynamic_tracking_train')
TEST_DIR  = os.path.join(SOURCE_BASE, 'experiments/dynamic_tracking_test')
EXP10_DIR = os.path.join(SOURCE_BASE, 'exp_temporal_new')
EMB_BASE  = os.path.join(EXP10_DIR, 'embeddings_proportional')
OUTPUT_BASE = os.path.join(EXP10_DIR, 'final_layer_probe')
MASTER_CSV_DIR = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results/_final_layer_extension'

BOXED_SUFFIX = "\\boxed{"

# Identical to exp10's MODELS
FULL_MODELS = [
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
SMOKE_MODELS = ['Qwen/Qwen2.5-Math-1.5B-Instruct', 'google/gemma-3-12b-it']

ABSTENTION_PHRASES = [
    "Insufficient", "Cannot", "Unable", "Unknown", "Indeterminate",
    "Impossible", "Undefined", "Undetermined", "Missing", "Not",
    "Ins", "Insuf",
]


# ============================================================================
# PROBE — IDENTICAL to exp10's train_probe()
# ============================================================================

def make_probe():
    """EXACTLY matches exp10's train_probe(). No random_state — exp10
    doesn't set it. lbfgs is deterministic so this doesn't affect numerics
    in practice; we drop the seed param for strict literal alignment."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced',
                           C=1.0, solver='lbfgs', n_jobs=-1),
    )


def probe_normal_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    n = np.linalg.norm(W)
    if n < 1e-12:
        return np.zeros_like(W, dtype=np.float32)
    return (W / n).astype(np.float32)


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)


# ============================================================================
# DATA LOADING — IDENTICAL to exp10's load_exp2_data
# ============================================================================

def load_exp2_data(model_name, dataset, split_dir):
    """Returns (data, labels). Label 1 = insufficient. Matches exp10."""
    slug = model_name.split('/')[-1]
    path = f"{split_dir}/math/{slug}/{dataset}_cot_generations.json"
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    labels = np.array([1 if not d.get('is_sufficient', True) else 0 for d in data])
    return data, labels


def load_exp10_cached_labels(slug, dataset):
    """Load y_train.npy / y_test.npy from exp10's cache for sanity comparison."""
    base = Path(EMB_BASE) / dataset / slug
    if not (base / 'y_train.npy').exists() or not (base / 'y_test.npy').exists():
        return None, None
    return np.load(base / 'y_train.npy'), np.load(base / 'y_test.npy')


def load_exp10_best_probe_direction(slug, dataset, best_layer):
    """Load v_probe from exp10's saved best-layer probe."""
    probe_path = (Path(EXP10_DIR) / 'probes_proportional' / dataset / slug
                  / f'unified_probe_layer{best_layer}.joblib')
    if not probe_path.exists():
        return None
    probe = joblib.load(probe_path)
    return probe_normal_direction(probe)


# ============================================================================
# MODEL LOADING — IDENTICAL to exp10
# ============================================================================

def load_model(model_name):
    """Matches exp10's loader, with two robustness tweaks added 2025-05:
      - dtype= (torch_dtype deprecated since transformers 4.46)
      - attn_implementation='eager' for gemma family. Multi-GPU device_map
        + sdpa on gemma-3 hybrid attention (sliding window + global) causes
        sticky CUDA context failures (cudaErrorLaunchFailure) that poison
        the process. Eager is ~2x slower per forward but stable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    memory_map = {0: "65GB"} if num_gpus > 0 else None
    if num_gpus > 1:
        for i in range(1, num_gpus):
            memory_map[i] = "78GB"

    attn_impl = 'eager' if 'gemma' in model_name.lower() else 'sdpa'

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", max_memory=memory_map,
        dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model, tokenizer


# ============================================================================
# HIDDEN STATE EXTRACTION — replicates exp10's protocol, subset of layers
# ============================================================================

@torch.no_grad()
def detect_num_hidden_outputs(model, tokenizer):
    enc = tokenizer("Hello.", return_tensors='pt').to(model.device)
    out = model(**enc, output_hidden_states=True)
    n = len(out.hidden_states)
    del out
    torch.cuda.empty_cache()
    return n


@torch.no_grad()
def extract_cutoffs_at_layers(model, tokenizer, data, labels, layers, desc=""):
    """Extract residual hidden state at each of the 6 PERCENTAGES cutoffs at
    each layer in `layers`. Drops samples with CoT length < 10 — IDENTICAL
    filter to exp10. IDENTICAL tokenize protocol, IDENTICAL target_indices
    formula. Returns:
      X:      [N_valid, n_pct, n_layers, D]  fp32
      y:      [N_valid]
      valid_indices: original-data indices of the kept samples
    """
    extracted = []
    valid_labels = []
    valid_idx = []
    layers = list(layers)
    for i, (item, label) in enumerate(tqdm(zip(data, labels),
                                            total=len(data), desc=desc)):
        prompt_text = item['prompt']
        gen_text = item.get('generated_response', '')
        # exp10's tokenization protocol (separate prompt / full)
        p_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'][0]
        full_ids = tokenizer(prompt_text + gen_text, return_tensors='pt')['input_ids'][0]
        p_len = p_ids.shape[0]
        total_len = full_ids.shape[0]
        cot_len = total_len - p_len
        if cot_len < 10:                                # exp10 filter
            continue
        target_indices = []
        for pct in PERCENTAGES:                          # exp10 formula
            idx = p_len + int(pct * cot_len) - (1 if pct == 1.0 else 0)
            target_indices.append(min(idx, total_len - 1))

        inputs = tokenizer(prompt_text + gen_text, return_tensors='pt').to(model.device)
        out = model(**inputs, output_hidden_states=True)
        per_layer = []
        for l in layers:
            h = out.hidden_states[l][0, target_indices, :].to(torch.float32).cpu()
            per_layer.append(h)
        stacked = torch.stack(per_layer, dim=0)         # [n_layers, n_pct, D]
        stacked = stacked.permute(1, 0, 2).numpy()      # [n_pct, n_layers, D]
        extracted.append(stacked)
        valid_labels.append(label)
        valid_idx.append(i)
        del out, inputs
        torch.cuda.empty_cache()
    X = (np.array(extracted, dtype=np.float32)
         if extracted else np.zeros((0, 0, 0, 0), dtype=np.float32))
    y = np.array(valid_labels)
    return X, y, valid_idx


def derive_valid_idx(data, labels, tokenizer, min_cot_len=10, desc='deriving valid_idx'):
    """Replay extract_cutoffs_at_layers's sample filter WITHOUT model forward,
    so we can recover which samples were kept by a previous run. Used by
    --boxed_only."""
    valid_idx = []
    kept_labels = []
    for i, (item, label) in enumerate(tqdm(zip(data, labels),
                                            total=len(data), desc=desc)):
        prompt_text = item['prompt']
        gen_text = item.get('generated_response', '')
        p_ids = tokenizer(prompt_text, return_tensors='pt')['input_ids'][0]
        full_ids = tokenizer(prompt_text + gen_text, return_tensors='pt')['input_ids'][0]
        cot_len = full_ids.shape[0] - p_ids.shape[0]
        if cot_len < min_cot_len:
            continue
        valid_idx.append(i)
        kept_labels.append(label)
    return valid_idx, np.array(kept_labels)


def make_boxed_input(prompt, generated_response):
    """Truncate generated_response at its first \\boxed{ and append our
    BOXED_SUFFIX, so the model is positioned at the decision token as if
    it were about to emit its answer for the first time.

    If no \\boxed{ is present (model never used LaTeX boxed), fall back
    to appending BOXED_SUFFIX to the full CoT; boxed state is then
    artificial for these samples (counted in meta as n_no_boxed)."""
    idx = generated_response.find(BOXED_SUFFIX)
    cot = generated_response if idx == -1 else generated_response[:idx]
    return prompt + cot + BOXED_SUFFIX


@torch.no_grad()
def extract_boxed_position_at_layers(model, tokenizer, data, valid_idx,
                                      layers, desc=""):
    """For each sample, build input = prompt + (CoT truncated at first
    \\boxed{) + BOXED_SUFFIX, forward pass, take last-token hidden state at
    each layer. Single-sample (no batching), so no padding setup needed.
    Returns [N_valid, n_layers, D] fp32."""
    layers = list(layers)
    out_states = []
    for i in tqdm(valid_idx, desc=desc):
        item = data[i]
        text = make_boxed_input(item['prompt'], item.get('generated_response', ''))
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        out = model(**inputs, output_hidden_states=True)
        seq_len = inputs['input_ids'].shape[1]
        per_layer = []
        for l in layers:
            h = out.hidden_states[l][0, seq_len - 1, :].to(torch.float32).cpu()
            per_layer.append(h)
        out_states.append(torch.stack(per_layer, dim=0).numpy())   # [n_layers, D]
        del out, inputs
        torch.cuda.empty_cache()
    return np.array(out_states, dtype=np.float32)


def verbalize_at_boxed(tokenizer, data, valid_idx, desc=""):
    """Read the model's ACTUAL decision token from exp2's cached
    generated_response. Find first \\boxed{ and the first token after.
    Classify as 'abstain' / 'numeric' / 'other' / 'no_boxed'."""
    abstain_ids = set(build_abstention_token_ids(tokenizer))
    numeric_ids = set(build_numeric_token_ids(tokenizer))
    rows = []
    for i in tqdm(valid_idx, desc=desc):
        item = data[i]
        gr = item.get('generated_response', '')
        idx = gr.find(BOXED_SUFFIX)
        if idx == -1:
            next_tid = -1
            kind = 'no_boxed'
            tok_str = ''
        else:
            after = gr[idx + len(BOXED_SUFFIX):]
            tids = tokenizer.encode(after, add_special_tokens=False) if after else []
            if not tids:
                next_tid = -1
                kind = 'other'
                tok_str = ''
            else:
                next_tid = int(tids[0])
                tok_str = tokenizer.decode([next_tid])
                if next_tid in abstain_ids:
                    kind = 'abstain'
                elif next_tid in numeric_ids:
                    kind = 'numeric'
                else:
                    kind = 'other'
        rows.append({
            'sample_idx': i,
            'is_sufficient': item.get('is_sufficient', True),
            'predicted_token_id': next_tid,
            'predicted_token': tok_str,
            'kind': kind,
        })
    return rows


# ============================================================================
# PROBE TRAINING — IDENTICAL protocol to exp10's unified-probe path
# ============================================================================

def train_cutoff_concat_probe(X_train, y_train, X_test, y_test):
    """Train one probe by flattening the (sample, cutoff) axis. EXACTLY
    matches exp10's unified-probe training (reshape + np.repeat + same
    sklearn pipeline). X_*: [N, 6_pct, D]."""
    N_tr, n_pct, D = X_train.shape
    Xtr = X_train.reshape(-1, D)
    ytr = np.repeat(y_train, n_pct)
    N_te = X_test.shape[0]
    Xte = X_test.reshape(-1, D)
    yte = np.repeat(y_test, n_pct)

    probe = make_probe()
    probe.fit(Xtr, ytr)
    concat_train_f1 = float(f1_score(ytr, probe.predict(Xtr)))
    concat_test_f1  = float(f1_score(yte, probe.predict(Xte)))
    per_cutoff_rows = []
    for pi, pct in enumerate(PERCENTAGES):
        Xtr_p = X_train[:, pi, :]
        Xte_p = X_test[:, pi, :]
        f1_tr = float(f1_score(y_train, probe.predict(Xtr_p)))
        f1_te = float(f1_score(y_test,  probe.predict(Xte_p)))
        per_cutoff_rows.append({
            'pct': f'{int(pct * 100)}%',
            'train_f1': round(f1_tr, 4),
            'test_f1':  round(f1_te, 4),
        })
    return probe, {
        'concat_train_f1': round(concat_train_f1, 4),
        'concat_test_f1':  round(concat_test_f1,  4),
    }, per_cutoff_rows


def train_pointwise_probe(X_train, y_train, X_test, y_test):
    """Train one probe on 2D X. X_*: [N, D]. Same sklearn pipeline as exp10."""
    probe = make_probe()
    probe.fit(X_train, y_train)
    f1_tr = float(f1_score(y_train, probe.predict(X_train)))
    f1_te = float(f1_score(y_test,  probe.predict(X_test)))
    return probe, {
        'train_f1': round(f1_tr, 4),
        'test_f1':  round(f1_te, 4),
    }


# ============================================================================
# Δu / MEASUREMENT A
# ============================================================================

def build_abstention_token_ids(tokenizer):
    ids = set()
    for phrase in ABSTENTION_PHRASES:
        for variant in [phrase, " " + phrase, phrase.lower(), " " + phrase.lower()]:
            tids = tokenizer.encode(variant, add_special_tokens=False)
            if tids:
                ids.add(tids[0])
    return sorted(ids)


def build_numeric_token_ids(tokenizer):
    ids = set()
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer)
    digit_starters = re.compile(r"^\s?[-+]?\.?\d")
    for tid in range(vocab_size):
        try:
            s = tokenizer.decode([tid])
        except Exception:
            continue
        if not s or len(s) > 5:
            continue
        if digit_starters.match(s):
            ids.add(tid)
    return sorted(ids)


def get_unembed_weight(model):
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head.weight
    if hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        return model.model.lm_head.weight
    raise RuntimeError("Could not locate lm_head")


def summarize_z(z, abs_ids, num_ids):
    z_abs = float(z[abs_ids].mean())
    z_num = float(z[num_ids].mean())
    ranks = (-z).argsort().argsort()
    return {
        'z_mean_abs': z_abs,
        'z_mean_num': z_num,
        'margin_abs_num': z_abs - z_num,
        'median_rank_abs_pct': float(np.median(ranks[abs_ids])) / z.shape[0],
        'median_rank_num_pct': float(np.median(ranks[num_ids])) / z.shape[0],
    }


def measurement_A(W_U_np, v, abs_ids, num_ids):
    z = W_U_np @ v
    return summarize_z(z, abs_ids, num_ids)


# ============================================================================
# PER-(MODEL, DATASET) PIPELINE
# ============================================================================

def process_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    pair_dir = Path(OUTPUT_BASE) / dataset / slug
    complete_marker = pair_dir / '_COMPLETE'
    if complete_marker.exists() and not args.force:
        print(f"[skip] {slug}/{dataset}: _COMPLETE marker present")
        return

    # 1) exp10 anchor
    exp10_csv = os.path.join(EXP10_DIR, 'results',
                              f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[skip] {slug}/{dataset}: no exp10 csv at {exp10_csv}")
        return
    df10 = pd.read_csv(exp10_csv)
    row10 = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row10.empty:
        print(f"[skip] {slug}/{dataset}: no exp10 entry for {slug}")
        return
    best_layer = int(row10['Optimal_Layer'].iloc[0])
    exp10_best_f1 = float(row10['Unified_Test_F1'].iloc[0])

    v_probe_best = load_exp10_best_probe_direction(slug, dataset, best_layer)

    # 2) data
    train_data, train_labels_raw = load_exp2_data(model_name, dataset, TRAIN_DIR)
    test_data,  test_labels_raw  = load_exp2_data(model_name, dataset, TEST_DIR)
    if train_data is None or test_data is None:
        print(f"[skip] {slug}/{dataset}: missing exp2 data")
        return

    # 3) load model — IDENTICAL to exp10
    print(f"\n[{slug}/{dataset}] loading model...")
    model = None
    tok = None
    try:
        model, tok = load_model(model_name)

        n_hidden_outputs = detect_num_hidden_outputs(model, tok)
        final_layer = n_hidden_outputs - 1
        print(f"[{slug}/{dataset}] best_layer={best_layer}, final_layer={final_layer}")

        # 4) extract final-layer hidden states at 6 cutoffs (train + test)
        print(f"[{slug}/{dataset}] extracting cutoff residuals at final layer...")
        X_train_cuts, y_train, train_valid = extract_cutoffs_at_layers(
            model, tok, train_data, train_labels_raw,
            layers=[final_layer], desc='train cutoffs')
        X_test_cuts,  y_test,  test_valid = extract_cutoffs_at_layers(
            model, tok, test_data, test_labels_raw,
            layers=[final_layer], desc='test cutoffs')
        if X_train_cuts.size == 0 or X_test_cuts.size == 0:
            print(f"[skip] {slug}/{dataset}: no valid samples after extraction")
            return

        X_train_final = X_train_cuts.squeeze(2)   # [N, 6, D]
        X_test_final  = X_test_cuts.squeeze(2)
        D = X_train_final.shape[-1]

        # Strict label check vs exp10 cache
        y_tr_exp10, y_te_exp10 = load_exp10_cached_labels(slug, dataset)
        if y_tr_exp10 is not None:
            if (len(y_tr_exp10) != len(y_train)
                    or not np.array_equal(y_tr_exp10, y_train)):
                print(f"  [WARN] {slug}/{dataset}: train labels differ from exp10 "
                      f"cache (ours={len(y_train)} vs exp10={len(y_tr_exp10)}).")
        if y_te_exp10 is not None:
            if (len(y_te_exp10) != len(y_test)
                    or not np.array_equal(y_te_exp10, y_test)):
                print(f"  [WARN] {slug}/{dataset}: test labels differ from exp10 "
                      f"cache.")

        # 5) extract boxed-position hidden states at both layers
        print(f"[{slug}/{dataset}] extracting boxed-position residuals at "
              f"L*={best_layer} and L={final_layer}...")
        X_train_box = extract_boxed_position_at_layers(
            model, tok, train_data, train_valid,
            layers=[best_layer, final_layer], desc='train boxed')
        X_test_box = extract_boxed_position_at_layers(
            model, tok, test_data, test_valid,
            layers=[best_layer, final_layer], desc='test boxed')
        X_train_boxed_best  = X_train_box[:, 0, :]
        X_train_boxed_final = X_train_box[:, 1, :]
        X_test_boxed_best   = X_test_box[:, 0, :]
        X_test_boxed_final  = X_test_box[:, 1, :]

        # 6) verbal accuracy at boxed
        print(f"[{slug}/{dataset}] parsing decision token from exp2 (test)...")
        verbal_rows = verbalize_at_boxed(tok, test_data, test_valid,
                                          desc='verbal@boxed')
        for r in verbal_rows:
            is_insuff = not r['is_sufficient']
            r['verbal_correct'] = int(
                (is_insuff and r['kind'] == 'abstain') or
                ((not is_insuff) and r['kind'] == 'numeric')
            )
        # Two versions: all samples vs only those where the model actually
        # emitted \boxed{ in sampling.
        n_no_boxed = sum(1 for r in verbal_rows if r['kind'] == 'no_boxed')
        boxed_only_rows = [r for r in verbal_rows if r['kind'] != 'no_boxed']
        verbal_acc_all = float(np.mean([r['verbal_correct'] for r in verbal_rows]))
        verbal_acc_boxed_only = (
            float(np.mean([r['verbal_correct'] for r in boxed_only_rows]))
            if boxed_only_rows else float('nan')
        )
        print(f"  verbal accuracy (all)        = {verbal_acc_all:.4f}")
        print(f"  verbal accuracy (boxed only) = {verbal_acc_boxed_only:.4f} "
              f"(n_no_boxed={n_no_boxed}/{len(verbal_rows)})")

        # 7) save embeddings — fp32 np.save (matches exp10)
        emb_dir = pair_dir / 'embeddings'
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / 'X_train_final.npy', X_train_final)
        np.save(emb_dir / 'X_test_final.npy',  X_test_final)
        np.save(emb_dir / 'X_train_boxed_best.npy',  X_train_boxed_best)
        np.save(emb_dir / 'X_test_boxed_best.npy',   X_test_boxed_best)
        np.save(emb_dir / 'X_train_boxed_final.npy', X_train_boxed_final)
        np.save(emb_dir / 'X_test_boxed_final.npy',  X_test_boxed_final)
        np.save(emb_dir / 'y_train.npy', y_train)
        np.save(emb_dir / 'y_test.npy',  y_test)

        # 8) train probes — same protocol as exp10
        print(f"[{slug}/{dataset}] training 3 probes...")
        probe_final, final_concat, final_per_cutoff = train_cutoff_concat_probe(
            X_train_final, y_train, X_test_final, y_test)
        probe_boxed_best,  boxed_best_f1  = train_pointwise_probe(
            X_train_boxed_best,  y_train, X_test_boxed_best,  y_test)
        probe_boxed_final, boxed_final_f1 = train_pointwise_probe(
            X_train_boxed_final, y_train, X_test_boxed_final, y_test)

        probes_dir = pair_dir / 'probes'
        probes_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(probe_final,       probes_dir / 'probe_final_layer.joblib')
        joblib.dump(probe_boxed_best,  probes_dir / 'probe_boxed_at_best.joblib')
        joblib.dump(probe_boxed_final, probes_dir / 'probe_boxed_at_final.joblib')

        v_probe_final       = probe_normal_direction(probe_final)
        v_probe_boxed_best  = probe_normal_direction(probe_boxed_best)
        v_probe_boxed_final = probe_normal_direction(probe_boxed_final)
        np.save(pair_dir / 'v_probe_final.npy',       v_probe_final)
        np.save(pair_dir / 'v_probe_boxed_best.npy',  v_probe_boxed_best)
        np.save(pair_dir / 'v_probe_boxed_final.npy', v_probe_boxed_final)

        # 9) Measurement A
        print(f"[{slug}/{dataset}] computing Measurement A...")
        W_U = get_unembed_weight(model).detach().to(torch.float32).cpu().numpy()
        abs_ids = build_abstention_token_ids(tok)
        num_ids = build_numeric_token_ids(tok)
        v_rand = random_direction(D, seed=42)

        candidates = []
        if v_probe_best is not None:
            candidates.append((v_probe_best, 'v_probe_best'))
        candidates += [
            (v_probe_final,       'v_probe_final'),
            (v_probe_boxed_best,  'v_probe_boxed_best'),
            (v_probe_boxed_final, 'v_probe_boxed_final'),
            (v_rand,              'v_random'),
        ]
        align_rows = []
        for v, name in candidates:
            s = measurement_A(W_U, v, abs_ids, num_ids)
            align_rows.append({
                'model': slug, 'dataset': dataset, 'direction': name,
                'margin_abs_num': s['margin_abs_num'],
                'z_mean_abs': s['z_mean_abs'], 'z_mean_num': s['z_mean_num'],
                'median_rank_abs_pct': s['median_rank_abs_pct'],
                'median_rank_num_pct': s['median_rank_num_pct'],
            })
        pd.DataFrame(align_rows).to_csv(pair_dir / 'alignment.csv', index=False)

        # 10) f1 summary
        f1_rows = []
        for r in final_per_cutoff:
            f1_rows.append({
                'model': slug, 'dataset': dataset,
                'probe_variant': 'final_layer_cutoff_concat',
                'layer': final_layer, 'pct': r['pct'],
                'train_f1': r['train_f1'], 'test_f1': r['test_f1'],
            })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'final_layer_cutoff_concat',
            'layer': final_layer, 'pct': 'concat',
            'train_f1': final_concat['concat_train_f1'],
            'test_f1':  final_concat['concat_test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'boxed_at_best',
            'layer': best_layer, 'pct': 'boxed',
            'train_f1': boxed_best_f1['train_f1'],
            'test_f1':  boxed_best_f1['test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'boxed_at_final',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': boxed_final_f1['train_f1'],
            'test_f1':  boxed_final_f1['test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'exp10_best_layer',
            'layer': best_layer, 'pct': '0%',
            'train_f1': float('nan'), 'test_f1': exp10_best_f1,
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'verbal@boxed_all',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': float('nan'),
            'test_f1':  round(verbal_acc_all, 4),
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'verbal@boxed_only',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': float('nan'),
            'test_f1':  round(verbal_acc_boxed_only, 4),
        })

        pd.DataFrame(f1_rows).to_csv(pair_dir / 'f1_summary.csv', index=False)
        pd.DataFrame(verbal_rows).to_csv(pair_dir / 'verbal_at_boxed.csv', index=False)

        # 11) headline gaps — two versions
        gap_final_all        = boxed_final_f1['test_f1'] - verbal_acc_all
        gap_final_boxed_only = boxed_final_f1['test_f1'] - verbal_acc_boxed_only
        gap_best_all         = boxed_best_f1['test_f1']  - verbal_acc_all
        gap_best_boxed_only  = boxed_best_f1['test_f1']  - verbal_acc_boxed_only

        # 12) sanity warnings
        warnings = []
        if final_concat['concat_test_f1'] < 0.5:
            warnings.append(f"final-layer concat F1 {final_concat['concat_test_f1']:.3f} "
                            f"< 0.5 (signal weak at L)")
        if abs(boxed_best_f1['test_f1'] - exp10_best_f1) > 0.2:
            warnings.append(f"boxed_at_best F1 ({boxed_best_f1['test_f1']:.3f}) "
                            f"differs from exp10 best-layer F1 ({exp10_best_f1:.3f}) "
                            f"by >0.2; position effect large")
        for w in warnings:
            print(f"  [WARN] {slug}/{dataset}: {w}")

        # 13) meta
        meta = {
            'model': slug, 'dataset': dataset,
            'best_layer': best_layer, 'final_layer': final_layer,
            'hidden_dim': int(D), 'vocab_size': int(W_U.shape[0]),
            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
            'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
            'exp10_anchor_f1': exp10_best_f1,
            'final_layer_concat_test_f1': final_concat['concat_test_f1'],
            'boxed_at_best_test_f1':  boxed_best_f1['test_f1'],
            'boxed_at_final_test_f1': boxed_final_f1['test_f1'],
            'verbal_accuracy_all_samples':  verbal_acc_all,
            'verbal_accuracy_boxed_only':   verbal_acc_boxed_only,
            'n_no_boxed_test': n_no_boxed,
            'frac_no_boxed_test': n_no_boxed / len(verbal_rows),
            'gap_at_boxed_final_all':         gap_final_all,
            'gap_at_boxed_final_boxed_only':  gap_final_boxed_only,
            'gap_at_boxed_best_all':          gap_best_all,
            'gap_at_boxed_best_boxed_only':   gap_best_boxed_only,
            'measurement_A_at_L_summary': {
                r['direction']: r['margin_abs_num'] for r in align_rows
            },
            'warnings': warnings,
        }
        with open(pair_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        complete_marker.touch()
        print(f"[{slug}/{dataset}] DONE  "
              f"(F1 final-concat={final_concat['concat_test_f1']:.3f}, "
              f"F1 boxed@L={boxed_final_f1['test_f1']:.3f}, "
              f"verbal_all={verbal_acc_all:.3f}, "
              f"verbal_boxed_only={verbal_acc_boxed_only:.3f})")
    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================================
# BOXED-ONLY RERUN
# ============================================================================

def process_one_boxed_only(model_name, dataset, args):
    """Redo only the boxed outputs for a pair previously completed with the
    buggy make_boxed_input. Reuses cached cutoff embeddings + probe_final +
    y labels. Strict label consistency check via np.array_equal."""
    slug = model_name.split('/')[-1]
    pair_dir = Path(OUTPUT_BASE) / dataset / slug
    if not (pair_dir / '_COMPLETE').exists():
        print(f"[skip] {slug}/{dataset}: no _COMPLETE — --boxed_only requires "
              f"a prior completed run")
        return
    emb_dir = pair_dir / 'embeddings'
    probes_dir = pair_dir / 'probes'
    required = ['X_train_final.npy', 'X_test_final.npy',
                'y_train.npy', 'y_test.npy']
    if not all((emb_dir / f).exists() for f in required):
        print(f"[skip] {slug}/{dataset}: missing cached cutoff embeddings")
        return
    if not (probes_dir / 'probe_final_layer.joblib').exists():
        print(f"[skip] {slug}/{dataset}: missing probe_final_layer.joblib")
        return

    print(f"\n[{slug}/{dataset}] BOXED_ONLY rerun")

    # 1) load cached
    y_train = np.load(emb_dir / 'y_train.npy')
    y_test  = np.load(emb_dir / 'y_test.npy')
    X_train_final = np.load(emb_dir / 'X_train_final.npy')
    X_test_final  = np.load(emb_dir / 'X_test_final.npy')
    probe_final = joblib.load(probes_dir / 'probe_final_layer.joblib')
    with open(pair_dir / 'meta.json') as f:
        prev_meta = json.load(f)
    best_layer = int(prev_meta['best_layer'])
    final_layer = int(prev_meta['final_layer'])
    D = int(prev_meta['hidden_dim'])
    exp10_best_f1 = float(prev_meta['exp10_anchor_f1'])

    v_probe_best = load_exp10_best_probe_direction(slug, dataset, best_layer)
    # Always re-derive v_probe_final from the cached probe — guarantees
    # the saved vector matches the saved probe.
    v_probe_final = probe_normal_direction(probe_final)
    np.save(pair_dir / 'v_probe_final.npy', v_probe_final)

    # 2) exp2 data
    train_data, train_labels_raw = load_exp2_data(model_name, dataset, TRAIN_DIR)
    test_data,  test_labels_raw  = load_exp2_data(model_name, dataset, TEST_DIR)
    if train_data is None or test_data is None:
        print(f"[skip] {slug}/{dataset}: missing exp2 data")
        return

    model = None
    tok = None
    try:
        # 3) tokenizer first (CPU only)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 4) strict label check
        train_valid, kept_y_train = derive_valid_idx(
            train_data, train_labels_raw, tok, desc='train valid_idx')
        test_valid, kept_y_test = derive_valid_idx(
            test_data, test_labels_raw, tok, desc='test valid_idx')
        if (len(kept_y_train) != len(y_train) or
                not np.array_equal(kept_y_train, y_train)):
            print(f"  [ERROR] {slug}/{dataset}: re-derived train valid_idx "
                  f"does not match cached y_train. Aborting.")
            return
        if (len(kept_y_test) != len(y_test) or
                not np.array_equal(kept_y_test, y_test)):
            print(f"  [ERROR] {slug}/{dataset}: re-derived test valid_idx "
                  f"does not match cached y_test. Aborting.")
            return

        # 5) load full model (replaces the CPU tokenizer with the loaded one)
        print(f"[{slug}/{dataset}] loading model...")
        del tok
        model, tok = load_model(model_name)

        # 6) re-extract boxed (FIXED make_boxed_input)
        print(f"[{slug}/{dataset}] re-extracting boxed-position residuals "
              f"at L*={best_layer} and L={final_layer}...")
        X_train_box = extract_boxed_position_at_layers(
            model, tok, train_data, train_valid,
            layers=[best_layer, final_layer], desc='train boxed')
        X_test_box = extract_boxed_position_at_layers(
            model, tok, test_data, test_valid,
            layers=[best_layer, final_layer], desc='test boxed')
        X_train_boxed_best  = X_train_box[:, 0, :]
        X_train_boxed_final = X_train_box[:, 1, :]
        X_test_boxed_best   = X_test_box[:, 0, :]
        X_test_boxed_final  = X_test_box[:, 1, :]
        np.save(emb_dir / 'X_train_boxed_best.npy',  X_train_boxed_best)
        np.save(emb_dir / 'X_test_boxed_best.npy',   X_test_boxed_best)
        np.save(emb_dir / 'X_train_boxed_final.npy', X_train_boxed_final)
        np.save(emb_dir / 'X_test_boxed_final.npy',  X_test_boxed_final)

        # 7) verbal
        print(f"[{slug}/{dataset}] parsing decision token from exp2 (test)...")
        verbal_rows = verbalize_at_boxed(tok, test_data, test_valid,
                                          desc='verbal@boxed')
        for r in verbal_rows:
            is_insuff = not r['is_sufficient']
            r['verbal_correct'] = int(
                (is_insuff and r['kind'] == 'abstain') or
                ((not is_insuff) and r['kind'] == 'numeric')
            )
        n_no_boxed = sum(1 for r in verbal_rows if r['kind'] == 'no_boxed')
        boxed_only_rows = [r for r in verbal_rows if r['kind'] != 'no_boxed']
        verbal_acc_all = float(np.mean([r['verbal_correct'] for r in verbal_rows]))
        verbal_acc_boxed_only = (
            float(np.mean([r['verbal_correct'] for r in boxed_only_rows]))
            if boxed_only_rows else float('nan')
        )

        # 8) retrain boxed probes
        print(f"[{slug}/{dataset}] training 2 boxed probes...")
        probe_boxed_best,  boxed_best_f1  = train_pointwise_probe(
            X_train_boxed_best,  y_train, X_test_boxed_best,  y_test)
        probe_boxed_final, boxed_final_f1 = train_pointwise_probe(
            X_train_boxed_final, y_train, X_test_boxed_final, y_test)
        joblib.dump(probe_boxed_best,  probes_dir / 'probe_boxed_at_best.joblib')
        joblib.dump(probe_boxed_final, probes_dir / 'probe_boxed_at_final.joblib')
        v_probe_boxed_best  = probe_normal_direction(probe_boxed_best)
        v_probe_boxed_final = probe_normal_direction(probe_boxed_final)
        np.save(pair_dir / 'v_probe_boxed_best.npy',  v_probe_boxed_best)
        np.save(pair_dir / 'v_probe_boxed_final.npy', v_probe_boxed_final)

        # 9) re-derive cutoff F1 from cached probe + cached X
        N_tr, n_pct, _ = X_train_final.shape
        Xtr_flat = X_train_final.reshape(-1, D)
        ytr_flat = np.repeat(y_train, n_pct)
        Xte_flat = X_test_final.reshape(-1, D)
        yte_flat = np.repeat(y_test, n_pct)
        concat_train_f1 = float(f1_score(ytr_flat, probe_final.predict(Xtr_flat)))
        concat_test_f1  = float(f1_score(yte_flat, probe_final.predict(Xte_flat)))
        final_per_cutoff = []
        for pi, pct in enumerate(PERCENTAGES):
            f1_tr = float(f1_score(y_train, probe_final.predict(X_train_final[:, pi, :])))
            f1_te = float(f1_score(y_test,  probe_final.predict(X_test_final[:, pi, :])))
            final_per_cutoff.append({
                'pct': f'{int(pct * 100)}%',
                'train_f1': round(f1_tr, 4),
                'test_f1':  round(f1_te, 4),
            })
        final_concat = {
            'concat_train_f1': round(concat_train_f1, 4),
            'concat_test_f1':  round(concat_test_f1,  4),
        }

        # 10) Measurement A
        print(f"[{slug}/{dataset}] recomputing Measurement A...")
        W_U = get_unembed_weight(model).detach().to(torch.float32).cpu().numpy()
        abs_ids = build_abstention_token_ids(tok)
        num_ids = build_numeric_token_ids(tok)
        v_rand = random_direction(D, seed=42)

        candidates = []
        if v_probe_best is not None:
            candidates.append((v_probe_best, 'v_probe_best'))
        candidates += [
            (v_probe_final,       'v_probe_final'),
            (v_probe_boxed_best,  'v_probe_boxed_best'),
            (v_probe_boxed_final, 'v_probe_boxed_final'),
            (v_rand,              'v_random'),
        ]
        align_rows = []
        for v, name in candidates:
            s = measurement_A(W_U, v, abs_ids, num_ids)
            align_rows.append({
                'model': slug, 'dataset': dataset, 'direction': name,
                'margin_abs_num': s['margin_abs_num'],
                'z_mean_abs': s['z_mean_abs'], 'z_mean_num': s['z_mean_num'],
                'median_rank_abs_pct': s['median_rank_abs_pct'],
                'median_rank_num_pct': s['median_rank_num_pct'],
            })
        pd.DataFrame(align_rows).to_csv(pair_dir / 'alignment.csv', index=False)

        # 11) f1_summary
        f1_rows = []
        for r in final_per_cutoff:
            f1_rows.append({
                'model': slug, 'dataset': dataset,
                'probe_variant': 'final_layer_cutoff_concat',
                'layer': final_layer, 'pct': r['pct'],
                'train_f1': r['train_f1'], 'test_f1': r['test_f1'],
            })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'final_layer_cutoff_concat',
            'layer': final_layer, 'pct': 'concat',
            'train_f1': final_concat['concat_train_f1'],
            'test_f1':  final_concat['concat_test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'boxed_at_best',
            'layer': best_layer, 'pct': 'boxed',
            'train_f1': boxed_best_f1['train_f1'],
            'test_f1':  boxed_best_f1['test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'boxed_at_final',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': boxed_final_f1['train_f1'],
            'test_f1':  boxed_final_f1['test_f1'],
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'exp10_best_layer',
            'layer': best_layer, 'pct': '0%',
            'train_f1': float('nan'), 'test_f1': exp10_best_f1,
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'verbal@boxed_all',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': float('nan'),
            'test_f1':  round(verbal_acc_all, 4),
        })
        f1_rows.append({
            'model': slug, 'dataset': dataset,
            'probe_variant': 'verbal@boxed_only',
            'layer': final_layer, 'pct': 'boxed',
            'train_f1': float('nan'),
            'test_f1':  round(verbal_acc_boxed_only, 4),
        })
        pd.DataFrame(f1_rows).to_csv(pair_dir / 'f1_summary.csv', index=False)
        pd.DataFrame(verbal_rows).to_csv(pair_dir / 'verbal_at_boxed.csv', index=False)

        # 12) meta
        gap_final_all = boxed_final_f1['test_f1'] - verbal_acc_all
        gap_final_boxed_only = boxed_final_f1['test_f1'] - verbal_acc_boxed_only
        gap_best_all = boxed_best_f1['test_f1']  - verbal_acc_all
        gap_best_boxed_only = boxed_best_f1['test_f1']  - verbal_acc_boxed_only
        warnings = []
        if final_concat['concat_test_f1'] < 0.5:
            warnings.append(f"final-layer concat F1 {final_concat['concat_test_f1']:.3f} "
                            f"< 0.5")
        if abs(boxed_best_f1['test_f1'] - exp10_best_f1) > 0.2:
            warnings.append(f"boxed_at_best F1 ({boxed_best_f1['test_f1']:.3f}) "
                            f"differs from exp10 best-layer F1 ({exp10_best_f1:.3f}) "
                            f"by >0.2; position effect large")
        for w in warnings:
            print(f"  [WARN] {slug}/{dataset}: {w}")
        meta = {
            'model': slug, 'dataset': dataset,
            'best_layer': best_layer, 'final_layer': final_layer,
            'hidden_dim': int(D), 'vocab_size': int(W_U.shape[0]),
            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
            'n_T_abs': len(abs_ids), 'n_T_num': len(num_ids),
            'exp10_anchor_f1': exp10_best_f1,
            'final_layer_concat_test_f1': final_concat['concat_test_f1'],
            'boxed_at_best_test_f1':  boxed_best_f1['test_f1'],
            'boxed_at_final_test_f1': boxed_final_f1['test_f1'],
            'verbal_accuracy_all_samples':  verbal_acc_all,
            'verbal_accuracy_boxed_only':   verbal_acc_boxed_only,
            'n_no_boxed_test': n_no_boxed,
            'frac_no_boxed_test': n_no_boxed / len(verbal_rows),
            'gap_at_boxed_final_all':         gap_final_all,
            'gap_at_boxed_final_boxed_only':  gap_final_boxed_only,
            'gap_at_boxed_best_all':          gap_best_all,
            'gap_at_boxed_best_boxed_only':   gap_best_boxed_only,
            'measurement_A_at_L_summary': {
                r['direction']: r['margin_abs_num'] for r in align_rows
            },
            'boxed_only_rerun': True,
            'warnings': warnings,
        }
        with open(pair_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        (pair_dir / '_COMPLETE').touch()
        print(f"[{slug}/{dataset}] BOXED_ONLY DONE")
    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate(out_root):
    out_root = Path(out_root)
    f1_dfs, align_dfs, meta_rows = [], [], []
    for meta_path in out_root.rglob('meta.json'):
        pair_dir = meta_path.parent
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta_rows.append({
                'model': meta['model'], 'dataset': meta['dataset'],
                'best_layer': meta['best_layer'],
                'final_layer': meta['final_layer'],
                'hidden_dim': meta['hidden_dim'],
                'n_train': meta['n_train'], 'n_test': meta['n_test'],
                'exp10_anchor_f1': meta['exp10_anchor_f1'],
                'final_layer_concat_test_f1': meta['final_layer_concat_test_f1'],
                'boxed_at_best_test_f1': meta['boxed_at_best_test_f1'],
                'boxed_at_final_test_f1': meta['boxed_at_final_test_f1'],
                'verbal_acc_all': meta.get('verbal_accuracy_all_samples'),
                'verbal_acc_boxed_only': meta.get('verbal_accuracy_boxed_only'),
                'n_no_boxed': meta.get('n_no_boxed_test'),
                'frac_no_boxed': meta.get('frac_no_boxed_test'),
                'gap_at_boxed_final_all': meta.get('gap_at_boxed_final_all'),
                'gap_at_boxed_final_boxed_only': meta.get('gap_at_boxed_final_boxed_only'),
                'A_v_probe_best':        meta['measurement_A_at_L_summary'].get('v_probe_best'),
                'A_v_probe_final':       meta['measurement_A_at_L_summary'].get('v_probe_final'),
                'A_v_probe_boxed_best':  meta['measurement_A_at_L_summary'].get('v_probe_boxed_best'),
                'A_v_probe_boxed_final': meta['measurement_A_at_L_summary'].get('v_probe_boxed_final'),
                'A_v_random':            meta['measurement_A_at_L_summary'].get('v_random'),
                'warnings': '; '.join(meta.get('warnings', [])),
            })
            if (pair_dir / 'f1_summary.csv').exists():
                f1_dfs.append(pd.read_csv(pair_dir / 'f1_summary.csv'))
            if (pair_dir / 'alignment.csv').exists():
                align_dfs.append(pd.read_csv(pair_dir / 'alignment.csv'))
        except Exception as e:
            print(f"  [aggregate] skipping {pair_dir}: {e}")
    Path(MASTER_CSV_DIR).mkdir(parents=True, exist_ok=True)
    if meta_rows:
        df = pd.DataFrame(meta_rows)
        df.to_csv(Path(MASTER_CSV_DIR) / 'final_layer_extension.csv', index=False)
        print(f"Wrote final_layer_extension.csv: {len(df)} pairs")
        cols = ['model', 'dataset', 'exp10_anchor_f1',
                'final_layer_concat_test_f1', 'boxed_at_best_test_f1',
                'boxed_at_final_test_f1', 'verbal_acc_all',
                'verbal_acc_boxed_only', 'frac_no_boxed',
                'gap_at_boxed_final_boxed_only',
                'A_v_probe_best', 'A_v_probe_final',
                'A_v_probe_boxed_final', 'A_v_random']
        existing = [c for c in cols if c in df.columns]
        print('\nHeadline numbers:')
        print(df[existing].to_string(index=False))
    if f1_dfs:
        pd.concat(f1_dfs, ignore_index=True).to_csv(
            Path(MASTER_CSV_DIR) / 'f1_by_variant.csv', index=False)
        print(f"Wrote f1_by_variant.csv")
    if align_dfs:
        pd.concat(align_dfs, ignore_index=True).to_csv(
            Path(MASTER_CSV_DIR) / 'alignment_all_pairs.csv', index=False)
        print(f"Wrote alignment_all_pairs.csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--aggregate_only', action='store_true')
    p.add_argument('--boxed_only', action='store_true',
                   help='For pairs with _COMPLETE: redo boxed extraction + '
                        'verbal + boxed probes + margins, reusing cached '
                        'cutoff outputs.')
    args = p.parse_args()

    if args.aggregate_only:
        aggregate(OUTPUT_BASE)
        return

    if args.smoke:
        models = SMOKE_MODELS
        datasets = ['umwp', 'treecut']
    else:
        if args.all_models:
            models = FULL_MODELS
        elif args.model:
            models = [args.model]
        else:
            raise SystemExit("Use --model, --all_models, or --smoke")
        if args.all_datasets:
            datasets = ['umwp', 'treecut']
        elif args.dataset:
            datasets = [args.dataset]
        else:
            raise SystemExit("Use --dataset or --all_datasets")

    Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
    runner = process_one_boxed_only if args.boxed_only else process_one
    for m in models:
        for d in datasets:
            try:
                runner(m, d, args)
            except Exception as e:
                print(f"\n[ERROR] {m}/{d}: {e}")
                import traceback; traceback.print_exc()

    aggregate(OUTPUT_BASE)


if __name__ == '__main__':
    main()