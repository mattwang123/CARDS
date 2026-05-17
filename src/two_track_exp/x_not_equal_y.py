"""
================================================================================
EXPERIMENT SB: Ablate Probe Direction, Measure Verbalization
================================================================================

THE SMOKING-GUN TEST FOR THE X != Y SUBSPACE HYPOTHESIS

Hypothesis:
    - Y subspace : the direction the linear probe (exp10 unified probe) uses
                   to read insufficiency. High F1 readout, but possibly does
                   NOT causally drive the model's output behavior.
    - X subspace : the direction(s) that actually drive verbalization. Unknown
                   location. Possibly non-linear / distributed / different from Y.

Critical prediction:
    If we surgically REMOVE the Y direction from the residual stream at every
    layer and every token position, then re-run forced-decode verbalization:
      * Probe F1 measured on the ablated states should collapse to chance
        (sanity check: we really removed the Y direction).
      * Verbalization accuracy should be largely UNCHANGED if Y != X.
      * Verbalization accuracy DROPS if Y == X (or overlaps significantly).

This script implements that test.

----------------------------------------------------------------------------
WHAT IT MEASURES (compared to exp14_cutoff_verbalize):
   exp14:   verbalization accuracy at cutoffs {0%, 20%, 40%, 60%, 100%} with
            NO intervention on activations.
   SB:      same forced-decode protocol, but with FOUR ablation conditions
            stacked side-by-side per sample:
              - baseline   : no ablation                  (replicates exp14)
              - ablate_y   : project out v_Y at every layer (the test)
              - ablate_rand: project out a random direction (control)
              - ablate_dim : project out DIM direction       (compare-to-X-proxy)
            We also report probe F1 (on the ablated activations) as a
            within-run sanity check.

----------------------------------------------------------------------------
PROTOCOL PER (model, dataset, sample, cutoff_pct):
   1. Truncate the CoT at the sentence boundary near cutoff_pct (exp14's
      find_sentence_boundary_cutoff).
   2. Build forced input: prompt + truncated_cot + "\n\n**Final Answer**\n\\boxed{".
   3. Register the relevant ablation hooks on every transformer block (or
      none for baseline).
   4. Greedy-decode up to ~50 tokens with StopOnCloseBrace.
   5. Parse the boxed answer.
   6. Classify the answer as numeric_answer / insufficient_claim / other.

PROTOCOL FOR THE PROBE SANITY CHECK:
   Once per (model, dataset, condition), we extract t=0 (end-of-prompt)
   hidden states with the ablation hooks active, run them through the
   exp10 unified probe, and compute F1 against ground-truth labels.
   If ablate_y reduces probe F1 to ~chance, we know the ablation worked.

----------------------------------------------------------------------------
OUTPUT FORMAT:

  Sample-major JSON per (model, dataset):
    /export/.../experiments/sb_ablate_probe/<slug>/<dataset>_sample_major.json

    Structure:
      [
        {
          "sample_id": ...,
          "question": ...,
          "is_sufficient": true/false,
          "quadrant": "Q1_Hallucination" | "Q2_Correct_Rejection" | ...,
          "cutoffs": {
            "0":   {"baseline": {...}, "ablate_y": {...}, ...},
            "20":  {...},
            ...
          }
        },
        ...
      ]
    where each inner dict has:
      {"truncated_cot_token_count": int, "actual_boundary_pct": float,
       "generated_answer": str|None, "raw_output": str,
       "answer_kind": "numeric" | "insufficient" | "other"}

  Probe-sanity CSV per (model, dataset):
    /export/.../experiments/sb_ablate_probe/<slug>/<dataset>_probe_sanity.csv
    columns: condition, n, probe_f1, mean_p_insuff_on_insuff, mean_p_insuff_on_suff

  Aggregate summary CSV (appended across all model x dataset runs):
    /export/.../experiments/sb_ablate_probe/summary.csv
    columns: model, dataset, condition, cutoff_pct, quadrant,
             n, verb_accuracy, insuff_claim_rate, numeric_rate,
             flip_to_insuff_from_q1_rate

----------------------------------------------------------------------------
INTERPRETATION CHEAT SHEET:

  Outcome 1 (THE PAPER):
    ablate_y reduces probe F1 from ~0.85 to ~0.50 (verified ablation),
    AND verb_accuracy at cutoff 100% under ablate_y is within +/- 5pp of
    baseline at the same cutoff.
    => Y direction is causally irrelevant for verbalization at the layer
       it was found. X subspace is elsewhere (or non-linear).

  Outcome 2 (FAIL OF HYPOTHESIS):
    ablate_y reduces probe F1 to ~0.50 AND verb_accuracy at cutoff 100%
    drops by >10pp.
    => Y direction overlaps with X. Single linear direction encodes both
       probe readout and verbalization control.

  Outcome 3 (METHODOLOGICAL):
    ablate_rand also drops verb_accuracy substantially.
    => The ablation procedure itself is too destructive (we are pushing
       out-of-distribution). Need to redesign (e.g. ablate at fewer layers).

----------------------------------------------------------------------------
USAGE:
    # single (model, dataset) run
    python SB_ablate_probe_verbalize.py \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --dataset umwp

    # representative subset across both datasets
    python SB_ablate_probe_verbalize.py --all_datasets

    # full sweep (warning: long)
    python SB_ablate_probe_verbalize.py --all_models --all_datasets

================================================================================
"""

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# CONFIG (matches the conventions in exp10/exp11/exp13/exp14)
# ============================================================================

# Where exp2 generations and exp11/exp13 metadata live (read-only):
SOURCE_EXPORT_BASE = '/export/fs06/hwang302/CARDS'

# Where we write SB outputs (separate from the canonical CARDS results dir):
SB_OUTPUT_BASE = '/export/fs06/hwang302/CARDS/result_two_track_exp'

DEFAULT_GENERATION_DIR = os.path.join(
    SOURCE_EXPORT_BASE, 'experiments/dynamic_tracking_test'
)
DEFAULT_EVAL_DIR = os.path.join(
    SOURCE_EXPORT_BASE, 'experiments/dynamic_tracking_test_evaluation'
)
DEFAULT_OUTPUT_DIR = os.path.join(
    SB_OUTPUT_BASE, 'experiments/sb_ablate_probe'
)
EXP_TEMPORAL_DIR = os.path.join(SOURCE_EXPORT_BASE, 'exp_temporal_new')


DATASETS = ['umwp', 'treecut']
# Match exp14's cutoffs so we can directly compare baseline numbers.
CUTOFF_PERCENTAGES = [0.0, 0.20, 0.40, 0.60, 1.00]
FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"

# The four ablation conditions (key, description). 'baseline' uses no hooks.
CONDITIONS = ['baseline', 'ablate_y', 'ablate_rand', 'ablate_dim']

# Model lists kept identical to exp14 for downstream comparability.
FULL_MODELS = [
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

REPRESENTATIVE_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
]


# ============================================================================
# CUTOFF UTILITIES (verbatim from exp14, kept inline to avoid cross-imports)
# ============================================================================

@dataclass
class CutoffResult:
    text: str
    actual_pct: float
    token_count: int
    target_token_idx: int
    total_tokens: int
    boundary_kind: str


def _token_ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)['input_ids']


def _sentence_boundary_candidates(text):
    candidates = []
    for match in re.finditer(r'\n{2,}', text):
        candidates.append((match.end(), 0, 'paragraph_break'))
    for match in re.finditer(r'[.!?](?:\s+)(?=[A-Z]|$)', text):
        candidates.append((match.end(), 1, 'sentence_punct_whitespace'))
    for match in re.finditer(r'[.!?]\n', text):
        candidates.append((match.end(), 2, 'sentence_punct_newline'))
    for match in re.finditer(r'[.!?]', text):
        candidates.append((match.end(), 3, 'bare_sentence_punct'))

    best_by_pos = {}
    for pos, priority, kind in candidates:
        if pos not in best_by_pos or priority < best_by_pos[pos][0]:
            best_by_pos[pos] = (priority, kind)
    return [(pos, priority, kind) for pos, (priority, kind) in best_by_pos.items()]


def zero_cutoff_result(cot_text, tokenizer):
    ids = _token_ids(tokenizer, cot_text)
    return CutoffResult("", 0.0, 0, 0, len(ids), "no_cot")


def find_sentence_boundary_cutoff(text, target_pct, tokenizer):
    ids = _token_ids(tokenizer, text)
    total_tokens = len(ids)
    if total_tokens == 0:
        return CutoffResult("", 0.0, 0, 0, 0, "empty")
    target_token_idx = max(1, min(total_tokens, int(target_pct * total_tokens)))
    if target_pct >= 1.0:
        return CutoffResult(text, 1.0, total_tokens, target_token_idx, total_tokens, "full_trace")

    candidates = _sentence_boundary_candidates(text)
    if not candidates:
        raw_text = tokenizer.decode(ids[:target_token_idx], skip_special_tokens=True)
        return CutoffResult(raw_text, target_token_idx / total_tokens,
                            target_token_idx, target_token_idx, total_tokens, "raw_token_cutoff")

    scored = []
    for pos, priority, kind in candidates:
        candidate_text = text[:pos].rstrip()
        candidate_tokens = len(_token_ids(tokenizer, candidate_text))
        if candidate_tokens == 0:
            continue
        distance = abs(candidate_tokens - target_token_idx)
        scored.append((distance, priority, candidate_tokens, pos, kind))
    if not scored:
        raw_text = tokenizer.decode(ids[:target_token_idx], skip_special_tokens=True)
        return CutoffResult(raw_text, target_token_idx / total_tokens,
                            target_token_idx, target_token_idx, total_tokens, "raw_token_cutoff")
    _, _, actual_tokens, char_pos, kind = min(scored, key=lambda x: (x[0], x[1]))
    truncated = text[:char_pos].rstrip()
    return CutoffResult(truncated, actual_tokens / total_tokens,
                        actual_tokens, target_token_idx, total_tokens, kind)


# ============================================================================
# FORCED DECODE (same protocol as exp14)
# ============================================================================

class StopOnCloseBrace:
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_len:]
        if generated_ids.numel() == 0:
            return False
        return "}" in self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def truncate_at_first_close_brace(text):
    close_idx = text.find("}")
    return text[:close_idx + 1] if close_idx != -1 else text


def generate_force_decode(model, tokenizer, forced_input, max_new_tokens=50):
    from transformers import StoppingCriteriaList
    inputs = tokenizer(forced_input, return_tensors="pt").to(model.device)
    prompt_len = inputs['input_ids'].shape[1]
    stopping = StoppingCriteriaList([StopOnCloseBrace(tokenizer, prompt_len)])
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    continuation = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    continuation = truncate_at_first_close_brace(continuation).strip()
    raw_output = FORCE_DECODE_SUFFIX + continuation
    return continuation, raw_output


# ============================================================================
# ANSWER CLASSIFICATION
# ============================================================================

# Strict pattern (matches exp13's strict regex but applied to the boxed content)
INSUFF_PATTERNS = (
    r'^\s*('
    r'insufficient|not enough(?:\s+information)?|cannot\s+(?:be\s+)?determined|'
    r'undetermined|missing(?:\s+information)?|unknown|unsolvable|'
    r'no\s+unique\s+answer|cannot\s+determine|unable\s+to\s+determine|'
    r'n/?a|impossible|indeterminate'
    r')\s*$'
)
INSUFF_REGEX = re.compile(INSUFF_PATTERNS, re.IGNORECASE)
NUMERIC_REGEX = re.compile(r'[-+]?\d')   # any digit anywhere = numeric attempt


def classify_answer(boxed_content):
    """Return one of 'insufficient', 'numeric', 'other'."""
    if boxed_content is None:
        return 'other'
    cleaned = boxed_content.strip().strip('.').strip()
    if INSUFF_REGEX.match(cleaned):
        return 'insufficient'
    if NUMERIC_REGEX.search(cleaned):
        return 'numeric'
    return 'other'


def extract_boxed_content(raw_output):
    """Pull what's inside the first \\boxed{...} of the raw output."""
    if not raw_output:
        return None
    m = re.search(r'\\boxed\s*\{([^{}]*)\}', raw_output)
    return m.group(1).strip() if m else None


def is_verbalize_correct(boxed_content, is_sufficient):
    """
    Verbalization-correct rule:
      - is_sufficient = True  => must contain a digit (numeric attempt; we
                                 don't have ground-truth numeric value here,
                                 so 'gave a number' is the right proxy).
      - is_sufficient = False => must classify as 'insufficient'.
    """
    kind = classify_answer(boxed_content)
    if is_sufficient:
        return kind == 'numeric'
    return kind == 'insufficient'


# ============================================================================
# ABLATION HOOK (same logic as exp13)
# ============================================================================

class AblationHook:
    """Project out v (assumed unit-norm) from this layer's residual stream."""
    def __init__(self, v):
        self.v = v
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if (self._cached is None
                or self._cached.device != hs.device
                or self._cached.dtype != hs.dtype):
            self._cached = self.v.to(device=hs.device, dtype=hs.dtype)
        proj = (hs * self._cached).sum(dim=-1, keepdim=True)
        modified = hs - proj * self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


def get_layer_modules(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    if hasattr(model, 'layers'):
        return model.layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    raise ValueError("Cannot locate transformer layers")


def register_ablation(layer_modules, v):
    """Register an AblationHook on every layer. Returns list of handles."""
    handles = []
    for lm in layer_modules:
        handles.append(lm.register_forward_hook(AblationHook(v)))
    return handles


# ============================================================================
# DIRECTION COMPUTATION
# ============================================================================

def probe_normal_direction(probe):
    """Effective unit-norm probe direction on raw activations."""
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def random_direction(hidden_dim, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(hidden_dim).astype(np.float32)
    return v / np.linalg.norm(v)


def extract_t0_states(model, tokenizer, prompts, layer, batch_size=4):
    """Extract end-of-prompt hidden states at <layer>, in batches."""
    states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt",
                        padding=True, truncation=False).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        attn = enc['attention_mask']
        last_idx = (attn.sum(dim=1) - 1).tolist()
        h = out.hidden_states[layer]
        for b, idx in enumerate(last_idx):
            states.append(h[b, idx, :].to(torch.float32).cpu().numpy())
        del out
        torch.cuda.empty_cache()
    return np.array(states)


def compute_dim_direction(model, tokenizer, dataset, model_slug, layer, n_per_class):
    """
    Compute v_DIM = mean(insuff acts) - mean(suff acts) at t=0, from balanced
    train prompts. Returns unit-norm np.float32 vector.
    """
    train_path = os.path.join(
        SOURCE_EXPORT_BASE,
        f"experiments/dynamic_tracking_train/math/{model_slug}/{dataset}_cot_generations.json"
    )
    if not os.path.exists(train_path):
        return None
    with open(train_path) as f:
        train = json.load(f)
    suff = [g['prompt'] for g in train if g.get('is_sufficient', True)][:n_per_class]
    insuff = [g['prompt'] for g in train if not g.get('is_sufficient', True)][:n_per_class]
    if not suff or not insuff:
        return None
    X_suff = extract_t0_states(model, tokenizer, suff, layer)
    X_insuff = extract_t0_states(model, tokenizer, insuff, layer)
    v = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_quadrant_lookup(model_slug, dataset):
    """Return dict: question_idx -> quadrant string."""
    eval_path = os.path.join(
        DEFAULT_EVAL_DIR, 'math', model_slug, f'{dataset}_evaluated_traces.json'
    )
    if not os.path.exists(eval_path):
        return {}
    with open(eval_path) as f:
        eval_data = json.load(f).get('data', [])
    lookup = {}
    for e in eval_data:
        key = str(e.get('question_idx', e.get('sample_id', e.get('id'))))
        lookup[key] = e.get('epistemic_quadrant', '')
    return lookup


def load_generation_data(model_slug, dataset, max_samples=None):
    path = os.path.join(
        DEFAULT_GENERATION_DIR, 'math', model_slug, f'{dataset}_cot_generations.json'
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data if max_samples is None else data[:max_samples]


# ============================================================================
# PROBE SANITY CHECK
# ============================================================================

def compute_probe_sanity(model, tokenizer, probe, layer, prompts, labels,
                        ablation_v=None, layer_modules=None):
    """
    Extract t=0 hidden states (with optional ablation active), feed through
    probe, return {n, probe_f1, mean_p_insuff_on_insuff, mean_p_insuff_on_suff}.

    labels: 1 = insufficient, 0 = sufficient
    """
    from sklearn.metrics import f1_score

    handles = []
    if ablation_v is not None and layer_modules is not None:
        v_t = torch.tensor(ablation_v, dtype=torch.float32)
        handles = register_ablation(layer_modules, v_t)
    try:
        X = extract_t0_states(model, tokenizer, prompts, layer)
    finally:
        for h in handles:
            h.remove()

    proba = probe.predict_proba(X)[:, 1]
    pred = (proba > 0.5).astype(int)
    f1 = float(f1_score(labels, pred, zero_division=0))
    p_on_insuff = float(proba[labels == 1].mean()) if (labels == 1).any() else float('nan')
    p_on_suff = float(proba[labels == 0].mean()) if (labels == 0).any() else float('nan')
    return {
        'n': int(len(prompts)),
        'probe_f1': round(f1, 4),
        'mean_p_insuff_on_insuff': round(p_on_insuff, 4),
        'mean_p_insuff_on_suff': round(p_on_suff, 4),
    }


# ============================================================================
# MAIN RUN PER (model, dataset)
# ============================================================================

def model_slug_of(model_name):
    return model_name.split('/')[-1]


def output_paths(output_dir, slug, dataset):
    base = Path(output_dir) / slug
    base.mkdir(parents=True, exist_ok=True)
    return {
        'sample_major': base / f'{dataset}_sample_major.json',
        'probe_sanity': base / f'{dataset}_probe_sanity.csv',
        'meta':         base / f'{dataset}_meta.json',
    }


def load_existing_sample_major(path):
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def run_model_dataset(model_name, dataset, args):
    slug = model_slug_of(model_name)
    paths = output_paths(args.output_dir, slug, dataset)

    # ------------------------------------------------------------------
    # 1) Locate exp10 unified probe + its best layer
    # ------------------------------------------------------------------
    exp10_csv = os.path.join(
        EXP_TEMPORAL_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv'
    )
    if not os.path.exists(exp10_csv):
        print(f"  ! Missing exp10 csv: {exp10_csv}")
        return
    exp10_df = pd.read_csv(exp10_csv)
    row = exp10_df[exp10_df['Model'] == slug]
    if row.empty:
        print(f"  ! No exp10 entry for {slug}")
        return
    best_layer = int(row['Optimal_Layer'].iloc[0])
    probe_path = os.path.join(
        EXP_TEMPORAL_DIR, 'probes_proportional', dataset, slug,
        f'unified_probe_layer{best_layer}.joblib'
    )
    if not os.path.exists(probe_path):
        print(f"  ! Missing probe: {probe_path}")
        return
    probe = joblib.load(probe_path)
    v_probe = probe_normal_direction(probe)
    hidden_dim = v_probe.shape[0]

    # ------------------------------------------------------------------
    # 2) Load test generations + quadrant labels
    # ------------------------------------------------------------------
    gen_data = load_generation_data(slug, dataset,
                                    max_samples=args.test_samples if args.test else None)
    if gen_data is None:
        print(f"  ! No CoT data for {slug}/{dataset}")
        return
    quad_lookup = load_quadrant_lookup(slug, dataset)

    print(f"\n[{slug} / {dataset}]")
    print(f"  best_layer={best_layer}, hidden_dim={hidden_dim}, n_samples={len(gen_data)}")

    # Resume sample-major file if present
    sample_records = load_existing_sample_major(paths['sample_major']) or []
    by_idx = {str(r['sample_id']): r for r in sample_records}

    # ------------------------------------------------------------------
    # 3) Load model
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_gpus = torch.cuda.device_count()
    memory_map = {0: "65GB"} if num_gpus > 0 else None
    if memory_map and num_gpus > 1:
        for i in range(1, num_gpus):
            memory_map[i] = "78GB"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", max_memory=memory_map,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()
    layer_modules = get_layer_modules(model)

    # ------------------------------------------------------------------
    # 4) Compute DIM direction (uses model + tokenizer once)
    # ------------------------------------------------------------------
    print(f"  Computing v_DIM at layer {best_layer}...")
    v_dim = compute_dim_direction(model, tokenizer, dataset, slug,
                                  best_layer, args.n_dim)
    if v_dim is None:
        print(f"  ! Could not compute v_DIM; ablate_dim will be skipped.")

    v_rand = random_direction(hidden_dim, seed=42)

    # Cosines for the meta file (informative for paper)
    cos_pd = float(np.dot(v_probe, v_dim)) if v_dim is not None else None
    cos_pr = float(np.dot(v_probe, v_rand))
    cos_dr = float(np.dot(v_dim, v_rand)) if v_dim is not None else None

    # ------------------------------------------------------------------
    # 5) PROBE SANITY CHECK (per condition)
    # ------------------------------------------------------------------
    # Take a manageable subset of prompts for sanity probing, ensuring
    # both classes are represented.
    sanity_n = min(args.probe_sanity_n, len(gen_data))
    sanity_items = gen_data[:sanity_n]
    sanity_prompts = [it['prompt'] for it in sanity_items]
    sanity_labels = np.array([
        0 if it.get('is_sufficient', True) else 1 for it in sanity_items
    ])

    print(f"  Probe sanity ({sanity_n} prompts):")
    sanity_rows = []
    v_map = {
        'baseline':    None,
        'ablate_y':    v_probe,
        'ablate_rand': v_rand,
        'ablate_dim':  v_dim,
    }
    for cond in CONDITIONS:
        v = v_map[cond]
        if cond == 'ablate_dim' and v is None:
            continue
        stats = compute_probe_sanity(
            model, tokenizer, probe, best_layer,
            sanity_prompts, sanity_labels,
            ablation_v=v, layer_modules=layer_modules,
        )
        stats.update({'condition': cond, 'model': slug, 'dataset': dataset})
        sanity_rows.append(stats)
        print(f"    {cond:<12}  f1={stats['probe_f1']:.3f}  "
              f"P(I|insuff)={stats['mean_p_insuff_on_insuff']:.3f}  "
              f"P(I|suff)={stats['mean_p_insuff_on_suff']:.3f}")
    pd.DataFrame(sanity_rows).to_csv(paths['probe_sanity'], index=False)

    # ------------------------------------------------------------------
    # 6) Write meta now (everything except final per-sample generations)
    # ------------------------------------------------------------------
    meta = {
        'model': slug, 'dataset': dataset,
        'best_layer': best_layer, 'hidden_dim': hidden_dim,
        'n_samples': len(gen_data),
        'cos_probe_dim': cos_pd, 'cos_probe_rand': cos_pr, 'cos_dim_rand': cos_dr,
        'cutoffs': CUTOFF_PERCENTAGES, 'conditions': CONDITIONS,
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # 7) Forced-decode under every (cutoff, condition) for every sample
    # ------------------------------------------------------------------
    # Strategy: outer loop over (cutoff, condition) -- register hooks ONCE
    # per (cutoff, condition) and iterate over samples. This avoids paying
    # the hook-register cost N times per sample.
    for cutoff_pct in CUTOFF_PERCENTAGES:
        cutoff_key = str(int(round(cutoff_pct * 100)))
        for cond in CONDITIONS:
            v = v_map[cond]
            if cond == 'ablate_dim' and v is None:
                continue

            # Which samples still need this (cutoff, cond)? Resume support.
            todo = []
            for idx, item in enumerate(gen_data):
                sid = str(item.get('sample_id',
                                   item.get('question_idx', item.get('id', idx))))
                rec = by_idx.get(sid)
                if rec is not None:
                    have = rec.get('cutoffs', {}).get(cutoff_key, {}).get(cond)
                    if have is not None:
                        continue
                todo.append((idx, item, sid))
            if not todo:
                continue

            # Register hooks for this condition (or none for baseline)
            handles = []
            if cond != 'baseline':
                v_t = torch.tensor(v, dtype=torch.float32)
                handles = register_ablation(layer_modules, v_t)

            try:
                for idx, item, sid in tqdm(
                    todo,
                    desc=f"  cutoff={cutoff_key}% / {cond}",
                ):
                    prompt = item.get('prompt', '')
                    cot_text = item.get('generated_response',
                                        item.get('model_output', ''))
                    if cutoff_pct <= 0.0:
                        cutoff = zero_cutoff_result(cot_text, tokenizer)
                    else:
                        cutoff = find_sentence_boundary_cutoff(
                            cot_text, cutoff_pct, tokenizer
                        )

                    # Skip very short cuts (consistent with exp14 behaviour)
                    if cutoff_pct > 0.0 and cutoff.token_count < args.min_cutoff_tokens:
                        per_variant = {
                            'status': 'skipped_short_cutoff',
                            'truncated_cot_token_count': cutoff.token_count,
                            'actual_boundary_pct': cutoff.actual_pct,
                            'generated_answer': None,
                            'raw_output': '',
                            'answer_kind': 'other',
                            'verbalize_correct': False,
                        }
                    else:
                        forced_input = prompt + cutoff.text + FORCE_DECODE_SUFFIX
                        continuation, raw_output = generate_force_decode(
                            model, tokenizer, forced_input,
                            max_new_tokens=args.max_new_tokens
                        )
                        boxed = extract_boxed_content(raw_output)
                        kind = classify_answer(boxed)
                        per_variant = {
                            'status': 'success',
                            'truncated_cot_token_count': cutoff.token_count,
                            'actual_boundary_pct': round(cutoff.actual_pct, 4),
                            'boundary_kind': cutoff.boundary_kind,
                            'generated_answer': boxed,
                            'raw_output': raw_output,
                            'answer_kind': kind,
                            'verbalize_correct': is_verbalize_correct(
                                boxed, item.get('is_sufficient', True)
                            ),
                        }

                    rec = by_idx.get(sid)
                    if rec is None:
                        rec = {
                            'sample_id': sid,
                            'question_idx': item.get('question_idx',
                                                      item.get('id', idx)),
                            'question': item.get('question'),
                            'is_sufficient': item.get('is_sufficient'),
                            'quadrant': quad_lookup.get(sid, ''),
                            'cutoffs': {},
                        }
                        sample_records.append(rec)
                        by_idx[sid] = rec
                    rec['cutoffs'].setdefault(cutoff_key, {})[cond] = per_variant

                    # Save every save_interval items
                    if (len(todo) > 0 and
                            (todo.index((idx, item, sid)) + 1) % args.save_interval == 0):
                        with open(paths['sample_major'], 'w') as f:
                            json.dump(sample_records, f, indent=2)
            finally:
                for h in handles:
                    h.remove()

            # End-of-condition save
            with open(paths['sample_major'], 'w') as f:
                json.dump(sample_records, f, indent=2)
            print(f"  -> saved through cutoff={cutoff_key}/{cond}")

    # ------------------------------------------------------------------
    # 8) Append summary rows to global summary
    # ------------------------------------------------------------------
    rows = []
    for rec in sample_records:
        sid = rec['sample_id']
        for cutoff_key, by_cond in rec.get('cutoffs', {}).items():
            for cond, v_data in by_cond.items():
                if v_data.get('status') != 'success':
                    continue
                rows.append({
                    'model': slug, 'dataset': dataset,
                    'condition': cond, 'cutoff_pct': int(cutoff_key),
                    'sample_id': sid,
                    'is_sufficient': bool(rec.get('is_sufficient', False)),
                    'quadrant': rec.get('quadrant', ''),
                    'answer_kind': v_data.get('answer_kind', 'other'),
                    'verbalize_correct': bool(v_data.get('verbalize_correct', False)),
                })

    if not rows:
        return

    sample_df = pd.DataFrame(rows)
    summary_path = Path(args.output_dir) / 'summary.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate per (condition, cutoff_pct, quadrant)
    agg_rows = []
    for (cond, cutoff_pct, quad), sub in sample_df.groupby(
            ['condition', 'cutoff_pct', 'quadrant']):
        n = len(sub)
        verb_acc = float(sub['verbalize_correct'].mean())
        insuff_rate = float((sub['answer_kind'] == 'insufficient').mean())
        num_rate = float((sub['answer_kind'] == 'numeric').mean())
        agg_rows.append({
            'model': slug, 'dataset': dataset, 'condition': cond,
            'cutoff_pct': cutoff_pct, 'quadrant': quad, 'n': n,
            'verb_accuracy': round(verb_acc, 4),
            'insuff_claim_rate': round(insuff_rate, 4),
            'numeric_rate': round(num_rate, 4),
        })

    # Also unconditional aggregate (across all quadrants)
    for (cond, cutoff_pct), sub in sample_df.groupby(['condition', 'cutoff_pct']):
        agg_rows.append({
            'model': slug, 'dataset': dataset, 'condition': cond,
            'cutoff_pct': cutoff_pct, 'quadrant': 'ALL', 'n': len(sub),
            'verb_accuracy': round(float(sub['verbalize_correct'].mean()), 4),
            'insuff_claim_rate': round(float((sub['answer_kind'] == 'insufficient').mean()), 4),
            'numeric_rate': round(float((sub['answer_kind'] == 'numeric').mean()), 4),
        })

    new_summary = pd.DataFrame(agg_rows)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Remove any prior rows for this (model, dataset) before appending fresh
        keep = ~((existing['model'] == slug) & (existing['dataset'] == dataset))
        existing = existing[keep]
        new_summary = pd.concat([existing, new_summary], ignore_index=True)
    new_summary.to_csv(summary_path, index=False)
    print(f"  -> global summary at {summary_path}")

    # Print the headline pivot for this run
    print(f"\n  --- {slug}/{dataset} verb_accuracy by condition x cutoff (ALL quadrants) ---")
    sub_all = new_summary[
        (new_summary['model'] == slug)
        & (new_summary['dataset'] == dataset)
        & (new_summary['quadrant'] == 'ALL')
    ]
    if not sub_all.empty:
        pivot = sub_all.pivot(index='condition', columns='cutoff_pct', values='verb_accuracy')
        print(pivot.round(3).to_string())

    # Cleanup model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# MAIN
# ============================================================================

def select_models(args):
    if args.model:
        return [args.model]
    return FULL_MODELS if args.all_models else REPRESENTATIVE_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None,
                        help='Single HF model id, e.g. Qwen/Qwen2.5-Math-1.5B-Instruct')
    parser.add_argument('--all_models', action='store_true',
                        help='Run on the full 21-model list (mirrors exp14).')
    parser.add_argument('--dataset', default=None, choices=DATASETS)
    parser.add_argument('--all_datasets', action='store_true')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--n_dim', type=int, default=200,
                        help='Number of train prompts per class for DIM.')
    parser.add_argument('--probe_sanity_n', type=int, default=200,
                        help='Number of test prompts used for probe-F1 sanity check.')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Forced-decode generation cap (matches exp14).')
    parser.add_argument('--min_cutoff_tokens', type=int, default=20,
                        help='Skip cutoffs that result in fewer than this many tokens.')
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--test', action='store_true',
                        help='Use a tiny subset (test_samples) for fast smoke testing.')
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    models = select_models(args)
    if args.all_datasets:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise ValueError("Use --dataset or --all_datasets")

    for model_name in models:
        for dataset in datasets:
            try:
                run_model_dataset(model_name, dataset, args)
            except Exception as e:
                print(f"\n[ERROR] {model_name}/{dataset}: {e}")
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()