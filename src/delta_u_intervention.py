"""
================================================================================
F10: Delta_u Intervention -- Positive Identification of the Abstention Direction
================================================================================

QUESTION (pre-registered):
  F7 establishes that v_probe has near-zero projection on the unembedding-side
  abstention-vs-numeric direction Delta_u. F9 shows that ablating or adding v_probe
  doesn't change behavior. The natural next question:

      Is Delta_u itself the direction that drives abstention in the unembedding?

  Headline test: inject alpha * Delta_u at L* (the probe's mid-network best layer) and
  let it propagate through subsequent attention/MLP/LN before reaching W_U. A
  positive result at L* identifies Delta_u as the abstention pathway in residual
  space at the layer where the recognition signal would be readable if the
  network used it. This is the result the paper rests on.

  Trivial control: injection at L (the final decoder block's output) directly
  shifts LN(h_L) along Delta_u and trivially biases the abstain-vs-numeric logit
  margin. A positive result at L is a wiring sanity check, NOT a mechanism
  finding. We run it to confirm the experimental machinery works.

  Either way, if Delta_u is in fact the direction the model uses for the
  abstain-vs-numeric decision at L*, we expect:
    - Increase abstention rate on Q1 (hallucinated insufficient samples), and
    - Not increase abstention rate on Q3 (correctly solved samples).

PRE-REGISTERED DECISION RULES (uses bootstrap 95% CI on dQ1, not point estimate):
  Clean positive identification at L*:
      Some alpha at layer='best' achieves, for direction='delta_u':
          dQ1 abstention rate (lower 95% CI)  >= +0.15
          dQ3 false-abstention (point)         <= +0.05  (Q3 doesn't collapse)
          Q1 coherence rate                    >= 0.80
      Verdict: 'CLEAN POSITIVE at L* -- Delta_u drives abstention at probe layer'.

  Positive at L only, null at L*:
      Verdict: 'POSITIVE at L only -- wiring sanity passes, but L* abstain is
      not Delta_u-aligned'. The activation-patching supplement (see below) then
      becomes the relevant test for whether abstention info is at L* as
      residual content without committing to a particular direction.

  Null on Delta_u at both layers:
      If other directions (v_probe, v_dim, v_rand) achieve threshold at L,
      verdict: 'NULL on Delta_u at both layers, but other directions work at L
      -- experiment is wired but Delta_u is the wrong direction'.
      Else verdict: 'NULL on everything -- possible experimental failure or
      coherence collapse'.

ACTIVATION PATCHING (prior-free supplement):
  For each Q1 prompt, find a matched Q2 prompt (same dataset, both
  is_sufficient=False, matched by token length within +/-20%). At the
  boxed-answer position, patch h^{L*} from the Q2 forward pass into the Q1
  forward pass. If Q1 flips to abstain, abstention information lives at L*
  as residual content (no direction commitment). Cleaner than Delta_u injection
  because it doesn't require us to specify a direction; complementary
  because it tests "is the info there?" rather than "is it in this
  direction?".

WHAT THIS SCRIPT DOES (per model, dataset):

  1. Load model + tokenizer + exp10 probe + best layer.
  2. Build T_abs (abstention-leading tokens) and T_num (numeric-leading tokens),
     same construction as probe_unembed_alignment.py.
  3. Compute four directions, all unit-normed:
        d_du     = Delta_u / ||Delta_u||,  where Delta_u = mean(W_U[T_abs]) -- mean(W_U[T_num])
        d_probe  = v_probe (from exp10 probe weight, StandardScaler-corrected)
        d_dim    = v_DIM at layer L* (data-derived diff-in-means at the probe layer)
        d_rand   = random Gaussian unit vector (seed-fixed)
  4. Load Q1 (hallucinated), Q3 (correctly solved) samples from baseline.
  5. For each direction x alpha x layer-choice:
        - Register a forward-hook on the chosen decoder layer that adds
          alpha * alpha_scale * direction to the residual stream at every
          position. alpha_scale = mean(||h||) / sqrt(d) so alpha=1 is "one
          per-dim residual magnitude".
        - Force-decode the boxed answer at cutoff 100% on all Q1 + Q3
          samples (same protocol as exp14 / causal_probe_test).
        - Classify the boxed content (insufficient / numeric / other) and
          check coherence.
        - Unregister hook.
  6. Aggregate to per-condition Q1/Q3 metrics. Apply pre-registered decision
     rule. Save summary + plot.

NOTES:
  - We test at TWO layer choices, with asymmetric paper-side roles:
        best   : the probe's best layer L* -- HEADLINE TEST. A positive
                 result here means Delta_u is the abstention direction in
                 residual space at the layer where v_probe is readable,
                 establishing positive identification of the mechanism.
        final  : the last decoder block's output -- TRIVIAL CONTROL. By
                 construction, adding alpha * Delta_u here directly shifts
                 LN(h_L) along Delta_u. A positive result is a wiring sanity,
                 not a mechanism finding.
    Reviewers can ask "why not layer X" -- all layers between L* and final
    either compress to one of these two, or are a follow-up.
  - HOOK SCOPE (alpha * v added at WHICH positions, controlled by --hook_scope):
        all  (default): every token position at the hooked layer's output.
                        Matches F9 (causal_probe_test) protocol. More
                        lenient test: Delta_u gets multiple opportunities to
                        propagate through subsequent attention/MLP. The
                        intervention is non-local at L* (perturbs the
                        K/V cache of the prompt + CoT).
        last         : only the boxed-answer position. Surgical, but a
                        weaker intervention since Delta_u only flows forward
                        through the layers above, not back through attention.
                        Use to sanity-check that 'all' positives aren't
                        artifacts of the K/V perturbation.
    The default is 'all' for headline reporting; 'last' is the robustness
    check. Disclose this choice in the paper.

DIRECTIONS SWEPT (per layer):
  delta_u   : Delta_u = mean(W_U[T_abs]) - mean(W_U[T_num]). The output-side
              direction whose alignment with LN(h_L) controls the abstain-
              vs-numeric logit margin. PRIMARY HYPOTHESIS.
  v_abstain : mean(h[boxed] | Q2) - mean(h[boxed] | Q1) at this layer
              (Marks & Tegmark diff-of-means). Both classes are insufficient;
              only behavior differs. Isolates the BEHAVIORAL DECISION, not
              input-class recognition. SECOND L*-headline candidate; cleaner
              than per-sample patching because averaging over many Q2/Q1
              pairs cancels prompt-specific semantic content.
  v_probe   : F7's recognition direction (probe normal at L*). Expected
              causally inert by F6/F7.
  v_dim     : t=0 diff-of-means insuff vs suff (input-class direction).
              Layer-specific.
  v_rand    : seed-fixed Gaussian unit vector. Null comparison.

ACTIVATION PATCHING SUPPLEMENT (optional, --do_patching):
  Per-sample patching of h^{L*}[boxed pos] from Q2 into Q1. Length-matched.
  Carries an unavoidable Q1/Q2 prompt-semantics confound (Q2 residual encodes
  not only the abstain decision but also Q2's semantic content); v_abstain
  in the main sweep is the cleaner aggregate test.

RUN:
  Smoke test (L*-headline at one model):
    python src/delta_u_intervention.py \\
        --model Qwen/Qwen2.5-Math-1.5B-Instruct --dataset umwp \\
        --n_q1 30 --n_q3 30 --n_q2 30 --layers best --alphas 1 2 4 8
  Full pre-registered sweep (5 models, both datasets, L* + L, all directions):
    nohup python src/delta_u_intervention.py --all_models --all_datasets \\
        --alphas 0.5 1 2 4 8 --layers best final \\
        > delta_u_run.log 2>&1 &
    echo $! > delta_u_run.pid
  With activation-patching supplement (slower; one extra Q2 forward pass per
  patch pair):
    python src/delta_u_intervention.py --all_models --all_datasets \\
        --alphas 0.5 1 2 4 8 --layers best final \\
        --do_patching --n_patch_pairs 50 \\
        > delta_u_run.log 2>&1 &
  Robustness: rerun with --hook_scope last to confirm 'all'-scope positives
  aren't artifacts of K/V cache perturbation.
================================================================================
"""

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


SOURCE_BASE = '/export/fs06/hwang302/CARDS'
OUTPUT_BASE = '/home/hwang302/.local/nlp/CARDS/experiment_result/causal_results'
EXP10_DIR = os.path.join(SOURCE_BASE, 'exp_temporal_new')

CUTOFFS_TO_TEST = [1.0]  # only cutoff 100% (the decision moment); can add more if needed
FORCE_DECODE_SUFFIX = "\n\n**Final Answer**\n\\boxed{"
MAX_NEW_TOKENS = 50

DEFAULT_MODELS = [
    'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'google/gemma-3-27b-it',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
]

ABSTENTION_PHRASES = [
    "Insufficient", "Cannot", "Unable", "Unknown", "Indeterminate",
    "Impossible", "Undefined", "Undetermined", "Missing", "Not",
    "Ins", "Insuf",
]


# =============================================================================
# Token sets, unembedding utilities
# =============================================================================
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


def get_final_norm_and_unembed(model):
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        norm = model.transformer.ln_f
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        norm = model.model.language_model.norm
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'norm'):
        norm = model.language_model.norm
    else:
        raise RuntimeError("Could not locate final norm")
    if hasattr(model, 'lm_head'):
        W_U = model.lm_head.weight
    else:
        raise RuntimeError("Could not locate lm_head")
    return norm, W_U


def get_layer_modules(model):
    """Robust layer-finder. Same as causal_probe_test.py."""
    import torch.nn as nn
    explicit_paths = [
        lambda m: m.model.layers,
        lambda m: m.layers,
        lambda m: m.transformer.h,
        lambda m: m.transformer.layers,
        lambda m: m.gpt_neox.layers,
        lambda m: m.model.language_model.model.layers,
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
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 5:
            if best is None or len(mod) > len(best):
                best = mod
    if best is not None:
        return best
    raise ValueError("Could not locate transformer layers")


# =============================================================================
# Forward hook for addition steering
# =============================================================================
class AdditionHook:
    """Add alpha_raw * v at the hooked layer's output.

    scope='all'  : add at every token position. Lenient intervention; matches
                   the F9 / causal_probe_test protocol. The default.
    scope='last' : add only at the final token position (the boxed-answer
                   position during force-decode). Surgical; weaker.

    v is unit-norm (np.ndarray, shape (d,))."""
    def __init__(self, v, alpha_raw, scope='all'):
        assert scope in ('all', 'last'), f"unknown scope: {scope}"
        self.v = v
        self.alpha_raw = float(alpha_raw)
        self.scope = scope
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = torch.tensor(self.v, dtype=hs.dtype, device=hs.device)
        if self.scope == 'all':
            modified = hs + self.alpha_raw * self._cached
        else:  # last
            modified = hs.clone()
            modified[:, -1, :] = modified[:, -1, :] + self.alpha_raw * self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


class PatchingHook:
    """Replace the last-position residual at the hooked layer with a cached
    donor vector (e.g. h^{L*} from a matched Q2 prompt).

    Used by run_activation_patching: for each Q1 prompt we run a Q2 forward
    pass once to extract h_q2 at the boxed-answer position, then patch h_q2
    into Q1's forward pass at the same layer. If Q1 flips to abstain, the
    abstention info at L* exists as residual content without commitment to
    a specific direction.
    """
    def __init__(self, h_donor):
        self.h_donor = h_donor  # 1D np.ndarray, shape (d,)
        self._cached = None

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._cached is None or self._cached.device != hs.device or self._cached.dtype != hs.dtype:
            self._cached = torch.tensor(self.h_donor, dtype=hs.dtype, device=hs.device)
        modified = hs.clone()
        modified[:, -1, :] = self._cached
        return (modified,) + output[1:] if isinstance(output, tuple) else modified


# =============================================================================
# Cutoff utilities (verbatim from causal_probe_test.py)
# =============================================================================
@dataclass
class CutoffResult:
    text: str
    token_count: int
    total_tokens: int


def _token_ids(tok, text):
    return tok(text, add_special_tokens=False)['input_ids']


def _sentence_boundary_candidates(text):
    candidates = []
    for m in re.finditer(r'\n{2,}', text):           candidates.append((m.end(), 0))
    for m in re.finditer(r'[.!?](?:\s+)(?=[A-Z]|$)', text):
        candidates.append((m.end(), 1))
    for m in re.finditer(r'[.!?]\n', text):          candidates.append((m.end(), 2))
    for m in re.finditer(r'[.!?]', text):             candidates.append((m.end(), 3))
    best = {}
    for pos, prio in candidates:
        if pos not in best or prio < best[pos]:
            best[pos] = prio
    return [(pos, prio) for pos, prio in best.items()]


def find_cutoff(text, target_pct, tok):
    ids = _token_ids(tok, text)
    total = len(ids)
    if total == 0:                       return CutoffResult('', 0, 0)
    if target_pct <= 0.0:                 return CutoffResult('', 0, total)
    target = max(1, min(total, int(target_pct * total)))
    if target_pct >= 1.0:                 return CutoffResult(text, total, total)
    cands = _sentence_boundary_candidates(text)
    if not cands:
        return CutoffResult(tok.decode(ids[:target], skip_special_tokens=True), target, total)
    scored = []
    for pos, prio in cands:
        ttext = text[:pos].rstrip()
        ttok = len(_token_ids(tok, ttext))
        if ttok == 0:
            continue
        scored.append((abs(ttok - target), prio, ttok, pos))
    if not scored:
        return CutoffResult(tok.decode(ids[:target], skip_special_tokens=True), target, total)
    _, _, actual, char_pos = min(scored, key=lambda x: (x[0], x[1]))
    return CutoffResult(text[:char_pos].rstrip(), actual, total)


# =============================================================================
# Forced-decode generation (same as causal_probe_test.py)
# =============================================================================
class StopOnCloseBrace(StoppingCriteria):
    def __init__(self, tokenizer, prompt_lens):
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens
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
def batched_forced_decode(model, tokenizer, forced_inputs, batch_size=8, max_new_tokens=MAX_NEW_TOKENS):
    results = [None] * len(forced_inputs)
    for i in range(0, len(forced_inputs), batch_size):
        batch = forced_inputs[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
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


# =============================================================================
# Answer classification
# =============================================================================
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


def is_coherent_forced(continuation, boxed):
    """Coherence check appropriate for force-decoded \\boxed{...} outputs.

    Force-decoded outputs are short by design ("5}", "Insufficient}", etc.).
    Word-count-based coherence is the wrong tool. We check instead for
    "model went off the rails" signatures: empty boxed, runaway length,
    extreme character repetition inside the boxed content, or exotic
    unicode (> U+FFFF) that almost never appears in well-formed math
    answers or English abstention phrases.
    """
    from collections import Counter
    s = continuation.strip()
    if boxed is None or not boxed.strip():
        return False
    if len(s) > 80:                                # runaway generation
        return False
    b = boxed.strip()
    if len(b) >= 6:
        c = Counter(b)
        if max(c.values()) / len(b) > 0.7:         # extreme repetition
            return False
    if any(ord(ch) > 0xFFFF for ch in b[:20]):     # exotic unicode (emoji, rare scripts)
        return False
    return True


# =============================================================================
# Data loading
# =============================================================================
def load_quadrant_samples(model_slug, dataset, max_q1=200, max_q3=100,
                          max_q2=100, max_q4=100):
    """Load Q1/Q2/Q3/Q4 samples from evaluated traces.

    Q1 (hallucinated, insufficient input + numeric answer) and Q3 (correctly
    solved, sufficient input) drive the pre-registered decision. Q2 (correctly
    abstained, insufficient input + abstention) and Q4 (over-cautious, sufficient
    input + abstention) are descriptive sanity:
      - Q2 baseline insuff_rate is already high; Delta_u injection should not
        substantially decrease it (sanity that we're not wrecking abstention).
      - Q4 baseline insuff_rate is already high; Delta_u injection should at most
        keep it high (consistency).
    Q2/Q4 are NOT used in the verdict -- only in the per-quadrant report.
    """
    eval_path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json')
    gen_path = os.path.join(SOURCE_BASE,
        f'experiments/dynamic_tracking_test/math/{model_slug}/{dataset}_cot_generations.json')
    if not os.path.exists(eval_path) or not os.path.exists(gen_path):
        return None, None, None, None
    with open(eval_path) as f:
        eval_data = json.load(f).get('data', [])
    with open(gen_path) as f:
        gen_data = json.load(f)
    q1, q2, q3, q4 = [], [], [], []
    caps = {
        'Q1_Hallucination':       (q1, max_q1),
        'Q2_Correct_Rejection':   (q2, max_q2),
        'Q3_Solved_Correctly':    (q3, max_q3),
        'Q4_Competence_Failure':  (q4, max_q4),
    }
    for idx, (e, g) in enumerate(zip(eval_data, gen_data)):
        quad = e.get('epistemic_quadrant', '')
        if quad not in caps:
            continue
        bucket, cap = caps[quad]
        if len(bucket) >= cap:
            continue
        bucket.append({
            'sample_id': str(e.get('question_idx', e.get('sample_id', idx))),
            'prompt': g.get('prompt'),
            'cot': g.get('generated_response', g.get('model_output', '')),
            'is_sufficient': g.get('is_sufficient', True),
            'quadrant': quad,
        })
        if all(len(b) >= c for b, c in caps.values()):
            break
    return q1, q2, q3, q4


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


# =============================================================================
# Direction computation
# =============================================================================
def probe_normal_direction(probe):
    scaler = probe.named_steps['standardscaler']
    clf = probe.named_steps['logisticregression']
    W = clf.coef_[0] / scaler.scale_
    return (W / np.linalg.norm(W)).astype(np.float32)


def random_direction(d, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(d).astype(np.float32)
    return v / np.linalg.norm(v)


@torch.no_grad()
def extract_t0_states(model, tokenizer, prompts, target_layer, batch_size=4):
    states = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        last_idx = (enc['attention_mask'].sum(dim=1) - 1).tolist()
        h = out.hidden_states[target_layer]
        for b, idx in enumerate(last_idx):
            states.append(h[b, idx, :].detach().to(torch.float32).cpu().numpy())
        del out
    return np.array(states)


@torch.no_grad()
def compute_abstain_direction(model, tokenizer, q1_samples, q2_samples,
                              target_layer, force_decode_suffix,
                              max_per_class=100, batch_size=4):
    """v_abstain_at_layer = mean(h[boxed pos] | Q2) - mean(h[boxed pos] | Q1).

    Both Q1 and Q2 are insufficient inputs; only their behavior differs (Q1
    hallucinates a number, Q2 correctly abstains). Averaging over many samples
    at the boxed-answer position cancels prompt-specific semantic content and
    leaves the shared behavioral signature.

    Distinct from v_DIM (which is insuff_class - suff_class at t=0, an input-
    class direction). v_abstain isolates the *behavioral decision* at L*.

    Marks & Tegmark (2024) style: diff-of-means in residual space.
    """
    def boxed_pos_residuals(samples):
        residuals = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            forced_inputs = []
            for s in batch:
                c = find_cutoff(s['cot'], 1.0, tokenizer)
                forced_inputs.append(s['prompt'] + c.text + force_decode_suffix)
            enc = tokenizer(forced_inputs, return_tensors='pt', padding=True,
                            truncation=False).to(model.device)
            out = model(**enc, output_hidden_states=True)
            last_idx = (enc['attention_mask'].sum(dim=1) - 1).tolist()
            h = out.hidden_states[target_layer]
            for b, idx in enumerate(last_idx):
                residuals.append(h[b, idx, :].detach().to(torch.float32).cpu().numpy())
            del out
        return np.array(residuals)

    q1_use = q1_samples[:max_per_class]
    q2_use = q2_samples[:max_per_class]
    if len(q1_use) < 30 or len(q2_use) < 30:
        return None, {'n_q1_used': len(q1_use), 'n_q2_used': len(q2_use),
                      'skipped': 'fewer than 30 samples per class'}
    h_q1 = boxed_pos_residuals(q1_use)
    h_q2 = boxed_pos_residuals(q2_use)
    v_raw = h_q2.mean(axis=0) - h_q1.mean(axis=0)
    norm = float(np.linalg.norm(v_raw))
    v = (v_raw / norm).astype(np.float32) if norm > 0 else None
    return v, {'n_q1_used': len(q1_use), 'n_q2_used': len(q2_use), 'raw_norm': norm}


@torch.no_grad()
def compute_layer_norm_stats(model, tokenizer, prompts, target_layer, batch_size=4):
    """Returns mean ||h|| at target_layer over the given prompts."""
    norms = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=False).to(model.device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[target_layer]
        per_tok = h.norm(dim=-1).flatten().to(torch.float32).cpu().numpy()
        norms.extend(per_tok.tolist())
        del out
    return float(np.mean(norms))


# =============================================================================
# Statistical uncertainty (Issue 3)
# =============================================================================
def bootstrap_ci(per_sample, quadrant, target_kind='insufficient',
                 n_boot=1000, seed=42):
    """Bootstrap 95% CI on the rate of `answer_kind == target_kind` within
    `quadrant`. Returns (lower, mean, upper). None if no samples."""
    sub = [1 if p['answer_kind'] == target_kind else 0
           for p in per_sample if p['quadrant'] == quadrant]
    if not sub:
        return None, None, None
    rng = np.random.RandomState(seed)
    sub_arr = np.array(sub)
    boots = np.array([rng.choice(sub_arr, size=len(sub_arr), replace=True).mean()
                      for _ in range(n_boot)])
    return (float(np.percentile(boots, 2.5)),
            float(boots.mean()),
            float(np.percentile(boots, 97.5)))


# =============================================================================
# Activation patching (Issue 2): prior-free supplement to Delta_u injection
# =============================================================================
@torch.no_grad()
def extract_hidden_at_last_pos(model, tokenizer, text, target_layer):
    """Return h^{target_layer} at the last token position for a single text."""
    enc = tokenizer(text, return_tensors='pt', truncation=False).to(model.device)
    out = model(**enc, output_hidden_states=True)
    last_idx = int(enc['attention_mask'].sum(dim=1).item() - 1)
    h = out.hidden_states[target_layer][0, last_idx, :].detach().to(torch.float32).cpu().numpy()
    del out
    return h


def match_q1_to_q2_by_length(q1_samples, q2_samples, tokenizer, length_tol=0.2):
    """Greedy match each Q1 prompt to its closest-length Q2 prompt (no
    replacement) where length differs by at most `length_tol` (relative).
    Returns list of (q1_item, q2_item) tuples."""
    def plen(s):
        return len(tokenizer(s['prompt'], add_special_tokens=False)['input_ids'])
    q1_lens = [(i, plen(s)) for i, s in enumerate(q1_samples)]
    q2_lens = [(j, plen(s)) for j, s in enumerate(q2_samples)]
    used = set()
    pairs = []
    # sort Q1 by length so longest find their match first (similar lengths
    # are rarer at the tails)
    for i, L1 in sorted(q1_lens, key=lambda x: x[1]):
        best_j, best_diff = None, None
        for j, L2 in q2_lens:
            if j in used:
                continue
            diff = abs(L1 - L2) / max(L1, 1)
            if diff > length_tol:
                continue
            if best_diff is None or diff < best_diff:
                best_j, best_diff = j, diff
        if best_j is not None:
            used.add(best_j)
            pairs.append((q1_samples[i], q2_samples[best_j]))
    return pairs


def run_activation_patching(model, tokenizer, layer_modules, target_layer,
                            q1_samples, q2_samples, force_decode_suffix,
                            max_pairs=50, batch_size=1):
    """For each matched (Q1, Q2) pair, patch h^{target_layer} from Q2's
    boxed-position into Q1's forward pass at the same position. Measure
    Q1 flip rate to abstention.

    Returns dict with: pairs_attempted, n_flipped, flip_rate, baseline_rate,
    bootstrap_ci, per_pair_records.
    """
    pairs = match_q1_to_q2_by_length(q1_samples, q2_samples, tokenizer)[:max_pairs]
    if not pairs:
        return {'pairs_attempted': 0, 'flip_rate': None,
                'note': 'no length-matched Q1/Q2 pairs available'}

    # Baseline: force-decode Q1 with no patching, record kinds.
    q1_baseline_inputs = []
    for q1, _ in pairs:
        c = find_cutoff(q1['cot'], 1.0, tokenizer)
        q1_baseline_inputs.append(q1['prompt'] + c.text + force_decode_suffix)
    baseline_gens = batched_forced_decode(model, tokenizer, q1_baseline_inputs,
                                          batch_size=batch_size,
                                          max_new_tokens=MAX_NEW_TOKENS)
    baseline_kinds = [classify_answer(extract_boxed(g['raw_output']))
                      for g in baseline_gens]
    baseline_abst = sum(1 for k in baseline_kinds if k == 'insufficient')

    # Patching: extract h_q2, then force-decode Q1 with PatchingHook.
    records = []
    n_flipped = 0
    for idx, (q1, q2) in enumerate(tqdm(pairs, desc='  activation patching')):
        c2 = find_cutoff(q2['cot'], 1.0, tokenizer)
        q2_input = q2['prompt'] + c2.text + force_decode_suffix
        try:
            h_q2 = extract_hidden_at_last_pos(model, tokenizer, q2_input, target_layer)
        except Exception as e:
            records.append({'q1_id': q1['sample_id'], 'q2_id': q2['sample_id'],
                            'baseline_kind': baseline_kinds[idx],
                            'patched_kind': 'error', 'error': str(e)})
            continue
        c1 = find_cutoff(q1['cot'], 1.0, tokenizer)
        q1_input = q1['prompt'] + c1.text + force_decode_suffix

        hook = PatchingHook(h_q2)
        handle = layer_modules[target_layer].register_forward_hook(hook)
        try:
            gen = batched_forced_decode(model, tokenizer, [q1_input],
                                        batch_size=1, max_new_tokens=MAX_NEW_TOKENS)
        finally:
            handle.remove()
        kind = classify_answer(extract_boxed(gen[0]['raw_output']))
        flipped = (baseline_kinds[idx] != 'insufficient' and kind == 'insufficient')
        if flipped:
            n_flipped += 1
        records.append({'q1_id': q1['sample_id'], 'q2_id': q2['sample_id'],
                        'baseline_kind': baseline_kinds[idx],
                        'patched_kind': kind, 'flipped_to_abstain': bool(flipped)})

    n = len(pairs)
    flip_rate = n_flipped / n if n else None
    # bootstrap CI on flip rate
    flip_arr = np.array([1 if r.get('flipped_to_abstain') else 0 for r in records])
    rng = np.random.RandomState(42)
    if len(flip_arr) > 0:
        boots = np.array([rng.choice(flip_arr, size=len(flip_arr), replace=True).mean()
                          for _ in range(1000)])
        ci = (float(np.percentile(boots, 2.5)),
              float(boots.mean()),
              float(np.percentile(boots, 97.5)))
    else:
        ci = (None, None, None)
    return {
        'pairs_attempted': n,
        'n_flipped_to_abstain': int(n_flipped),
        'flip_rate': flip_rate,
        'flip_rate_ci_95': {'lower': ci[0], 'mean': ci[1], 'upper': ci[2]},
        'baseline_q1_abst_rate': baseline_abst / n if n else None,
        'per_pair': records,
    }


# =============================================================================
# Main run
# =============================================================================
def run_one(model_name, dataset, args):
    slug = model_name.split('/')[-1]
    out_dir = Path(OUTPUT_BASE) / slug / dataset / 'delta_u'
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / 'DONE'
    if done.exists() and not args.force:
        print(f"[DONE] {slug}/{dataset}/delta_u -- skipping")
        return

    # ---- resolve exp10 probe + best layer ----
    exp10_csv = os.path.join(EXP10_DIR, 'results', f'exp10_ultimate_proportional_{dataset}.csv')
    if not os.path.exists(exp10_csv):
        print(f"[skip] missing exp10 csv"); return
    df10 = pd.read_csv(exp10_csv)
    row = df10[(df10['Model'] == slug) & (df10['Percentage'] == '0%')]
    if row.empty:
        print(f"[skip] no exp10 entry for {slug}/{dataset}"); return
    best_layer = int(row['Optimal_Layer'].iloc[0])
    probe_path = os.path.join(EXP10_DIR, 'probes_proportional', dataset, slug,
                              f'unified_probe_layer{best_layer}.joblib')
    if not os.path.exists(probe_path):
        print(f"[skip] missing probe at {probe_path}"); return
    probe = joblib.load(probe_path)
    v_probe = probe_normal_direction(probe)
    hidden_dim = v_probe.shape[0]

    # ---- load samples (Q1/Q3 drive the verdict; Q2/Q4 are descriptive) ----
    q1_samples, q2_samples, q3_samples, q4_samples = load_quadrant_samples(
        slug, dataset,
        max_q1=args.n_q1, max_q2=args.n_q2,
        max_q3=args.n_q3, max_q4=args.n_q4,
    )
    if not q1_samples:
        print(f"[skip] no Q1 samples"); return
    print(f"[RUN]  {slug}/{dataset}/delta_u   L*={best_layer}   "
          f"nQ1={len(q1_samples)} nQ2={len(q2_samples)} "
          f"nQ3={len(q3_samples)} nQ4={len(q4_samples)}")

    # ---- load model ----
    print("  loading model...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    n_gpus = torch.cuda.device_count()
    max_mem = {i: '78GiB' for i in range(n_gpus)} if n_gpus else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map='auto', max_memory=max_mem,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation='sdpa',
    )
    model.eval()
    layer_modules = get_layer_modules(model)
    n_layers = len(layer_modules)

    # ---- build Delta_u, v_DIM, v_random ----
    abs_ids = build_abstention_token_ids(tok)
    num_ids = build_numeric_token_ids(tok)
    _, W_U = get_final_norm_and_unembed(model)
    W_U_f32 = W_U.detach().to(torch.float32)
    delta_u_raw = W_U_f32[abs_ids].mean(dim=0) - W_U_f32[num_ids].mean(dim=0)
    delta_u_norm = float(delta_u_raw.norm())
    d_du = (delta_u_raw / delta_u_raw.norm()).cpu().numpy().astype(np.float32)
    d_probe = v_probe
    d_rand = random_direction(hidden_dim, seed=42)

    # ---- which layer indices to test ----
    layer_choices = []
    for L in args.layers:
        if L == 'final':
            layer_choices.append(('final', n_layers - 1))
        elif L == 'best':
            layer_choices.append(('best', best_layer))
        elif L.isdigit():
            li = int(L)
            if 0 <= li < n_layers:
                layer_choices.append((f'layer{li}', li))

    # ---- per-layer alpha_scale and per-layer v_DIM (design issues 1+2 fix) ----
    # Residual stream norm grows with depth; alpha=1 means the SAME relative
    # perturbation only if alpha_scale is computed at the same layer where the
    # intervention applies. Likewise v_DIM is a data-driven direction defined
    # at the layer where it's measured; using L*-trained v_DIM at the final
    # layer has no principled interpretation. We compute both per layer.
    norm_pool = ([s['prompt'] for s in q1_samples[:50]] +
                 [s['prompt'] for s in q3_samples[:50]])[:100]
    suff_p, insuff_p = load_train_balanced_prompts(slug, dataset, args.n_dim)
    have_dim_data = bool(suff_p and insuff_p and len(suff_p) >= 50 and len(insuff_p) >= 50)

    alpha_scale_per_layer = {}
    v_dim_per_layer = {}
    v_abstain_per_layer = {}
    v_abstain_meta_per_layer = {}
    have_abstain_data = len(q1_samples) >= 30 and len(q2_samples) >= 30
    for layer_name, layer_idx in layer_choices:
        mean_h = compute_layer_norm_stats(model, tok, norm_pool, layer_idx, batch_size=4)
        alpha_scale_per_layer[layer_name] = float(mean_h / np.sqrt(hidden_dim))
        if have_dim_data:
            X_suff = extract_t0_states(model, tok, suff_p, layer_idx)
            X_insuff = extract_t0_states(model, tok, insuff_p, layer_idx)
            v_dim_raw = X_insuff.mean(axis=0) - X_suff.mean(axis=0)
            v_dim_per_layer[layer_name] = (v_dim_raw / np.linalg.norm(v_dim_raw)).astype(np.float32)
        else:
            v_dim_per_layer[layer_name] = None
        # v_abstain = mean(h[boxed] | Q2) - mean(h[boxed] | Q1), at this layer.
        # Both classes are insufficient inputs; only behavior differs.
        if have_abstain_data:
            v_abs, abs_meta = compute_abstain_direction(
                model, tok, q1_samples, q2_samples, layer_idx,
                FORCE_DECODE_SUFFIX,
                max_per_class=args.n_abstain_per_class, batch_size=4,
            )
            v_abstain_per_layer[layer_name] = v_abs
            v_abstain_meta_per_layer[layer_name] = abs_meta
        else:
            v_abstain_per_layer[layer_name] = None
            v_abstain_meta_per_layer[layer_name] = {'skipped': 'need >=30 Q1 and Q2 samples'}
    if not have_dim_data:
        print("  ! insufficient train data, v_DIM will be skipped at all layers")
    if not have_abstain_data:
        print(f"  ! insufficient Q1/Q2 samples (n_q1={len(q1_samples)}, n_q2={len(q2_samples)}), "
              f"v_abstain will be skipped at all layers")
    print(f"  |T_abs|={len(abs_ids)} |T_num|={len(num_ids)} ||delta_u||={delta_u_norm:.3f}")
    for name, scale in alpha_scale_per_layer.items():
        print(f"    alpha_scale at layer '{name}' = {scale:.3f}")
        if v_abstain_per_layer.get(name) is not None:
            print(f"    v_abstain at layer '{name}': "
                  f"raw_norm={v_abstain_meta_per_layer[name]['raw_norm']:.3f} "
                  f"(n_q1={v_abstain_meta_per_layer[name]['n_q1_used']}, "
                  f"n_q2={v_abstain_meta_per_layer[name]['n_q2_used']})")

    # ---- direction registry (layer-independent directions only) ----
    base_directions = {'delta_u': d_du, 'v_probe': d_probe, 'v_rand': d_rand}

    # cosines for the meta file (v_DIM cosine reported at L*)
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    cosines = {
        'cos(delta_u, v_probe)': cos(d_du, d_probe),
        'cos(delta_u, v_rand)':  cos(d_du, d_rand),
    }
    if v_dim_per_layer.get('best') is not None:
        cosines['cos(delta_u, v_DIM_at_best)'] = cos(d_du, v_dim_per_layer['best'])
    if v_abstain_per_layer.get('best') is not None:
        cosines['cos(delta_u, v_abstain_at_best)']  = cos(d_du, v_abstain_per_layer['best'])
        cosines['cos(v_probe, v_abstain_at_best)']  = cos(d_probe, v_abstain_per_layer['best'])
        if v_dim_per_layer.get('best') is not None:
            cosines['cos(v_DIM_at_best, v_abstain_at_best)'] = cos(
                v_dim_per_layer['best'], v_abstain_per_layer['best'])
    print(f"  cosines: {cosines}")

    # ---- pre-compute forced inputs (cutoff=100%) ----
    # Q1/Q3 drive the decision rule; Q2/Q4 are descriptive sanity (no thresholds).
    all_samples = ([(s, 'Q1') for s in q1_samples] +
                   [(s, 'Q2') for s in q2_samples] +
                   [(s, 'Q3') for s in q3_samples] +
                   [(s, 'Q4') for s in q4_samples])
    forced_inputs = []
    for s, q in all_samples:
        c = find_cutoff(s['cot'], 1.0, tok)
        forced_inputs.append(s['prompt'] + c.text + FORCE_DECODE_SUFFIX)

    # ---- run baseline + conditions ----
    rows = []
    per_condition_samples = {}  # cond_name -> per_sample list (for bootstrap CIs)

    def run_condition(cond_name, layer_idx, direction_v, alpha_raw):
        """Register hook, generate, classify, record. alpha_raw=None means baseline (no hook).
        Hook scope (per-position vs last-only) is controlled by args.hook_scope."""
        if alpha_raw is not None and direction_v is not None:
            hook = AdditionHook(direction_v, alpha_raw, scope=args.hook_scope)
            handle = layer_modules[layer_idx].register_forward_hook(hook)
        else:
            handle = None
        try:
            gens = batched_forced_decode(model, tok, forced_inputs,
                                         batch_size=args.batch_size, max_new_tokens=MAX_NEW_TOKENS)
        finally:
            if handle is not None:
                handle.remove()

        # classify
        per_sample = []
        for (s, q), g in zip(all_samples, gens):
            boxed = extract_boxed(g['raw_output'])
            kind = classify_answer(boxed)
            coh = is_coherent_forced(g['continuation'], boxed)
            per_sample.append({'quadrant': q, 'answer_kind': kind, 'coherent': coh,
                               'boxed': boxed, 'sample_id': s['sample_id']})
        # aggregate per quadrant (Q1/Q2/Q3/Q4)
        agg = {}
        for q in ('Q1', 'Q2', 'Q3', 'Q4'):
            sub = [p for p in per_sample if p['quadrant'] == q]
            n = len(sub)
            if n == 0: continue
            insuff = sum(1 for p in sub if p['answer_kind'] == 'insufficient') / n
            num = sum(1 for p in sub if p['answer_kind'] == 'numeric') / n
            coh = sum(1 for p in sub if p['coherent']) / n
            agg[q] = {'n': n, 'insuff_rate': insuff, 'numeric_rate': num, 'coherence': coh}
        per_condition_samples[cond_name] = per_sample
        return agg, per_sample

    # ---- baseline ----
    print("  baseline (no hook) ...")
    base_agg, base_samples = run_condition('baseline', 0, None, None)
    for q in base_agg:
        rows.append({
            'condition': 'baseline', 'layer_name': '--', 'layer_idx': -1, 'direction': '--',
            'alpha': 0.0, 'quadrant': q, **base_agg[q],
        })
    base_line_parts = [f"{q} insuff={base_agg[q]['insuff_rate']:.3f}" for q in sorted(base_agg)]
    print("  baseline  " + "  ".join(base_line_parts))

    # ---- interventions ----
    # For each (layer, direction, alpha): build the direction dict for this layer.
    #   delta_u, v_probe, v_rand : layer-independent
    #   v_DIM                    : layer-specific (input-class diff-of-means at t=0)
    #   v_abstain                : layer-specific (Q2-vs-Q1 behavioral diff at boxed pos)
    total = 0
    for layer_name, _ in layer_choices:
        n_dirs = len(base_directions)
        if v_dim_per_layer.get(layer_name) is not None: n_dirs += 1
        if v_abstain_per_layer.get(layer_name) is not None: n_dirs += 1
        total += n_dirs * len(args.alphas)
    pbar = tqdm(total=total, desc='  interventions')
    for layer_name, layer_idx in layer_choices:
        layer_alpha_scale = alpha_scale_per_layer[layer_name]
        layer_directions = dict(base_directions)
        if v_dim_per_layer.get(layer_name) is not None:
            layer_directions['v_dim'] = v_dim_per_layer[layer_name]
        if v_abstain_per_layer.get(layer_name) is not None:
            layer_directions['v_abstain'] = v_abstain_per_layer[layer_name]
        for dir_name, d_vec in layer_directions.items():
            for alpha in args.alphas:
                alpha_raw = alpha * layer_alpha_scale
                cond_name = f'steer_{dir_name}_a{alpha}_at_{layer_name}'
                agg, _ = run_condition(cond_name, layer_idx, d_vec, alpha_raw)
                for q in agg:
                    rows.append({
                        'condition': cond_name,
                        'layer_name': layer_name, 'layer_idx': layer_idx,
                        'direction': dir_name, 'alpha': alpha,
                        'alpha_scale_used': layer_alpha_scale,
                        'quadrant': q, **agg[q],
                    })
                pbar.update(1)
    pbar.close()

    # ---- save ----
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'summary.csv', index=False)

    # ---- apply pre-registered decision rule ----
    # Q1 abstention should INCREASE by at least 0.15 (intervention works);
    # we now require this on the LOWER 95% bootstrap CI of dQ1, not point estimate.
    # Q3 abstention should NOT increase by more than 0.05 (specificity).
    # Q1 coherence >= 0.80 (intervention doesn't break generation).
    threshold_dq1_min = 0.15            # min dQ1 lower-CI threshold
    threshold_dq3_max = 0.05            # max dQ3 point-estimate threshold
    threshold_coh    = 0.80             # min Q1 coherence

    base_q1 = base_agg.get('Q1', {}).get('insuff_rate', 0)
    base_q3 = base_agg.get('Q3', {}).get('insuff_rate', 0)
    # Baseline CI on Q1 abstention rate (for the dQ1 CI calculation)
    base_q1_ci = bootstrap_ci(base_samples, 'Q1', target_kind='insufficient')

    clean_hits = []
    for _, r in df[df['condition'] != 'baseline'].iterrows():
        if r['quadrant'] != 'Q1':
            continue
        cond_name = r['condition']
        q3_row = df[(df['condition'] == cond_name) & (df['quadrant'] == 'Q3')]
        if q3_row.empty:
            continue
        q3_dq3 = q3_row.iloc[0]['insuff_rate'] - base_q3
        coh = r['coherence']
        # Bootstrap CI on the intervention's Q1 abstention rate
        cond_samples = per_condition_samples.get(cond_name, [])
        cond_q1_ci = bootstrap_ci(cond_samples, 'Q1', target_kind='insufficient')
        # Conservative dQ1 lower CI: subtract baseline upper CI from cond lower CI
        # (i.e., worst-case treatment effect within the joint CI).
        if cond_q1_ci[0] is None or base_q1_ci[2] is None:
            dq1_lower = float(r['insuff_rate']) - float(base_q1)
            dq1_point = dq1_lower
        else:
            dq1_lower = cond_q1_ci[0] - base_q1_ci[2]
            dq1_point = float(r['insuff_rate']) - float(base_q1)
        if (dq1_lower >= threshold_dq1_min and
            q3_dq3   <= threshold_dq3_max and
            coh      >= threshold_coh):
            clean_hits.append({
                'condition': cond_name, 'direction': r['direction'],
                'alpha': r['alpha'], 'layer_name': r['layer_name'],
                'delta_Q1_point': float(dq1_point),
                'delta_Q1_lower_CI': float(dq1_lower),
                'delta_Q3': float(q3_dq3),
                'Q1_coherence': float(coh),
            })

    # ---- verdict logic ----
    # L* headline = Delta_u at 'best'. v_abstain at 'best' is a second L* test that
    # isolates the behavioral decision from input-class recognition (Q2 vs Q1
    # diff-of-means at boxed position). L = 'final' is the trivial wiring control.
    du_lstar = [h for h in clean_hits
                if h['direction'] == 'delta_u' and h['layer_name'] == 'best']
    du_lfinal = [h for h in clean_hits
                 if h['direction'] == 'delta_u' and h['layer_name'] == 'final']
    abst_lstar = [h for h in clean_hits
                  if h['direction'] == 'v_abstain' and h['layer_name'] == 'best']
    if du_lstar and abst_lstar:
        verdict = ('CLEAN POSITIVE at L* (double confirmation) -- Delta_u AND '
                   'v_abstain both drive abstention at probe layer')
    elif du_lstar:
        verdict = ('CLEAN POSITIVE at L* -- Delta_u drives abstention at probe layer '
                   '(v_abstain null or absent)')
    elif abst_lstar:
        verdict = ('POSITIVE at L* via v_abstain only -- abstain pathway exists '
                   'at L* but is NOT in the Delta_u direction')
    elif du_lfinal:
        verdict = ('POSITIVE at L only -- wiring sanity passes, but L* abstain '
                   'is not Delta_u-aligned')
    else:
        any_at_lfinal = any(h['layer_name'] == 'final' for h in clean_hits)
        if any_at_lfinal:
            verdict = ('NULL on Delta_u and v_abstain at L*; other directions work '
                       'at L -- experiment is wired but L* mechanism unidentified')
        else:
            verdict = ('NULL on everything -- possible experimental failure or '
                       'coherence collapse')

    # ---- optional: activation patching supplement (Issue 2) ----
    patching_result = None
    if args.do_patching and q2_samples:
        target_layer = best_layer  # patching is the L* supplement
        print(f"  activation patching at L*={best_layer} "
              f"(n_pairs={min(args.n_patch_pairs, len(q1_samples), len(q2_samples))}) ...")
        patching_result = run_activation_patching(
            model, tok, layer_modules, target_layer,
            q1_samples, q2_samples, FORCE_DECODE_SUFFIX,
            max_pairs=args.n_patch_pairs, batch_size=1,
        )
        if patching_result.get('flip_rate') is not None:
            ci = patching_result['flip_rate_ci_95']
            print(f"    flip rate = {patching_result['flip_rate']:.3f} "
                  f"(95% CI [{ci['lower']:.3f}, {ci['upper']:.3f}], "
                  f"n={patching_result['pairs_attempted']})")

    # ---- meta + verdict ----
    meta = {
        'model': slug, 'dataset': dataset,
        'best_layer': best_layer, 'final_layer_idx': n_layers - 1,
        'hidden_dim': hidden_dim, 'n_layers': n_layers,
        'alpha_scale_per_layer': alpha_scale_per_layer,
        'T_abs_count': len(abs_ids), 'T_num_count': len(num_ids),
        'delta_u_norm': delta_u_norm,
        'cosines': cosines,
        'n_q1': len(q1_samples), 'n_q2': len(q2_samples),
        'n_q3': len(q3_samples), 'n_q4': len(q4_samples),
        'alphas': args.alphas, 'layer_choices': [name for name, _ in layer_choices],
        'hook_scope': args.hook_scope,
        'baseline_q1_insuff': float(base_q1),
        'baseline_q3_insuff': float(base_q3),
        'baseline_q1_insuff_ci_95': {'lower': base_q1_ci[0], 'mean': base_q1_ci[1],
                                     'upper': base_q1_ci[2]},
        'baseline_q1_coherence': float(base_agg.get('Q1', {}).get('coherence', 0)),
        'baseline_q3_coherence': float(base_agg.get('Q3', {}).get('coherence', 0)),
        'baseline_q2_insuff': float(base_agg.get('Q2', {}).get('insuff_rate', float('nan'))),
        'baseline_q4_insuff': float(base_agg.get('Q4', {}).get('insuff_rate', float('nan'))),
        'pre_registered_thresholds': {
            'min_delta_Q1_lower_CI': threshold_dq1_min,
            'max_delta_Q3_point':    threshold_dq3_max,
            'min_Q1_coherence':      threshold_coh,
        },
        'clean_hits': clean_hits,
        'hits_du_at_lstar_count':       len(du_lstar),
        'hits_du_at_lfinal_count':      len(du_lfinal),
        'hits_v_abstain_at_lstar_count': len(abst_lstar),
        'v_abstain_per_layer_meta':      v_abstain_meta_per_layer,
        'verdict': verdict,
        'activation_patching': patching_result,
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # ---- plot ----
    make_plot(df, base_agg, out_dir, slug, dataset, args.alphas)

    done.touch()
    print(f"  -> done. verdict: {meta['verdict']}")
    print(f"     clean hits: {len(clean_hits)}")
    for h in clean_hits[:5]:
        print(f"       {h}")

    del model, tok, W_U
    gc.collect()
    torch.cuda.empty_cache()


def make_plot(df, base_agg, out_dir, slug, dataset, alphas):
    """dQ1 and dQ3 by (direction, alpha) at each layer choice."""
    base_q1 = base_agg.get('Q1', {}).get('insuff_rate', 0)
    base_q3 = base_agg.get('Q3', {}).get('insuff_rate', 0)
    layer_names = sorted(df[df['layer_name'] != '--']['layer_name'].unique())
    direction_names = sorted(df['direction'].unique())
    direction_names = [d for d in direction_names if d != '--']

    fig, axes = plt.subplots(2, len(layer_names), figsize=(6 * len(layer_names), 8), sharex=True)
    if len(layer_names) == 1:
        axes = axes.reshape(2, 1)

    colors = {'delta_u':   '#d62728', 'v_probe': '#1f77b4',
              'v_dim':     '#2ca02c', 'v_rand':  '#7f7f7f',
              'v_abstain': '#9467bd'}
    for j, ln in enumerate(layer_names):
        for q_idx, q in enumerate(['Q1', 'Q3']):
            ax = axes[q_idx, j]
            for dname in direction_names:
                sub = df[(df['layer_name'] == ln) & (df['direction'] == dname) & (df['quadrant'] == q)].sort_values('alpha')
                if sub.empty: continue
                base = base_q1 if q == 'Q1' else base_q3
                ys = sub['insuff_rate'].values - base
                ax.plot(sub['alpha'].values, ys, marker='o', color=colors.get(dname, '#000'), label=dname)
            ax.axhline(0, color='black', lw=0.5)
            if q == 'Q1':
                ax.axhline(0.15, color='red', linestyle=':', label='dQ1 threshold = 0.15')
            elif q == 'Q3':
                ax.axhline(-0.05, color='red', linestyle=':', label='dQ3 threshold = -0.05')
            role = '(L* HEADLINE)' if ln == 'best' else '(L trivial control)' if ln == 'final' else ''
            ax.set_title(f'{slug}/{dataset}: d{q} insuff_rate at {ln} {role}')
            ax.set_xlabel('alpha (units of mean(||h||)/sqrt(d))')
            ax.set_ylabel(f'd{q} insuff_rate')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'fig_delta_u_intervention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved figure to {out_dir / 'fig_delta_u_intervention.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=None)
    p.add_argument('--all_models', action='store_true')
    p.add_argument('--dataset', default=None, choices=['umwp', 'treecut'])
    p.add_argument('--all_datasets', action='store_true')
    p.add_argument('--n_q1', type=int, default=200,
                   help='max Q1 (hallucinated, insufficient) samples (verdict-relevant).')
    p.add_argument('--n_q2', type=int, default=100,
                   help='max Q2 (correctly abstained) samples. Used for v_abstain '
                        'direction construction AND descriptive sanity (not in verdict).')
    p.add_argument('--n_q3', type=int, default=100,
                   help='max Q3 (correctly solved) samples (verdict-relevant: '
                        'specificity check).')
    p.add_argument('--n_q4', type=int, default=100,
                   help='max Q4 (over-cautious) samples. Descriptive only.')
    p.add_argument('--n_dim', type=int, default=200,
                   help='max samples per class for v_DIM at t=0.')
    p.add_argument('--n_abstain_per_class', type=int, default=100,
                   help='max Q1 and Q2 samples used to compute v_abstain '
                        '(diff-of-means at boxed-position).')
    p.add_argument('--alphas', type=float, nargs='+', default=[0.5, 1.0, 2.0, 4.0, 8.0],
                   help='alpha in units of per-layer typical-per-dim residual magnitude. '
                        'Pre-registered sweep is 0.5 1 2 4 8.')
    p.add_argument('--layers', type=str, nargs='+', default=['best', 'final'],
                   help='layer choices: "best" (L*, HEADLINE) and/or "final" '
                        '(L, trivial control), or integer indices. Order matters '
                        'only for display.')
    p.add_argument('--hook_scope', type=str, default='all', choices=['all', 'last'],
                   help='where the addition hook fires. "all" (default) adds at '
                        'every token position (matches F9 protocol; lenient). '
                        '"last" adds only at the boxed-answer position (surgical; '
                        'use as robustness check).')
    p.add_argument('--do_patching', action='store_true',
                   help='also run per-sample activation patching at L* (Q2 boxed-pos '
                        'residual patched into Q1 forward pass). Length-matched. '
                        'Carries Q2 prompt-semantic confound; the v_abstain direction '
                        'in the main sweep is the cleaner aggregate test.')
    p.add_argument('--n_patch_pairs', type=int, default=50,
                   help='max length-matched (Q1, Q2) pairs for activation patching.')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    if args.all_models:    models = DEFAULT_MODELS
    elif args.model:       models = [args.model]
    else:                  raise SystemExit("--model or --all_models required")

    if args.all_datasets:  datasets = ['umwp', 'treecut']
    elif args.dataset:     datasets = [args.dataset]
    else:                  raise SystemExit("--dataset or --all_datasets required")

    for m in models:
        for ds in datasets:
            try: run_one(m, ds, args)
            except Exception as e:
                print(f"\n[ERROR] {m}/{ds}: {e}")
                import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
