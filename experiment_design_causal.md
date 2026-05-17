# Experiment Design — Causal Test of the Insufficiency Direction

A single, principled experiment that unifies steering and ablation under one protocol, with a rank-K extension for the ablation arm and pre-registered decision rules. Replaces the current `exp13_steering_new.py` and `two_track_exp/x_not_equal_y.py` for paper-quality results.

---

## 1. What we are trying to prove

The probe finds a linearly separable direction in residual stream that distinguishes insufficient from sufficient inputs with F1 in the 0.70--0.90 range. The verbalization trajectory and the probe trajectory diverge (Findings 4 and 5). The question is:

> **Is the probe direction $v_Y$ a causal driver of behavioral abstention, or is it a passive observer with no causal role?**

We commit to two competing hypotheses and pre-register the test:

| | $H_\text{observer}$ | $H_\text{causal}$ |
|---|---|---|
| Adding $\alpha v_Y$ to residual stream | does not flip Q1 hallucinations to abstention | flips Q1 to abstention at small to moderate $\alpha$ |
| Ablating $v_Y$ across all layers | does not change Q1 / Q2 verbalization rates | reduces Q2 abstention rate |
| Ablating rank-K subspace containing $v_Y$ for small K | does not change verbalization rates even when probe F1 reaches chance | changes verbalization rates as soon as probe collapses |
| Behavioral effect of $v_Y$ interventions vs random direction | indistinguishable | substantially larger |

The current data already favors $H_\text{observer}$ but is not yet conclusive. This experiment is designed to make it conclusive.

---

## 2. Scope: which (model, dataset) pairs to run

Quality control eliminates the noise that ruined the previous experiments. Run only on pairs that satisfy all three filters:

1. **Non-degenerate probe**: `exp10 Unified_Test_F1` at 0% cutoff $\in [0.70, 0.95]$.
2. **Confounder-validated probe**: passes Finding 3 (D1 Mean_Gap $\ge 0.40$ on UMWP, D3 cosines $\le 0.10$).
3. **Sufficient Q1 / Q2 sample counts**: from exp2 evaluation, $n_\text{Q1} \ge 200$ and $n_\text{Q2} \ge 50$.

This yields a pre-defined subset, expected to be roughly 6--8 (model, dataset) pairs, all on UMWP plus the two Gemma models on TreeCut. **Drop everything else. Report the drop count transparently.**

Recommended initial models (subject to filter pass):

| Model | Size | Why |
|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 1.5B | Small, math-specialized, instruct |
| Qwen2.5-Math-7B-Instruct | 7B | Medium, math-specialized, instruct |
| Meta-Llama-3.1-8B | 8B | Medium general, base — for instruct comparison |
| Meta-Llama-3.1-8B-Instruct | 8B | Medium general, instruct |
| gemma-3-12b-it | 12B | Mid-large general, instruct |
| gemma-3-27b-it | 27B | Large general, instruct |
| DeepSeek-R1-Distill-Llama-70B | 70B | Large reasoning-distilled |

Datasets: UMWP first. TreeCut second on the subset that passes the filter.

---

## 3. The unified protocol

Three intervention families, one shared evaluation harness, one shared decision rule.

### 3.1 Directions tested

For each (model, dataset):

- **$v_Y$**: probe direction. StandardScaler-corrected weight from the exp10 unified probe, unit-normed. The direction of interest.
- **$v_\text{DIM}$**: diff-in-means direction. $\text{mean}(\text{insuff } t{=}0 \text{ acts}) - \text{mean}(\text{suff } t{=}0 \text{ acts})$, unit-normed. Orthogonal but class-correlated control.
- **$v_\text{rand}$**: random unit vector. Seed-fixed. Methodology control.
- **$\{v_Y^{(1)}, v_Y^{(2)}, \ldots, v_Y^{(K)}\}$ for $K \le 16$**: rank-K probe basis. Train $v_Y^{(1)} = v_Y$. Ablate $\{v_Y^{(1)}\}$ from training activations, retrain a new probe to get $v_Y^{(2)}$. Iterate. Each $v_Y^{(k)}$ is orthogonal to its predecessors by construction.

Record cosines: $\cos(v_Y, v_\text{DIM})$, $\cos(v_Y, v_\text{rand})$, and pairwise cosines among $\{v_Y^{(k)}\}$.

### 3.2 Intervention A: Steering (ADD test of sufficiency)

For each direction $v \in \{v_Y, v_\text{DIM}, v_\text{rand}\}$:

- **Layer sweep**: apply at four layers — early ($L \approx \tfrac{1}{6}$), middle ($L \approx \tfrac{1}{2}$), probe's best layer (from exp10), and late ($L \approx \tfrac{5}{6}$).
- **Position-restricted addition**: hook only triggers at the force-decode position (the suffix `\n\n**Final Answer**\n\boxed{`) and the few tokens generated after. This is more principled than current exp13 which adds at every position including prompt tokens.
- **Magnitude-normalized α**: define $\alpha=1$ as "one standard deviation of the per-layer residual stream norm in training distribution." Sweep $\alpha \in \{0.5, 1, 2, 4, 8\}$. Anchors the intervention magnitude in interpretable units instead of arbitrary scalar values.
- **Coherence-guarded reporting**: only count flips with $\text{coherence} > 0.80$. Otherwise label as "model broken at this α," do not interpret as causal effect.

Evaluation on $n_\text{Q1}=200$, $n_\text{Q3}=100$ samples (5x current sizes).

### 3.3 Intervention B: Rank-K Ablation (REMOVE test of necessity)

For each $K \in \{1, 2, 4, 8, 16\}$:

- Build subspace basis $V_K = \{v_Y^{(1)}, \ldots, v_Y^{(K)}\}$ (from §3.1).
- Hook every layer: project out $V_K$ from residual stream at every token position.
- Match with two control conditions:
  - Ablate $V_K^\text{rand}$ = K random orthogonal directions (controls for "any K-dim ablation breaks the model").
  - Ablate $V_K^\text{DIM}$ = $\{v_\text{DIM}, v_\text{DIM}^{(2)}, \ldots, v_\text{DIM}^{(K)}\}$ where the DIM basis is built analogously (compare K-rank causality of class-mean direction vs probe direction).

Evaluation: same as 3.2.

### 3.4 Shared evaluation: behavior at every cutoff

Reuse the exp14 force-decode protocol exactly: truncate CoT at sentence boundary near cutoff $\tau \in \{0, 20, 40, 60, 100\}$%, append `\n\n**Final Answer**\n\boxed{`, greedy-decode 50 tokens, judge with LLM.

For each (intervention condition, cutoff, sample):

- `boxed_content`: extracted boxed answer
- `answer_kind`: numeric / insufficient / other (classified via regex, identical to current SB)
- `is_correct`: judge-graded against ground truth, identical to exp14
- `coherence`: standard repetition-rate check
- `quadrant_observed`: derived

Aggregate to per-quadrant rates (Q1 abstention rate, Q3 false abstention rate, etc.) per condition and cutoff.

### 3.5 Probe sanity check (run for every intervention condition)

For each intervention condition, extract $t{=}0$ residual states **with the intervention hook active**, run them through the exp10 unified probe, compute F1 against ground-truth labels. This tells us whether the intervention actually moved the probe-readable signal.

A condition is **interpretable** only if:
- For ablation conditions: baseline F1 $\ge 0.65$ AND $|\Delta \text{F1}| \ge 0.10$ vs baseline.
- For addition conditions: $\Delta \text{F1} \ne 0$ (the addition produces non-trivial activation shift).

Report uninterpretable conditions as such; do not include them in headline statistics.

---

## 4. Pre-registered decision rules

We commit to these interpretations *before* looking at results.

### 4.1 Single-direction observer test ($K=1$)

A. **Confirms $H_\text{observer}$** if both hold on $\ge 70\%$ of in-scope (model, dataset) pairs:
- Probe sanity check confirms intervention works (F1 drops $\ge 0.10$ for ablate_y; F1 changes for add).
- Mean $|\Delta Q1\text{-abstention-rate}|$ for $v_Y$ interventions is within $\pm 0.05$ of the same quantity for $v_\text{rand}$ interventions.

B. **Confirms $H_\text{causal}$ (single-direction)** if both hold:
- Probe sanity works.
- Mean $|\Delta Q1\text{-abstention-rate}|$ for $v_Y$ is at least $0.10$ larger than for $v_\text{rand}$, in the predicted direction (ADD increases Q1, ABLATE decreases Q2).

C. **Methodology issue (cannot interpret)** if:
- $v_\text{rand}$ ablation produces $|\Delta| > 0.10$ on Q3. The protocol is too destructive.

### 4.2 Rank-K test ($K \in \{1, 2, 4, 8, 16\}$)

Let $K^*_\text{probe}$ = smallest $K$ where ablating $V_K$ reduces probe F1 to within $\pm 0.05$ of chance (0.5).
Let $K^*_\text{behavior}$ = smallest $K$ where ablating $V_K$ produces $|\Delta Q1| > 0.10$.

| Pattern | Conclusion |
|---|---|
| $K^*_\text{probe} \le 4$ and $K^*_\text{behavior} > K^*_\text{probe}$ by $\ge 4$ | Probe-readable signal is low-dim; behavior is decoupled from it. **Strong $H_\text{observer}$.** |
| $K^*_\text{probe} \le 4$ and $K^*_\text{behavior} \le K^*_\text{probe}$ | Probe direction is causal, just multi-dim. **Multi-dim $H_\text{causal}$.** |
| $K^*_\text{probe} > 8$ | Probe signal is highly distributed. Single-direction methodology in original SB and steering was inadequate. Report this honestly. |
| Rank-K random ablation at matched $K$ produces $|\Delta Q1| > 0.05$ | Protocol too destructive at this $K$. Cap $K$ at the largest where random is still null. |

### 4.3 Steering test

Define a "clean intervention" as $\Delta Q1 \ge +0.10$ AND $\Delta Q3 \le +0.05$ AND coherence $\ge 0.80$.

- Pre-registered prediction under $H_\text{observer}$: **zero clean interventions for $v_Y$ across all layers, all $\alpha$, all in-scope (model, dataset) pairs.**
- $v_\text{DIM}$ may produce clean interventions in some configurations. This is consistent with $H_\text{observer}$ for $v_Y$ specifically, not against it.

---

## 5. Artifacts to save (for future experiments)

This is critical. The experiment must save everything needed to run DLA, activation patching, and SAE analysis later without re-running the model.

For each (model, dataset, sample, cutoff):

### 5.1 Residual stream caches at force-decode position
- Format: `{out_dir}/{model}/{dataset}/residual/{quadrant}_{cutoff}.npy`
- Shape: `(N_samples, n_layers + 1, hidden_dim)`, dtype `float16`
- Indexed list of `sample_id` saved separately.
- Captured under baseline (no intervention) only — for clean DLA later.
- Stored in `float16` to keep size manageable. Cost estimate: 4 GB per 1.5B model per dataset, larger for big models.

### 5.2 Layer-wise component contributions at force-decode position
- For each layer, save the MLP output and attention output at the force-decode position separately. Same shape/dtype as residual cache.
- Format: `{out_dir}/{model}/{dataset}/components/{quadrant}_{cutoff}_{mlp,attn}.npy`
- Enables DLA decomposition without re-running the model.

### 5.3 Top-K next-token logits
- At the force-decode position (after the `\boxed{` suffix), save the top-50 logits and their token ids.
- Format: `{out_dir}/{model}/{dataset}/logits/{quadrant}_{cutoff}.npz`
- Critical for the planned DLA analysis on the "Insufficient" token.

### 5.4 Probe iterates
- $v_Y^{(1)}, \ldots, v_Y^{(16)}$ saved as joblib.
- Format: `{out_dir}/{model}/{dataset}/probes_rank/{k}.joblib`

### 5.5 Direction metadata
- Format: `{out_dir}/{model}/{dataset}/meta.json`
- Fields: $\cos(v_Y, v_\text{DIM})$, $\cos(v_Y, v_\text{rand})$, pairwise cosines among rank-K probe iterates, residual stream norm statistics per layer, intervention magnitudes used.

### 5.6 Per-sample full intervention results
- Format: `{out_dir}/{model}/{dataset}/results_sample_major.json`
- One record per (sample, cutoff, intervention condition): `{boxed_content, answer_kind, is_correct, coherence, probe_sanity_F1_under_intervention}`.

**Total storage estimate**: ~50--150 GB for 6 models × 2 datasets. Plan for it.

---

## 6. What this experiment does NOT do (deferred to follow-ups)

- **DLA on the "Insufficient" token**. The artifacts saved in §5 enable this as a 1-2 day follow-up forward decomposition without re-running models.
- **Activation patching Q1 ↔ Q2**. The residual caches in §5.1 enable this.
- **SAE feature analysis**. Requires separate SAE training, deferred.

These are deliberately kept out of the current experiment to keep its purpose clean: **establish whether the probe direction is causally active or observer-only.** DLA addresses the next question ("where is the actual mechanism?"). Mixing them muddles both.

---

## 7. Required code changes

A single new script: `src/causal_probe_test.py`. Replaces both `exp13_steering_new.py` and `two_track_exp/x_not_equal_y.py` for paper results. Both old files stay in the repo as reference / for the supplementary appendix.

Structure:

```
src/causal_probe_test.py
  - filter_models_by_quality()
  - load_unified_probe_and_compute_directions()
  - train_rank_k_probe_basis()         # iteratively ablate + retrain
  - register_position_restricted_hooks()
  - run_intervention_at_cutoff()       # reuses exp14 cutoff machinery
  - probe_sanity_check_under_intervention()
  - save_artifacts()                   # §5
  - aggregate_to_summary_csv()
```

Aim for 700--900 lines, modular, with the §5 saving as a separate function so we can call it standalone if needed.

---

## 8. Expected outputs (the paper figures and tables)

### Figure 1: Steering doesn't flip behavior, even at the probe's best layer
- One panel per (model, dataset), bar chart: clean Q1-flip rate ($\alpha=1, 2, 4, 8$) for $v_Y$, $v_\text{DIM}$, $v_\text{rand}$ at the probe's best layer.
- Prediction under $H_\text{observer}$: $v_Y$ bars are flat near 0; $v_\text{rand}$ is flat near 0; $v_\text{DIM}$ may show selectivity in some configurations.

### Figure 2: Rank-K ablation — probe collapses before behavior does
- Line plot. X-axis: K. Two lines per (model, dataset): probe F1 under $V_K$ ablation, Q1 abstention rate under $V_K$ ablation.
- Headline: the probe F1 line drops to chance well before the behavioral line moves.

### Table 1: Pre-registered decision results
- Per (model, dataset): conclusion (Observer / Causal / Inconclusive) per §4.1 and §4.2.
- Honest count of how many in-scope pairs land in each conclusion.

### Section text claim (the headline)
> "Across 6 (model, dataset) pairs that pass strict probe-quality filters, the probe direction is causally inert at K=1 (mean $|\Delta Q1|$ matches the random-direction control within 0.02). Increasing the ablation rank from 1 to 16 collapses probe F1 to chance ($K^*_\text{probe}=4$ on average) without producing behavioral change beyond the random-direction control ($K^*_\text{behavior} > 16$ on $X/6$ pairs). Sufficient and necessary tests both confirm: the linearly separable probe-readable signal of insufficiency is not the direction the model uses to verbally abstain."

---

## 9. Honest pre-mortem

What could go wrong:

1. **Random direction ablation breaks the model** at K $\ge 4$. Then we cannot interpret rank-K. Mitigation: pilot K=1 first; only increase K if random K-ablation remains null.
2. **Layer-restricted steering does flip behavior on one or two models** at one specific layer. Then $H_\text{observer}$ is partially refuted. Honest report: "Y is mostly observer; in $X/N$ configurations a layer-localized intervention has a small causal effect ($\Delta Q1 < 0.20$)."
3. **TreeCut behavioral effects are tiny regardless of intervention** (consistent with current data). Then we cannot conclude on TreeCut. Mitigation: pre-register that TreeCut is a stretch goal, not a required dataset.
4. **The "in-scope" filter passes only 2--3 models**. Then statistical claims are weak. Mitigation: explicitly broaden filter and report stratified results (strong-probe vs weak-probe pairs).

These are written out so we cannot rationalize after the fact.
