# CARDS — Findings Summary

Curated findings only. Each item lists what we can defensibly claim, the evidence, the code that produced it, and one honest caveat. Old experiments superseded by redesigned ones have been removed. Inconclusive experiments are labeled as such — not reported as null evidence for any hypothesis.

Paths relative to `/home/hwang302/.local/nlp/CARDS/`. Server outputs under `/export/fs06/{hwang302,xwang397}/CARDS/`.

Last updated: 2026-05-14.

---

## Current state, stated honestly

We have established a robust **representation–verbalization gap** with phenomenon-level evidence. Across 21 models and 2 datasets, models internally distinguish underspecified from solvable inputs (high linear-probe F1 at $t=0$ and throughout CoT), but behaviorally produce numeric answers at much higher rates than their internal recognition would suggest. Chain-of-thought improves solving accuracy roughly $4\times$ more than abstention accuracy.

We attempted **two causal interventions** on the probe-readable direction. **Both are currently inconclusive**:

- **F7 (rank-K ablation)** failed its manipulation check. Probe F1 did not collapse to chance even at $K=16$, meaning we did not successfully remove the probe-readable signal. We therefore cannot draw a causal conclusion either way from F7.
- **F8 (additive steering)** is underpowered. The $\alpha$ range tested approaches the model's coherence limit before producing any substantial behavioral effect. "Zero clean intervention at $\alpha \le 8$" does not establish causal inertness; it establishes that we did not find a working intervention in the parameter range tested.

We do **not** yet have direct evidence either for or against the geometric-mismatch account ($H_\text{geom}$) or the magnitude/override account ($H_\text{magnitude}$). The experiment designed to produce direct evidence — `probe_unembed_alignment.py`, measurement A: $W_U \cdot v_\text{probe}$ projected onto the abstention-vs-numeric direction — has not yet been run.

This is the honest state of the work: a robust phenomenon, an open mechanism question, and an in-progress causal investigation that has not yet produced a clean answer.

---

## F1 — The static representation–verbalization gap motivates the work

**Claim.** At $t=0$ (end of prompt, before generation), on a binary yes/no probe of solvability vs the model's behavioral yes/no answer, linear probe accuracy exceeds behavioral accuracy by 5–40 percentage points across the (base, instruct) pairs we tested.

**Caveat.** Binary verbalization setup, different prompt from the 4-quadrant solving setup used in F2–F5. Motivation only, not the foundation of any mechanism claim.

**Code.** [src/exp1_run_all_probing.py](src/exp1_run_all_probing.py), [src/exp1_run_all_verbalization.py](src/exp1_run_all_verbalization.py), [src/exp1_analyze_gap.py](src/exp1_analyze_gap.py).

**Result file.** [experiment_result/experiments/analysis/gap_analysis.csv](experiment_result/experiments/analysis/gap_analysis.csv).

---

## F2 — Probe discriminability follows a universal drop-then-recover trajectory across CoT

**Claim.**
- 21/21 models on both datasets show a drop in Unified test F1 between cutoff 0% and 20% (UMWP mean Δ = $-0.127$, TreeCut mean Δ = $-0.167$).
- The drop is concentrated in that single first step; the middle three cutoffs (20–80%) are roughly flat.
- On UMWP, 15/21 models recover to within 2 pp of the $t=0$ baseline by 100%. On TreeCut only 6/21 recover.
- Separate (oracle, re-trained per cutoff) probe F1 matches the Unified probe within mean $|\Delta F1| < 0.01$, so the shape is a property of the representation, not a probe-pooling artifact.

**Caveat.** We do not yet explain *why* discriminability collapses precisely at the 0→20% transition. Per-token analyses are too noisy to localize this further.

**Code.** [src/exp10_unified_proportional_probe.py](src/exp10_unified_proportional_probe.py).

**Result files.** [exp10_ultimate_proportional_umwp.csv](experiment_result/exp_temporal_new/results/exp10_ultimate_proportional_umwp.csv), [exp10_ultimate_proportional_treecut.csv](experiment_result/exp_temporal_new/results/exp10_ultimate_proportional_treecut.csv).

---

## F3 — Probe and verbalization trajectories move in opposite directions at the onset of reasoning

**Claim.** At the 0%→20% cutoff transition, population means across 21 models:

| | UMWP | TreeCut |
|---|---|---|
| Δ Probe F1 | $-0.127$ | $-0.167$ |
| Δ Q2 abstention accuracy | $+0.049$ | $+0.040$ |

Same models, same input data, same cutoff position. The two trajectories disagree in direction at this transition.

**Caveat.** Population-level statistical claim. Not causal at the sample level.

**Code.** [src/exp14_early_cutoff_generate.py](src/exp14_early_cutoff_generate.py), [src/exp14_early_cutoff_evaluate.py](src/exp14_early_cutoff_evaluate.py); probe trajectories from `exp10`.

**Result files.** Probe F1 from `exp10_ultimate_proportional_{umwp,treecut}.csv`. Verbalization per-cutoff in [xwang397_results_new/experiments/early_cutoff_evaluation/](xwang397_results_new/experiments/early_cutoff_evaluation/).

---

## F4 — Chain-of-thought improves solving roughly $4\times$ more than abstention

**Claim.** Over the full CoT (cutoff 0%→100%), population means across 21 models:

| Dataset | Δ Q3 (solving) | Δ Q2 (abstention) | Ratio |
|---|---|---|---|
| UMWP | $+0.495$ | $+0.112$ | $4.4\times$ |
| TreeCut | $+0.378$ | $+0.096$ | $3.9\times$ |

**Caveat.** Descriptive. The Q3 gain partly reflects entry from a no-CoT baseline.

**Code & result files.** Same as F3.

---

## F5 — Quadrant-conditional probe behavior: model "internally recognizes" while behaviorally hallucinating

**Claim.** Soft-output probe activations averaged within quadrant:

- **Q1 (hallucinate; insufficient input)**: P(insuff) stays 0.60–0.80 throughout CoT on UMWP. Residual representation reads input as insufficient *while the model behaviorally produces a numeric answer*.
- **Q2 (correctly abstain; insufficient input)**: 0.85–1.00.
- **Q3 (correctly solve; sufficient input)**: 0.10–0.30. Control quadrant is clean.

Q1 vs Q2 mean gap on UMWP: ~0.15–0.25. On TreeCut: ~0.03–0.10.

**Caveat.** Per-sample probe trajectories are highly noisy; population means tell a clean story, individual samples do not.

**Code.** [src/exp11_quadrant_proportional.py](src/exp11_quadrant_proportional.py).

**Result file.** [exp11_average_trajectories.csv](experiment_result/exp_temporal_new/results/exp11_average_trajectories.csv).

---

## F6 — Probe direction is not reducible to four surface confounders on UMWP

**Claim (UMWP).** Across 6 representative models spanning 1.5B–70B and four families:

- **D1 Minimal Contrastive Pairs**: a single-word edit that preserves length, scenario, and lexical content flips the probe by mean $0.71$ probability mass (range $[0.52, 0.86]$).
- **D3 Geometric Separability**: $|\cos(v_\text{probe}, v_\text{confounder})| \le 0.08$ for length, numcount, complexity, and lexical confounder directions.

**Caveat.** D1 is weaker on TreeCut: only 2/6 representative models retain a meaningful gap (the rest have saturated probes). D2 (lexical injection) is reported only on non-saturated baselines, as supplementary support. The four confounders we tested are not exhaustive; other surface dimensions (syntactic structure, operator distribution, discourse markers) are not controlled here.

**Metrics dropped from the original confounder pipeline**:
1. D1 `Pairwise_Acc` — passes at 0.501 vs 0.499; uninformative.
2. D1 `Strict_Acc` — too harsh, not informative beyond `Mean_Gap`.
3. D3 `F1_Drop` — baseline-F1 calculation does not match the canonical Unified probe F1; uninterpretable.

**Code.** [src/confounder_exps/build_pairs.py](src/confounder_exps/build_pairs.py), [src/confounder_exps/confounder_validation.py](src/confounder_exps/confounder_validation.py).

**Result files.** Curated: [experiment_result/exp_confounder/results_curated/](experiment_result/exp_confounder/results_curated/). Drafted section: [confounder_section_draft.tex](confounder_section_draft.tex).

---

## F7 — Rank-K probe ablation: INCONCLUSIVE (failed manipulation check)

**Intended test.** Project the K-dim subspace spanned by iteratively-retrained probe directions out of the residual stream at every layer and every position ($K \in \{1, 2, 4, 8, 16\}$). If the probe-readable signal is successfully removed and Q1 abstention behavior is unchanged, that is causal evidence the probe direction is not the driver.

**What actually happened.** **The manipulation check failed.** Probe F1 on the sanity-check set did not collapse to chance even at $K=16$; it changed by less than 0.05 from baseline in every direction (probe, DIM, random) on every (model, dataset) pair tested. Two possible reasons, neither verified:
1. The probe-readable signal is genuinely higher-dimensional than 16.
2. Our iteratively-trained orthonormal basis does not span the subspace the probe actually uses at inference.

**What we cannot conclude.** Because the manipulation did not remove the signal we intended to remove, this experiment **does not provide evidence about the probe direction's causal role**. It is an inconclusive intervention, not a null finding.

**What we observed anyway, for the record (but cannot interpret causally):**
- Across 130 (model, dataset, cutoff) cells, `ablate_probe` produces mean $|\Delta Q_1| = 0.011$; `ablate_random` produces mean $|\Delta Q_1| = 0.007$. These are indistinguishable, but since neither intervention removed the probe signal, this fact alone does not bear on the probe direction's causal role.
- `ablate_dim` produces ~$10\times$ larger behavioral changes (0.104) — but on some models (e.g., Gemma-3-12B at $K \ge 2$) collapses the probe to "predict everything insufficient" (P(insuff | both classes) → 1.0). Part real effect, part model breakdown.

**What needs to happen for F7 to produce evidence.** A revised manipulation that demonstrably reduces probe F1 to chance. Concrete proposed first step: directly project out the **1D span of $v_\text{probe}$ itself** (not the iteratively-retrained K-dim basis) and verify probe F1 collapses on a properly-matched evaluation set (the same test set on which the canonical Unified F1 = 0.78–0.92 was reported). If that works, the existing causal_probe_test pipeline can re-run with that as the manipulation.

**Code.** [src/causal_probe_test.py](src/causal_probe_test.py).

**Result files.** [experiment_result/causal_results/{slug}/{dataset}/](experiment_result/causal_results/) — `summary.csv`, `probe_sanity.csv`, `sample_major.json`, `meta.json`, `probes_rank/probe_v{1..16}.npy`, `artifacts/`.

---

## F8 — Additive steering of $v_\text{probe}$: INCONCLUSIVE (underpowered)

**Intended test.** Add $\alpha \cdot v_\text{probe}$ at the probe's best layer. If $v_\text{probe}$ is the direction the unembedding uses to abstain, this should increase Q1 abstention while leaving Q3 alone.

**What actually happened.** Across 5 (model, dataset) pairs tested, $v_\text{probe}$ steering produced no clean intervention at any $\alpha \in \{0.5, 1.0, 2.0, 4.0, 8.0\}$. **However, by $\alpha = 8$ the model's coherence is at or near its limit** (random-direction addition at $\alpha = 8$ also produces non-trivial effects on some pairs through model degradation). This means the parameter range we tested is bounded above by model breakdown before it produces a substantial effect.

**What we cannot conclude.** "No clean intervention at $\alpha \le 8$" is a statement about a specific bounded parameter range, not about the probe direction's causal role in general. The intervention could fail to produce effect because the probe direction is not causally relevant ($H_\text{geom}$), or because $\alpha$ was insufficient, or because single-layer intervention is the wrong site, or because multi-layer simultaneous intervention is required.

**On the n=1 reversal previously highlighted.** Qwen2.5-Math-7B-Instruct/TreeCut showed $\Delta Q_1 = -0.265$ under $\alpha = 8$ probe-direction steering. This is **one data point among 8 (model, dataset) pairs**, on a model whose probe-sanity P(insuff | both classes) suggests saturation issues at $t=0$ on TreeCut. We retract the earlier suggestion that this represents a "wrong-direction" causal finding. It is plausibly a saturation artifact.

**What needs to happen for F8 to produce evidence.** Either (a) find an $\alpha$ range and intervention site that produces a substantial effect on $v_\text{probe}$ steering without producing comparable effects under random-direction steering — confirming the test is well-powered; or (b) replace addition steering with an intervention that the alignment-style measurement A indicates is more likely to produce effect.

**Code.** [src/causal_probe_test.py](src/causal_probe_test.py).

**Result files.** Same directory as F7. Steering conditions appear in `summary.csv` with names `steer_{probe,dim,rand}_a{0.5,1.0,2.0,4.0,8.0}`.

**For full disclosure on this experiment's history**: an earlier version of the script applied steering only when layer output sequence length was 1 (position-restricted heuristic). In recent HuggingFace + SDPA versions this condition was never satisfied during generation, so the hook silently never fired, producing $\Delta Q_1 = 0$ across all conditions. Fixed by applying addition at every position. All numbers reported here are post-fix.

---

## Mechanism: open question

Two hypotheses remain competing. We have not yet produced direct evidence for either.

| Hypothesis | What it predicts | Direct test |
|---|---|---|
| $H_\text{geom}$ — Probe direction is geometrically orthogonal (in $W_U$-space) to the abstention-vs-numeric distinction. The unembedding never reads it for that decision. | $W_U \cdot v_\text{probe}$ has near-zero projection on the abstention-vs-numeric direction in unembedding space. | R2 measurement A. **Not yet run.** |
| $H_\text{magnitude}$ — Probe direction is the right direction, but $v_\text{probe}$ in real residuals is too small or gets overwhelmed by competing CoT-generated content. | A sufficiently strong, well-targeted addition of $v_\text{probe}$ should flip behavior. Removal of $v_\text{probe}$ at scale should reduce abstention. | A well-powered version of F7/F8 with manipulation checks that pass. **Not yet achieved.** |

Both F7 and F8 in their current form are inconclusive. The earlier framing of "data is consistent with $H_\text{geom}$" was overclaim; the data is consistent with the experiments not having worked. Until R2 measurement A runs (or until F7/F8 are redesigned with successful manipulation checks), we cannot rank these hypotheses on evidence.

---

## Next experiments, in priority order

### 1. R2 measurement A — *direct* geometric test (highest priority)

**Question.** Does $v_\text{probe}$ project onto the abstention-vs-numeric direction in unembedding space?

**Procedure.** For each (model, dataset), compute $z = W_U \cdot v_\text{probe}$ and the scalar $\text{mean}_{t \in T_\text{abs}}(z_t) - \text{mean}_{t \in T_\text{num}}(z_t)$. Compare against the same scalar computed for $v_\text{DIM}$ and a random direction.

**Why it is decisive.** This is the only experiment that produces direct evidence about $H_\text{geom}$ regardless of intervention difficulty. Two possible outcomes:
- **Margin near zero relative to random-direction baseline** → first direct evidence for $H_\text{geom}$. Reframe F7 and F8 as "consistent with H_geom now that there is a positive geometric finding to anchor them."
- **Margin substantially non-zero** → $H_\text{geom}$ refuted by our own data. Mechanism section must be rewritten; F7 and F8 stay inconclusive.

**Status.** Script written: [src/probe_unembed_alignment.py](src/probe_unembed_alignment.py). Has not been run on any model. **Should run today on one (model, dataset) pair, not after R1 finishes.**

### 2. Direct 1D probe ablation as a corrected manipulation check

**Question.** If we directly project out the 1D span of $v_\text{probe}$ from every layer, does probe F1 collapse on the same test set where the canonical Unified F1 was 0.78–0.92?

**Why it matters.** Currently we do not know whether the rank-K basis fails to crash probe F1 because (a) probe signal is distributed across many directions, or (b) our basis is the wrong basis. Direct ablation of $v_\text{probe}$ itself (1D span) tests this distinction.

**Outcomes.**
- Probe F1 collapses → we have a valid manipulation. F7 can be re-run on this manipulation and produce interpretable evidence.
- Probe F1 does not collapse → the probe-readable signal is genuinely distributed, the "$v_\text{probe}$ direction" framing is itself a simplification, and we need to rethink the mechanism in terms of subspaces rather than directions.

**Status.** Not yet written. Small script — fast.

### 3. Positive identification: which direction *does* drive abstention?

Even if R2 measurement A confirms $H_\text{geom}$, the paper still needs an answer to "if not $v_\text{probe}$, then what?" Candidates to test under the same R2 framework:
- $v_\text{DIM}$ at the same layer (`W_U \cdot v_\text{DIM}`).
- The abstention column of $W_U$ itself, treated as a residual-space direction.
- DAS-style learned directions (distributed alignment search).

This is a follow-up to (1); without (1) we don't know whether it's needed.

---

## Code → experiment → conclusion table

| Code | Experiment | Conclusion |
|---|---|---|
| `exp1_run_all_*` + `exp1_analyze_gap.py` | Binary solvability probe vs verbal yes/no at $t=0$ | **F1**: 5–40 pp gap, motivates the work |
| `exp10_unified_proportional_probe.py` | Train unified probe across CoT cutoffs, evaluate per-cutoff F1 | **F2**: universal drop-then-recover in probe F1 |
| `exp11_quadrant_proportional.py` | Apply unified probe to per-sample states, aggregate by quadrant | **F5**: Q1 P(insuff) stays elevated while model behaviorally hallucinates |
| `exp14_early_cutoff_*.py` | Cutoff CoT at sentence boundary, force-decode, judge | **F3, F4**: opposite directions at 0→20%; $4\times$ CoT asymmetry |
| `confounder_exps/{build_pairs,confounder_validation}.py` | Minimal contrastive pairs + geometric separability | **F6**: probe not reducible to length/numcount/complexity/lexical |
| `causal_probe_test.py` (ablation arm) | Rank-K probe-subspace ablation with controls | **F7: INCONCLUSIVE** — manipulation check failed; probe F1 not crashed at $K=16$ |
| `causal_probe_test.py` (steering arm) | Additive steering with magnitude-normalized $\alpha$ | **F8: INCONCLUSIVE** — underpowered, $\alpha$ at coherence limit before producing substantial effect |
| `probe_unembed_alignment.py` | Direct geometric test of $W_U \cdot v_\text{probe}$ alignment with abstention direction | **Not yet run** |

---

## What is honestly removed from earlier drafts

- **"Consistent with $H_\text{geom}$"** language in the mechanism section. The "null results" in F7 and F8 came from a failed manipulation check (F7) and an underpowered test (F8) respectively. Neither bears on $H_\text{geom}$ until corrected.
- **"Leading hypothesis is $H_\text{geom}$"**. The hypotheses remain competing. R2 measurement A is what would rank them on evidence.
- **The n=1 "wrong direction" steering result on Qwen-Math-7B/TreeCut as evidence**. Plausibly a saturation artifact in a single (model, dataset) pair. Mentioned in F8 for full disclosure but no longer treated as a finding.
- **The U-shape claim for F2**. The shape is asymmetric drop-then-recover, not a symmetric U.
- **Shape-disagreement statistic (1/42)**. Threshold-dependent and weaker than the direct numerical contrast at 0%→20%.
- **Reasoning-vs-Instruct contrast.** Out of scope — counts datapoints rather than answering the mechanism question.
- **Layer-by-layer DLA design.** Replaced by the three-measurement alignment test in `probe_unembed_alignment.py`.
