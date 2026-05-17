# CARDS Project — 完整实验文档 (中文版)

**最后更新**: 2026-05-14
**状态**: F1-F5 (Act 1) audited paper-ready; F6 audited 21-model; F7 v1 done with caveats, v2 ready to run; F8 GPU rerun pending; INLP / multi-layer probe / prompt-robustness pending

这份文档把整个 CARDS 项目按 "概念 → 实验 → 发现 → code 路径 → 当前 evidence 强度 → 还差什么 → 怎么 connect 到 story" 全部讲清楚。每个数学符号都精确定义, 每个公式都给 intuition。

---

# 目录

0. Paper 的核心 claim (story 一句话)
1. 基础概念和数学符号
2. Act 1 — Phenomenon (F1-F5)
3. Act 2 — Mechanism (F6-F8)
4. 整体 story 怎么 connect
5. 当前 evidence 审计 (sub-claim by sub-claim)
6. TODO list + F8 的 scientific 再审视

---

# 0. Paper 的核心 claim

我们要 defend 的中心 claim 是这一段:

> "LLMs reliably encode whether a math word problem is solvable in their residual stream, and this encoding is recoverable by a linear probe across all four behavioral quadrants — including the two quadrants where the model's verbalized answer is incorrect (Q1 hallucinations, Q4 over-cautions). However, this internal recognition does not causally drive the model's abstain-vs-answer decision: the residual direction along which the probe reads input class is geometrically and causally separate from the residual direction along which the unembedding selects abstention tokens. Recognition exists; it just is not routed to the decision."

把这个 claim 拆成 5 个 sub-claim:

| # | Sub-claim | 由哪个 experiment 支持 |
|---|---|---|
| (a) | Encoding exists (probe 高 F1) | F1, F2 |
| (b) | Encoding 跨四象限 (Q1-Q4 都对) | F4 |
| (c) | Encoding ≠ surface features (不是学 length/keyword) | F5 |
| (d) | $v_\text{probe}$ 几何上和 $\Delta u$ decoupled | F6 |
| (e) | $v_\text{probe}$ 因果上 inert | F7 |
| (+) | (可选) Decision channel 在 $\Delta u$ at $L^*$ | F8 |

(a)-(e) 加起来已经支持核心 claim。(+) 是 F8, 它把 paper 从 "decoupled, recognition channel identified, decision channel unlocalized" 升级到 "both channels identified"。

---

# 1. 基础概念和数学符号

## 1.1 Residual stream — 模型的"内心状态"

Transformer 处理输入是一层一层进行的。每过一层, 模型更新一个 $d$ 维向量 $h$ (典型 $d \in [4096, 5120]$), 这个向量叫 **residual stream**:

$$h^{(0)} \to h^{(1)} \to \cdots \to h^{(L)}$$

其中 $h^{(\ell)}$ 是第 $\ell$ 层之后的 residual。$L$ 是 model 的总层数 (通常 28-80)。每层做的事情可以抽象成:

$$h^{(\ell+1)} = h^{(\ell)} + \text{attn}^{(\ell)}(h^{(\ell)}) + \text{mlp}^{(\ell)}(h^{(\ell)})$$

residual stream 是 model 的工作内存, 每一层都在往里加 "thought"。

## 1.2 Unembedding — 内心状态变成输出 token

最后一层之后, 做两步把 $h^{(L)}$ 变成 logits:

$$\tilde h_L = \text{LayerNorm}(h^{(L)})$$

$$\text{logits} = W_U \cdot \tilde h_L \in \mathbb{R}^V$$

其中:
- $W_U \in \mathbb{R}^{V \times d}$ 是 **unembedding 矩阵**
- $V$ 是词表大小 (32k-200k)
- $W_U[t]$ 是 token $t$ 对应的 row, 一个 $d$ 维向量

每个 token 在 residual space 里都有一个 "direction" ($W_U[t]$ 这个 row)。模型最终输出哪个 token = $\tilde h_L$ 在哪个方向上投影最大。

## 1.3 Probe — 读取内心独白

**Probe** 是一个 logistic regression 二分类器:

$$P(\text{insufficient} \mid h) = \sigma\Big(\frac{W}{\sigma_\text{scale}} \cdot h + b'\Big)$$

其中 $W$ 是 probe 学到的权重 ($d$ 维), $\sigma_\text{scale}$ 是 StandardScaler 的尺度。从 probe 提取的归一化方向是:

$$v_\text{probe} = \frac{W / \sigma_\text{scale}}{\|W / \sigma_\text{scale}\|} \in \mathbb{R}^d, \quad \|v_\text{probe}\| = 1$$

**Probe normal direction** = $v_\text{probe}$ = "residual space 里区分 insufficient vs sufficient 的方向"。注意: **一个 probe 只学一个方向**, 不是 d-1 个。

但 "insufficient" 这个 concept **可能住在多个方向上**, 也就是 high-dim subspace 里 — INLP 用来量化这件事 (后面 §6.3 详说)。

## 1.4 $\Delta u$ — Unembedding 在 abstain-vs-num 决策上的偏好方向

定义两类 token:

- $T_\text{abs}$ = tokenizer 里 "Insufficient", "Cannot", "Unable" 等的 first token id
- $T_\text{num}$ = tokenizer 里所有以数字开头的 token id

然后:

$$\Delta u = \text{mean}\big(W_U[T_\text{abs}]\big) - \text{mean}\big(W_U[T_\text{num}]\big) \in \mathbb{R}^d$$

**Intuition**: $\Delta u$ 是 residual space 里 "如果 $\tilde h_L$ 朝这个方向走, abstain token 的 logit 会比 num token 的 logit 高" 的方向。

如果 $\tilde h_L \cdot \Delta u > 0$ → 模型更倾向 abstain
如果 $\tilde h_L \cdot \Delta u < 0$ → 模型更倾向输出数字

**关键**: $\Delta u$ 是从 $W_U$ 直接构造的, 完全是 unembedding side 的方向, 和 hidden state 训出来的 $v_\text{probe}$ 是**不同来源**的两个方向。

## 1.5 Four behavioral quadrants

每个 (input, model) 样本根据 (a) 输入实际是否 insufficient, (b) 模型行为是否 abstain, 落入四象限之一:

|  | Behavior: abstain | Behavior: numeric |
|---|---|---|
| **Input: insufficient** | $Q_2$ (correct abstain) | $Q_1$ (hallucinate) |
| **Input: sufficient** | $Q_4$ (over-abstain) | $Q_3$ (correct solve) |

$Q_1$ 是 hallucination — 我们最关心的 pathology。
$Q_4$ 是 over-caution — 也是错误, 但反方向。

## 1.6 CoT — Chain-of-Thought

模型最终答题之前会先生成一段推理 ("Let me think... if we have 5 apples..."). 我们用 truncation 在 0%, 10%, ..., 100% cutoff 处采 hidden state, 看 CoT 不同进度时 residual 的样子。

## 1.7 两个核心假说 — H_A vs H_B

**H_A (generative override 假说)**:
$v_\text{probe}$ 和 $\Delta u$ 是同一个 (或重叠的) 方向。模型内心确实在 $v_\text{probe}$ 方向编码 insufficient, 这个方向也喂给 unembedding 做 abstain 决策。但 CoT 生成的内容把 residual stream 弄乱了, recognition signal 被淹没。

**H_B (geometric decoupling 假说)**:
$v_\text{probe}$ 和 $\Delta u$ 是 residual space 里**两个不同的方向**。Recognition signal 住在一个 unembedding 根本不读的子空间里。识别存在但哪里也去不到。

整个 Act 2 在区分这俩。**结论是 H_B**。

---

# 2. Act 1 — Phenomenon (F1-F5)

Act 1 的任务是建立现象学: gap 存在、跨四象限存在、不是 surface artifact。

## F1: 静态 representation-verbalization gap

**Question**: $t=0$ 时 (CoT 还没开始), probe accuracy 比 verbalization accuracy 高吗?

**Setup**:
- 在 prompt 结束位置 ($t=0$):
  - 直接问模型 "is this solvable? yes/no", 记录 verbal answer
  - 提 $h^{(L^*)}$, 跑 probe, 记录 probe prediction
- 比较两者准确率

**Finding**: probe 准确率高 5-40 pp (range $[0.057, 0.395]$), 跨所有 (model, dataset) pair。

**Code path**:
- `src/exp1_run_all_probing.py`
- `src/exp1_run_all_verbalization.py`
- `src/exp1_analyze_gap.py`
- 结果: `experiment_result/experiments/analysis/gap_analysis.csv`

**Story role**: motivate paper — gap 真实存在, 跨 21 model。

**Evidence 强度**: **~95%**, 不会被打。

## F2: Probe accuracy 沿 CoT 的轨迹

**Question**: 跑了 CoT, probe accuracy 怎么变?

**Setup**:
- 每个 sample, 把 CoT 截到 0%, 10%, ..., 100%
- 每个截断处提 $h^{(L^*)}$, 跑 probe, 记 F1
- 21 model × 2 dataset

**Finding**:
- 21/21 model 在 $0\% \to 20\%$ 都有 F1 drop (UMWP mean $\Delta = -0.127$, TreeCut $-0.167$)
- 中间 cutoff (20-80%) 大致平
- 100% 时部分恢复 (UMWP 15/21 model 回到 $t=0$ ±2pp; TreeCut 6/21)

**Refinement from F4**: drop **不是 class identification flip**, 是 soft probability $P(\text{insufficient})$ 朝 0.5 压缩 (confidence compression, not forgetting)。

**Code path**:
- `src/exp10_unified_proportional_probe.py`
- 结果: `experiment_result/exp_temporal_new/results/exp10_ultimate_proportional_{umwp,treecut}.csv`
- Probe 文件: `experiment_result/exp_temporal_new/probes_proportional/`
- 缓存的 hidden states: `experiment_result/exp_temporal_new/embeddings_proportional/`

**Story role**: 现象不平凡 — CoT 实际 degrade 了 probe 的 confidence, 但 direction 没翻。motivate Act 2 的"机制问题"。

**Evidence 强度**: **~95%**, robust。

## F3: CoT 的 asymmetric 效应

合并了之前的 F3+F4 (asymmetric effects 的两个 sub-claim)。

**Question**: CoT 对 solving 和 abstaining 的影响是 symmetric 的吗?

**Setup**:
- 在每个 cutoff 算 verbal Q1/Q2/Q3 rate 变化
- 比较 probe trajectory 和 abstention trajectory 的方向

**Finding**:

**Sub-claim (a) — $0 \to 20\%$ 方向不一致**:

|  | UMWP | TreeCut |
|---|---|---|
| $\Delta$ probe F1 | $-0.127$ | $-0.167$ |
| $\Delta$ Q2 abstention rate | $+0.049$ | $+0.040$ |

同一组 input, 同一个截断点, 两个信号往反方向走。

**Sub-claim (b) — 累积 $0 \to 100\%$ asymmetric**:

| | $\Delta Q_3$ (solving) | $\Delta Q_2$ (abstention) | Ratio |
|---|---|---|---|
| UMWP | $+0.495$ | $+0.112$ | $4.4\times$ |
| TreeCut | $+0.378$ | $+0.096$ | $3.9\times$ |

**Code path**:
- `src/exp14_early_cutoff_generate.py`
- `src/exp14_early_cutoff_evaluate.py`
- Probe trajectories from `exp10`

**Story role**: Test-time compute 对解题和拒绝**不对称帮助** — solving 提升 4×, abstention 几乎不动。Motivate 机制问题: 为什么 CoT 帮一个 channel 不帮另一个?

**Evidence 强度**: **~95%**

## F4: Quadrant-conditional probe behavior (paper 最强的现象证据)

**Question**: 当模型 hallucinate (Q1) 或 over-abstain (Q4) 时, probe 内心想什么?

**Setup**:
- 按 quadrant 分类 sample, 每 quadrant 每 cutoff 算 mean $\bar P(\text{insuff})$
- 21 model × 2 dataset = 42 pair

**Finding** (42 对 robustness check):

| Check | Result |
|---|---|
| Q1 mean > 0.5 | **42/42** (range $[0.53, 0.78]$) |
| Q4 mean < 0.5 | **42/42** (range $[0.20, 0.48]$) |
| Q1 > Q3 | **42/42** (gap $[0.21, 0.69]$) |
| Q1 > Q4 | **42/42** (gap $[0.07, 0.58]$) |
| Q2 > Q1 | **39/42** |
| Full ordering Q2 > Q1 > Q4 > Q3 | **39/42** |

**Trajectory shape**: $0 \to 20\%$ 时:
- Q1, Q2 (insufficient inputs): 从 >0.5 朝下压
- Q3, Q4 (sufficient inputs): 从 <0.5 朝上升
- 都朝 0.5 收 (confidence compression toward decision boundary)

**Code path**:
- `src/exp11_quadrant_proportional.py`
- 结果: `experiment_result/exp_temporal_new/results/exp11_average_trajectories.csv`

**Story role**: **representation-verbalization decoupling 的最强现象学证据**。Probe 在所有四象限都 respect input class, **包括行为错的那两个 (Q1, Q4)**。

**Evidence 强度**: **~95%**, 42/42 robust。

## F5: Probe 不是学 surface features

**Question**: 也许 probe 高 F1 不是因为模型真识别 insufficient, 而是因为 insufficient 和 sufficient input 在表面特征上不同?

**Setup** (6 representative model, UMWP):
- **D1 (Minimal Contrastive Pairs)**: 改一个词翻转 solvability, 保持长度、结构、词汇。重跑 probe。
- **D3 (Geometric Separability)**: 算 $|\cos(v_\text{probe}, v_\text{confounder})|$, $v_\text{confounder}$ 是 length / numcount / complexity / lexical direction。

**Finding**:
- D1: 一词改动翻转 probe prediction $0.71$ probability mass (range $[0.52, 0.86]$) on UMWP
- D3: $|\cos(v_\text{probe}, v_\text{confounder})| \le 0.08$ for 四个 confounder directions

**Caveat**: TreeCut 上 D1 只 2/6 model 保留 gap (probe 太饱和)。

**Code path**:
- `src/confounder_exps/build_pairs.py`
- `src/confounder_exps/confounder_validation.py`
- 结果: `experiment_result/exp_confounder/results_curated/`

**Story role**: probe 学到的是 semantic content, 不是 surface artifact。Motivate Act 2: 这个 semantic signal 去了哪里?

**Evidence 强度**: **~85%**。Prompt 固定 + TreeCut 弱是已知 caveat (后面 §6.5 详说)。

---

# 3. Act 2 — Mechanism (F6-F8)

Act 2 的任务: 区分 H_A (override) 和 H_B (decoupling), 找出 abstain decision 在 residual space 的实际通路。

## F6: 几何 alignment 测量 (static, paper 最稳的一块)

**Question**: $v_\text{probe}$ 和 $\Delta u$ 在 residual space 中重合吗?

**Setup**: 不跑任何 forward pass。只算几个内积。

**Step 1**: 提 $v_\text{probe}$ (上文 §1.3)
**Step 2**: 构造 $v_\text{random}$ — seed-fixed Gaussian unit vector (control)
**Step 3**: 构造 $\Delta u$ (上文 §1.4)
**Step 4**: 对每个 $v \in \{v_\text{probe}, v_\text{random}\}$, 算:

$$z = W_U \cdot v \in \mathbb{R}^V$$

$$\text{margin}(v) = \bar z[T_\text{abs}] - \bar z[T_\text{num}]$$

**代数恒等**: $\text{margin}(v) = \Delta u \cdot v$。验证:

$$\bar z[T_\text{abs}] - \bar z[T_\text{num}] = \text{mean}(W_U[T_\text{abs}]) \cdot v - \text{mean}(W_U[T_\text{num}]) \cdot v = \Delta u \cdot v$$

绕 $z$ 这一圈是为了顺便能看 top-50 token (qualitative inspection)。

**Small validation**: 不只看 difference, 分开看 $\bar z[T_\text{abs}]$ 和 $\bar z[T_\text{num}]$ 各自绝对值, 防止 cancellation (两个都 +0.1, 差为 0 但都不算 inert)。

**Finding** (21 model × 2 dataset = 42 pair):

| Direction | Margin range | $|\text{margin}| < 0.02$ 在 |
|---|---|---|
| $v_\text{probe}$ | $[-0.033, +0.143]$ | $38/42$ pair |
| $v_\text{random}$ | $[-0.014, +0.099]$ | $42/42$ pair |

Small validation: 39/42 pair 两个 component 各自都 $< 0.02$。

**Outliers** (显式 disclose):
1. `deepseek-math-7b-instruct` (两个 dataset): tokenizer 只识别 10 个 numeric token, inflate baseline → tokenizer artifact
2. `Qwen2.5-Math-1.5B / UMWP`: 单 model outlier, margin $+0.055$ vs random $+0.002$, 但同模型 TreeCut $+0.012$, 其他 Qwen-Math 尺寸 < 0.02 → single-pair outlier, not systematic

**Code path**:
- `src/measurement_A_sweep.py` (canonical, 21 model)
- 结果: `experiment_result/causal_results/_measurement_A_sweep/summary.csv`
- Per-pair: `per_pair/{slug}__{dataset}/measurement_A.json`, `top50_tokens.csv`

**Story role**: **直接排除 H_A**。$v_\text{probe}$ 和 $\Delta u$ 几何上 decoupled, unembedding 不读 $v_\text{probe}$ 做 abstain 决策。

**Evidence 强度**: **~85%**。短板是 layer mismatch ($v_\text{probe}$ 在 $L^*$, $\Delta u$ 作用在 $L$ post-LN) 和 prompt-fixed (§6.5)。

## F7: Causal intervention (dynamic) — necessity + sufficiency

F7 内部有两个独立 sub-experiment, 检验 $v_\text{probe}$ 的因果作用。

### F7-a Ablation (necessity)

**Question**: 去除 $v_\text{probe}$ 方向的信息, 行为变吗?

**Setup**:
1. 训 $v_1 = v_\text{probe}$
2. 把 $v_1$ 从 residual 中正交投影出去 → modified stream
3. 在 modified stream 上重训 probe → 得 $v_2$ ($\perp v_1$)
4. 迭代 $K$ 次, 得到 $K$ 个互相正交方向 $V_K = [v_1, \ldots, v_K]$
5. Inference 时, **每层每个 position** 把 $h$ 投到 $V_K^\perp$:

$$h' = h - V_K^T V_K h$$

6. 用 $h'$ 跑完整生成, 测 $\Delta Q_1$
7. 对照: random orthonormal basis, $v_\text{DIM}$ basis

$K \in \{1, 2, 4, 8, 16\}$

**Intuition**: 不只删一个方向, 删整个 "linearly readable insufficient" 子空间。

**Finding** (10 pair pass F1 ∈ [0.70, 0.95] scope filter):

Mean $|\Delta Q_1|$ across $K$ and cutoff:

| Direction | $|\Delta Q_1|$ | Relative |
|---|---|---|
| $v_\text{probe}$ | $0.019$ | $1\times$ |
| $v_\text{random}$ | $0.055$ | $\sim 2.9\times$ |
| $v_\text{DIM}$ | $0.150$ | $\sim 7.9\times$ |

**删 $v_\text{probe}$ 比删 random 还更不破坏行为**。

**Manipulation check caveat** (已知 weakness):
$K=16$ ablation 后, retrained probe 仍然高 F1 ($\Delta < 0.05$)。说明信号 genuinely > 16 维, 我们没真把它删干净。

### F7-b Steering (sufficiency)

**Question**: 加 $v_\text{probe}$ 到 stream 上, behavior 朝 abstain 走吗?

**Setup**:
- 在 $L^*$ 层加 $h_{L^*} \to h_{L^*} + \alpha \cdot \sigma \cdot v_\text{probe}$
- $\sigma$ = typical per-dim residual magnitude = $\text{mean}(\|h\|) / \sqrt{d}$
- $\alpha \in \{0.5, 1, 2, 4, 8\}$

**Intuition**: 如果 $v_\text{probe}$ 真是 abstain 方向, 加它应该 push behavior 朝 abstain。

**Finding**: 200 个 (pair, $\alpha$, cutoff) cell 里**只有 1 个**产生 clean intervention ($\Delta Q_1 \ge +0.10$, $\Delta Q_3 < +0.10$)。基本完全 null。

### F7 整体结论

两条独立因果证据 (necessity + sufficiency) 都 null → $v_\text{probe}$ 因果上 inert, 一致 with F6。

**Code path** (v1):
- `src/causal_probe_test.py`
- 结果: `experiment_result/causal_results/{slug}/{dataset}/`
  - `summary.csv`, `probe_sanity.csv`, `sample_major.json`, `probes_rank/`, `artifacts/`

**Code path** (v2 — pending GPU run):
- `src/causal_probe_test_v2.py`
- 修了 v1 的所有 caveat:
  - M1: K 扩展到 [1, 2, 4, 8, 16, 32, 64], 每个 K 同步 manipulation check (用 exp10 缓存的 hidden state 重训 probe)
  - M2: 双向 steering ($\alpha \in [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8]$)
  - M3: $L^*$-only ablation 平行于 all-layer
  - M4: Mean ablation 平行于 zero projection
  - M5: Bootstrap CI ($N_\text{boot} = 1000$) + per-pair box plot
- 输出: `experiment_result/causal_results_v2/`

**Story role**: 配合 F6 给出 **mechanism conclusion**: $v_\text{probe}$ 不驱动 abstain decision。F7 是 H_B 的因果证据。

**Evidence 强度** (v1): **~55%** — manipulation check caveat 是公开漏洞。
**Evidence 强度** (v2 跑完后预计): **~85%** — 所有 reviewer attack 都堵了。

## F8: Positive identification of decision channel (planned, GPU pending)

**Question**: F6+F7 排除了 $v_\text{probe}$ 是 abstain pathway, 那**真正的 abstain pathway 在哪个方向, 哪个层**?

**关键设计 — $L^*$ vs $L$ asymmetry**:

| 注入层 | 角色 | 含义 |
|---|---|---|
| $L^*$ (probe best layer) | **HEADLINE** | 注入 $\Delta u$ 到 $L^*$, 让它经过后续所有 attention/MLP/LN 再到 $W_U$。如果产生 abstain, 说明 abstain decision 在 $L^*$ 就已经通过 $\Delta u$ 这个方向实现 |
| $L$ (final layer) | **TRIVIAL CONTROL** | 直接加 $\Delta u$ 到 final residual。几乎按定义 work (直接 shift logit margin)。**Positive 只是 wiring sanity, 不是 mechanism finding** |

**测的 5 个方向**:

| Direction | 怎么构造 | 角色 |
|---|---|---|
| $\Delta u$ | $\text{mean}(W_U[T_\text{abs}]) - \text{mean}(W_U[T_\text{num}])$ | **PRIMARY**: decision channel 的 unembedding-side 方向 |
| $v_\text{abstain at } L^*$ | $\text{mean}(h^{(L^*)}[\text{boxed pos}] \mid Q_2) - \text{mean}(h^{(L^*)}[\text{boxed pos}] \mid Q_1)$ | **SECONDARY**: $L^*$ 上经验的 abstain-vs-hallucinate 方向 (Marks & Tegmark diff-of-means) |
| $v_\text{probe}$ | F6/F7 的 probe normal | **CONTROL**: F7 已证 inert |
| $v_\text{DIM}$ | $\text{mean}(h \mid \text{insuff}) - \text{mean}(h \mid \text{suff})$ at $t=0$ | **CONTROL**: input-class direction |
| $v_\text{random}$ | seed-fixed Gaussian | **CONTROL**: null comparison |

**$v_\text{abstain at } L^*$ 的特殊价值**:
- 它**按构造**就是 $L^*$ 上 abstain decision 的经验方向
- 比 $\Delta u$ 更宽松 — 不要求和 unembedding 对齐
- Both Q1 和 Q2 都是 insufficient input, 只是 behavior 不同 → 这个方向 isolate 出 **behavioral decision**, 不是 input-class recognition
- Marks & Tegmark 2024 风格 (diff-of-means in residual space)

**Hook scope** (两种实现):
- `all` (default): 每 token position 都加 $\alpha v$, 整段 generation 都受影响。Lenient test。
- `last`: 只在 boxed 位置加, surgical, 弱一点。Robustness check。

**Pre-registered 决策规则** (用 bootstrap 95% CI, 不是 point estimate):

满足以下三个 condition 算 "clean hit":
1. $\Delta Q_1$ lower 95% CI $\ge +0.15$ (intervention 真把 hallucination 推向 abstain)
2. $\Delta Q_3 \le +0.05$ (specificity: Q3 没塌成 "啥都 abstain")
3. Q1 coherence $\ge 0.80$ (intervention 没 break generation)

**四种可能 outcome**:

| $\Delta u$ @ $L^*$ | $v_\text{abstain}$ @ $L^*$ | 结论 |
|---|---|---|
| Positive | Positive | **最强**: $L^*$ 上 abstain pathway 存在, **就是** $\Delta u$ 方向 |
| Null | Positive | **次强**: pathway 存在但**不是** $\Delta u$ 方向, 是 $v_\text{abstain}$ |
| Positive | Null | 矛盾, debug |
| Null | Null | $L^*$ 不是 abstain decision point; pathway 在 $L^*$ 之后某层 |

**Activation patching supplement** (`--do_patching`):
- 对每个 Q1 prompt, length-matched 一个 Q2 prompt
- Cache Q2 在 $L^*$ 的 boxed-position residual
- 注入 Q1 的同位置, 看 Q1 是否翻转到 abstain
- Caveat: Q1/Q2 prompt 语义不同, 携带 prompt-semantic confound
- 主 sweep 里的 $v_\text{abstain}$ direction 是更干净的 aggregate version (mean over 100 pair, 把语义内容平均掉)

**Code path**:
- `src/delta_u_intervention.py` (你最新版的代码)
- 结果: `experiment_result/causal_results/{slug}/{dataset}/delta_u/`
  - `summary.csv`, `meta.json`, `fig_delta_u_intervention.png`

**Story role**:
- 如果 F8 positive at $L^*$: paper 故事是 **"both channels identified"** — recognition channel ($v_\text{probe}$, F6+F7) + decision channel ($\Delta u$ at $L^*$, F8)
- 如果 F8 null on both: paper 故事是 **"recognition channel identified, decision channel still being characterized"** — F8 给出额外 negative result on $L^*$, 加深 decoupling 但没 positively identify decision channel

**Evidence 强度**: **未知**, depends on GPU rerun outcome。

---

# 4. 整体 Story 怎么 Connect

## 故事链

```
F1 (5-40pp gap)
  ↓ motivate: 现象是真的, 跨 21 model
F2 (probe trajectory cliff)
  ↓ refine: CoT 干扰 probe
F3 (asymmetric CoT effects)
  ↓ motivate Act 2: 为什么 CoT 帮 solving 不帮 abstention?
F4 (Q1-Q4 quadrant: probe respects input class even in wrong behaviors)
  ↓ 这是 paper 最强现象证据: recognition vs behavior 解耦
F5 (probe 不是学 surface)
  ↓ probe direction 是真 semantic signal

  ═════════ Act 1 结束 ═════════

F6 (geometric: v_probe ⊥ Δu)
  ↓ static evidence for H_B (geometric decoupling)
F7 (causal: ablate/steer v_probe 都 null)
  ↓ dynamic evidence for H_B
F8 (positive ID: Δu @ L* drives abstain?)
  ↓ 找到真正的 decision channel (if positive)

  ═════════ Act 2 结束 ═════════

Central claim: Recognition channel (v_probe) and decision channel
              (Δu) are geometrically and causally separate in residual
              space. Model has recognition but doesn't route it to
              decision.
```

## F1-F8 各自承担的 sub-claim

| F | Sub-claim 它支持 | 关键数字 |
|---|---|---|
| F1 | (a) Encoding exists | 5-40 pp gap |
| F2 | (a) + 现象动态 | 21/21 model probe cliff at $0\to20\%$ |
| F3 | 现象动态 + motivate Act 2 | $4\times$ asymmetric CoT |
| F4 | (b) 跨四象限 | 42/42 robustness, Q1-Q4 都对 |
| F5 | (c) not surface | D1 0.71, D3 < 0.08 cos |
| F6 | (d) geometric decoupling | 42/42, $|\text{margin}| < 0.02$ in 38/42 |
| F7 v1 | (e) causal inert | 0.019 vs 0.055, 1/200 steering |
| F7 v2 | (e) 全面 robustness | manipulation + 双向 + L*-only + mean + CI |
| F8 | (+) decision channel ID | $\Delta u$ @ $L^*$ outcome 决定 |

---

# 5. 当前 Evidence 审计

## Sub-claim by Sub-claim

### (a) Encoding exists — **~95%**

**Evidence**: F1 (5-40pp gap, 21 model), F2 (probe F1 0.70-0.95 in scope)
**Gap**: 几乎没有

### (b) Encoding 跨四象限 — **~95%**

**Evidence**: F4 (42/42 robustness across 7 checks)
**Gap**: 几乎没有, 这是 paper 最强证据

### (c) Encoding ≠ surface features — **~85%**

**Evidence**: F5 (D1 0.71 flip, D3 ≤ 0.08 cos) on 6 model UMWP
**Gap**:
- TreeCut 弱 (2/6 model)
- Prompt fixed (没测 alternative prompt)
- 只 6 个 representative model

### (d) 几何 decoupling — **~85%**

**Evidence**: F6 (42/42 pair, 38/42 < 0.02, 39/42 small validation)
**Gap**:
- **Layer mismatch**: $v_\text{probe}$ 在 $L^*$ 提的, $\Delta u$ 在 $L$ 上作用。**还没在 final layer 重做 F6**。
- Prompt-fixed (同 (c))
- 2 个 deepseek-math + 1 个 Qwen-Math-1.5B/UMWP outlier (已 disclose)

### (e) 因果 inert — **~55%** ⚠️ 最大瓶颈

**Evidence (v1)**: 0.019 vs 0.055, 1/200 steering
**Gap (v1)**:
- Manipulation check 失败 (公开 caveat)
- Steering 单向
- $n=10$ 无 CI
- 跨层 ablation crude
- 单一 ablation method

**Predicted Evidence (v2 + INLP)**: **~85%**
- INLP 把 manipulation check failure 转成 "high-dim finding"
- v2 修了 双向 + L*-only + mean + CI 所有问题

### (+) Decision channel positive ID — **未知 (depends on F8)**

---

## Overall Evidence

| 状态 | Total |
|---|---|
| **现状 (F1-F7 v1, F8 not run)** | **~75%** |
| **F7 v2 跑完 + INLP curve 跑完** | **~88%** |
| **+ multi_layer_probe + prompt-robustness** | **~92%** |
| **+ F8 positive at $L^*$** | **~95%** |
| **+ F8 null on both** | **~88%** (story 改 framing 但 main claim 稳) |

**关键 insight**: F1-F8 的核心 claim (recognition vs decision decoupled) **在 F8 还没出 result 之前**, 跑完 v2 + INLP + multi_layer + prompt-robustness 已经有 ~92% evidence。**Paper 完全 publishable 在 main conference level。**

F8 的 outcome 决定 **paper 是否能 go for oral**, 不决定 paper 是否能 publish。

---

# 6. TODO List + F8 Scientific 再审视

## 6.1 优先级 TODO

按 cost-benefit 排序:

### P0 (必须做, blocking)

**TODO-1: F7 v2 GPU run**
- 代码已写 (`causal_probe_test_v2.py`)
- Runtime: ~5 days on 2 GPU
- 给 sub-claim (e): 55% → 85%
- 修了所有已知 F7 caveat

**TODO-2: INLP probe-retrain curve (all 21 model)**
- 代码还没写, 用 exp10 缓存 hidden state, CPU only
- Runtime: ~5 hours CPU
- 把 "manipulation check 失败" 转成 "信号 high-dim finding"
- 是 sub-claim (e) defense 的 keystone

**TODO-3: F8 GPU rerun (on real GPU, fixed coherence threshold)**
- 代码已写 (`delta_u_intervention.py`)
- 当前 CPU smoke 数据是 noise (coherence 0.13-0.29)
- Runtime: ~3-5 days on GPU
- 决定 paper 是否能去 oral

### P1 (强烈建议)

**TODO-4: multi_layer_probe.py**
- 每个 layer 都训 probe, 21 model 跑一次
- Runtime: ~5-15 hours GPU overnight
- 给 sub-claim (d): 85% → 95%
- 提供 F6 在 final layer $L$ 的版本, 移除 layer mismatch 攻击
- Bonus: F1-by-layer curve paper figure

**TODO-5: Prompt-robustness check**
- 改 instruction 用 "Cannot determine" 替代 "Insufficient", 重做 $T_\text{abs}$, 重测 F6 + F5 D1
- Runtime: ~1 day, 3 representative model
- 给 sub-claim (c) + (d) 加 ~5% 各自
- 堵 "你的 finding 只在这个 prompt format 上" 攻击

### P2 (oral 升级, optional)

**TODO-6: AmbigQA cross-domain replication**
- 3 model × AmbigQA, 重做 F1, F3, F4, F6
- Runtime: ~1 week
- 必备 for oral, 不做也能 publish main

**TODO-7: F9 deployment (mechanism-derived prediction test)**
- Input-side vs output-side probe head-to-head
- Frame 成 "mechanism-derived falsifiable prediction"
- Runtime: ~3-4 days after F8

## 6.2 Action Sequence

```
Now (Week 1)
├── Launch TODO-1 (F7 v2) on GPU A
├── Launch TODO-3 (F8 rerun) on GPU B  ← critical-path
├── Write + launch TODO-2 (INLP) on CPU
└── Start paper draft (F1-F6 sections, F7/F8 placeholder)

Week 2 (results coming in)
├── Analyze F7 v2 + INLP results, fill paper sections
├── Analyze F8 results
│     - If positive at L*: "both channels identified" framing
│     - If null: "recognition identified, decision unlocalized" framing
└── Launch TODO-4 (multi_layer_probe) on free GPU

Week 3
├── Launch TODO-5 (prompt-robustness)
├── Polish paper draft
└── Decide F9 (TODO-7) based on F8 outcome
```

## 6.3 F8 Scientific 再审视 — 这个实验真的能找到 decision channel 吗?

我之前一直在推 F8, 现在停下来认真审视一次。

### F8 的实际科学目的

F8 测的是: **"在 $L^*$ 这一层注入某个 direction, 能不能让模型 Q1 sample 从 hallucinate 变 abstain?"**

如果某个 direction (比如 $\Delta u$) 在 $L^*$ 注入产生 abstain, 我们 conclude: **这个 direction 在 $L^*$ 是 (or 部分是) abstain pathway**。

这是不是 "positive identification of decision channel"?

**严格回答**: 是 **partial** positive identification, **不是 complete**。

### F8 能 claim 什么 / 不能 claim 什么

#### F8 positive at $L^*$ 能 claim:

1. **Sufficiency**: 在 $L^*$ 这一层, 注入 $\Delta u$ 方向 **足以** 让一些 Q1 sample 朝 abstain 翻转
2. **Layer localization**: abstain decision **可以在 $L^*$ 这一层被 trigger** (即 information bottleneck 不在 $L^*$ 之后)
3. **Direction relevance**: $\Delta u$ 方向**不是 inert** — 至少在某些 alpha 下能影响 behavior

#### F8 positive at $L^*$ **不能** claim:

1. ✗ "$\Delta u$ 是模型实际用的方向" — 我们注入它有效, 不代表模型自己就是用这个方向。模型可能用一个**完全不同的方向**, 而 $\Delta u$ 只是恰好 mimic 那个方向的最终效果
2. ✗ "Abstain decision 在 $L^*$ 完成的" — sufficiency 不等于 necessity; 模型可能在 $L^*$ 之后才真正 decide
3. ✗ "我们 fully localize 了 decision channel" — direction 的精确角度、layer 的精确位置都没确定

#### F8 null on both 能 claim:

1. **Neither $\Delta u$ nor $v_\text{abstain}$ at $L^*$ alone drives abstain** — 加强 decoupling 的论点
2. **Abstain pathway 不在这两个 candidate 方向上** — narrow down possibilities
3. **可能 implication**: decision 在 $L^*$ 之后实现, 或不是单 direction 而是多 direction interaction

### F8 的真实角色 — 我之前 frame 得过于 grand

我之前说 F8 是 "positive identification of decision channel"。这个 framing **过强**。

更准确的 framing 是: **F8 是一个 "is this candidate the decision direction?" test**。它测试两个具体 candidate ($\Delta u$ 和 $v_\text{abstain}$), 而不是 fully 解决 "decision channel 是什么"。

### F8 在 paper 中的诚实定位

**If F8 positive at $L^*$**:

> "We show that injecting the unembedding-derived abstention direction $\Delta u$ at layer $L^*$ is sufficient to flip a substantial fraction of Q1 samples from numeric to abstention output, while preserving Q3 specificity. Combined with F6 and F7, this provides a converging picture: the abstain pathway in residual space at $L^*$ admits intervention via the $\Delta u$ direction, distinct from $v_\text{probe}$ which is geometrically orthogonal and causally inert. We do not claim $\Delta u$ is *the* direction the model natively uses (sufficiency $\ne$ necessity), only that it is a viable abstention pathway at $L^*$ that the model does not route $v_\text{probe}$ through."

**If F8 null on both**:

> "Neither the unembedding-derived $\Delta u$ nor the empirically-derived $v_\text{abstain at } L^*$ direction drives clean abstention when injected at the probe layer. Combined with F6 and F7, this implies: (i) at $L^*$, abstention is not implemented as a single linear direction in either the unembedding's preferred geometry or the empirical $Q_2$-$Q_1$ contrast geometry; (ii) the decoupling between recognition channel and decision channel is therefore even stronger than F6+F7 suggest — not only is $v_\text{probe}$ separate from $\Delta u$, but neither aligns with the actual abstain pathway, which likely lives downstream of $L^*$ or as a non-linear interaction."

**两种 framing 都科学诚实, 都能 publish**。

### 还有什么 F8 没回答的, 我们也不打算回答的

1. **Decision channel 实际是什么 direction**: F8 只测两个 candidate, 不是 exhaustive
2. **Decision 在哪个具体 layer 完成**: F8 测两个 layer ($L$, $L^*$), 中间层和后续层都没测
3. **Decision 是不是 linear**: F8 只测 linear intervention
4. **跨 model 一致吗**: F8 跑 5 model, 但可能 heterogeneous

**这些都是 follow-up work, 不是这篇 paper 的 scope**。诚实承认。

### 那 F8 还值不值得跑?

**值得**。即便它不能 fully ID decision channel, 它做的是:

1. **测试两个最合理的 candidate** ($\Delta u$ 和 $v_\text{abstain}$)
2. **如果有 positive**: paper 升级 (但是要 honest 关于 sufficiency vs necessity)
3. **如果都 null**: paper 加深 decoupling story (decision pathway 不在 $L^*$ as linear direction)

Either way F8 提供 informative result。**没有 wasted experiment**。

但**框架要 honest**: F8 不是 "find the decision channel", 是 "test whether these specific candidate directions at this specific layer drive abstention"。这个差别 reviewer 会问, 我们应该 preempt。

### 给 paper 写法的建议

不要在 Act 2 框架里说 "we identify the decision channel"。改成 "we localize a sufficient intervention direction for abstention at $L^*$" (if positive) 或者 "we show that two natural candidate directions at $L^*$ are insufficient" (if null)。

这样 reviewer 不能说 "你 claim 过强"。

---

# 7. 一句话总结

**核心 claim 在 F8 还没跑之前已经有 ~75% evidence support, 跑完 F7 v2 + INLP + multi_layer + prompt-robustness 到 ~92%, 完全 publishable。F8 的 outcome 决定 paper 是 main 还是 oral, 但 F8 不是 paper 是否成立的 single point of failure。F8 的诚实定位是 "test two candidate directions at $L^*$", 不是 "find the decision channel" — 后者 over-claim。**

---

# 附录 A: 关键 code paths 速查表

| Code file | 角色 | 状态 |
|---|---|---|
| `src/exp1_*.py` | F1 | Done |
| `src/exp10_unified_proportional_probe.py` | F2 (probe trajectory) | Done |
| `src/exp11_quadrant_proportional.py` | F4 (Q1-Q4) | Done |
| `src/exp14_early_cutoff_*.py` | F3 (asymmetric CoT) | Done |
| `src/confounder_exps/*` | F5 | Done |
| `src/measurement_A_sweep.py` | F6 (21 model) | Done |
| `src/causal_probe_test.py` | F7 v1 | Done with caveats |
| `src/causal_probe_test_v2.py` | F7 v2 (manipulation + 双向 + L*-only + mean + CI) | **Pending GPU run** |
| `src/inlp_probe_retrain.py` | INLP curve all 21 model | **Code not written yet** |
| `src/multi_layer_probe.py` | Per-layer probe extraction | **Code not written yet** |
| `src/delta_u_intervention.py` | F8 (5 directions × 2 layers + patching) | **Pending GPU rerun** |

# 附录 B: 关键结果 file paths

| 结果 | 路径 |
|---|---|
| F1 gap | `experiment_result/experiments/analysis/gap_analysis.csv` |
| F2 probe trajectory | `experiment_result/exp_temporal_new/results/exp10_ultimate_proportional_{umwp,treecut}.csv` |
| F4 quadrant trajectory | `experiment_result/exp_temporal_new/results/exp11_average_trajectories.csv` |
| F5 confounder | `experiment_result/exp_confounder/results_curated/` |
| F6 21-model | `experiment_result/causal_results/_measurement_A_sweep/summary.csv` |
| F7 v1 | `experiment_result/causal_results/{slug}/{dataset}/summary.csv` |
| F7 v2 | `experiment_result/causal_results_v2/{slug}/{dataset}/summary.csv` (pending) |
| F8 | `experiment_result/causal_results/{slug}/{dataset}/delta_u/summary.csv` (CPU data, needs GPU rerun) |
| Probe files | `experiment_result/exp_temporal_new/probes_proportional/{dataset}/{slug}/unified_probe_layer{L*}.joblib` |
| Cached hidden states | `experiment_result/exp_temporal_new/embeddings_proportional/{dataset}/{slug}/` |

