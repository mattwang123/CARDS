# Research Summary & Storyboard: Epistemic Dynamics in Generative Reasoning
**Target Venue:** EMNLP 2026 (Oral / Best Paper Track)
**Strategic Framing:** Methodology & Phenomenological Discovery (机制发现与可解释性方法论)
**Research Intersection:** Test-Time Compute (长思维链) $\times$ Mechanistic Interpretability (机制可解释性) $\times$ Faithful Reasoning (忠实推理)

---

## 👑 暂定论文题目 (Working Titles)
1. **Generative Onset Shock:** How Test-Time Compute Disrupts Epistemic State in Large Language Models
2. **The Representation-Verbalization Gap:** Epistemic Collapse and Recovery During Long Chain-of-Thought
3. **Into the Cognitive Abyss:** Scaling Laws of Epistemic Retention in Generative Reasoning

---

## 一、 战略定位与核心叙事 (Strategic Framing & Core Narrative)

### 1. 为什么这不是一篇普通的分析报告？(The Impact & Framing)
在当前的学术界与工业界（如 OpenAI o1, DeepSeek-R1），"Test-Time Compute"（让模型思考得越久越好）被狂热推崇，视为通向 AGI 的银弹。本文的价值在于**极其罕见的“逆向发声与敲响警钟”**：
我们不提供新的 Benchmark 进行刷榜，也不仅仅是浅层的 "We analyze X and find Y"。我们通过严格的机制可解释性（Mechanistic Probing）揭示了一个极度反直觉的底层物理法则——**长程的推理算力（Test-Time Compute）本身是一把双刃剑，它会作为一种“认知噪音（Cognitive Noise）”，覆盖甚至扭曲模型本应具备的初始清醒认知。** 这类揭示大模型基础生成缺陷的 Phenomenological Discovery，天然具备冲击 Best Paper 的潜质。

### 2. 核心现象：生成式启动休克 (Generative Onset Shock)
为了提升理论深度，我们将此现象与经典的检索衰减进行对标：
* **Nelson's Retrieval LitM (Liu et al., 2023):** 这是关于**“读 (Reading)”**的注意力衰减。静态长文本的中间信息被 Attention 机制天然遗忘。
* **Our Generative LitM (本文核心):** 这是一个关于**“写/推理 (Reasoning)”**的内部表征冲击。在 $t=0$ 时，模型能够精准判断题目条件是否充足（Lucid Start）。然而，当模型决定开始生成思维链的**最初 10%**（例如写下 `Let x = ...` 的解题模板）时，隐层经历了从“全局阅读”到“局部算术”的剧烈状态切换，导致代表“条件不足”的全局认知遭遇**断崖式坠落，即“启动休克 (Onset Shock)”**。此后在漫长的推理中，模型实际上是在缓慢地“找回自我”，但在输出时却被惯性裹挟。

---

## 二、 Q1-Q4 轨迹的深度因果剖析 (Deep Causal Analysis of Quadrants)

基于比例化探针 (Proportional Probing, Exp 11) 获取的 Average 概率轨迹（以 Treecut 为例），我们对模型的四个行为象限进行了深度的因果关系（Causality）剖析：

### 1. 表征-输出鸿沟 (The Representation-Verbalization Gap)
* **Q2 (Correct Reject, "知行合一"):** $P(\text{insuff})$ 均值高达 **0.739**。模型隐层清醒，表层也成功拒绝，这是理想状态。
* **Q1 (Hallucination, "心口不一"):** 极度反直觉的是，Q1 的隐层 $P(\text{insuff})$ 均值在末期同样恢复到了高达 **0.669**。隐层已经拉响了警报，为何表层依然强行瞎编答案？
  * **深层机制解释:** 解码器（Decoder）在生成时不仅受隐层状态影响，更受到**局部语法惯性 (Local Syntactic Momentum)** 和 SFT 带来的**指令跟随偏置 (Instruction-Following Bias)** 的强烈劫持。一旦模型在开头写下了标准解题模板（如 `Let x = ...`），它就像一列刹车失灵的火车。即便隐层不断发出“条件不足”的高频警报（0.669），这股警报也无法跨越这道**表征-输出鸿沟 (Representation-Verbalization Gap)** 去扭转解码器的 Logit 分布。

### 2. 局部计算受阻引发的全局自我怀疑 (The Causality of Reasoning Doubt)
* **Q3 (Correct Solve, "自信推导"):** 题目条件充足且做对时，概率极低（**0.185**）。说明当模型顺畅进行代数推导时，它内心非常笃定“条件是充足的”。
* **Q4 (Competence Failure, "认知动摇"):** 这是一个极其炸裂的发现！题目条件**明明是充足的**，仅仅因为模型在长程计算中算错了、卡壳了或逻辑混乱了，其隐层 $P(\text{insuff})$ 竟然异常升高至接近临界点的 **0.412**。
  * **因果推论 (The Causality):** **局部的计算流畅度 (Local Computational Fluency) 是全局认知状态 (Global Epistemic State) 的因变量 (Causal Driver)。** 当计算陷入泥潭时（Q4），这种局部的“计算受阻感”会反向污染它对“题目是否合规”的全局判断，导致隐层产生广泛的“认知不自信 (Epistemic Doubt)”。
  * *(注：在评估脚本中，我们需将 Q4 细分为 `Q4_Math_Error` [算错数字] 与 `Q4_False_Reject` [错答Insufficient]，对比这两种截然不同的思维模式将进一步丰富结论。)*

---

## 三、 缩放定律：对抗覆盖的物理护城河 (Scaling Laws of Retention)

我们首次定义了**认知信息容量上限的缩放定律 (Scaling Law of Information Upper Bound)**。通过对比 100% 与 0% 时的 Separate Probe F1 差值 (Exp 10)：
* **小模型的物理性失忆:** (如 Qwen-1.5B, $\Delta F1 \approx -0.20$)。在经历数千 token 的运算后，其狭窄的残差流带宽被完全挤占，彻底丧失了判断全局条件的能力。
* **大模型的金身不坏:** (如 DeepSeek-32B, Qwen-72B, $\Delta F1 \approx -0.02$)。凭借海量的高维带宽，在历经计算风暴后依然能完美锁住 $t=0$ 时的初始认知，呈现极美的对数函数拟合。

---

## 四、 实验体系骨架 (The Core Experimental Pipeline)

我们摒弃了所有冗余的分析，正文逻辑由四大基石构成（严密咬合，形成闭环）：
1. **Exp 1 (Native Gap):** 确立 $t=0$ 时隐层识别与表层输出的原生鸿沟 (`gap_analysis.csv`)。
2. **Exp 2 (CoT Gen & Eval):** 将模型行为严格切分为 Q1-Q4 象限，为轨迹分析奠定基石。
3. **Exp 10 (Proportional Extraction):** 按百分比提取特征，证明 F1 信息容量上限的 Scaling Law (`exp10_ultimate_proportional_*.csv`)。
4. **Exp 11 (Trajectory Averages):** 获取动态演变规律，绘制 Q1-Q4 均值曲线，揭示 Onset Shock 核心图表 (`exp11_average_trajectories.csv`)。
*(附加: Exp 13 Causal Steering 作为一个重要的 Discussion，证明了强行在残差流注入向量会导致语法瘫痪，反向证实了 Generative Momentum 的不可逆性。)*

---

## 五、 Next Steps 冲刺计划 (Crucial Actions for Oral Defense)

为了防守顶会审稿人（尤其是 Reviewer 2）的苛刻攻击，必须立即落实以下三项“保命与升华”行动：

### Action 1: 绝对防御 —— 最小反事实对验证 (Minimal Contrastive Pairs) `[最高优先级]`
* **防守目标:** 彻底击碎 "Clever Hans Effect" (探针作弊) 质疑。证明探针测的是深层“逻辑缺失”，而非浅层对 `assume`, `not specified` 等特定词汇的过拟合。
* **执行方案:** 整合已有的 `run_all_confounders.py` 和 `comprehensive_confounder_results.json`。在论文独立章节展示：面对文本高达 95% 相似但逻辑截然相反的 Prompt Pairs，在**不生成任何 CoT** 的情况下，探针依然能给出极度悬殊的概率预测。这是确立方法论合法性的唯一免死金牌。

### Action 2: 领域泛化 —— 扩展非数学数据集 (Cross-Domain Generalization) `[提升 Impact 必备]`
* **防守目标:** 如果仅有 UMWP 和 TREECUT，评委会认为 Onset Shock 只是大模型处理“数学题”时的独有 Artifact。
* **执行方案:** 必须引入非数学任务，证明这是大模型生成的**普适定律 (Fundamental Flaw)**：
  1. **Logical Reasoning (逻辑推导):** 改造 `ProofWriter` 或 `PrOntoQA`，故意漏掉一条三段论的前提。
  2. **Real-world QA (真实世界问答):** 改造 `HotpotQA`，删掉多跳推理中的关键桥接文档。

### Action 3: Token 级别的微观病灶对齐 (Temporal Token Alignment) `[定性可视化]`
* **目标:** 让“启动休克”变得肉眼可见。
* **执行方案:** 从 `exp11_sample_wise` 文件夹中提取经典的 Q1 样本，将 $P(\text{insuff})$ 的暴跌轨迹与生成的具体文本对齐。定性证明概率从 0.8 砸到 0.4 的瞬间，精确对应着模型写下第一步复杂代数设定（如 `Let x = ...`）的那个 Token。