# HVR 救援执行计划 (Rescue Plan)

> **定位**：本文档是**初稿提交 (2026-04-15) → 最终答辩**之间的分阶段执行路线图与决策逻辑。实现细节（config 模板、`do.sh` 任务、loss 类骨架、manifest 扩展、per-chapter HVR 分解、μ sweep 决策树的数值阈值）见姊妹文档 [`2026-04-14-hierarchy-loss-improvements-future.md`](./2026-04-14-hierarchy-loss-improvements-future.md)（以下简称 **未来工作文档**）。本文不复述 code/config，只规定**做什么、按什么顺序做、看到什么数字就做什么决定、最终章节怎么写**。

---

## 1. 问题定义

2026-04-14 完整评测结果：E1（μ=0.1，hierarchical head + soft chap-head penalty）相对 E0（flat BCE）**Mean AUROC 仅 +0.009**（0.671 → 0.680），**HVR (mean) 仅下降 0.005**（0.495 → 0.490，噪声量级），而显式 chap head 上测得的 **HVR (head) = 0.364** — 说明 hierarchy loss 在它自己定义的约束目标上都**没有收敛到接近 0**。E2–E4 同步呈现 HVR_head ∈ [0.32, 0.39]，HVR_mean 不动。

这直接威胁论文 Innovation 2 (**hierarchical loss 能强制预测与疾病分类层次一致**) 的立论：当前数据既不支持"loss 有效"、也没给出有价值的 negative result。距答辩约 7 周，必须在这段时间内：要么找到让 HVR_head → 0 的配置，要么把失败重塑成"我们量化诊断了默认配置的失败 + 提出了 HVR 评测框架"这种**仍具贡献度**的叙事。否则 chapter 5 只剩 Innovation 1 (multi-task) 和 Innovation 3 (PEFT)，单薄且与文献重叠度高。

## 2. 救援假设

### H1 — μ 量级不足 (weight imbalance)

**表述**：`μ_hierarchy=0.1` 与 `L_leaf` (BCE weight=1.0) 相差一个数量级，hierarchy penalty 在总梯度中被淹没；loss 项虽在场，优化器无压力去降它。**预测**：若 H1 成立，μ ↑ 10–100× 后 HVR_head 会单调下降，在某个 μ* 处趋近 0；AUROC 可能轻微退化但幅度 < 0.02。**验证**：Phase 1 μ sweep（§3.1）。**拒绝条件**：μ=10 时 HVR_head 仍 ≥ 0.2，或 HVR_head 先降后反弹 → 单纯加权不够，转 H2/H3。**拒绝含义**：问题不在标量权重，而在 loss 形式或 architecture。

### H2 — Loss target 选错 (chap-head 非 leaf 聚合)

**表述**：默认 `L_hierarchy` 比较两个**独立 Linear head** 的 sigmoid 输出。两 head 仅共享上游 256-d projection，下游参数解耦；chap head 可以自行把输出压低以满足 `σ(leaf) ≤ σ(chap)`，而 leaf head 分布完全不受约束（"escape valve"）。**预测**：若 H2 成立，μ 很大时 HVR_head 会降但 HVR_mean 不动（已在 E1 数据中观察到端倪）。改 target 为 `mean_aggregated_leaf`（对 chapter 内 leaf probs 求均值作 RHS）后，HVR_mean 应显著下降。**验证**：Phase 2.1（§3.2）。**拒绝条件**：mean-aggregated 变体下 HVR_mean 仍 ≥ 0.4。**拒绝含义**：leaf head 自身表达力未受约束，需动架构（H3）。

### H3 — Architecture 不够 coupled (两 head 独立)

**表述**：即使 loss target 正确，两 head 在梯度路径上几乎独立——chap head 的梯度只经 projection 反传一层 Linear 就到 backbone，leaf head 同理；hierarchy 约束无法通过"结构归纳偏置"强制传播。**预测**：把 chap head 重构为 leaf logits 的**可学习聚合**（`chap_logits = aggregate(leaf_logits) + residual`），或 warm-up curriculum 先 BCE 后 hierarchy，应能在不靠暴力加 μ 的前提下同时压低 HVR_head 和 HVR_mean。**验证**：Phase 3（§3.3）。**拒绝条件**：残差聚合 head + curriculum 组合下 HVR_head 仍 ≥ 0.2 且 AUROC 退化 > 0.03 → 承认 soft hierarchy loss 在本数据规模下存在 fundamental tension，走 Tier C negative result。

## 3. 三阶段执行计划

每个实验都以下述格式给出：**H**（所测假设）/ **Config**（改什么，引用未来工作文档章节）/ **Runtime**（Nebula 4090 单卡 E1 ≈ 2.5h；见未来工作文档估算）/ **Go/No-Go**（推进到下一阶段的判据）。

### Phase 1 — μ sweep (最高 P(success)，一夜 Nebula，先跑)

**统一前提**：沿用 E1 backbone（`sft_full_ft.yaml`，lambda_chapter=1.0，40 epoch，bs=512，lr=2e-5），只改 `mu_hierarchy`。E1 原始 μ=0.1 结果 symlink 进 sweep 目录（见未来工作文档 §A.1）。

| 实验 ID | μ | H | Go/No-Go |
|---|:---:|:---:|---|
| P1.0 | 0.0 | 基线（hierarchy off） | 无 go/no-go，只作 anchor |
| P1.1 | 0.5 | H1 | 若 HVR_head < 0.36（比 μ=0.1 进一步下降），确认"压更大更低"单调性 |
| P1.2 | 1.0 | H1 | 配对 BCE 量级；若 HVR_head < 0.2 已达 Tier A 候选 |
| P1.3 | 3.0 | H1 | 若 HVR_head < 0.05 但 AUROC drop > 0.05，触发 Tier B |
| P1.4 | 10.0 | H1 extreme | 若训练发散/NaN，启动 curriculum warmup；若 HVR_head 仍 ≥ 0.2，H1 **被拒绝**，直接跳 Phase 2 |

**Metrics（每个 run）**：Mean AUROC (fair), per-disease AUROC, HVR (mean), HVR (head), **per-chapter HVR 分解**（未来工作文档 §A.5），训练曲线中 `loss_hierarchy.item()` 轨迹（确认 loss 真的在降而不是一开始就 ≈ 0）。

**Decision gate（Phase 1 结束时）**：按未来工作文档 Appendix B 决策树定位 (μ=10 的 HVR_head, ΔAUROC) 落点。

- 若存在 μ* 使 HVR_head < 0.05 ∧ ΔAUROC < 0.01 → **直接走 Tier A**，Phase 2/3 作为 supplementary 跑 1–2 个对照即可。
- 若 μ=10 出现 "HVR_head≈0 但 HVR_mean 不变" → H2 确认，**优先跑 Phase 2.1**（mean-aggregated leaf target）。
- 若 μ=10 出现 NaN/发散 → **优先跑 Phase 3.1** (curriculum warmup)，不盲目降 μ。
- 若所有 μ 下 HVR_head 都 ≥ 0.2 → H1 拒绝，**跳 Phase 2 全部实验**。

### Phase 2 — Loss re-target (并行 ~half-night)

基于 Phase 1 的最佳 μ（记为 μ†，若 Phase 1 无合格点则固定 μ†=1.0）作为 Phase 2 所有实验的 μ。

#### P2.1 — Mean-aggregated leaf hierarchy loss  **[H2]**

**Config**：`configs/sft_full_ft_leaf_target.yaml`（未来工作文档 §A.1–A.2、A.4 已给出 `HierarchicalLossMeanAggregated` 实现与 `hierarchy_loss_variant: mean_aggregated` 字段分发）。`mu_hierarchy = μ†`。**Runtime**：1 × 2.5h。**Go/No-Go**：若 HVR_mean 从 ~0.49 降到 < 0.2 → H2 成立，核心变体确认；若 HVR_mean 仍 ≥ 0.4 → 转 Phase 3。

#### P2.2 — Hard projection (inference-time endpoint)  **[μ=∞ anchor]**

**Config**：`configs/sft_full_ft_hard_proj.yaml`，在模型 forward 出口对 leaf probs 执行 `min(σ(leaf), σ(chap))` clamp（未来工作文档 §A.4 末段）。训练时 loss 同默认，只在 eval 用；也可选在训练末 5 epoch 加入作为 regulariser。**Runtime**：0.5h（或 0h，仅 re-eval 既有 E1 ckpt）。**Go/No-Go**：HVR 必为 0；记录 AUROC 作为 hierarchy 约束**强加后的性能上限代价**——给 Pareto curve 提供右端点。无论 Phase 1 结果都应跑，成本极低。

#### P2.3 — Per-chapter HVR breakdown 分析  **[诊断]**

**Config**：无训练，改 eval 脚本（未来工作文档 §A.5），对 P1.0–P1.4 + P2.1 所有 ckpt 重跑，输出 per-chapter HVR 矩阵（6 chap × N runs）。**Go/No-Go**：若 max/min per-chapter HVR > 5×，说明单个 chapter 主导违例 → Phase 3 需考虑 chapter-aware reweighting；否则维持 uniform μ。这一步是 Chapter 5 "我们刻画了违例的结构不均匀性" 的数据来源，**必做**。

### Phase 3 — Structural changes (conditional，仅在 P1+P2 不足以达 Tier A/B 时跑)

**触发条件**：Phase 1 Go/No-Go 未给出 μ*，**且** Phase 2.1 未将 HVR_mean 降至 < 0.2。此时说明 "加权 + loss target 修正" 都不足，需要结构性干预。**预算**：最多 3 个 E1-scale job（~7.5h on Nebula 单卡，双卡并行 ≈ 4h）。

#### P3.1 — Warm-up curriculum  **[H3a]**

**Config**：前 20% epoch `μ=0`（纯 BCE），20–30% 线性 ramp 到 μ†，30–100% 保持 μ†。实现：在 `train_multitask.py` scheduler 侧加 μ scheduler（非 future doc 显式覆盖范围，但只需 ~10 行新代码）。**Runtime**：2.5h。**Go/No-Go**：若相对 P1 同 μ† 基线，HVR_head 再降 ≥ 0.1 且 AUROC 不退化 → 加入论文。否则标为 "tried, ineffective"。

#### P3.2 — Chap head init as leaf aggregator  **[H3b]**

**Config**：`chapter_head.weight` 初始化为 leaf-to-chapter 邻接矩阵（按每 chapter leaf 数归一化），`bias=0`；与 H3a 独立可叠加。训练期允许继续更新。**Runtime**：2.5h。**Go/No-Go**：若仅改 init 即把 HVR_head 从 0.36 → < 0.15 → 写入方法贡献；否则作为 ablation 附录。

#### P3.3 — Residual chap head (显式结构耦合)  **[H3c]**

**Config**：`chap_logits = A @ leaf_logits + residual_head(proj)`，其中 A 为固定（或可学习 low-rank）leaf→chap 聚合矩阵，`residual_head` 是一个小 Linear；前向路径保证 chap 与 leaf **共享大部分容量**，独立逃逸通道被削减为 residual。**Runtime**：3h（多一层 fwd/bwd）。**Go/No-Go**：这是 Phase 3 的 "terminal" 实验；若它仍达不到 HVR_head < 0.2 且 AUROC ≥ 0.65 → 接受 Tier C，不再投入算力。

---

## 4. 成功判据 (Three-tier success criteria)

所有阈值均在 **fair eval**（16 disease × mean AUROC 口径）上定义，HVR 同时报 head 与 mean 两种。

### Tier A — Full rescue

**数字条件**：存在 μ*（可能配合 Phase 2.1）使：
- HVR_head < 0.05 **且** HVR_mean < 0.15
- Mean AUROC (fair) ≥ 0.670（E0 baseline 水平或更高）
- 至少 4/6 chapter 的 per-chapter HVR < 0.1

**Chapter 5 narrative**："我们证明 hierarchy loss 在 metabolomic BERT + 16 disease 设定下**有效**，但其效果对 μ 高度敏感；在 μ=μ* 工作点，HVR 从 E0 的 0.49 降至 X，AUROC 持平或略升。我们进一步通过 mean-aggregated leaf target 变体揭示了默认 chap-head-based 设计的 escape-valve 失效模式。"

**Future work 框架**：KL-divergence 重构、multi-level hierarchy、跨 cohort 泛化。

### Tier B — Pareto rescue

**数字条件**：μ sweep 给出清晰 HVR-AUROC 权衡曲线：
- 存在 μ 使 HVR_head < 0.1，但对应 AUROC 退化 ∈ [0.01, 0.05]
- 或：能画出单调 Pareto 前沿（μ ↑ → HVR ↓ ∧ AUROC ↓）**且**至少 2 个点落在前沿上
- 推荐工作点 μ_rec 明确可报（通常是 "HVR 降幅最大 且 AUROC drop ≤ 0.02" 的点）

**Chapter 5 narrative**："我们刻画了 hierarchy loss 权重 μ 所诱导的 HVR-AUROC Pareto 前沿，并建议 μ=μ_rec 作为默认工作点。该结果本身是本文第二项贡献——首次定量呈现 metabolomic 多标签分类中层次一致性与判别性能之间的 tradeoff。"

**Future work 框架**：自适应 μ、基于 validation HVR 的 μ scheduler、与 label correlation matrix 的联合优化。

### Tier C — Negative rescue (fallback)

**数字条件**：Phase 1–3 所有实验中，无 run 达成 HVR_head < 0.2；或达成的配置 AUROC 退化 > 0.08。

**Chapter 5 narrative**："默认 soft hierarchy penalty 在 metabolomic BERT + 小样本多标签场景下**存在结构性张力**：BCE 梯度量级主导、两 head 独立导致的 escape-valve、leaf-level prevalence 不均匀共同作用，使得 μ sweep、loss re-target、结构耦合三类干预均未能将 HVR_head 压至 < 0.2 而不牺牲判别性能。本工作的贡献转化为：(i) HVR 评测框架的提出，(ii) 三类干预的系统化消融，(iii) 对未来工作采用 KL-divergence / hyperbolic embedding 等替代形式的明确动机。"

**Future work 框架**：KL-divergence 形式（约束 conditional distribution 而非 marginal probs）、hyperbolic / Poincaré embedding for tree-structured labels、label hierarchy encoder（而非 loss term）。

---

## 5. 时间线 (到答辩的 7 周)

假设最终答辩在 **2026-06-02 前后**，初稿 2026-04-15 提交。

| 日期 | 任务 | 交付物 |
|---|---|---|
| 04-14 (今) | 写救援计划 + 未来工作文档（已完成） | 本文档 + implementation doc |
| 04-15 | 初稿提交（按当前 E0-E4 数据 + Tier C-style 写 Innovation 2） | 初稿 v1 |
| 04-16 | Phase 1 configs 生成（未来工作文档 §A.1）、`do.sh` 接入（§A.2）、manifest 扩展（§A.3） | 5 个 μ config + `e1_mu_sweep` task |
| 04-17 夜 | Phase 1 μ sweep 串行或并行跑（4090 单卡 5×2.5h ≈ 12h，或 4 卡并行 ≈ 3h） | P1.0–P1.4 ckpt + metrics |
| 04-18 | Phase 1 分析、画 (μ, HVR, AUROC) 三轴图、决策树定位 | Phase 1 report + 下一步分支 |
| 04-19 | Phase 2.1 (mean-agg)、P2.2 (hard proj)、P2.3 (per-chap breakdown) 并行 | P2 ckpts + per-chap HVR 矩阵 |
| 04-20 | Phase 2 分析；决定是否进 Phase 3 | Tier 判定初版（A/B/C 候选） |
| 04-21 ~ 04-24 | （条件）Phase 3 P3.1/P3.2/P3.3 | P3 ckpts |
| 04-25 | 全结果汇总，锁定最终 Tier | 终稿数据表 |
| 04-26 ~ 05-03 | Chapter 5 重写（按 §6 版本 A/B/C 之一），全文 consistency check | 论文 v2 |
| 05-04 ~ 05-10 | 导师 + 答辩委员会反馈迭代 | 论文 v3 |
| 05-11 ~ 05-24 | 答辩 slides、附录 supplementary、demo 图 | slides v1 |
| 05-25 ~ 06-01 | Mock defense、QA 准备（尤其是 Tier C 时的"为什么这仍是贡献"问答） | 答辩包 |
| 06-02 | 答辩 | — |

**缓冲**：Phase 3 若全部触发，04-21 ~ 04-24 刚好 4 天。若 Nebula 排队紧张，可把 Phase 3 压缩到 P3.3 一个 terminal experiment。

---

## 6. Writeup contingency — Chapter 5 段落三版本

以下三段可**逐字复制**进 Chapter 5 "Innovation 2: Hierarchical Consistency Loss" 小节的核心 paragraph，按最终 Tier 选一。

### 版本 A — Tier A 成功

> 我们提出的层次一致性损失 `L_hierarchy = mean(max(0, σ(p_leaf) - σ(p_chap)))` 在默认权重 μ=0.1 下并未收敛至有效区间（HVR_head=0.364）。为厘清失效原因，我们在 E1 基线上对 μ 做了系统 sweep（μ ∈ {0.0, 0.1, 0.5, 1.0, 3.0, 10.0}），发现 μ=μ\*={VALUE} 处 HVR_head 降至 {VALUE}、HVR_mean 降至 {VALUE}，而 Mean AUROC 在 fair 评测下维持 {VALUE}（≥ E0 flat baseline 0.671）。进一步，将 loss 右端由独立 chapter head 换为同一 chapter 内 leaf probs 的均值聚合后，HVR_mean 进一步下降 {VALUE}，印证了原始设计中两 head 独立逃逸通道的 escape-valve 失效机制。本节贡献有三：(i) 给出 μ 的可操作推荐区间；(ii) 揭示并修复 soft hierarchy loss 的 target 缺陷；(iii) 引入 HVR 指标量化层次一致性，作为多标签医学分类的通用评测工具。

### 版本 B — Tier B 成功

> 默认权重 μ=0.1 下的层次一致性损失仅带来微弱改善（HVR_mean 下降 0.005，AUROC +0.009），表明该 loss 项的梯度量级被 BCE 主导。我们通过 μ ∈ {0.0, 0.1, 0.5, 1.0, 3.0, 10.0} 的完整 sweep，首次刻画了 metabolomic 多标签分类中**层次一致性（HVR）与判别性能（AUROC）之间的 Pareto 前沿**：μ ↑ 时 HVR_head 单调下降至 {VALUE} (μ=10)，但 AUROC 同步退化至 {VALUE}；μ=μ_rec={VALUE} 处取得最佳折中——HVR_head={VALUE}, AUROC={VALUE}（相对 E0 退化 {VALUE}）。Per-chapter 分解进一步显示违例集中在 prevalence 较低的 {VALUE} 个 chapter 上，暗示未来 chapter-aware 重加权方向。本节贡献在于：(i) 首次定量呈现该 tradeoff 曲线；(ii) 给出工程上可复用的 μ 推荐值；(iii) 提出 HVR 指标作为跨实验的统一评测口径。

### 版本 C — Tier C（失败转贡献）

> 本节报告一个**经过三阶段系统干预仍未达预期**的负面结果，并论证其方法论价值。默认 hierarchy loss (μ=0.1) 下 HVR_head=0.364、HVR_mean 相对 E0 仅降 0.005。我们依次测试了三类假设：(i) 权重失衡（μ sweep 至 10），(ii) loss target 错配（mean-aggregated leaf 变体），(iii) 架构解耦（残差 chap head + warmup curriculum）。在 {TOTAL_N} 组实验中，没有任何配置同时满足 HVR_head < 0.2 与 AUROC ≥ 0.65；最佳配置为 {CONFIG}，达成 HVR_head={VALUE}、AUROC={VALUE}。这一结果的诊断价值在于：(i) 本文提出并发布了 **HVR 评测框架**（head-based 与 mean-aggregated 双口径），使层次一致性首次在本领域可度量；(ii) 通过系统消融识别出默认 soft penalty 设计的**三重张力**（量级、target、结构），为后续 KL-divergence / hyperbolic embedding 重构提供明确动机；(iii) dual-head 架构本身在 chapter 级预测任务上仍给出可解释的中间表示，作为方法贡献独立于 loss 有效性成立。

---

## 7. 关键风险与 Plan D

**风险清单**：

1. **Nebula 排队 / 算力不足**：Phase 1 若无法一夜跑完，优先跑 μ ∈ {1.0, 10.0} 两端 + μ=0.1 既有结果做粗三点 sweep，Phase 2/3 相应压缩。
2. **Phase 1 全发散（μ=3 即 NaN）**：跳过 μ=10，直接进 Phase 3.1 curriculum，避免在不稳定训练上浪费预算。
3. **Per-chapter HVR 极度不均**：某 chapter 零 prevalence 导致 HVR 计算无意义——用 未来工作文档 §A.5 的 `None` 语义处理，不污染 overall HVR。
4. **Loss variant 代码 bug**：强制用 `HierarchicalLossMeanAggregated` 在 μ=0 下的 loss 值与 default 相等来回归测试（两者在 μ=0 时应返回数值相同的 total loss）。
5. **最终所有 tier 都 miss**（e.g. Phase 3.3 也没压住 HVR）：走 Plan D。

### Plan D — 最小可辩护的 Innovation 2

即使整套 rescue 完全失败，Chapter 5 **仍可以成立**为独立创新的最小集合：

- **D1. HVR 评测框架本身**。本文首次在 metabolomic disease prediction 任务上提出并部署 head-based + mean-aggregated 双口径 HVR 指标；框架独立于任何具体 loss 是否奏效，是可被其他研究复用的评测工具。这一项**单独就足以支撑一节**，可放 Chapter 5 §5.X "Hierarchy Violation Rate: A Dual-View Metric for Label Consistency"。
- **D2. 公平评测协议**。对应 fair eval 部分（16 disease × uniform average），解决 E0 原实现的 per-disease averaging artifact；与 HVR 并列为"评测贡献"。
- **D3. Dual-head hierarchical architecture**。即使 hierarchy loss 不 work，dual-head 本身提供了可解释的 chapter-level probability（HVR_head 数字本身就是该 head 在训练后的状态证明），可作为 "method contribution independent of loss效果" 写入。
- **D4. Systematic ablation across three intervention families**（Phase 1+2+3 实验矩阵）。负面结果的系统性本身是贡献：未来研究者可以直接引用本文的 "μ sweep + target variant + structural coupling 三类都不 work" 作为 baseline，不必重复这一工作。

**最终兜底 narrative**："本文将 Innovation 2 重新定位为**评测与诊断贡献**（HVR 框架 + fair eval + dual-head architecture + 三类干预的系统消融），并明确指出 soft hierarchy penalty 在 metabolomic 小样本场景下的失效模式，为后续 KL-divergence / hyperbolic 重构提供起点。" 此陈述可在 §6 版本 C 基础上进一步收缩使用。

---

**执行入口**：Phase 1 开跑命令见未来工作文档 §A.2 `e1_mu_sweep` task。本文档结束。
