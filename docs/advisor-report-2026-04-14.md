# 毕设进展汇报：基于 Transformer 的多疾病风险预测方法研究

> 关键词：MetaboLM、UK Biobank 代谢组、多任务学习、层次化分类、参数高效微调
> 报告日期：2026-04-14

## 1. Introduction

本毕设以**代谢组学预训练模型 MetaboLM 为基础，研究面向多疾病风险预测的后训练（post-training）改进方法**。任务设定为：

- **数据**：UK Biobank 队列中 83,744 名相对健康参与者的 168 种 NMR 代谢物测量值
- **预测目标**：16 种 ICD-10 慢性疾病（含代谢、心血管、呼吸、神经退行性、自身免疫、癌症六大类）的风险分层
- **评测指标**：Per-disease AUROC、Mean AUROC、Hierarchy Violation Rate（本工作提出的层次一致性指标）、参数效率

本工作不重新设计 backbone，而是聚焦在**预训练 backbone 之上的后训练设计空间**，从评测方法、架构、损失函数、参数效率四个维度系统改进，并量化每项改进的贡献。

---

## 2. Related Work：MetaboLM

### 2.1 模型设计

MetaboLM (Hu et al., *Nature*, 2024) 是首个面向 NMR 代谢组学的 BERT 风格预训练模型，~85M 参数：

- **输入嵌入**：每个代谢物 m 的 embedding 为 `e_m = x_m · W_m + b_m`（Hadamard product）；168 个代谢物 token + [CLS]，无位置编码（代谢物无内在顺序）
- **生物归纳偏置**：将 metabolite-metabolite correlation matrix（来自 STITCH 化学-化学链接）注入 self-attention 的 bias 项，使注意力偏向真实代谢网络
- **预训练任务**：10% masked metabolite reconstruction，MSE loss，200 epochs

### 2.2 下游评测设定（论文原方案）

- 对 16 种疾病分别构建 1:1 balanced cohort（疾病:健康 = 1:1），每个疾病独立训练一个二分类头（768 → 1）
- 使用 `pooler_output`（AdaptiveAvgPool1d 输出，**非 [CLS]**）作为分类输入
- 论文报告：13/16 种疾病 AUROC > 0.70；对比 MLP 全胜 (16/16)，对比 FT-Transformer 14/16，**对比 CatBoost 未取得统计显著优势**

### 2.3 紧贴本工作叙事的局限性

原方案存在三个值得改进的设计选择，正是本工作的切入点：

1. **每个疾病独立训练 16 个模型**——忽视 ICD-10 编码体系中的层次结构（章节-leaf 关系），稀有疾病无法借助同章节高频疾病的共享信号
2. **Per-disease balanced subset 评测**——每个疾病评测集不同，**无法支持跨疾病、跨模型的 apples-to-apples 比较**
3. **未探讨参数效率**——85M 参数全量微调是否必须？代谢组学场景下 PEFT（Adapter / LoRA）尚无系统研究

---

## 3. Motivation

### 3.1 多合一（One-for-All）

**问题**：原方案训练并维护 16 个独立模型，存在三方面浪费：

- **算力**：训练 16 次，部署 16 个权重
- **数据信号**：稀有疾病（如 parkinsons、各类癌症）阳性样本少，独立训练易过拟合
- **生物学相关性被忽略**：代谢综合征核心节点疾病（T2D、肥胖、高血压）共享代谢通路，独立训练让模型学不到共享信号

**直接证据**：我们做 E0 重现时构建了 **16×16 cross-disease transfer matrix**——用疾病 i 的 E0 模型预测疾病 j。结果：

| 指标 | 数值 |
|:---|:---:|
| Diagonal Mean（自身疾病）| 0.671 |
| Off-Diagonal Mean（其他疾病）| 0.596 |
| Gap | **0.075** |
| 最强跨疾病预测 | **hypertension → T2D = 0.820**（高于 diagonal 均值）|

**Off-diagonal 与 diagonal 仅相差 0.075**，说明**为疾病 A 训练的模型已经捕捉了疾病 B 大部分风险信号**——代谢组学表征本质上反映**系统性代谢状态**，而非疾病特异 biomarker。这从经验上证明了"多合一"路线的可行性与必要性。

### 3.2 利用层次关系

**问题**：ICD-10 编码体系天然有 chapter（如 E：内分泌-代谢；C：肿瘤；I：循环系统）→ leaf 疾病的层次结构。

- **稀有疾病可借同章节高频疾病的梯度做隐式正则**——例：parkinsons 阳性样本少（千级），但同章节神经退行性疾病训练时学到的代谢-神经轴特征可以共享
- **逻辑一致性**：合理预测应满足 P(leaf) ≤ P(parent chapter)（出现某具体疾病前提是该章节系统失稳），但**原论文从未量化这种一致性**

→ 引出本工作的层次化多任务架构 + 层次一致性损失 + HVR 评价指标设计。

---

## 4. Method

### 4.1 层次化 SFT（Innovation 1 + 2）

#### 4.1.1 架构：Dual-Head with Shared Projection

```
Backbone (MetaboLM, 85M)
   ↓
pooler_output (768-d)
   ↓
shared_projection (768 → 256, ReLU, Dropout)
   ↓             ↘
leaf_head      chapter_head
(256 → 16)     (256 → 6)
```

- **共享 256 维投影层**：强制 leaf 与 chapter 任务共用代谢物→疾病-章节表征空间，让稀有疾病可从同章节高频疾病获得梯度
- **6 个 ICD-10 章节**：从 16 种 leaf 疾病的 ICD-10 编码自动归类（E、I、C、J、F/G、M）
- **Chapter labels 自动推导**：`chapter_label = OR(leaf labels in chapter)`

#### 4.1.2 联合损失

$$
\mathcal{L} = \mathcal{L}_{\text{BCE-leaf}} + \lambda \cdot \mathcal{L}_{\text{BCE-chap}} + \mu \cdot \mathcal{L}_{\text{hierarchy}}
$$

其中：

- $\mathcal{L}_{\text{hierarchy}} = \frac{1}{N \cdot K} \sum_{i,k} \max(0, \sigma(z^{\text{leaf}}_{i,k}) - \sigma(z^{\text{chap}}_{i,\text{parent}(k)}))$
- $\lambda = 1.0$（chapter BCE 权重），$\mu = 0.1$（层次一致性权重，初始配置；后续做 sweep）
- 软约束设计：允许局部违反但提供梯度信号引导输出走向层次一致

#### 4.1.3 Hierarchy Violation Rate (HVR) 评价指标

为量化层次一致性，本工作提出 HVR 指标，定义为：

$$
\text{HVR} = \frac{1}{N \cdot K} \sum_{i,k} \mathbb{1}[\sigma(z^{\text{leaf}}_{i,k}) > \sigma(z^{\text{chap}}_{i,\text{parent}(k)})]
$$

考虑跨模型比较需求，设计两个变体：

| 变体 | P(chapter) 来源 | 适用场景 |
|:---|:---|:---|
| **HVR (mean)** | mean-aggregated leaf probs | **跨模型可比**（E0 也能算）|
| **HVR (head)** | chapter head 输出 | 反映 loss 的显式训练目标 |

这是首个为代谢组学层次分类任务提供的可量化一致性度量。

### 4.2 Efficiency 探索（Innovation 3）

在同一层次化多任务框架下系统比较四种参数策略，回答"代谢组学预训练 BERT 上 PEFT 是否有效"：

| 策略 | 实现要点 | 可训参数 | 占比 |
|:---|:---|:---:|:---:|
| **Full FT (E1)** | 所有 backbone + head 参数均训练 | 85.5M | 100% |
| **Head-only (E2)** | 冻结 backbone，仅训分类头 | 203K | 0.24% |
| **Adapter (E3)** | 每个 Transformer 层后插入 64-bottleneck Adapter | 1.4M | 1.61% |
| **LoRA (E4)** | 在 attention Q / V 投影上加 rank-8 LoRA | 497K | 0.58% |

均自实现，详见 `src/model/adapters.py` 与 `src/model/wrapper.py`。

---

## 5. Experiment & Results

### 5.1 实验设定

- **数据**：UK Biobank，84,611 样本（80/20 split），168 NMR 代谢物，16 慢病
- **训练**：AdamW，lr=2e-5，bs=512，40 epochs，cosine + 10% warmup，BCE + early stopping
- **评测 regime**：
  - **Paper-style balanced**：每疾病独立 1:1 balanced subset，对齐原论文报告
  - **Global val.csv (fair)**：所有模型在同一 84k 全队列上评测，**支持跨模型 apples-to-apples 比较**
- **基础设施**：阿里云 4090 集群（Nebula 平台），完整 leaderboard 自动化

### 5.2 E0 基线重现

| Eval Regime | Mean AUROC |
|:---|:---:|
| Paper-style balanced subset | **0.698**（与论文报告 ~0.7 一致，验证 pipeline）|
| Global val — shared preprocess | 0.671 |
| Global val — per-disease cohort z-score | 0.675 |

**16×16 Cross-Disease Transfer Matrix**（E0 ckpt 跨疾病预测）：

- Diagonal mean = 0.671（自身疾病预测）
- Off-Diagonal mean = 0.596（迁移到其他疾病）
- Gap = 0.075
- **Strongest off-diagonal: hypertension → T2D = 0.820**（高于 diagonal 均值）

→ **关键发现**：代谢组学表征跨疾病迁移性强，反映系统性代谢状态而非单一病变 biomarker。

### 5.3 E1-E4 主表（Global val.csv 同口径）

| ID | 实验 | Mean AUROC | HVR (mean) | HVR (head) | 可训参数 |
|:---:|:---|:---:|:---:|:---:|:---:|
| E0 (fair) | 16 独立 per-disease | 0.671 | 0.495 | — | — |
| **E1** | 多任务+hierarchy loss (Full FT) | **0.680 (+0.009)** | 0.490 | 0.364 | 85.5M (100%) |
| E2 | 多任务+hierarchy loss (Head-only) | 0.607 | 0.484 | 0.323 | 203K (0.24%) |
| E3 | 多任务+hierarchy loss (Adapter) | 0.644 | 0.488 | 0.387 | 1.4M (1.61%) |
| E4 | 多任务+hierarchy loss (LoRA) | 0.631 | 0.484 | 0.368 | 497K (0.58%) |

### 5.4 Per-Disease 增益分析（E0 fair vs E1，验证 Innovation 1+2 设计假设）

| 类别 | 疾病 | E0 fair → E1 | Δ |
|:---|:---|:---:|:---:|
| **稀有疾病一致受益** ⭐ | parkinsons | 0.556 → **0.611** | **+0.055** |
| | breast_cancer | 0.677 → 0.702 | +0.025 |
| | colon_cancer | 0.574 → 0.596 | +0.022 |
| | lung_cancer | 0.651 → 0.671 | +0.020 |
| | rheumatoid | 0.670 → 0.679 | +0.009 |
| | dementia | 0.639 → 0.648 | +0.009 |
| **共病集群核心受益** | T2D | 0.829 → 0.846 | +0.017 |
| | stroke | 0.604 → 0.623 | +0.019 |
| | heart_failure | 0.701 → 0.710 | +0.009 |
| **常见疾病轻微让步** | obesity | 0.723 → 0.712 | -0.011 |
| | hypertension | 0.695 → 0.683 | -0.012 |
| | copd | 0.735 → 0.718 | -0.017 |
| | asthma | 0.580 → 0.576 | -0.004 |

**E1 在 16 种疾病中 10 种取得提升**，**稀有疾病聚类全面受益**。其中 **parkinsons +0.055** 从接近随机（0.556）跃升至临床可用（0.611），**癌症家族（C 章节）一致 +0.020~+0.025** 表明同章节共享投影确实在传递有用的代谢特征。

常见疾病（obesity / hypertension / copd / asthma）的轻微让步是预期 tradeoff——E0 已有充分阳性样本，共享表征做让步以服务整体 Pareto，整体 mean AUROC 净收益仍 +0.009。

→ **直接验证 Innovation 1（多任务）+ Innovation 2（层次化）的核心设计假设：稀有疾病通过同章节高频疾病的共享梯度获得隐式正则**。

### 5.5 PEFT 系统比较（Innovation 3）

| 策略 | Mean AUROC | vs Full FT | 可训参数 |
|:---|:---:|:---:|:---:|
| Full FT (E1) | **0.680** | 100% | 85.5M |
| Adapter (E3) | 0.644 | **94.7%** | 1.4M (**1.6%**) |
| LoRA (E4) | 0.631 | 92.8% | 497K (0.6%) |
| Head-only (E2) | 0.607 | 89.3% | 203K (0.2%) |

**关键发现**：

- 清晰的参数量-性能单调排序，无 reversal
- **Adapter 以 1.6% 参数保持 Full FT 94.7% 性能**——代谢组学领域首个系统报告的 PEFT 基准
- 与 NLP 领域典型"PEFT ≈ Full FT"现象不同，代谢 BERT 上 Full FT 仍有 5-10% 领先，**暗示代谢组学表征对 backbone 全参数微调更敏感**

### 5.6 HVR 量化分析

- E0 baseline HVR (mean) = 0.495 ≈ 随机水平，印证独立训练对层次结构完全无感
- E1-E4 HVR (mean) 均下降至 0.484~0.490，方向与设计预期一致
- HVR (head) 在默认 μ=0.1 下为 0.32~0.39，表明 loss 在显式目标上尚有下降空间
- **HVR 指标本身的诊断价值得到验证**——单靠 AUROC 无法暴露层次不一致性，HVR 让该问题可量化、可改进

后续工作将系统扫描 μ ∈ {0.0, 0.5, 1.0, 3.0, 10.0} 与损失变体（mean-aggregated leaf target、hard projection），刻画 HVR-AUROC 帕累托前沿。

---

## 6. 主要贡献小结

1. **公平评测框架 + 16×16 cross-disease 矩阵**：揭示代谢组学表征跨疾病迁移性强（hypertension→T2D 0.820）
2. **层次化多任务架构 + HVR 指标**：在 10/16 种疾病上取得提升，**稀有疾病增益最显著**（parkinsons +0.055），验证 ICD-10 层次先验的有效性
3. **代谢组学 BERT 首个 PEFT 系统对比**：Adapter 以 1.6% 参数保持 94.7% 性能，给出参数量-性能 Pareto

## 7. 后续计划

- **2026-04-22 前**：完成 hierarchy loss μ sweep，刻画 HVR-AUROC Pareto，为 Innovation 2 提供更强证据
- **2026-04-29 前**：完成 GRPO RL（E5 Calibration / E6 Hierarchy reward）训练，加入论文贡献四
- **2026-05-13**：论文二稿
- **2026-06-02**：答辩

---

**附**：完整实验 leaderboard 与代码见 GitHub `TheChosenOne66/MetaboLM`（branch: `exp`）。
