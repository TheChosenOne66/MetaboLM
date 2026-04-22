# Chapter 4 实验数据素材索引

本目录包含论文第4章（实验与分析）所需的全部数据表格，按章节组织。

## 文件清单

| 文件 | 对应论文章节 | 内容 |
|:---|:---|:---|
| `4.1_dataset.md` | 4.1 数据集与实现 | 数据集统计、验证集分布、模型配置 |
| `4.2_e0_reproduction.md` | 4.2 E0 重现与公平评测 | 三种评测口径对比、E0 vs 原论文逐疾病AUROC |
| `4.3_cross_disease_transfer.md` | 4.3 Cross-Disease 迁移分析 | 16×16矩阵汇总、逐疾病三种口径完整表 |
| `4.4_multitask_hierarchy.md` | 4.4 多任务 + Hierarchy Loss 结果 | E0 vs E1 公平对比、关键发现总结 |
| `4.5_hvr_analysis.md` | 4.5 HVR 量化分析 | E0-E4 HVR对比、μ Sweep综合、Per-Chapter HVR、hF1全面对比 |
| `4.6_peft_comparison.md` | 4.6 PEFT 系统比较 | 参数量-性能帕累托表、逐疾病PEFT梯度、模块实现细节 |
| `4.7_grpo_rl.md` | 4.7 GRPO RL 结果 | 实现状态、细节、预期设计 |

## 数据来源

- E0-E4 AUROC / HVR: `LEADERBOARD.md`, `outputs/E{0-4}_*/`, `scripts/eval_e0_global_cohort.py`
- μ Sweep: `outputs/E1_mu_sweep/summary.csv`
- hF1: `outputs/E1_mu_sweep/hf1_comparison.json`, `outputs/hf1_all_experiments.json`
- Cross-disease matrix: `outputs/E0_reproduction/cross_disease_matrix.csv`

## 使用说明

所有表格均为 Markdown 格式，可直接复制到 LaTeX 或 Word 中。
数值已四舍五入到合适精度，原始精确值见对应 JSON/CSV 源文件。
