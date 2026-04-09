# MetaboLM Post-Training

基于 [Qiu-Shizheng/MetaboLM](https://github.com/Qiu-Shizheng/MetaboLM) 的后训练流水线，实现从原始 UK Biobank 数据到 16 种慢性疾病预测的端到端复现与扩展。

## 概述

MetaboLM 是一个在 83,744 名健康参与者的血浆代谢组学数据上预训练的 Transformer（BERT-like）模型，捕捉 168 种 NMR 代谢物之间的交互模式。本项目在预训练模型的基础上构建了完整的下游训练与评估流水线。

**支持的功能：**

- **E0 复现**：逐疾病监督微调（SFT），复现原论文 16 种疾病的 AUROC 结果
- **预训练评估**：Masked metabolite reconstruction，验证预训练 checkpoint 有效性
- **数据流水线**：Raw UK Biobank → 清洗/标注/切分/归一化 → 训练就绪数据集
- **模型架构**：MetaboliteBERTModel（12 层, 8 头, hidden=768, ~85M 参数）+ 分类头

## 项目结构

```
metabolm_posttrain/
├── scripts/
│   ├── train.py                # E0 微调主脚本
│   ├── prepare_data.py         # 数据预处理流水线
│   ├── eval_pretrain.py        # 预训练 checkpoint 评估
│   └── download_weights.sh     # 下载预训练权重
├── configs/
│   ├── reproduce.yaml          # E0 训练配置
│   ├── base.yaml               # 数据预处理配置
│   └── eval_pretrain.yaml      # 预训练评估配置
├── src/
│   ├── config.py               # 配置加载（dataclass + YAML）
│   ├── data/                   # 数据流水线（biomarkers, cohort, preprocessing, dataset）
│   ├── model/                  # 模型（backbone, heads, wrapper）
│   ├── training/               # 训练（SFT trainer, losses, metrics）
│   └── evaluation/             # 评估模块
├── tests/                      # 单元测试
├── data/processed/             # 预处理后的数据（gitignore）
├── weights/                    # 模型权重（gitignore）
├── outputs/                    # 训练输出（gitignore）
├── docs/                       # 运行指南、环境说明、进度追踪
└── _reference*/                # 原始论文参考代码
```

## 快速开始

### 环境准备

```bash
conda activate metabolm
cd /path/to/metabolm_posttrain
pip install -r requirements.txt
```

依赖：torch>=2.4.0, numpy>=1.26, pandas>=2.2, scikit-learn>=1.4, scipy>=1.13, pyyaml>=6.0, tqdm

### 1. 下载预训练权重

```bash
bash scripts/download_weights.sh
# -> weights/best_metabolite_bert_model.pt (326 MB)
```

权重来源：[Figshare](https://figshare.com/s/bd69f74785946802b585)

### 2. 数据预处理

```bash
python scripts/prepare_data.py --config configs/base.yaml
```

从原始 UK Biobank CSV 生成训练数据：
- 提取 168 种 NMR 代谢物（instance 0）
- 解析 ICD-10 诊断码，构建 16 种疾病标签（incident/prevalent/healthy）
- 80/20 分层切分，中位数插补，Z-score 归一化（仅在训练集上拟合，防止数据泄露）

输出至 `data/processed/`：train.csv, val.csv, correlation_matrix.pt, normalization_stats.csv

### 3. E0 微调训练

```bash
# 训练全部 16 种疾病（串行，约 15-20 小时）
python -u scripts/train.py --config configs/reproduce.yaml \
  > outputs/E0_train.log 2>&1 &

# 仅训练指定疾病
python scripts/train.py --config configs/reproduce.yaml \
  --diseases T2D obesity hypertension
```

每种疾病独立训练：1:1 平衡采样 → Z-score 归一化 → AdamW(lr=2e-5) + cosine scheduler → 40 epochs → 按 val AUROC 选最优模型。

训练结果输出至 `outputs/E0_reproduction/`：
```
E0_reproduction/
├── finetune_summary_metrics.csv        # 16 种疾病 AUROC 汇总表
├── T2D/
│   ├── best_finetune_model_T2D.pt      # 最优模型
│   ├── finetune_metrics_T2D.csv        # 逐 epoch 指标
│   └── finetune_predictions_val_T2D.csv
└── ...
```

### 4. 预训练评估

```bash
python scripts/eval_pretrain.py \
  --metabolomics /path/to/metabolomics.csv \
  --diagnosis /path/to/diagnosis_icd10.csv \
  --cause-of-death /path/to/cause_of_death.csv \
  --checkpoint weights/best_metabolite_bert_model.pt \
  --output-dir outputs/pretrain_eval \
  --mask-rate 0.10 --num-passes 5
```

通过 masked metabolite reconstruction 验证预训练模型：随机遮盖 10% 代谢物，评估重建的 MSE / MAE / R² / Accuracy。

## 支持的 16 种疾病

| 疾病 | ICD-10 | 疾病 | ICD-10 |
|------|--------|------|--------|
| 2 型糖尿病 (T2D) | E11 | 房颤 | I48 |
| 肥胖 | E66 | 心衰 | I50 |
| 高血压 | I10 | 类风湿关节炎 | M05/M06 |
| 缺血性心脏病 | I25 | 哮喘 | J45 |
| COPD | J44 | 痴呆 | F00-F03/G30 |
| 帕金森 | G20 | 脑卒中 | I60-I64 |
| 乳腺癌 | C50 | 结肠癌 | C18 |
| 肺癌 | C34 | 前列腺癌 | C61 |

## 训练超参数（E0）

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Batch size | 512 |
| Epochs | 40 |
| Warmup | 10% linear |
| Scheduler | Cosine |
| Loss | BCEWithLogitsLoss |
| 模型选择 | Best validation AUROC |

## 运行测试

```bash
python -m pytest tests/
```

覆盖模型架构（前向传播 shape、参数量）、代谢物字段映射、疾病端点定义。

## 参考

- 原始论文权重：[Figshare](https://figshare.com/s/bd69f74785946802b585)
- 微调模型权重：[10.6084/m9.figshare.30744284](https://doi.org/10.6084/m9.figshare.30744284)
- DOI：https://doi.org/10.5281/zenodo.17083417
- 上游仓库：[Qiu-Shizheng/MetaboLM](https://github.com/Qiu-Shizheng/MetaboLM)
