# MetaboLM 运行指南（E0 复现 + E1-E4 SFT 创新）

## 迁移到新机器

### 目录结构

`metabolm_posttrain/` 是**完全自包含**的，所有训练需要的代码、数据、权重都在目录内。所有 config 使用相对路径。

```
metabolm_posttrain/           # 总计 ~2.3 GB，整个目录打包迁移即可
├── scripts/                  # 入口脚本
│   ├── train.py              # E0 训练主脚本（per-disease 独立训练）
│   ├── train_multitask.py    # E1-E4 训练主脚本（multi-task 联合训练）  ← NEW
│   ├── prepare_data.py       # 数据预处理（已跑完，新机器不需要重跑）
│   └── eval_pretrain.py      # pretrain checkpoint 评估
├── configs/
│   ├── reproduce.yaml        # E0: 官方复现（per-disease 1:1 平衡）
│   ├── sft_full_ft.yaml      # E1: 层次化多任务 + 全量微调             ← NEW
│   ├── sft_head_only.yaml    # E2: 层次化多任务 + 仅训练 head           ← NEW
│   ├── sft_adapter.yaml      # E3: 层次化多任务 + Adapter              ← NEW
│   ├── sft_lora.yaml         # E4: 层次化多任务 + LoRA                 ← NEW
│   └── base.yaml             # 数据预处理配置（引用原始 UKB 数据路径）
├── data/processed/           # 预处理好的数据（1.4 GB）✅ 已就绪
│   ├── train.csv             # 338,442 人 × 185 列 (eid + 168 metabolites + 16 labels)
│   ├── val.csv               # 84,611 人
│   ├── correlation_matrix.pt # 169×169 代谢物相关矩阵 tensor
│   └── normalization_stats.csv
├── weights/                  # 模型权重（652 MB）
│   ├── best_metabolite_bert_model.pt      # pretrained backbone (326MB, 85M params) ✅ 必需
│   └── best_finetune_model_diabetes.pt    # 官方 T2D fine-tuned（参考用，非必需）
├── src/                      # Python 模块代码
│   ├── config.py
│   ├── data/                 # biomarkers, endpoints, cohort, preprocessing, dataset
│   ├── model/                # backbone, heads, adapters, wrapper
│   └── training/             # losses, metrics, sft_trainer
├── tests/                    # 单元测试（43 个，全部通过）
├── _reference/               # 官方原始代码（参考用，非必需）
└── docs/                     # 文档
```

### 打包迁移

```bash
# 方法一：只打包训练必需文件（~2 GB，E0 + E1-E4 全部可跑）
cd /SPXvePFS/users/jytang
tar czf metabolm_posttrain_train.tar.gz \
    metabolm_posttrain/scripts/ \
    metabolm_posttrain/configs/ \
    metabolm_posttrain/src/ \
    metabolm_posttrain/tests/ \
    metabolm_posttrain/data/processed/ \
    metabolm_posttrain/weights/best_metabolite_bert_model.pt \
    metabolm_posttrain/requirements.txt \
    metabolm_posttrain/docs/

# 方法二：整个目录（~2.3 GB，含参考代码）
tar czf metabolm_posttrain_full.tar.gz metabolm_posttrain/
```

### 新机器环境搭建

```bash
# 解压
tar xzf metabolm_posttrain_train.tar.gz
cd metabolm_posttrain

# 创建 conda 环境
conda create -n metabolm python=3.10 -y
conda activate metabolm
pip install -r requirements.txt

# 验证环境
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

依赖列表（`requirements.txt`）：
```
torch>=2.0
transformers>=4.30
scikit-learn>=1.2
pandas>=2.0
numpy>=1.24
scipy>=1.10
PyYAML>=6.0
tqdm>=4.60
```

---

## 运行步骤

### Step 1: 数据准备（✅ 已完成，新机器不需要重跑）

预处理好的数据在 `data/processed/` 下，随目录一起迁移。

如需重跑（比如换了原始数据），需先修改 `configs/base.yaml` 中的原始数据绝对路径，然后：

```bash
python -u scripts/prepare_data.py --config configs/base.yaml
```

### Step 2: E0 Fine-tuning 训练

#### 训练全部 16 个疾病（完整 E0 复现）

```bash
cd /path/to/metabolm_posttrain

nohup python -u scripts/train.py --config configs/reproduce.yaml \
    > outputs/E0_train.log 2>&1 &

# 查看实时日志
tail -f outputs/E0_train.log

# 查看训练进度（哪些疾病完成了）
grep "Best Val AUC" outputs/E0_train.log
```

#### 只训练指定疾病

```bash
python -u scripts/train.py --config configs/reproduce.yaml \
    --diseases T2D obesity hypertension
```

16 个疾病名称：
```
T2D  obesity  hypertension  ischemic_heart  atrial_fib  heart_failure
rheumatoid  asthma  dementia  copd  stroke  parkinsons
breast_cancer  colon_cancer  lung_cancer  prostate_cancer
```

### Step 3: 查看结果

```bash
# AUC 汇总表（训练完成后生成）
cat outputs/E0_reproduction/finetune_summary_metrics.csv

# 单个疾病的详细指标
cat outputs/E0_reproduction/T2D/finetune_metrics_T2D.csv

# 验证集预测
head outputs/E0_reproduction/T2D/finetune_predictions_val_T2D.csv
```

---

## 训练细节

### 每个疾病的训练流程

1. 从 `data/processed/train.csv` + `val.csv` 加载全量数据（423,053 人）
2. 按疾病构建 1:1 平衡队列（incident 病例 : 16 标签全阴性的健康对照）
3. 80/20 stratified split
4. Per-disease z-score 归一化（fit on train split）
5. 加载 pretrained MetaboliteBERTModel (85M params) + Linear(768,1) 分类头
6. 训练 40 epochs，按 val AUC 保存最优模型

### 超参数（匹配论文 `MetaboLM_Fine-tuning.py`）

| 参数 | 值 | 来源 |
|------|-----|------|
| batch_size | 512 | 官方代码 line 676 |
| learning_rate | 2e-5 | 官方代码 line 696 |
| epochs | 40 | 官方代码 line 697 |
| optimizer | AdamW | 官方代码 line 709 |
| scheduler | cosine + 10% warmup | 官方代码 line 711 |
| loss | BCEWithLogitsLoss | 官方代码 line 708 |
| model selection | best val AUC | 官方代码 line 803 |
| data balance | 1:1 disease:healthy | 官方代码 line 628 |

### 时间估算（基于 H20 GPU 实测）

| 疾病 | 样本量(病例) | 每 epoch 耗时 | 40 epochs 估计 |
|------|-------------|-------------|--------------|
| hypertension | ~87K | ~4 min | ~160 min |
| T2D / obesity / asthma / ischemic_heart | ~20-27K | ~2.5 min | ~100 min |
| 其余小样本疾病 | 2-13K | ~1-2 min | ~40-80 min |
| **全部 16 个疾病串行** | - | - | **约 15-20 小时** |

### 输出文件结构

```
outputs/E0_reproduction/
├── finetune_summary_metrics.csv              # 16 疾病 AUC 汇总表（核心交付物）
├── T2D/
│   ├── best_finetune_model_T2D.pt            # 最优模型权重
│   ├── finetune_metrics_T2D.csv              # AUC, Acc, Precision, Recall, F1
│   └── finetune_predictions_val_T2D.csv      # eid, True_Label, Predicted_Probability
├── obesity/
│   └── ...
├── ...（共 16 个子目录）
```

---

## 已验证的结果

T2D 训练跑了 9 个 epoch 后因 GPU 占满被中断，AUC 趋势正常：

| Epoch | Val AUC |
|-------|---------|
| 1 | 0.7828 |
| 3 | 0.8076 |
| 5 | 0.8423 |
| 7 | 0.8513 |
| 9 | 0.8566 |

论文报告 T2D AUC ~0.85-0.87，我们 epoch 9 已达到 0.8566 且仍在上升，**说明代码和数据流水线正确**。

---

## 常见问题

### Q: GPU 显存不够怎么办？
- 模型本身只需 ~2-3 GB 显存（85M 参数 + batch_size=512 的中间激活）
- 如果显存紧张，修改 `configs/reproduce.yaml` 中 `batch_size: 256` 或 `128`
- 不影响最终精度，只影响训练速度

### Q: 如何从中断恢复？
当前不支持 epoch 级断点续训。但已完成的疾病结果保存在各自子目录，可用 `--diseases` 只跑剩余的：
```bash
# 假设 T2D, obesity 已完成，只跑剩下的
python -u scripts/train.py --config configs/reproduce.yaml \
    --diseases hypertension ischemic_heart atrial_fib heart_failure \
    rheumatoid asthma dementia copd stroke parkinsons \
    breast_cancer colon_cancer lung_cancer prostate_cancer
```

### Q: 数据加载太慢？
首次加载 1.4G CSV 约需 10-15 秒。加载后所有疾病共享同一份数据，不会重复读取。

### Q: `configs/base.yaml` 中的绝对路径报错？
`base.yaml` 中的路径指向原始 UKB 数据（`storage_tmp/`），**只在重新预处理时需要**。E0 训练只用 `reproduce.yaml`，不需要原始数据。

---

## E1-E4: 层次化多任务 SFT 创新实验

### 与 E0 的关键区别

| | E0 (复现) | E1-E4 (创新) |
|---|---|---|
| **训练脚本** | `scripts/train.py` | `scripts/train_multitask.py` |
| **模型数量** | 16 个独立模型 | 1 个联合模型 |
| **训练集** | per-disease 1:1 平衡子集 | 全量 train.csv (338K) |
| **分类头** | `SingleTaskHead` Linear(768,1) | `HierarchicalMultiTaskHead` 共享投影 → leaf(16) + chapter(6) |
| **类别平衡** | 物理采样 | `pos_weight = N_neg / N_pos` per disease |
| **Loss** | BCEWithLogitsLoss | `L_leaf + λ·L_chapter + μ·L_hierarchy` |
| **归一化** | per-disease 子集重算 z-score | 使用 prepare_data 已算好的全局 z-score |
| **模型选择** | 单疾病 val AUC | 16 疾病 mean val AUC |

### 实验矩阵

| 实验 | 配置文件 | Freeze 策略 | 可训练参数 | 占比 |
|------|---------|------------|-----------|------|
| E1 | `sft_full_ft.yaml` | none（全量微调） | ~85M | 100% |
| E2 | `sft_head_only.yaml` | head_only（冻结 backbone） | ~203K | 0.24% |
| E3 | `sft_adapter.yaml` | adapter（冻结 + 瓶颈适配器） | ~1.4M | 1.61% |
| E4 | `sft_lora.yaml` | lora（冻结 + Q/V 低秩分解） | ~500K | 0.58% |

### 新增模块说明

| 文件 | 说明 |
|------|------|
| `src/model/heads.py` | 新增 `HierarchicalMultiTaskHead`: 共享投影(768→256→ReLU→Dropout) → leaf_head(256→16) + chapter_head(256→6) |
| `src/model/adapters.py` | `AdapterLayer`: 瓶颈 768→64→768 + 残差 + GELU，插入每层 FFN 之后 |
| | `LoRALinear`: 包裹 nn.Linear，冻结原始权重，添加 A(768×8)·B(8×768) 低秩矩阵 |
| `src/model/wrapper.py` | 支持 4 种冻结策略 (none/head_only/adapter/lora)，自动注入 adapter 或 LoRA |
| `src/training/losses.py` | `HierarchicalLoss`: leaf BCE + chapter BCE + hierarchy violation penalty |
| `src/training/metrics.py` | `compute_multitask_metrics`: 16 疾病 per-disease + mean AUC |
| | `compute_hierarchy_violation_rate`: P(leaf) > P(chapter) 违规率 |
| `src/training/sft_trainer.py` | `MultiTaskSFTTrainer`: 联合训练，按 mean AUC 选最优模型 |
| `src/data/dataset.py` | `MultiTaskDataset`: 自动从 leaf labels 推导 chapter labels (OR) |

### 运行 E1-E4

```bash
cd /path/to/metabolm_posttrain

# E1: 全量微调（最慢，显存需求最高）
nohup python -u scripts/train_multitask.py --config configs/sft_full_ft.yaml \
    > outputs/E1_train.log 2>&1 &

# E2: 仅训练 head（最快，lower-bound 参考）
nohup python -u scripts/train_multitask.py --config configs/sft_head_only.yaml \
    > outputs/E2_train.log 2>&1 &

# E3: Adapter（推荐先跑这个）
nohup python -u scripts/train_multitask.py --config configs/sft_adapter.yaml \
    > outputs/E3_train.log 2>&1 &

# E4: LoRA
nohup python -u scripts/train_multitask.py --config configs/sft_lora.yaml \
    > outputs/E4_train.log 2>&1 &

# 查看训练进度
grep "Epoch\|Best" outputs/E3_train.log
```

**注意**：E1 (全量微调) 显存需求较高（~85M 参数全部可训练），如果 GPU 显存紧张，建议先跑 E3 (adapter) 或 E4 (lora)，它们冻结了 backbone，显存需求小很多。

### 输出文件结构

```
outputs/E3_sft_adapter/
├── best_model.pt              # 最优模型权重（整个模型的 state_dict）
├── multitask_metrics.csv      # 16 疾病 + MEAN 的 AUC / F1 表
└── summary.json               # 实验配置、最优 epoch、参数量等元信息
```

### 查看 E1-E4 结果

```bash
# 各实验的 AUC 表
cat outputs/E1_sft_full_ft/multitask_metrics.csv
cat outputs/E2_sft_head_only/multitask_metrics.csv
cat outputs/E3_sft_adapter/multitask_metrics.csv
cat outputs/E4_sft_lora/multitask_metrics.csv

# 各实验的汇总信息
cat outputs/E3_sft_adapter/summary.json
```

### Hierarchy Loss 超参数

默认值在 config 的 `training` 部分：

```yaml
training:
  lambda_chapter: 1.0   # chapter BCE 权重
  mu_hierarchy: 0.1     # hierarchy violation penalty 权重
```

消融实验可以通过修改这两个值来做：
- `mu_hierarchy: 0` → 无 hierarchy 约束
- `lambda_chapter: 0` → 无 chapter loss（只有 leaf）

### 运行单元测试

```bash
cd /path/to/metabolm_posttrain
python -m pytest tests/ -v
# 或逐个运行：
python tests/test_backbone.py
python tests/test_heads.py
python tests/test_adapters.py   # 包含 wrapper 集成测试（共 14 个）
python tests/test_biomarkers.py
python tests/test_endpoints.py
```

共 43 个测试，全部通过。
