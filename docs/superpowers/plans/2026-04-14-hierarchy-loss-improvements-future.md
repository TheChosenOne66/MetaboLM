# Hierarchy Loss 改进方向（future work）

> **背景**: 2026-04-14 拿到 E1-E4 完整 HVR 数字后发现，默认 `μ_hierarchy=0.1` 下 hierarchy loss **未能在其显式训练目标上收敛**（HVR_head ≈ 0.32-0.39，远离 0），更没传播到 leaf head 输出（HVR_mean E0→E1 仅 -0.005）。这份文档列出**论文初稿提交后**的迭代方向。

## 数据现状（2026-04-14）

| 实验 | Mean AUROC (fair) | HVR (mean) | HVR (head) |
|---|:---:|:---:|:---:|
| E0 | 0.671 | 0.495 | — |
| E1 (μ=0.1) | 0.680 | 0.490 | 0.364 |
| E2 (μ=0.1) | 0.607 | 0.484 | 0.323 |
| E3 (μ=0.1) | 0.644 | 0.488 | 0.387 |
| E4 (μ=0.1) | 0.631 | 0.484 | 0.368 |

诊断：BCE (weight 1.0) 推 leaf prob 上去的梯度远大于 hierarchy loss (μ=0.1) 的反向梯度。Loss 项在场，但量级上不足以约束。

## 改进方向（按优先级）

### 1. μ sweep（最直接，必做）

在 E1 上跑 `μ ∈ {0.0, 0.1, 0.5, 1.0, 3.0, 10.0}`，画 (μ, HVR_head, HVR_mean, AUROC) 三轴图。

**预期产出**：
- 找到 μ* 使 HVR_head 接近 0 且 AUROC 退化 < 0.01 → **Innovation 2 救活**，论文 chapter 5 改写成"hyperparameter-sensitive 但有效"
- 或者证明 hierarchy loss 与 BCE 存在 fundamental tradeoff → Pareto curve 也是发表级发现

**实现**：
```bash
# configs/sft_full_ft_mu_{0.0,0.5,1.0,3.0,10.0}.yaml
# 复制 sft_full_ft.yaml，改一行 mu_hierarchy
# 跑 5 个并行 job
```

输出建议放到 `outputs/E1_mu_sweep/μ=X/`，然后 leaderboard 加个 mu_sweep 子表。

### 2. Loss target 改到 leaf head 直接输出（高优先级）

当前 `L_hierarchy = mean(max(0, σ(leaf_logit) - σ(chap_logit)))` 比的是两个独立 head 的输出。leaf 和 chap head 都是从 256 维共享 proj 出去的独立 Linear，**它们之间没有结构约束**——chap_logit 可以独立变化来让 loss 看起来下降，而 leaf 输出本身分布完全不受影响。

**改进**：把 chap target 改成从 leaf probs **算出来的** mean-aggregated chap：
```python
mean_chap = mean_aggregate_chapter_probs(leaf_probs)
L_hierarchy = mean(max(0, leaf_probs - mean_chap))
```
这样 loss 直接施压到 leaf head 上，不能通过"调 chap head"作弊。

### 3. Hard constraint 变体（消融对照）

替代 soft penalty，直接在前向过 sigmoid 时投影 leaf 到 chap 上界：
```python
leaf_probs = torch.minimum(σ(leaf_logits), σ(chap_logits))
```
HVR 必为 0。看 AUROC 怎么受影响。这是 hierarchy loss 的**极限版本**，给 μ sweep 一个 endpoint。

### 4. 检查 lambda_chapter（次优先级）

`λ_chapter=1.0`（chapter BCE 权重）也可能影响 chapter head 学到什么。如果 chapter head 自己都没学好，hierarchy loss 比的就是噪声。建议先验证 chapter head 单独的 AUROC。

### 5. 训练 curriculum（实验性）

假设 1：BCE 和 hierarchy loss 同时上，BCE 收敛快、hierarchy loss 来不及生效。
- Phase 1（前 50% epochs）：仅 BCE，让 leaf head 学会预测
- Phase 2（后 50%）：加入 hierarchy loss，让 leaf head 调整以满足层次

## 可发表性评估

| 改进 | 工作量 | 论文价值 |
|---|---|---|
| μ sweep | 5 个 job × 40 epoch | **必须做**，要么救 Innovation 2，要么给 negative result 配上 Pareto |
| Loss target → leaf | 改 5 行代码 + 1 个 job | 高 — 揭示原始 design flaw |
| Hard constraint | 改 3 行代码 + 1 个 job | 中 — μ sweep 的 μ=∞ endpoint |
| λ_chapter sweep | 5 个 job | 低 — diagnostic |
| Curriculum | 1 个 job | 中 — 工程 trick |

总计 ~12 个 E1-scale 训练 job。Nebula 4090 上每个 job ~2-3 小时，全部并行可一晚搞定。

## 与论文初稿的关系

论文初稿（2026-04-15 提交）按当前数据写：
- Innovation 2 重新框架为"提出 HVR 指标 + 量化诊断默认配置失败"，未来工作引用这份 doc
- 答辩前如果上述实验跑完，可用作 supplementary 或 final draft 修订

## 落地 checklist（提交初稿后）

- [ ] 写 5 个 `configs/sft_full_ft_mu_*.yaml`
- [ ] 写 `do.sh` 子命令 `bash do.sh e1_mu_sweep`
- [ ] 在 `scripts/update_leaderboard.py` 加 `mu_sweep` 实验组（manifest 扩展）
- [ ] 跑完后画 (μ, HVR, AUROC) 图
- [ ] 实现 mean-aggregated chap loss 变体（src/training/losses.py 里加新函数）
- [ ] 实现 hard projection 变体
- [ ] 扩展 `compute_hierarchy_violation_rate` 输出 per-chapter 分解
- [ ] 写 supplementary section 报告结果

---

## Appendix A — 具体实现细节

下面是把 checklist 翻成可执行 artifacts 的最小集合。每个子节都给到「直接复制即可跑」的程度。

### A.1 μ sweep 配置模板

基线 `configs/sft_full_ft.yaml` 里只有一行需要变 (`training.mu_hierarchy`)。直接生成 5 个独立 YAML，避免引入额外 config 继承机制（当前 codebase 不支持 `defaults:`）。

`configs/sft_full_ft_mu_0.0.yaml`:
```yaml
# E1 ablation: μ=0.0 (hierarchy loss disabled — pure leaf+chapter BCE)
experiment: E1_sft_full_ft_mu_0.0
seed: 42

data:
  train_path: data/processed/train.csv
  val_path: data/processed/val.csv
  correlation_matrix_path: data/processed/correlation_matrix.pt
  num_metabolites: 168

model:
  pretrained_ckpt: weights/best_metabolite_bert_model.pt
  head_type: hierarchical
  freeze_strategy: none
  hidden_size: 768
  num_diseases: 16
  proj_size: 256

training:
  mode: sft
  batch_size: 512
  learning_rate: 2.0e-5
  num_epochs: 40
  warmup_ratio: 0.1
  weight_decay: 0.01
  lambda_chapter: 1.0
  mu_hierarchy: 0.0     # <-- swept

output_dir: outputs/E1_mu_sweep/mu_0.0
```

剩下四个 (`mu_0.5`, `mu_1.0`, `mu_3.0`, `mu_10.0`) 用同样模板，仅改 `experiment`、`training.mu_hierarchy`、`output_dir` 三行。简洁起见用 diff 形式表示：

```diff
# configs/sft_full_ft_mu_0.5.yaml — diff vs sft_full_ft_mu_0.0.yaml
- experiment: E1_sft_full_ft_mu_0.0
+ experiment: E1_sft_full_ft_mu_0.5
- mu_hierarchy: 0.0
+ mu_hierarchy: 0.5
- output_dir: outputs/E1_mu_sweep/mu_0.0
+ output_dir: outputs/E1_mu_sweep/mu_0.5
```

`mu_1.0` / `mu_3.0` / `mu_10.0` 对应替换数字即可。注意 `mu=0.1` 已经由现有 `outputs/E1_sft_full_ft` 提供，不必重跑——画图时把它当作 sweep 的 (μ=0.1) 数据点 symlink 进 `outputs/E1_mu_sweep/mu_0.1` 即可。

一行 shell 生成 4 个变体（生成完务必 diff 检查）：
```bash
for mu in 0.5 1.0 3.0 10.0; do
  sed -e "s/mu_0\.0/mu_${mu}/g" \
      -e "s/mu_hierarchy: 0\.0/mu_hierarchy: ${mu}/" \
      configs/sft_full_ft_mu_0.0.yaml > configs/sft_full_ft_mu_${mu}.yaml
done
```

### A.2 `do.sh` 扩展

当前仓库**没有** `do.sh`（只有 `scripts/download_weights.sh`），README 直接列 `python scripts/train_multitask.py --config ...`。新建 `do.sh` 作为命令路由器，承载 ablation 编排：

```bash
#!/usr/bin/env bash
# do.sh — task router for MetaboLM ablations.
# Usage: bash do.sh <task_name>
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-}" in

  e1_mu_sweep)
    # 5 个 μ 配置串行跑（单卡安全）；并行跑见 e1_mu_sweep_parallel。
    for mu in 0.0 0.5 1.0 3.0 10.0; do
      echo "=== μ=${mu} ==="
      python scripts/train_multitask.py \
        --config configs/sft_full_ft_mu_${mu}.yaml
      python scripts/eval_hierarchy_violation.py multitask \
        --config configs/sft_full_ft_mu_${mu}.yaml \
        --output-dir outputs/E1_mu_sweep/mu_${mu}
    done
    python scripts/update_leaderboard.py
    ;;

  e1_mu_sweep_parallel)
    # 4090 多卡场景：每张卡跑一个 μ。CUDA_VISIBLE_DEVICES 需调用方设置外层。
    for mu in 0.0 0.5 1.0 3.0 10.0; do
      CUDA_VISIBLE_DEVICES=$((i++ % $(nvidia-smi -L | wc -l))) \
        python scripts/train_multitask.py \
          --config configs/sft_full_ft_mu_${mu}.yaml &
    done
    wait
    ;;

  e1_loss_target_leaf)
    # 用 mean-aggregated leaf 作为 hierarchy loss target 的变体。
    # 由 configs/sft_full_ft_leaf_target.yaml 提供 (training.hierarchy_loss_variant: mean_aggregated)。
    python scripts/train_multitask.py \
      --config configs/sft_full_ft_leaf_target.yaml
    python scripts/eval_hierarchy_violation.py multitask \
      --config configs/sft_full_ft_leaf_target.yaml \
      --output-dir outputs/E1_leaf_target
    ;;

  e1_hard_projection)
    python scripts/train_multitask.py \
      --config configs/sft_full_ft_hard_proj.yaml
    python scripts/eval_hierarchy_violation.py multitask \
      --config configs/sft_full_ft_hard_proj.yaml \
      --output-dir outputs/E1_hard_proj
    ;;

  *)
    echo "Unknown task: ${1:-<empty>}"
    echo "Available: e1_mu_sweep | e1_mu_sweep_parallel | e1_loss_target_leaf | e1_hard_projection"
    exit 2
    ;;
esac
```

注意：`hierarchy_loss_variant` 字段需要在 `src/config.py` 添加（默认 `"soft_chapter_head"`，新增 `"mean_aggregated"`、`"hard_projection"`），并由 `train_multitask.py` 在构造 `HierarchicalLoss` 时分发。

### A.3 `configs/leaderboard_manifest.yaml` 扩展

把 μ sweep 作为独立实验组（而非平铺到主 experiments 列表，避免污染主表），新增 `experiment_groups` 段：

```yaml
# 追加到现有 manifest 末尾（在 diseases: 之前）

experiment_groups:
  - id: mu_sweep
    display_name: "Hierarchy Loss μ Sweep (E1 ablation)"
    parent: E1
    description: "μ ∈ {0.0, 0.1, 0.5, 1.0, 3.0, 10.0} on sft_full_ft baseline"
    render_as: subtable     # update_leaderboard.py 据此生成 LEADERBOARD.md 子节
    sort_by: mu_hierarchy
    members:
      - id: E1_mu_0.0
        config: configs/sft_full_ft_mu_0.0.yaml
        output_dir: outputs/E1_mu_sweep/mu_0.0
        mu_hierarchy: 0.0
      - id: E1_mu_0.1
        config: configs/sft_full_ft.yaml         # 复用既有 E1
        output_dir: outputs/E1_sft_full_ft
        mu_hierarchy: 0.1
      - id: E1_mu_0.5
        config: configs/sft_full_ft_mu_0.5.yaml
        output_dir: outputs/E1_mu_sweep/mu_0.5
        mu_hierarchy: 0.5
      - id: E1_mu_1.0
        config: configs/sft_full_ft_mu_1.0.yaml
        output_dir: outputs/E1_mu_sweep/mu_1.0
        mu_hierarchy: 1.0
      - id: E1_mu_3.0
        config: configs/sft_full_ft_mu_3.0.yaml
        output_dir: outputs/E1_mu_sweep/mu_3.0
        mu_hierarchy: 3.0
      - id: E1_mu_10.0
        config: configs/sft_full_ft_mu_10.0.yaml
        output_dir: outputs/E1_mu_sweep/mu_10.0
        mu_hierarchy: 10.0

  - id: loss_variants
    display_name: "Hierarchy Loss Variants (E1 ablation)"
    parent: E1
    description: "Soft / mean-aggregated leaf target / hard projection"
    render_as: subtable
    members:
      - id: E1_soft_default
        config: configs/sft_full_ft.yaml
        output_dir: outputs/E1_sft_full_ft
        variant: soft_chapter_head
      - id: E1_leaf_target
        config: configs/sft_full_ft_leaf_target.yaml
        output_dir: outputs/E1_leaf_target
        variant: mean_aggregated
      - id: E1_hard_proj
        config: configs/sft_full_ft_hard_proj.yaml
        output_dir: outputs/E1_hard_proj
        variant: hard_projection
```

`update_leaderboard.py` 端需要的最小改动（伪码）：

```python
# scripts/update_leaderboard.py 增加：
def render_experiment_group(group: dict, output_md: list[str]) -> None:
    output_md.append(f"\n### {group['display_name']}\n")
    output_md.append(f"_{group['description']}_\n\n")
    headers = ["Variant", "μ" if group["id"] == "mu_sweep" else "Variant",
               "Mean AUROC", "HVR (mean)", "HVR (head)"]
    output_md.append("| " + " | ".join(headers) + " |\n")
    output_md.append("|" + "|".join(["---"] * len(headers)) + "|\n")
    members = group.get("members", [])
    if group.get("sort_by"):
        members = sorted(members, key=lambda m: m[group["sort_by"]])
    for m in members:
        row = load_metrics_for(m["output_dir"])  # 既有 helper
        output_md.append(format_row(m, row))

# 主流程末尾追加：
for group in manifest.get("experiment_groups", []):
    render_experiment_group(group, lines)
```

### A.4 Mean-aggregated-chap loss 实现

直接在 `src/training/losses.py` 末尾追加新函数（保留原 `HierarchicalLoss` 不动，避免破坏既有实验复现性）。注意调用方需在 `train_multitask.py` 根据 `training.hierarchy_loss_variant` 选择使用哪个。

```python
class HierarchicalLossMeanAggregated(HierarchicalLoss):
    """Variant of :class:`HierarchicalLoss` whose hierarchy penalty compares
    each leaf prob against the **mean of its sibling leaf probs** instead of
    the independent chapter-head output.

    Why: the default ``HierarchicalLoss`` compares ``σ(leaf_logit)`` against
    ``σ(chapter_head_logit)``. The two heads share only the 256-d projection
    upstream; below that they are independent ``nn.Linear`` layers, so the
    chapter head can drift downward to satisfy the inequality without ever
    constraining the leaf head's distribution. Empirically this is exactly
    what happens — see HVR_head ≈ 0.36 in 2026-04-14 results.

    Replacing the RHS with ``mean(σ(sibling_leaf_logits))`` removes the
    "independent escape valve": the only way to reduce the penalty is to
    actually push leaf probs down, which is what the loss is supposed to do.
    Mirrors the recipe used by ``mean_aggregate_chapter_probs`` in
    ``scripts/eval_hierarchy_violation.py``, so the trained model's HVR_mean
    becomes a directly-optimised quantity rather than an emergent one.

    Note: ``L_chapter`` (chapter-head BCE) is still computed against
    ``chapter_logits`` so the chapter head keeps learning a real target
    (useful for downstream interpretability + the explicit-head HVR sidecar).
    """

    def forward(
        self,
        leaf_logits: torch.Tensor,
        chapter_logits: torch.Tensor,
        leaf_labels: torch.Tensor,
        chapter_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_leaf = self.leaf_criterion(leaf_logits, leaf_labels)
        l_chapter = self.chapter_criterion(chapter_logits, chapter_labels)

        leaf_probs = torch.sigmoid(leaf_logits)                 # (B, 16)

        # Aggregate sibling leaves into a per-chapter mean using a one-hot
        # incidence matrix (precomputed once; lives on the same device as
        # the buffers registered in __init__).
        # incidence[c, l] = 1/|chapter c| if leaf l ∈ chapter c, else 0
        if not hasattr(self, "_chap_incidence"):
            n_leaves = self._leaf_idx.numel()
            n_chaps = int(self._chapter_idx.max().item()) + 1
            mat = torch.zeros(n_chaps, n_leaves, device=leaf_logits.device)
            for l, c in zip(self._leaf_idx.tolist(),
                            self._chapter_idx.tolist()):
                mat[c, l] = 1.0
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=1.0)
            self.register_buffer("_chap_incidence", mat, persistent=False)

        # mean_chap[:, c] = (1/|c|) Σ_{l∈c} σ(leaf_l)
        mean_chap = leaf_probs @ self._chap_incidence.T          # (B, n_chap)
        # Broadcast each leaf against its parent chapter's mean.
        parent_p = mean_chap[:, self._chapter_idx]               # (B, 16)
        leaf_p = leaf_probs[:, self._leaf_idx]                   # (B, 16)
        violations = torch.clamp(leaf_p - parent_p, min=0)
        l_hierarchy = violations.mean()

        total = (
            l_leaf
            + self.lambda_chapter * l_chapter
            + self.mu_hierarchy * l_hierarchy
        )
        components = {
            "loss_leaf": l_leaf.item(),
            "loss_chapter": l_chapter.item(),
            "loss_hierarchy": l_hierarchy.item(),
            "loss_hierarchy_target": "mean_aggregated_leaves",
            "loss_total": total.item(),
        }
        return total, components


def build_hierarchy_loss(variant: str, **kwargs) -> HierarchicalLoss:
    """Factory used by train_multitask.py. ``variant`` from cfg.training."""
    if variant == "soft_chapter_head":
        return HierarchicalLoss(**kwargs)
    if variant == "mean_aggregated":
        return HierarchicalLossMeanAggregated(**kwargs)
    raise ValueError(f"Unknown hierarchy_loss_variant: {variant!r}")
```

Hard-projection 变体（`variant="hard_projection"`）不需要新 loss 类，应该在模型 forward 出口处 clamp leaf_probs（在 `src/model/heads.py` 里加 `apply_hard_projection: bool` flag 即可），保持 loss 端不变。

### A.5 Per-chapter HVR 分解

当前 `compute_hierarchy_violation_rate` 只返回标量。如果 chapter A 的 4 个 leaves 全部违例，chapter B-F 都通过，整体 HVR 也只有 ~0.25——靠总数看不到这个 pattern。建议保留原签名向后兼容，加一个 `compute_hierarchy_violation_rate_breakdown`：

```python
# src/training/metrics.py 追加：

def compute_hierarchy_violation_rate_breakdown(
    leaf_probs: np.ndarray,
    chapter_probs: np.ndarray,
    disease_to_chapter_idx: dict[int, int],
    chapter_names: list[str] | None = None,
) -> dict:
    """Per-chapter violation rates + the overall rate (one pass).

    Returns:
        {
          "overall": float,                       # same as compute_hierarchy_violation_rate
          "per_chapter": {chap_idx: rate, ...},   # rate within (sample, leaf) pairs of that chapter
          "per_chapter_named": {chap_name: rate}  # only if chapter_names is provided
        }

    A chapter with zero leaves mapped to it (possible under partial sub-
    mappings used by E0 baseline mode) is reported as ``None`` for that
    chapter to keep "unobserved" distinct from "0.0 violations".
    """
    from collections import defaultdict
    chap_violations: dict[int, int] = defaultdict(int)
    chap_pairs: dict[int, int] = defaultdict(int)
    n = len(leaf_probs)
    for d_idx, c_idx in disease_to_chapter_idx.items():
        v = int((leaf_probs[:, d_idx] > chapter_probs[:, c_idx] + 1e-7).sum())
        chap_violations[c_idx] += v
        chap_pairs[c_idx] += n
    overall_v = sum(chap_violations.values())
    overall_p = sum(chap_pairs.values())
    overall = float(overall_v) / overall_p if overall_p > 0 else 0.0
    per_chap = {
        c: (chap_violations[c] / chap_pairs[c]) if chap_pairs[c] > 0 else None
        for c in chap_pairs
    }
    out = {"overall": overall, "per_chapter": per_chap}
    if chapter_names is not None:
        out["per_chapter_named"] = {
            chapter_names[c]: per_chap[c] for c in per_chap
        }
    return out
```

`scripts/eval_hierarchy_violation.py::compute_and_persist_hvr` 可在 payload 里追加 `"per_chapter_violation_rate"` 字段——所有现有 reader 仍能读 `"hierarchy_violation_rate"`，不破坏向后兼容。`update_leaderboard.py` 可选择加一列 `HVR worst chapter` 来突出失衡。

---

## Appendix B — μ sweep 决策树

跑完 Appendix A.1-A.2 后，按 (HVR_head, AUROC) 落点决定下一步：

```
                ┌────────── μ=10 结果 ──────────┐
                │                                │
       HVR_head ≈ 0?                       HVR_head 仍 ≈ 0.3+?
       │                                          │
   ┌───┴─────┐                                    ▼
   │         │                            metric/loss 不一致——
AUROC drop   AUROC drop                   loss 已经把 head 输出
< 0.01       > 0.05                       压平却没影响 leaf；
   │             │                        立即跳 A.4 (mean-agg
   ▼             ▼                        leaf target)。如果连
Innovation 2   Pareto curve                A.4 也救不回，问题
被救活——       是核心结论——               在「leaf head 表达力
论文 chapter 5  把 μ sweep 图作            过强 + chap head 形同
改成「需要     supplementary，论文          虚设」，写进 limitations
仔细调 μ，    重写为「hierarchy             并跳 A 的 hard-projection
但能找到工作  loss 与 BCE 存在               变体作为 endpoint。
点」。        fundamental tension」。
```

具体阈值规则（执行时按这套决断）：

| 观察 | 解读 | 下一步 |
|---|---|---|
| `μ=10`: HVR_head≈0, ΔAUROC < 0.01 | μ 调对了，loss 设计本身没问题 | 写论文，submit；no further work needed |
| `μ=10`: HVR_head≈0, ΔAUROC ∈ [0.01, 0.05] | 存在中等 tradeoff，但 μ* 可调 | 二分 μ ∈ (1, 10) 找最优工作点；画 Pareto |
| `μ=10`: HVR_head≈0, ΔAUROC > 0.05 | 显式 chap head 学到「全 0」trivial 解 | 跳 A.4 mean-aggregated leaf target |
| `μ=10`: HVR_head 仍 ≥ 0.2 | loss 没生效——梯度 / 数值问题 | 检查 `loss_hierarchy.item()` 在训练中是否 > 0；查 grad scale；查是否被 mixed precision 截断 |
| `μ=10`: HVR_head≈0 但 HVR_mean 不变 | 「chap head 作弊」假设证实 | A.4 + per-chapter breakdown，看是哪个 chapter 在拖 |
| `μ=10`: 训练发散 / loss NaN | μ 太大 | 加 warmup（前 5 epoch μ=0，再线性 ramp 到目标） + grad clip |
| 任意 μ: per-chapter HVR 高度不均（max/min > 5×） | 单 chapter 主导违例 | 检查该 chapter 的 leaf 数量、prevalence；可能需要 chapter-aware reweighting |

**触发 mean-aggregated leaf target (A.4) 的条件**：上表第 3、5 行任一发生即跑。
**触发 hard-projection (A.5) 的条件**：A.4 跑完仍未把 HVR_mean 推到 < 0.1，则 hard projection 作为 sanity check（必为 0），用来确认 metric/data path 本身无 bug。
**触发 curriculum 的条件**：μ sweep 跑完后，所有 μ 都出现 NaN / 训练崩；说明 hierarchy loss 在早期 epoch 与 BCE 梯度方向相悖，需要 phase-wise 训练。
