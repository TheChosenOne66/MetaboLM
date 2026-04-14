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
- [ ] 写 supplementary section 报告结果
