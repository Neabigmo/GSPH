# GSPH Feature Representation 2.0 — 消融报告（Ablation-2.0）

日期：2026-11-14。基于 Ablation-1.0 修正协议：固定候选池、固定 fold（复用 v1 cv_splits）、固定 XGBRanker 预算（80 rounds, depth4, lr0.05, sub/colsample 0.9）；膜任务严格 LOPO 回归（插补→方差过滤→标准化→±10 截断→Ridge/ENet）。**未重新提取任何基础 GSPH 特征**；v2 全部特征由已有 parquet 派生/精选。

## 0. 不可恢复项声明（诚实记录）

- **T1 逐路径支撑分布**：引擎 `extract_gsph6_atoms` 在内存中计算了逐路径 `tau`（`birth` dict），但 driver 仅落盘 `tau_{2,3,4}_{mean,min,max}` 聚合值；parquet/audit 中无分位数源数据。按方案"禁止重解析原子/虚构"，**未实现**。
- **T3 真实 persistence barcode**：引擎返回 `persistence_details`（含 birth/death/right_censored 的真实 intervals），同样未落盘；parquet 仅有 `persistence_interval_count_k{0..2}`。**未实现**，不允许凭 count 虚构 barcode。
- 两者若需实现，需在引擎侧加 dump 并重跑（本轮禁止）。这是管线落盘字段的设计缺口，已记录。

## 1. v2 特征组（互斥、含派生归一化 GLMY）

| 组 | RNA | 膜 | PD b/u | 内容 |
|---|---|---|---|---|
| T2 | 159 | 159 | 159 | betti/kernel/dim_gap/filling/admissibility + **归一化 GLMY**（÷dim_Omega，0 保护）+ pathchem 比例/熵/转移 + path span/support + tau/eta/Δtau + persistence counts |
| H2 | 87 | 87 | 87 | scale_0.5 低模态读出（lse/ler/lht0.1/gap/swm）+ base 谱 + 6 跨通道相关 |
| PcV2 | 125 | 118 | 121 | 全通道 global patch（非 rdf_ordered）+ token mean + RNA 专属 + rdf_ordered 降采样 41 bin |
| S | 63 | 62 | 62 | 计数类 + 原始 GLMY dim/rank（规模控制） |

## 2. 主结果（target-macro Spearman；膜=LOPO macro-Spearman 最优预测器）

| 模型 | RNA | PD bound | PD unbound | 膜(ridge) |
|---|---|---|---|---|
| B1 P-full (A1) | 0.529 | 0.243 | 0.200 | ~0.05–0.10 |
| B2 Pcompact-v2 | 0.533 | 0.211 | 0.171 | **+0.156** |
| B3 T-old (A2) | 0.477 | 0.208 | 0.118 | ~0 |
| B4 T2 | 0.458 | 0.203 | 0.123 | −0.151 |
| B5 H-old (A3) | 0.497 | 0.229 | 0.136 | ~0 |
| B6 H2 | 0.457 | 0.210 | 0.099 | −0.212 |
| B7 Pc+T2 | 0.500 | 0.203 | 0.162 | +0.003 |
| B8 Pc+H2 | 0.518 | 0.217 | 0.161 | +0.004 |
| B9 Pc+T2+H2 | 0.523 | 0.223 | 0.161 | −0.069 |
| B10 Pc+T2res+H2res | 0.479 | 0.208 | 0.159 | — |
| B11 Size (A8) | 0.130 | 0.219 | 0.142 | ~−0.10 |
| P_PCA142/40/20 | .498/.516/.502 | .219/.217/.205 | .135/.127/.151 | — |

## 3. 交互效应正式检验（核心问题）

### v1 全特征空间（已有 OOF）

I_TH = [S(P+T+H)−S(P+T)] − [S(P+H)−S(P)]，10k paired bootstrap + 10k sign-flip permutation + Holm：

| 任务 | I_TH | 95% CI | p_perm | p_holm |
|---|---|---|---|---|
| RNA | +0.015 | [−0.019, +0.052] | 0.419 | 0.837 |
| **PD bound** | **+0.034** | **[+0.014, +0.056]** | **0.0025** | **0.007** |
| PD unbound | +0.004 | [−0.013, +0.020] | 0.668 | 0.837 |

→ **用户假设证实**：在完整特征空间中，PD bound 的 Hodge 增量依赖拓扑存在（协同显著）；unbound 为负对照≈0 符合预期。

### v2 紧凑空间（Pc/T2/H2）

| 任务 | I_TH(v2) | 95% CI | p_perm |
|---|---|---|---|
| RNA | +0.038 | [−0.031, +0.122] | 0.422 |
| PD bound | +0.014 | [−0.007, +0.036] | 0.211 |
| PD unbound | +0.008 | [−0.020, +0.037] | 0.565 |

→ **协同在紧凑空间衰减至不显著**：交互信号部分来自精简时丢弃的特征（全谱 H、原始 GLMY 维度等）。dH|T bound = +0.020 (p_perm=0.029, **p_holm=0.323** 不显著)。

## 4. 其余关键增量（v2，Holm 后全不显著）

- ΔT=PcT2−Pc：bound −0.008 / unbound −0.009 / RNA（见 topk_bootstrap_v2.csv）
- ΔTH=PcT2H2−Pc：bound +0.012 / unbound −0.009 —— 紧凑空间净增量仍≈0
- PcV2 vs P-full：bound −0.032 (p=0.012, holm 0.14)——紧凑 P 在 PD 上丢失少量信息；RNA 持平
- T2 vs T-old / H2 vs H-old：均≈0 微负——归一化与精简几乎无损保留信息但也没增强

## 5. 回答方案中的五个问题

1. **持久区间扩展（barcode）是否改善 T？** —— 无法回答：barcode 未落盘，本轮禁止重提取。需引擎侧补持久化字段后另测。
2. **归一化 GLMY 是否保留独立信息？** —— T2(159d, 归一化+去原始维数) ≈ T-old 微降（bound −0.005, RNA −0.020, ns），说明其信息大部分可与原始维度互换；独立增量依然未见。
3. **Hodge 精简后 PD bound 条件增量是否保留？** —— 方向保留（dH|T +0.020）但 Holm 后不显著；完整空间显著交互（I_TH +0.034, p_holm=0.007）在紧凑空间衰减为 +0.014 ns。
4. **Top-k 是否比 Spearman 更体现优势？** —— 否：top-1 配对差全部 ns（见 topk_bootstrap_v2.csv）。
5. **物理压缩到 ~120 维后拓扑是否仍有增量？** —— 否：Pc+T2、Pc+T2+H2 均未显著超过 PcV2。

## 6. 归因终判

- **P 承载绝大部分信息**，且可压缩至 ~120 维（RNA 无损、PD 微损 ~0.03、**膜 LOPO 上 PcV2 ridge +0.156 反而为全部实验最佳**——紧凑化去噪有益）。
- **T 与 H 单用真实但弱**；对 P 的净增量≈0；规模混杂在 PD bound 严重（Size≈0.22 vs 全模型 0.24）。
- **唯一可重复的非常规信号**：全特征空间中 PD bound 的 T×H 协同（p_holm=0.007）。它提示 Hodge 谱读出能补充 GLMY 路径组织的部分缺失信息，但该效应在 87+159 维紧凑读出下不再显著——**协同依赖于被精简掉的高维谱/维度细节，尚不足以成为实用的预测增量**。
- **下一步应按方案的停止标准执行**：不再堆特征维数，而是回到数学对象——先将逐路径 support 与真实 barcode 落盘（轻量改动），再判断高阶支撑结构与标签的关系；膜任务当前 7 蛋白规模不足以支撑通用 LOPO 结论。

## 7. 产物清单（results/ablation/）

`feature_manifest_v2.csv`、`feature_health_v2.csv`、`{topology,hodge,physical_compact}_features_v2_{task}.parquet`、`oof_v2_{task}.parquet`、`predictions_oof_v2.parquet`、`metrics_v2{,_task}.json`、`per_target_v2.csv`、`topk_bootstrap_v2.csv`、`interaction_test_v2.csv`（全特征空间）、`interaction_test_compact_v2.csv`（紧凑空间）、`interaction_per_target_v2.csv`、代码 `GSPH_H3/ablation/*.py`。
