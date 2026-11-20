# 证据清单（2026-11-20，RNA 单接触明细已补齐）

本清单列出论文可以引用的每一条证据，每条写明结论、数字、数据范围和对应的结果表。"核对"一栏的含义：
- **已核对**：本清单中的数字是 2026-11-20 直接从所列结果表重新读出或重新计算的。
- **生成时核对**：数字在结果表生成的当天核对过，本次没有重算。

所有外部方法都用官方程序零样本运行；本方法的数字是按靶标分组交叉验证的折外结果。

## 1. 数学表示与定理检验

| # | 结论 | 数字 | 数据范围 | 结果表 | 核对 |
|---|---|---|---|---|---|
| 1.1 | 普通点云 Čech 是共同支撑在零半径点支撑下的特例 | 零半径 τ 与精确 miniball 半径的最大差 1.1×10⁻¹⁴ Å；Čech 与 α 复形端点最大差 4.4×10⁻¹⁶ Å | RNA 控制结构 1A9L、1A60（不是原审计的 16 个结构） | `results/tables/cech_rips_compare.csv`，`results/reports/cech_rips_compare.md` | 生成时核对 |
| 1.2 | 在普通点云上，Čech 与 Rips 也不同 | "结构×阶"中 2/6 不同 | 同上 | 同上 | 生成时核对 |
| 1.3 | 蛋白-DNA：按精确 Å，共同支撑与两两支撑的条形码不同 | 有向图 5/8 个结构至少一阶不同（H1 5/8，H2 3/8）；对称化图 8/8（H1 6/8，H2 8/8）；H0 都相同。条数口径：长度不超过 10⁻⁶ Å 的条属于数值精度，不计入结论；用公开坐标独立重导出后，8 个结构的条数和有限总持续度与本表逐行相同。对称化 H2 在最小条长 10⁻⁶、10⁻³、10⁻² Å 下，每个结构仍多出 1–111、1–90、1–55 根，8/8 不变 | 原审计队列中的 8 个 FTDMP 诱饵（1FOK、1HJC、1K79、1TRO 各 2 个），坐标取自官方 Zenodo 12804208 | `results/tables/union_audit_pd_barcodes.csv`，`results/reports/union_audit_pd.md`，`results/tables/union_barcode_count_by_min_length.csv` | 已核对 |
| 1.4 | RNA：按精确 Å，共同支撑与两两支撑的条形码不同 | 有向图和对称化图都是 8/8 个结构在 H1、H2 不同，H0 都相同。对称化 H2 多出 48–118 个条形码，有限总持续度多 2.33–5.96 Å。条数口径：长度不超过 10⁻⁶ Å 的条属于数值精度，不计入结论；用公开坐标独立重导出后，“是否不同”和有限总持续度逐行一致（差小于 10⁻¹²），条数在 12/48 行相差 1–2 根，都是这类短条，所以论文以有限总持续度为主、条数为辅并注明最小条长。对称化 H2 在最小条长 10⁻⁶、10⁻³、10⁻² Å 下，每个结构多出 48–118、43–94、28–62 根，8/8 不变 | 原审计队列中的 8 个 RNA 诱饵（clash 比 0.6） | `results/tables/union_rna_barcode_summary.csv`，`results/tables/union_barcode_count_by_min_length.csv` | 已核对 |
| 1.5 | 全队列满足仿射 Jung 界 | 蛋白-DNA 132,138 条路径、RNA 340,874 条路径，违反数都是 0。检验量最大值：蛋白-DNA 1.130，RNA 1.141，都低于 √(3/2)≈1.225 | 原审计的 16 个结构，有向和对称化两种图，阶 0–3 | `results/tables/union_audit_pd_ratio_distribution.csv`，`results/tables/union_rna_ratio_distribution.csv` | 已核对 |
| 1.6 | 不需要备选 patch，也会出现超出两两支撑的缺口 | 蛋白-DNA：只含单接触 patch、δ=0 而 η>0 的路径有 9,616 条，分布在 7/8 个结构中，最大 η 为 0.676 Å。RNA：8/8 个结构、有向加对称化共 71,279 条，其中检验量大于 1 的 43,078 条，违反界 0 条，检验量最大 1.141。RNA 的结果表只保留了检验量最高的 10 条路径的 η（其中最大 0.497 Å），全体 71,279 条中的最大 η 不在已入库的表里 | 蛋白-DNA 8 个诱饵；RNA 8 个诱饵 | `results/tables/union_audit_pd_per_structure.csv`，`results/tables/union_rna_single_contact_summary.csv`，`results/tables/union_rna_single_contact_top.csv` | 已核对 |
| 1.7 | 归档路径数可复现 | 16 个结构在两种图上、各阶的路径数与归档一致 | 原审计的 16 个结构 | `results/tables/path_landscape.parquet` | 已核对（蛋白-DNA 由批处理的队列检查通过；RNA 由超算会话导出） |

## 2. 与外部打分方法的同池对比

| # | 结论 | 数字 | 数据范围 | 结果表 | 核对 |
|---|---|---|---|---|---|
| 2.1 | RNA 上，本方法与经典统计势持平 | 宏平均 Spearman：rsRNASP 0.580、cgRNASP 0.560、DFIRE-RNA 0.518；本方法最佳臂 0.550（20 次划分的均值）。按单次划分，A7_PTH 与 rsRNASP 的配对差为 −0.032 [−0.170, 0.087] | PM 诱饵集：20 个靶标，806 个 decoy，不含 native | `results/tables/external_scoring_pm_full_per_target.csv`，`external_scoring_pm_bootstrap_single_split.csv` | 已核对 |
| 2.2 | 蛋白-DNA 上，本方法明显好于 VoroMQA | bound：0.244 对 −0.004，配对差 +0.248 [+0.194, +0.305] | FTDMP bound，47 个靶标，9,400 个候选 | `results/tables/external_scoring_pd_summary.csv`，`external_scoring_pd_bootstrap.csv` | 已核对 |
| 2.3 | FTDMP 官方共识排序在 bound 上好于本方法（只作参照） | 0.324 对 0.244，配对差 −0.079 [−0.152, −0.005]；Top-1 CAD 0.596 对 0.356 | 同上 | 同上 | 已核对 |
| 2.4 | VoroMQA 分数可复现 | 用 Voronota v1.29.4816 抽检 21/21 逐位一致 | 18,800 条中随机抽 20 条，另加 1BDT 那 1 条 | `results/tables/pd_voromqa_scores.jsonl`，`results/reports/external_scoring_pd.md` | 生成时核对 |

## 3. 难负例层（冻结特征）

| # | 结论 | 数字 | 数据范围 | 结果表 | 核对 |
|---|---|---|---|---|---|
| 3.1 | 在 bound 上，加入条形码层后难例对胜率提高；只加支撑差不提高 | T3−T2 +0.0269 [+0.0129, +0.0399]，Holm p=0.0006；T2−T1 +0.0006 [−0.0054, +0.0064]，p=0.84 | FTDMP bound，47 个靶标，种子 0–9 合并 | `results/tables/hardpair_frozen_summary_pooled.csv` | 已核对 |

## 4. 离子与水位点（共同见证点）

| # | 结论 | 数字 | 数据范围 | 结果表 | 核对 |
|---|---|---|---|---|---|
| 4.1 | Zn：共同见证点几乎覆盖全部真实位点 | 640 个位点，1 Å 召回 0.902，2 Å 召回 0.992；最近位点误差中位数 0.142 Å；证书失败 0/157,715 | 300 个 Zn 结构全部完成 | `results/reports/ion_full_measurements.json`，`results/tables/ion_full_summary.csv` | 已核对 |
| 4.2 | Zn：两两通过而共同支撑不通过的难负例占比高 | 三配体候选中占 51.9%，去掉冲突后 26.8%；经典配体子层为 8.6% | 同上 | `results/tables/ion_full_summary.csv` | 已核对 |
| 4.3 | 水：2 Å 内大部分可覆盖，1 Å 内只覆盖一半多 | 9,441 个位点，1 Å 召回 0.579，2 Å 召回 0.945；误差中位数 0.852 Å；证书失败 0/14,824,091 | 300 个结构全部完成 | 同上 | 已核对 |

## 5. 未完成或不进正文

| 项目 | 现状 | 处理 |
|---|---|---|
| Mg（RNA）位点 | 16/20 个结构完成。56 个位点中 1 Å 召回只有 0.089，2 Å 召回 0.482，误差中位数 2.18 Å，结果很差 | 不进正文；需要时在补充材料中如实报告 |
| Mg 被截断的 4 个结构 | 4V9P、5FDU、6XQE、9DFD 超过 25 万原子或 5 万 patch 的上限，被截断（`ion_full_per_structure.csv` 中 status=size_censored）。放宽上限的重跑脚本（`scripts/slurm/ion_mg_relaxed.sh`）已写好，但没有结果 | 未完成 |
| 外部离子位点对比（Metal3D、AllMetal3D） | 超算 CPU 作业脚本和环境已就绪（`experiments/external_metal_site_run/`），还没有运行 | 未完成；没有结果前，正文不写与 Metal3D 的比较 |
| FTDMP unbound | 候选几乎全是错误构象：47 个靶标中只有 8 个有 CAD ≥ 0.3 的候选。本方法 0.192，FTDMP −0.011，VoroMQA −0.032 | 不进正文，放补充材料并说明口径（`results/reports/external_scoring_pd.md`） |
| lociPARSE | PM 上宏平均 0.596。作者的训练集清单（Zenodo 12669705）返回 403，无法核对是否包含 PM 靶标 | 不进正文，放补充材料 |
| 1BDT 那 1 个 unbound 候选 | 本方法特征表本来就缺这一条，很可能是当时特征提取失败或被排除 | 同池对比中已排除，用 18,799 对，并作说明 |
| 3dRNAscore、ARES、DDNA3、ITScore-PR、3dRPC | 没有公开可复现的程序或权重（2026-11-19 查证） | 不进正文（`docs/EXTERNAL_BASELINE_PLAN.md` 第 7 节） |
| 四元严格超出 | 所有路径最多只含 4 个不同 patch，无法检验 | 不进正文，只作理论陈述 |
| 原 16 个结构的 Čech/Rips 精确对比 | 第 1.1、1.2 条只在控制结构上计算 | 正文改用第 1.3、1.4 条的按 Å 共同支撑对两两支撑比较 |
