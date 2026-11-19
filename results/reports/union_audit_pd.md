# 蛋白-DNA 八个原诱饵：按 Å 的条形码重核与定理检验

**结论：** 在精确的 Å 尺度上，原审计队列中的 8 个蛋白-DNA 诱饵，有向图有 5/8、对称化图有 8/8 的共同支撑条形码与两两支撑不同。全部 132,138 条路径都满足仿射 Jung 界，没有一条违反。旧的三尺度粗网格只报出 2/8 不同，低估了差异。

## 输入核对

- 结构：`manifests/union_audit_structures.tsv` 中 task=pd 的 8 个诱饵，即 1FOK、1HJC、1K79、1TRO 每个靶标各 2 个。
- 坐标来源：FTDMP 官方 Zenodo 12804208 `Protein-DNA.tar.gz`，路径为 `Bound_docking/<靶标>/relaxed_top_complexes/`。坐标不入库，各输入文件的 SHA256 记录在 `results/tables/union_audit_pd_per_structure.csv`。
- 文件名与编号：8 个 ID 都在官方包中，逐字一致。
- 标签：官方 `cad_scores.txt` 的界面 CAD-score 与 `ftdmp_candidates.csv` 中的 `label_cadscore` 逐个相同。每个靶标是 1 个 CAD = 0 的错误诱饵加 1 个 CAD 为 0.72–0.82 的近天然诱饵。
- 路径数：8 个结构在两种图上、各阶的路径数都与归档的 `path_landscape.parquet` 完全一致（批处理的队列检查通过）。

## 计算

使用仓库现有代码，未作改动：
- `experiments.exp05_molecular_landscape.deep_complex_dump --fixed-ids` 做逐路径导出；
- `experiments.union_audit_batch` 做按 Å 的条形码比较和 Jung 界检验。

在云端 8 核上运行，最大的 1HJC_ftd142（101,438 条对称化路径）用时 44 分钟。

## 结果

**条形码（各阶中共同支撑与两两支撑的区间多重集是否不同）**

| 图 | H0 | H1 | H2 | 至少一阶不同 |
|---|---|---|---|---|
| 有向 | 0/8 | 5/8 | 3/8 | 5/8 |
| 对称化 | 0/8 | 6/8 | 8/8 | 8/8 |

- 对称化图的 H2 上，共同支撑比两两支撑多 1–111 个条形码，有限总持续度多 0.03–7.39 Å。
- 逐结构明细见 `results/tables/union_audit_pd_barcodes.csv`。

**定理检验**

- 仿射 Jung 界的检验量 (r_min + τ)/(r_max + τ_pw + δ)（界为 √(3/2)）的最大值：有向图为 1.113，对称化图为 1.130，都低于界 √(3/2) ≈ 1.225。违反界的路径数为 0。分布见 `results/tables/union_audit_pd_ratio_distribution.csv`。
- 只含单接触 patch、δ = 0 而 η > 0 的路径有 9,616 条，分布在 7 个结构中，最大 η 为 0.676 Å。这说明不需要备选 patch，也会出现超出两两支撑的缺口。
- 所有路径最多含 4 个不同 patch，所以仍然不能检验严格的四元超出。
