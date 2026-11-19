# 蛋白-DNA 同池外部对比：VoroMQA 与 FTDMP 官方排序

**结论：** 在本方法的同一候选池上比较。VoroMQA 在 bound 和 unbound 两个条件下都几乎不能给候选排序，本方法明显好于它。FTDMP 官方排序在 bound 条件下好于本方法；在 unbound 条件下，三者的 Top-1 都接近 0，但本方法的逐靶标 Spearman 明显更高。FTDMP 官方排序是多个打分函数的共识排序，不是单一打分函数，只作参照。

## 候选池与数据来源

- 候选池：`results/tables/ftdmp_candidates.csv`，即本方法实际使用的候选，共 18,799 个（bound 47 靶标 × 200 个；unbound 47 靶标，共 9,399 个）。
  - unbound 的 1BDT 只有 199 个，缺 `1BDT_ftd161s0d690072p4p56p25p63p144p345`。FTDMP 官方包和 VoroMQA 分数里都有这个候选，但它不在本方法的候选池中，所以不进入计算。
- VoroMQA：`results/tables/pd_voromqa_scores.jsonl`，共 18,800 条，没有缺失值。
  - 由仓库脚本 `experiments/baselines_ext/score_pd_voromqa.py` 调用 `voronota-voromqa -i` 生成，原始计算时没有记录程序版本。
  - 2026-11-19 用 Voronota 官方发布版 v1.29.4816 从源码编译，随机抽 20 个候选，加上 1BDT 那一个，共 21 个重新计算。VoroMQA 分数、残基数、原子数全部逐位一致。
- FTDMP 官方排序：Zenodo 12804208 的 `Protein-DNA.tar.gz`，取各靶标 `scoring_results-ranks.txt` 中的 `relaxed_RJS_rank`，数值越小越好。
  - unbound 的排序文件把每个候选重复列了两次，两次的排名相同，已去重。
  - 逐候选排名见 `results/tables/external_scoring_ftdmp_official_ranks.csv`。
- 标签：候选清单中的 `label_cadscore`，即界面 CAD-score。

## 指标

与 EXP-06 相同：
- 每个靶标内计算打分与 CAD-score 的 Spearman，再对 47 个靶标取宏平均。
- Top-k 指排名前 k 的候选中 CAD-score 的最大值，再对靶标取平均。

本方法的数值取自已保存的单次划分折外结果（`metrics_pd_*.json`），按靶标与外部方法配对，做 2,000 次靶标层 bootstrap。

| 条件 | 方法 | 宏平均 Spearman | Top-1 | Top-5 | Top-10 |
|---|---|---|---|---|---|
| bound | FTDMP 官方排序 | 0.324 | 0.596 | 0.648 | 0.677 |
| bound | VoroMQA | −0.004 | 0.112 | 0.285 | 0.361 |
| bound | 本方法 A1_P（物理） | 0.243 | 0.326 | 0.574 | 0.644 |
| bound | 本方法 A7_PTH（物理+拓扑+谱） | 0.244 | 0.356 | 0.579 | 0.645 |
| unbound | FTDMP 官方排序 | −0.011 | 0.024 | 0.035 | 0.045 |
| unbound | VoroMQA | −0.032 | 0.014 | 0.041 | 0.054 |
| unbound | 本方法 A1_P | 0.200 | 0.019 | 0.059 | 0.077 |
| unbound | 本方法 A7_PTH | 0.192 | 0.030 | 0.066 | 0.079 |

## 配对差（本方法 A7_PTH 减外部方法，逐靶标 Spearman，95% CI）

| 条件 | 对 FTDMP 官方排序 | 对 VoroMQA |
|---|---|---|
| bound | −0.079 [−0.152, −0.005] | +0.248 [+0.194, +0.305] |
| unbound | +0.202 [+0.134, +0.272] | +0.224 [+0.164, +0.284] |

A1_P 和 CleanFull_PTHS 的配对差与此相差不超过 0.01，见 `results/tables/external_scoring_pd_bootstrap.csv`。

## unbound 口径核查（2026-11-19）

在 unbound 上，FTDMP 官方排序的宏平均 Spearman 接近 0（−0.011）。我检查了这个反常值，确认它不是计算错误：
- 排序方向：用同一段代码、同样以 `-relaxed_RJS_rank` 为分数，bound 上得到 0.324。在 unbound 上，排名与配体 RMSD 的相关也是正的（+0.027），方向没有反。
- 候选对齐：本方法的 `label_cadscore` 与官方 `cad_scores.txt` 逐候选完全相同，最大差为 0。官方排序文件里重复的两段排名逐条一致。
- 原因在候选池本身：unbound 候选中 48% 的 CAD-score 恰好为 0，只有 3.5% 大于 0.1；47 个靶标中只有 8 个至少有一个 CAD ≥ 0.3 的候选。几乎全是错误构象时，以挑出近天然构象为目标的排序就给不出有意义的序。
- 只看这 8 个有可接受候选的靶标：FTDMP 官方排序的 Spearman 为 0.019，VoroMQA 为 −0.009，本方法 A7_PTH 为 0.301，A1_P 为 0.343。

**论文写法：** 本方法是在同一数据集上按靶标交叉验证、以 CAD-score 为标签训练的，能学到 CAD 接近 0 的候选之间的细微差别；FTDMP 和 VoroMQA 是零样本使用的。因此 unbound 上本方法的优势只放补充材料并说明上述口径，正文的外部对比以 bound 为准。

## 局限

- 本方法的数值是单次划分的结果，不是多次划分的平均。
- VoroMQA 是单模型质量评估，零样本使用；本方法是在同一数据集上按靶标分组交叉验证训练的，两者的训练条件不同。
- 按主人"现有能用就用"的规则，没有补跑 HADDOCK3 打分和 DDNA3。

复现命令：`cd experiments && python external_scoring_pd.py --tables ../results/tables`
