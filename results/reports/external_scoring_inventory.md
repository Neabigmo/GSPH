# 核酸外部打分：已有分数与小规模试算

原则是先复用已有分数，不重跑完整数据集。本次只对 PM 的 **5 个靶标、210 个结构（含 5 个 native）**执行了 rsRNASP、cgRNASP、DFIRE-RNA；没有运行全部 826 个结构，没有重跑 FTDMP/VoroMQA，也没有生成 GSPH 折外预测。

## 结论与缺口

| 方法／来源 | 现成分数 | 与本次目标口径是否对齐 | 本次可用数字 | 还缺什么 |
|---|---|---|---|---|
| FTDMP 官方 | Zenodo 12804208 中逐候选 `relaxed_RJS_rank` | ID 命名空间可用，但没有 GSPH 实际候选清单，不能认证同池 | 每条件 47 靶标 × 200 个不同模型 = 9,400 | 本方法每条件的完整 `(protocol,target,model)` 清单；不能仅凭数量相同认证 |
| VoroMQA | Zenodo 包中**没有** VoroMQA 分数；旧打分 JSONL 未找到 | 未对齐 | 不计算同池性能 | `pd_voromqa_scores.jsonl` 的逐候选 `pdb,voromqa`，或另行授权重打分 |
| rsRNASP | 2022 原文 PM 逐靶标表；旧 RNAtive 表也有该列 | 原文是 PCC/DI，不是 Spearman/RMSD；RNAtive 不是 PM | PM 五靶标平均 Spearman **0.708484** | 全部 20 靶标的当前版本能量尚未计算；GSPH 冻结池逐候选哈希未取得 |
| cgRNASP | 2023 原文 PM+RNA-Puzzles 汇总；旧 RNAtive 表有该列 | 合并集不是 PM 单集；RNAtive 不是 PM | PM 五靶标平均 Spearman **0.646706** | 同上；PM-only 补充表未成功取得 |
| DFIRE-RNA | rsRNASP 原文 PM 逐靶标表；旧 RNAtive 表有 `DFIRE` 列 | 原文 PCC/DI 不是目标指标；RNAtive 不是 PM | PM 五靶标平均 Spearman **0.637858** | 全部 20 靶标当前版本能量尚未计算 |
| RNAtive 导入表内其他评分列 | RNAtive 各变体、3dRNAscore、ARES、RNA3DCNN、lociPARSE 等 | 只能直接复用到这些 CSV 自己的候选行，不是 PM 现成分数 | 96 个靶标表、43,630 行；与 PM 规范化模型 ID 交集 **0** | 若要跨命名体系证明结构重合，需要坐标或结构哈希及模型映射 |

RNA 的五靶标数字是逐靶标 Spearman 的算术平均，不是拼接全部结构后的相关性；不能当作 20 靶标最终基准。

## FTDMP：实际下载内容与候选池

来源：[Zenodo 12804208](https://zenodo.org/records/12804208)，文件 `Protein-DNA.tar.gz`，**1,668,339,414 B（1.668 GB，1.554 GiB）**，保存于仓库外。完整文件 MD5 已核对：`a84cf14a3f5917b726c724cd2f09196f`。只流式读取归档目录与文本，没有将大批 PDB 解压进仓库。

目录为 `Protein-DNA/{Bound_docking,Unbound_docking}/{target}/`；PDB 位于 `relaxed_top_complexes/`。每个条件都是 47 靶标、每靶标 **200 个 PDB / 200 个不同的 rank ID**，共 18,800 个 PDB；不是每靶标 400 个不同模型。

| 文件 | 实际列／格式 | 意义 |
|---|---|---|
| `scoring_results-ranks.txt` | `ID relaxed_RJS_rank` | 逐候选官方序位，越小越好；没有原始 RJS 分数或 VoroMQA 列 |
| `cad_scores.txt` | `model iface_cadscore iface_site_based_cadscore` | 参考结构 CAD 标签，不是 VoroMQA 预测分数 |
| `rmsd_results.txt` | 无表头：`model RMSD` | 逐候选 ligand RMSD 标签 |
| `binding_site_CAD-scores.txt` | 无表头：`model value` | 参考结构 binding-site CAD 标签，不是外部打分预测 |

全包非 PDB 文件只有上述四种，各 94 份。ID 示例：`1R4O_ftd162s0d695731p17p13p27p162p54p150`；rank 的 `ID` 加 `.pdb` 后与 PDB 文件名、CAD/RMSD 的 `model` 对齐。必须保留 protocol 和 target，不能只按 PDB 靶标名合并 bound/unbound。

**重复记录与文档纠偏：**46 个 unbound 靶标的 rank 文件含两个完全相同的 200 行排序及中间的重复表头，pandas 读取后各 401 行；`1KSY` 只有 200 行。重复 ID 的 rank 值没有冲突。bound 均为 200 行。unbound `1KSY/cad_scores.txt` 另有一行重复表头，读取后的 201 行中只有 200 个真实模型。清理表头、按 ID 去重后，94 个靶标的 rank、PDB、CAD、RMSD 都有完整的 200 个模型交集。因此仓库旧注释中的“unbound 400/靶标”不能解读为 400 个不同候选。逐靶标计数见 `../tables/external_scoring_ftdmp_archive.csv`，其中 `n_rank_rows`、`n_cad` 是 pandas 读取后的原始行数，`n_joined` 是真实模型交集数。

### 为什么没有给出 FTDMP/VoroMQA 同池 Spearman 与 Top-1

`experiments/baselines/protein_dna/eval_ftdmp_official_samepool.py` 的 `main()` 从 JSONL 的 `pdb` 路径提取 `(protocol,target,basename)`，作为实际共同候选池；按当前脚本位置解析的预期文件是 **`experiments/baselines_ext/pd_voromqa_scores.jsonl`**。源码旧布局的 `baselines_ext/pd_voromqa_scores.jsonl` 也检索过，两个可读仓库均未找到该文件。`experiments/baselines_ext/score_pd_voromqa.py` 的 `PD_OUT_JSONL` 默认值指向旧工作站，并非云端现存文件。

`manifests/targets_pd.csv` 目前仅有 `target_id,n_decoys,task` 表头，连靶标行也没有，更不是逐候选清单。`hardpair_oof_v2_pd_*.csv` 是引用缺失 parquet 的汇总，不含完整模型 ID。没有找到可替代 JSONL 的 9,400 候选完整 manifest 或逐候选 OOF。

公共归档恰好也是每条件 9,400 个，说明补回 GSPH 清单后很可能可以直接 join，但**数量相同不是 ID 相同的证据**。本次不把公共全池冒充 GSPH 同池，故不输出“samepool”性能数字，也不凭官方序位伪造 VoroMQA 分数。

补充检索了可读主目录及 `/mnt`、`/srv`、临时目录，仍未发现上述 JSONL、`rna_structure_ids.csv` 或引用的 PD OOF parquet；不覆盖无权限的 `/root`。恢复 JSONL 后的原脚本口径是：FTDMP 用 `Spearman(-relaxed_RJS_rank, iface_cad)`；VoroMQA 用 `Spearman(voromqa, iface_cad)`，越大越好；Top-1 报所选候选的 CAD 与 ligand RMSD，另可汇总 `CAD >= 0.5` 比例。对应代码是上述脚本及同目录 `eval_ftdmp_voromqa.py`。它们不是 RNA 的能量对 RMSD 指标。缺少 GSPH OOF 不妨碍未来只评外部方法，但本次首先缺的是候选池及 VoroMQA 分数。

## PM 公开数据与论文

公开入口为 [Tan-group/rsRNASP 的 PM_decoy_set](https://github.com/Tan-group/rsRNASP/tree/73803d275850ff4f0d0a81340ace15ae633e1c8f/PM_decoy_set)，已从官方仓库取得。该目录文件总量 **97,553,504 B（97.55 MB）**，低于 1 GB；包含 20 个靶标、826 个 PDB，另有 `RMSD/`、`DI/` 与 README。826 = **806 个 decoy + 20 个 native**，单靶标含 native 为 36–43 个结构，不是全部恰好 40 个。逐靶标数量见 `../tables/external_scoring_pm_pool.csv`。

PM 来自 FARFAR2、RNAComposer、SimRNA、3dRNA v2.0；模型名例如 `RNAComposer-1Z43-3.pdb`、`SimRNA-1Z43-3.pdb`、`3dRNA-1Z43-2.pdb`，native 是 `1Z43.pdb`。标签文件直接给出模型文件名与 RMSD。原文说的是约 40 个 decoy/靶标，没有给出本次下载文件的精确 826 计数。该数量与 `manifests/datasets.yaml` 的 PM/clash06 标称池吻合；clash06 是本方法的接触过滤配置，本次不以此修改公开 PDB。尚未取得旧 GSPH 冻结输入的逐候选哈希，不能仅以总数独立认证冻结版本一致。

### 原文中已有的数值

| 论文／位置 | 池 | 已有指标与数字 | 相对本次 Spearman(energy,RMSD) |
|---|---|---|---|
| [rsRNASP，Biophysical Journal 121 (2022), 142–156](https://doi.org/10.1016/j.bpj.2021.11.016)，Table 1 | PM，20 靶标 | native 排第一：rsRNASP **16/20**，DFIRE-RNA **10/20**；排前五：18/20、14/20 | native 识别率，不是逐靶标排序相关 |
| 同文 [Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8758408/#tbl3) | PM，20 靶标 | 每靶标给最低能量结构 DI、nearest-native rank、**Pearson(energy,DI)**；平均 PCC：rsRNASP **0.64**、DFIRE-RNA **0.54**；最低能量 DI 平均 3.3、10.2 | 已有逐靶标数字，但系数和质量标签均不同，不能改名为 Spearman/RMSD |
| 同文 PM 正文与 Table S8 引用 | PM | 正文明确说 RMSD 评价在 **Table S8**，并给出全 PM 的 RMSD-energy scatterplots（Fig. S11） | 补充表未成功下载，不能声称已核到其中逐靶标 RMSD 数字或 Spearman |
| [cgRNASP，NAR Genomics and Bioinformatics 5(1) (2023), lqad016](https://doi.org/10.1093/nargab/lqad016)，正文 realistic-datasets 节 | **PM + RNA-Puzzles，20+22=42 靶标** | cgRNASP native top-1 **33/42**；mean DI 3.9，排除 native 后 12.7；PCC(energy,DI) **0.60**，rsRNASP 对应 **0.61** | 合并集不是 PM-only，更不是 Spearman/RMSD |
| 同文 Tables S2–S5、Figs S4–S6 引用 | 含 PM | S2 为 native 识别，S3/S4 为最低能量 DI，S5 为 PCC；正文另指向 RMSD-energy 图 | 只核实了正文引用的表意，未读到补充表，PM-only 行待补 |

**文献纠正：cgRNASP 这篇 2023 原文并非 PLoS Computational Biology，而是 NAR Genomics and Bioinformatics。**

rsRNASP Table 3 的 20×5 方法原文数字已转录至 `../tables/external_scoring_published_pm.csv`，保留 `pearson_energy_DI` 等原指标名；从表内四舍五入的逐靶标值再求均值可与论文汇总末位稍有不同。不是逐候选原始能量，无法由此恢复 Spearman。

补充材料的实际入口已定位：[rsRNASP mmc1.pdf](https://pmc.ncbi.nlm.nih.gov/articles/instance/8758408/bin/mmc1.pdf)、[cgRNASP supplemental_file.pdf](https://pmc.ncbi.nlm.nih.gov/articles/instance/9985339/bin/lqad016_supplemental_file.pdf)。本环境请求返回 HTML 的 `Preparing to download ...` 页面，而非 PDF；远程文本提取也返回 `No content retrieved.`，因此**补充材料数值核对未完成**，没有把未读取表格当成“没有 Spearman”的证据。两个原文正文已读取；没有发现可直接复用为 20/826 口径的逐候选能量表。

## 五靶标 CPU 试算

按公开 PM 靶标名排序取前五个：`1Z43,3A3A,3IVN,3L0U,3LA5`，各 42 个结构，共 210 个；每方法均得到完整的 210 个有限能量。只运行主 cgRNASP（三珠 P/C4′/N 模型），没有运行 cgRNASP-C/PC 等变体。

| 方法 | 官方代码与固定提交 | 编译与运行 |
|---|---|---|
| rsRNASP | [Tan-group/rsRNASP](https://github.com/Tan-group/rsRNASP), `73803d275850ff4f0d0a81340ace15ae633e1c8f` | `gcc -O2 rsRNASP.c -lm -o rsRNASP`；逐 PDB 执行 `./rsRNASP input.pdb output.txt`，工作目录含 `data/` |
| cgRNASP | [Tan-group/cgRNASP](https://github.com/Tan-group/cgRNASP), `ff8c5a23727b725e636d0b52dead2cecdb624a87` | 在主 `cgRNASP/` 目录编译 `gcc -O2 cgRNASP.c -lm -o cgRNASP`；每靶标按文件名排序链接成 `1.pdb`…`42.pdb`，`./cgRNASP input_dir/ 42 output.txt`，无限栈，工作目录含该模型 `data/` |
| DFIRE-RNA | [tcgriffith/dfire_rna](https://github.com/tcgriffith/dfire_rna), `b6cb379d31de931f93e862311da35d0ed559f2d1` | `make`；设置 `DFIRE_RNA_HOME`，每靶标一次 `bin/DFIRE_RNA -l sorted_pdb_list.txt` |

原源码未修改。rsRNASP/cgRNASP 在输出有效分数后源码明确 `return 6`，试算据此接收退出码 6，并检查候选 ID、行数与有限值；没有把任意非零退出当成功。rsRNASP 输出以追加方式写文件，逐次读取最后一行并核对 ID，避免旧诊断行污染。DFIRE-RNA 返回 0。CPU 串行；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，编译时间、下载时间不计入评分耗时；计时包含各评分进程启动和势能表加载。

Spearman 为 `scipy.stats.spearmanr(energy,RMSD)`：能量与误差都越低越好，正值表示一致。主列含 native，另给 decoys-only 列。CSV 还记录最低能量**非 native**候选的 Top-1 RMSD（不是论文 Table 1 的 native 识别率）；这 15 组的最低 decoy 能量均无并列。

| 靶标 | n（含 native） | rsRNASP ρ | cgRNASP ρ | DFIRE-RNA ρ |
|---|---:|---:|---:|---:|
| 1Z43 | 42 | 0.572644 | 0.558383 | 0.447857 |
| 3A3A | 42 | 0.719958 | 0.644923 | 0.670043 |
| 3IVN | 42 | 0.649785 | 0.584150 | 0.640710 |
| 3L0U | 42 | 0.738271 | 0.685601 | 0.609594 |
| 3LA5 | 42 | 0.861762 | 0.760473 | 0.821084 |
| 逐靶标平均 | 210 | **0.708484** | **0.646706** | **0.637858** |

| 方法 | 210 结构实测秒 | 826 结构线性估算秒 | 估算分钟 |
|---|---:|---:|---:|
| rsRNASP | 100.001 | 393.338 | 6.56 |
| cgRNASP | 9.545 | 37.544 | 0.63 |
| DFIRE-RNA | 37.142 | 146.092 | 2.43 |

估算公式 `pilot_seconds * 826 / 210`，总和约 **9.62 分钟**；只表示本机、上述调用方式和相似长度下的量级。不同靶标长度、批次启动数会影响实际耗时，不能把估算当全量实测。五靶标合计评分 146.688 秒。当前势能版本不一定等于 2022/2023 论文版本，不能据与论文差异判定复现失败。

输出：`../tables/external_scoring_pm_pilot_per_target.csv`（15 行，含每靶标耗时）、`external_scoring_pm_pilot_scores.csv`（630 行，逐结构能量与 RMSD）、`external_scoring_pm_runtime_estimate.csv`。PM 文件树指纹为 SHA256 `3e9392f5f0a9b2daed59dd78bbed63e9407ab51f895658398aa3e32fa4d74449`，算法为按相对路径排序，依次哈希 `相对路径 UTF-8 + NUL + SHA256(文件内容)的32字节摘要`，覆盖 PM 目录内全部文件。

## 旧 RNAtive 分数：能复用在哪里

源根目录为另一个仓库的 `cspph-ws/RNA_QA_Benchmark_Project/data/processed/rnative_score_tables/`。该目录 README 明确区分 CASP15 与 randstr；使用 `decoys/unified-unique/*.csv`、`casp/unified-unique/*.csv`，而不是把同一数据的其他处理副本叠加计数。

| 存储表范围 | 靶标文件 | 行数（含 native） | 规范化 PM 模型 ID 交集 |
|---|---:|---:|---:|
| randstr，全体存储 CSV | 84 | 42,084（84×501） | 0 |
| CASP15，全体存储 CSV | 12 | 1,546 | 0 |

这些行有 `rsRNASP,cgRNASP,DFIRE` 等现成评分列，因此**在它们自己的行上**可直接评估，不需要再调用这些工具。逐样本表同时有原生 RNAtive 多个变体，不能把“RNAtive 的分数”本身改称 rsRNASP 分数。randstr 的质量列是 `RMSD-ALL`，CASP15 为 `RMSD`，不擅自假定与 PM 的 RMSD 原子选择一致。

核验先去 `.pdb`、忽略大小写比较模型 ID：96 张表无重复 ID，和 PM 的 826 个模型 ID 交集为 0。再比较前四位 PDB 靶标 ID：只有 **PM `1Z43` / randstr `1z43A`** 重合，后者还带链后缀。randstr 的模型为 `1z43A_M1`…`1z43A_M500`，不是 PM 的四种预测器模型。native 的 PDB 来源可能相同，但坐标/链选择是否相同未核实；没有证明任何 PM decoy 可以直接取用这些行的能量。不能把“一个 PDB 来源相同”当作共享诱饵池；跨命名体系结构同一性还需坐标哈希证明。

**过滤与错误边界：**README 的 `decoys/excluded.txt` 列出 19 个无碱基对靶标，这 19 个仍存在于上述存储 CSV 中。因此 84 靶标是存储覆盖量，不是 RNAtive 论文排除后的最终分析量。此次导出按全部存储 CSV 计算，明确保留这些靶标；`1zhoH`、`2i82H` 的 DFIRE 为常数，Spearman 未定义，CSV 留空，不填 0。其他已选方法/靶标相关性均有限。

`external_scoring_rnative_overlap.csv` 列出全部 96 个靶标、行数、native 数及交集；`external_scoring_rnative_per_target.csv` 为三个现成评分列的 288 行逐靶标相关性，分别按各表自己的 RMSD 标签计算，**不是 PM 结果，也不声称重现 RNAtive 论文筛选后的汇总**。旧工作区 `results/eval/summary_l1_train.csv` 的 rsRNASP=0.243145、cgRNASP=0.160862、DFIRE=0.208399、RNAtive=0.236158 均为旧 level-1 的 8 靶标汇总，同样不能当 PM 数字。

## 下载／存储账与尚未完成事项

| 对象 | 已取得体积 | 说明 |
|---|---:|---|
| FTDMP `Protein-DNA.tar.gz` | **1,668,339,414 B** | 仓库外，MD5 通过；唯一明确要求的大文件 |
| PM_decoy_set | **97,553,504 B** | 目录文件内容总量，含在 rsRNASP 克隆中，不重复计下载 |
| rsRNASP 源码仓库 | Git pack **82,192,972 B** | 克隆连带训练集；当前编译后目录占用 466,149,584 B，非网络传输量 |
| cgRNASP 源码仓库 | Git pack **77,811,201 B** | 克隆连带训练集；当前编译后目录占用 415,497,552 B，非网络传输量 |
| DFIRE-RNA 源码仓库 | Git pack **2,022,668 B** | 当前编译后目录占用 7,120,432 B，非网络传输量 |

Git pack 是本地取得的压缩对象文件大小，不含协议开销，不冒充精确网络流量；所有源码、原始数据、编译产物均留在仓库外。其余网络访问是论文 HTML/XML、Zenodo 元数据和未成功返回 PDF 的小型下载响应。RNAtive 分数直接读旧工作区，没有重新下载或打分。

未完成项只有：GSPH 的 PD 完整候选 manifest／VoroMQA JSONL、GSPH 冻结 PM 的输入哈希、两篇论文补充表的实际 PDF 数值。没有全量 PM 结果、GSPH OOF、FTDMP 同池性能或新训练模型；这些缺口不以文档声称、同数量或其他池现成分数替代。
