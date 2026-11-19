# 外部基线对比方案（待确认后执行）

本文只讨论**别人已发表的方法**。仓库里的两两支撑、边权滤过、两两规则、配体质心、纯物理特征模型等是本方法的消融或对照实验，见 [BASELINES.md](BASELINES.md)，不算基线。外部方法的选题依据见 [EXTERNAL_COMPARISONS.md](EXTERNAL_COMPARISONS.md)，离子位点细节见 [ION_WATER_SITE_TASK_DESIGN.md](ION_WATER_SITE_TASK_DESIGN.md)。

## 结论

1. 现在仓库里的外部对比之所以乱，原因有四个：文献数字和自己跑的数字混在一起；候选池和切分不一致（47 与 57 靶标，826/806/970 个结构，9,400 候选子集与 400/靶标全集）；只有文档里的数字，没有逐候选分数文件（rsRNASP 约 0.58、FTDMP、VoroMQA）；方法注册了却从没跑通（HADDOCK、pyDockDNA、3dRPC、ITScore 等均为 `implemented:false`）。
2. 统一规则：外部方法在**我们的候选池上零样本运行官方程序**，逐候选分数入库；本方法用按靶标分组交叉验证的折外预测；在同一组候选上计算同一组逐靶标指标，用靶标层面的配对 bootstrap 比较。跑不了的方法写明原因，不进正文主表。
3. 正文主表共 **13 个外部方法**，分三类任务：
   - 拓扑表示：Čech/α 复形、Rips、持久路径同调、持久路径拉普拉斯、元素特异持久同调。
   - 核酸打分：rsRNASP、cgRNASP、DFIRE-RNA、FTDMP/VoroMQA、HADDOCK 打分。
   - 离子位点：Metal3D、AllMetal3D、PinMyMetal。
4. 需要下载的东西合计约 **2.5 GB**，不到 10 GB 的上限。按每个方法单条结构用时估算（未实测），全部外部方法约需 **100–150 CPU 核时**，另加 AllMetal3D 的 GPU 时间。
5. 开跑前有两个前提，需要超算会话提供：
   - 本方法在 FTDMP 蛋白-DNA 上实际用的候选清单（每个条件 9,400 个）和逐候选特征表或折外预测。
   - rsRNASP PM 诱饵集（826 个结构的冻结版本）。
   这两样在云端都没有。

## 1. 现有外部对比为什么乱

| 问题 | 例子 | 后果 |
|---|---|---|
| 文献数字和自跑数字混用 | rsRNASP "约 0.58" 只写在报告文字里；膜蛋白工具用的是 mCSM-membrane 论文附表里的预测 | 不能确认是同一批结构、同一个指标 |
| 候选池不一 | FTDMP：本方法用每个条件 9,400 个候选的子集，官方全集为每靶标 400 个；旧工作区是 57 靶标、每靶标 999 个刚体扰动候选；RNA 有 826/806/970 三种结构数 | 数字放进同一张表就失真 |
| 切分不一 | 外部方法是零样本；本方法是按靶标 5 折；膜蛋白有留一蛋白和 5 折两种口径 | 需要明确"外部零样本对本方法折外" |
| 没有分数文件 | FTDMP 官方排序、VoroMQA、rsRNASP、cgRNASP、DFIRE-RNA、3dRNAscore | 无法复核 |
| 注册了但没跑通 | HADDOCK、pyDockDNA、3dRPC、ITScore-PR/PP、DDNA 类在注册表里为 `implemented:false`；旧排行榜多数靶标只有 0 或 1 个候选 | 旧排行榜整体作废 |
| 版本漂移 | rsRNASP 2024-07-23、cgRNASP 2024-08-08 都改过势能表 | 必须固定代码版本并写进记录 |

## 2. 统一协议

| 项目 | 规定 |
|---|---|
| 数据 | 拓扑表示：16 个审计结构（8 个蛋白-DNA、8 个 RNA），与定理检验用同一批。打分：FTDMP 蛋白-DNA 47 靶标，bound 和 unbound 各 9,400 个候选；rsRNASP PM 诱饵集，20 个靶标、826 个结构。离子位点：首次发布日不早于 2024-05-01 的时间外测集 |
| 切分 | 外部方法零样本，不调参。本方法用按靶标分组的 5 折折外预测，划分与 EXP-06 相同；离子位点用时间切分，另做 30% 序列同源去重 |
| 指标 | 打分：逐靶标 Spearman、Top-1 成功率、难负例层的成对正确率。拓扑：条形码是否相同、瓶颈距离、区间数差。离子位点：一对一匹配下 1 Å 和 2 Å 的精确率、召回率、F1，定位误差，以及难负例层 AUROC |
| 统计 | 以靶标为单位做配对 bootstrap，重采样 ≥2,000 次；离子位点按靶标簇做 bootstrap |
| 记录 | 每个方法保存代码版本或提交号、命令、逐候选分数 CSV、失败条目清单；失败条目在比较中计为缺失，并报告覆盖率 |

## 3. 外部方法逐项可行性

核时为单条结构用时乘以候选数的估算，**未实测**；先各跑 50 条校准再报实际核时。

### 3.1 拓扑表示（16 个审计结构，同一点集或同一有向图）

| 方法 | 论证职责 | 代码与许可 | 依赖 | 下载 | 估计核时 | 能否重跑 |
|---|---|---|---|---|---|---|
| Čech / α 复形持久同调 | 定位新意：零半径点支撑下共同支撑等于 Čech 出生值；真实 patch 上共同支撑是加厚并集支撑集族的 Čech 型神经出生值，两两支撑是其 Rips 型对应 | GUDHI，MIT/GPL | pip | <50 MB | <1 | 能 |
| Vietoris–Rips 持久同调 | 标准两两做法会丢掉什么 | Ripser，MIT | pip | <10 MB | <1 | 能 |
| 持久路径同调（Chowdhury–Mémoli） | 直接前身：同一复形只换出生规则 | 官方 pph-matlab（BSD-3，需 MATLAB）；Wei 组 PathHom（MIT，Python） | Python | 小 | 1–5（路径数随阶增长） | 能，用 PathHom，并与仓库的边权滤过交叉核对 |
| 持久路径拉普拉斯 | 回应"和 Wei 组路径方法有何不同"：它的滤过仍是两两距离型 | 官方 HERMES 需 MATLAB 和 CGAL；PathHom 可能覆盖 | — | 小 | 1–5 | 能，优先用 PathHom；不行就按 Wang & Wei 2023 的定义复现并标注"复现" |
| 元素特异持久同调（Cang–Wei） | 分子打分里最常被引用的拓扑特征 | TopologyNet/TopBio，MIT（依赖 JavaPlex 和 R-TDA） | Java、R | 小 | 5–10（打分池上提特征） | 能；放进同一排序器、同一折，作为特征组比较 |
| 超图持久上同调 | 高阶相互作用已有超图表示 | 无通用发布代码 | — | — | — | 只给理论反例，不进数值表 |

### 3.2 核酸对接候选打分

| 方法 | 任务 | 论证职责 | 代码与许可 | 依赖 | 下载 | 估计核时 | 能否重跑 |
|---|---|---|---|---|---|---|---|
| rsRNASP | RNA PM | 该诱饵集的原始发布者，最重要的参照 | GitHub，GPL-3.0 | gcc | <50 MB | <1 | 能；固定 2024-07 前后两个版本中与原文一致的那个 |
| cgRNASP | RNA PM | 粗粒化统计势代表 | GitHub，GPL-3.0 | gcc | <50 MB | <1 | 能 |
| DFIRE-RNA | RNA PM | 经典统计势 | GitHub，许可未写明 | make | <50 MB | <1 | 能（仓库已有修补脚本） |
| 3dRNAscore | RNA PM | 经典统计势 | 官网登录后才能下载 | Perl | 小 | <1 | 需要注册账号；拿不到就只放补充说明 |
| ARES | RNA PM | 深度学习代表 | Zenodo，CC-BY-NC-4.0 | Torch 1.5 | 7 MB | 1 | **不能**：公开的只有示例 dummy 权重。改用 lociPARSE（GPL-3.0，自带权重 8 MB，可 CPU 运行）作深度学习参照 |
| FTDMP 官方排序 / VoroMQA | 蛋白-DNA | 数据集来自 FTDMP，最重要的参照 | Voronota 与 FTDMP，MIT | 静态二进制 / conda | 候选结构包 1.7 GB | 约 5（每个候选约 1 秒，18,800 个） | 能；需要本方法的候选清单 |
| HADDOCK 打分 | 蛋白-DNA | 核酸对接最常用的物理打分 | HADDOCK3，Apache-2.0；注册表里写的是 2.4 版 | pip、CNS | 15 MB | 约 25–50 | 能，用 HADDOCK3 的 emscoring 模块并写明版本 |
| DDNA3 | 蛋白-DNA | 蛋白-DNA 统计势 | 来自 Zhou 实验室，可获得性待查 | — | 小 | <5 | 待查；拿不到就不进主表 |
| ITScore-PR、3dRPC | 蛋白-RNA | 蛋白-RNA 打分代表 | 可获得性待查 | — | — | — | 暂不进主表：蛋白-RNA 候选池只在旧工作区，要先重建 |

### 3.3 离子位点（时间外测集）

| 方法 | 论证职责 | 代码与许可 | 依赖 | 下载 | 估计核时 | 能否重跑 |
|---|---|---|---|---|---|---|
| Metal3D / Metal1D | 主要结构对手；"可比并带证书" | GitHub，代码 MIT，权重 CC-BY-4.0 | conda | 14 MB | 约 10–20（体素网络在 CPU 上） | 能；加载权重时需加 `map_location` 才能用 CPU，这只改设备、不改模型，写明 |
| AllMetal3D | 泛金属定位加种类 | MIT | 必须 CUDA | 约 0.4 GB | GPU 数小时 | 需要 GPU；超算如果是非 CUDA 加速卡，兼容性待查 |
| PinMyMetal | 混合学习与几何定位；训练用 2024-04 快照 | MIT | Docker 和 PostgreSQL | 待查 | 待查 | 可能能跑；若超算不允许 Docker，放补充材料 |
| 其余（MIB2、BioMetAll、LMetalSite、GPSite、M-Ionic、ZincBindPredict 等） | 只做残基级比较，或训练边界不明 | — | — | — | — | 放补充材料 |
| 结构水：SuperWater、HydraProt | 结构水对手 | MIT | SuperWater 不支持 CPU，另需约 2.5 GB 的 ESM-2 | — | — | 不进正文；如需比较放补充材料 |

时间泄漏：Metal3D 和 AllMetal3D 的训练数据截至 2021-03-05，PinMyMetal 截至 2024-04，都早于 2024-05-01，所以对这三个方法时间外测集成立。但它们的训练清单不全，序列同源泄漏只能靠 30% 去重来控制，正文照此说明。

## 4. 正文主表与补充材料

**正文主表**（同数据、同切分、同指标，外部方法由我们自己跑）：

- 拓扑表示：Čech/α、Rips、持久路径同调、持久路径拉普拉斯。比的是条形码差异和反例，不是比赛。
- 元素特异持久同调：作为特征组进入同一排序器。
- RNA：rsRNASP、cgRNASP、DFIRE-RNA。
- 蛋白-DNA：FTDMP/VoroMQA、HADDOCK3 打分。
- 离子位点：Metal3D、AllMetal3D、PinMyMetal。

**补充材料**：lociPARSE、3dRNAscore（若拿到）、DDNA3（若拿到）、残基级离子方法、结构水方法。

**不出现**：ARES（没有公开权重）；旧排行榜；膜蛋白工具（论文附表导入，并且与主线无关）。

口径沿用 EXTERNAL_COMPARISONS.md：整体 Spearman 和离子位点总体精度如实列出，不当成比赛；结论放在同图同路径的条形码差异、精确反例、证书可复核和难负例层。

## 5. 执行顺序

1. 在云端装工具并各用 50 条做冒烟和计时：GUDHI、Ripser、PathHom、rsRNASP、cgRNASP、DFIRE-RNA、Voronota、HADDOCK3、Metal3D、lociPARSE。
2. 拓扑表示比较在 16 个审计结构上直接开跑（不需要超算数据）。
3. 下载 FTDMP 候选结构包（1.7 GB）。拿到本方法的候选清单后，按清单取同一子集，跑 VoroMQA、FTDMP 和 HADDOCK3。
4. 拿到 PM 诱饵集后跑 RNA 统计势。
5. 离子位点在时间外测集定稿后跑 Metal3D 和 PinMyMetal；AllMetal3D 交给有 GPU 的节点。
6. 每个方法产出逐候选分数 CSV，统一评估脚本出表，写回本文档。

## 6. 执行状态（按"现成能用就不重跑"收窄后）

| 方法 | 状态 | 结果 | 出处 |
|---|---|---|---|
| Čech / α、Rips | 已完成：在控制结构上计算，原归档结构做汇总 | 零半径 τ 与精确 miniball 半径的最大差为 1.1×10⁻¹⁴ Å；patch 层无向 τ 与两两支撑的条形码，RNA 有 12/24 个"结构×阶"不同，蛋白-DNA 有 2/21 个不同 | [cech_rips_compare.md](../results/reports/cech_rips_compare.md) |
| 持久路径同调（Chowdhury–Mémoli） | 直接用仓库已有的 `edgew` 结果 | 尺度是共价步数，只作参照 | 同上 |
| rsRNASP / cgRNASP / DFIRE-RNA | 已在 PM 全量上重跑（806 个 decoy、20 个靶标），共 8 分钟 CPU | 宏平均 Spearman 分别为 0.580 / 0.560 / 0.518；本方法 0.550，差异不显著 | [external_scoring_inventory.md](../results/reports/external_scoring_inventory.md) |
| FTDMP 官方排序 | 已下载官方逐候选排序（每个条件 9,400 个候选） | 等本方法的候选清单到了再做同池对齐 | 同上 |
| VoroMQA | 官方包里没有它的分数 | 等超算找到原分数文件；找不到再在子集上补跑 | 同上 |
| Metal3D | 在 32 个合格的 Zn 时间外结构上推理中（只把设备改为 CPU） | 69 个 Zn 位点作参考，但外部质控只做了一部分 | [external_metal_site.md](../results/reports/external_metal_site.md) |
| AllMetal3D | 移出主表：CPU 上单个小结构 15 分钟也跑不完，按成本决定不租 GPU | 补充材料引用原文数字，注明是序列隔离测试集、训练截至 2021-03-05 | — |
| 持久路径拉普拉斯、PinMyMetal、HADDOCK3、元素特异持久同调 | 暂缓 | — | — |
| 3dRNAscore、DDNA3、ITScore-PR、3dRPC | 需要注册账号或代码来源不明，已列给 Robin | — | — |
| ARES | 公开的只有示例 dummy 权重 | 不进入比较 | — |
