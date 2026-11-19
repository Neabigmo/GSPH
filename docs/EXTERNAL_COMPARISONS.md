# 外部对比方法清单（选题视角）

本文补充 [BASELINES.md](BASELINES.md)：那里登记的是仓库里已有的对照及其口径，这里列出审稿人会要求、我们也应当主动比较的**已发表方法**，每一个写明论证职责。离子与结构水位点的对照沿用 [ION_WATER_SITE_TASK_DESIGN.md](ION_WATER_SITE_TASK_DESIGN.md)，这里不重复。

## 结论（交建模实验官）

1. **最要紧的对照是 Čech 与 Vietoris–Rips**。共同支撑不是 patch 的最小包围球半径：patch 的支撑集是若干接触透镜的并集，一般非凸。准确说法是，共同支撑是加厚支撑集族 {S_P(t)} 的 Čech 型神经出生值，两两支撑是它对应的 Rips 型（flag）规则；普通 Čech 是原子取零半径点、每个 patch 只有一个点时的特例（数值核对最大差 1.1e-14 Å，见 `results/reports/cech_rips_compare.md`）。审稿人第一句会问“这是不是路径复形上的 Čech 对 Rips”，所以必须在同一批点上跑 Čech/α 复形持久同调和 Rips 持久同调，说清楚新意在“有向路径复形上的 Čech 型出生规则”，而不是“高阶胜过两两”本身。
2. **拓扑表示类**再比四个：持久路径同调（Chowdhury–Mémoli），持久路径拉普拉斯，元素特异持久同调（Cang–Wei），超图持久上同调；单纯复形神经网络只放讨论。
3. **核酸打分**：RNA 诱饵用 rsRNASP、cgRNASP、3dRNAscore、DFIRE-RNA、ARES；蛋白-DNA 用 FTDMP/VoroMQA、HADDOCK 打分、DDNA3；蛋白-RNA 用 ITScore-PR、3dRPC 打分。
4. **必赢口径**：同图同路径的条形码差异、难负例层、证书精确性和反例。**不设比赛**：整体 Spearman、离子位点总体精度、拓扑对物理的增量。

## 1. 拓扑或几何分子表示

| 方法 | 文献 | 论证职责 | 比较方式 |
|---|---|---|---|
| Čech / α 复形持久同调 | Edelsbrunner & Mücke, *ACM TOG* 1994（α 形）；Čech–Rips 夹逼见 de Silva & Ghrist, *AGT* 2007 | 定位新意：零半径点支撑下共同支撑退化为 Čech 出生值；真实 patch 上差别来自并集型非凸支撑集和有向路径复形 | 同一批原子坐标；报告零半径点退化情形下与 Čech/α 条形码一致（一致性检查）；真实 patch 上共同支撑与两两规则的条形码差异（RNA 12/24、蛋白-DNA 2/21 个“结构×同调阶”不同） |
| Vietoris–Rips 持久同调 | 同上；实现用 Ripser（Bauer, *J Appl Comput Topol* 2021） | 说明两两相容的标准做法会丢掉什么；与仓库 `pairwise` 对照互为印证 | 同点同尺度，瓶颈距离、区间数差 |
| 持久路径同调（GLMY 路径同调 + 边权距离滤过） | Grigor'yan, Lin, Muranov & Yau, *Pure Appl Math Q* 2012；Chowdhury & Mémoli, SODA 2018 | 我们的直接前身：同一复形只换出生规则，差值即共同见证点的贡献 | 即仓库 `edgew`，须注明出处为 Chowdhury–Mémoli |
| 持久路径拉普拉斯 / 分子路径拓扑 | Wang & Wei, *Found Data Sci* 2023；Chen, Liu, Wu, Wei, Pan & Yau, *J Phys Chem Lett* 2023 | 最近的同类分子路径表示；审稿人会问“和 Wei 组路径方法有何不同” | 同复形同滤过输出谐波谱与零特征值个数，说明其滤过仍是两两距离型 |
| 元素特异持久同调（TopologyNet 一系） | Cang & Wei, *PLoS Comput Biol* 2017；Cang, Mu & Wei, *PLoS Comput Biol* 2018 | 分子打分里最常被引用的拓扑特征基线 | 作为特征组放入同一排序器、同折（旧工作区已有“普通持久同调”特征组，需按其元素分组重做） |
| 超图持久上同调、超图嵌入同调 | Liu, Wang, Wu & Xia, *Brief Bioinform* 2021；Bressan, Li, Ren & Wu, *Asian J Math* 2019 | 回应“高阶相互作用已有超图表示”：超图超边由组合规则给出，没有共同见证点 | 理论上对比定义；给一个超图不变量相同而共同支撑不同的反例 |
| Dowker / 见证复形 | de Silva & Carlsson, SPBG 2004；Chowdhury & Mémoli, “A functorial Dowker theorem”, 2018 | 名称撞车：“witness”在 TDA 里另有所指，须说明区别 | 只在相关工作中界定，不做数值比较 |
| 单纯复形 / CW 消息传递网络 | Bodnar et al., ICML 2021；Bodnar et al., NeurIPS 2021 | 高阶表示学习的代表 | 只放讨论；不训练比赛 |

## 2. 核酸相关的对接候选打分

| 任务 | 公认方法 | 文献 | 备注 |
|---|---|---|---|
| RNA 结构诱饵（PM 诱饵集） | rsRNASP、cgRNASP | Tan et al., *Biophys J* 2022；Tan et al., *PLoS Comput Biol* 2023 | rsRNASP 是该诱饵集的原始发布者，必须同池入库逐样本分数 |
| | 3dRNAscore、DFIRE-RNA、RASP | Wang et al., *NAR* 2015；Zhang et al., *J Comput Biol* 2020；Capriotti et al., *Bioinformatics* 2011 | 经典统计势 |
| | ARES | Townshend et al., *Science* 2021 | 深度学习代表；只作参照，不比赛 |
| 蛋白-DNA | FTDMP 排序与 VoroMQA | Olechnovič & Venclovas, *Proteins* 2017 及其 CASP15 流程 | 数据集即来自 FTDMP，必须同池 |
| | HADDOCK 打分 | van Dijk & Bonvin, *NAR* 2010 | 核酸对接最常用的物理打分 |
| | DDNA3 | Zhao, Yang & Zhou, *Bioinformatics* 2010 | 蛋白-DNA 统计势 |
| 蛋白-RNA | ITScore-PR、3dRPC 打分 | Huang & Zou, *NAR* 2014；Huang, Li & Xiao, *Bioinformatics* 2016 | 旧工作区蛋白-RNA 池使用；HDOCK（Yan et al., *NAR* 2017）可作对接来源说明 |

全部采用：同一候选池、零样本运行官方程序、入库逐候选分数文件，再和本方法同指标比较；只有文献数字的不进正文（与 BASELINES.md 一致）。

## 3. 必赢口径与不设比赛的比较

**必赢（由定理或构造保证，报告的是频率和大小）**

- 同图同路径换出生规则：共同支撑不小于两两支撑，真实结构上条形码不同的比例（RNA 14/16，蛋白-DNA 4/16）和出生值差。
- 精确反例：两两支撑相同而三重支撑不同（1.426 vs 3.521 Å）；对超图、Rips 同样给反例。
- 证书化求解：每个选定接触组合输出加权最小包围球（至多四个活跃球）的证书，并集 patch 的支撑由对组合的取最小得到，可逐 patch 复核；数值近似（随机采样或迭代优化）无此保证。
- 离子位点难负例层：两两规则按构造 AUROC=0.5，共同见证点可排除。

写法上要坦白这些优势来自定义，价值在于“真实分子里有多常见”，不要包装成性能胜利。

**不设比赛**

- 整体 Spearman：RNA 0.550 低于 rsRNASP 约 0.58，蛋白-DNA 0.19–0.24；如实列出，作外部参照。
- 拓扑叠加物理的增量约为零：只报告难对 +0.018 这一稳定信号，表述为“补充信息”而非“更优”。
- 离子位点总体精度对 Metal3D：表述为“可比并带证书”。
- 与 Čech 在零半径点支撑下的比较：两者按定义相同，是一致性检查，不是比赛。
- 与深度学习（ARES、单纯复形网络）：训练数据和规模不同，只作参照。
