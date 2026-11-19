# 金属离子与结构水位点预测：任务设计（草案）

状态：设计与文献阶段，尚未下载数据、尚未运行任何实验。下文的数据规模是估计，需在小规模试点中核实。

## 0. 结论

1. **任务定义**：把结构中的金属离子和水全部删掉，只看大分子原子，恢复每个离子和结构水的三维位置（以及离子种类、配位原子）。**真值本身就是一个空间点**，而共同支撑问题的最优解 x\* 正是这个点：离子位点 = 若干配体相互作用片的共同见证点。这是共同见证点在定义上就必需、而不是"多加一组特征"的任务。
2. **先做哪一个**：主任务做蛋白中的 Zn²⁺ 及其他过渡金属（Mn、Fe、Co、Ni、Cu）位点；第二任务做桥连 ≥3 个极性原子的结构水；第三任务做 RNA 中内层配位的 Mg²⁺，与仓库已有的 RNA 流程衔接。Zn 先做的理由：配位几何最规整（四面体，Zn–N/O/S 约 2.0–2.3 Å）、晶体学标注最可靠、主要对照方法 Metal3D 就是在 Zn 上训练的，比较最直接。
3. **共同见证点为什么是必需的**：(a) 输出要求一个坐标，两两相容只能给出"这几个原子两两都够近"，给不出点；(b) 我们把"两两都相容但没有共同见证点"的配体组合单独列成难负例层，在这一层上两两规则的 AUROC 按构造等于 0.5，共同支撑可以把它们全部排除；(c) 在同一批复合物、同一路径复合形上换出生规则（`cs`、`pairwise`、`facetmax`、`edge`、`shuffled`，沿用仓库已有的对照），差值即为共同见证点的贡献。
4. **一个最小的数字例子**（离子—配体理想距离 d = 2.1 Å，容差 δ = 0.3 Å）：三个配体原子组成边长 4.6 Å 的正三角形时，任一对的共同支撑是 4.6/2 − 2.1 = 0.20 Å ≤ δ（两两都通过），三者的共同支撑是 4.6/√3 − 2.1 = 0.56 Å > δ（没有离子能同时配位三者）。四个原子组成边长 4.4 Å 的正四面体时，两两为 0.10 Å，四者为 0.59 Å。真实四面体 Zn 位点的配体间距约 3.35 Å，而 4.4–4.6 Å 的侧链原子间距在蛋白中非常常见，所以这类伪位点在数据中会大量出现——其比例是试点要测的第一个数。
5. **对现有方法的定位**：不宣称全面超过 Metal3D / AllMetal3D 这类深度学习方法；宣称的是 (i) 给出带数值证书的精确位点和配位集合；(ii) 在两两相容的难负例层上显著优于两两规则和只看局部原子环境的方法；(iii) 在总体指标上与专门方法可比。若试点发现难负例比例很低（见 §6 的停止规则），该任务就不能证明必需性，应改以结构水为主任务。

## 1. 与数学规范的对应

规范中的共同支撑为 `tau(P_1..P_k) = min_x max_i min_{c in P_i} clearance(x, c)`。离子任务只需改变"球"的半径，不改变数学对象：

- 配体相互作用片 P：同一配体基团中可互相替代的配位原子组成一个并集型片。例如 Asp/Glu 羧基 {OD1, OD2} / {OE1, OE2}（单齿或双齿），His 咪唑 {ND1, NE2}，Cys {SG}，主链羰基 {O}，磷酸 {OP1, OP2}，核糖/碱基上的 O/N。片的并集结构正好对应"哪个原子去配位都可以"的化学事实。
- 每个配体原子 a 的球半径取离子—配体元素的理想距离 d(M, a)（按 Harding 1999/2001 与 CheckMyMetal 的统计表），clearance(x, a) = |x − a| − d(M, a)。于是 τ ≤ δ 当且仅当存在一点同时位于所有配体的配位壳内；最优点 x\* 就是预测的离子位置，活跃原子（3D 中至多 4 个）就是预测的配位集合。
- 结构水：配体换成极性 N/O，d 取氢键 O···N/O 距离约 2.8 Å；片按供体/受体身份分组，与现有微接触的供体/受体约定一致。
- 排斥条件（预测点周围不能有非配体重原子太近）会让问题变成非凸的极小极大，不放进 τ 里，以免破坏第 1 项定理的适用条件；作为证书之后的独立过滤条件，并在消融中报告它单独的作用。
- 难负例的定义直接用规范中的量：两两界 `tau_pw ≤ δ` 而 `tau > δ`，即高阶支撑增量 `eta_k = tau − tau_pw` 把它推出了容差。

## 2. 数据集与筛选规则

### 2.1 金属离子（主任务）

- **来源**：PDB 实验结构；位点清单取自 MetalPDB（含最小功能位点 MFS 与配位几何），用 BioLiP2 的"生物学相关"标记交叉核对，用 CheckMyMetal 的规则做验证。
- **结构级筛选**：X 射线；分辨率 ≤ 2.0 Å；R_free ≤ 0.25；去掉含未建模关键残基（配位壳内有缺失原子）的链。
- **位点级筛选**：
  - 离子种类：Zn（主）；Mn、Fe、Co、Ni、Cu（扩展）；Mg、Ca 单列一层（水介导配位多，几何更松）；Na、K 不进入主评测（误标率高，Zheng 等人多次报告）。
  - 离子占有率 ≥ 0.9（或按 altloc 取最高占有率构象），离子 B 因子不超过其周围配体平均 B 的 2 倍。
  - CheckMyMetal 的配位数、键价和（valence）、向量和（vecsum）与几何项不为"离群"。
  - **至少 3 个大分子配位原子，来自至少 2 个残基**（否则不是多片共同见证问题）；1–2 个大分子配体、其余为水的位点单独放一层，不进主指标。
  - 排除辅因子决定几何的金属：血红素 Fe、Fe–S 簇、叶绿素 Mg、钴胺素 Co 等（这些位点由辅因子而非蛋白配体组合决定）。
  - 配位壳中有对称拷贝原子的位点（晶体接触）标记出来，主评测中排除，单独报告。
- **去冗余与划分**：MMseqs2 按 30% 序列一致性聚类，聚类作为分组交叉验证的组（与仓库现有分组 CV 一致）；另设时间外测集：只用对照方法训练截止日之后发布的结构（每个对照的截止日需逐一核实；AlphaFold 3 训练截止为 2021-09-30）。
- **apo 设定（更难，第二阶段）**：同一 UniProt 的无金属结构，与 holo 结构叠合后，把 holo 的离子位置作为真值，检验构象变化下的鲁棒性。

### 2.2 结构水（第二任务）

- X 射线；分辨率 ≤ 1.5 Å（严格集）或 ≤ 1.8 Å（扩展集）。
- 水 O 占有率 = 1；B 因子按 Carugo（1999）的方法对占有率与整体 B 做归一化后不超过阈值（阈值在试点中按分布定）。
- 至少 3 个极性伙伴原子（N/O，距离 2.5–3.3 Å），来自至少 2 个残基；在删去其他水后的溶剂可及面积约为 0（埋藏）。
- **保守水**（推荐作为最干净的真值）：同一 UniProt 至少 3 个独立高分辨率结构，叠合后 ≥ 50% 的结构在 1.0 Å 内都有这个水。
- 划分同上。

### 2.3 RNA 中的 Mg²⁺（第三任务）

- 按 Zheng 等（2015）对 RNA Mg²⁺ 位点的验证与分类，只取至少 2 个内层配位 RNA 原子（磷酸 OP1/OP2、N7、O6 等）的位点；与仓库 RNA 流程共享原子解析与片构建。

### 2.4 规模估计（需核实）

全部按 mmCIF 压缩文件估计：金属主任务去冗余后约数千个结构，约 3–5 GB；结构水严格集约 1–3 万个结构，约 5–10 GB。试点（几百个结构）约 1 GB 以内。任一阶段若总量超过 10 GB，先请示。

## 3. 候选点与真值标签

- 对删掉离子和水后的结构，枚举空间上相邻的配体片三元组与四元组（候选数用现有分支定界的原子间隔下界剪枝），对每个组合求共同支撑和见证点 x\*，并做数值证书。
- **正例**：x\* 与某个真离子（或真水）在一对一匹配后的距离 ≤ 1.0 Å（宽松标准 2.0 Å 同时报告）。
- **负例**：其余候选点。按以下维度分层报告：
  - 难负例层：`tau_pw ≤ δ` 且 `tau > δ`；
  - `eta_k` 分箱；
  - 配位数（3/4/5/6）、离子种类、holo/apo、是否有水介导配体。
- 预测的配位集合与真配位集合（距离 ≤ 理想距离 + 0.5 Å 的大分子原子）比较，作为次要真值。

## 4. 对照方法

按输入分三组（均为已发表方法，DOI 见 §7）：

| 组 | 方法 | 说明 |
|---|---|---|
| 结构、深度学习 | Metal3D / Metal1D；AllMetal3D | 3D 卷积预测离子密度，Metal3D 以 Zn 为主，AllMetal3D 扩展到常见金属并给出种类与几何；主要对手 |
| 结构、模板或几何 | MIB / MIB2；BioMetAll；CHED；FINDSITE-metal；PinMyMetal；AlphaFill | 模板片段比对、主链预组织几何、三联体几何规则、同源迁移；PinMyMetal 为混合学习，AlphaFill 从同源结构转移离子与辅因子 |
| 序列 | IonCom；LMetalSite；GPSite；M-Ionic；ZincBindPredict | 残基级预测，只能在残基级指标上比较 |
| 结构预测 | AlphaFold 3（给定序列与离子） | 共折叠给出离子坐标；注意训练截止日带来的泄漏 |
| 结构水 | GalaxyWater-CNN；HydraProt；SuperWater；Dowser | 3D CNN、深度学习、生成模型、基于能量的空腔水放置 |
| 本仓库内部 | 两两规则（所有配体对都 ≤ δ）；配体质心；`facetmax`、`edge`、`shuffled` 出生规则 | 同复合物同复合形，只换出生规则 |

其中"两两规则"和"质心"是必需性检验的核心对照：两两规则没有点，质心有点但不保证同时配位。

## 5. 评价指标

- **位点级（主指标，沿用 Metal3D 的口径以便直接比较）**：一对一匹配后，在 1.0 Å / 2.0 Å 距离阈值下的精度、召回、F1；真阳性的定位误差（中位数与均值）；按候选打分的 AUPRC。
- **残基级（与序列方法比较）**：配位残基的 MCC、AUPRC。
- **种类与配位**：离子种类准确率；配位集合的 Jaccard；配位数准确率。
- **结构水**：在每残基固定预测数下，0.5 / 1.0 / 1.5 Å 内的真水召回率（GalaxyWater-CNN、HydraProt、SuperWater 采用的口径）。
- **必需性指标（本任务的核心）**：
  1. 难负例占两两通过候选的比例（试点即可得到，不需训练）；
  2. 难负例层上的 AUROC：两两规则按构造为 0.5，报告共同支撑及各对照方法；
  3. 同复合形下 `cs` 与 `pairwise` / `facetmax` / `edge` / `shuffled` 的差值，20 次分组 CV，目标聚类自助法（≥ 2000 次）给区间（与仓库实验规则一致）；
  4. 质心与 x\* 的定位误差之差。

## 6. 试点与停止规则（下载前确定）

1. 试点：约 300 个 Zn 结构 + 约 300 个高分辨率结构水结构（约 1 GB）。只跑候选枚举与共同支撑，不训练。
2. 记录：难负例占两两通过候选的比例；真位点中 `eta_k > 0` 的比例；证书通过率。
3. 停止规则：难负例比例 ≥ 20%（δ = 0.3 Å）则必需性成立，进入全量；5%–20% 之间，加入四元组与并集片后再测；< 5% 则离子任务只作为定位精度的展示，主任务改为结构水。
4. 仅凭几何（不训练）在 Zn 上的召回率与 Metal3D 报告值对比，作为是否需要学习打分器的依据。

## 7. 文献（DOI 已逐条核对）

- Dürr, Levy, Rothlisberger. Metal3D: a general deep learning framework for accurate metal ion location prediction in proteins. Nat Commun 2023. 10.1038/s41467-023-37870-6
- Dürr et al. AllMetal3D: joint prediction of localization, identity and coordination geometry of common metal ions in proteins. bioRxiv 2025. 10.1101/2025.02.05.636627
- Lin et al. MIB: Metal Ion-Binding Site Prediction and Docking Server. J Chem Inf Model 2016. 10.1021/acs.jcim.6b00407
- Lu et al. MIB2: metal ion-binding site prediction and modeling server. Bioinformatics 2022. 10.1093/bioinformatics/btac534
- Sánchez-Aparicio et al. BioMetAll: Identifying Metal-Binding Sites in Proteins from Backbone Preorganization. J Chem Inf Model. 10.1021/acs.jcim.0c00827
- Babor et al. Prediction of transition metal-binding sites from apo protein structures (CHED). Proteins 2007. 10.1002/prot.21587
- Brylinski, Skolnick. FINDSITE-metal. Proteins 2010. 10.1002/prot.22913
- Zhang et al. PinMyMetal: a hybrid learning system to accurately model transition metal binding sites in macromolecules. Nat Commun 2025. 10.1038/s41467-025-57637-5
- Hekkelman et al. AlphaFill: enriching AlphaFold models with ligands and cofactors. Nat Methods 2022. 10.1038/s41592-022-01685-y
- Abramson et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature 2024. 10.1038/s41586-024-07487-w
- Hu et al. Recognizing metal and acid radical ion-binding sites by integrating ab initio modeling with template-based transferals (IonCom). Bioinformatics 2016. 10.1093/bioinformatics/btw396
- Yuan et al. Alignment-free metal ion-binding site prediction from protein sequence through pretrained language model and multi-task learning (LMetalSite). Brief Bioinform 2022. 10.1093/bib/bbac444
- Yuan et al. Genome-scale annotation of protein binding sites via language model and geometric deep learning (GPSite). eLife 2024. 10.7554/eLife.93695
- Shenoy et al. M-Ionic: Prediction of metal ion binding sites from sequence using residue embeddings. bioRxiv 2023. 10.1101/2023.04.06.535847
- Ireland, Martin. ZincBind—the database of zinc binding sites. Database 2019. 10.1093/database/baz006
- Ireland, Martin. ZincBindPredict—Prediction of Zinc Binding Sites in Proteins. Molecules 2021. 10.3390/molecules26040966
- Putignano et al. MetalPDB in 2018. Nucleic Acids Res 2018. 10.1093/nar/gkx989
- Zhang et al. BioLiP2: an updated structure database for biologically relevant ligand–protein interactions. Nucleic Acids Res 2024. 10.1093/nar/gkad630
- Zheng et al. Validation of metal-binding sites in macromolecular structures with the CheckMyMetal web server. Nat Protoc 2014. 10.1038/nprot.2013.172；CheckMyMetal: a macromolecular metal-binding validation tool. Acta Cryst D 2017. 10.1107/s2059798317001061
- Zheng et al. Magnesium-binding architectures in RNA crystal structures. Nucleic Acids Res 2015. 10.1093/nar/gkv225
- Philips et al. MetalionRNA: computational predictor of metal-binding sites in RNA structures. Bioinformatics 2012. 10.1093/bioinformatics/btr636
- Harding. The geometry of metal–ligand interactions relevant to proteins. Acta Cryst D 1999. 10.1107/s0907444999007374；同作者 II（2000）10.1107/s0907444900005849；Na/K（2002）10.1107/s0907444902003712
- Park, Seok. GalaxyWater-CNN. J Chem Inf Model 2022. 10.1021/acs.jcim.2c00306
- Zamanos et al. HydraProt. J Chem Inf Model 2024. 10.1021/acs.jcim.3c01559
- Kuang et al. SuperWater as a generative AI framework to predict water molecule positions on protein structures. Commun Chem 2025. 10.1038/s42004-025-01789-4
- Zhang, Hermans. Hydrophilicity of cavities in proteins (Dowser). Proteins 1996. 10.1002/(sici)1097-0134(199604)24:4<433::aid-prot3>3.0.co;2-f
- Carugo. Correlation between occupancy and B factor of water molecules in protein crystal structures. Protein Eng 1999. 10.1093/protein/12.12.1021
- Steinegger, Söding. MMseqs2. Nat Biotechnol 2017. 10.1038/nbt.3988

## 8. 风险

- 单次选定配体后的共同支撑就是加权最小包围球问题，审稿人会指出几何探针类方法（BioMetAll、CHED）也在做"找一个同时靠近几个配体的点"。区别必须落在：并集型片、按阶数的高阶增量 `eta_k`、持久结构，以及带证书的精确解；论文中要正面讨论。
- 晶体学离子与水的标注本身有误标（尤其 Na/K/Mg 与水互相混淆），所以主评测限于过渡金属与严格筛选的水。
- 对照方法的训练集大多包含 PDB 中的这些结构，时间外测集是必需的。
