# 论文提纲与证据对应表

**工作标题：共同见证点让分子路径的高阶几何进入条形码**

**头条主张：两两相容不等于共享一个空间点；把共同见证点作为路径出生的条件，能辨认两两滤过看不见的高阶几何。** 论文定位为数学表示论文。证书回答“这个点能否复核”，仿射 Jung 界回答“两种出生值相差多远”，分子条形码回答“这个区别是否进入同调”，Zn 与结构水回答“这个点在应用中代表什么”。候选打分不是主卖点。

## 先定三幅主图

| 图 | 标题与图注要说的结论 | 画法与证据边界 |
|---|---|---|
| 图 1 | **三对都相交，三片却没有共同点。** 在同一尺度，两两规则填上三角形，共同支撑留下空心三角形。 | 左侧画三个空间中的两两见证点，并用统一颜色标识各片；右侧并列画三片的神经与其旗复形。让“少一张面”一眼可见；明确这是支撑集族的神经示意，不是 GLMY 条形码。 |
| 图 2 | **审计队列的每条路径都落在 √(3/2) 界内。** | RNA 与蛋白-DNA 共用横轴和同一条理论界线，按图类型与路径阶分面；注明样本数，以分位数或基于逐路径数据的分布展示，不把分位数还原成密度。 |
| 图 3 | **换出生规则，RNA 和蛋白-DNA 的高阶条形码都改变。** 在这批结构中，RNA 的 H1、H2 差异更一致，H0 在两组均不变。 | 对照案例用相同的 Å 横轴、共同与两两规则固定配色，以 H1/H2 配对；旁列两组、两种图的全队列差异计数。区间数和有限总持续度分别标注，无穷区间另记；不把案例的差幅当成队列结论。 |

## 逐节提纲

| 节 | 节标题与论点 | 定理、图、表 |
|---|---|---|
| 摘要与 1 引言 | **为什么两两相容还不够。** 从图 1 切入：每对片可由不同的点见证，而一个路径要在同一位置满足所有片。随后指向图 2 的界、图 3 的同图差异和位点坐标应用，不以排名立论。 | 图 1 先于定理叙述；摘要按“反例—界—条形码—位点”组织，不写成工作过程。 |
| 2 表示与证书 | **一个共同点给路径出生值一个可核验的几何含义。** 从有向供体–受体微接触的透镜，定义相互作用片（patch）的备选透镜并集；共同支撑 `τ` 是各片首次有共同交点的增厚尺度。固定每片的一个微接触，求加权 1-center，再用分支定界在所有选择中取最小；输出见证点 `x*` 与数值证书。区分三维至多四个活跃球的证书，以及单接触片才有的四片 Helly 精确性。 | 选择公式（引理 1）、Helly 与选择–Helly（定理 1）；表 1：零半径单点支撑恢复普通 Čech 的对照，列出独立最小包围球与 α 复形核验，并将点云 Rips 区别限于控制结构。这里的 Čech 型／Rips 型指支撑集族的神经出生与其旗复形；不以神经定理推断非凸支撑并集的拓扑。 |
| 3 反例与界 | **两两支撑不能恢复共同支撑，但差距有几何上界。** 等边布置三个位置，每片占其中两个：两两共享位置，三片的共同出生值为 `D/2 − r > 0`，两两出生值为 0。再用等半径正四面体说明单接触片也会出现缺口并达到界。先对每片求备选接触的同角色端点之间的最大距离，再在路径各片上取最大值得片内展宽 `δ`；用路径上的半径极值，仿射 Jung 界约束 `(r_min + τ)/(r_max + τ_pw + δ) ≤ √(3/2)`，不是 `τ/τ_pw` 的界。 | 图 1；并集反例（命题 2）、仿射 Jung 界（定理 3）；图 2 用原 16 个结构、两种图、0–3 阶的全审计路径检验，并标注单接触子层，说明现象不依赖片内备选。 |
| 4 分子条形码 | **只换出生条件，高阶条形码就改变。** 固定 patch 图与允许路径，用 `τ` 或 `τ_pw` 生成 GLMY 路径滤过；交错结论给出模块的受控比较，不意味着区间相同。两种规则在联合出生事件值上按 Å 计算 H0–H2，不用三尺度粗网格或事件序号替代。 | 交错（定理 4）；图 3。以原队列的 8 个 RNA 诱饵（clash 比 0.6）和 8 个 FTDMP 蛋白-DNA 诱饵比较；有向、对称化分别列“结构×同调维”的区间多重集差异。RNA 的改变更一致，不把差异归因于片展宽，也不声称每个 RNA 案例的差幅都更大。 |
| 5 位点应用 | **配体共享的点就是要定位的位点。** 隐去待定位的 Zn 或结构水，用大分子配位片求 `x*`，与实验坐标比较。在各 300 个结构的覆盖评估中，1 Å 内有见证点的实验位点占 Zn 的 90.2%、结构水的 57.9%；2 Å 内分别为 99.2% 和 94.5%。把排斥作为求解后的独立过滤；离子难负例为 `τ_pw ≤ t_tol < τ`，其中 `t_tol` 是容差，不是第 3 节的片内展宽 `δ`。 | 表 2：分开报告 1 Å / 2 Å 逐位点覆盖、最近见证点定位误差及证书分母；另列 Zn 三配体难负例比例、排斥过滤与经典配体子层。时间外比较按 PDB **首次发布日 2024-05-01 起（含当日）**划分；外部表只有完成同池质控和一对一匹配后才纳入。 |
| 6 讨论与结语 | **几何区分度不是排序领先的同义词。** 收束为“共同点可复核、高阶差异可定位、两种出生规则的关系有界”。如需讨论排序，正文用一句话交代任务目标不同：固定的难例对用于检查条形码是否带来额外区分度；整体 Spearman 作为同池背景放补充材料。 | 难例配对差与同池参照列补表。严格四元超出留在理论，不以已审计的分子路径为实证。 |

## 各节的证据对应表

编号来自[证据清单](EVIDENCE_REGISTER.md)；“§5：项目名”是该清单中的未编号条目。“未编号（规范／实现）”表示已有可查依据，不另造登记号。表中相对链接均指向仓库中已有文件；待补的结果不虚列路径。

| 节 | 要支持的论点 | 证据条目 ID | 文件路径 | 状态 | 缺口／待办 |
|---|---|---|---|---|---|
| 摘要、1、3 | 空心三角形：两两可行与共同可行不同；并集片不存在由任意固定低阶子族决定全体的一般规则 | 未编号（命题 2） | [数学论证](THEORY_UNION_SUPPORT.md)、[合成校验](../experiments/theory_checks/README.md) | 理论可用；图待交接 | 图 1 由作图会话制作；图注明确是支撑集族的神经与旗复形。 |
| 2 | 选择公式、单接触四片精确性、加权 1-center 的活跃球证书、带剪枝的选择求解 | 未编号（引理 1、定理 1、实现） | [数学规范](MATHEMATICAL_SPEC.md)、[证明](THEORY_UNION_SUPPORT.md)、[求解器](../src/cspph/geometry/patch_support.py) | 理论和实现可引用 | 区分子问题证书与跨选择的最小值；不把未完成或未通过证书的计算填成零。 |
| 2，表 1 | 普通 Čech 是零半径单点支撑特例；同一点云的 Čech 与 Rips 可不同 | 1.1、1.2；§5：原 16 个结构的 Čech/Rips 精确对比 | [控制结果](../results/tables/cech_rips_compare.csv)、[报告](../results/reports/cech_rips_compare.md) | 生成时核对；两个 RNA 控制结构 | 不作为原 16 个结构的点云对比；原队列使用第 4 节的共同对两两滤过。 |
| 3，图 2 | 半径变量中的仿射 Jung 界；凸特例可取等号 | 未编号（定理 3）、1.5 | [定理](THEORY_UNION_SUPPORT.md)、[蛋白-DNA 比值分布](../results/tables/union_audit_pd_ratio_distribution.csv)、[RNA 比值分布](../results/tables/union_rna_ratio_distribution.csv) | 定理可用；两组分别核对 132,138 与 340,874 条路径，均无越界 | 已入库为分组分位数。图 2 由作图会话制作；若画逐路径分布，缺逐路径比值归档，绘图输入需对齐现有统计。 |
| 3 | 单接触片也出现 `η = τ − τ_pw > 0`；与严格四元超出是两个命题 | 1.6 | [蛋白-DNA 逐结构](../results/tables/union_audit_pd_per_structure.csv)、[RNA 子层汇总](../results/tables/union_rna_single_contact_summary.csv)、[RNA 高比值路径](../results/tables/union_rna_single_contact_top.csv) | 已核对 | RNA 的 η 明细对应比值最高的十条；缺全单接触子层的最大 η，不把十条中的最大值当总体最大值。 |
| 3、4 | 原队列与图类型、路径阶固定，提取后各阶路径数与归档吻合 | 1.7 | [固定清单](../manifests/union_audit_structures.tsv)、[路径归档](../results/tables/path_landscape.parquet)、[复现说明](HPC_EVALUATION.md) | 已核对 | 路径数复现不替代对区间端点或比值的核对。 |
| 4 | 同图上的两种路径滤过夹逼与交错；按实际出生事件值计算区间 | 未编号（定理 4、实现） | [数学论证](THEORY_UNION_SUPPORT.md)、[事件比较实现](../experiments/exp03_filtration_comparison/compare_exact.py)、[持久计算](../src/cspph/topology/event_persistence.py) | 理论和实现可引用 | 交错给上界，不把条形码差异表读成实测瓶颈距离。 |
| 4，图 3 | 有向图蛋白-DNA 5/8 至少一阶不同、对称化 8/8；RNA 两种图均为 H1、H2 各 8/8 不同；两组 H0 相同 | 1.3、1.4 | [蛋白-DNA 条形码汇总](../results/tables/union_audit_pd_barcodes.csv)、[蛋白-DNA 报告](../results/reports/union_audit_pd.md)、[RNA 条形码汇总](../results/tables/union_rna_barcode_summary.csv) | 已核对；图待交接 | 图 3 由作图会话制作；表给区间计数与总持续度，不给逐条 Å 端点。演示区间需要原始事件输出，并与表核对。 |
| 5，表 2 | Zn 和结构水的见证点覆盖、定位与求解证书 | 4.1、4.3 | [位点测量](../results/reports/ion_full_measurements.json)、[候选汇总](../results/tables/ion_full_summary.csv)、[评价设计](ION_WATER_SITE_TASK_DESIGN.md) | 已核对；Zn 与水各完成 300 个结构 | 逐真位点看是否存在邻近见证点，不等于一对一匹配的召回、精确率或 F1，也不是时间外子集结果。 |
| 5，表 2 | Zn 三配体候选包含两两通过、共同支撑不通过的难负例；分开呈现排斥与经典配体口径 | 4.2 | [候选分层汇总](../results/tables/ion_full_summary.csv)、[评价设计](ION_WATER_SITE_TASK_DESIGN.md) | 已核对 | 标清每个比例的分母；难负例占比不是难例分类 AUROC。 |
| 5，外部比较（条件纳入） | Metal3D、AllMetal3D 在同一个时间外 Zn 子集作位点级比较 | §5：外部离子位点对比（Metal3D、AllMetal3D） | [外部协议](EXTERNAL_BASELINE_PLAN.md)、[执行说明](../experiments/external_metal_site_run/README.md)、[运行状态](../results/reports/external_metal_site.md)、[位点质控](../results/tables/external_metal_site_qc.csv)、[外部质控状态](../results/tables/external_metal_external_qc_status.csv)、[指标状态](../results/tables/external_metal_metrics_status.csv) | 缺：待回收外部运行结果 | 分开记录候选结构、自动质控位点与外部质控状态；在同池上做 1 Å / 2 Å 一对一匹配与定位误差，注明覆盖率和未能核对的训练／同源边界。结果未齐不作比较，空值不记作零。 |
| 6，补表 | 冻结难例中，条形码层为 FTDMP bound 带来额外的成对区分度；支撑差单独加入未显示同样的增量 | 3.1 | [合并种子的冻结评估](../results/tables/hardpair_frozen_summary_pooled.csv)、[协议](HPC_EVALUATION.md) | 已核对 | 固定特征、难例集合和靶标分组；不外推到一般排名或位点预测。 |
| 补表 | RNA 的同池 Spearman 给出排序背景，不替代几何对照 | 2.1 | [逐靶标对齐分数](../results/tables/external_scoring_pm_full_per_target.csv)、[单次划分配对区间](../results/tables/external_scoring_pm_bootstrap_single_split.csv) | 已核对 | PM 20 靶标、806 个诱饵，不含天然结构；分列 20 次划分均值和单次划分配对区间，不用区间跨零宣称等效。 |
| 补表 | 蛋白-DNA bound 的同池 Spearman 给出排序背景；VoroMQA 有复现抽检，官方 FTDMP 共识排序另列为参照 | 2.2、2.3、2.4；§5：FTDMP unbound、1BDT 那 1 个 unbound 候选 | [汇总](../results/tables/external_scoring_pd_summary.csv)、[配对区间](../results/tables/external_scoring_pd_bootstrap.csv)、[VoroMQA 分数](../results/tables/pd_voromqa_scores.jsonl)、[对齐报告](../results/reports/external_scoring_pd.md) | 2.2、2.3 已核对；2.4 生成时核对 | bound 与 unbound 分列；后者如保留在补充材料，注明候选质量构成与共同比较池排除的那条 1BDT 候选。 |
| 补充材料（可选） | RNA Mg 位点结果与规模截断单列 | §5：Mg（RNA）位点、Mg 被截断的 4 个结构 | [位点测量](../results/reports/ion_full_measurements.json)、[逐结构状态](../results/tables/ion_full_per_structure.csv)、[超算重跑脚本](../scripts/slurm/ion_mg_relaxed.sh) | 16/20 个结构完成；缺超算重跑结果 | 公开已完成分母与截断口径；不进正文，也不并入严格时间外 Zn 比较。 |
| 补充材料／排除项 | 训练边界无法核对或没有可复现程序、权重的方法，不承担主张 | §5：lociPARSE；3dRNAscore、ARES、DDNA3、ITScore-PR、3dRPC | [打分清单](../results/reports/external_scoring_inventory.md)、[外部协议](EXTERNAL_BASELINE_PLAN.md) | lociPARSE 已运行，训练重叠未核对；其余缺可复现程序或权重 | lociPARSE 如列出须标明训练重叠未核对；其余不进正文比较。 |
| 3、6（理论边界） | 并集片可严格超出四片子族决定的尺度 | §5：四元严格超出 | [命题 2](THEORY_UNION_SUPPORT.md)、[队列审计](../results/reports/union_support_audit.md) | 理论可用；缺分子实证 | 审计路径最多含四个不同片，不以单接触的两两缺口替代严格四元超出的证据。 |
