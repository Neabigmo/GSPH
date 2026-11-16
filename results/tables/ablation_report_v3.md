# GSPH Ablation-3.0 — 条件互补性验证 + 深层对象落盘（R3）

日期：2026-11-15。按战略转向执行：不再追求三任务全面涨点，围绕两个问题投入——(1) PD Bound 的 T×H 条件互补是否真实且特异；(2) 逐路径 τ 与真实 barcode 能否落盘并产生信息。

## 1. 交付一致性修复（P0）

- `cv_splits_membrane.json`：已保存 7 个 LOPO 蛋白列表。
- `interaction_test_v2.csv` 与 `interaction_test_compact_v2.csv`：**两文件内容一致**（均为 `stats_v2.py` 的全部交互臂输出，canonical 文件名为后者）。pd_unbound 只有 I_full 与 I_compact 两臂——交叉压缩模型 X_P_T_H2 等只在 rna 与 pd_bound 上训练（`run_ablation_v2.py` 有意为之，交互检验聚焦 bound），不是文件残缺。
- `summary_all.csv`：已重建为 v1+v2 全部模型指标。
- `run_ablation_v2.py`：v1/v2 OOF 合并处新增 **structure_id 集合相等断言**，杜绝位置对齐风险。

## 2. 实验一：I_bound − I_unbound 配对检验（核心新结果）

同一批 47 个 target 上，逐 target 计算 I_TH = [S(PTH)−S(PT)]−[S(PH)−S(P)]（v1 全特征模型 A1/A4/A5/A7，target-macro Spearman），再对 bound−unbound 做配对 bootstrap + sign-flip：

| 量 | 值 |
|---|---|
| I_bound 均值 | +0.0341 (p_holm=0.023) |
| I_unbound 均值 | +0.0036 (n.s.) |
| **I_bound − I_unbound** | **+0.0305, 95% CI [+0.0040, +0.0585], p_perm=0.035** |

→ 在 seed-0 folds 上，T×H 条件互补显著强于 unbound 对照。**但 §6 多 seed 结果表明该显著性不可复现——此处数字应视为单次划分的描述值，而非稳定发现。** RNA target id 命名不同无法配对（已注明）。

## 3. 实验二（完成）：逐路径 τ / eta / barcode 落盘

- **引擎补丁已应用**（`gsph6/engine.py`，仅序列化层改动，数学内核未动）：
  - `res["path_support"]` = 每条路径 {tau, valid}（含 certificate 判定）；
  - `res["path_eta"]` = 每条 3/4 阶路径的 η 增量（原仅存均值）；
  - `res["persistence"]` 本就存在：`orders[k].intervals = [birth_idx, death_idx, right_censored]`，birth/death 为**离散滤过的尺度索引**（scales=[0,0.5,2]），非连续坐标——已如实标注。
- `ablation/deep_dump.py`：序列化为 `*.barcode.parquet` / `*.support.parquet` / `*.deep_manifest.json`。
- `ablation/deep_subset.py`：RNA + PD，`--shard/--n-shards` 支持 SLURM array 并行。
- **R3.1 选择器修正**：`_stratified_targets()` 按 难度（标签中位数三分位）× 结构规模（原子数三分位）× 候选质量离散度 贪心覆盖选 target，不再偏向易例；stratum 写入 manifest。

### 3a. 实际产物（HPC `wuzh02.hpccube.com` SLURM array，job 49162479/49168830）

**PD Bound（8 结构 / 4 target，`results/deep_subset_pd/`）**：分层覆盖 d0_s0/d0_s1/d0_s2/d1_s0；路径数 60–10,256；单结构 5–1,708s。

**RNA（8 结构 / 4 target，`results/deep_subset_rna/`）**：3RKF、3L0U、5SWD、3IVN × {3dRNA/FARFAR2/SimRNA}；路径数 5,211–10,827；单结构 710–951s。

**关键工程修正**：首版 RNA 任务误用全事件集（等价 sparse_fraction=1.0），单结构 >5h 无产出；核对 `gsph_h3_feat/rna` 生产 audit（970 结构，`sparse_fraction=0.25, per_nt_quota=8`，中位 661s）后改为与生产完全一致的稀疏事件滤波 + `rna_quality_sparse` 场景，8 结构全部 ~12–16min 完成。**deep subset 的数学对象与生产特征矩阵同一复合体**，可直接对接。

### 3b. 落盘数据概览（真实分子，非合成）

| 任务 | 结构 | 路径数 | ord≥2 路径 η>0 占比 | τ 中位数 | barcode 区间数 (H0/H1/H2) |
|---|---|---|---|---|---|
| PD | 8 结构 | 60–10,256 | 3–12% | 0.79–2.41 | 5–58（个别含 H2） |
| RNA | 8 结构 | 5,211–10,827 | 8–12% | 1.41–1.85 | 234–354（H0≈125, H1≈100, H2≈2–9） |

- **真实数据上 η>0 普遍存在**：RNA 每结构数百条三/四阶路径存在正的高阶支撑增量（τ₃ > max 两两 τ）——T4a 合成构造的现象在真实分子中同样成立。
- **H2 持久区间真实存在**：RNA 每结构 2–9 条 H2 barcode（dim=2），PD 个别结构含 H2——高阶持久模不是空集。
- barcode 为离散尺度索引（birth_idx/death_idx ∈ {0,1,2}），lifespan_idx 为指数差；right_censored 标志保留。

### 3c. 后续可做（数据已备好）

- T1 支撑分布：quantiles / P(τ>t) / η 分布按 order×化学类型分解；
- T3 barcode 统计：total persistence、persistent entropy、lifespan 分布 vs 标签的小样本相关；
- GSPH vs 普通边权滤过的 barcode 差异（需对照提取，成本≈同量级）。

## 3b. 合成几何验证（R3.1 完成，`ablation/geometry_props.py` → `geometry_props.json`）

在 `compute_patch_support` 精确求解器上直接构造 InteractionPatch，隔离数学内核（无 PDB/无提取管线）。C++ support 后端本机不可用（torch DLL），全部走 certified python cascade——数学相同、仅慢。

| 测试 | 结果 |
|---|---|
| T1 单调性 π⊆π′⇒τ(π)≤τ(π′) | 60 随机嵌套构型，**0 违反** |
| T2 刚体不变性 SE(3) | 40 组随机旋转+平移，max\|Δτ\|=**1.6×10⁻¹³**（机器精度） |
| T3 扰动稳定性 | 原子高斯抖动 ε∈{0.1,0.25,0.5,1.0}Å：max\|Δτ\|/ε ≈ **1.03–1.28**，与 τ 作为支撑函数的 1- Lipschitz 性质一致 |
| T4a **两两 τ 相同、三重不同（精确构造）** | 构型 A：τ12=1.301, τ13=τ23=0.8006, **τ123=1.426**；构型 B（P3 加一个远端备选微接触）：τ12=1.301, τ13=τ23=0.8008, **τ123=3.521**。两两支撑在两构型间相同（差 <3×10⁻⁴，来自接触偶极微偏移），三重共同支撑相差 2.5 倍——**普通两两/边权滤过在数学上无法表达此差异** |
| T4b η₃=τ₃−max τ₂>0 普遍性 | 300 随机三 patch 构型中 **67%** 存在正的高阶增量（mean +0.23, max +2.04） |
| T4c 两两 τ 对 τ₃ 的决定力 | corr(τ₃, max两两τ)=0.87，但残差 std=0.38（τ₃ 范围 [0,3.06]）——两两信息外仍有大量高阶方差 |
| T5 方向性 | τ(path) 为**集合值**：路径反转 Δτ=**0.0**（精确）；化学方向仅通过有向边枚举进入路径集合（合成图：有向 12 条路径 vs 无向 68 条）——**方向信息载体是 GLMY 链复形而非 τ 值**，这是对"τ 无向"的如实确认 |
| T6 滤过对照 | τ-滤过 vs 边权-滤过（max 原子距代理）在随机点集上 2700 对排序中仅 **0.67%** 不一致——随机构型上两者高度一致，区分能力依赖 T4a 这类结构化构型；τ 洗牌 100% 破坏序（ sanity ） |

**诚实解读**：T4a 是本项目首个"两两统计完全相同但高阶支撑不同"的严格构造——它证明共同几何支撑诱导的路径滤过**表达能力严格强于**两两关系。T6 同时提示：在无结构的随机点集上 τ 与朴素距离滤过排序几乎一致，判别力来自接触组合的非凸并结构（union-of-lenses 的 min-over-contacts），这正是数学贡献点。Hodge 谱扰动稳定性（Spec(Δ_k)）需完整 GLMY 链，留待 deep subset 真实结构验证。

## 4. 实验三：独立泛化 —— 可行性评估

- FTDMP 是当前唯一带大量 decoy + CAD 标签的蛋白-DNA 对接集；**本地无第二个独立 decoy 集**。生成新 decoy 需对接流水线（HPC）。
- 已有的 target-grouped 5-fold CV 本身就是"未见 target 泛化"检验；unbound 作为同 target 不同协议对照已用。
- 建议：独立验证留待 HPC 生成外部 decoy 集后进行；当前以 bound-vs-unbound 配对 + hard-pair 分析作为内部泛化证据。

## 5. Hard-pair discrimination（新分析，本地完成）

定义：target 内 |Δlabel| ≥ 75 分位 的"难对"，且物理模型分差处于其自身分布下 50%（物理上不可分）→ 比较物理-only 与全模型的成对排序正确率（target 配对 bootstrap）：

| 任务 | 对比 | Δwin(full−phys) | 95% CI | p_perm |
|---|---|---|---|---|
| PD bound | B9_PcT2H2 vs B2_PcV2 | **+0.0247** | [+0.005, +0.043] | **0.016** |
| PD bound | A7_PTH vs A1_P | +0.0124 | [−0.006, +0.031] | 0.197 |
| PD unbound | B9 vs B2 | −0.0190 | [−0.039, +0.001] | 0.076 |
| RNA | CleanFull vs A1 | **+0.0246** | [−0.031, +0.084] | 0.430 |
| RNA | B9 vs B2 | −0.0094 | [−0.063, +0.041] | 0.741 |

（R3.1 修正：RNA 的 y_true 为 RMSD（低优）、y_pred 为质量分（高优，已验证与 RMSD 负相关），正确判据为 sign(Δs)=sign(−ΔRMSD)。旧版对两者同乘 −1 导致符号抵消，RNA 两行符号已翻转；PD 方向本就正确、数值不变。）

解读：**紧凑全模型在 bound 的物理模糊难对上显著更优**（p=0.016；6 对比族 Holm≈0.094，边际）。修正后 RNA 的 CleanFull−A1 方向转正（+0.0246, n.s.），与 CleanFull PTHS 的正向趋势一致但仍不显著。unbound 为负向趋势——再次支持互补特异于结合界面。

**多 seed 复核结果（`multiseed_hardpair.py`，10 个 fold 划分，A1_P vs A7_PTH）——本轮最重要的正面结果：**

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| d = win(full)−win(phys) | +.012 | +.012 | +.022 | +.025 | +.030 | +.014 | +.017 | +.008 | +.031 | +.011 |

**10/10 seed 全部为正，均值 +0.0182，范围 [+0.0078, +0.0307]；4/10 seed 单独显著（p<0.05）。** 与 I_TH 的符号不稳形成鲜明对比——hard-pair 判据度量的是"物理模糊难对上的成对排序正确率"，比宏观 Spearman 交互项对模型方差更鲁棒。这是目前唯一经 10 次独立划分复核的 GSPH 正向信号：**GSPH 在物理描述符难以区分的候选对上，确实比纯物理模型更常排对**。注意它衡量的是排序正确率而非 Spearman 增量，且定义依赖物理模糊子集——应作为"高阶结构提供非冗余判别信息"的证据，而非整体性能优势。逐 seed 明细：`hardpair_multiseed_pd_bound.csv`，OOF：`oof_mshp{1..9}_pd_bound.parquet`。

## 6. 多随机种子稳健性 —— 结果推翻了交互结论（关键负面结果）

`ablation/multiseed_bound.py`：pd_bound 上 A1/A4/A5/A7 四模型以 fold seed=0(正式)/1/2 重跑（canonical `cv_splits_pd_bound.json` 已恢复验证）：

| fold seed | I_TH | PTH−P |
|---|---|---|
| 0（正式） | **+0.0341** | +0.0012 |
| 1 | **−0.0004** | +0.0060 |
| 2 | **−0.0140** | +0.0073 |

**I_TH 均值 ≈ +0.007，三个 seed 变化 −0.014 ~ +0.034，符号不稳定。**

→ **R2 的"显著" T×H 交互（p_holm=0.023）是该特定 fold 划分下的产物，不可复现。** bound-vs-unbound 配对差（+0.030, p=0.035）同样建立在 seed-0 folds 上，同样存疑。Hard-pair 的 +0.025 优势也需在多 seed 下复核后才能采信。

诚实结论更新：目前**没有任何经多 seed/独立数据复核的 GSPH 增量证据**。交互信号应降级为"假设"而非"发现"。这与用户"是否偶然现象"的担忧完全一致，且答案是：很可能是。

## 7. 当前结论状态（R3.1 修正）

- **已证伪（就当前证据）**：PD Bound T×H 交互的稳定性——3 个 fold seed 下 I_TH ∈ [−0.014, +0.034]，均值≈0。
- **新获正面证据（R3.1）**：
  1. **Hard-pair 优势 10/10 seed 全正**（均值 +0.0182）——GSPH 在物理模糊难对上比纯物理更常排对，经独立划分复核；
  2. **几何核数学验证完成**：单调性 0 违反、刚体不变性达机器精度、扰动 Lipschitz≈1、以及**严格构造证明两两 τ 相同而三重共同支撑不同**（τ123=1.43 vs 3.52）——共同几何支撑滤过的表达能力严格强于两两关系；67% 随机构型存在正 η₃ 高阶增量。
- **如实边界**：τ 为集合值、反转不变——化学方向信息载体是 GLMY 链复形而非 τ 本身；随机点集上 τ 与朴素距离滤过排序 99% 一致，判别力依赖真实接触组合结构；Hodge 谱扰动稳定性待真实结构验证。
- **仍成立**：P 承载绝大部分信息；PcV2(~115d) 是高效紧凑表示；膜任务 PcV2 enet +0.178 为唯一正 LOPO 信号；规模混杂在 PD bound 结构性存在（严格 S 仍达 0.208）。
- **已完成**：PD + RNA 各 4 target × 2 decoy 的逐路径 τ/eta + 离散尺度 barcode 落盘（HPC 提取，产物已拉回本地 `results/deep_subset_{pd,rna}/`，含合并 manifest）。
- **论文定位建议（与用户一致，且更保守）**：在拿到 barcode/τ 与多 seed 复核之前，GSPH 应定位为**数学表示框架 + 探索性应用**，而非已验证的预测优势算法。四个理论定理（有限 minimax 表示、单调性、Lipschitz 稳定性、Hodge-拓扑对应）是当前最有价值的可交付贡献。

## 8. P0 数学审计实验（对应 V0.3 论文补充实验方案）

产物：`p0_audits.json`、`p0_chain_audit.json`、`filtration_compare.json` + `filtration_compare_summary.csv`、`deep_subset_{pd,rna}_complex/*.complex_{paths,edges,tau_table,patch_audit}.*`。脚本：`ablation/p0_audits.py`、`p0_chain_audit.py`、`filtration_compare.py`、`deep_complex_dump.py`。

### P0-1 同图同路径复形滤过对照（15 结构：PD 7 + RNA 8）

对每个结构的**同一 GLMY 路径复形**（dir=化学有向边 / und=对称化），用同一 `sparse_chain → Omega 边界 → inclusion chain maps → persistence` 流水线比较五种出生规则：`tau`（GSPH 共同支撑）、`pairwise`（canon 内所有 2-子集 τ 的最大值）、`facetmax`（递归 max-face）、`edgew`（路径连续边权最大值）、`shuffled`（τ 置换对照）。

**出生值层面（核心数学证据）**：τ − pairwise-max 在 **0–41% 的路径上严格为正**（15 结构合计；dir 图结构均值 0.003–0.043 Å，und 图 0.008–0.043 Å，p90 最高 0.18，最大 0.68 Å）——高阶共同支撑确实携带超出任意两两支撑的几何信息。RNA und 图比例最高（~31–41%）。`facetmax ≡ tau` 恒等（由 τ 对子集的单调性平凡推出，非独立对照）。

**Barcode 层面（如实边界）**：在 3 个离散尺度 [0, 0.5, 2.0] 下，tau 与 pairwise 滤过的持久区间 **86/90 个（结构×图×阶）完全一致**；仅 4 处不同且全部是 tau 多出一条区间（1FOK und 的 H2、3 个 RNA 的 H1）。即：**高阶支撑差异在出生值上普遍存在，但在粗粒度离散 barcode 上的持久同调足迹较小**；差异在对称化（und）复形和 RNA 上最明显。edgew 滤过产生显著不同的 barcode（H0 区间数约 2 倍）。shuffled 对照在多数复形上**响亮失败**（Omega 坐标映射非整数 / dense fallback 超限）——说明 GLMY 滤过的可容许性是实质约束，不是任意赋值都能成持久模。

**链映射审计**：所有可计算的 tau/pairwise/facetmax/edgew 复形 chain-map 方程全部通过（max residual = 0）。

### P0-2 有界坐标扰动 → 1-Lipschitz 定理直接验证

每个原子在 ε-球内均匀有界位移（非高斯抖动），40 随机世界 × 4 ε × 10 patch 组合 × 2 次求解 = 400 对/ε：

| ε (Å) | max\|Δτ\| | 界 ε | 违反 | 比值 |
|---|---|---|---|---|
| 0.1 | 0.0685 | 0.1 | 0 | 0.69 |
| 0.25 | 0.1852 | 0.25 | 0 | 0.74 |
| 0.5 | 0.3677 | 0.5 | 0 | 0.74 |
| 1.0 | 0.8136 | 1.0 | 0 | 0.81 |

**1600 对位移-支撑对全部满足 |Δτ| ≤ ε**，比值 0.69–0.81（松弛来自最坏方向未必对齐）——Theorem 3 的 1-Lipschitz 界获直接数值证实。

### P0-3 精确 Support 求解器审计

独立实现：枚举全部接触选择 → 每叶原子集的加权 1-中心（scipy SLSQP epigraph 多起点）对照 `compute_patch_support`（B&B + 证书）：

- **120/120 随机试验 τ 完全一致**，max|Δ| = 3.8e-11 Å，mean 6.9e-13；0 证书失败、0 不完整。
- 路由分布：`branch_and_bound` + `closed_form_two_ball`。
- 退化用例：空路径 τ=0 认证通过；空 patch → τ=∞/incomplete/**绝不当作有效值**；重复 patch_id 集合值一致（τ1=τ2=0）；共点原子 τ=0；`max_combinations=0` 诊断模式正确返回 incomplete。

### P0-4 GLMY/Persistence/Hodge 审计（60 随机嵌套复形 × 3 尺度）

- chain-map 交换性 `dF=Fd`：60/60 通过，max residual 7.6e-13。
- Ω 坐标边界平方 `d̃²=0`：max 8.9e-13；forbidden 核残差 ≤9.7e-15。
- Ω 维数 vs dense SVD 参考：60/60 一致。
- **发现并修复真实 bug**：`dense_svd_fallback`（float Ω 基）下 `boundary_ranks`/`betti` 原本用 mod-素数求秩，`int()` 截断会静默出错；已改为按 dtype 分流实算秩。修复后 betti 60/60 与独立实算一致。
- Hodge：全部 PSD（min eigenvalue ≥ −1e-6），调和维数=Betti（4 例 dense-fallback 退化复形上机器精度容差内 off-by-one，如实记录）。
- **真实分子复形上 dense fallback 触发率 = 0**（15 结构 × 全部规则 × 全部尺度 = 360/360 chains 均为 `structured_sparse`）——该 bug 在真实数据上不会触发，但防御性修复已入库。

### P0-5 化学角色/Patch 审计

- PD（7 结构，400 接触）：donor/acceptor 成员重叠 0、非法接触 0、空 patch 0、τ 抽查 15/15 一致；接触距离 2.02–4.99 Å。
- RNA（8 结构，2248 接触，1631 patch）：τ 抽查 54/54 一致、非法接触 0；**13/1631 patch 存在同原子双重 donor/acceptor 角色**（RNA 中磷酸/糖氧双向参与，化学上可解释，已标记待 SI 检查）；**2/8 结构出现 0.82–1.88 Å 的超短接触距离**（低于典型非键范围，疑为共价/别构异构体伪接触，`intra_min_bond_sep` 排除需在 SI 中核实）。

## 9. 负面结论复核审计（negative-result audit）

对全部失败/阴性结果逐一排查"是否为代码实现问题"：

| 结论 | 复核结果 | 判定 |
|---|---|---|
| I_TH 交互 3-seed 符号翻转 | 用 `interaction_test.per_target` 在 oof_ms1/ms2 上独立重算：0.0341/−0.0004/−0.0140 **逐值复现**；四臂均覆盖全部 47 target，无目标集混杂；y_true 方向（PD=CAD 高优、RNA=−RMSD）正确 | **真实不稳定**，非 bug |
| PTH−P≈0 | topology_features_v2 四任务 206 列：无全 NaN 列，绝大多数列有方差（恒定列仅为 k0 等退化阶的合法占位） | **特征未退化**，零增量为真实结果 |
| tau≈pairwise barcode 86/90 | tau_table 覆盖审计：318,196 个 2-子集 **0 缺失**——pairwise 出生值是真实独立计算，非回退假象；birth 层面 τ>pairwise 在 0–41% 路径成立 | **真实结果**：差异存在于出生值但粗粒度 barcode 足迹小 |
| RNA hard-pair 方向 | 已修并验证 sign(Δs)=sign(−ΔRMSD) | 修复正确 |
| membrane LOPO 失败 | cv_splits_membrane.json 确认 7 蛋白级 LOPO（BR/DSBB/GLPG/GLYA/OMPA/OMPLA/PAGP） | 真实小样本局限，非划分错误 |
| T6 τ vs 距离滤过 99% 一致 | 复核实现："edge-weight" 实为全体接触原子**直径**（max pairwise distance），在随机单球 patch 上与 τ 同测空间弥散度——对照本身较弱，99% 一致是预期而非 bug；但说明该对照不能区分"空间紧致度"与"共同支撑" | **对照设计偏弱**，论文应改用真实边权/逐边定义重测 |
| RNA 超短接触（0.82Å） | 原子级探针：SimRNA-3IVN-5 中 N2(G49)···N1(A53) 距离 0.82Å、bond-graph 距离 >8（intra_sep=4 正确放行）；为 **SimRNA decoy 的真实空间冲突**（重建伪影），非事件检测 bug；但提示缺少**最小距离下限**——clash 接触会使 τ→负值恒活跃 | **数据问题+设计缺口**：建议加 d_min 下限或 clash 过滤，列入 SI |
| shuffled 滤过失败 | `_integer_csr` 对非整数 Omega 映射**响亮报错**而非静默——设计正确 | 预期行为，非 bug |
| dense-fallback rank | 已修复（dtype 分流实算秩）；真实复形 360/360 structured_sparse | 已修复 |

**复核结论**：所有主要负面结果（交互不稳、PTH 零增量、τ≈pairwise、膜 LOPO、RNA 无增益）均确认为真实结果而非实现错误。唯一需要行动的是：(a) T6 对照定义偏弱应重设；(b) 分子接触缺最小距离下限，SimRNA 类 decoy 的 clash 会进入接触集——建议补 `d_min` 过滤并在 SI 量化影响。

## 10. Clash 过滤修复与 RNA 全量重跑（clash06）

**修复**：`contact_min_vdw_ratio=0.6`（接触距离 <0.6×vdW 和判为 steric clash）注入 `intramolecular.extract_intra`，覆盖全部内分子提取路径；PD 互分子不受影响。阈值依据：合法接触最小 vdW ratio≈0.68，SimRNA clash≤0.6。

**重提取**（HPC job 49178410+49196422）：826/826 complete，18 shards，特征与旧版在无 clash 结构上逐位一致（max diff 1e-14）。287/826 结构特征变化（SimRNA 151、FARFAR2 46、RNAComposer 43、3dRNA 43、native≈0）。

**RNA 结论对比（clash 前 → 后）**：

| 指标 | pre-clash | clash06 | Δ |
|---|---|---|---|
| A1_P macro \|ρ\| | 0.529 | 0.515 | −0.014 |
| A4_PT | 0.503 | 0.520 | +0.018 |
| A5_PH | 0.546 | **0.551** | +0.005 |
| A7_PTH | 0.535 | 0.548 | +0.013 |
| CleanFull_PTHS | 0.561 | 0.535 | −0.026 |
| A9_Pcompact | 0.532 | 0.549 | +0.018 |
| hard-pair CleanFull−A1 | +0.0246 | +0.0246 | 0 |
| hard-pair B9−B2 | −0.0094 | −0.0094 | 0 |
| I_TH 交互 | n.s. | −0.0089, p_holm=1.0 | 仍 n.s. |

**结论**：clash 过滤是正确的建模修复（去除了非物理接触），但 **RNA 阴性结果稳健**——所有指标变化在噪声范围内，GSPH 仍未超过 rsRNASP (~0.58)。RNA 无增益不是 clash 伪影导致的假阴性。

**T6 对照修复**：edge-weight 代理从"全体原子直径"改为"相邻 patch 对最小接触距离瓶颈"，discordant fraction 1%→5.4%，对照区分度真实提升。

## 11. Clash06 复合体重跑后的 P0-1 滤过对照（重要修正）

RNA deep complex（8 结构）用 clash06 代码重跑后重算 `filtration_compare`（产物 `filtration_compare_rna_clash06.json`）：

| 指标 | clash 前 | clash06 |
|---|---|---|
| tau vs pairwise barcode 一致 | 86/90 (96%) | **2/16 (12.5%)** |
| τ>pairwise 出生值占比（dir） | 0–41% | 27–34% |
| und 复形差异 | 仅 4 处 | **几乎全部结构** |

示例（SimRNA-3IVN-5 und）：tau 136 区间（H1=5,H2=6）vs pairwise 129（H1=2,H2=2）——τ 滤过产生显著更多的高阶持久结构；edgew 滤过差异同样显著（k0 区间 205 vs 125）。

**解释**：旧结果中海量 clash 接触（τ≈0 恒活跃边）压缩了支撑差异，使 tau≈pairwise barcode 高度一致——**这是数据污染导致的假阴性**，已被 clash 修复纠正。清洁数据上 τ-滤过与两两滤过的持久同调确实不同，高阶共同支撑信息在 barcode 层面可检测。

注意：facetmax 与 tau 在此批结构上区间完全一致（max-pair 支撑与并集支撑在这些复形中给出相同持久结构）；edgew 滤过显著不同。PD 复合体不受 clash 影响，旧 PD 结果仍有效。
