# Metal3D / AllMetal3D：时间外子集 CPU 运行状态

**本次没有得到可发表的位点评价数字，不能把下表空值或未运行记作性能 0。**PM 全量评分及 EXP-06 对齐完成后才开始本项。当前阻断是：Metal3D 的官方体素化强制 CUDA，超出了“只加 map_location”的修改许可；AllMetal3D 最小设备修改已进入 CPU 前向计算，但单个小结构的有界试运行未完成，尚未证实端到端 CPU 推理成功。以下保留下载、原始离子和阻断证据，等待继续执行的决定，不冒称 41 结构已评价。

## 清单与结构质量

按 `experiments/exp09_metal_site_pilot/{zn,mg}_ids.csv` 中 `time_split_test=True` 原样选取 **Zn 38、Mg 3**，从 `https://files.rcsb.org/download/<PDB_ID>.cif.gz` 下载，41/41 成功，压缩总量 **20,233,779 B**（Zn 10,579,929 B；Mg 9,653,850 B）。没有替换、补选或把时间标志当成质量合格标志。文件均保存在仓库外；每个 URL、压缩字节数、SHA256、耗时和原始清单字段见 `external_metal_downloads.csv`。

Gemmi 0.7.5 读取第一模型；同名原子保留占有率最高的 altloc，平局按 altloc 字符顺序。所有 Zn/Mg 元素原始离子坐标单独导出，**不是完成设计筛选后的参考集**。全结构原始离子 Zn=113（occupancy≥0.9 的 103）、Mg=2,811（全部 occupancy≥0.9）；模型只用第一模型，不将多模型重复计数。

| 队列 | 指定/下载结构 | 满足 X-ray、分辨率≤2 Å、R_free≤0.25 | 原始离子 | 完成全部设计 QC 的结构 | 已完成预测 |
|---|---:|---:|---:|---|---:|
| Zn | 38/38 | 32 | 113 | 未核完 | 0 |
| Mg | 3/3 | **0** | 2,811 | 结构级已全部不合格 | 0 |

`ION_WATER_SITE_TASK_DESIGN.md` 的全量质量规则并不由 exp09 的 time_split 标志保证。结构级具体排除：

| 结构 | 队列 | 现有 mmCIF 所示不合格项 |
|---|---|---|
| 9EA0、9E6Q、37YA | Zn | ELECTRON MICROSCOPY，非 X-ray，未给 R_free |
| 8TRU | Zn | R_free=0.2562 |
| 8WZY | Zn | R_free=0.2601 |
| 10ZJ | Zn | R_free=0.2688 |
| 8EPK | Mg | 分辨率 2.65 Å、R_free=0.2607 |
| 9DFD | Mg | 分辨率 2.60 Å、R_free=0.2658 |
| 9DSX | Mg | 分辨率 2.05 Å |

因此严格执行设计后 **Mg 主评价没有合格结构，必须 N/A 而非 F1=0**；不能为凑齐“Mg 3 个”偷偷放宽规则。`external_metal_structure_audit.csv` 保存全部 41 个结构的数字、第一模型重原子数、CA 数及原始离子数。余下 Zn 尚需完成：离子 B≤邻近配体平均 B 的 2 倍、≥3 个大分子配位原子且来自≥2 残基、缺失配位壳/关键残基、辅因子决定几何、晶体接触/对称拷贝及 MetalPDB/BioLiP/CheckMyMetal 的参考质量核查。未伪造这些注释，也未把仅通过 occupancy 的离子提升为合格真值。

## Metal3D：只加 map_location 仍然不能在本 CPU 环境执行

- 官方代码：`https://github.com/lcbc-epfl/metal-site-prediction`。
- 固定提交：**`09cbfb3c30e71e4fbea7e535072506c66af97cd9`**。
- 权重在该提交仓库中，13,815,931 B；SHA256 见 `external_metal_weights.csv`。
- 唯一源码修改在 `Metal3D/metal3d.py`：`torch.load(..., map_location=device)`；没有修改体素化、设备分配、网络或候选定义。
- 实际诊断输入是请求子集中的 **12UM**，由下载的 mmCIF 用 Gemmi 转写 PDB，不改变坐标；运行官方 CLI 的 `--metalbinding --softexit --writeprobes --batch-size 1`。输入离子由官方 `protein and not hydrogen` 清理排除，不把真值喂给网络。

CPU 诊断已实达下列调用链，并非仅凭源码猜测：`metal3d.py:84` 调用 `processStructures(args.pdb, ids)` → `utils/voxelization.py:205` 执行硬编码 CUDA 分配。实际 stderr 末尾：

```text
voxels = torch.empty(len(results), 8, 32, 32, 32, device="cuda")
AssertionError: Torch not compiled with CUDA enabled
```

此处位于 `metal3d.py` 的 device 选择和权重加载之前，因此加 map_location 不会绕过它。反证检查：已排除仅仅漏装 Tkinter/权重映射的解释，补齐环境后，同一输入确实到达 CUDA 分配才失败。没有在其余 40 个结构上重复同一确定性错误。继续 CPU 执行至少需要获准把这一个分配改成 CPU/可用设备选择；这不是 map_location，因此当前未擅改。

## AllMetal3D：最小设备尝试，不宣称“CPU 不支持”

独立 Python 3.10 venv 中 `pip install allmetal3d==0.7.3`，Torch **2.5.1+cpu**，MoleculeKit **1.9.4**，NumPy **1.26.4**，SciPy **1.12.0**，scikit-learn **1.4.1.post1**，Gradio **4.20.0**。原 wheel 139,056 B；与其声明版本一致安装。权重按官方常量 URL 下载，未自行训练或替换：identity 7,252,515 B，metal/water 各 192,011,387 B，合计 **391,275,289 B**；三者下载 wall time **3.511 秒**，URL 与 SHA256 见表。

未修改版导入即实报：

```text
ImportError: No CUDA detected by PyTorch, please install a CUDA compatible PyTorch build
```

只作设备相关最小尝试：移除模块导入时的无 CUDA 拒绝；`utils/voxelizationNew.py` 的 CUDA 分配按 `torch.cuda.is_available()` 回退 CPU；`utils/main.py::predict_cli` 的三个实际加载路径增加 `map_location=device`。网络、权重、原子类型、体素分辨率、采样和聚类均未修改，GUI/旧 PyTorch 分支未改。第一次设备修改漏掉 CLI 的另一套加载路径时实报 CUDA 反序列化错误，已将 map_location 放在真实 `predict_cli` 调用点，未用该错误冒称最终不兼容。

在同一 **12UM** PDB 上，以官方 `predict(..., models='allmetal3d', mode='fast', batch_size=1, threshold=7, pthreshold=0.25)` 执行，OMP/OpenBLAS/MKL 各限制 1 线程。实际日志：

```text
models loaded in 0.725 seconds
voxelization completed in 3.901 seconds
running AllMetal3D
```

单结构运行超过 5 分钟仍在 CPU 前向计算，未生成完整位点预测；为遵守“不硬耗”，中止有界试运行，没有继续排入 41 结构。**这不是 CUDA 仍不可绕过的证据，更不是 CPU 端到端成功；也不是 8 核 VM 的充分性能测试。**是否使用更多 CPU 线程或给更长时间预算，需要明确继续决定；当前不把未完成的单线程诊断外推为全量耗时。

环境处理均在仓库外：系统 Python 缺 `Python.h`，本地解包发行版匹配开发头后编译 MoleculeKit；Metal3D 的 `turtle` 导入需 Tkinter/Tk/BLT，也只本地解包满足，没有 sudo 或修改系统安装。补装 requests、rdkit；Gradio 4.20.0 所需 `HfFolder` 在自动解析的 huggingface-hub 2.4.0 中已移除，按其依赖范围选用兼容 **0.36.2**。这些是环境安装，不是算法修改；原始错误未算成方法性能失败。

## 指标状态与下一步

`external_metal_metrics_status.csv` 对两个方法×两个队列×1/2 Å 各占一行，P/R/F1、TP 误差、TP/FP/FN 均明确留空，状态 `not_evaluated`。没有预测就不能把全部参考离子计为 FN；未核完设计 QC 就不能报告严格参考数。尚未执行一对一匹配，因而没有 TP 定位误差数值。完成后应在各阈值下作最大可匹配数的一对一二部匹配（同数再最小总距离），分别汇总 Zn/Mg 的 TP/FP/FN、P/R/F1、TP 欧氏距离均值和中位数，不复用 exp09 几何先导结果冒充外部模型。

需要继续决定：Metal3D 是否许可上述额外一个设备分配改动；AllMetal3D 是否继续多线程/更长 CPU 试运行。即使获准，Mg 严格主池仍为空，Zn 仍需补齐位点级设计筛选；不得将原始离子 CSV 直接用于对外比较。
