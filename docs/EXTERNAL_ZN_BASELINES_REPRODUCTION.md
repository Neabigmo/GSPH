# 锌位点外部基线复现说明：Metal3D 与 AllMetal3D

本文说明如何在论文使用的 32 个 Zn 时间外测结构上重跑 Metal3D 和 AllMetal3D，并按与共同支撑方法相同的口径评测。结果见 `results/reports/external_metal_gpu_comparison.md`。

## 1. 结构清单

清单在 `experiments/external_metal_site_run/zn_time_split_32.txt`，共 32 个 PDB ID，顺序即任务编号 0–31：

| 编号 | PDB ID |
|---|---|
| 0 | 12is |
| 1 | 12um |
| 2 | 30ct |
| 3 | 31lp |
| 4 | 43ea |
| 5 | 8yva |
| 6 | 8yvb |
| 7 | 9cp7 |
| 8 | 9csx |
| 9 | 9dgx |
| 10 | 9dih |
| 11 | 9eyg |
| 12 | 9f30 |
| 13 | 9feh |
| 14 | 9flm |
| 15 | 9g0j |
| 16 | 9g0l |
| 17 | 9htu |
| 18 | 9jko |
| 19 | 9jrr |
| 20 | 9m4s |
| 21 | 9n2y |
| 22 | 9pq8 |
| 23 | 9rfp |
| 24 | 9rjd |
| 25 | 9rje |
| 26 | 9sa1 |
| 27 | 9ug7 |
| 28 | 9vzj |
| 29 | 9x77 |
| 30 | 9zef |
| 31 | 9zsc |

- 来源：`experiments/exp09_metal_site_pilot/zn_ids.csv` 中 `time_split_test=True` 的结构（PDB 首次发布于 2024-05-01 及以后），再按 X-ray、分辨率 ≤2 Å、R_free ≤0.25 筛选，筛选记录在 `results/tables/external_metal_structure_audit.csv`。
- 输入文件：`experiments/external_metal_site_run/inputs/<ID>.pdb.gz`，随仓库提供。它们由 RCSB 的 mmCIF 只保留第一个 model 后写成 PDB，解压后的 SHA256 记录在 `results/tables/external_metal_inputs.csv`。两个方法都使用这 32 个文件。

## 2. 程序与权重

| 方法 | 代码 | 权重 |
|---|---|---|
| Metal3D | GitHub `lcbc-epfl/metal-site-prediction`，提交 `09cbfb3c30e71e4fbea7e535072506c66af97cd9`，使用其中 `Metal3D/` 目录 | 随代码提供：`Metal3D/weights/metal_0.5A_v3_d0.2_16Abox.pth` |
| AllMetal3D | PyPI `allmetal3d==0.7.3` | Hugging Face `simonduerr/allmetal3d` 上的三个 `.pth` 文件 |

四个权重文件的 SHA256 记录在 `prepare_tools.py` 和 `results/tables/external_metal_gpu_software.csv` 中。

准备步骤：运行 `python experiments/external_metal_site_run/prepare_tools.py --tools <工具目录>`。它下载 Metal3D 代码和 AllMetal3D 权重，校验 SHA256，并对源码做五处不改变模型输出的最小改动（每处改动前核对原文出现次数，对不上即报错退出）：
1. 设备：去掉导入时的强制 CUDA 检查，有 GPU 时使用 GPU；
2. 加载权重时加 `map_location=device`；
3. 工作进程数改读环境变量 `SLURM_CPUS_PER_TASK`，不再按 `multiprocessing.cpu_count()` 开进程；
4. 删除 Metal3D 中未使用的 `from turtle import width`；
5. 推理放进 `torch.no_grad()`。

模型定义和阈值保持官方默认：聚类阈值 `threshold=7`，概率阈值 Metal3D `pthreshold=0.10`、AllMetal3D `pthreshold=0.25`（`run.py` 中的 `THRESHOLD` 和 `P_THRESHOLD`）；AllMetal3D 使用 `models='allmetal3d', mode='fast'`。推理批大小为 8，每次处理一个结构。

## 3. 依赖

论文结果使用的版本（`results/tables/external_metal_gpu_software.csv`）：

| 组件 | 版本 |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.5.1（CUDA 12.1） |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scipy | 1.12.0 |
| moleculekit | 1.9.4 |
| allmetal3d | 0.7.3 |

其余包的版本见 `experiments/external_metal_site_run/constraints.txt`，安装时用 `-c constraints.txt` 约束。PyTorch 需 ≥1.13，因为 AllMetal3D 加载权重时使用 `weights_only=True`。另需 `tqdm`、`py3Dmol`、`typing_extensions`。运行入口 `run.py` 记录峰值内存时用到 Python 的 `resource` 模块，因此需在 Linux 环境中运行。

## 4. 输出

`run.py` 为每个方法、每个结构写一个 `result.json`，格式为 `{"record": <运行记录>, "sites": [<预测位点>], "details": <AllMetal3D 的 details 或 null>}`，再由 `collect()` 汇总成四张表：

- `external_metal_gpu_predictions.csv`：每个预测位点一行，列为 `method, pdb_id, rank, element, x, y, z, probability`。坐标和概率读自两个程序输出的探针 PDB（概率在 occupancy 列），读法见 `run.py` 的 `read_sites()`。
- `external_metal_gpu_runtime.csv`：每个方法、每个结构一行，共 64 行，列为 `method, pdb_id, status, seconds, n_sites, device, batch_size, peak_rss_mb, peak_vram_mb, torch`。
- `external_metal_gpu_software.csv`：软件版本、Metal3D 提交号和权重 SHA256。
- `external_allmetal_gpu_details.jsonl`：每个结构一行，保存 `allmetal3d.predict` 返回的第五项（金属种类概率等）。

## 5. 评测口径

- 参考位点：`results/tables/external_metal_site_qc.csv` 中这 32 个结构的 Zn 位点。主口径为自动质控通过的 69 个位点，另报全部 86 个位点。三种方法使用同一参考集。
- 对每个参考位点，取距离它最近的预测位点，两者距离记为定位误差；误差 ≤1 Å、≤2 Å 分别记为 1 Å、2 Å 命中，召回为命中的参考位点所占比例；另报定位误差中位数。
- 外部方法的预测按官方阈值输出，不再过滤。共同支撑方法取 τ ≤ 0.3 Å 的全部候选见证点，按同一规则计算。
