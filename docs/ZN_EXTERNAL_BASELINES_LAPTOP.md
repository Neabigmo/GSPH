# 在 Windows + RTX 4070 上运行 Metal3D 与 AllMetal3D：要求与规划

本文写给在这台笔记本上执行任务的人或 agent，读者不需要事先了解项目背景。

## 1. 任务是什么

在 32 个 Zn 结合蛋白结构（时间外测集）上，用两个已发表的外部方法 **Metal3D** 和 **AllMetal3D** 预测金属位点，交回逐位点的预测坐标。命中判定和召回由项目方用同一个匹配脚本统一计算，这台机器只负责推理和交回原始预测。

- 代码仓库：`github.com/Neabigmo/GSPH`，分支 `paper-strengthening`。
- 需要用到的目录：`experiments/external_metal_site_run/`。
- 取法：`git clone -b paper-strengthening https://github.com/Neabigmo/GSPH.git`，克隆到 E 盘的一个新目录。整个任务只用到上面这个子目录。

## 2. 输入：同一批结构，不得更换

- 清单：`experiments/external_metal_site_run/zn_time_split_32.txt`，共 32 个 PDB ID，顺序就是任务编号 0–31：

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

- 来源：`experiments/exp09_metal_site_pilot/zn_ids.csv` 中 `time_split_test=True` 的结构，再按 X-ray、分辨率 ≤2 Å、R_free ≤0.25 筛选，筛选记录在 `results/tables/external_metal_structure_audit.csv`。
- 输入文件：`experiments/external_metal_site_run/inputs/<ID>.pdb.gz`，共 32 个、约 4 MB，已随仓库提供，**不要重新下载**。它们是从 RCSB 的 mmCIF 只保留第一个 model 后写成的 PDB，解压后的 SHA256 记录在 `results/tables/external_metal_inputs.csv`。
- 两个方法都必须用这 32 个文件，不增、不减、不换版本。

## 3. 官方程序与权重（只用这些）

| 方法 | 代码 | 权重 |
|---|---|---|
| Metal3D | GitHub `lcbc-epfl/metal-site-prediction`，提交 `09cbfb3c30e71e4fbea7e535072506c66af97cd9`，只用其中 `Metal3D/` 目录 | 随代码提供：`Metal3D/weights/metal_0.5A_v3_d0.2_16Abox.pth` |
| AllMetal3D | PyPI `allmetal3d==0.7.3` | Hugging Face `simonduerr/allmetal3d` 上的三个 `.pth` 文件，SHA256 写在 `prepare_tools.py` 里，下载后必须校验 |

在笔记本上的准备步骤：激活第 4 节的环境，设好第 5 节的环境变量，然后运行 `python experiments/external_metal_site_run/prepare_tools.py --tools <WORK>\tools`。它会把 Metal3D 代码放进 `<WORK>\tools\Metal3D`，把 AllMetal3D 权重放进 `<WORK>\tools\home\.local\share\allmetal3d\weights`，校验 SHA256，然后对源码做五处最小改动；每处改动前都会先核对原文出现的次数，对不上就报错退出：
1. 设备：去掉导入时的强制 CUDA 检查，有 GPU 就用 GPU；
2. 加载权重时加 `map_location=device`；
3. 工作进程数：不再按 `multiprocessing.cpu_count()` 开 joblib 进程，改读环境变量 `SLURM_CPUS_PER_TASK`；
4. 删除 Metal3D 里没用到的 `from turtle import width`（这一行会要求装 Tk）；
5. 推理放进 `torch.no_grad()`，输出不变、内存更省。

模型定义和阈值都不改：聚类阈值 `threshold=7`，概率阈值 Metal3D `pthreshold=0.10`、AllMetal3D `pthreshold=0.25`（`run.py` 里的 `THRESHOLD` 和 `P_THRESHOLD`），AllMetal3D 用 `models='allmetal3d', mode='fast'`。

## 4. 已知可用的依赖组合

- Python 3.10。
- numpy 1.26.4、pandas 2.2.3、scipy 1.12.0、moleculekit 1.9.4；其余包的版本见 `constraints.txt`，安装时用 `-c constraints.txt` 约束。
- PyTorch **必须 ≥1.13**，因为 AllMetal3D 加载权重用了 `weights_only=True`。在 4070 上用官方 CUDA 版 `torch==2.5.1`（cu121 或 cu124 均可），用官方源安装：`pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`。
- 再装 `allmetal3d==0.7.3 tqdm py3Dmol typing_extensions`。

Windows 上已知的问题和替代办法：
- **`run.py` 用到了 `resource` 模块，Windows 上没有。** 请复制一份 Windows 版的 runner，把峰值内存改用 `psutil` 记录（父进程加子进程的 `memory_info().rss` 取最大值），其余逻辑不变。
- **moleculekit**：如果 pip 装不上，改用 conda 安装：`conda install -c acellera -c conda-forge moleculekit=1.9.4`。
- **AllMetal3D 的权重目录**：它按用户主目录查找 `.local/share/allmetal3d/weights`。Windows 上 Python 认的是 `USERPROFILE`，不是 `HOME`，所以运行前要把 `HOME` 和 `USERPROFILE` **都**设成非系统盘上的一个目录（例如 `<WORK>\tools\home`），否则权重会写到 C 盘。
- 如果 conda 环境创建失败，用 venv 加上述 pip 预编译包同样可以。
- **备选：在 WSL2（Ubuntu）里运行。** 这时 `run.py` 可以原样使用，moleculekit 的 Linux 预编译包也能直接装。注意 WSL 默认最多能用到物理内存的一半：必须先在 Windows 用户目录的 `.wslconfig` 里写上 `[wsl2]` 和 `memory=8GB`、`swap=0` 来限内存（这是唯一需要写到 C 盘的一个小文本文件）。项目目录和缓存仍放在 E 盘（WSL 里对应 `/mnt/e/...`）。GPU 需要 Windows 上的 NVIDIA 驱动支持 WSL，在 WSL 里运行 `nvidia-smi` 能看到 4070 才可以。

## 5. 这台笔记本的硬约束（必须遵守）

背景：上次在这台机器上不设上限地跑这两个程序，内存被占满、机器卡死；C 盘几乎没有剩余空间。

1. **不碰 C 盘，全部放 E 盘**：环境、代码、权重、缓存、临时文件和输出都放在非系统盘的一个新目录下（下文记作 `<WORK>`，放在 E 盘）。启动前设置以下环境变量：`TMP`、`TEMP`、`PIP_CACHE_DIR`、`TORCH_HOME`、`HF_HOME`、`XDG_CACHE_HOME`、`HOME`、`USERPROFILE`，全部指向 `<WORK>` 下的子目录；pip 一律加 `--no-cache-dir`。
2. **一次只跑一个结构、一个推理进程**，不并行。
3. **批大小 8**（`--batch-size 8`），不要用默认的 32。
4. **CPU 工作进程 ≤4**：设 `SLURM_CPUS_PER_TASK=4`，被修补过的代码会读它。同时设 `OMP_NUM_THREADS=4`，并在代码里调用 `torch.set_num_threads(4)`。
5. **显存上限**：进程开头调用 `torch.cuda.set_per_process_memory_fraction(0.8)`，即 4070 的 12 GB 中最多用约 9.6 GB。
6. **内存上限**：runner 里加一个看门狗，每 5 秒用 `psutil` 检查一次；推理进程树的内存超过 8 GB，或者系统可用内存低于 4 GB，就结束当前结构，记为 `error_memory_limit`，然后继续下一个。
7. **单结构时限**：Metal3D 每个结构 30 分钟，AllMetal3D 每个结构 60 分钟，超时就结束并记为 `error_timeout`。
8. **逐结构落盘、可续跑**：每个结构跑完立即写 `<WORK>\out\<method>\<ID>\result.json`（格式见第 6 节）；再次启动时跳过已有 `result.json` 的结构。
9. **看得见、随时能停**：在一个可见的命令行窗口里前台运行，不用隐藏窗口或后台服务。停止办法：在这个窗口按 Ctrl+C，或者关掉窗口；必要时在任务管理器里结束 `python.exe`。停下后重新启动会从未完成的结构接着跑。
10. 先只跑两个结构做校准（12is 和 12um，两个方法各跑一次），记下每个结构的耗时、峰值显存和峰值内存。确认没有超限后，再跑其余 30 个。

## 6. 要交回的文件

全部放在 `<WORK>\out\tables\` 下，用 UTF-8 编码的 CSV（逗号分隔，带表头）。

**`external_metal_gpu_predictions.csv`**：每个预测位点一行。
| 列 | 含义 |
|---|---|
| method | `metal3d` 或 `allmetal3d` |
| pdb_id | 清单里的 ID |
| rank | 该结构内按输出顺序的序号，从 1 开始 |
| element | 探针 PDB 里的残基名列（第 18–20 列） |
| x, y, z | 位点坐标（Å），取探针 PDB 第 31–54 列 |
| probability | 位点概率，取探针 PDB 的 occupancy 列（第 55–60 列） |

两个程序都把位点写成探针 PDB，概率放在 occupancy 列。`run.py` 里的 `read_sites()` 就是这个读法，请原样沿用。

**`external_metal_gpu_runtime.csv`**：每个方法、每个结构一行，共 64 行。列为 `method, pdb_id, status, seconds, n_sites, device, batch_size, peak_rss_mb, peak_vram_mb, torch`。`status` 取 `success`、`error_<返回码或异常名>`、`error_timeout`、`error_memory_limit` 或 `missing` 之一。

**`external_metal_gpu_software.csv`**：两列 `item, value`。至少包含：Python、torch、CUDA、显卡驱动、numpy、pandas、scipy、moleculekit、allmetal3d 的版本，Metal3D 的提交号，以及四个权重文件的 SHA256。

**`external_allmetal_gpu_details.jsonl`**：每个结构一行 JSON，`{"pdb_id": ..., "details": ...}`，`details` 是 `allmetal3d.predict` 返回的第五项（金属种类概率等），原样保存。

每个结构的 `result.json` 格式为 `{"record": <runtime 那一行>, "sites": [<predictions 的行，不含 method/pdb_id/rank>], "details": <AllMetal3D 的 details 或 null>}`。四张表都由这些 `result.json` 汇总生成，逻辑同 `run.py` 的 `collect()`。

## 7. 怎么算跑完

- 64 次运行（2 个方法 × 32 个结构）都有 `result.json`，并且 runtime 表里没有 `missing`。
- 失败的结构允许重跑一次，第二次仍失败就保留错误状态，不改参数、不换结构。
- 交回第 6 节的四个文件，再附一句话：校准时每个结构的耗时、峰值显存和峰值内存。

## 8. 评测口径（供参考，由项目方计算）

- 真实位点：这 32 个结构中的 Zn 位点。
- 对每个真实位点，取距离它最近的预测位点，两者距离记为定位误差。
- 命中：误差 ≤1 Å 记为 1 Å 命中，误差 ≤2 Å 记为 2 Å 命中。1 Å 召回和 2 Å 召回分别是命中的真实位点所占的比例。
- 另报全部真实位点定位误差的中位数，以及平均每个真实位点对应的预测位点数。
- 本方法（共同见证点）在同一批结构上用完全相同的规则计算，所以外部方法只需交回原始预测坐标，不要自行过滤或换阈值。
