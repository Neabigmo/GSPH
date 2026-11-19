# Metal3D 与 AllMetal3D 的超算 CPU 运行

在离子位点时间外测集的 32 个合格 Zn 结构上运行 Metal3D 和 AllMetal3D 两个外部方法，供与共同见证点方法比较。两者都用官方代码和官方权重，在 CPU 上推理，模型定义和阈值不变。

## 输入

- `zn_time_split_32.txt`：32 个 PDB ID。它们来自 `experiments/exp09_metal_site_pilot/zn_ids.csv` 中 `time_split_test=True` 的结构，再按 X-ray、分辨率 ≤2 Å、R_free ≤0.25 筛选（见 `results/tables/external_metal_structure_audit.csv`）。
- `inputs/*.pdb.gz`：上面 32 个结构的输入文件，共 4 MB，已随仓库提供，计算节点不需要联网。
  - 生成方式：从 `https://files.rcsb.org/download/{ID}.cif.gz` 下载，用 gemmi 0.7.5 只保留第一个 model 后写成 PDB。
  - 解压后的 SHA256 记录在 `results/tables/external_metal_inputs.csv`。

## 代码与权重来源

| 方法 | 代码 | 权重 | 体积 |
|---|---|---|---|
| Metal3D | GitHub `lcbc-epfl/metal-site-prediction`，提交 `09cbfb3c`；只取其中的 `Metal3D/` 目录 | 随代码提供：`Metal3D/weights/metal_0.5A_v3_d0.2_16Abox.pth` | 代码 zip 约 100 MB |
| AllMetal3D | PyPI `allmetal3d==0.7.3` | Hugging Face `simonduerr/allmetal3d`，下载后校验 SHA256 | 0.39 GB |

`prepare_tools.py` 对源码只做五处改动，每处都先核对原文出现的次数，对不上就报错退出：
1. 设备：去掉 AllMetal3D 导入时的 CUDA 检查；两者的体素缓冲区在没有 GPU 时建在 CPU 上。
2. 权重：加载时加 `map_location=device`。
3. 进程数：原代码按整台节点的核数（`multiprocessing.cpu_count()`）开 joblib 进程，在共享节点上会开出远超分配的进程，内存也随之暴涨；改为使用 `SLURM_CPUS_PER_TASK`。
4. 删除 Metal3D `utils/helpers.py` 里没用到的 `from turtle import width`，这一行会要求装 Tk 图形库。
5. 前向推理包在 `torch.no_grad()` 里：输出不变，但不再为反向传播保留中间激活，内存更省。

## 步骤

1. **登录节点，一次性准备（需要联网）。** `ENV_DIR` 和 `TOOLS` 都用新建的目录，Python 需为 3.10。
   ```bash
   ENV_DIR=<新环境目录> TOOLS=<新工具目录> BASE_PYTHON=python3.10 \
     bash experiments/external_metal_site_run/setup_env.sh
   ```
   - 依赖版本固定在 `constraints.txt` 中，torch 用 2.5.1 的 CPU 版。
   - 有 conda（或 micromamba）时，脚本先用 conda 建 Python 3.10 环境，装好 numpy 1.26.4、pandas 2.2.3、scipy 1.12.0 的预编译包。可以用 `CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge` 走清华镜像。
   - 其余编译型依赖只装 manylinux2014 预编译包（glibc 2.17 就能用），不会退回到源码编译。只有纯 Python 的 halo 和带小型 C 扩展的 moleculekit 从源码安装，gcc 7.3 就能编译。
   - 脚本设置了 `PYTHONNOUSERSITE=1`，避免读到 `~/.local` 里的旧包。
2. **提交作业数组。** 一个结构一个任务，每个任务 16 核、32 GB 内存上限、最长 12 小时，32 个结构同时排队，batch 默认为 8。两种方法各提交一次；分区在提交时用 `-p` 指定。
   ```bash
   export PYTHON=<新环境目录>/bin/python TOOLS=<新工具目录> OUT_DIR=<仓库外的输出目录>
   sbatch -p <分区> --export=ALL,METHOD=metal3d scripts/slurm/external_metal.sh
   sbatch -p <分区> --export=ALL,METHOD=allmetal3d scripts/slurm/external_metal.sh
   ```
   - 建议先用 `--array=0-1` 校准耗时和峰值内存。每个任务会把峰值内存（`peak_rss_mb`）写进结果。
   - 若某个任务超内存，用 `BATCH_SIZE=2` 重提这一个下标即可。如果队列核数充足，可以用 `sbatch -c 32 --mem=48G` 进一步加速；进程数会自动跟随分配的核数。
3. **汇总。** 全部任务结束后在登录节点运行：
   ```bash
   $PYTHON experiments/external_metal_site_run/run.py --collect --out $OUT_DIR
   ```
   只把生成的小表提交推送到 `paper-strengthening`。原始输出留在 `OUT_DIR`，不入库。

## 输出（写入 `results/tables/`，合计几十 KB）

| 文件 | 列 |
|---|---|
| `external_metal_cpu_predictions.csv` | method, pdb_id, rank, element, x, y, z, probability（每个预测位点一行，按官方探针文件的顺序；probability 取自探针文件的占有率列） |
| `external_metal_cpu_runtime.csv` | method, pdb_id, status, seconds, n_sites, cpus, batch_size, peak_rss_mb, torch |
| `external_allmetal_cpu_details.jsonl` | AllMetal3D 给出的位置置信度及种类、几何概率，每个结构一行 |

指标在云端计算：采用一对一匹配，统计 1 Å 和 2 Å 下的精确率、召回率、F1 和定位误差。参考位点为 `results/tables/external_metal_site_qc.csv` 中通过自动质控的 69 个 Zn。
