# Metal3D 与 AllMetal3D 的 GPU 运行

在离子位点时间外测集的 32 个合格 Zn 结构上运行 Metal3D 和 AllMetal3D 两个外部方法，供与共同见证点方法比较。两者都用官方代码和官方权重，设备设为 CUDA，不改模型。

## 输入

- `zn_time_split_32.txt`：32 个 PDB ID。它们来自 `experiments/exp09_metal_site_pilot/zn_ids.csv` 中 `time_split_test=True` 的结构，再按 X-ray、分辨率 ≤2 Å、R_free ≤0.25 筛选（见 `results/tables/external_metal_structure_audit.csv`）。
- `inputs/*.pdb.gz`：上面 32 个结构的输入文件，共 4 MB，已随仓库提供，运行时不必再下载。
  - 生成方式：从 `https://files.rcsb.org/download/{ID}.cif.gz` 下载，用 gemmi 0.7.5 只保留第一个 model 后写成 PDB。
  - 解压后的 SHA256 记录在 `results/tables/external_metal_inputs.csv`。

## 代码与权重来源

| 方法 | 代码 | 权重 | 体积 |
|---|---|---|---|
| Metal3D | GitHub `lcbc-epfl/metal-site-prediction`，提交 `09cbfb3c`；脚本只取其中的 `Metal3D/` 目录 | 随代码，`Metal3D/weights/metal_0.5A_v3_d0.2_16Abox.pth` | 代码 zip 约 100 MB |
| AllMetal3D | PyPI `allmetal3d==0.7.3` | Hugging Face `simonduerr/allmetal3d`，下载后校验 SHA256 | 0.39 GB |

## 运行（Windows）

一条命令完成依赖、下载和推理。`--work` 是临时目录，用来放代码、权重和原始输出，合计约 0.6 GB。

```
E:\anaconda3\envs\pytorch-clean\python.exe experiments\external_metal_site_gpu\run_gpu.py --work E:\gsph_external_metal --method both
```

- 缺少的依赖用 `pip install --no-deps --target <work>\pydeps` 装进临时目录，不改动 conda 环境本身。
- 如果 moleculekit 没有可用的 Windows wheel，先运行 `conda install -c acellera moleculekit` 装好，再重跑上面的命令。
- AllMetal3D 读取的 `HOME` 和 `USERPROFILE` 只在本脚本进程内指向 `<work>\home`。

## 输出（写入 `results/tables/`，都很小）

| 文件 | 列 |
|---|---|
| `external_metal_gpu_predictions.csv` | method, pdb_id, rank, x, y, z, confidence（每个预测位点一行，按置信度排名） |
| `external_metal_gpu_runtime.csv` | method, pdb_id, seconds, status, n_predictions, device |
| `external_metal_gpu_software.csv` | package, version |
| `external_allmetal_gpu_details.jsonl` | AllMetal3D 给出的种类和几何概率，每个结构一行 |

只需把这四个文件提交推送到 `paper-strengthening`，临时目录不入库。指标计算在云端完成：一对一匹配，统计 1 Å 和 2 Å 下的精确率、召回率、F1 和定位误差，参考位点用 `results/tables/external_metal_site_qc.csv` 中通过自动质控的 69 个 Zn。
