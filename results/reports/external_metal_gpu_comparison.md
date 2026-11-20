# Metal3D 与 AllMetal3D 在 32 个 Zn 时间外测结构上的结果

## 结论
在同一批 32 个结构上，Metal3D 的 1 Å 召回为 0.957、2 Å 召回为 0.986，AllMetal3D 分别为 0.855 和 0.870。参考位点是自动质控通过的 69 个 Zn 位点。两种方法的 64 次运行全部成功，没有超时或超出内存上限。本方法在这 32 个结构上的逐位点数字还没有从计算端取回；下表中本方法一行用的是它所在的整个 Zn 队列（300 个结构）的数字，两者不是同一批结构，不能直接比较。

## 口径
- 结构：`experiments/external_metal_site_run/zn_time_split_32.txt` 中的 32 个结构，交回的结果与这份清单逐一对上。
- 参考位点：`results/tables/external_metal_site_qc.csv` 中这 32 个结构的 Zn 位点。主口径是 `automatic_QC_pass=True` 的 69 个位点，另报全部 86 个位点。本方法自己的参考位点总数也是 69，但有 2 个结构各差 1 个位点（30ct 多 1 个，9rfp 少 1 个），所以两套参考集并不完全相同。
- 对每个参考位点，取距离它最近的预测位点，两者距离记为定位误差；误差 ≤1 Å 记为 1 Å 命中，≤2 Å 记为 2 Å 命中；召回是命中的参考位点所占的比例。外部方法的预测按官方阈值输出，没有再做过滤。

## 结果

| 方法 | 参考集 | 位点数 | 1 Å 召回 | 2 Å 召回 | 中位定位误差（Å） | 每个参考位点对应的预测数 |
|---|---|---|---|---|---|---|
| Metal3D | 自动质控通过 | 69 | 0.957 | 0.986 | 0.225 | 4.51 |
| AllMetal3D | 自动质控通过 | 69 | 0.855 | 0.870 | 0.409 | 2.71 |
| Metal3D | 全部 Zn | 86 | 0.860 | 0.895 | 0.254 | 3.73 |
| AllMetal3D | 全部 Zn | 86 | 0.744 | 0.756 | 0.455 | 2.28 |
| 本方法（整个 Zn 队列，300 个结构，非同一批） | 本方法参考集 | 640 | 0.902 | 0.992 | 0.142 | 48.3（过滤冲突后的聚类数） |

## 运行情况
- 设备：RTX 4070，torch 2.5.1+cu121，CUDA 12.1，batch 8，一次一个结构。
- 每个结构的耗时中位数：Metal3D 117 秒，AllMetal3D 216 秒；最长分别为 1378 秒和 1162 秒。
- 峰值内存最高 5.9 GB，峰值显存最高 2.3 GB，都在上限之内。

## 与说明的偏差
- Python 用的是 3.12.12，说明里要求 3.10。其余版本都与说明一致：torch 2.5.1、numpy 1.26.4、pandas 2.2.3、scipy 1.12.0、moleculekit 1.9.4、allmetal3d 0.7.3、Metal3D 提交 09cbfb3c；AllMetal3D 三个权重文件的 SHA256 与 `prepare_tools.py` 中的记录一致。

## 文件
`results/tables/external_metal_gpu_predictions.csv`、`external_metal_gpu_runtime.csv`、`external_metal_gpu_software.csv`、`external_allmetal_gpu_details.jsonl`、`external_metal_gpu_metrics.csv`。
