# -*- coding: utf-8 -*-
"""Merge per-seed OOF parquets into oof_predictions.parquet schema."""
import glob
import pandas as pd

parts = []
for f in glob.glob("GSPH_H3/results/ablation/oof_*_s*.parquet"):
    seed = int(f.rsplit("_s", 1)[1].split(".")[0])
    df = pd.read_parquet(f)
    df["seed"] = seed
    parts.append(df)
out = pd.concat(parts, ignore_index=True)
out.to_parquet("results/tables/oof_predictions.parquet", index=False)
print(out.groupby(["task" if "task" in out else out.columns[0], "seed"]).size().head())
print("rows:", len(out))
