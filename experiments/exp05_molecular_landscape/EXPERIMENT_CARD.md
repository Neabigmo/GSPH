# EXP-05: Path / Barcode Landscape

- **Purpose**: Landscape-level statistics of the path complex across
  structures, graphs, and orders.
- **Data**: 125 structure x graph x order combinations (deep subset).
- **Method**: `deep_subset.py` / `deep_complex_dump.py` (per-structure
  path/support/complex dumps incl. directed vs undirected path counts);
  `aggregate.py` -> path_landscape / barcode_landscape tables.
- **Key results**: deep-subset totals (clash06): RNA 59,945 support
  records / 1,873 H0-H2 intervals; PD 12,488 / 294; combined
  72,433 / 2,167 -- 100% certified valid, orders k0-k3.
- **Status**: DONE.
- **Artifacts**: `results/tables/path_landscape.parquet`,
  `barcode_landscape.parquet`, `results/reports/deep_subset_v33.csv`.
