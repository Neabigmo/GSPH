# -*- coding: utf-8 -*-
"""Unified experiment dispatcher.

Usage:
  python scripts/run_experiment.py --config configs/experiments/exp03_filtration_comparison.yaml [--dry-run]

Resolves the config's script entry/entries, verifies required manifests,
and executes them. Large runs should set --shard/--n-shards for SLURM
array execution (pass-through args appended to the script command).
"""
import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # minimal fallback for the flat key: value style used in our configs
    cfg = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" in line and not line.startswith(("-", " ")):
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--n-shards", type=int, default=None)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()

    cfg_path = Path(a.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    scripts = cfg.get("scripts") or ([cfg["script"]]
                                   if cfg.get("script") else
                                   [str(p) for p in
                                    (ROOT / "experiments" /
                                     cfg.get("experiment_id", "")).glob("*.py")]
                                   if (ROOT / "experiments" /
                                       cfg.get("experiment_id", "")).exists()
                                   else [])
    if not scripts:
        sys.exit(f"no scripts resolved for {cfg_path}")
    env_note = ("Set PYTHONPATH or `pip install -e .` so `cspph`, "
                "`gsph6`, `runners`, and task adapters resolve.")
    for s in scripts:
        sp = Path(s)
        if not sp.is_absolute():
            sp = ROOT / sp
        cmd = [sys.executable, str(sp)]
        if a.shard is not None:
            cmd += ["--shard", str(a.shard)]
        if a.n_shards is not None:
            cmd += ["--n-shards", str(a.n_shards)]
        cmd += a.extra
        print("+", " ".join(cmd))
        if not a.dry_run:
            r = subprocess.run(cmd, cwd=str(ROOT))
            if r.returncode != 0:
                sys.exit(r.returncode)
    print("[run_experiment] done")


if __name__ == "__main__":
    main()
