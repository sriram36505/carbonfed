#!/usr/bin/env python3
"""Prepare carbon-intensity and workload traces for the simulator.

Expects a CSV per region with a `carbon_intensity_gco2_kwh` column at the
control interval, and a workload CSV with a normalized `demand` column.
Outputs a single .npz consumed by DataCenterFleetEnv.

Usage:
    python scripts/prepare_data.py --carbon data/raw/*.csv \
        --workload data/raw/workload.csv --out data/traces.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_csv_column(path: Path, column: str) -> np.ndarray:
    rows = path.read_text().splitlines()
    header = [h.strip() for h in rows[0].split(",")]
    if column not in header:
        raise KeyError(f"{path}: column '{column}' not found; have {header}")
    idx = header.index(column)
    vals = []
    for line in rows[1:]:
        if not line.strip():
            continue
        try:
            vals.append(float(line.split(",")[idx]))
        except (ValueError, IndexError):
            continue
    return np.asarray(vals, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon", nargs="+", required=True,
                    help="one CSV per region, in region order")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--carbon-column", default="carbon_intensity_gco2_kwh")
    ap.add_argument("--demand-column", default="demand")
    ap.add_argument("--out", default="data/traces.npz")
    a = ap.parse_args()

    series = [load_csv_column(Path(p), a.carbon_column) for p in a.carbon]
    T = min(len(s) for s in series)
    carbon = np.stack([s[:T] for s in series])
    demand = load_csv_column(Path(a.workload), a.demand_column)[:T]
    if demand.max() > 1.0:
        demand = demand / demand.max()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, carbon=carbon, demand=demand)
    print(f"regions={carbon.shape[0]} steps={T} -> {out}")


if __name__ == "__main__":
    main()
