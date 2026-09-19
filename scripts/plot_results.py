#!/usr/bin/env python3
"""Generate the result figures from training logs and evaluation output.

Produces the convergence, constraint-satisfaction, carbon-reduction and
daily-mechanism figures used in Section 5. All curves are drawn from logged
runs -- nothing here is synthetic.

Usage:
    python scripts/plot_results.py --logs runs --eval results/evaluation.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_runs(log_dir: Path, pattern: str):
    """Load train.jsonl from every matching run directory."""
    out = []
    for p in sorted(log_dir.glob(f"{pattern}*/train.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if rows:
            out.append(rows)
    return out


def mean_std(runs, key):
    """Stack a metric across seeds, truncating to the shortest run."""
    n = min(len(r) for r in runs)
    arr = np.array([[row[key] for row in r[:n]] for r in runs])
    return arr.mean(0), arr.std(0)


def plot_convergence(log_dir: Path, out: Path):
    """Normalized episodic return with seed variance band (paper Fig. 7)."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for pattern, label, style in [("federated", "CarbonFed (proposed)", "-"),
                                  ("centralized", "Centralized (oracle)", "--"),
                                  ("isolated", "Isolated GST-TD3", ":")]:
        runs = load_runs(log_dir, pattern)
        if not runs:
            continue
        m, s = mean_std(runs, "carbon_kg")
        ret = 1.0 - (m - m.min()) / (m.max() - m.min() + 1e-9)
        band = s / (m.max() - m.min() + 1e-9)
        x = np.arange(len(ret))
        ax.plot(x, ret, style, label=label)
        ax.fill_between(x, ret - band, ret + band, alpha=0.18, linewidth=0)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Normalized episodic return")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(out / "fig_convergence.png", dpi=300); plt.close(fig)


def plot_constraint(log_dir: Path, out: Path, budget: float = 0.02):
    """SLO violation rate and dual variable over training (paper Fig. 8)."""
    runs = load_runs(log_dir, "federated")
    if not runs:
        return
    v, vs = mean_std(runs, "violation_rate")
    lam, _ = mean_std(runs, "lam")
    x = np.arange(len(v))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, 100 * v, "-", color="k", label="SLO violation rate")
    ax.fill_between(x, 100 * (v - vs), 100 * (v + vs), alpha=0.18, color="k", linewidth=0)
    ax.axhline(100 * budget, ls="--", c="gray", lw=1)
    ax.set_xlabel("Training episode"); ax.set_ylabel("SLO violation rate (%)")
    ax2 = ax.twinx(); ax2.plot(x, lam, ":", color="C3", label=r"dual $\lambda$")
    ax2.set_ylabel(r"Dual variable $\lambda$")
    fig.tight_layout(); fig.savefig(out / "fig_constraint.png", dpi=300); plt.close(fig)


def plot_reduction(eval_json: Path, out: Path):
    """Carbon reduction by method with seed error bars (paper Fig. 9)."""
    data = json.loads(eval_json.read_text())["summary"]
    names = [k for k in data if k != "tests"]
    means = [data[k]["carbon_reduction_mean"] for k in names]
    stds = [data[k]["carbon_reduction_std"] for k in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(names, means, yerr=stds, capsize=3, color="0.75", edgecolor="k")
    ax.set_ylabel("Carbon reduction (%)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_reduction.png", dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="runs")
    ap.add_argument("--eval", default="results/evaluation.json")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log_dir = Path(a.logs)
    if log_dir.exists():
        plot_convergence(log_dir, out)
        plot_constraint(log_dir, out)
    if Path(a.eval).exists():
        plot_reduction(Path(a.eval), out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
