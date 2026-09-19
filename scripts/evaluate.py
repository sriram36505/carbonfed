#!/usr/bin/env python3
"""Evaluate CarbonFed against the baselines of Section 5.

Reports operational carbon reduction relative to round-robin, the SLO violation
rate and p95 tail latency, aggregated over seeds with mean and standard
deviation, and runs a Wilcoxon signed-rank test against the chosen reference.

Usage:
    python scripts/evaluate.py --checkpoint runs/federated_seed0/checkpoint.pt
    python scripts/evaluate.py --baselines-only
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=UserWarning)

from carbonfed.agent import CarbonFedAgent
from carbonfed.baselines import REGISTRY
from carbonfed.env import DataCenterFleetEnv, FleetConfig
from carbonfed.utils import ring_adjacency, set_seed


def make_env(cfg, seed):
    e = cfg["env"]
    return DataCenterFleetEnv(FleetConfig(
        n_regions=e["n_regions"], dt_minutes=e["dt_minutes"],
        episode_steps=e["episode_steps"], carbon_min=e["carbon_min"],
        carbon_max=e["carbon_max"], pue_range=tuple(e["pue_range"]),
        slo_latency_ms=e["slo_latency_ms"], deferral_max=e["deferral_max"],
        freq_min=e["freq_min"], seed=seed))


def rollout(env, policy):
    o = env.reset()
    carbon, viols, lat, steps = 0.0, 0, 0.0, 0
    while True:
        a = policy(o)
        o, cost, viol, done, info = env.step(a)
        carbon += info["carbon_g"]
        viols += int(viol > 0)
        lat += info["p95_latency_ms"]
        steps += 1
        if done:
            break
    return {"carbon_kg": carbon / 1000.0,
            "violation_rate": viols / steps,
            "p95_latency_ms": lat / steps}


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank test (normal approximation).

    Returns (statistic, p-value). For n < 10 treat the p-value as indicative
    only; SciPy's exact test is preferable when available.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 0.0, 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = r[d > 0].sum()
    w_minus = r[d < 0].sum()
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return float(w), 1.0
    z = (w - mu) / sigma
    from math import erfc, sqrt
    return float(w), float(erfc(abs(z) / sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--out", default="results/evaluation.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    n = cfg["env"]["n_regions"]
    results = {}

    for name, cls in REGISTRY.items():
        runs = []
        for s in args.seeds:
            set_seed(s)
            env = make_env(cfg, s)
            sched = cls(n, freq_min=cfg["env"]["freq_min"],
                        deferral_max=cfg["env"]["deferral_max"])
            runs.append(rollout(env, sched.act))
        results[name] = runs

    if args.checkpoint and not args.baselines_only:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        runs = []
        for s in args.seeds:
            set_seed(s)
            env = make_env(cfg, s)
            agent = CarbonFedAgent(cfg["agent"], node_dim=env._obs()["node"].shape[1],
                                   n_regions=n, adjacency=ring_adjacency(n))
            agent.load_parameters(ck["global"])
            runs.append(rollout(env, lambda o: agent.act(o, noise=0.0)))
        results["carbonfed"] = runs

    ref = np.array([r["carbon_kg"] for r in results["round_robin"]])
    print(f"{'method':<20}{'carbon red. (%)':>18}{'SLO viol. (%)':>16}{'p95 (ms)':>12}")
    print("-" * 66)
    summary = {}
    for name, runs in results.items():
        c = np.array([r["carbon_kg"] for r in runs])
        red = 100.0 * (ref - c) / ref
        v = 100.0 * np.array([r["violation_rate"] for r in runs])
        l = np.array([r["p95_latency_ms"] for r in runs])
        summary[name] = {
            "carbon_reduction_mean": float(red.mean()),
            "carbon_reduction_std": float(red.std()),
            "violation_mean": float(v.mean()),
            "latency_mean": float(l.mean()),
        }
        print(f"{name:<20}{red.mean():>12.1f} ± {red.std():<4.1f}"
              f"{v.mean():>12.2f}    {l.mean():>10.1f}")

    if "carbonfed" in results:
        for other in ["carbon_aware_opt", "heuristic_carbon"]:
            a = [r["carbon_kg"] for r in results["carbonfed"]]
            b = [r["carbon_kg"] for r in results[other]]
            w, p = wilcoxon(a, b)
            print(f"Wilcoxon carbonfed vs {other}: W={w:.1f}, p={p:.4f}")
            summary.setdefault("tests", {})[other] = {"W": w, "p": p}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
