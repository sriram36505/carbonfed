#!/usr/bin/env python3
"""Train CarbonFed across federated regions (Algorithm 1).

Each client runs K local steps on its own environment, then contributes a
clipped and noised parameter update to the carbon-weighted aggregation. The
refreshed global model is personalized locally via the Ditto objective before
the next round.

Usage:
    python scripts/train.py --config configs/default.yaml --seed 0
    python scripts/train.py --mode isolated       # no federation (ablation)
    python scripts/train.py --mode centralized    # pooled data oracle
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=UserWarning)

from carbonfed.agent import CarbonFedAgent
from carbonfed.env import DataCenterFleetEnv, FleetConfig
from carbonfed.federated import (carbon_weighted_aggregate, clip_and_noise,
                                 parameter_delta)
from carbonfed.utils import (JsonLogger, ReplayBuffer, flatten_action,
                             ring_adjacency, set_seed)


def build_client(cfg, seed, client_id):
    e = cfg["env"]
    fleet = FleetConfig(
        n_regions=e["n_regions"], dt_minutes=e["dt_minutes"],
        episode_steps=e["episode_steps"], carbon_min=e["carbon_min"],
        carbon_max=e["carbon_max"], pue_range=tuple(e["pue_range"]),
        p_idle_w=e["p_idle_w"], p_peak_w=e["p_peak_w"],
        slo_latency_ms=e["slo_latency_ms"], deferral_max=e["deferral_max"],
        freq_min=e["freq_min"], seed=seed * 100 + client_id,
    )
    env = DataCenterFleetEnv(fleet)
    n = e["n_regions"]
    agent = CarbonFedAgent(cfg["agent"], node_dim=env._obs()["node"].shape[1],
                           n_regions=n, adjacency=ring_adjacency(n),
                           device=cfg["train"]["device"])
    buf = ReplayBuffer(cfg["train"]["buffer_capacity"], n,
                       env._obs()["node"].shape[1], env.history_len,
                       n * n + 2 * n, device=cfg["train"]["device"])
    return env, agent, buf


def run_episode(env, agent, buf, cfg, noise, train=True):
    o = env.reset()
    totals = {"carbon": 0.0, "viol": 0.0, "steps": 0, "latency": 0.0}
    n = env.n
    for _ in range(env.cfg.episode_steps):
        a = agent.act(o, noise=noise if train else 0.0)
        o2, cost, viol, done, info = env.step(a)
        if train:
            buf.add(o, flatten_action(a, n), cost / 1e6, viol, o2, done)
            if buf.size > cfg["train"]["warmup_steps"]:
                for _ in range(cfg["train"]["updates_per_step"]):
                    agent.update(buf.sample(cfg["train"]["batch_size"]))
        totals["carbon"] += info["carbon_g"]
        totals["viol"] += float(viol > 0)
        totals["latency"] += info["p95_latency_ms"]
        totals["steps"] += 1
        o = o2
        if done:
            break
    s = max(totals["steps"], 1)
    return {
        "carbon_kg": totals["carbon"] / 1000.0,
        "violation_rate": totals["viol"] / s,
        "p95_latency_ms": totals["latency"] / s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--mode", default="federated",
                    choices=["federated", "isolated", "centralized"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.episodes:
        cfg["train"]["episodes"] = args.episodes
    set_seed(args.seed)

    fed = cfg["federated"]
    n_clients = 1 if args.mode == "centralized" else fed["n_clients"]
    clients = [build_client(cfg, args.seed, i) for i in range(n_clients)]

    out = Path(args.out or f"{cfg['train']['log_dir']}/{args.mode}_seed{args.seed}")
    logger = JsonLogger(out / "train.jsonl")
    print(f"[CarbonFed] mode={args.mode} clients={n_clients} seed={args.seed}")

    global_params = clients[0][1].get_parameters()
    gen = torch.Generator().manual_seed(args.seed)

    for ep in range(cfg["train"]["episodes"]):
        noise = cfg["train"]["exploration_noise"]
        metrics, updates, sizes, potentials = [], [], [], []

        for env, agent, buf in clients:
            before = agent.get_parameters()
            m = run_episode(env, agent, buf, cfg, noise, train=True)
            metrics.append(m)
            if args.mode == "federated":
                updates.append(parameter_delta(before, agent.get_parameters()))
                sizes.append(buf.size)
                potentials.append(env.carbon_saving_potential())

        # ---- periodic federated aggregation (Algorithm 1, lines 11-13)
        if args.mode == "federated" and (ep + 1) % fed["aggregation_interval"] == 0:
            priv = [clip_and_noise(u, fed["dp_clip_norm"], fed["dp_epsilon"],
                                   fed["dp_delta"], generator=gen) for u in updates]
            global_params = carbon_weighted_aggregate(
                global_params, priv, sizes, potentials, fed["carbon_kappa"])
            for _, agent, _ in clients:      # broadcast, then personalize (Ditto)
                agent.load_parameters(global_params)

        agg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        agg.update(episode=ep, lam=float(clients[0][1].lam))
        logger.log(**agg)
        if ep % max(1, cfg["train"]["eval_every"]) == 0:
            print(f"ep {ep:4d}  carbon {agg['carbon_kg']:10.1f} kg  "
                  f"viol {agg['violation_rate']:.3f}  "
                  f"p95 {agg['p95_latency_ms']:6.1f} ms  lambda {agg['lam']:.3f}")

    torch.save({"global": global_params,
                "clients": [a.state_dict() for _, a, _ in clients]},
               out / "checkpoint.pt")
    print(f"[CarbonFed] saved -> {out}")


if __name__ == "__main__":
    main()
