"""Smoke tests: run with `pytest -q`."""
import sys, warnings
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", category=UserWarning)

from carbonfed.agent import CarbonFedAgent, cvar_from_quantiles, quantile_huber_loss
from carbonfed.baselines import REGISTRY
from carbonfed.env import DataCenterFleetEnv, FleetConfig
from carbonfed.federated import (carbon_weighted_aggregate, carbon_weights,
                                 clip_and_noise, flat_norm, parameter_delta)
from carbonfed.utils import ReplayBuffer, flatten_action, ring_adjacency, set_seed

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "configs/default.yaml"))


def small_env(steps=40, seed=0):
    return DataCenterFleetEnv(FleetConfig(episode_steps=steps, seed=seed))


def test_env_step_shapes_and_conservation():
    env = small_env()
    o = env.reset()
    assert o["node"].shape == (4, 5)
    a = {"routing": np.full((4, 4), .25), "freq": np.ones(4), "deferral": np.zeros(4)}
    o2, cost, viol, done, info = env.step(a)
    assert cost > 0 and 0.0 <= viol
    assert info["p95_latency_ms"] > 0
    assert np.all(env.util >= 0) and np.all(env.util <= 1)


def test_routing_rows_sum_to_one():
    set_seed(0)
    env = small_env()
    agent = CarbonFedAgent(CFG["agent"], 5, 4, ring_adjacency(4))
    a = agent.act(env.reset(), noise=0.1)
    assert np.allclose(a["routing"].sum(axis=1), 1.0, atol=1e-5)
    assert np.all(a["freq"] >= CFG["agent"]["freq_min"] - 1e-6)
    assert np.all(a["deferral"] <= CFG["agent"]["deferral_max"] + 1e-6)


def test_cvar_monotonic_in_alpha():
    q = torch.linspace(0, 1, 50).unsqueeze(0)
    assert cvar_from_quantiles(q, 0.1) > cvar_from_quantiles(q, 1.0)


def test_quantile_huber_positive():
    pred = torch.randn(8, 25)
    target = torch.randn(8, 25)
    assert quantile_huber_loss(pred, target) >= 0


def test_carbon_weights_normalize_and_reduce_to_fedavg():
    w = carbon_weights([100, 200], [0.2, 0.8], kappa=1.5)
    assert abs(sum(w) - 1.0) < 1e-9
    w0 = carbon_weights([100, 100], [0.2, 0.8], kappa=0.0)
    assert abs(w0[0] - w0[1]) < 1e-9          # kappa=0 -> plain FedAvg


def test_dp_clipping_bounds_norm():
    upd = {"a": torch.ones(100) * 10.0}
    out = clip_and_noise(upd, clip_norm=1.0, epsilon=8.0, delta=1e-5)
    assert out["a"].shape == upd["a"].shape
    # stronger privacy injects more noise
    lo = clip_and_noise(upd, 1.0, 0.5, 1e-5)
    assert flat_norm(lo) > flat_norm(out)


def test_aggregate_preserves_keys():
    set_seed(0)
    agents = [CarbonFedAgent(CFG["agent"], 5, 4, ring_adjacency(4)) for _ in range(2)]
    g = agents[0].get_parameters()
    ups = [parameter_delta(g, a.get_parameters()) for a in agents]
    new = carbon_weighted_aggregate(g, ups, [10, 20], [0.3, 0.6], 1.5)
    assert set(new) == set(g)


def test_update_runs_and_moves_lambda():
    set_seed(0)
    env = small_env()
    agent = CarbonFedAgent(CFG["agent"], 5, 4, ring_adjacency(4))
    buf = ReplayBuffer(500, 4, 5, env.history_len, 4 * 4 + 8)
    o = env.reset()
    for _ in range(40):
        a = agent.act(o, noise=0.2)
        o2, c, v, d, _ = env.step(a)
        buf.add(o, flatten_action(a, 4), c / 1e6, v, o2, d)
        o = o2
    m = [agent.update(buf.sample(16)) for _ in range(4)][-1]
    assert np.isfinite(m["critic_loss"]) and m["lambda"] > 0


def test_all_baselines_produce_valid_actions():
    env = small_env()
    o = env.reset()
    for name, cls in REGISTRY.items():
        a = cls(4).act(o)
        assert np.allclose(a["routing"].sum(axis=1), 1.0, atol=1e-6), name
        env.step(a)
