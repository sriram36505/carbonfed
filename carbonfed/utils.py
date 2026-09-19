"""Replay buffer, seeding and logging helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """Fixed-size replay buffer holding graph observations."""

    def __init__(self, capacity: int, n_regions: int, node_dim: int,
                 history_len: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr, self.size = 0, 0
        self.node = np.zeros((capacity, n_regions, node_dim), dtype=np.float32)
        self.hist = np.zeros((capacity, n_regions, history_len), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.cost = np.zeros(capacity, dtype=np.float32)
        self.viol = np.zeros(capacity, dtype=np.float32)
        self.node2 = np.zeros_like(self.node)
        self.hist2 = np.zeros_like(self.hist)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action_flat, cost, viol, obs2, done):
        i = self.ptr
        self.node[i] = obs["node"]
        self.hist[i] = obs["carbon_history"]
        self.act[i] = action_flat
        self.cost[i] = cost
        self.viol[i] = viol
        self.node2[i] = obs2["node"]
        self.hist2[i] = obs2["carbon_history"]
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return (t(self.node), t(self.hist), t(self.act), t(self.cost),
                t(self.viol), t(self.node2), t(self.hist2), t(self.done))


def flatten_action(action, n_regions: int) -> np.ndarray:
    return np.concatenate([
        np.asarray(action["routing"], dtype=np.float32).ravel(),
        np.asarray(action["freq"], dtype=np.float32),
        np.asarray(action["deferral"], dtype=np.float32),
    ])


def ring_adjacency(n: int, self_loops: bool = True) -> torch.Tensor:
    """Inter-region topology: ring plus optional self-loops.

    Replace with the real migration topology of the deployment when known.
    """
    adj = torch.zeros(n, n)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[i, (i - 1) % n] = 1
        if self_loops:
            adj[i, i] = 1
    return adj


def fully_connected_adjacency(n: int) -> torch.Tensor:
    return torch.ones(n, n)


class JsonLogger:
    """Append-only JSONL logger for training metrics."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, **kw):
        with self.path.open("a") as f:
            f.write(json.dumps(kw) + "\n")

    def load(self):
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
