"""Baseline schedulers used in Section 5 of the paper.

Each baseline exposes ``act(obs)`` returning the same action dict as the agent:
round-robin (carbon-agnostic reference), greedy lowest-latency, heuristic
carbon-greedy, and a carbon-aware optimization in the spirit of Radovanovic
et al. (2022).
"""

from __future__ import annotations

import numpy as np


class BaseScheduler:
    def __init__(self, n_regions: int, freq_min: float = 0.6, deferral_max: float = 0.30):
        self.n = n_regions
        self.freq_min = freq_min
        self.deferral_max = deferral_max

    def act(self, obs):
        raise NotImplementedError

    def _uniform(self):
        return np.full((self.n, self.n), 1.0 / self.n)


class RoundRobin(BaseScheduler):
    """Carbon-agnostic reference: split load evenly, no DVFS, no deferral."""

    def act(self, obs):
        return {
            "routing": self._uniform(),
            "freq": np.ones(self.n),
            "deferral": np.zeros(self.n),
        }


class GreedyLatency(BaseScheduler):
    """Serve locally and run at full speed: minimises latency, ignores carbon."""

    def act(self, obs):
        return {
            "routing": np.eye(self.n),
            "freq": np.ones(self.n),
            "deferral": np.zeros(self.n),
        }


class HeuristicCarbonGreedy(BaseScheduler):
    """Send everything to the momentarily cleanest region and defer aggressively."""

    def act(self, obs):
        ci = obs["node"][:, 2]                     # normalized carbon intensity
        target = int(np.argmin(ci))
        routing = np.zeros((self.n, self.n))
        routing[:, target] = 1.0
        return {
            "routing": routing,
            "freq": np.ones(self.n),
            "deferral": np.full(self.n, self.deferral_max),
        }


class CarbonAwareOptimization(BaseScheduler):
    """Forecast-driven softmax allocation with a queue-aware penalty.

    Approximates the production carbon-aware optimizer: route in proportion to
    exp(-beta * predicted carbon intensity), damped by current queue occupancy,
    and defer in proportion to how far above its daily mean a region currently is.
    """

    def __init__(self, n_regions, beta: float = 6.0, **kw):
        super().__init__(n_regions, **kw)
        self.beta = beta

    def act(self, obs):
        ci = obs["node"][:, 2]
        queue = obs["node"][:, 1]
        score = -self.beta * ci - 2.0 * queue
        w = np.exp(score - score.max())
        w = w / w.sum()
        routing = np.tile(w, (self.n, 1))

        rel = ci - ci.mean()
        defer = np.clip(rel, 0, None)
        defer = self.deferral_max * (defer / (defer.max() + 1e-9))
        freq = np.clip(1.0 - 0.3 * (ci - ci.min()), self.freq_min, 1.0)
        return {"routing": routing, "freq": freq, "deferral": defer}


REGISTRY = {
    "round_robin": RoundRobin,
    "greedy_latency": GreedyLatency,
    "heuristic_carbon": HeuristicCarbonGreedy,
    "carbon_aware_opt": CarbonAwareOptimization,
}
