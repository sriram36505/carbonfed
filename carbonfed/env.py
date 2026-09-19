"""Discrete-time simulator for a geographically distributed data-center fleet.

The environment advances in fixed control steps (default 5 min). At each step the
agent chooses, for every region: the fraction of incoming requests routed to each
region, a normalized server frequency (DVFS), and the fraction of delay-tolerant
work deferred to a later step. The environment returns the per-step operational
carbon and a latency SLO-violation signal.

Power model (Eq. 1-2 of the paper):

    C_tot = sum_t sum_i I_i(t) * (P_IT_i(t) + P_cool_i(t)) * dt
    P_cool_i(t) = (PUE_i(t) - 1) * P_IT_i(t)
    P_IT_i(t)   = P_idle_i + (P_peak_i - P_idle_i) * u_i(t)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FleetConfig:
    n_regions: int = 4
    dt_minutes: float = 5.0
    episode_steps: int = 2016              # ~1 week at 5 min
    carbon_min: float = 120.0              # gCO2/kWh
    carbon_max: float = 430.0
    pue_range: tuple = (1.2, 1.6)
    p_idle_w: float = 120.0
    p_peak_w: float = 350.0
    servers_per_region: int = 8000
    slo_latency_ms: float = 200.0          # p95 target
    queue_capacity: float = 1.0
    deferral_max: float = 0.30             # max share of load deferrable
    deferral_horizon: int = 12             # steps a deferred job may wait
    freq_min: float = 0.6                  # normalized DVFS floor
    seed: int = 0
    # per-region phase offsets de-synchronize the carbon profiles
    phase_offsets: tuple = (0.0, 0.35, 0.7, 0.5)


class DataCenterFleetEnv:
    """Multi-region fleet environment.

    Observation (per region): utilization, queue occupancy, normalized carbon
    intensity, capacity headroom, plus a short history of carbon intensity used
    by the temporal-attention forecaster.
    """

    def __init__(self, cfg: FleetConfig | None = None, carbon_traces=None, demand_trace=None):
        self.cfg = cfg or FleetConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.n = self.cfg.n_regions
        self.dt_h = self.cfg.dt_minutes / 60.0

        self.carbon_traces = (
            np.asarray(carbon_traces, dtype=np.float64)
            if carbon_traces is not None
            else self._synth_carbon()
        )
        self.demand_trace = (
            np.asarray(demand_trace, dtype=np.float64)
            if demand_trace is not None
            else self._synth_demand()
        )
        self.history_len = 24
        self.reset()

    # ------------------------------------------------------------------ traces

    def _synth_carbon(self) -> np.ndarray:
        """Fallback synthetic carbon traces (used only when no real trace given).

        Real studies should pass measured carbon-intensity traces via
        ``carbon_traces``; see scripts/prepare_data.py.
        """
        T, n = self.cfg.episode_steps, self.n
        steps_per_day = int(24 * 60 / self.cfg.dt_minutes)
        t = np.arange(T)
        out = np.zeros((n, T))
        for i in range(n):
            phase = 2 * np.pi * self.cfg.phase_offsets[i % len(self.cfg.phase_offsets)]
            daily = np.sin(2 * np.pi * t / steps_per_day + phase)
            weekly = 0.25 * np.sin(2 * np.pi * t / (7 * steps_per_day) + phase)
            noise = 0.08 * self.rng.standard_normal(T).cumsum() / np.sqrt(T)
            sig = 0.5 * (1 + daily) + weekly + noise
            sig = (sig - sig.min()) / (np.ptp(sig) + 1e-9)
            out[i] = self.cfg.carbon_min + sig * (self.cfg.carbon_max - self.cfg.carbon_min)
        return out

    def _synth_demand(self) -> np.ndarray:
        T = self.cfg.episode_steps
        steps_per_day = int(24 * 60 / self.cfg.dt_minutes)
        t = np.arange(T)
        base = 0.55 + 0.20 * np.sin(2 * np.pi * (t / steps_per_day) - np.pi / 2)
        base += 0.05 * self.rng.standard_normal(T)
        return np.clip(base, 0.15, 0.95)

    # ------------------------------------------------------------------- reset

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.queues = np.zeros(self.n)
        self.deferred = np.zeros((self.n, self.cfg.deferral_horizon))
        self.pue = self.rng.uniform(*self.cfg.pue_range, size=self.n)
        self.util = np.full(self.n, 0.4)
        self._carbon_hist = np.tile(
            self.carbon_traces[:, :1], (1, self.history_len)
        )
        return self._obs()

    # --------------------------------------------------------------- observation

    def _norm_carbon(self, c):
        return (c - self.cfg.carbon_min) / (self.cfg.carbon_max - self.cfg.carbon_min)

    def _obs(self) -> dict:
        idx = min(self.t, self.carbon_traces.shape[1] - 1)
        carbon_now = self.carbon_traces[:, idx]
        node = np.stack(
            [
                self.util,
                self.queues / self.cfg.queue_capacity,
                self._norm_carbon(carbon_now),
                1.0 - self.util,
                self.deferred.sum(axis=1) / max(self.cfg.deferral_max, 1e-6),
            ],
            axis=1,
        )
        return {
            "node": node.astype(np.float32),                       # (N, F)
            "carbon_history": self._norm_carbon(self._carbon_hist).astype(np.float32),
            "carbon_now": carbon_now.astype(np.float32),
        }

    # -------------------------------------------------------------------- step

    def step(self, action: dict):
        """Apply one control step.

        action = {
          "routing":  (N, N) row-stochastic matrix, routing[i, j] = share of
                      region i's arrivals served by region j,
          "freq":     (N,) in [freq_min, 1],
          "deferral": (N,) in [0, deferral_max],
        }
        """
        cfg = self.cfg
        idx = min(self.t, self.carbon_traces.shape[1] - 1)
        carbon = self.carbon_traces[:, idx]
        arrivals = self.demand_trace[idx] * np.ones(self.n)

        routing = np.asarray(action["routing"], dtype=np.float64)
        routing = np.clip(routing, 0.0, None)
        routing = routing / np.clip(routing.sum(axis=1, keepdims=True), 1e-9, None)
        freq = np.clip(np.asarray(action["freq"], dtype=np.float64), cfg.freq_min, 1.0)
        defer = np.clip(np.asarray(action["deferral"], dtype=np.float64), 0.0, cfg.deferral_max)

        # route arrivals, hold back the deferred share
        served_in = routing.T @ (arrivals * (1.0 - defer))
        self.deferred[:, self.t % cfg.deferral_horizon] = arrivals * defer

        # deferred work matures after the horizon and is added back
        matured = self.deferred[:, (self.t + 1) % cfg.deferral_horizon].copy()
        self.deferred[:, (self.t + 1) % cfg.deferral_horizon] = 0.0
        offered = served_in + matured + self.queues

        # service capacity scales with DVFS frequency
        capacity = freq * 1.0
        done_work = np.minimum(offered, capacity)
        self.queues = np.clip(offered - done_work, 0.0, cfg.queue_capacity * 3)
        self.util = np.clip(done_work / np.clip(capacity, 1e-9, None), 0.0, 1.0)

        # power and carbon (Eq. 1-2)
        p_it = cfg.p_idle_w + (cfg.p_peak_w - cfg.p_idle_w) * self.util
        p_it *= cfg.servers_per_region * freq**2      # DVFS: dynamic power ~ f^2
        p_cool = (self.pue - 1.0) * p_it
        energy_kwh = (p_it + p_cool) * self.dt_h / 1000.0
        carbon_g = float(np.sum(carbon * energy_kwh))

        # latency model: queueing delay grows with utilization and backlog
        # M/M/1-style queueing delay, saturated to reflect admission control
        rho = np.clip(self.util, 0.0, 0.95)
        latency_ms = 40.0 + 60.0 * rho / (1.0 - rho) + 120.0 * self.queues
        latency_ms = np.clip(latency_ms, 0.0, 2000.0)
        p95 = float(np.percentile(latency_ms, 95))
        violation = float(max(0.0, p95 - cfg.slo_latency_ms) / cfg.slo_latency_ms)

        # roll carbon history for the forecaster
        self._carbon_hist = np.concatenate(
            [self._carbon_hist[:, 1:], carbon[:, None]], axis=1
        )

        self.t += 1
        done = self.t >= cfg.episode_steps
        info = {
            "carbon_g": carbon_g,
            "energy_kwh": float(energy_kwh.sum()),
            "p95_latency_ms": p95,
            "violation": violation,
            "mean_carbon_intensity": float(carbon.mean()),
            "utilization": self.util.copy(),
        }
        return self._obs(), carbon_g, violation, done, info

    # ------------------------------------------------------------------ helpers

    def carbon_saving_potential(self) -> float:
        """Region carbon-saving potential g_i used by the aggregation rule (Eq. 13)."""
        mean_ci = self.carbon_traces.mean(axis=1)
        spread = self.carbon_traces.std(axis=1)
        pot = spread / (mean_ci + 1e-9)
        return float(pot.mean())
