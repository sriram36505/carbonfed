"""Risk-aware actor-critic agent (Sections 4.1, 4.3 of the paper).

Distributional twin critics trained with the quantile-Huber loss and truncated
target quantiles (TQC), a CVaR objective over the return distribution, and a
learned Lagrangian dual variable that enforces the latency SLO.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import Actor, CostCritic, QuantileCritic, StateEncoder


def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, kappa: float = 1.0):
    """Quantile-Huber loss, Eq. (9).

    pred:   (B, M)       quantile estimates theta_m(s, a)
    target: (B, M_t)     target samples y_m'
    """
    B, M = pred.shape
    tau = (torch.arange(M, device=pred.device, dtype=pred.dtype) + 0.5) / M
    u = target.unsqueeze(1) - pred.unsqueeze(2)                   # (B, M, M_t)
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    weight = (tau.view(1, M, 1) - (u.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean(dim=2).sum(dim=1).mean()


def cvar_from_quantiles(q: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR_alpha of a cost distribution given its quantiles (Eq. 8).

    Cost convention: higher is worse, so the risky tail is the UPPER alpha
    fraction. alpha -> 1 recovers the mean.
    """
    M = q.shape[-1]
    k = max(1, int(round(alpha * M)))
    worst, _ = torch.topk(q, k, dim=-1, largest=True)
    return worst.mean(dim=-1)


class CarbonFedAgent:
    def __init__(self, cfg, node_dim: int, n_regions: int, adjacency: torch.Tensor,
                 device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n = n_regions

        self.encoder = StateEncoder(
            node_dim=node_dim,
            history_len=cfg["history_len"],
            horizon=cfg["forecast_horizon"],
            gat_hidden=cfg["gat_hidden"],
            gat_heads=cfg["gat_heads"],
        ).to(self.device)
        self.encoder.set_adjacency(adjacency.to(self.device))

        z_dim = self.encoder.out_dim
        act_dim = n_regions * n_regions + 2 * n_regions

        self.actor = Actor(z_dim, n_regions,
                           freq_min=cfg["freq_min"],
                           deferral_max=cfg["deferral_max"]).to(self.device)
        self.critic1 = QuantileCritic(z_dim, act_dim, n_regions, cfg["n_quantiles"]).to(self.device)
        self.critic2 = QuantileCritic(z_dim, act_dim, n_regions, cfg["n_quantiles"]).to(self.device)
        self.cost_critic = CostCritic(z_dim, act_dim, n_regions).to(self.device)

        self.actor_t = copy.deepcopy(self.actor)
        self.critic1_t = copy.deepcopy(self.critic1)
        self.critic2_t = copy.deepcopy(self.critic2)
        self.encoder_t = copy.deepcopy(self.encoder)

        lr = cfg["learning_rate"]
        self.opt_actor = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.encoder.parameters()), lr=lr)
        self.opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters())
            + list(self.cost_critic.parameters()), lr=lr)

        # learned Lagrangian dual variable (Eq. 11)
        self.log_lambda = torch.tensor(float(np.log(cfg["lambda_init"])),
                                       requires_grad=True, device=self.device)
        self.opt_lambda = torch.optim.Adam([self.log_lambda], lr=cfg["dual_lr"])
        self.updates = 0

    # ------------------------------------------------------------------ helpers

    @property
    def lam(self) -> torch.Tensor:
        return self.log_lambda.exp()

    def _encode(self, obs, enc=None):
        enc = enc or self.encoder
        node = torch.as_tensor(obs["node"], dtype=torch.float32, device=self.device)
        hist = torch.as_tensor(obs["carbon_history"], dtype=torch.float32, device=self.device)
        if node.dim() == 2:
            node, hist = node.unsqueeze(0), hist.unsqueeze(0)
        dual = self.lam.detach().expand(node.shape[0])
        z, _ = enc(node, hist, dual)
        return z

    # -------------------------------------------------------------------- act

    @torch.no_grad()
    def act(self, obs, noise: float = 0.0):
        z = self._encode(obs)
        routing, freq, defer = self.actor(z)
        if noise > 0:
            routing = torch.softmax(
                torch.log(routing + 1e-8) + noise * torch.randn_like(routing), dim=-1)
            freq = (freq + noise * torch.randn_like(freq)).clamp(self.cfg["freq_min"], 1.0)
            defer = (defer + noise * torch.randn_like(defer)).clamp(0.0, self.cfg["deferral_max"])
        return {
            "routing": routing.squeeze(0).cpu().numpy(),
            "freq": freq.squeeze(0).cpu().numpy(),
            "deferral": defer.squeeze(0).cpu().numpy(),
        }

    # ----------------------------------------------------------------- update

    def update(self, batch):
        cfg = self.cfg
        node, hist, act, cost, viol, node2, hist2, done = batch
        B = node.shape[0]
        dual = self.lam.detach().expand(B)

        z, _ = self.encoder(node, hist, dual)
        with torch.no_grad():
            z2, _ = self.encoder_t(node2, hist2, dual)
            a2 = self.actor_t.flat_action(z2)
            eps = (cfg["target_noise"] * torch.randn_like(a2)).clamp(
                -cfg["noise_clip"], cfg["noise_clip"])
            a2 = a2 + eps

            q1 = self.critic1_t(z2, a2)
            q2 = self.critic2_t(z2, a2)
            q_cat = torch.cat([q1, q2], dim=-1)
            q_sorted, _ = torch.sort(q_cat, dim=-1)
            # TQC: drop the largest target quantiles (optimistic tail)
            keep = q_sorted.shape[-1] - 2 * cfg["top_quantiles_to_drop"]
            q_keep = q_sorted[:, :keep]
            target = cost.unsqueeze(-1) + cfg["gamma"] * (1 - done.unsqueeze(-1)) * q_keep

        loss_c = (quantile_huber_loss(self.critic1(z, act), target)
                  + quantile_huber_loss(self.critic2(z, act), target))
        qv = self.cost_critic(z, act)
        with torch.no_grad():
            qv_t = viol + cfg["gamma"] * (1 - done) * self.cost_critic(z2, a2)
        loss_c = loss_c + F.mse_loss(qv, qv_t)

        self.opt_critic.zero_grad(set_to_none=True)
        loss_c.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters())
            + list(self.cost_critic.parameters()), 10.0)
        self.opt_critic.step()

        loss_a = torch.tensor(0.0)
        self.updates += 1
        if self.updates % cfg["policy_delay"] == 0:
            z_pi, _ = self.encoder(node, hist, dual)
            a_pi = self.actor.flat_action(z_pi)
            q_pi = torch.min(self.critic1(z_pi, a_pi), self.critic2(z_pi, a_pi))
            z_alpha = cvar_from_quantiles(q_pi, cfg["cvar_alpha"])      # risk-adjusted cost
            cost_pen = self.cost_critic(z_pi, a_pi)
            # minimise risk-adjusted carbon plus the dual-priced SLO term (Eq. 12)
            loss_a = (z_alpha + self.lam.detach() * cost_pen).mean()

            self.opt_actor.zero_grad(set_to_none=True)
            loss_a.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.encoder.parameters()), 10.0)
            self.opt_actor.step()

            # dual ascent on the constraint violation (Eq. 11)
            with torch.no_grad():
                excess = viol.mean() - cfg["slo_budget"]
            loss_lambda = -(self.log_lambda.exp() * excess)
            self.opt_lambda.zero_grad(set_to_none=True)
            loss_lambda.backward()
            self.opt_lambda.step()
            with torch.no_grad():
                self.log_lambda.clamp_(np.log(1e-6), np.log(100.0))

            self._polyak()

        return {"critic_loss": float(loss_c.detach()), "actor_loss": float(loss_a.detach()),
                "lambda": float(self.lam)}

    def _polyak(self):
        tau = self.cfg["tau"]
        for net, net_t in [(self.actor, self.actor_t), (self.critic1, self.critic1_t),
                           (self.critic2, self.critic2_t), (self.encoder, self.encoder_t)]:
            for p, pt in zip(net.parameters(), net_t.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)

    # ------------------------------------------------------- federated interface

    def get_parameters(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def state_dict(self) -> dict:
        out = {}
        for name, mod in [("encoder", self.encoder), ("actor", self.actor),
                          ("critic1", self.critic1), ("critic2", self.critic2),
                          ("cost", self.cost_critic)]:
            for k, v in mod.state_dict().items():
                out[f"{name}.{k}"] = v
        return out

    def load_parameters(self, params: dict):
        buckets = {"encoder": {}, "actor": {}, "critic1": {}, "critic2": {}, "cost": {}}
        for k, v in params.items():
            mod, sub = k.split(".", 1)
            buckets[mod][sub] = v
        self.encoder.load_state_dict(buckets["encoder"], strict=False)
        self.actor.load_state_dict(buckets["actor"])
        self.critic1.load_state_dict(buckets["critic1"])
        self.critic2.load_state_dict(buckets["critic2"])
        self.cost_critic.load_state_dict(buckets["cost"])
