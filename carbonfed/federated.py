"""Carbon-weighted, differentially private federated aggregation (Section 4.4).

Implements:
  * clip_and_noise            - Gaussian mechanism on parameter updates (Eq. 15)
  * carbon_weighted_aggregate - weighting by carbon-saving potential (Eq. 13)
  * ditto_regularizer         - personalization penalty (Eq. 14)
"""

from __future__ import annotations

import math

import torch


def flat_norm(update: dict) -> float:
    """L2 norm of a parameter update, taken over all tensors jointly."""
    total = 0.0
    for v in update.values():
        total += float(v.pow(2).sum())
    return math.sqrt(total)


def clip_and_noise(update: dict, clip_norm: float, epsilon: float, delta: float,
                   generator: torch.Generator | None = None) -> dict:
    """Clip an update to ``clip_norm`` then add calibrated Gaussian noise, Eq. (15).

        sigma_dp  proportional to  sqrt(2 ln(1.25 / delta)) / epsilon

    The noise is scaled by the clipping bound S, so a smaller epsilon (stronger
    privacy) injects proportionally more noise.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    norm = flat_norm(update)
    scale = min(1.0, clip_norm / (norm + 1e-12))
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

    out = {}
    for k, v in update.items():
        clipped = v * scale
        noise = torch.normal(
            mean=0.0, std=sigma * clip_norm, size=v.shape,
            device=v.device, dtype=v.dtype, generator=generator,
        )
        out[k] = clipped + noise
    return out


def carbon_weights(n_samples, carbon_potentials, kappa: float) -> list[float]:
    """Aggregation weights of Eq. (13).

        w_i = n_i (1 + kappa g_i) / sum_j n_j (1 + kappa g_j)

    kappa = 0 recovers ordinary sample-count FedAvg.
    """
    raw = [n * (1.0 + kappa * g) for n, g in zip(n_samples, carbon_potentials)]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [r / total for r in raw]


def carbon_weighted_aggregate(global_params: dict, updates: list[dict],
                              n_samples, carbon_potentials, kappa: float) -> dict:
    """Form the next global model from privatized client updates, Eq. (13)."""
    w = carbon_weights(n_samples, carbon_potentials, kappa)
    new = {}
    for k, base in global_params.items():
        acc = torch.zeros_like(base)
        for wi, upd in zip(w, updates):
            acc += wi * upd[k]
        new[k] = base + acc
    return new


def ditto_regularizer(local_params, global_params, mu: float) -> torch.Tensor:
    """Ditto proximal term of Eq. (14): (mu / 2) * || phi_i - omega_glob ||^2."""
    total = None
    for k, p in local_params.items():
        if not torch.is_floating_point(p):
            continue
        diff = (p - global_params[k]).pow(2).sum()
        total = diff if total is None else total + diff
    return 0.5 * mu * total


def parameter_delta(before: dict, after: dict) -> dict:
    """Elementwise update delta_omega_i = phi_after - phi_before."""
    return {k: (after[k] - before[k]) for k in before}
