"""Neural modules for CarbonFed.

Contains:
  * GATLayer          - graph attention over the inter-region topology (Eq. 5)
  * TemporalAttention - short-horizon carbon-intensity forecaster (Eq. 6)
  * StateEncoder      - fuses graph, forecast and local features (Eq. 7)
  * Actor             - routing / frequency / deferral policy (Eq. 4)
  * QuantileCritic    - distributional return critic (Eq. 9)
  * CostCritic        - scalar SLO-violation critic used by the dual (Eq. 12)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single-head graph attention (Velickovic et al., ICLR 2018), Eq. (5).

        h'_i = sigma( sum_{j in N(i)} alpha_ij W h_j )
        alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )
    """

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.normal_(self.a_src, std=0.1)
        nn.init.normal_(self.a_dst, std=0.1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, N, F_in), adj: (N, N) with 1 where an edge exists
        Wh = self.W(h)                                   # (B, N, F_out)
        e_src = (Wh * self.a_src).sum(-1).unsqueeze(-1)  # (B, N, 1)
        e_dst = (Wh * self.a_dst).sum(-1).unsqueeze(-2)  # (B, 1, N)
        e = F.leaky_relu(e_src + e_dst, self.negative_slope)
        e = e.masked_fill(adj.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        return F.elu(torch.bmm(alpha, Wh))


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = nn.ModuleList(
            [GATLayer(in_dim, out_dim // heads) for _ in range(heads)]
        )

    def forward(self, h, adj):
        return torch.cat([head(h, adj) for head in self.heads], dim=-1)


class TemporalAttention(nn.Module):
    """Scaled dot-product self-attention forecaster, Eq. (6).

    Consumes a window of past carbon-intensity observations per region and
    predicts the next ``horizon`` steps.
    """

    def __init__(self, history_len: int, horizon: int, d_model: int = 64, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, history_len, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        # hist: (B, N, L) -> forecast (B, N, horizon)
        B, N, L = hist.shape
        x = self.inp(hist.reshape(B * N, L, 1)) + self.pos
        z = self.encoder(x)
        return self.head(z[:, -1]).reshape(B, N, -1)


class StateEncoder(nn.Module):
    """Fuses graph embedding, carbon forecast and local signals into z_i (Eq. 7)."""

    def __init__(self, node_dim: int, history_len: int, horizon: int,
                 gat_hidden: int = 128, gat_heads: int = 4, d_model: int = 64):
        super().__init__()
        self.gat = MultiHeadGAT(node_dim, gat_hidden, gat_heads)
        self.forecaster = TemporalAttention(history_len, horizon, d_model)
        self.out_dim = gat_hidden + horizon + node_dim + 1   # + dual variable

    def forward(self, node, carbon_hist, dual):
        h = self.gat(node, self.adj)
        fc = self.forecaster(carbon_hist)
        B, N, _ = node.shape
        lam = dual.view(B, 1, 1).expand(B, N, 1)
        return torch.cat([h, fc, node, lam], dim=-1), fc

    def set_adjacency(self, adj: torch.Tensor):
        self.register_buffer("adj", adj, persistent=False)


class Actor(nn.Module):
    """Outputs routing logits, normalized frequency and deferral fraction (Eq. 4)."""

    def __init__(self, z_dim: int, n_regions: int, hidden: int = 256,
                 freq_min: float = 0.6, deferral_max: float = 0.30):
        super().__init__()
        self.n = n_regions
        self.freq_min = freq_min
        self.deferral_max = deferral_max
        self.body = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.route_head = nn.Linear(hidden, n_regions)
        self.freq_head = nn.Linear(hidden, 1)
        self.defer_head = nn.Linear(hidden, 1)

    def forward(self, z):
        # z: (B, N, z_dim)
        f = self.body(z)
        routing = torch.softmax(self.route_head(f), dim=-1)          # (B, N, N)
        freq = self.freq_min + (1 - self.freq_min) * torch.sigmoid(self.freq_head(f))
        defer = self.deferral_max * torch.sigmoid(self.defer_head(f))
        return routing, freq.squeeze(-1), defer.squeeze(-1)

    def flat_action(self, z):
        r, f, d = self(z)
        return torch.cat([r.flatten(1), f, d], dim=-1)


class QuantileCritic(nn.Module):
    """Distributional critic returning M quantiles of the carbon-cost return (Eq. 9)."""

    def __init__(self, z_dim: int, action_dim: int, n_regions: int,
                 n_quantiles: int = 25, hidden: int = 256):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(z_dim * n_regions + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_quantiles),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z.flatten(1), a], dim=-1))


class CostCritic(nn.Module):
    """Scalar critic for expected discounted SLO violations (used in Eq. 12)."""

    def __init__(self, z_dim: int, action_dim: int, n_regions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * n_regions + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, a):
        return self.net(torch.cat([z.flatten(1), a], dim=-1)).squeeze(-1)
