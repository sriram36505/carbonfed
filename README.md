# CarbonFed

Risk-aware federated graph reinforcement learning for carbon-aware workload
orchestration in geographically distributed data centers.

Each operator runs a local agent that encodes the fleet with a graph attention
network over the inter-region topology and a temporal self-attention module that
forecasts short-horizon carbon intensity. Learning uses distributional critics
under a conditional-value-at-risk (CVaR) objective and a learned Lagrangian dual
variable that enforces latency SLOs. Operators cooperate through a
carbon-weighted, differentially private federated averaging scheme with
Ditto-style personalization; no raw data is exchanged.

---

---

## Installation

```bash
git clone https://github.com/sriram36505/carbonfed.git
cd carbonfed
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q            # 9 smoke tests
```

Tested with Python 3.10–3.12 and PyTorch 2.x on CPU. A GPU is not required.

---

## Data

The simulator accepts real carbon-intensity and cloud-workload traces. Place one
CSV per region plus a workload CSV under `data/raw/`, then:

```bash
python scripts/prepare_data.py \
    --carbon data/raw/region_*.csv \
    --workload data/raw/workload.csv \
    --out data/traces.npz
```



---

## Usage

```bash
# federated training (Algorithm 1)
python scripts/train.py --config configs/default.yaml --seed 0

# ablations
python scripts/train.py --mode isolated      --seed 0   # no federation
python scripts/train.py --mode centralized   --seed 0   # pooled-data oracle

# evaluation against all baselines, 5 seeds
python scripts/evaluate.py --checkpoint runs/federated_seed0/checkpoint.pt \
    --seeds 0 1 2 3 4

# figures from logged runs
python scripts/plot_results.py --logs runs --eval results/evaluation.json
```

To reproduce the full study, run all five seeds for each mode:

```bash
for m in federated isolated centralized; do
  for s in 0 1 2 3 4; do
    python scripts/train.py --mode $m --seed $s
  done
done
```

---

## Layout

```
carbonfed/
  env.py         fleet simulator: power, PUE, queues, DVFS, deferral   (Eq. 1–2)
  models.py      GAT encoder, temporal-attention forecaster, actor,
                 quantile critics, cost critic                          (Eq. 5–7)
  agent.py       CVaR objective, quantile-Huber loss, TQC truncation,
                 TD3 targets, learned Lagrangian dual                   (Eq. 8–12)
  federated.py   DP clipping and noise, carbon-weighted aggregation,
                 Ditto personalization                                  (Eq. 13–15)
  baselines.py   round-robin, greedy-latency, carbon-greedy, carbon-aware opt
  utils.py       replay buffer, seeding, adjacency, JSONL logging
configs/default.yaml   all hyperparameters (Table 2 of the paper)
scripts/               prepare_data, train, evaluate, plot_results
tests/                 smoke tests
```

---

## Key hyperparameters

All values live in `configs/default.yaml` and mirror Table 2.

| Parameter | Value |
|---|---|
| Regions *N* | 4 |
| Control interval Δt | 5 min |
| Episode length | 2016 steps (~1 week) |
| Carbon-intensity range | 120–430 gCO₂/kWh |
| PUE | 1.2–1.6 |
| Idle / peak server power | 120 / 350 W |
| GAT heads / hidden dim | 4 / 128 |
| Forecast horizon *h* | 12 steps (1 hour) |
| Quantiles *M* | 25 |
| CVaR level α | 0.5 |
| Discount γ | 0.99 |
| Aggregation interval *K* | 10 rounds |
| Carbon weight κ / Ditto μ | 1.5 / 0.1 |
| Privacy budget ε (δ = 1e-5) | 8 |

Setting `carbon_kappa: 0` recovers ordinary sample-count FedAvg;
`cvar_alpha: 1.0` recovers mean optimization. Both are useful ablations.

---

## Reproducibility

`scripts/train.py` seeds Python, NumPy and PyTorch through
`carbonfed.utils.set_seed`. Training metrics are written to
`runs/<mode>_seed<k>/train.jsonl`, one JSON object per episode, and checkpoints
to `checkpoint.pt`. The differential-privacy noise draws from a seeded
`torch.Generator`, so federated runs are reproducible given the same seed.

Note that exact bitwise reproducibility across PyTorch versions and hardware is
not guaranteed; report mean and standard deviation over seeds rather than single
runs.

---

## Citation

```bibtex
@article{carbonfed2026,
  title  = {CarbonFed: risk-aware federated graph reinforcement learning for
            carbon-aware workload orchestration in geographically distributed
            data centers},
  author = {Viji, D. and Saranya, P. and Gao, Xiao-Zhi and Sriram, R.},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
