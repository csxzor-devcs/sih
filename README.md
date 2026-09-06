# SIH 26153 — Latent-Dynamics Network Attack Forecasting

A learned GRU-based latent-dynamics model that forecasts the onset and class of network attacks several minutes ahead on CIC-IDS-2017. The system rolls a learned network latent state forward in time and predicts from the rolled state. Strict temporal leakage controls. Three-machine split: dev (CPU) / training (RTX 4060 8GB) / deploy (CPU).

See `docs/superpowers/specs/2026-09-06-sih26153-latent-dynamics-attack-forecasting.md` for the full design.

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -m "not slow and not gpu"
```

## Hardware

- Dev laptop (your machine, CPU): code, tests, smoke runs, API, dashboard
- Training laptop (friend's, RTX 4060 8GB): full 5-seed training only
- Deploy / demo (CPU): API + dashboard for the SIH presentation

## Environment variables

- `SIH_DATA_DIR` (default: `./data`) — where raw and processed data live
- `SIH_ARTIFACTS_DIR` (default: `./artifacts`) — where checkpoints, scaler, vocab, demo data live
