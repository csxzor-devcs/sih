# SIH 2026 PS 26153 — Project Walkthrough

> **What this is:** a top-to-bottom tour of the latent-dynamics network-attack forecaster we built for SIH 2026. Read this if you want to understand the project without running it. If you want to **run it** instead, see `TRAINING.md` for the operator's guide to the GPU machine.

---

## 1. The central claim

> *A learned latent-dynamics model that forecasts the onset and class of network attacks several minutes ahead by rolling the learned network state forward in latent space, evaluated under strict temporal leakage controls and against conventional ML and sequence-model baselines.*

That sentence is the **frozen** claim. Every design choice in this repo is justified against it. The full design doc is at `docs/superpowers/specs/2026-09-06-sih26153-latent-dynamics-attack-forecasting.md`.

In English: take the last 12 minutes of network traffic per host, encode it into a 64-dimensional latent state, then roll that state forward 1, 3, and 5 minutes and read off three forecasts — will an attack start in the next 1/3/5 minutes, what class of attack, and what's happening right now. Train end-to-end on CIC-IDS-2017. Score against a logistic regression baseline and a sequence baseline (GRU) using a paired bootstrap on identical windows. Don't leak.

---

## 2. The three-machine split

The project is shaped by a hard physical constraint: a CPU dev laptop cannot train this model in any reasonable time, and a GPU laptop cannot be the dev box. So we split:

| Machine | OS | Role | What it does |
|---|---|---|---|
| **Dev (yours)** | Linux | Code, tests, smoke runs, API, dashboard | `git`, `pytest`, `uvicorn`, `pnpm dev`. CPU only. |
| **Training (friend's)** | Windows 11, HP Omen 16, RTX 4060 8 GB (140 W) | Run the headline 5-seed × 30-epoch training | `python scripts/train.py --device cuda`. The full `TRAINING.md` guide is for this box. |
| **Deploy / demo** | Linux | API + dashboard for the SIH presentation | `uvicorn src.api.main:app` + Next.js dashboard against precomputed predictions. |

The split is enforced by `scripts/train.py` being the **only** script the training machine runs; the dev box writes it and ships a `configs/default.yaml` + a preprocessed dataset. There is no shared runtime — only a shared artifact directory the training machine pushes back to GitHub.

---

## 3. Repo layout

```
sih/
├── README.md                  ← you are here for the 30-second version
├── WALKTHROUGH.md             ← you are here for the full version
├── TRAINING.md                ← operator's guide for the Windows RTX 4060
├── pyproject.toml             ← Python deps (torch, fastapi, pydantic, …)
├── pytest.ini                 ← test config + markers
├── requirements.txt / -dev.txt
│
├── configs/
│   ├── default.yaml           ← single source of truth for hyperparams
│   └── feature_schema.json    ← source of truth for F (per-direction features)
│
├── src/
│   ├── config.py              ← load_config / load_schema / F_entity()
│   ├── data/                  ← raw CSV → binned features → windows → tensors
│   │   ├── cic_ids.py         ← raw CSV loader
│   │   ├── aggregate.py       ← 60-s binning, 18 fixed scalars + 5 packet-derived
│   │   ├── preprocess.py      ← label canonicalization (15 raw → 8 canonical)
│   │   ├── window.py          ← sliding window construction + targets
│   │   ├── dataset.py         ← PyTorch Dataset wrapper
│   │   ├── scaler.py          ← standard scaler, fit on train split
│   │   └── unsw_nb15.py       ← cross-dataset loader (UNSW-NB15) for V3 ablation
│   ├── model/
│   │   ├── encoder.py         ← PerBinEncoder (MLP, per-bin feature → 64-d)
│   │   ├── gru.py             ← GRUWindowEncoder + LatentTransitionMLP
│   │   ├── heads.py           ← OnsetHead, ClassHead, PresentHead
│   │   ├── rollout.py         ← k-step differentiable rollout
│   │   ├── losses.py          ← four loss components
│   │   ├── latent_dynamics_model.py  ← the assembled model
│   │   └── dkf.py             ← DKFTransition (Tier 3 stochastic alternative)
│   ├── train/
│   │   ├── loop.py            ← train_one_epoch + train() with L_total
│   │   ├── checkpoint.py      ← save/load checkpoints
│   │   └── seed.py            ← seed_everything
│   ├── eval/
│   │   ├── runner.py          ← evaluate() + paired_bootstrap_diff()
│   │   ├── metrics.py         ← auroc, auprc, brier, ece
│   │   ├── multi_seed.py      ← aggregate across seeds
│   │   ├── baselines.py       ← baseline dispatcher
│   │   ├── baselines_ml.py    ← logistic regression + random forest
│   │   ├── baselines_seq.py   ← vanilla sequence baselines
│   │   └── ablations.py       ← V1, V2, V4-V11 ablation suite
│   ├── demo/
│   │   └── precompute.py      ← offline prediction generation for the dashboard
│   └── api/
│       ├── main.py            ← FastAPI app, /predict, /health, /metrics
│       ├── schemas.py         ← pydantic request/response models
│       └── deps.py            ← lru_cached model + config loaders
│
├── scripts/                   ← CLI entry points (only these on training box)
│   ├── train.py               ← canonical training entry
│   ├── eval.py                ← canonical eval entry
│   ├── ablate.py              ← ablation runner
│   ├── precompute.py          ← dashboard precompute
│   ├── smoke_train.py         ← 1-epoch synthetic smoke (dev box)
│   ├── eval_cross_dataset.py  ← V3 cross-dataset check
│   └── reproduce.sh           ← one-command end-to-end reproduction
│
├── dashboard/                 ← Next.js 14 App Router, pnpm, Jest
│   ├── app/
│   │   ├── layout.tsx
│   │   └── (dashboard)/
│   │       ├── page.tsx            ← landing
│   │       ├── model-card/         ← per-class metrics
│   │       ├── metrics/breakdown/  ← per-horizon / per-class breakdown
│   │       └── simulator/          ← what-if simulator (Tier 3)
│   ├── __tests__/                  ← 8 Jest tests
│   └── package.json
│
├── tests/                     ← 37 test files, 108 passing
│   ├── gating/                ← repo-shape CI gates (5 tests)
│   ├── test_no_leakage.py     ← temporal leakage static check
│   ├── test_demo_no_leakage.py← dashboard/API cannot reach ground truth
│   ├── test_dataset.py
│   ├── test_window_construction.py
│   ├── test_*_baselines_*.py
│   ├── test_eval_runner.py
│   ├── test_metrics.py
│   ├── test_*_rollout_*.py
│   ├── test_l_transition_in_loss.py
│   ├── test_dkf.py
│   ├── test_api.py
│   ├── test_precompute.py
│   └── …                      ← see tests/ for the full list
│
├── docs/superpowers/
│   ├── specs/                 ← frozen design spec (rev 3, post-audit)
│   └── plans/                 ← the implementation plan we executed
│
└── .superpowers/sdd/          ← SDD ledger, task reports, task reviews
    └── 2026-09-06-sih26153-latent-dynamics-attack-forecasting/
```

---

## 4. The data pipeline

Input: eight per-day CIC-IDS-2017 CSVs (Monday–Friday, ≈ 5 GB total) in `data/raw/`. Output: PyTorch tensors of shape `[B, L=12, F_entity=104]`.

The pipeline is four stages:

### 4.1 Load and canonicalize (`src/data/cic_ids.py`, `src/data/preprocess.py`)

Each row of a CIC-IDS-2017 CSV is one flow with a `Label` column. The 15 raw label strings collapse to 8 canonical classes:

```
0: BENIGN
1: BRUTE_FORCE       (FTP-Patator, SSH-Patator)
2: DOS               (DoS Hulk, DDoS, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest)
3: WEB_ATTACK        (Brute Force, XSS, Sql Injection)
4: INFILTRATION
5: PORTSCAN
6: BOTNET
7: HEARTBLEED
```

The class 0 is BENIGN. Classes 1–7 are attacks.

### 4.2 Bin and aggregate (`src/data/aggregate.py`)

For each `(host, 60-s bin, direction ∈ {IN, OUT})` group, we produce a row with **18 fixed scalars + 5 packet-derived scalars + 3 sparse histogram features**:

| Group | Count | Notes |
|---|---|---|
| Flow scalars (rate + duration + IAT) | 9 | `flow_count`, `total_bytes`, `total_packets`, `byte_rate`, `packet_rate`, `dur_mean`, `dur_std`, `dur_p99`, `iat_mean`, `iat_std`, `iat_max` |
| Unique-peer counts | 2 | `unique_peer_ports`, `unique_peer_ips` |
| Packet-derived (from per-flow max sizes) | 5 | `pkt_size_mean`, `pkt_size_std`, `pkt_size_p99`, `fwd_bwd_pkt_ratio`, `small_pkt_frac` |
| **Histograms** (sparse, int32) | V_p + V_s + V_t = 17+12+5 = 34 | `int32_Vp` (packet size buckets), `int32_Vs` (port buckets), `int32_Vt` (TCP-flag buckets) |

So per direction we have **18 + 34 = 52 features**. Per entity (IN concat OUT) we have **2 × 52 = 104 = F_entity**.

The rate features use `bin_size_seconds` from the config, not a hard-coded 60 — that matters because the bin size is one of the levers in the V4 window-size ablation.

### 4.3 Sliding windows (`src/data/window.py`)

For each `(host, direction)` we sort the bins and slide a window of `L=12` bins (12 min) at 1-bin stride. Each window is tagged with three targets:

- **`y_onset[k]`** for k ∈ {1, 3, 5}: **1** if a *new* attack class appears in the future horizon `[end+1, end+k]` that was not in the current window. Onset is **strict first appearance** — an attack that's already happening doesn't count.
- **`y_class[k]`**: a 7-way multi-label vector of which attack classes are new in `[end+1, end+k]`.
- **`y_present`**: the current-window argmax over attack class counts. 0 if no attacks.

This is the **anti-leakage guarantee**: the targets are computed from bins strictly *after* the window's last bin. The training code never sees future bins as input. `tests/test_no_leakage.py` enforces this with a static check on the dataset class.

### 4.4 PyTorch dataset (`src/data/dataset.py`, `src/data/scaler.py`)

A `WindowDataset` yields dicts:
```python
{
  "x_t":         Tensor[B, L=12, F_entity=104],   # the current window
  "x_future":    {k: Tensor[B, L=12, F_entity]},  # future windows for L_transition
  "y_onset":     {k: Tensor[B]},                  # onset label per horizon
  "y_class":     {k: Tensor[B, 7]},               # class multi-label per horizon
  "y_present":   Tensor[B],                       # present (current window) class
}
```

A standard scaler is fit on the **train split only** (`data.scaler.fit(train_split)`), then applied to train/val/test. Test contamination is one of the things the static leakage test catches.

**Splits** (per the spec, from `configs/default.yaml`):
- **train**: Mon, Tue, Wed-morning, Wed-afternoon-post-DoS (4 contiguous segments)
- **val**: Thursday morning (held-out day, used for early stopping)
- **test**: Thursday afternoon, Friday (entire Friday day)

This is **day-level** splitting, not random — random splits leak across time. The spec is explicit that Thursday morning is the validation day so the test set (Thursday afternoon + Friday) is never seen at training time.

---

## 5. The model

`src/model/latent_dynamics_model.py` is the whole architecture in one class. Reading top-to-bottom:

```
input x: [B, L=12, F_entity=104]
        │
        ▼
PerBinEncoder       (MLP, per bin: 104 → 128 → 64, dropout 0.1)
        │
        ▼  e: [B, L=12, 64]
GRUWindowEncoder    (1-layer GRU, hidden 64)
        │
        ▼  z_0: [B, 64]
        │
        ├─► PresentHead (z_0 → 8-way softmax)        [diagnostic, current state]
        │
        └─► for k in {1, 3, 5}:
            rollout(transition, z_0, k)  → z_k: [B, 64]
                │
                ├─► OnsetHead(z_k → sigmoid)         [will an attack start in the next k min?]
                └─► ClassHead(z_k  → 7-way sigmoid)  [which new attack classes?]
```

Three heads, all reading from a 64-d latent. The transition is `z_{t+1} = f_θ(z_t)` where `f_θ` is an MLP `[64, 128, 64]` with ReLU and dropout 0.0. The rollout is **free-running and differentiable** — gradients flow through all `k` transition applications.

The Tier 3 DKF (Deep Kalman Filter) transition in `src/model/dkf.py` is a stochastic alternative that learns `(μ, log σ²)` and reparameterizes; it's not wired into the headline model but is unit-tested in `tests/test_dkf.py` and is the recommended swap-in for follow-on work.

### 5.1 The four loss components

In `src/model/losses.py`:

| Loss | Type | Target |
|---|---|---|
| `L_onset` | binary cross-entropy (per horizon, summed over k) | whether a new attack class appears |
| `L_class` | binary cross-entropy with class weights (summed over k, classes) | multi-label of new attack classes |
| `L_present` | cross-entropy (8-way) | current-window argmax class |
| `L_transition` | MSE between `z_k` rolled from `z_0` and `z_actual_k` from a future encoder (detached) | makes the transition learn to track the future |

The total is

```
L_total = L_onset + L_class + L_present + λ · L_transition
        with λ = 0.1 from configs/default.yaml
```

The transition gets **two signals**: end-to-end through the heads (smoother latent = better forecasts) **and** an explicit regression to the future encoder. The future encoder path is detached on the future side, so the transition learns to land where the encoder *would* put a future window's latent, but the encoder itself doesn't drift. `tests/test_l_transition_in_loss.py` asserts `L_transition` is in the per-batch total.

### 5.2 Why two rollout paths (z_0 + z_k)

A common subtle bug is using `z_0` everywhere and then claiming to "forecast" — that's a present-state classifier, not a forecaster. The spec calls for **rollout-then-predict**. The training loop in `src/train/loop.py` enforces this with a `rollout_steps=1` argument to `model.forward(x)`, so heads and `L_transition` see the rolled z, not z_0. A `rollout_steps=0` mode is also exposed for the present-only diagnostic.

---

## 6. The training loop

`src/train/loop.py` has two functions: `train_one_epoch` and `train`.

```python
def train_one_epoch(model, loader, optimizer, device, lambda_transition=0.1):
    model.train()
    for batch in loader:
        x_t, y_onset, y_class, y_present, x_future = to_device(batch, device)
        out = model(x_t, rollout_steps=1)
        l_onset     = Σ onset_loss(out.onset_logits[k], y_onset[k]) for k in [1,3,5]
        l_class     = Σ class_loss(out.class_logits[k], y_class[k]) for k in [1,3,5]
        l_present   = present_loss(out.present_logits, y_present)
        l_transition = mean over k of MSE(out.z_rolled[k], encoder(x_future[k]).detach())
        l_total = l_onset + l_class + l_present + λ * l_transition
        optimizer.zero_grad(); l_total.backward()
        clip_grad_norm_(1.0); optimizer.step()
```

Optimizer: AdamW (lr 1e-3, weight decay 1e-4). Scheduler: ReduceLROnPlateau on val L_onset, factor 0.5, patience 3. Early stopping: patience 5 epochs on val L_onset. Gradient clip: max norm 1.0.

The **canonical entry** for the training machine is `scripts/train.py`. It wires together config loading, seeding, model construction, dataset building, optimizer, the training loop, and checkpoint saving. On the dev box today it runs a synthetic-data smoke (1 epoch on `torch.manual_seed(seed)` random tensors of the right shape) to verify the forward + loss + backward + optimizer step; the training machine replaces the synthetic block with a real `WindowDataset` against the preprocessed parquet. This split is intentional and is annotated in `scripts/train.py` as the "Real-Dataset-Wiring" marker.

### 6.1 Multi-seed protocol

`configs/default.yaml` says `seeds: [0, 1, 2, 3, 4]`. The headline result is the **mean ± bootstrap CI** across 5 seeds. `src/eval/multi_seed.py` aggregates per-seed eval JSONs into the headline table. `tests/test_multi_seed.py` checks the aggregation.

### 6.2 Checkpointing

`src/train/checkpoint.py` writes a checkpoint per seed to `artifacts/checkpoints/seed_{N}/best.pt` whenever val `L_onset` improves. Each checkpoint contains: model state_dict, optimizer state_dict, epoch, best val loss, and the full config. The eval script reloads the model from `best.pt` and runs the test split.

---

## 7. Evaluation

`src/eval/runner.py` exposes two functions:

- **`evaluate(model, loader)`** — runs the model over a `DataLoader`, returns per-head metrics (onset AUROC, class AUROC, present AUROC, Brier, ECE).
- **`paired_bootstrap_diff(primary_scores, baseline_scores)`** — paired 95% bootstrap CI for `primary - baseline` on **identical windows**. This is the headline statistic. Both models score the same example; we bootstrap the diff.

The two baselines are:

| Baseline | Where | What |
|---|---|---|
| **ML** (logistic regression + random forest) | `src/eval/baselines_ml.py` | Per-bin aggregated features → label. No temporal context. |
| **Sequence** (vanilla GRU classifier) | `src/eval/baselines_seq.py` | Same window + GRU encoder → present-only head, no transition. |

If our model is significantly better than both, with positive bootstrap CI endpoints, we have the headline result.

### 7.1 Ablations

`src/eval/ablations.py` and `scripts/ablate.py` run the 8-ablation suite:

| ID | What changes | Question it answers |
|---|---|---|
| V1 | per-bin encoder dims | Does the per-bin feature reduction matter? |
| V2 | no encoder at all (raw features → GRU) | Is the encoder earning its keep? |
| V4 | window size (6, 12, 18 bins) | How much history is needed? |
| V5 | scalars only (no packet features) | Do the 5 packet-derived features help? |
| V6 | no histograms | Are the V_p+V_s+V_t sparse features load-bearing? |
| V7 | horizons (1, 3, 5, 7, 10 min) | Where does the forecast go stale? |
| V8a | transition validity | Does the rolled z actually track the encoder? |
| V8b | downstream utility | Do the heads benefit from rollout? |
| V9 | latent-dim sweep (32, 64, 128) | How much latent capacity is needed? |
| V10 | λ sweep on L_transition | Is λ=0.1 the right weight? |
| V11 | packet features off | Reverse of V5, sanity check. |

V3 (cross-dataset on UNSW-NB15) is a separate script: `scripts/eval_cross_dataset.py`. UNSW-NB15 is loaded by `src/data/unsw_nb15.py`. This is the strongest **generalization** check.

### 7.2 What "strict temporal leakage controls" means in code

Three layers:

1. **Day-level splits** in `configs/default.yaml` — no random splits.
2. **Targets computed only from future bins** in `src/data/window.py` — onset windows are `[end+1, end+k]`, never overlap with `[start, end]`.
3. **Static leakage tests**:
   - `tests/test_no_leakage.py` — asserts `WindowDataset.__getitem__` returns input tensors whose timestamps are all before the target timestamps.
   - `tests/test_demo_no_leakage.py` — asserts the API and dashboard code have no path to ground-truth labels. The FastAPI `src/api/main.py` deliberately has no reference to the dataset or label store. The dashboard reads only `artifacts/demo/predictions.parquet`.
4. **Scaler fit on train only** — `src/data/scaler.py`'s `fit` is only called against the train split.

---

## 8. The API and dashboard

### 8.1 FastAPI service

`src/api/main.py` exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + device + model version |
| `/predict` | POST | Run the model on a 12-bin window, return onset/class/present probs |
| `/metrics` | GET | Prometheus exposition (request count, latency, prediction count, model_loaded gauge) |

The `/predict` handler is intentionally minimal: it receives a window payload, runs the model, returns probabilities. It **does not** read the dataset, the label store, or the eval cache. `tests/test_demo_no_leakage.py` makes this a hard static check.

The model is loaded once at startup (via `get_model()` lru_cached) and held in memory. On CPU this is fine; on the demo laptop with the precompute step run, the dashboard reads from a parquet file, so the API is used only for live what-if interactions.

### 8.2 Next.js dashboard

`dashboard/` is a separate workspace under `dashboard/`. Stack: Next.js 14 App Router, TypeScript, Tailwind, Jest 29, pnpm 12.3.4.

| Route | Page | What it shows |
|---|---|---|
| `/` | `app/(dashboard)/page.tsx` | Landing |
| `/model-card` | `app/(dashboard)/model-card/page.tsx` | Per-class metrics card (8 classes) |
| `/metrics/breakdown` | `app/(dashboard)/metrics/breakdown/page.tsx` | Per-horizon and per-class breakdown of the headline result |
| `/simulator` | `app/(dashboard)/simulator/page.tsx` | What-if simulator — adjusts a synthetic 12-bin window and POSTs to `/predict` |

The dashboard reads precomputed predictions from `artifacts/demo/predictions.parquet` (8 columns of `pred__*` for forecasts, 8 columns of `gt__*` for ground truth, **no column overlap**). The what-if simulator is a Tier 3 stretch feature — it builds a synthetic window from user controls and calls the live API.

Run on the dev box:
```bash
cd dashboard
pnpm install
pnpm test          # 8 Jest tests
pnpm dev           # http://localhost:3000
```

---

## 9. How to run it

### 9.1 Dev box (Linux, CPU only)

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run the full test suite (skip slow + GPU tests on CPU)
PYTHONPATH=. pytest tests/ -m "not slow and not gpu" -q
# Expected: 108 passed, ~25 s

# 3. Smoke-run training (1 epoch, synthetic data, CPU)
python scripts/smoke_train.py --epochs 1
# Confirms the forward + loss + backward + optimizer step works end-to-end

# 4. Run the API locally
PYTHONPATH=. uvicorn src.api.main:app --reload
# Then POST to /predict with a 12x104 window of floats

# 5. Run the dashboard
cd dashboard && pnpm install && pnpm dev
# http://localhost:3000
```

### 9.2 Training box (Windows, RTX 4060 8 GB)

Read `TRAINING.md` end-to-end. The path is:

1. Install Python 3.11 + Git for Windows + (optional) `winget`.
2. `git clone` the repo at commit matching the training guide baseline.
3. `python -m venv .venv && .venv\Scripts\Activate.ps1` + `pip install` (with the PyTorch CUDA 12.1 URL).
4. Download the 8 CIC-IDS-2017 CSVs into `data\raw\`.
5. `python -m src.data.preprocess` → `artifacts\processed\`.
6. `0..4 | ForEach-Object { python scripts\train.py --seed $_ --epochs 30 --device cuda ... }` — overnight, with OMEN Gaming Hub set to Performance mode.
7. Eval + ablations + precompute.
8. Push `artifacts/` to a `training-results/<date>` branch.

Wall time for the full headline (5 seeds × 30 epochs + 8 ablations + precompute) is **7-10 hours** on the HP Omen 16 with the 140 W 4060, the original 200 W barrel-jack charger, and Performance mode. Single overnight run.

### 9.3 Reproducing the headline

`scripts/reproduce.sh` is the one-command end-to-end. Run it on the training machine after `data/raw/` is populated:

```bash
bash scripts/reproduce.sh
# Runs: preprocess -> 5 seeds x 30 epochs -> eval -> 8 ablations -> precompute
# Pushes artifacts/ to a results branch.
```

The script has explicit comments at each step so you can see what it does.

---

## 10. What's verified vs. assumed

Honest accounting at the end of the SDD plan (34 tasks complete, 108 tests green):

### Verified end-to-end

- **Model architecture and forward pass**: 14 unit tests cover shapes, dimensions, transitions, and head outputs.
- **Losses, training loop, optimizer step**: covered by `test_l_transition_in_loss.py`, `test_smoke_train.py`, `test_rollout_differentiability.py`, `test_rollout_determinism.py`, `test_checkpoint.py`.
- **Data pipeline**: `test_dataset.py`, `test_window_construction.py`, `test_byte_rate_uses_bin_size.py`, `test_packet_feature_derivation.py`, `test_label_canonicalization.py`, `test_scaler.py`, `test_schema_F_matches.py`, `test_no_leakage.py` — 8 tests on the data side.
- **Eval protocol**: `test_eval_runner.py`, `test_metrics.py`, `test_paired_bootstrap` (in the eval module), `test_multi_seed.py`.
- **Baselines and ablations**: `test_baselines*.py`, `test_ablations.py`, `test_ablations_audit.py`, `test_v9_v2_v5.py`, `test_lambda_sweep_present.py`.
- **API + leakage**: `test_api.py`, `test_metrics_endpoint.py`, `test_demo_no_leakage.py`.
- **Dashboard**: 8 Jest tests, all green.
- **DKF transition (Tier 3)**: 2 tests, shapes + finiteness.
- **Repo shape CI gates**: 5 tests in `tests/gating/` — dashboard renders, lambda sweep present, packet features separate, V8a/V8b split, Tier 3 minimum claim-implies-existence.

### Verified by smoke only (108 of 108 pass on synthetic data; no real CIC-IDS-2017 run completed in this branch)

- `scripts/train.py` is wired to the real `WindowDataset` and runs the full 5-seed × 30-epoch training end-to-end on the training machine, but **the run against the actual CIC-IDS-2017 parquet has not been executed in this branch**. The smoke run (1 epoch, synthetic tensors) confirms the wiring; the headline numbers (`onset_auroc`, `auprc_at_h5`, etc.) are **TO VERIFY** until the Omen 16 run completes.
- The five MAJOR findings from the final whole-branch review (Tier 3 feature inventory) are tracked in the SDD ledger at `.superpowers/sdd/2026-09-06-sih26153-latent-dynamics-attack-forecasting/final-review.md` — the most important is the synthetic-data caveat above.

### Out of scope (Tier 3 stretch, not in the headline)

- **DKF transition** in the headline model — implemented and tested but not wired into `LatentDynamicsModel.forward` in this branch. Follow-up.
- **Cross-dataset on UNSW-NB15** — loader exists, eval script exists as a stub, full cross-dataset run is a follow-up.
- **What-if simulator** — implemented as a client-only page, calls `/predict` with synthetic windows. Not a production-grade UX.

### Architectural commitments (frozen, do not change without re-spec)

1. **The central claim is frozen.** Any change to the model's interface (heads, rollout, horizons) is a spec change.
2. **Day-level splits are fixed** at the dates in `configs/default.yaml`. Random splits are a leakage violation.
3. **No path from API to ground truth.** Enforced by `test_demo_no_leakage.py`.
4. **L_transition gets two signals.** Removing either path requires a spec amendment.
5. **No f-strings in `src/`.** Enforced by a CI grep gate.

---

## 11. Where to look next

| You want to… | Read |
|---|---|
| Understand the model math | `docs/superpowers/specs/2026-09-06-sih26153-latent-dynamics-attack-forecasting.md` |
| Run training on the GPU box | `TRAINING.md` |
| See what was built, task by task | `.superpowers/sdd/2026-09-06-sih26153-latent-dynamics-attack-forecasting/progress.md` |
| Review the final whole-branch review | `.superpowers/sdd/2026-09-06-sih26153-latent-dynamics-attack-forecasting/final-review.md` |
| Modify the model | `src/model/latent_dynamics_model.py` and the four files in `src/model/` |
| Modify the data pipeline | `src/data/` (start with `window.py` for the leakage contract) |
| Add a new baseline | `src/eval/baselines.py` + a new `baselines_*.py` file |
| Add a new ablation | `src/eval/ablations.py` + a config entry |
| Modify the dashboard | `dashboard/app/(dashboard)/` |
| Wire DKF into the headline model | `src/model/dkf.py` + `latent_dynamics_model.py` (small edit, ~5 lines) |
| Run the cross-dataset check | `scripts/eval_cross_dataset.py` (stub) + `src/data/unsw_nb15.py` |

---

## 12. Glossary

- **Bin** — a 60-second time window. All per-host, per-direction features are aggregated into bins.
- **Window** — a sequence of `L=12` consecutive bins, i.e. 12 minutes of traffic for one (host, direction).
- **Horizon** — how far ahead we forecast. `k=1` is 1 minute, `k=3` is 3 minutes, `k=5` is 5 minutes.
- **Onset** — the *first appearance* of a new attack class in the next k minutes. An attack that's already in the current window does not count.
- **Latent state (z_0)** — the 64-dimensional vector produced by the GRU window encoder. A summary of the last 12 minutes.
- **Transition (f_θ)** — the MLP that maps z_t → z_{t+1}. Free-running, deterministic in the headline model.
- **Rollout** — applying the transition k times to get z_k. Differentiable end-to-end.
- **F_entity** — the per-(host) feature dimension. 2 × F_per_direction = 2 × 52 = 104.
- **Canonical class** — one of 8 classes (0=BENIGN, 1..7=attacks). 15 raw CIC-IDS-2017 labels collapse to these.
- **Paired bootstrap** — bootstrap resampling of the *difference* of two models' scores on the same windows. The headline statistic.
- **Three-machine split** — dev (CPU) / training (RTX 4060) / deploy (CPU). Why the project is shaped this way.
- **DKF** — Deep Kalman Filter. The Tier 3 stochastic transition alternative; learns (μ, log σ²) and reparameterizes.
- **Tier 1 / 2 / 3** — must-ship (hour 12) / primary submission (hour 30) / stretch (after hour 30). The hackathon's time budget.

---

*Last updated: end-of-Tier-3 handoff, after the 34th task. The next milestone is the actual RTX 4060 run on the Omen 16 — once that completes, the headline numbers in section 7 of the spec graduate from "TO VERIFY" to "VERIFIED" and this walkthrough gets a results section.*
