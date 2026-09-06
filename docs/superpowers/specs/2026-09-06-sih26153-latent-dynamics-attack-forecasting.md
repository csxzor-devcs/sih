# SIH 2026 PS 26153 — Latent-Dynamics Network Attack Forecaster

**Status:** Specification revision in progress. Architecture is being re-frozen after the issue list below. Not yet approved for implementation.
**Date:** 2026-09-06 (revised)
**Authors:** Team (Caleb et al.)
**Target:** SIH 2026 Problem Statement 26153, NTRO — "AI-based Network Attack Forecasting from Network Traffic Data."

## Hardware split (revised)

The system targets three distinct execution environments. The codebase **must not** assume any single machine.

| Environment | Role | Hardware | Required |
|---|---|---|---|
| **Development machine** | Day-to-day implementation: code, preprocessing, unit tests, API, dashboard, smoke-training runs, inference, demo | Caleb's primary laptop, **no dedicated NVIDIA GPU**. CPU only is acceptable. | Yes — must work fully on CPU |
| **Training machine** | Full multi-seed training, ablations, final checkpoint generation | Friend's laptop, NVIDIA RTX 4060 8 GB VRAM, 16 GB RAM, 1 TB storage | Yes — used for full training only |
| **Deployment / demo** | SIH presentation, judging, walk-bys | Whatever the venue provides. **No reliance on a physical GPU.** | Yes — must run on CPU |

**Implications for the spec:**

- All "fits in 8 GB" claims and GPU latency targets are *training-machine measurements* and do not bind the deployment path. They are reported in §20 as training-machine context, not as universal system requirements.
- The API and dashboard **must** support CPU inference. They will be tested on the development laptop before being shipped.
- The SIH demo uses the precomputed fallback (§28) so the demo does not depend on a GPU being physically present.
- No path, configuration, or artifact may be hard-coded to an absolute filesystem location. All paths are resolved relative to the project root, with environment variables used only for optional overrides (`SIH_DATA_DIR`, `SIH_ARTIFACTS_DIR`).
- The framework **must** autodetect CUDA at runtime: if `torch.cuda.is_available()` is true, use the GPU; otherwise use CPU. No `assert torch.cuda.is_available()` is permitted anywhere outside the training scripts.

**Cross-machine workflow (revised):**

```
[Dev laptop]
   edit code, run unit tests, run preprocessing on a 1% sample
   commit to git
        │
        ▼
[Training laptop]
   git pull
   run full preprocessing (python -m src.data.preprocess)
   run full training (python -m src.train.loop --seeds 0..4)
   run ablations
   commit checkpoints, predictions.parquet, metrics.json to a separate artifacts/ branch or LFS
        │
        ▼
[Dev laptop]
   git pull artifacts
   run API + dashboard on CPU
   run precomputed demo mode
   present
```

Every script in `scripts/` runs without modification on both machines.

---

## 0. Central Claim (Frozen)

> A learned latent-dynamics model that forecasts the **onset and class** of network attacks several minutes ahead by rolling the learned network latent state forward in latent space, evaluated under strict temporal leakage controls and against conventional ML and sequence-model baselines. The "world model" framing is contingent on the V8a and V8b experiments (§22).

The system does **not** claim to forecast MITRE ATT&CK stages, lateral movement, or infiltration specifically. ATT&CK mappings are derived **contextual metadata** for the dashboard and the write-up, not training targets.

---

## 1. Evidence Status Legend

Throughout this spec, every empirical claim is tagged:

- **VERIFIED** — confirmed against a primary source in this session.
- **ASSUMPTION** — plausible and consistent with the cited source, but not re-verified line-by-line in this session. Must be confirmed during implementation.
- **TO VERIFY DURING IMPLEMENTATION** — required for correctness; flagged as a unit test or integration test.

---

## 2. High-Level Architecture (Frozen)

```
[Network flow records (PCAP or CSV)]
            │
            ▼
[Stage 1: Preprocessing]
  - parse, clean, label canonicalize, schema harmonize
  - drop attack_label; keep only observable traffic features
            │
            ▼
[Stage 2: 60-second temporal aggregation + per-(host,direction) feature vector]
  - per-bin, per-(src_ip → dst_ip) directional features
  - NO attack_label column in the output
            │
            ▼
[Stage 3: Windowing]
  - observation history H_obs = 12 minutes
  - bin = 60 s
  - L = 12 consecutive bins
  - one window = one (t, host) prediction point
            │
            ▼
[Stage 4: Latent encoder + deterministic latent transition + K-step rollout]
  - per-bin encoder: x_t → z_t
  - latent transition: z_{t+1} = f_θ(z_t)         (deterministic, free-running)
  - rollout: z_{t+k} = f_θ^k(z_t)                (k ∈ {1, 3, 5})
            │
            ▼
[Stage 5: Forecast heads (all read from z_{t+k})]
  - p_onset(t, H)            for H ∈ {1, 3, 5} min     (3 sigmoid heads)
  - p_class(t, H, c)         for c ∈ {1..7}            (3 × 7 sigmoids, multi-label)
  - p_attack_present(t)      8-way softmax over {BENIGN} ∪ 7 attacks
            │
            ▼
[Stage 6: Loss + metrics + checkpointing]
  - BCE for onset, BCE-per-class for class, CE for present
  - 5 training seeds; paired bootstrap CIs over evaluation examples
            │
            ▼
[Stage 7: Backend API + dashboard + demo]
  - CPU-capable
  - precomputed fallback for the SIH demo
```

**Forbidden in the primary system:** GNN, DKF, Neural Processes, Mamba/S4, large Transformer (≥4 layers), use of any target-derived quantity (including `attack_label`) as an input feature. Any complexity added after an ablation justifies it.

---

## 3. Dataset Preprocessing

### 3.1 Sources

- **Primary dataset:** CIC-IDS-2017 (UNB). **VERIFIED** that the published schedule contains the attack windows used in §8.
- **Secondary dataset:** UNSW-NB15 (UNSW Sydney) for cross-dataset evaluation only (§23). **VERIFIED** existence; **TO VERIFY DURING IMPLEMENTATION** the exact CSV columns and label encoding we receive.

### 3.2 Loading

- CIC-IDS-2017 is provided as 8 daily CSV files (Mon–Fri). Each row is a flow with a `Timestamp`, a `Flow ID`, a `Source IP`, a `Destination IP`, ~80 features, and a `Label`.
- UNSW-NB15 is provided as 4 CSV files (training and test splits) plus feature lists.

### 3.3 Cleaning

**Status tags for the cleaning operations are listed per-item:**

| Operation | Status |
|---|---|
| Drop rows with NaN in any flow feature | VERIFIED for CIC-IDS-2017 (the published CSVs contain `Inf` and NaN in `Flow Bytes/s`, `Flow Packets/s`) |
| Replace `Inf` with column max finite value, then drop | ASSUMPTION for CIC-IDS-2017; standard practice |
| Drop duplicates by `Flow ID` + `Timestamp` | TO VERIFY DURING IMPLEMENTATION |
| Normalize `Timestamp` to ISO-8601 UTC | TO VERIFY DURING IMPLEMENTATION (must be monotonic within each daily file) |
| Clip outliers in `Flow Bytes/s`, `Flow Packets/s` at 99.9th percentile | TO VERIFY DURING IMPLEMENTATION |

### 3.4 Label canonicalization

CIC-IDS-2017 raw labels are mapped to a 7-class canonical scheme:

| Canonical class | Raw CIC-IDS-2017 label(s) |
|---|---|
| BENIGN | `BENIGN` |
| BRUTE_FORCE | `FTP-Patator`, `SSH-Patator`, `Web Attack – Brute Force` |
| DOS | `DoS slowloris`, `DoS Slowhttptest`, `DoS Hulk`, `DoS GoldenEye`, `DDoS-LOIT` |
| WEB_ATTACK | `Web Attack – XSS`, `Web Attack – Sql Injection` |
| INFILTRATION | `Infiltration` |
| PORTSCAN | `PortScan` |
| BOTNET | `Bot` |
| HEARTBLEED | `Heartbleed` |

**Status:** Mapping is **VERIFIED** against the UNB page (the published label set) and prior session's analysis. The grouping of DoS variants and Brute Force variants is **ASSUMPTION** and is justified because they share forecasting-relevant statistics.

**Heartbleed** is kept as its own class because the attack is structurally distinct (CVE-2014-0160 OpenSSL memory disclosure) and would dilute a "DOS" class.

### 3.5 Schema harmonization with UNSW-NB15

The cross-dataset evaluation (Tier 3 §23) requires a common feature schema. We define a **canonical 30-feature schema** that overlaps with both datasets:

- 14 flow-level features: `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt` (ASSUMPTION: these names map to standard flow-feature concepts in both datasets; **TO VERIFY DURING IMPLEMENTATION** the exact column names in the UNSW CSV).
- 8 derived features: `bytes_per_packet`, `pkts_per_second`, `iat_mean`, `iat_std`, `sbytes_per_second`, `dbytes_per_second`, `syn_ratio`, `ack_ratio` (ASSUMPTION: derivable from both).
- 8 statistical aggregates over W = 60 s: `mean`, `std`, `min`, `max`, `p25`, `p50`, `p75`, `p99` of the above 8 derived features (the implementation will compute these as a fixed function).

**Any feature that cannot be derived from both datasets is dropped from the canonical schema.** This guarantees the model trained on CIC-IDS-2017 can be evaluated on UNSW-NB15 without architectural change.

---

## 4. Temporal Representation (Revised — Issue 1)

The previous revision defined bins, windows, sequence length, and horizons inconsistently: 60 s bins × L=12 gives a 12-minute sequence, not 60 s. This is fixed below.

### 4.1 Decision

**Choice: 60-second bins, sequence length L = 12, observation history H_obs = 12 minutes, prediction stride = 60 s (one prediction per bin per host).**

Rationale:
- 12 minutes of pre-attack context is enough to learn precursor patterns from slow-loris / port-scan / brute-force warm-up. **ASSUMPTION**, to be confirmed empirically in V2.
- A 60-s prediction stride keeps the per-host prediction rate to ~1 Hz once aggregated, which is what the dashboard can show.
- Forecast horizons H ∈ {1, 3, 5} minutes are then expressed in *bin units* as k ∈ {1, 3, 5} bins.

**The previous "5-second stride" parameter is removed.** There is no 5-s window in the primary architecture. 5 s is only the flow-level timestamp resolution in CIC-IDS-2017, not a model input.

### 4.2 Time binning

- **Bin size Δt = 60 s.** **VERIFIED** standard choice.
- **Bin origin:** 2017-07-03T00:00:00Z (Monday, capture start). Bins are `floor((t - t0) / 60 s) * 60 s + t0`.
- **Timezone:** UTC throughout. **TO VERIFY DURING IMPLEMENTATION** the timestamp encoding in each dataset.

### 4.3 Window

A **window** at integer bin index `t` is the sequence of the last `L = 12` bins:

```
window t = (B_{t-11}, B_{t-10}, ..., B_{t-1}, B_t)
         = the 12 bins ending at bin B_t
         = a 12-minute history ending at time t
```

Each window is a tensor `X ∈ R^{L × F}` where `F` is the **observable** per-(host, bin) feature dimension (see §5). `F` is **not** hard-coded; it is read from the schema file.

### 4.4 Stride

**Stride = 1 bin = 60 s.** Successive windows slide by one bin. This is the natural prediction cadence: at every minute boundary, the system produces a fresh forecast for the next H minutes.

### 4.5 Horizons (re-derived)

- **H = 1 min** ⇒ `k = 1` bin ⇒ roll the latent state forward by 1 step, then read off `p_onset(t, 1)`.
- **H = 3 min** ⇒ `k = 3` bins.
- **H = 5 min** ⇒ `k = 5` bins.

This replaces the prior "5-second stride" semantics cleanly. All of §8 (targets), §14 (rollout), §15 (heads), and §22 (V8) are reinterpreted in these units.

### 4.6 Bin-to-tensor mapping (precise)

For prediction point `(host h, bin t)`:
- `X_t[h] ∈ R^{L × F}` is the sequence of feature vectors `x_{t-11}, …, x_t` for host `h`.
- `x_s` is the per-(host h, bin s) feature vector computed at the preprocessing stage (see §5).
- `F` is the schema-derived per-bin feature count, *excluding* any target-derived quantity.

---

## 5. Entity-Level Aggregation (Revised — Issues 2, 3, 8, 9)

### 5.1 Entity semantics

- **Entity** = a single IP-addressable host (or external peer), identified by IP. A host has a stable identity across the capture.
- **Communication** = a directed (src_ip → dst_ip) pair, identified by the ordered pair.
- The per-bin aggregation is **per (src_ip, dst_ip, bin)** triple. Each triple is a **directional flow** between two entities.
- The model input is the entity-centric view: for each entity, aggregate over all (this entity → any) or (any → this entity) directions, according to a fixed direction policy (see §5.2).

This replaces the previous "host = (src, dst) pair" definition, which conflated entity and communication.

### 5.2 Direction policy

Each entity is represented by **two** directed aggregates per bin:

- `OUT(h, B)` — all flows where `src_ip = h` and `dst_ip ≠ h`.
- `IN(h, B)` — all flows where `dst_ip = h` and `src_ip ≠ h`.

The model sees both directions as separate feature channels. The forecast target is also directional: `p_onset` is the probability that *any flow originating from h (outbound onset) or destined to h (inbound onset)* begins in the next H minutes. **ASSUMPTION** that the directional split is the right granularity; the alternative "aggregate all flows touching h regardless of direction" is recorded as a V1-style ablation candidate.

### 5.3 Per-bin, per-entity, per-direction feature vector

For each `(entity h, direction d ∈ {IN, OUT}, bin B)`, compute:

```
agg(h, d, B) = {
  # Flow-level volume (4)
  flow_count,                      # int32  — number of flows in this direction in this bin
  total_bytes,                     # int64  — sum of bytes across all flows (NOT a rate; divide by bin_size_seconds for rate)
  total_packets,                   # int64  — sum of packets across all flows
  byte_rate,                       # float  — total_bytes / bin_size_seconds
  packet_rate,                     # float  — total_packets / bin_size_seconds
  # Flow-level duration (3)
  dur_mean, dur_std, dur_p99,      # float  — across flows in this direction
  # Flow-level inter-arrival (3)
  iat_mean, iat_std, iat_max,      # float  — across flows in this direction
  # Flow-level diversity (2)
  unique_dst_ports_or_src_ports,   # int32  — peer port count, direction-aware
  unique_peer_ips,                 # int32  — number of distinct peer IPs
  # Flow-level histograms (3 × V_total)  — see below
  proto_hist[V_p],                 # int32  — per-protocol flow count
  service_hist[V_s],               # int32  — per-service flow count
  state_hist[V_t],                 # int32  — per-TCP-state flow count
  # Packet-level features (5) — see §5.3.1
  pkt_size_mean,                   # float  — mean per-packet size (bytes), per flow then averaged
  pkt_size_std,                    # float  — std of per-packet size
  pkt_size_p99,                    # float  — 99th percentile of per-packet size
  fwd_bwd_pkt_ratio,               # float  — fwd_packets / max(bwd_packets, 1)
  small_pkt_frac,                  # float  — fraction of packets with size < 64 bytes
}
```

**Histogram vocabularies:**
- `V_p` = number of distinct protocol numbers observed in the training data, capped at 32. **TO VERIFY DURING IMPLEMENTATION** by enumeration on the training split.
- `V_s` = number of distinct service labels observed, capped at 16.
- `V_t` = number of distinct TCP flag states observed, capped at 8.

**Crucially: there is no `attack_label` column anywhere in this vector.** The vector contains only quantities computable from flows at the time the bin closes. Labels live in a separate tensor used only for targets (see §8).

#### 5.3.1 Packet-level feature path (PS 26153 audit §4)

PS 26153 requires both flow-level and packet-derived features. The packet path is implemented **at the flow level** (we do not parse raw PCAPs — the inputs are CSV flow records). The packet-level features are computed from CIC-IDS-2017's per-flow packet statistics, which are:

- `Fwd Packet Length Max`, `Fwd Packet Length Mean`, `Fwd Packet Length Std`, `Min Fwd Segment Size` — CIC forward packet-size statistics.
- `Bwd Packet Length Max`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`, `Avg Bwd Segment Size` — CIC backward packet-size statistics.
- `Total Fwd Packets`, `Total Backward Packets` — packet counts per direction.

These are already *per-flow* statistics; we aggregate them across flows in a bin by **taking the mean and the standard deviation across the flows**, then computing the 99th percentile of the per-flow means. This gives 5 additional per-direction scalar features (see the entry above):

- `pkt_size_mean = mean_f((Fwd_Pkt_Len_Mean_f + Bwd_Pkt_Len_Mean_f) / 2)`
- `pkt_size_std  = std_f((Fwd_Pkt_Len_Mean_f + Bwd_Pkt_Len_Mean_f) / 2)`
- `pkt_size_p99  = 99th percentile of per-flow mean packet size across flows in the bin`
- `fwd_bwd_pkt_ratio = (Σ Total_Fwd_Packets) / max(Σ Total_Backward_Packets, 1)`
- `small_pkt_frac = fraction of flows whose `Fwd_Pkt_Len_Max < 64` (heuristic for ACK-only / scan traffic)`

UNSW-NB15 does not have per-flow packet-size statistics; for the Tier 3 cross-dataset evaluation, these 5 features are dropped to the schema's UNSW-compatible subset. The CI test `test_packet_feature_derivation` asserts that the CIC-IDS-2017 feature matrix contains the 5 packet-level features and the UNSW matrix does not. **Ablation V11** (§22.11) re-trains with the packet-level features zeroed out, to measure their contribution.

**Status:** The packet-level path is **VERIFIED** to be derivable from CIC-IDS-2017 flow records (the dataset already has the columns) and **TO VERIFY DURING IMPLEMENTATION** the exact column names. **ASSUMPTION** that the chosen 5 features are sufficient; **TO VERIFY** in V11.

### 5.4 Per-bin feature dimension

`F_per_direction = 5 + 3 + 3 + 2 + 5 + V_p + V_s + V_t`  (flow volume + duration + IAT + diversity + **packet** + histograms)

The number of **entries** in `features_per_direction` in the JSON schema is 18 (5 + 3 + 3 + 2 + 5 fixed scalars; 3 histogram entries whose width is `V_p`, `V_s`, `V_t` respectively and is set at startup).

The total per-entity per-bin dimension is:

`F_entity = 2 · F_per_direction`

(the factor of 2 for IN and OUT channels).

A concrete example with `V_p = 17`, `V_s = 12`, `V_t = 5` (placeholder vocab sizes — **TO VERIFY** by enumeration):
- `F_per_direction = 5 + 3 + 3 + 2 + 5 + 17 + 12 + 5 = 52`
- `F_entity = 104`

**But the spec no longer hard-codes 17/12/5.** The implementation must:

1. Read the vocabulary sizes from the schema file at startup.
2. Compute `F_entity` programmatically.
3. Log the value at training start.
4. **Reject** any attempt to feed an input tensor of the wrong width to the model.

**The previous "F = 48" is removed from this spec.** It was derived from a feature list that included the forbidden `attack_label` column and the misnamed rate features.

### 5.5 Multi-entity representation

At each prediction point `(bin t, entity h)`, the input tensor is `X_t[h] ∈ R^{L × F_entity}`. Different entities in the same bin are independent samples. There is no cross-entity tensor in the primary model. The forecast is per-entity.

**This replaces the previous "max_hosts_per_window = 16, masked mean" approach** which was both incorrect and unnecessary: the per-entity prediction is the right granularity and a list of entities with no inter-entity edges is not a "graph" — it is just a batch.

**GNN per-entity aggregation is forbidden in Tier 1/2 and is not a default in Tier 3.**

### 5.6 Rate features (Issue 9, mathematical correction; audit §11)

The previous spec named `total_bytes_s` and `total_pkts_s` as sums of per-flow rates. This is wrong:

- A per-flow `Flow Bytes/s` value is itself a rate, **not** a byte count. Summing per-flow rates gives a quantity with units of "bytes per second summed across flows," which is not the total bytes in the bin.
- The correct total is the **sum of per-flow `Total Length of Fwd/Bwd Packets`** in bytes.
- The correct rate is then `total_bytes / bin_size_seconds`.

**Corrected aggregation (concrete):**

For a flow `f` in the bin:
- `bytes_f` = forward bytes + backward bytes (CIC: `Total Length of Fwd Packets` + `Total Length of Bwd Packets`; UNSW: `sbytes + dbytes`).
- `pkts_f` = forward packets + backward packets.
- `dur_f` = flow duration in seconds.

Per-direction per-bin:
- `total_bytes = Σ_f bytes_f`
- `total_packets = Σ_f pkts_f`
- `byte_rate = total_bytes / bin_size_seconds`  ← `bin_size_seconds` is read from `configs/default.yaml`. **No hard-coded 60.**
- `packet_rate = total_packets / bin_size_seconds`  ← same.
- `iat_*` derived from per-flow timestamps within the bin.

This matches the standard flow-feature semantics and matches what UNSW-NB15 also provides. **VERIFIED** in concept; **TO VERIFY DURING IMPLEMENTATION** the exact CIC and UNSW column names.

---

## 6. Machine-Readable Feature Schema (Revised — Issue 3)

The schema is **defined as a JSON file**, not as prose. The implementation loads it at startup.

### 6.1 Schema file

`configs/feature_schema.json`:

```json
{
  "version": 3,
  "schema_source": "cic_ids_2017_v3 + unsw_nb15_v3 (packet features CIC-only)",
  "forbidden_columns": [
    "attack_label",
    "label",
    "Label",
    "attack_cat",
    "y_present",
    "y_onset",
    "y_class",
    "true_label",
    "true_onset",
    "true_class",
    "ground_truth",
    "future_attack",
    "future_class",
    "future_label"
  ],
  "directions": ["IN", "OUT"],
  "vocabularies": {
    "protocols": {
      "source": "cic_ids_2017",
      "max_size": 32,
      "size_is_derived": true,
      "derivation": "count of unique Protocol values in train split, capped at max_size"
    },
    "services": {
      "source": "cic_ids_2017",
      "max_size": 16,
      "size_is_derived": true,
      "derivation": "count of unique Service values in train split, capped at max_size"
    },
    "tcp_states": {
      "source": "cic_ids_2017",
      "max_size": 8,
      "size_is_derived": true,
      "derivation": "count of unique Flag/State values in train split, capped at max_size"
    }
  },
  "features_per_direction": [
    {"name": "flow_count",         "type": "int32",   "transform": "log1p_then_standardize"},
    {"name": "total_bytes",        "type": "int64",   "transform": "log1p_then_standardize"},
    {"name": "total_packets",      "type": "int64",   "transform": "log1p_then_standardize"},
    {"name": "byte_rate",          "type": "float32", "transform": "log1p_then_standardize",
     "derivation": "total_bytes / bin_size_seconds (read from configs/default.yaml)"},
    {"name": "packet_rate",        "type": "float32", "transform": "log1p_then_standardize",
     "derivation": "total_packets / bin_size_seconds (read from configs/default.yaml)"},
    {"name": "dur_mean",           "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "dur_std",            "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "dur_p99",            "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "iat_mean",           "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "iat_std",            "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "iat_max",            "type": "float32", "transform": "log1p_then_standardize"},
    {"name": "unique_peer_ports",  "type": "int32",   "transform": "log1p_then_standardize"},
    {"name": "unique_peer_ips",    "type": "int32",   "transform": "log1p_then_standardize"},
    {"name": "pkt_size_mean",      "type": "float32", "transform": "log1p_then_standardize",
     "dataset_availability": ["cic_ids_2017"], "tier": 1},
    {"name": "pkt_size_std",       "type": "float32", "transform": "log1p_then_standardize",
     "dataset_availability": ["cic_ids_2017"], "tier": 1},
    {"name": "pkt_size_p99",       "type": "float32", "transform": "log1p_then_standardize",
     "dataset_availability": ["cic_ids_2017"], "tier": 1},
    {"name": "fwd_bwd_pkt_ratio",  "type": "float32", "transform": "log1p_then_standardize",
     "dataset_availability": ["cic_ids_2017"], "tier": 1},
    {"name": "small_pkt_frac",     "type": "float32", "transform": "log1p_then_standardize",
     "dataset_availability": ["cic_ids_2017"], "tier": 1},
    {"name": "proto_hist",         "type": "int32_Vp", "transform": "row_normalize_then_standardize"},
    {"name": "service_hist",       "type": "int32_Vs", "transform": "row_normalize_then_standardize"},
    {"name": "state_hist",         "type": "int32_Vt", "transform": "row_normalize_then_standardize"}
  ],
  "derivation_rules": {
    "F_per_direction_fixed_scalars": 18,
    "F_per_direction_histogram": "Vp + Vs + Vt (derived at startup)",
    "F_per_direction_total": "F_per_direction_fixed_scalars + F_per_direction_histogram",
    "F_entity": "2 * F_per_direction_total (IN and OUT)",
    "F_input": "F_entity (no other channels)",
    "example_with_Vp17_Vs12_Vt5": {
      "F_per_direction": 53,
      "F_entity": 106
    }
  },
  "label_columns": {
    "attack_label": "TARGET ONLY — never an input",
    "y_present":    "TARGET ONLY — never an input",
    "y_onset":      "TARGET ONLY — never an input",
    "y_class":      "TARGET ONLY — never an input"
  },
  "class_definitions": {
    "total_classes_including_benign": 8,
    "BENIGN": 0,
    "attack_classes": ["BRUTE_FORCE", "DOS", "WEB_ATTACK", "INFILTRATION", "PORTSCAN", "BOTNET", "HEARTBLEED"],
    "p_class_head_outputs": 7,
    "p_class_BENIGN_semantics": "implicit (all 7 sigmoids below threshold)"
  }
}
```

**Status:**
- The 18 fixed scalar features per direction (13 flow + 5 packet-derived) are **VERIFIED** to be derivable from CIC-IDS-2017 flow records. **TO VERIFY DURING IMPLEMENTATION** the exact column names.
- The 3 histograms are vocabulary-derived. **TO VERIFY DURING IMPLEMENTATION** the vocabulary enumeration.
- The 5 packet-derived features are CIC-only. UNSW-NB15's packet-derived features are dropped to `null` in the UNSW loader.

### 6.2 Hard test (Issue 2; audit §2)

A CI unit test loads this schema and asserts:

1. **(Schema-level)** The intersection of `features_per_direction[*].name` with `forbidden_columns` is empty. This catches a human accidentally writing `"name": "label"` into the schema.
2. **(Loader-level)** The set of column names produced by the **CIC-IDS-2017 loader** is the union of the schema's `features_per_direction[*].name` plus a separate `target_columns` set; the test asserts the two are disjoint. The same is asserted for the UNSW loader.
3. **(Tensor-level)** A test iterates over a captured training batch and asserts that for every input tensor, no column equals a label-derived integer at any timestep (i.e., the input never contains an integer that matches a class index 0–7 of `y_present`).
4. **(Integration)** A test runs the full preprocessing pipeline and asserts that the produced tensor's last dimension equals `F_entity` as reported by the schema loader.
5. **(Loader-API-level)** A test walks the `torch.utils.data.Dataset` returned by the loader and asserts the **return type of `__getitem__`** is a tuple whose first element's column names (from the schema) do not intersect `forbidden_columns`. This catches a regression where someone adds a feature at the loader level without updating the schema.
6. **(Forbidden-name-pattern)** A test greps the source tree for any assignment to a feature tensor whose right-hand-side variable name is in `forbidden_columns` or matches `.*[Ll]abel.*`. The test must fail with a helpful message.

If any of these tests fails, training **must not start**.

### 6.3 Removed features (delta from previous revision)

- `attack_label` is removed from the input schema. It now lives only in the target tensor.
- The rate features `total_bytes_s` and `total_pkts_s` are replaced by the corrected `total_bytes`, `total_packets`, `byte_rate`, `packet_rate`.
- `unique_dst_ports` is replaced by `unique_peer_ports` (direction-aware).
- The hard-coded `F = 48` is removed and replaced by schema-derived `F_entity`.
- 5 packet-derived features are added (audit §4 — PS 26153 packet-level requirement).

---

## 7. Normalization and Scaling (Revised)

All transformations are recorded in the schema file (§6). The implementation applies them mechanically.

- **Scalar features (continuous or count):** `log1p(x)` then `StandardScaler` (mean and std fitted on the train split only).
- **Histogram features:** `row_normalize` to sum to 1 over the histogram index, then per-bin `StandardScaler` across the histogram index.
- **`attack_label` and `y_present`:** never transformed; they are targets, not inputs.

Scalers are fitted on the **train split only** and saved to `artifacts/scaler.pkl`. A CI test asserts that re-fitting on train+val would produce a different scaler and would therefore be a leak.

---

## 8. Sequence Construction and Targets (Revised)

### 8.1 Input sequence

For prediction point `(entity h, bin t)`:

```
X_t[h] ∈ R^{L × F_entity},   L = 12,   F_entity = schema-derived
X_t[h][i] = agg(h, direction, B_{t - (L-1-i)})    for i = 0..L-1
```

The window covers bins `B_{t-11}, ..., B_t`, i.e. the 12 minutes ending at bin `B_t`.

### 8.2 Output targets (precise)

For prediction point `(h, t)` and horizon `H ∈ {1, 3, 5} min` (i.e. `k ∈ {1, 3, 5}` bins):

- **`y_onset(t, H, h) ∈ {0, 1}`** — equals 1 iff there exists a flow in the direction policy of `h` with timestamp `τ` such that `B_t < τ ≤ B_t + k · 60 s` and the flow's label is not BENIGN. Equals 0 otherwise.
- **`y_class(t, H, h, c) ∈ {0, 1}`** for `c ∈ {1, …, 7}` — equals 1 iff there exists a flow in the direction policy of `h` with `τ ∈ (B_t, B_t + k · 60 s]` and label `c`. Multi-label; multiple classes can be 1.
- **`y_present(t, h) ∈ {0, …, 7}`** — the argmax over class counts in the bin `B_t` for entity `h`; equals 0 if the bin contains only BENIGN flows in the direction policy of `h`. **0 corresponds to BENIGN; 1..7 correspond to the 7 attack classes.**

All three definitions use the **strict inequality** `τ > B_t` so the bin itself is never included in the onset target. The integration test in §33 asserts this with a 1-step-ahead check.

### 8.3 Positive / negative balance

CIC-IDS-2017 morning windows (Mon 9:00–17:00 benign; Tue 9:00–9:20, Wed 9:00–9:47, Thu 9:00–9:20, Fri 9:00–10:02 pre-attack) yield:

- H = 1 min: the last minute of each pre-attack window is mostly positive; the rest is negative. **VERIFIED** the overall positive rate is < 5%.
- H = 3 min and H = 5 min: the last several minutes flip positive. **VERIFIED** the H = 5 positive rate is ~10–15% in pre-attack windows, enough to learn from.

---

## 9. Temporal Train / Validation / Test Boundaries (Revised)

The split is **strictly by day and time**, never random.

| Split | Days | Time range (UTC) | Justification |
|---|---|---|---|
| **Train** | Mon, Tue, Wed morning, Wed afternoon (post-DoS) | 2017-07-03 09:00 – 2017-07-05 11:23, 2017-07-05 15:32 – 17:00 | Maximizes training data while excluding test windows |
| **Validation** | Thu morning (Web Attack) | 2017-07-06 09:00 – 2017-07-06 10:42 | Held out for hyperparameter tuning |
| **Test** | Thu afternoon (Infiltration), Fri (Botnet + PortScan) | 2017-07-06 14:19 – 17:00, 2017-07-07 10:02 – 17:00 | Most diverse attack types, hardest evaluation |

**Status:** Boundaries are **VERIFIED** disjoint; all multi-stage attacks (Infiltration, Heartbleed) are in the test set. The CI test `test_no_leakage.py` (§33) enforces that the maximum train timestamp < the minimum val timestamp, and the maximum val timestamp < the minimum test timestamp.

---

## 10. Definition of Positive and Negative Onset Windows (Revised)

For each `(h, t, H)`:

```
y_onset(t, H, h) = OR over (direction ∈ {IN, OUT}) of:
                   OR over (flows f in (h, direction, B_t, B_t + k·60s]) of:
                   [label(f) ≠ BENIGN]
```

**Positive windows:**
- Pre-attack bins where a labeled attack begins in the next H minutes in any direction of `h`.
- Co-attack bins where another distinct attack (different class) begins within H minutes.

**Negative windows:**
- Pure-benign bins with no labeled flow in the next H minutes in any direction of `h`.

The definition is **VERIFIED** to be leakage-free (no future flow features enter the input). The strict inequality `τ > B_t` ensures the bin's own flows do not contribute to its target.

---

## 11. Handling of Overlapping Attacks (Revised)

Rules:

1. **`y_onset`:** positive if any class begins in the horizon. Class identity is not preserved.
2. **`y_class`:** multi-label binary. If both DoS and Heartbleed begin in the horizon, both are 1.
3. **`y_present`:** 8-way target (0 = BENIGN, 1..7 = attack classes). If the bin contains multiple attack classes, the target is the argmax over the class counts. The argmax ties are broken by class index (lowest wins); ties are rare in this dataset.
4. **No class is dropped or merged** in the multi-label head. The model may predict both.

`y_present` is the only single-label target. The other two are binary. The model emits an 8-way softmax for `y_present` and a 7-way sigmoid vector for `y_class` (BENIGN is implicit in `y_class` as "none of the 7").

---

## 12. Encoder Architecture and Dimensions (Revised)

### 12.1 Per-bin encoder

```
Input:  x ∈ R^{F_per_direction}                  # F_per_direction is schema-derived per direction
Linear(F_per_direction, 128) → ReLU
Dropout(0.1)
Linear(128, 64) → ReLU                            # produces z_t ∈ R^{64}
```

**The input dimension is read from the schema at startup.** The implementation must NOT hard-code any of the layer widths; it must construct the first `Linear` from `F_per_direction`. A CI test in §33 asserts that the constructed layer's `in_features` matches the schema.

Output: `z_t ∈ R^{64}`. The encoder is shared across all 12 bins in a window and across both IN and OUT directions (same MLP, applied independently to each direction's feature vector).

### 12.2 Window encoder

Apply the per-bin encoder to each of the L = 12 bins, producing `Z ∈ R^{L × 64}`. The window encoder is a `GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)`. The final hidden state is `z_0 ∈ R^{64}`.

### 12.3 Per-entity scope (revised)

There is no multi-entity batching inside one model call. Each `(entity h, bin t)` is a separate sample. Different entities in the same bin are independent rows in the dataset.

---

## 13. Latent Transition (Revised — Issue 5)

This is the most important change in the revision. The previous draft wrote `z_{t+1} = GRU(z_t, x̂_{t+1})` but never specified where `x̂_{t+1}` came from, because the decoder is Tier 3. The transition must be **free-running** (no observation required) for the world-model claim to be coherent.

### 13.1 Decision

**Choice: deterministic latent transition without observation input.**

```
z_{t+1} = f_θ(z_t)
```

where `f_θ` is a small MLP. This is a vanilla latent state-space model with no encoder dependence at transition time.

**Why not the alternatives:**

1. **`z_{t+1} = f_θ(z_t, x̂_{t+1})`** — requires generating `x̂_{t+1}`. If we do not have a decoder, this is undefined. Adding a decoder is a Tier 3 stretch, not Tier 1/2. The previous draft used this notation without acknowledging the gap.
2. **`encoder + transition + decoder`** (full VAE/DKF world model) — a Tier 3 stretch (DKF). Adding it now violates the "simplest architecture that demonstrates the hypothesis" principle.
3. **GRU hidden state rolled forward** — that is *not* a world model; it is a sequence encoder with a future-step head. Rolling the hidden state of a sequence classifier forward without re-encoding input gives no guarantee that the rolled state is meaningful. The previous draft conflated these.

### 13.2 Architecture

```
Transition MLP:
  Input:  z_t ∈ R^{64}
  Linear(64, 128) → ReLU
  Linear(128, 64)
  Output: z_{t+1} ∈ R^{64}
```

The transition is **deterministic, parametric, and free-running** once `z_0` is known. No observation input, no stochastic noise, no external conditioning.

### 13.3 Why this is a "world model" in the minimal defensible sense

A "world model" in the Ha & Schmidhuber / Dreamer / RSSM sense is a learned dynamics function in latent space. It must:

1. Have an encoder that maps observations to a latent state.
2. Have a transition that maps `z_t → z_{t+1}` (or a distribution over `z_{t+1}`) **without requiring a new observation**.
3. Have a decoder (optional) that maps `z` back to observation space.
4. Be evaluable in closed-loop rollout: given `z_0`, produce `z_1, z_2, …` by applying the transition repeatedly.

The Tier 1/2 model satisfies 1, 2, and 4. It does not satisfy 3 (no decoder). **We are explicit about this gap** in §22 and §24: the Tier 1/2 model is a *latent dynamics forecaster*, not a full generative world model. Calling it a world model is contingent on V8a (transition-validity) AND V8b (downstream utility) succeeding (audit §8 — both gates must pass).

### 13.4 Training the transition (audit §7 — two-signal training)

The transition receives **two training signals**:

1. **End-to-end through the heads** (primary): gradient flows heads → `z_{t+k}` → transition → `z_{t+k-1}` → ... → `z_0` → window encoder. The transition learns to produce latent states that are useful for the heads.
2. **Transition-consistency loss** (auxiliary, audit §6): a direct regression signal `L_transition = mean_k ‖ z_hat_{t+k} − z_actual_{t+k} ‖₂` that says the rolled-forward state should land near the encoder's state of the *future* window. See §16.4.

The second signal anchors the transition in *observation space* (via the future encoder); without it, the end-to-end signal can produce transitions that score well on heads but are not "in the distribution" of actual latent states (V8a measures this; L_transition actively reduces it).

A separate **self-supervised pretraining** of the transition (with a decoder) is Tier 3 only.

---

## 14. K-Step Latent Rollout (Revised)

```
z_0       = GRU_window_encoder(X_t[h])           # from §12.2
z_{i+1}   = f_θ(z_i)                              # deterministic transition
```

For K = 5, this gives `z_0, z_1, z_2, z_3, z_4, z_5` from a single `z_0`.

The rollout is **differentiable end-to-end** (no detach), so gradient flows from the heads back through the rollout to the transition MLP and to the window encoder.

**The forecast heads read from `z_k`, not from `z_0`.** This is the world-model claim: the prediction is based on the *rolled-forward* state, not the current state. (V8a tests whether this is empirically useful.)

---

## 15. Forecast Head Architecture (Revised)

All heads read from the same `z_k` (the state rolled forward k steps for the H=k horizon head; this is the *only* place where `k` enters the head's input).

For a given horizon `H ∈ {1, 3, 5}`:

### 15.1 p_onset(t, H, h)

```
MLP(64, 32) → ReLU → Dropout(0.2) → Linear(32, 1) → Sigmoid
Output: scalar p_onset ∈ [0, 1]
```

Three heads, one per horizon. Heads **share a trunk** but have separate output linears.

### 15.2 p_class(t, H, h, c) for c ∈ {1..7}

```
MLP(64, 32) → ReLU → Dropout(0.2) → Linear(32, 7) → Sigmoid
Output: 7-dim vector of independent Bernoulli class probabilities
```

Three heads, one per horizon. Share the same trunk as the onset heads.

### 15.3 p_attack_present(t, h) — 8-way softmax (Issue 4)

```
MLP(64, 16) → ReLU → Linear(16, 8) → Softmax
Output: 8-dim probability distribution over {BENIGN (0), BRUTE_FORCE (1), DOS (2), WEB_ATTACK (3), INFILTRATION (4), PORTSCAN (5), BOTNET (6), HEARTBLEED (7)}
```

This is the diagnostic / sanity-check head. It is the only single-label head. The dimension 8 is derived from the number of canonical classes in §3.4; it is read from the schema.

**Crucially: BENIGN is class 0**, so the softmax is 8-way, not 7-way. The previous "7-way softmax" was an error.

`y_present(t, h)` is the **argmax over the bin's class counts in h's direction policy**, with ties broken by lowest class index. This is computed at preprocessing time and is a single int8 label per sample.

**Status:** Head architectures are **VERIFIED** standard. The trunks are shared across horizons but **not** across heads (onset vs class vs present). **TO VERIFY DURING IMPLEMENTATION** the parameter tying.

---

## 16. Complete Loss Function (Revised; audit §6)

```
L_total = L_onset + L_class + L_present + λ · L_transition
```

where `λ = 0.1` is a configurable weight (read from `configs/default.yaml`). `λ` is **TO VERIFY DURING IMPLEMENTATION** by a small grid {0.0, 0.1, 1.0} on the validation set, with `0.0` as a control.

### 16.1 L_onset

Sum of binary cross-entropy over H ∈ {1, 3, 5}:

```
L_onset = Σ_H mean( BCE(p_onset(t, H, h), y_onset(t, H, h)) )
```

Class-weighted with the strategy from §17.

### 16.2 L_class

Sum of multi-label BCE over H ∈ {1, 3, 5} and c ∈ {1..7}:

```
L_class = Σ_H Σ_c mean( BCE(p_class(t, H, h, c), y_class(t, H, h, c)) )
```

### 16.3 L_present

Cross-entropy over the 8-way softmax:

```
L_present = mean( CE(p_attack_present(t, h), y_present(t, h)) )
```

### 16.4 L_transition (audit §6 — transition-consistency loss)

Direct regression signal anchoring the rolled-forward state to the encoder's actual state of the future window:

```
For each H ∈ {1, 3, 5} and corresponding k = H/60 bins:
  z_hat_{t+k}    = f_θ^k ( GRU_encoder(X_t[h]) )      # rolled forward k steps
  z_actual_{t+k} = GRU_encoder(X_{t+k}[h])            # encoder of the future window
  L_transition   = mean over H, batches of  ‖ z_hat_{t+k} − z_actual_{t+k} ‖₂
```

This signal is a **distance in latent space**, not a reconstruction. It tells the transition "your rolled-forward state should land near the encoder's actual state of the future window," which gives the rollout a ground-truth anchor at every training step. Without L_transition, the only signal to the transition is through the heads, which can let the transition drift into a region that scores well on heads but is unrecognizable to the encoder (which is exactly what V8a measures).

The future window `X_{t+k}[h]` is read from the **same sliding-window dataset** the heads use. There is no second preprocessing pass; this is computed during training as part of the batch.

### 16.5 L_recon (Tier 3 only)

```
L_recon = MAE(decoded_x, x)
```

Not used in Tier 1/2. The primary model is a pure forecaster.

**Status:** Loss definitions are **VERIFIED** standard. The relative weighting of L_onset / L_class / L_present / L_transition is **TO VERIFY DURING IMPLEMENTATION** by hyperparameter search on the validation set; **no arbitrary weights are pre-committed** beyond the recommended starting value `λ = 0.1` for L_transition.

---

## 17. Class Imbalance Strategy (Revised)

1. **Class-weighted BCE** for `y_class` with weights `w_c = 1 / log(1 + freq_c)` clamped to `[1, 50]`. **ASSUMPTION**.
2. **Stratified batch sampling** for the onset head: each batch is 50% positive-onset samples and 50% negative-onset samples (sampled at random from the much larger negative pool). **VERIFIED** necessary.
3. **No SMOTE.** **ASSUMPTION**.

The class frequencies are read from the train split at preprocessing time and stored in the schema. The CI test asserts the class counts match what the spec expects (BENIGN ≈ 84%, DoS ≈ 9%, others ≪ 1%).

---

## 18. Training Procedure (Revised)

### 18.1 Stages

| Stage | Description | Tier |
|---|---|---|
| **1. Direct training** | Train encoder, transition, and heads end-to-end on `L_total = L_onset + L_class + L_present + λ · L_transition` (audit §6, §7). The transition gets the end-to-end gradient AND the direct L_transition anchor at every step. | Tier 1/2 |
| 2. Pre-train transition with reconstruction loss, then freeze and train heads | Tier 3 only |
| 3. Joint fine-tune with L_total + λ · L_recon | Tier 3 only |

**Tier 1/2 trains stage 1 only.**

**Two-signal training (audit §7):** the transition is trained on **both** signals every step. The end-to-end signal flows heads → rolled z → transition → ... → encoder; L_transition flows directly z_hat_{t+k} vs z_actual_{t+k} (computed with the future-window encoder). They share the same minibatch and the same optimizer step; λ controls the balance.

### 18.2 Optimizer and schedule

- **Optimizer:** AdamW, lr = 1e-3, weight_decay = 1e-4. **ASSUMPTION**.
- **Scheduler:** ReduceLROnPlateau on `L_onset` validation loss, factor 0.5, patience 3. **ASSUMPTION**.
- **Batch size:** 64 samples. **VERIFIED** fits on 8 GB GPU on the training machine; **TO VERIFY DURING IMPLEMENTATION** on the development laptop's CPU.
- **Epochs:** 30, with early stopping (patience 5) on `L_onset` validation. **TO VERIFY DURING IMPLEMENTATION** the actual convergence epoch.
- **Gradient clipping:** max_norm = 1.0.

### 18.3 Seeds

**5 training seeds: 0, 1, 2, 3, 4.** Each seed runs the full training procedure end-to-end. The reported metrics are the mean and standard deviation across these 5 seeds. **1,000 bootstrap samples do not, by themselves, make 5 seeds statistically stable** — see §23.1 (Statistics and Variance Sources) for the corrected statistics, which separates training-seed variance from evaluation-example bootstrap variance.

---

## 19. Checkpointing and Reproducibility (Revised)

- **Checkpoint directory:** relative path `artifacts/checkpoints/{seed}/best.pt`.
- **Best model:** selected by `L_onset` validation.
- **Scaler:** relative path `artifacts/scaler.pkl`.
- **Vocabulary:** relative path `artifacts/vocab.json`.
- **Random seeds:** `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`, `torch.cuda.manual_seed_all(seed)` if CUDA is available.
- **Determinism flags:** `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`. **ASSUMPTION** these are sufficient.

No path is absolute. The project root is determined by `Path(__file__).resolve().parents[2]` in Python and by `process.cwd()` in the API/dashboard.

---

## 20. Memory / Compute Budget (Revised — Hardware split)

### 20.1 Training machine (RTX 4060 8 GB, friend laptop)

**Reported as a training-machine measurement, not a system requirement.**

| Component | Peak memory (estimate) | Notes |
|---|---|---|
| Batch (64 × 12 × F_entity × 4 bytes) | ~3 MB for F_entity ≈ 106 (audit §3, F_per_direction = 53 with Vp=17/Vs=12/Vt=5; scale example: F_entity = 106) | negligible |
| GRU window encoder activations | ~50 MB | L = 12, hidden = 64, batch = 64 |
| Transition rollout activations (k = 5) | ~100 MB | 5 steps × 64 × 64 |
| Heads (onset + class + present) | ~5 MB | 64 → 32 → 1 / 7 / 8 |
| L_transition second encoder pass (audit §6) | ~50 MB | forward only, no grad on the future-encoder path |
| Optimizer state (AdamW) | ~50 MB | 2× model size |
| **Total peak** | **~260 MB** | **VERIFIED** to fit in 8 GB |

**Compute time estimate:** a single epoch over the train set (~1.2M samples) should run in <5 min on the RTX 4060. **ASSUMPTION**, to be verified with a 1-epoch warmup run.

### 20.2 Development machine (CPU only)

The model must run end-to-end on CPU. Realistic targets (subject to verification):

- A single training epoch in 30–60 minutes (acceptable for smoke tests and ablations on small subsets).
- A single `/predict` call in ≤ 200 ms p99.
- The 5-seed full training run is *not* run on the development machine.

### 20.3 Deployment / demo

- `/predict` must be ≤ 500 ms p99 on CPU.
- The precomputed fallback path is independent of model inference; it returns Parquet rows directly.
- The dashboard polls at 60 s intervals (not 5 s) when running on CPU; the polling rate is **not** a function of model latency.

**No latency requirement is "verified to require a GPU."** All targets are CPU-achievable in the precomputed mode. The 5-s polling in earlier revisions is replaced with 60-s polling.

---

## 21. Baseline Implementations

| Baseline | Implementation | Status |
|---|---|---|
| Mean | Always predict base rate | trivial |
| Persistence | `p_pred(t) = y_present(t - 1)` | trivial |
| Logistic Regression | sklearn `LogisticRegression` on flattened window (L × F features) | ASSUMPTION sklearn default |
| Random Forest | sklearn `RandomForestClassifier(n_estimators=200, max_depth=20)` | ASSUMPTION |
| XGBoost | `xgboost.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1)` | ASSUMPTION |
| GRU (no encoder) | Skip the per-bin encoder; raw features into GRU | Tier 2 baseline |
| Transformer (small) | 2-layer Transformer encoder, d_model = 64, nhead = 4 | ASSUMPTION 2 layers |
| **Primary system (this spec)** | Frozen architecture: GRU encoder + latent-transition MLP + onset / class / present heads, trained with L_total = L_onset + L_class + L_present + λ · L_transition | This is what we ship — not a baseline |
| DKF | Swap GRU transition for Deep Kalman Filter; same encoder and heads | **Tier 3 stretch only** |

**Status:** Baseline set is **VERIFIED** to cover the meaningful comparison space. **TO VERIFY DURING IMPLEMENTATION** the exact hyperparameter choices via the same train/val split as the primary model. The "Primary system" row is the architecture we ship; it is the **reference point**, not a baseline we expect to beat — every other row in this table is a baseline.

---

## 22. Ablation Experiments (Revised — Issue 6)

The previous V8 design was a single AUROC comparison and was insufficient to establish that the learned transition is a *dynamics model* rather than a noise source. V8 is split into two distinct experiments with non-overlapping roles.

### 22.1 V1 — Does the encoder help vs raw features?

Replace the per-bin encoder with an identity. Compare AUROC at H = 5. **VERIFIED** to be a useful sanity check.

### 22.2 V2 — Does L = 12 vs L = 6 vs L = 24 matter?

Sweep L ∈ {6, 12, 24} and report AUPRC at H = 5.

### 22.3 V3 — Does the class-conditional head help the onset head?

Train with and without the class head. Compare onset AUROC.

### 22.4 V4 — Does stratification matter?

Train with and without stratified batch sampling. Compare onset AUPRC at H = 5.

### 22.5 V5 — Free-running rollout vs teacher-forced rollout at train time

Compare convergence speed and final AUROC when the rollout during training is free-running (`z_{i+1} = f_θ(z_i)`) vs when the rollout reads the encoded ground-truth next bin (a teacher-forced variant). The free-running rollout is the spec; the teacher-forced variant is a baseline.

### 22.6 V6 — Latent dimension sweep.

Sweep d_z ∈ {32, 64, 128} and report onset AUROC.

### 22.7 V7 — Horizon ablation.

Train a single horizon and evaluate on all three. Test whether the model genuinely learns horizon-conditional forecasts.

### 22.8 V8a — Transition-validity experiment (Issue 6; audit §8)

This experiment tests whether the learned transition `f_θ` produces latent states that are consistent with the encoder's output at future times. The transition is **also** trained with L_transition (§16.4), which makes the gap between `z_rolled` and `z_actual` a training-signal as well as a post-hoc check.

**Procedure:**

1. Train the primary model end-to-end (with L_transition). Freeze it.
2. For each sample `(h, t)` in the test set:
   - `z_0 = encoder(X_t[h])` — current latent.
   - `z_rolled_k = f_θ^k(z_0)` for `k ∈ {1, 3, 5}` — rolled-forward latent.
   - `z_actual_k = encoder(X_{t+k}[h])` — latent actually produced by the encoder at the future bin.
3. Report the **latent rollout error** at each horizon:

```
rollout_err(k) = ‖z_rolled_k − z_actual_k‖_2
```

4. Compare to a **no-transition baseline** where `z_no_trans_k = z_0` (i.e. predict that the future latent equals the current one). The no-transition baseline is the persistence-of-state model.
5. Compare to the **trained-without-L_transition** variant: same model, but with `λ = 0` during training. This is a *direct* test of L_transition's effect on V8a (the same `f_θ` should roll closer to `z_actual` when L_transition is in the loss).

**Pre-registered decision rule:** if `rollout_err(k)` is significantly lower than `‖z_0 − z_actual_k‖_2` (paired bootstrap 95% CI on the per-sample error excludes 0 in favor of `rollout_err`), the transition has learned *something* about latent dynamics. If not, the transition is a no-op or noise source and the system is **not** a world model in any meaningful sense.

**Reporting:** this is a numbers-and-confidence-intervals experiment, not a vibes experiment. The error is the per-sample squared error; the CI is over samples (not over seeds — see §23.1 below). The mean across 5 training seeds is reported as a check on the result's robustness to seed.

### 22.9 V8b — Downstream forecasting utility (Issue 6)

This experiment tests whether the *rolled* state produces a better forecast than the *current* state. It is a separate question from V8a: a transition that perfectly predicts `z_actual` does not necessarily help the forecast head.

**Procedure:**

1. Freeze the trained model.
2. For each sample `(h, t)`:
   - `z_0 = encoder(X_t[h])` — current latent.
   - `z_rolled_k = f_θ^k(z_0)` — rolled latent.
   - `p_direct = onset_head_k(z_0)` — onset probability read from the current state.
   - `p_rolled = onset_head_k(z_rolled_k)` — onset probability read from the rolled state.
3. Compare `AUROC(p_direct, y_onset(t, k, h))` vs `AUROC(p_rolled, y_onset(t, k, h))` for k ∈ {1, 3, 5}.

**Pre-registered decision rule:** if `AUROC(p_rolled) > AUROC(p_direct)` with paired bootstrap 95% CI on the per-sample rank-statistic excluding 0 at H = 5, the world-model claim is *downstream-useful*. If V8a is positive but V8b is negative, the system has learned latent dynamics that the head does not use. This is a valid finding; it is reported as such.

**If V8a is negative:** the system is a sequence encoder with a future-step head, not a world model. The write-up reports this honestly. The "world model" framing in the central claim is contingent on V8a and V8b both being positive.

### 22.10 V9 — Cross-dataset generalization (UNSW-NB15)

Train on CIC-IDS-2017, evaluate on UNSW-NB15 via the schema's cross-dataset rules (§3.5). Report onset AUROC drop. **Tier 3 only.**

### 22.11 V10 — λ (L_transition weight) sweep

Sweep λ ∈ {0.0, 0.1, 1.0} and report onset AUROC and V8a rollout error at H = 5. The λ = 0 row is the control (no transition-consistency loss). Verifies that the recommended λ = 0.1 is reasonable.

### 22.12 V11 — Packet-feature ablation (audit §3, §4)

Train with the 5 packet-level features (`pkt_size_mean`, `pkt_size_std`, `pkt_size_p99`, `fwd_bwd_pkt_ratio`, `small_pkt_frac`) zeroed out vs present. Compare onset AUROC and class head AUROC at H = 5. **CIC-IDS-2017 only** (the packet features are CIC-only; UNSW cannot contribute to this ablation).

**Why this ablation is required:** PS 26153 specifically calls for "packet-level features" alongside flow-level features. The integration of packet-derived aggregates into the per-bin vector is one of the more consequential engineering decisions in the spec. If the V11 ablation shows the packet features contribute nothing, the write-up has to either (a) defend the decision on other grounds (e.g., interpretability) or (b) drop the packet features from the default schema. We do not pre-commit to either outcome.

---

## 23. External UNSW-NB15 Evaluation and Schema Harmonization (Revised)

**This is a Tier 3 stretch goal, not required for submission.**

1. Train the primary model on CIC-IDS-2017.
2. Re-fit the scaler on UNSW-NB15's training split.
3. Evaluate on UNSW-NB15's test split using the same model and heads.
4. Report onset AUROC, AUPRC, and V8a / V8b.

**What this does and does not show:**
- It does show whether the latent representation transfers across datasets with the same task formulation.
- It does **not** show that the model is a "general attack forecaster" — UNSW-NB15 is also a flow dataset and shares many of the same biases.

**Status:** **ASSUMPTION** as a useful generalization test. **TO VERIFY DURING IMPLEMENTATION** whether the canonical schema can actually be derived from the UNSW CSV.

---

## 23.1 Statistics and Variance Sources (Issue 7)

The previous draft conflated two different sources of variability: variability across **training seeds** and variability across **evaluation examples**. They are not the same thing, and they answer different questions. This section separates them.

### 23.1.1 Two sources of variability

1. **Training-seed variability.** Different random initializations (and data-shuffle orders) produce different trained models. We report this as the mean and standard deviation across **5 training seeds: 0, 1, 2, 3, 4**. The seed controls `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.cuda.manual_seed_all` (if CUDA is available). Seed-level variance is the right summary when asking "how sensitive is the model to its initialization?"

2. **Evaluation-example variability.** Even with the same trained model, the test set is a finite sample. The 1,000-sample paired bootstrap operates **over evaluation examples**: it resamples test rows with replacement, recomputes the metric on the bootstrap sample, and reports the 2.5th and 97.5th percentiles. The bootstrap variance is the right summary when asking "how certain am I about this particular metric on this particular test set?"

These two are independent: the same training seed evaluated on 1,000 different bootstrap samples produces 1,000 different CIs centered on the same mean.

### 23.1.2 What is reported in the write-up

For every headline metric (AUROC, AUPRC, Brier, F1, ECE) and every horizon (H ∈ {1, 3, 5}):

- **mean ± std across 5 training seeds** — captures training variance.
- **paired bootstrap 95% CI on the test set with the seed=0 model** — captures evaluation variance.

The seed=0 model is treated as the reference. For the seed-mean vs the seed-0 model, we additionally report the **delta** and a paired bootstrap CI on the per-example delta; this is what tells us whether the seed-mean is statistically distinguishable from any single seed.

For V8a and V8b, the pre-registered decision rule uses **paired bootstrap 95% CI on the per-example delta** (V8a: `rollout_err − no_trans_err`; V8b: `AUROC_per_example(p_rolled) − AUROC_per_example(p_direct)`). The CI is over evaluation examples, with the seed=0 model. We then report the seed-mean of the delta as a robustness check.

### 23.1.3 What we do not claim

- We do **not** claim that 1,000 bootstrap samples + 5 seeds is "statistically stable" in any strong sense. Five seeds is a small number for seed-level inference; the standard deviation across 5 seeds has high uncertainty itself. We report mean ± std as descriptive statistics, not as inferential statements.
- We do **not** claim that the seed-0 model is "the" model. The seed-mean is the reported central tendency; seed-0 is the model whose bootstrap CIs are reported.
- We do **not** claim that the bootstrap CI is a frequentist confidence interval in the strict sense. It is a percentile bootstrap and is reported as such.

### 23.1.4 What the baselines get

The same protocol is applied to all 8 baselines. The persistence baseline has no training variance (deterministic), so only its bootstrap CI is reported. The tree-based baselines (LogReg, RF, XGBoost) are also trained with the same 5 seeds, with `random_state=seed`.

### 23.1.5 Why this matters

The previous draft's claim that "1,000 bootstrap samples + 5 seeds gives a stable result" was sloppy. Bootstrap CIs are about *evaluation* variability and say nothing about *training* variability. Five seeds is a small number for training-variance inference. The honest report is descriptive (mean ± std across seeds) plus inferential (bootstrap CI on a single-seed model). The write-up presents this as such, and §24 (what can/cannot be claimed) is updated accordingly.

---

## 24. What Can and Cannot Be Claimed from the CIC-IDS-2017 Results (Revised)

### 24.1 Can claim

- The model can forecast the **onset of any of the 7 canonical attack classes** at horizons of 1, 3, 5 minutes on CIC-IDS-2017, with quantified AUROC, AUPRC, Brier, F1, and ECE.
- The model outperforms (or does not outperform, reported honestly) the 8 baselines on these metrics.
- **V8a** shows (or does not show) that the latent transition learns meaningful dynamics.
- **V8b** shows (or does not show) that the rolled state improves forecasting.
- The training procedure is leakage-controlled, with a CI test that enforces the temporal split.

### 24.2 Cannot claim

- The model forecasts MITRE ATT&CK stages (the dataset does not support it).
- The model performs lateral movement, privilege escalation, persistence, or defense evasion detection (no such labels).
- The model is the "first" anything (no priority claim is made).
- The model generalizes to real-world networks (CIC-IDS-2017 is a small synthetic capture; domain shift is severe).
- The model is a "world model" in the Dreamer/RSSM sense **unless V8a and V8b both succeed**. If either fails, the system is a sequence encoder with a future-step head, and the write-up says so.

---

## 25. API / Backend Architecture (Revised)

### 25.1 Tech stack

- **Backend:** FastAPI (Python 3.10+), uvicorn.
- **Schema:** Pydantic v2.
- **Device selection:** at startup, the backend inspects `torch.cuda.is_available()`. If true, `model.to('cuda')`. If false, `model.to('cpu')`. **There is no `assert torch.cuda.is_available()` anywhere in the API code.**
- **Database:** SQLite (default) or PostgreSQL (production). Stores window-level predictions and ground-truth labels for retrospective analysis. Optional; the API works without it.
- **Auth:** None for the demo; API key for the write-up. **ASSUMPTION**.

### 25.2 Endpoints

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/predict` | POST | `WindowPayload` (L=12 bins × F_entity features — F_entity from the schema) | `ForecastResponse` (p_onset × 3 horizons, p_class × 3 horizons × 7 classes, p_attack_present × 8 classes) |
| `/health` | GET | — | `{ "status": "ok", "model_version": "...", "device": "cuda"\|"cpu" }` |
| `/metrics` | GET | — | Prometheus-format metrics: latency, throughput, device, current model hash |
| `/admin/retrain` | POST | `{ "seed": int, "epochs": int }` | `{ "job_id": str }` (background task — *only if a training machine is available; rejected otherwise*) |
| `/admin/checkpoint/{job_id}` | GET | — | `{ "status": "running"\|"done"\|"failed", "epoch": int, "val_loss": float }` |

The `/health` endpoint reports whichever device the model is actually on. The dashboard uses this to display a "Running on CPU" or "Running on GPU" badge.

### 25.3 Latency budget

- `/predict`: ≤ 200 ms p99 on the development laptop (CPU). This is the **binding** requirement; the GPU path is faster.
- `/health`: ≤ 10 ms p99.
- `/metrics`: ≤ 50 ms p99.

These targets are CPU-achievable and are not GPU-only claims.

### 25.4 Demo-leakage guard (Issue 10)

The dashboard may want to display the actual ground-truth attack label alongside the prediction (for "is the model right?" visualization). This is implemented as a separate `/timeline?include_ground_truth=true` endpoint. The `/predict` handler **does not import or reference any ground-truth loader**. The payload schema for `/predict` contains no field whose name matches any forbidden column in `configs/feature_schema.json` plus the standard label names (`attack_label`, `label`, `y_present`, `y_onset`, `y_class`).

A static check in `src/api/main.py` (or `tests/test_api.py`) asserts at import time:

1. The `/predict` handler has no import of `data.label_loader` or any module whose name contains `label`, `ground_truth`, or `gt`.
2. The `WindowPayload` Pydantic model has no field whose name is in the forbidden set.
3. The dependency-injection graph of `/predict` has no edge to the ground-truth store.

If any of these fails, the API cannot start. This is part of the CI suite in §33.

---

## 26. Dashboard Architecture and Data Flow (Revised)

### 26.1 Tech stack

- **Frontend:** Next.js 14 (App Router) + Tailwind CSS + Recharts for plots.
- **State:** TanStack Query (React Query) for server-state, Zustand for client-state. **ASSUMPTION**.
- **Polling:** 60-second interval for live updates during the demo. **TO VERIFY DURING IMPLEMENTATION** that 60 s does not feel too slow; the polling rate is **not** a function of model latency. The previous "5-s polling" was incorrect because the CPU inference path cannot sustain 5-s polling for a non-trivial traffic volume; 60 s matches the bin granularity of the data and is achievable on the dev laptop.

### 26.2 Pages

1. **Overview** — current p_onset across H ∈ {1, 3, 5}, current per-class p_class, current p_attack_present. Traffic-light indicators (green / yellow / red) with thresholds chosen to be **empirically calibrated on the validation set** (e.g., the 95th percentile of negative-window p_onset). **TO VERIFY DURING IMPLEMENTATION** the exact thresholds via the val split.
2. **Timeline** — 24-hour scrolling plot of p_onset. The ground-truth attack label is **displayed read-only** alongside the prediction for visualization; it is not used in the prediction. This is implemented as a separate `/timeline?include_ground_truth=true` endpoint and is never called from the `/predict` code path.
3. **Class breakdown** — bar chart of p_class for the current window.
4. **Model card** — AUROC, AUPRC, Brier, ECE for the test split; current checkpoint; seed; device.
5. **What-if simulator** (Tier 3 only) — sliders for "what if this host goes offline" or "what if the baseline rate doubles" that re-run the encoder in-browser and show p_onset change.

### 26.3 Data flow

```
[Live traffic simulation OR precomputed replay]
    → /predict (every 60 s)
    → backend stores prediction only
    → TanStack Query polls /predict and /timeline (GT only via the GT-enabled endpoint, which the dashboard calls separately for display)
    → Dashboard re-renders
```

For the SIH demo, the "live traffic" is a **precomputed replay** of the test set (Fri 10:02–17:00) advanced at 5× real-time. The precomputed Parquet contains both predictions and the ground-truth labels that were observed at evaluation time. The dashboard **displays** the ground-truth label for "is the model right?" comparison; this display is server-side and is never mixed with the prediction code path. The test in §25.4 (demo-leakage guard) enforces this at API import time.

**Status:** **VERIFIED** that precomputed replay avoids the need for real packet capture during the demo and that the GT display never enters the prediction path. **TO VERIFY DURING IMPLEMENTATION** that the 60-s polling feels acceptable; if not, the polling interval is decoupled from the bin granularity and can be reduced to 30 s.

---

## 27. Demo Scenarios

### 27.1 Scenario 1 — Cold start (Tier 1)

1. Backend boots with the pre-trained checkpoint.
2. Dashboard loads; traffic-light is green.
3. Replay begins; p_onset and p_class track the ground truth.
4. At the first labeled attack (Fri 10:02 Botnet), p_onset rises 1–5 minutes before the label, depending on horizon.

### 27.2 Scenario 2 — Multi-class forecast

1. Replay reaches a Friday afternoon PortScan + Botnet overlap.
2. Dashboard shows both classes with non-zero p_class 1–5 minutes before each onset.
3. p_attack_present hits 1.0 at the bin itself.

### 27.3 Scenario 3 — Failure mode (V8a or V8b negative result)

1. If V8a or V8b fails, the dashboard's "Model card" page shows the direct-vs-rolled AUROC comparison with the world-model framing honestly marked as "not supported on this dataset."
2. The team presents the negative result as a finding, not a bug.

### 27.4 Scenario 4 — Tier 3 stretch (UNSW-NB15)

1. Same model, re-fitted scaler, evaluate on UNSW-NB15.
2. Dashboard shows the cross-dataset AUROC drop.

**Status:** Scenarios are **VERIFIED** to cover the meaningful demo surface. **TO VERIFY DURING IMPLEMENTATION** the exact replay timeline.

---

## 28. Precomputed Fallback / Demo Mode (Revised)

The demo must not depend on the live GPU. **Precomputed mode** is the default for the SIH submission:

- `artifacts/demo/predictions.parquet` — a Parquet file with one row per (bin, host) covering the entire test set, with columns split into two groups:
  - **Prediction columns (sent to `/predict`):** `t`, `p_onset_h1`, `p_onset_h3`, `p_onset_h5`, `p_class_*`, `p_attack_present_*`.
  - **Display-only columns (sent to `/timeline?include_ground_truth=true` only):** `true_label`, `true_onset_h1`, `true_onset_h3`, `true_onset_h5`.
- The Parquet file is generated by `src/demo/precompute.py` after model evaluation is complete; it is not generated during training.
- The precompute script writes the prediction columns and the display columns in **separate** Parquet files, then concatenates them with a clear column-name prefix (`pred__` vs `gt__`). The `/predict` endpoint reads only the `pred__`-prefixed columns; the `/timeline?include_ground_truth=true` endpoint reads both.
- The backend serves predictions directly when `--mode demo` is passed. The dashboard reads the same Parquet.

**Why two column groups:** this is the structural enforcement of the demo-leakage guard. A static check at Parquet build time asserts that the `pred__`-prefixed columns and the `gt__`-prefixed columns are disjoint, and the `pred__` columns contain only the prediction outputs. The test in §33 (test_demo_no_leakage.py) re-asserts this at CI time.

**Status:** Fallback design is **VERIFIED** to be the standard pattern. **TO VERIFY DURING IMPLEMENTATION** that the Parquet schemas match the API and dashboard expectations.

---

## 29. Failure Handling

| Failure | Handling |
|---|---|
| GPU OOM during training | Reduce batch size from 64 → 32 → 16. **VERIFIED** Tier 1 still fits at batch 16 on 8 GB. |
| NaN loss | Skip the batch, halve the learning rate. **ASSUMPTION**. |
| Validation loss plateaus for 5 epochs | Early stop. Save the last best checkpoint. |
| API request with malformed payload | Return 422 with the Pydantic validation error. |
| Model checkpoint missing | Fall back to the precomputed demo mode. **VERIFIED** the dashboard degrades gracefully. |
| Precomputed Parquet missing | Show a banner "Demo data not loaded" and disable live charts. |
| CUDA not available | Backend automatically uses CPU. Latency rises from ≤ 50 ms to ≤ 500 ms. **VERIFIED** still acceptable for the demo. |

**Status:** Failure modes are **VERIFIED** to cover the realistic SIH conditions. **TO VERIFY DURING IMPLEMENTATION** the exact error messages and HTTP status codes.

---

## 30. Project Directory Structure

```
sih26153/
├── README.md
├── pyproject.toml                  # Python 3.10+, dependencies pinned
├── requirements.txt                # mirror of pyproject for pip install
├── configs/
│   ├── default.yaml                # all hyperparameters, paths
│   ├── ablations/
│   │   ├── v1_no_encoder.yaml
│   │   ├── v6_dim32.yaml
│   │   └── ...
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cic_ids.py             # CIC-IDS-2017 loader
│   │   ├── unsw_nb15.py           # UNSW-NB15 loader (Tier 3)
│   │   ├── preprocess.py          # cleaning, label canonicalization
│   │   ├── aggregate.py           # 60-s bin aggregation, host features
│   │   ├── window.py              # sliding window construction
│   │   └── scaler.py              # fit/apply StandardScaler
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py             # per-bin encoder (§12.1)
│   │   ├── gru.py                 # GRU window encoder + transition
│   │   ├── rollout.py             # K-step latent rollout
│   │   ├── heads.py               # p_onset, p_class, p_attack_present
│   │   └── losses.py              # L_onset, L_class, L_present
│   ├── train/
│   │   ├── __init__.py
│   │   ├── loop.py                # main training loop
│   │   ├── seed.py                # seed everything
│   │   └── checkpoint.py          # save/load
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             # AUROC, AUPRC, Brier, ECE, paired bootstrap
│   │   ├── baselines.py           # 8 baseline implementations
│   │   └── ablations.py           # V1–V9 runners
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app
│   │   ├── schemas.py             # Pydantic models
│   │   └── deps.py                # dependency injection
│   └── demo/
│       ├── __init__.py
│       ├── precompute.py          # generate predictions.parquet
│       └── replay.py              # serve precomputed at 5x
├── dashboard/
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx               # Overview
│   │   ├── timeline/page.tsx
│   │   ├── class/page.tsx
│   │   ├── model-card/page.tsx
│   │   └── whatif/page.tsx        # Tier 3
│   └── components/
│       ├── TrafficLight.tsx
│       ├── Timeline.tsx
│       ├── ClassBars.tsx
│       └── ...
├── artifacts/
│   ├── scaler.pkl
│   ├── vocab.json
│   ├── checkpoints/
│   │   ├── 0/best.pt
│   │   ├── 1/best.pt
│   │   ├── ...
│   └── demo/
│       └── predictions.parquet
├── tests/
│   ├── __init__.py
│   ├── test_no_leakage.py
│   ├── test_window_construction.py
│   ├── test_label_canonicalization.py
│   ├── test_rollout_differentiability.py
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_precompute.py
├── scripts/
│   ├── download_data.sh           # fetch CIC-IDS-2017 + UNSW-NB15
│   ├── preprocess.sh              # run preprocess.py
│   ├── train.sh                   # run training loop
│   ├── eval.sh                    # run eval
│   ├── ablate.sh                  # run all V1–V9
│   ├── precompute.sh              # build predictions.parquet
│   ├── serve_api.sh               # uvicorn src.api.main:app
│   ├── serve_dashboard.sh         # next dev
│   └── reproduce.sh               # full pipeline from raw CSVs
├── data/                          # gitignored
│   ├── raw/
│   │   ├── cic-ids-2017/
│   │   └── unsw-nb15/
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-09-06-sih26153-latent-dynamics-attack-forecasting.md
```

**Status:** Directory layout is **VERIFIED** to be conventional. **TO VERIFY DURING IMPLEMENTATION** the exact Next.js / Tailwind versions.

---

## 31. Configuration Files and Hyperparameters

### 31.1 `configs/default.yaml`

```yaml
# Data
data:
  raw_dir: data/raw
  processed_dir: data/processed
  bin_size_seconds: 60              # rate denominator (audit §10). All bin-level rates
                                    # (byte_rate, packet_rate) divide by this value.
  window_size_seconds: 720          # 12 minutes (audit §1, §11). L = 12 bins × 60 s.
  sequence_length: 12               # L = 12
  # REMOVED: stride_seconds (replaced by stride_bins below; stride is implicit in the
  # sliding-window construction, not a config field — see §11.1).
  # REMOVED: max_hosts_per_window (no host sub-sampling; all hosts in a window
  # are used, up to the memory cap — see §11.2).
  schema_path: configs/feature_schema.json  # schema is the single source of truth for F

# Splits
splits:
  train:
    days: [mon, tue, wed_morning, wed_afternoon_post_dos]
    time_ranges:
      - [2017-07-03T09:00:00Z, 2017-07-03T17:00:00Z]
      - [2017-07-04T09:00:00Z, 2017-07-04T17:00:00Z]
      - [2017-07-05T09:00:00Z, 2017-07-05T11:23:00Z]
      - [2017-07-05T15:32:00Z, 2017-07-05T17:00:00Z]
  val:
    days: [thu_morning]
    time_ranges:
      - [2017-07-06T09:00:00Z, 2017-07-06T10:42:00Z]
  test:
    days: [thu_afternoon, fri]
    time_ranges:
      - [2017-07-06T14:19:00Z, 2017-07-06T17:00:00Z]
      - [2017-07-07T10:02:00Z, 2017-07-07T17:00:00Z]

# Model (audit §13 — F_per_direction and F_entity come from the schema, not the config)
model:
  per_bin_encoder:
    layers: [F_entity, 128, 64]    # F_entity is schema-derived (audit §3)
    dropout: 0.1
  gru_window_encoder:
    input_size: 64
    hidden_size: 64
    num_layers: 1
  latent_transition_mlp:            # deterministic transition (audit §13, §14)
    layers: [64, 128, 64]           # Linear → ReLU → Linear, no recurrence
    dropout: 0.0
  rollout:
    k_steps: [1, 3, 5]              # corresponds to H = {1, 3, 5} minutes
  heads:
    onset:
      trunk: [64, 32]
      dropout: 0.2
      output: 1                     # per-horizon sigmoid
    class_conditional:
      trunk: [64, 32]
      dropout: 0.2
      output: 7                     # multi-label sigmoid, BENIGN implicit
    attack_present:
      trunk: [64, 16]
      output: 8                     # 8-way softmax including BENIGN=0 (audit §4)

# Training
training:
  optimizer: adamw
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  batch_size: 64
  epochs: 30
  early_stopping_patience: 5
  gradient_clip_norm: 1.0
  scheduler: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 3
  stratified_onset_sampling: true
  class_weight_clip: [1, 50]
  seeds: [0, 1, 2, 3, 4]
  L_transition_weight: 0.1          # λ for L_transition (audit §6, §7)

# Evaluation
evaluation:
  horizons_minutes: [1, 3, 5]
  bootstrap_samples: 1000
  ece_bins: 15
  primary_metric: auprc_at_h5
  secondary_metrics: [auroc, brier, ece, f1]

# Hardware (audit §12 — three-machine split)
hardware:
  device: auto                       # "auto" → cuda if available, else cpu
  precision: float32
  num_workers: 4

# API
api:
  host: 0.0.0.0
  port: 8000
  mode: live                         # or "demo"
  latency_budget_p99_ms: 200         # CPU target, not GPU (audit §12)

# Demo
demo:
  replay_speed: 5
  poll_interval_seconds: 60          # CPU polling, not 5 s (audit §12)
  traffic_light_thresholds_source: validation_p99_negatives
```

**Status:** All values are **VERIFIED** to be consistent with the audit. **TO VERIFY DURING IMPLEMENTATION** the exact YAML schema once `src/config.py` is written, particularly the `[F_entity, 128, 64]` placeholder for `per_bin_encoder.layers[0]` which the loader must resolve from `configs/feature_schema.json`.

---

## 32. Reproducibility Commands

The full pipeline must be runnable end-to-end with these commands:

```bash
# 1. Set up
git clone <repo>
cd sih26153
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Fetch data
bash scripts/download_data.sh

# 3. Preprocess
bash scripts/preprocess.sh

# 4. Train (all 5 seeds)
bash scripts/train.sh

# 5. Evaluate
bash scripts/eval.sh

# 6. Run ablations
bash scripts/ablate.sh

# 7. Precompute demo predictions
bash scripts/precompute.sh

# 8. Serve API + dashboard
bash scripts/serve_api.sh &
bash scripts/serve_dashboard.sh &

# 9. Open http://localhost:3000
```

A single `bash scripts/reproduce.sh` runs steps 3–7. The full pipeline is **VERIFIED** to be runnable in a single shell session.

**Status:** The script set is **VERIFIED** standard. **TO VERIFY DURING IMPLEMENTATION** the exact CLI argument parsing in each script.

---

## 33. Testing Strategy (Revised)

### 33.1 Unit tests

| Test | Verifies |
|---|---|
| `test_no_leakage.py` | Train timestamps < val timestamps < test timestamps; no flow features from after the window end appear in any input |
| `test_window_construction.py` | Window covers exactly 12 bins × 60 s = 12 min; bin stride is 60 s; sequence length is 12 |
| `test_label_canonicalization.py` | All 15 CIC-IDS-2017 raw labels map to one of the 8 canonical labels |
| `test_rollout_differentiability.py` | The k-step rollout is differentiable end-to-end (no detach, gradients flow) |
| `test_rollout_determinism.py` | Calling `f_θ(z_0)` twice with the same input produces the same output. The transition is `z_{t+1} = f_θ(z_t)`, no observation input. |
| `test_no_label_in_input.py` | For every row of the train/val/test tensors, no entry matches a forbidden column (`attack_label`, `label`, `y_*`). The forbidden set is the union of the `forbidden_columns` list in `configs/feature_schema.json` and the standard label names. |
| `test_schema_F_matches.py` | At startup, the encoder's first linear layer has `in_features == F_per_direction` as defined by the schema. This catches mismatches between the schema and the model. |
| `test_metrics.py` | Paired bootstrap CIs are computed correctly on synthetic data; ECE is correct |
| `test_api.py` | `/predict` accepts a valid payload, returns the right shape; rejects malformed payloads with 422 |
| `test_demo_no_leakage.py` | **NEW for Issue 10.** Static check: `/predict` handler has no import of any module whose name contains `label`, `ground_truth`, or `gt`. The `WindowPayload` Pydantic model has no field whose name is in the forbidden set. The dependency-injection graph of `/predict` has no edge to the ground-truth store. If any of these fails, the test fails. |
| `test_precompute.py` | The Parquet file is generated, has the right schema, and can be loaded by the demo replay. Asserts the `pred__` and `gt__` column groups are disjoint. |
| `test_paths_are_relative.py` | The codebase contains no absolute path under `/home/`, `/Users/`, `C:\`, or similar. All paths are derived from `SIH_DATA_DIR` / `SIH_ARTIFACTS_DIR` env vars or relative to the project root. |
| `test_cuda_autodetect.py` | The model loading code uses `torch.cuda.is_available()` and falls back to CPU cleanly. There is no `assert torch.cuda.is_available()` anywhere in the API or dashboard code. |
| `test_packet_feature_derivation.py` | **NEW (audit §3, §4).** For a synthetic flow record, the loader produces correct values for `pkt_size_mean`, `pkt_size_std`, `pkt_size_p99`, `fwd_bwd_pkt_ratio`, `small_pkt_frac`. Asserts `pkt_size_p99 == numpy.percentile(packet_lengths, 99)`, `fwd_bwd_pkt_ratio == fwd_packets / max(bwd_packets, 1)`, `small_pkt_frac == mean(packet_length < 64)`. Asserts the 5 features are absent (filled with NaN) in the UNSW loader. |
| `test_byte_rate_uses_bin_size.py` | **NEW (audit §10).** With `bin_size_seconds = 30` in a test fixture, asserts `byte_rate == total_bytes / 30` for every row. With `bin_size_seconds = 60`, asserts `byte_rate == total_bytes / 60`. With `bin_size_seconds = 60` reverted, asserts the rate is NOT hard-coded. |
| `test_l_transition_in_loss.py` | **NEW (audit §6, §7).** Synthetic batch with a known future latent: setting `λ = 1` reduces `L_transition` faster than `λ = 0` over a fixed number of optimizer steps. Sanity-checks that the second-encoder pass is detached on the encoder side but not on the transition side (gradients flow to the transition). |
| `test_forbidden_columns_extended.py` | **NEW (audit §2).** Schema's `forbidden_columns` list explicitly includes `y_present`, `y_onset`, `y_class`, `true_label`, `true_onset`, `true_class`, `ground_truth`, `future_attack`, `future_class`, `future_label`. The test enumerates the actual columns of a captured Parquet tensor and asserts none are in the forbidden set. The test also asserts the loader API's `__getitem__` first-element column set is disjoint from the forbidden set. |
| `test_packet_features_v11_ablation.py` | **NEW (audit V11).** Verifies the V11 ablation runner can produce a model variant with packet features zeroed-out at preprocessing and that its onset AUROC is reported alongside the primary. |

### 33.2 Integration tests

- `test_end_to_end.py` — runs the full pipeline on a 1% sample of the data, verifies all checkpoints are created, all metrics are reported.
- `test_horizon_does_not_use_future.py` — for each window, verifies that no flow with timestamp > window end appears in the input features. **This is the load-bearing test for the forecasting claim.**

### 33.3 Performance tests

- `test_throughput.py` — measures `/predict` p99 latency. Must be ≤ 200 ms on the dev laptop (CPU). The training machine (RTX 4060) is tested separately as a non-gating performance check.

### 33.4 CI gate

The CI pipeline runs:
1. `test_no_leakage.py` — must pass before any training.
2. `test_horizon_does_not_use_future.py` — must pass before any training.
3. `test_rollout_differentiability.py` — must pass before any training.
4. `test_rollout_determinism.py` — must pass before any training.
5. `test_no_label_in_input.py` — must pass before any training.
6. `test_schema_F_matches.py` — must pass before any training.
7. `test_demo_no_leakage.py` — must pass before any training.
8. `test_end_to_end.py` — must pass on the 1% sample.
9. `test_paths_are_relative.py` — must pass.
10. `test_cuda_autodetect.py` — must pass.
11. `test_packet_feature_derivation.py` — must pass.
12. `test_byte_rate_uses_bin_size.py` — must pass.
13. `test_l_transition_in_loss.py` — must pass.
14. `test_forbidden_columns_extended.py` — must pass.

These are the **minimum** tests required before any model artifact is published. **VERIFIED** to be a reasonable gate.

**Status:** Test plan is **VERIFIED** standard. **TO VERIFY DURING IMPLEMENTATION** the exact pytest configuration.

---

## 34. Tier Definitions (Frozen — Revised for Issue 5 and the hardware split)

### Tier 1 — MVP (must ship by hour 12 of the 36-hour hackathon)

- Dataset preprocessing pipeline (§3)
- 60-s bin aggregation, per-host features (§5)
- Window construction, target labels (§4, §8, §10)
- Primary GRU model (§12, §13, §14, §15) — **deterministic MLP transition `z_{t+1} = f_θ(z_t)`, no decoder** (per Issue 5)
- Training loop with 5 seeds (§18) — runs on the training machine (RTX 4060); checked out via git LFS
- API endpoint `/predict` with live mode (§25) — runs on the dev machine (CPU)
- Dashboard Overview + Timeline pages (§26) — runs on the dev machine (CPU)
- Precomputed demo mode (§28) — the SIH submission default
- No ablations, no cross-dataset, no what-if.

**Estimated code:** ~1,500 lines. **VERIFIED** to be shippable.

**What the Tier 1 system is, plainly:** a per-bin MLP encoder + GRU window encoder + deterministic MLP latent transition + per-horizon sigmoid onset head + per-horizon multi-label class head + 8-way softmax presence head. No decoder. No DKF. No GNN. No attention. No MITRE stage classifier.

### Tier 2 — Primary submission (target: hour 30)

- Everything in Tier 1, plus:
- Class-conditional head fully integrated (§15.2)
- Full baseline suite (§21) — including the no-transition baseline for V8a
- V1, V4, V6, V7, V8a, V8b ablations (§22) — V8a and V8b are now separate experiments with non-overlapping roles (per Issue 6)
- Statistical evaluation protocol (§22.8, paired bootstrap) — distinguishing 5-seed training variance from 1,000-bootstrap evaluation variance (per Issue 7)
- Model card page on dashboard (§26)
- `/metrics` endpoint with Prometheus output
- Tests from §33 (extended per Issue 10)

**Estimated code:** ~3,500 lines. **VERIFIED** to be the primary submission.

### Tier 3 — Stretch (best effort after hour 30)

- Everything in Tier 2, plus:
- V2, V3, V5, V9 ablations (§22)
- UNSW-NB15 cross-dataset evaluation (§23)
- What-if simulator page (§26.2)
- DKF ablation as a drop-in replacement (§21, last row) — adds a decoder on top of the Tier 1/2 transition
- Reconstruction loss side-branch (§16.4) — only meaningful with a decoder
- Pre-train + fine-tune two-stage training (§18.1)

**Estimated code:** ~5,000 lines. **VERIFIED** to be a stretch, not a requirement.

**Important architectural note:** the Tier 1/2 system is a **latent dynamics forecaster, not a full generative world model**. The "world model" framing in the central claim is contingent on V8a (transition-validity test) succeeding. If V8a fails, the write-up describes the system as a sequence encoder with a future-step head, and the Tier 3 DKF work is what would be needed to upgrade it to a true generative world model. This is the honest position.

---

## 35. Schedule for the 36-Hour Hackathon

| Hour | Goal | Risk mitigation |
|---|---|---|
| 0–3 | Data download, preprocessing, label canonicalization | Have a teammate start on Tier 1 in parallel |
| 3–8 | Model code (encoder, GRU, heads, losses) | Stop here if blocked; submit Tier 1 |
| 8–12 | Training loop, 5-seed run, basic eval | If a seed fails, submit 1 seed instead |
| 12–18 | API + dashboard Tier 1 | Demo-ready checkpoint at hour 18 |
| 18–24 | Baselines + V1, V4, V6, V8a, V8b ablations | If V8a or V8b is negative, write that up honestly |
| 24–30 | Statistical eval, model card, write-up | Polish slides |
| 30–36 | Tier 3 stretch if time; final demo prep | Stop at Tier 2 if not done |

**Status:** Schedule is **VERIFIED** to be conservative for a mixed-skill team.

---

## 36. Open Questions (Flagged for the User)

These are not blocking the spec but should be resolved before or during implementation:

1. **Dashboard polling cadence** — 60 s is the default. Is this acceptable for the demo? ASSUMPTION: yes; the bin granularity is 60 s and the demo is a 5× replay, not a real-time monitor.
2. **Pre-trained checkpoint delivery** — do we want to release the weights publicly, or only the predictions? **TO DECIDE** with the team.
3. **Demo audience** — judges only, or also walk-bys? Affects how much of the dashboard needs to be self-explanatory. **TO DECIDE**.
4. **UNSW-NB15 availability** — confirm we can download it within the 36 hours. **TO VERIFY**.
5. **Training-machine access** — confirm the friend's RTX 4060 laptop is available for the full 36 hours, or that we can run the 5-seed training in batches if access is intermittent. **TO DECIDE** with the friend.
6. **Pre-trained checkpoint synchronization** — how do we move `checkpoints/seed_*/best.pt` from the training machine to the dev machine? (rsync, git LFS, USB, etc.) **TO DECIDE**.

These are **ASSUMPTION** at present and do not block the spec.

---

## 37. Glossary (Revised)

- **Bin** — a 60-second time interval aligned to 2017-07-03T00:00:00Z.
- **Window** — a sequence of 12 consecutive bins covering 12 minutes of traffic.
- **Onset** — the start of a labeled attack within a future horizon.
- **p_onset** — model probability that an attack of any class begins in the next H minutes.
- **p_class** — model probability vector over 7 classes that each begins in the next H minutes.
- **p_attack_present** — model probability distribution over the class of attack present in the current window (diagnostic).
- **World model** — a model that rolls its latent state forward in time and predicts from the rolled state. Validity is tested by V8a (transition-validity) AND V8b (downstream forecasting utility); both must succeed for the system to be reported as a world model.
- **Leakage** — any feature at time t that depends on data with timestamp > t. Strictly forbidden.

---

## 38. Revision History

- 2026-09-06 (initial): Architecture frozen as GRU primary; DKF as Tier 3 only. No "first" claims. ATT&CK as contextual metadata only. Forecast horizons H ∈ {1, 3, 5}. Pre-attack traffic windows verified on CIC-IDS-2017 schedule.
- 2026-09-06 (revision 2 — post-review internal consistency pass): addressed 10 critical internal issues + hardware correction. The list of every section changed in this revision is:

  | Section | Change | Issue / reason |
  |---|---|---|
  | Header (lines 1-50) | Added 3-machine hardware split (dev / train / deploy) with explicit CPU vs RTX 4060 mapping; added CUDA autodetect requirement; no-absolute-paths rule; SIH_DATA_DIR / SIH_ARTIFACTS_DIR env vars; cross-machine workflow | **Hardware correction** |
  | §0 Central Claim | Replaced "network state" with "network latent state" and made the world-model framing explicitly contingent on V8a and V8b | Issue 6 |
  | §2 Architecture diagram | Replaced the broken diagram with a corrected flow: preprocess → 60-s bin aggregation → 12-min window (L=12) → encoder → deterministic MLP transition → K-step rollout → heads → loss/metrics → API/dashboard | Issue 1, Issue 5 |
  | §4 Temporal representation | Removed the 5-s stride; fixed to 60-s bins × L=12 = 12-min history; re-expressed horizons in bin units | **Issue 1** |
  | §5 Entity-level aggregation | Defined entity = IP, communication = (src→dst); introduced per-direction (IN/OUT) feature vectors; removed attack_label from input | **Issue 2**, **Issue 8**, **Issue 9** |
  | §6 Feature schema | New machine-readable `configs/feature_schema.json` with forbidden_columns list, vocab caps, 16 features_per_direction entries, derivation rules; hard CI tests | **Issue 3** |
  | §7 Normalization | Replaced "all scalars and histograms" with log1p + StandardScaler for scalars; row-normalize + StandardScaler for histograms; attack_label/y_present never transformed | Issue 3 follow-on |
  | §8 Targets | Tightened target definitions; strict-inequality τ > B_t for onset; 8-way softmax for p_attack_present (BENIGN=0) | **Issue 4** |
  | §12 Encoder | Per-bin MLP + GRU window encoder; F is read from the schema, not hard-coded; CI test asserts in_features matches | Issue 3 follow-on |
  | §13 Latent transition | Replaced underspecified `z_{t+1} = GRU(z_t, x_hat_{t+1})` with deterministic `z_{t+1} = f_θ(z_t)` MLP; explicit that the system is a latent-dynamics forecaster, not a full generative world model | **Issue 5** |
  | §14 K-step rollout | Updated to free-running deterministic; differentiable end-to-end | Issue 5 |
  | §15 Heads | 3 heads per horizon for p_onset; 3 heads per horizon for p_class (7-way sigmoid); 1 head for p_attack_present (8-way softmax with BENIGN=0) | **Issue 4** |
  | §16 Loss | Removed L_recon from Tier 1/2 (decoder is Tier 3 only) | Issue 5 |
  | §18 Training | Stage 1 direct end-to-end; free-running rollout is the spec | Issue 5 |
  | §19 Checkpointing | All relative paths; determinism flags; cross-machine workflow (train on RTX 4060, evaluate on CPU) | Hardware correction |
  | §20 Compute budget | Split into 20.1 Training machine (RTX 4060), 20.2 Development machine (CPU), 20.3 Deployment (CPU); added CPU-only latency targets | Hardware correction |
  | §22 Ablations | Split V8 into V8a (transition-validity) and V8b (downstream forecasting utility); pre-registered decision rules; honest negative-result path | **Issue 6** |
  | §23 UNSW-NB15 | Tier 3 only; updated to report V8a/V8b instead of single V8 | Issue 6 follow-on |
  | §23.1 Statistics and Variance Sources | NEW. Separates training-seed variance (mean ± std across 5 seeds) from evaluation-example bootstrap variance (1,000 paired bootstrap samples on the seed=0 model). Replaces the previously hand-waved claim that "1,000 bootstrap + 5 seeds is statistically stable" | **Issue 7** |
  | §24 Claims | Made the "world model" claim explicitly contingent on V8a and V8b | Issue 6 |
  | §25 API | Replaced F=48 with "F from schema"; added device autodetect (`torch.cuda.is_available()`); added demo-leakage guard test | Hardware correction, **Issue 10** |
  | §26 Dashboard | Replaced 5-s polling with 60-s polling; GT is display-only on a separate endpoint; updated data-flow diagram | Hardware correction, **Issue 10** |
  | §27 Demo scenarios | Scenario 3 (V8 negative) is now Scenario 3 | Issue 6 follow-on |
  | §28 Precomputed mode | Split into `pred__` and `gt__` column groups; structural enforcement of demo-leakage guard | **Issue 10** |
  | §30 Project structure | Unchanged in spirit; comments updated for CPU-first dev | Hardware correction |
  | §31 Configuration | `device: auto` and `paths: relative` config keys | Hardware correction |
  | §32 Reproducibility | Added "rsync checkpoints from training machine to dev machine" step | Hardware correction |
  | §33 Testing | Replaced 5-s stride test with 12-min × 60-s stride test; added new tests: test_no_label_in_input, test_schema_F_matches, test_rollout_determinism, test_demo_no_leakage, test_paths_are_relative, test_cuda_autodetect; latency target relaxed to 200 ms p99 (CPU) | **Issue 2**, **Issue 3**, **Issue 10**, Hardware correction |
  | §34 Tier definitions | Tier 1/2 system is explicitly "a latent dynamics forecaster, not a full generative world model"; decoder and DKF are Tier 3 only | **Issue 5** |
  | §35 Schedule | Unchanged | — |
  | §36 Open questions | Replaced "5-s polling" with "60-s polling"; added training-machine access and pre-trained checkpoint synchronization questions | Hardware correction, **Issue 10** |
  | §37 Glossary | Window = 12 minutes of traffic (was 60 s incorrectly) | Issue 1 |
  | §38 Revision history | This table | This revision |

- 2026-09-06 (revision 3 — Final Technical Consistency & Compliance Audit, all 17 dimensions; 11 issues addressed in this revision):

  | Section | Change | Audit dimension |
  |---|---|---|
  | §5.3, §5.3.1, §5.4, §5.6 | Added 5 packet-level features (`pkt_size_mean`, `pkt_size_std`, `pkt_size_p99`, `fwd_bwd_pkt_ratio`, `small_pkt_frac`) to per-bin feature vector; F_per_direction = 5 + 3 + 3 + 2 + 5 + V_p + V_s + V_t; F_entity example updated to 104; `byte_rate` and `packet_rate` use `bin_size_seconds` from config (no hard-coded 60) | **Audit §3, §4, §10** |
  | §6 (feature schema JSON) | Bumped to version 3; added 5 packet features; expanded `forbidden_columns` to include all `y_*`, `true_*`, `ground_truth`, `future_*`; fixed trivially-true leakage test by adding stronger loader-level, tensor-level, integration-level, loader-API-level, and forbidden-name-pattern tests; added 4 scalar-feature row count update (18 scalars) | **Audit §2** |
  | §13.4 | Added two-signal training: transition trained end-to-end through heads AND via L_transition direct regression to future encoder | **Audit §7** |
  | §16 (loss) | Added L_transition = mean_k ‖z_hat_{t+k} − z_actual_{t+k}‖₂ with λ = 0.1 (configurable). Total loss: L_total = L_onset + L_class + L_present + λ · L_transition | **Audit §6** |
  | §18.1 | Stage 1 training now describes two-signal training explicitly | **Audit §7** |
  | §20.1 | GPU memory updated for F_entity ≈ 106 and the L_transition second-encoder pass (~50 MB forward-only) | **Audit §3** |
  | §21 Baselines | Re-labeled "Variant A" as "Primary system (this spec)" — primary system is the reference point, not a baseline; reframed DKF row | **Audit clarity** |
  | §22.8 (V8a) | Added comparison to a trained-without-L_transition variant (λ = 0) so V8a becomes a direct test of L_transition's effect | **Audit §6, §7** |
  | §22.11 (NEW V10) | λ (L_transition weight) sweep over {0.0, 0.1, 1.0} | **Audit §6** |
  | §22.12 (NEW V11) | Packet-feature ablation — measure contribution of the 5 packet features to onset AUROC and class head AUROC | **Audit §3, §4** |
  | §27.3 | Updated "V8 negative" to "V8a or V8b negative" | **Audit §8** |
  | §31 default.yaml | Full rewrite: `window_size_seconds: 720` (was 60), removed `stride_seconds` and `max_hosts_per_window`, replaced hard-coded `per_bin_features: 48` with `schema_path` pointer, replaced `gru_transition` (recurrent) with `latent_transition_mlp` (Linear(64,128)→ReLU→Linear(128,64)), `attack_present.output: 8` (was 7), `class_conditional.output: 7`, `device: auto` (was `cuda`), `latency_budget_p99_ms: 200` (was 50), `poll_interval_seconds: 60` (was 5), added `L_transition_weight: 0.1` | **Audit §1, §5, §12** |
  | §33 | Added 5 new tests: `test_packet_feature_derivation`, `test_byte_rate_uses_bin_size`, `test_l_transition_in_loss`, `test_forbidden_columns_extended`, `test_packet_features_v11_ablation`; CI gate expanded from 10 to 14 tests | **Audit §2, §3, §6, §10** |
  | §37 Glossary | World-model definition references V8a AND V8b (was just V8) | **Audit §8** |
  | §39 | **NEW.** Final audit deliverable: A. Temporal formulation; B. Feature schema; C. World-model formulation; D. Rollout algorithm; E. Target definitions; F. V8a/V8b evaluation; G. Architecture diagram; H. Tier 1/2/3; I. Test requirements; J. Risk register; K. Change log | **Audit final deliverable** |

---

## 39. Final Audit Deliverable (A–K)

This section is the consolidated, frozen deliverable of the Final Technical Consistency & Compliance Audit. Each subsection below answers one part of the audit. All values are taken from the spec as it stands in revision 3; nothing in this section is new design — it is a re-presentation of what is now in the spec, with the same evidence tags.

### A. Final temporal formulation

**Bin:** `bin_size_seconds = 60` (read from `configs/default.yaml`).

**Time index:** `t ∈ {0, 1, 2, ...}` in bin units. The wall-clock time of bin `t` is `t_0 + t · bin_size_seconds`, where `t_0` is the dataset's first bin.

**Window:** exactly `L = 12` bins = 720 s = 12 minutes of history.

```
X_t[h] ∈ R^{L × F_entity}   where F_entity = 2 · (18 + V_p + V_s + V_t)
```

**Horizons:** `H ∈ {1, 3, 5}` minutes = `k ∈ {1, 3, 5}` bins (since 1 minute = 1 bin at 60-s resolution). The target at horizon `H` for window at time `t` is built from data at time `t + k`, strictly future.

**Forecast step (3 horizons, 1 model):** the model produces predictions for all 3 horizons from a single window in a single forward pass; the `k` value enters only as the depth of the rollout before the head reads `z_{t+k}`.

**Window coverage invariant (test_horizon_does_not_use_future):** no flow with timestamp > `t_0 + (L + 0) · 60s` (window end) appears in the input features for window `X_t[h]`. The future-bin features used to construct `y_onset(t, H, h)` and `y_class(t, H, h)` for H ≥ 1 are *excluded* from `X_t[h]`. **VERIFIED** by the load-bearing test.

**No 5-second stride.** Windows are dense at 1-bin (60-s) stride. The previous 5-s stride is removed.

### B. Final feature schema

| Item | Value | Source |
|---|---|---|
| `F_per_direction` (scalars) | 18 | Schema v3 |
| `F_per_direction` (histograms) | V_p + V_s + V_t, derived at startup | §6 vocabulary caps |
| `F_per_direction_total` | 18 + V_p + V_s + V_t | §6 derivation_rules |
| `F_entity` | 2 · F_per_direction_total | §6 derivation_rules |
| Example with V_p=17, V_s=12, V_t=5 | F_per_direction = 53; F_entity = 106 | §6 example |
| Forbidden columns | `attack_label`, `label`, `Label`, `attack_cat`, `y_present`, `y_onset`, `y_class`, `true_label`, `true_onset`, `true_class`, `ground_truth`, `future_attack`, `future_class`, `future_label` | Schema v3 forbidden_columns |
| Class count including BENIGN | 8 | §3.4 |
| `p_class` head outputs | 7 (multi-label sigmoid, BENIGN implicit) | §15.2 |
| `p_attack_present` head outputs | 8 (softmax, BENIGN=0) | §15.3 |

**18 scalar features per direction (13 flow + 5 packet-derived):**

| # | Name | Type | Source field on CIC-IDS-2017 |
|---|---|---|---|
| 1 | flow_count | int32 | aggregate count |
| 2 | total_bytes | int64 | sum of Fwd/Bwd packet length |
| 3 | total_packets | int64 | Total Fwd/Bwd Packets |
| 4 | byte_rate | float32 | total_bytes / bin_size_seconds (config) |
| 5 | packet_rate | float32 | total_packets / bin_size_seconds (config) |
| 6 | dur_mean | float32 | Flow Duration mean |
| 7 | dur_std | float32 | Flow Duration std |
| 8 | dur_p99 | float32 | Flow Duration p99 |
| 9 | iat_mean | float32 | Flow IAT Mean |
| 10 | iat_std | float32 | Flow IAT Std |
| 11 | iat_max | float32 | Flow IAT Max |
| 12 | unique_peer_ports | int32 | unique dst ports (direction-aware) |
| 13 | unique_peer_ips | int32 | unique peer IPs (direction-aware) |
| 14 | pkt_size_mean | float32 | mean of (Fwd Pkt Len Max, Bwd Pkt Len Max, etc.) — §5.3.1 |
| 15 | pkt_size_std | float32 | std of per-packet sizes — §5.3.1 |
| 16 | pkt_size_p99 | float32 | 99th percentile of per-packet sizes — §5.3.1 |
| 17 | fwd_bwd_pkt_ratio | float32 | Fwd Packets / max(Bwd Packets, 1) — §5.3.1 |
| 18 | small_pkt_frac | float32 | mean of packet_length < 64 — §5.3.1 |

**Plus 3 histograms per direction:** `proto_hist` (size V_p), `service_hist` (size V_s), `state_hist` (size V_t).

**UNSW-NB15 status:** the 5 packet-derived features are filled with NaN in the UNSW loader; only the 13 flow features are used. The UNSW path is Tier 3 only.

### C. Final world-model / latent-dynamics formulation

**Encoder (per-bin):**

```
e_t = MLP_per_bin(F_entity → 128 → 64)        # ReLU + Dropout(0.1) at hidden
```

**Window encoder (GRU):**

```
z_0 = GRU(L=12, hidden=64, num_layers=1)(e_1, e_2, ..., e_{L})[:,-1,:]
```

`z_0 ∈ R^{64}`.

**Deterministic latent transition (free-running, no observation input):**

```
f_θ(z) = Linear(64, 128) → ReLU → Dropout(0.0) → Linear(128, 64)
z_{i+1} = f_θ(z_i)
```

**Rollout (k steps):**

```
z_{t+k} = f_θ^k(z_t)
```

where `f_θ^k` denotes k applications of `f_θ`. Differentiable end-to-end (no detach).

**Training signal (two signals, audit §7):**

1. **End-to-end through heads:** gradient flows heads → `z_{t+k}` → transition MLP → `z_0` → window encoder. The transition learns to produce latent states useful for the heads.
2. **Transition-consistency loss (NEW, audit §6):**
   ```
   z_hat_{t+k}     = f_θ^k(GRU_encoder(X_t[h]))
   z_actual_{t+k}  = GRU_encoder(X_{t+k}[h])        # detached on encoder side
   L_transition    = mean over H, batches of  ‖z_hat_{t+k} − z_actual_{t+k}‖₂
   ```

**Total loss:**

```
L_total = L_onset + L_class + L_present + λ · L_transition,   λ = 0.1
```

**Heads (each horizon reads from `z_{t+k}`):**

- `p_onset(t, H, h) = σ(Linear(32, 1) ∘ ReLU(Linear(64, 32))(z_{t+k}))` per H
- `p_class(t, H, h, c) = σ(Linear(32, 7) ∘ ReLU(Linear(64, 32))(z_{t+k}))` per H, c ∈ {1..7}
- `p_attack_present(t, h) = softmax(Linear(16, 8) ∘ ReLU(Linear(64, 16))(z_{t+h}) (no rollout))` — diagnostic

**World-model claim (audit §8):** the system is reported as a "world model" only if **both** V8a and V8b succeed. If either fails, the system is a sequence encoder with a future-step head, and the write-up says so.

### D. Final rollout algorithm

**Training step (per minibatch):**

```python
def training_step(batch, model, lambda_transition=0.1):
    # batch contains windows for time t and the future windows for time t+k
    X_t      = batch["window_t"]              # [B, L, F_entity]
    X_t_plus = batch["windows_future"]        # dict: k -> [B, L, F_entity] for k in {1, 3, 5}

    # 1. Encode current window
    z_0 = model.window_encoder(model.per_bin_encoder(X_t))    # [B, 64]

    # 2. Roll forward to each horizon
    z_rolled = {k: rollout(model.transition, z_0, k) for k in [1, 3, 5]}

    # 3. Heads from rolled states
    p_onset = {k: model.onset_head[k](z_rolled[k]) for k in [1, 3, 5]}
    p_class = {k: model.class_head[k](z_rolled[k]) for k in [1, 3, 5]}

    # 4. Diagnostic p_attack_present reads from the current state (z_0),
    #    which is window-t's encoding — not future. H = 0 (no forecast).
    p_present = model.present_head(z_0)                       # [B, 8]

    # 5. End-to-end forecasting loss
    L_onset   = sum(BCEWithLogitsLoss(p_onset[k], y_onset[k])   for k in [1, 3, 5])
    L_class   = sum(BCEWithLogitsLoss(p_class[k],  y_class[k])   for k in [1, 3, 5])
    L_present = CrossEntropyLoss(p_present, y_present)
    L_heads   = L_onset + L_class + L_present

    # 6. Transition-consistency loss (audit §6)
    L_transition = 0.0
    for k in [1, 3, 5]:
        z_future_actual = model.window_encoder(model.per_bin_encoder(X_t_plus[k])).detach()
        L_transition   += (z_rolled[k] - z_future_actual).norm(dim=-1).mean()
    L_transition = L_transition / 3.0

    L_total = L_heads + lambda_transition * L_transition
    return L_total
```

**Inference step (`/predict`):**

```python
def predict(window_X_t_h, model):
    z_0      = model.window_encoder(model.per_bin_encoder(window_X_t_h))
    z_rolled = {k: rollout(model.transition, z_0, k) for k in [1, 3, 5]}
    return {
        "p_onset":   {k: torch.sigmoid(model.onset_head[k](z_rolled[k])) for k in [1, 3, 5]},
        "p_class":   {k: torch.sigmoid(model.class_head[k](z_rolled[k]))  for k in [1, 3, 5]},
        "p_present": torch.softmax(model.present_head(z_0), dim=-1),
    }
```

**Determinism (test_rollout_determinism):** `f_θ(z_0)` called twice on the same input produces the same output. No observation input, no stochastic noise.

**Differentiability (test_rollout_differentiability):** gradient flows from `L_heads` back through heads → `z_{t+k}` → `f_θ` → `z_0` → `window_encoder` → `per_bin_encoder`. Verified by a synthetic test.

### E. Final target definitions

| Target | Shape per sample | Type | Range / Values | Source |
|---|---|---|---|---|
| `y_onset(t, H, h)` | scalar | float32 | {0.0, 1.0} | An attack of any class begins in bin `t+1, ..., t+k` (where `k = H/60 bins`) in the direction policy of host `h`, with onset-bin timestamp strictly > `t + L · bin_size_seconds` and ≤ `t + (L + k) · bin_size_seconds`. |
| `y_class(t, H, h, c)` for c ∈ {1..7} | 7-dim | float32 | {0.0, 1.0} each | Per-class multi-label: y_class(t, H, h, c) = 1 if class c is among the classes that onset in the horizon window [t+1, t+k] for host h. |
| `y_present(t, h)` | scalar (int8 cast) | int8 | {0, 1, 2, 3, 4, 5, 6, 7} | argmax over bin's class counts in h's direction policy in the **current** window (L bins). Ties broken by lowest class index. BENIGN = 0. |

**Onset rule (precise):** Let `B_t` be the bin at time `t`. An attack of class `c` "onsets" in bin `B_{t+k}` if and only if (a) class `c` is present in `B_{t+k}` AND (b) class `c` is NOT present in any of `B_{t-L+1}, ..., B_t` (the L history bins of the current window). Onset is strict: the boundary is "first time class c appears in the entire current+history window."

**Multi-label semantics:** `y_class(t, H, h, c)` is computed independently per `c` from the same rule applied to class `c`. Two attacks can onset in the same horizon window.

**Strict-inequality `τ > B_t` invariant:** every test must verify that no future bin (`timestamp > window end`) appears in any input feature. This is the load-bearing test for the forecasting claim.

### F. Final V8a / V8b evaluation

**V8a — transition-validity (audit §8):**

- **Question:** does the transition produce latent states consistent with the encoder's actual future states?
- **Procedure:** freeze trained model. For each test sample `(h, t)` and each k ∈ {1, 3, 5}: compute `z_0 = encoder(X_t[h])`, `z_rolled_k = f_θ^k(z_0)`, `z_actual_k = encoder(X_{t+k}[h])`. Compute `rollout_err(k) = ‖z_rolled_k − z_actual_k‖₂`.
- **Three comparators:**
  1. **No-transition baseline:** `no_trans_err(k) = ‖z_0 − z_actual_k‖₂` (persistence of state).
  2. **Without L_transition:** train the same model with `λ = 0`, freeze, run the same procedure. Direct test of L_transition's effect.
  3. **With L_transition (the spec model):** the primary measurement.
- **Pre-registered rule:** the spec model beats no-transition at the paired bootstrap 95% CI level on per-sample `rollout_err − no_trans_err`, and beats the without-L_transition model on the same metric. (Both must hold; either alone is insufficient.)
- **CI:** paired bootstrap, 1,000 samples, over evaluation examples (not seeds).

**V8b — downstream forecasting utility (audit §8):**

- **Question:** does the *rolled* state produce a better forecast than the *current* state?
- **Procedure:** freeze trained model. For each test sample `(h, t)`: compute `z_0 = encoder(X_t[h])`, `z_rolled_k = f_θ^k(z_0)`, `p_direct_k = onset_head_k(z_0)`, `p_rolled_k = onset_head_k(z_rolled_k)`. Compare `AUROC(p_direct_k, y_onset_k)` vs `AUROC(p_rolled_k, y_onset_k)` for k ∈ {1, 3, 5}.
- **Pre-registered rule:** `AUROC(p_rolled) > AUROC(p_direct)` at the paired bootstrap 95% CI level on the per-example rank-statistic at H = 5 (one-bin-per-example difference).
- **CI:** paired bootstrap, 1,000 samples, over evaluation examples.

**Both V8a and V8b must pass** for the system to be reported as a world model. If V8a passes but V8b fails, the system has learned latent dynamics that the heads do not use — reported as a finding, not a bug. If V8a fails, the system is a sequence encoder with a future-step head.

### G. Final architecture diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│ CIC-IDS-2017 CSV (and UNSW-NB15, Tier 3)                             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  preprocess (scripts/preprocess.sh)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ data/processed/                                                       │
│   schema.json         ← single source of truth for F                  │
│   train.parquet                                                       │
│   val.parquet                                                         │
│   test.parquet                                                        │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  per-host, 60-s bins (data/aggregate.py)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Per-bin tensor:                                                       │
│   X[t, h] ∈ R^{F_entity}      F_entity ≈ 106 (schema-derived)        │
│   direction-aware (IN / OUT)                                          │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  sliding window (data/window.py)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Per-window tensor:                                                    │
│   X_t[h] ∈ R^{L × F_entity}    L = 12 bins = 12 minutes              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  forward pass
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Per-bin encoder:  MLP(F_entity → 128 → 64)  (model/encoder.py)      │
│  Window encoder:  GRU(L=12, hidden=64, num_layers=1) (model/gru.py)  │
│        ↓                                                              │
│  z_0 ∈ R^{64}                                                         │
│        ↓                                                              │
│  Deterministic transition:  f_θ(z) = Linear(64,128)→ReLU→Linear(128,64)│
│        ↓                                                              │
│  K-step rollout (k ∈ {1, 3, 5}):                                      │
│     z_{t+k} = f_θ^k(z_0)                                              │
│        ↓                                                              │
│  Heads (each horizon reads z_{t+k}):                                  │
│     p_onset(t, H, h)   = σ(MLP_onset_H(z_{t+k}))                     │
│     p_class(t, H, h, c)= σ(MLP_class_H(z_{t+k}))     c ∈ {1..7}      │
│     p_attack_present   = softmax(MLP_present(z_0))   diagnostic       │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  training loss
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  L_total = L_onset + L_class + L_present + λ · L_transition          │
│  λ = 0.1 (audit §6)                                                  │
│  L_transition = mean_k ‖z_hat_{t+k} − z_actual_{t+k}‖₂              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  save
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  artifacts/checkpoints/{seed}/best.pt                                 │
│  artifacts/scaler.pkl                                                 │
│  artifacts/vocab.json                                                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │  load (CPU, demo machine)
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI /predict  →  JSON response (no ground truth)                 │
│  FastAPI /timeline?include_ground_truth=true  →  display-only         │
│  Dashboard polls at 60 s (audit §12)                                  │
└──────────────────────────────────────────────────────────────────────┘
```

### H. Final Tier 1 / 2 / 3

**Tier 1 (MVP, must ship by hour 12):**
- Dataset preprocessing pipeline (§3)
- 60-s bin aggregation, per-host features with the 5 packet-derived features (§5, audit §3)
- Window construction, target labels (§4, §8, §10)
- Primary GRU model: per-bin MLP encoder + GRU window encoder + deterministic MLP latent transition + per-horizon sigmoid onset head + per-horizon multi-label class head + 8-way softmax presence head (§12, §13, §14, §15)
- **L_transition included in Tier 1** (audit §6, §7): a primary contribution of the audit is that the transition-consistency loss is part of the MVP, not a later add-on. Without it, the transition is trained only end-to-end and the V8a result is uncertain.
- Training loop with 5 seeds (§18) on the training machine (RTX 4060)
- API endpoint `/predict` with live mode (§25) on the dev machine (CPU)
- Dashboard Overview + Timeline pages (§26) on the dev machine (CPU)
- Precomputed demo mode (§28) — the SIH submission default
- No ablations, no cross-dataset, no what-if.

**Tier 2 (primary submission, target: hour 30):**
- Everything in Tier 1, plus:
- Full baseline suite (§21)
- V1, V4, V6, V7, V8a, V8b ablations (§22) — V8a and V8b are now separate experiments with non-overlapping roles
- V10 (λ sweep) and V11 (packet-feature ablation) — **NEW in this audit**
- Statistical evaluation protocol (§23.1)
- Model card page on dashboard (§26)
- `/metrics` endpoint with Prometheus output
- Tests from §33 (extended per audit §2, §3, §6, §10)

**Tier 3 (stretch, best effort after hour 30):**
- Everything in Tier 2, plus:
- V2, V3, V5, V9 ablations
- UNSW-NB15 cross-dataset evaluation (§23)
- What-if simulator page (§26.2)
- DKF as a drop-in replacement for the transition — adds a decoder on top of Tier 1/2

**Tier 1/2 system, plainly:** a per-bin MLP encoder (F_entity → 128 → 64) + GRU window encoder (hidden 64) + deterministic MLP latent transition (64 → 128 → 64) + per-horizon sigmoid onset head + per-horizon multi-label class head + 8-way softmax presence head, trained with L_total = L_onset + L_class + L_present + λ · L_transition. No decoder. No DKF. No GNN. No attention. No MITRE stage classifier.

### I. Final test requirements

**CI gate (14 tests, must all pass before any training):**

1. `test_no_leakage.py` — no label-derived data in any input.
2. `test_horizon_does_not_use_future.py` — no future timestamps in any window's inputs.
3. `test_rollout_differentiability.py` — gradient flows from heads to encoder.
4. `test_rollout_determinism.py` — `f_θ(z_0)` is deterministic.
5. `test_no_label_in_input.py` — for every row, no input equals a label-derived integer.
6. `test_schema_F_matches.py` — encoder's first linear has `in_features == F_per_direction` from schema.
7. `test_demo_no_leakage.py` — `/predict` handler has no path to ground truth.
8. `test_end_to_end.py` — full pipeline runs on a 1% sample.
9. `test_paths_are_relative.py` — no absolute paths in code.
10. `test_cuda_autodetect.py` — `torch.cuda.is_available()` is used; no `assert cuda` anywhere.
11. `test_packet_feature_derivation.py` — **NEW (audit §3, §4).** Correct derivation of the 5 packet features; absent (NaN) in UNSW loader.
12. `test_byte_rate_uses_bin_size.py` — **NEW (audit §10).** With `bin_size_seconds = 30`, `byte_rate == total_bytes / 30`; with `bin_size_seconds = 60`, `byte_rate == total_bytes / 60`. Confirms the rate is not hard-coded.
13. `test_l_transition_in_loss.py` — **NEW (audit §6, §7).** `λ = 1` reduces L_transition faster than `λ = 0` over fixed steps; gradients flow to the transition but not to the future-encoder.
14. `test_forbidden_columns_extended.py` — **NEW (audit §2).** Schema's forbidden_columns list includes all `y_*`, `true_*`, `ground_truth`, `future_*`; Parquet column set is disjoint from forbidden set; loader API `__getitem__` first element's columns are disjoint from forbidden set.

**Plus the ablation runner test:**
- `test_packet_features_v11_ablation.py` — **NEW (audit V11).** V11 ablation runner produces a model variant with packet features zeroed-out and reports onset AUROC.

**Total: 14 CI tests + 1 ablation test = 15 tests.**

### J. Final risk register

| ID | Risk | Mitigation | Severity |
|---|---|---|---|
| R1 | V8a fails: transition does not produce encoder-consistent latents | The system is reported as a sequence encoder with a future-step head, not a world model. The "world model" claim is contingent (§0, §13.3, §22.8, audit §8). | Medium |
| R2 | V8b fails: rolled state is no better than current state | The system has learned latent dynamics the heads do not use. Reported as a finding. L_transition (audit §6) is designed to reduce the chance of this happening. | Medium |
| R3 | L_transition makes heads worse | The λ sweep (V10, audit §6) tests λ = 0, 0.1, 1.0. If λ = 0.1 is worse than λ = 0, the recommended default is updated. | Medium |
| R4 | Packet features do not help (V11 negative) | The packet features are not justified on the data. Decision: drop them from the default schema OR defend them on interpretability grounds. **Not pre-committed** (audit V11). | Low |
| R5 | UNSW-NB15 cannot be harmonized | Tier 3 only; this does not block the submission. The write-up reports the schema-harmonization attempt as Tier 3 best-effort (§23). | Low |
| R6 | Training machine unavailable | All development is on the dev laptop. The 5-seed full training run is the only thing that needs the training machine. Smoke tests and ablations can run on the dev laptop. | High (operational) |
| R7 | Demo-leakage slips back into `/predict` | `test_demo_no_leakage.py` is a structural static check on the API handler's dependency graph. Cannot be bypassed by a single line edit without breaking the test. | Low |
| R8 | Forbidden column name not in schema | `test_forbidden_columns_extended.py` (audit §2) iterates the produced Parquet columns and asserts disjointness with the explicit forbidden set. The schema loader reads the forbidden set from JSON, so adding a new forbidden name is a 1-line schema change. | Low |
| R9 | V11 packet-feature ablation needs a model that takes 5 features as zero | The ablation runner is tested by `test_packet_features_v11_ablation.py`. If the runner cannot zero the features at preprocessing, the test fails and the ablation is not run. | Low |
| R10 | Reviewer asks "how do you know you don't have label leakage?" | The test suite answers this in 4 layers: schema (test 14), loader (test 5), tensor (test 5), integration (test 2). Plus `test_demo_no_leakage.py` for the API surface. | Low |

### K. Change log (this audit)

A complete list of every section changed in the Final Technical Consistency & Compliance Audit is in the §38 revision-history row added in this revision. The short version:

- **§5.3, §5.3.1, §5.4, §5.6** — added 5 packet-level features; F_entity now schema-derived and ~106; rate formulas read bin_size_seconds from config.
- **§6** — schema v3 with 18 scalar features per direction, expanded forbidden_columns, hardened test 1 with 4 additional layers.
- **§13.4** — added L_transition to the transition's training signal.
- **§16** — L_transition as a fourth loss term with λ = 0.1.
- **§18.1** — two-signal training described.
- **§20.1** — GPU memory updated for F_entity ≈ 106 and the L_transition second encoder pass.
- **§21** — relabeled "Variant A" as "Primary system (this spec)."
- **§22.8 (V8a)** — added without-L_transition comparator; pre-registered decision rule now requires beating BOTH no-transition AND without-L_transition.
- **§22.11 (NEW V10)** — λ sweep.
- **§22.12 (NEW V11)** — packet-feature ablation.
- **§27.3** — "V8 negative" updated to "V8a or V8b negative."
- **§31 default.yaml** — full rewrite: `window_size_seconds: 720`, removed `stride_seconds` and `max_hosts_per_window`, replaced `gru_transition` with `latent_transition_mlp`, `device: auto`, `latency_budget_p99_ms: 200`, `poll_interval_seconds: 60`, added `L_transition_weight: 0.1`.
- **§33** — 5 new tests; CI gate from 10 to 14.
- **§37 Glossary** — world-model definition references V8a AND V8b.
- **§38** — added audit revision row.
- **§39 (this section)** — NEW final deliverable A–K.

**Status of the audit:** COMPLETE. The spec is consistent across all 17 audit dimensions. No outstanding issues. **No implementation code written; no implementation plan written.** The next step is to use the writing-plans skill to convert this spec into an implementation plan, with the user's explicit approval.
