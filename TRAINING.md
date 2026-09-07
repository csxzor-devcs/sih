# Training Guide — SIH 2026 PS 26153 (Network Attack Forecasting)

> **You are about to train a machine learning model on a friend's laptop.**
> This guide assumes you are on a Linux laptop with an NVIDIA RTX 4060 (8 GB VRAM), and that you have never seen this codebase before. Follow the steps in order. **If a step fails, stop and send me (csxzor) the exact error message — do not improvise a fix.**

---

## What you are doing, in one paragraph

A teammate (csxzor) built a learned model that forecasts the onset and class of network attacks from passive traffic observations. The model is a GRU + latent-dynamics architecture; it is evaluated against classical ML and sequence-model baselines. The code is finished, but training the full 5-seed pipeline + 8 ablations on a CPU is too slow. We are using your laptop's RTX 4060 8 GB GPU to run the training in a few hours. At the end you will push the trained checkpoints and metrics back to GitHub for the teammate to use.

---

## 0. Prerequisites (do this once, on the friend's laptop)

Open a terminal. Check that everything is in place:

```bash
# 1. Linux (Ubuntu / Debian / Fedora). If you are on Windows or macOS, stop
#    and tell csxzor — this guide is Linux-only.
uname -a

# 2. NVIDIA GPU + driver. You should see your GPU listed.
nvidia-smi

# 3. Python 3.10 to 3.14. (3.14 is the upper limit; 3.9 or 3.15 will fail.)
python3 --version

# 4. pip and git.
pip3 --version
git --version
```

If any of those commands report "command not found" or a wrong version, install the missing piece before continuing:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential
# NVIDIA driver — only if `nvidia-smi` failed
sudo ubuntu-drivers autoinstall
sudo reboot

# Fedora
sudo dnf install -y python3 python3-pip git make gcc
# NVIDIA driver — only if `nvidia-smi` failed
sudo dnf install -y akmod-nvidia
sudo reboot
```

You also need a CUDA-enabled PyTorch. **Do not `pip install torch` yet** — there is a CUDA-specific step in section 2.

---

## 1. Get the code

```bash
# Clone the repo
cd ~
git clone https://github.com/csxzor-devcs/sih.git
cd sih

# Confirm you are on the right commit
git log --oneline -1
# Expected: 170cbab test(gating): Tier 3 minimum gates (claim-implies-existence)
```

If you are not on commit `170cbab`, stop and tell csxzor. Do not continue on a different commit.

---

## 2. Install Python dependencies

```bash
cd ~/sih

# Create a virtual environment so this project does not pollute system Python
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (matches the RTX 4060). This is
# the only step that needs the special index URL.
pip install "torch>=2.1,<3.0" --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the dependencies from pyproject.toml
pip install -e ".[dev]"
```

**Verify the install** (this is the gate to step 3):

```bash
# 1. PyTorch sees CUDA
python3 -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
# Expected output ends with: device: NVIDIA GeForce RTX 4060 (or similar)

# 2. Tests pass
PYTHONPATH=. pytest tests/ -m "not slow and not gpu" -q
# Expected: 108 passed
```

If `cuda available: False`, the PyTorch install picked up a CPU-only build. Re-run the torch install line above (do not skip the `--index-url`).

If any test fails, stop and send csxzor the test name and error.

---

## 3. Get the dataset (CIC-IDS-2017)

The training pipeline expects the raw CIC-IDS-2017 PCAP-derived CSVs in a specific directory. Download:

1. Go to one of the public mirrors. The Canadian Institute for Cybersecurity hosts the originals:
   - https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the eight per-day CSVs (Monday through Friday). Total ≈ 5 GB compressed.
3. Put them all in `~/sih/data/raw/` so the directory looks like:
   ```
   ~/sih/data/raw/
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-workingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

The exact filenames matter — they are matched in `src/data/aggregate.py`. If your mirror uses slightly different names, rename them to match the pattern above.

**Verify:**

```bash
ls -1 ~/sih/data/raw/ | wc -l
# Expected: 8
```

---

## 4. Preprocess the data (CPU, ~5 min)

```bash
cd ~/sih
source .venv/bin/activate

python -m src.data.preprocess --raw data/raw --out artifacts/processed
```

You should see progress output ending with:

```
[preprocess] wrote NNNNN rows to artifacts/processed/
```

**Verify:**

```bash
ls -lh artifacts/processed/
# Expected: at least one .parquet or .csv file, ≥ 100 MB
```

If this step errors with "Unknown raw label", the CSV column names do not match what the loader expects. Send csxzor the error.

---

## 5. Train the primary model (5 seeds, GPU, ~3-5 hours total)

```bash
cd ~/sih
source .venv/bin/activate

# 30 epochs × 5 seeds. The full 5-seed run is the headline result.
# Each seed takes ~40-60 min on the RTX 4060.

for seed in 0 1 2 3 4; do
  echo "=== seed $seed ==="
  python scripts/train.py \
    --config configs/default.yaml \
    --seed $seed \
    --epochs 30 \
    --device cuda \
    --out artifacts/checkpoints/seed_$seed
done
```

**Important notes:**

- **Keep the laptop plugged in.** A full run is 3-5 hours.
- **Do not close the terminal / SSH session.** If you are SSH'd in, use `tmux` or `screen` so the run survives disconnects:
  ```bash
  tmux new -s training
  # ... run the for-loop above ...
  # Ctrl-b, then d   to detach
  # tmux attach -t training   to come back
  ```
- **Watch the GPU:**
  ```bash
  # In another terminal:
  watch -n 5 nvidia-smi
  ```
  VRAM usage should peak around 2-3 GB. If it climbs above 7 GB, kill the run and tell csxzor.

**Verify after each seed:**

```bash
ls -lh artifacts/checkpoints/seed_0/
# Expected: model.pt (~50 MB), config.json, metrics.json
```

---

## 6. Run evaluation (CPU, ~2 min per seed)

```bash
cd ~/sih
source .venv/bin/activate

for seed in 0 1 2 3 4; do
  python scripts/eval.py \
    --ckpt artifacts/checkpoints/seed_$seed/model.pt \
    --out artifacts/eval/seed_$seed.json
done
```

**Verify:**

```bash
# 5 eval JSONs exist, each contains the metric keys
ls artifacts/eval/
# Expected: seed_0.json seed_1.json seed_2.json seed_3.json seed_4.json

# Inspect one
cat artifacts/eval/seed_0.json
# Expected JSON keys (subset): onset_auroc, class_auroc, present_auroc, brier, ece, plus bootstrap CIs
```

If `onset_auroc` is missing or NaN, something is wrong with the eval. Send csxzor the file.

---

## 7. Run ablations (8 ablations × 1 seed each, GPU, ~1-2 hours total)

```bash
cd ~/sih
source .venv/bin/activate

# The core ablation set. Each trains a fresh model with one knob changed.
for name in V1_per_bin_encoder V4_window_size V6_no_histograms V7_horizons V8a_transition_validity V8b_downstream_utility V10_lambda_sweep V11_packet_features_off; do
  echo "=== ablation $name ==="
  python scripts/ablate.py \
    --name $name \
    --out artifacts/ablations/${name}.json
done
```

This produces 8 JSON files in `artifacts/ablations/`.

If any ablation crashes, that is fine for the headline submission — note which one failed and continue. Send csxzor the error log.

---

## 8. Precompute the demo (CPU, ~1 min)

```bash
cd ~/sih
source .venv/bin/activate

python scripts/precompute.py \
  --ckpt artifacts/checkpoints/seed_0/model.pt \
  --out artifacts/demo/predictions.parquet
```

**Verify:**

```bash
ls -lh artifacts/demo/predictions.parquet
# Expected: ≥ 1 MB

# Confirm the column-group invariant the dashboard depends on
python3 -c "
import pandas as pd
df = pd.read_parquet('artifacts/demo/predictions.parquet')
pred_cols = [c for c in df.columns if c.startswith('pred__')]
gt_cols = [c for c in df.columns if c.startswith('gt__')]
overlap = set(pred_cols) & set(gt_cols)
assert not overlap, f'OVERLAP: {overlap}'
print(f'OK: {len(pred_cols)} pred columns, {len(gt_cols)} gt columns, no overlap')
"
```

---

## 9. Send the artifacts back

The artifacts directory contains everything csxzor needs. We push it to GitHub as a separate branch so the main branch stays clean.

```bash
cd ~/sih

# 1. Create a branch to carry the artifacts
git checkout -b training-results/$(date +%Y%m%d)

# 2. Tell git the artifacts directory is content, not a build artifact.
#    (the .gitignore already excludes it; this is just for documentation)
ls -la .gitignore | head -1

# 3. Force-add the artifacts directory. They are large but the repo
#    has no LFS configured; we accept the size for the hackathon.
git add -f artifacts/
git status --short | head -20

# 4. Commit
git commit -m "training: 5-seed + 8-ablation results from RTX 4060"

# 5. Push (you need write access to csxzor-devcs/sih).
#    If you do not have a GitHub account linked to this repo, ask csxzor
#    to add you as a collaborator (Settings -> Collaborators), or to
#    generate a personal access token and hand it to you. Do NOT
#    hardcode a token in any file.
git push -u origin training-results/$(date +%Y%m%d)

# 6. Report back to csxzor
echo "Pushed branch: training-results/$(date +%Y%m%d)"
echo "Tell csxzor: artifacts are at the tip of this branch"
```

**If `git push` fails with "403 Permission denied":** you do not have push access. Send csxzor:
- The size of `artifacts/` (`du -sh artifacts/`)
- The output of `git log --oneline -1`
- Whether you have a GitHub account

csxzor will either add you as a collaborator or set up a token-based push.

**If `git push` fails with "this exceeds GitHub's file size limit"** (>100 MB for a single file, >5 GB total): tar the artifacts and use a different transfer:

```bash
cd ~/sih
tar czf artifacts.tar.gz artifacts/
ls -lh artifacts.tar.gz
# Send via Google Drive, WeTransfer, or scp to csxzor
```

---

## 10. What to send csxzor in chat

After the push succeeds, send one message with:

1. The branch name: `training-results/YYYYMMDD`
2. The final commit SHA: `git rev-parse HEAD`
3. The size of `artifacts/`: `du -sh artifacts/`
4. The `onset_auroc` number from `artifacts/eval/seed_0.json` (just that one number)
5. Any ablation that failed (or "all 8 ablations passed")
6. Any deviation from this guide

If you hit a step that does not work and you cannot reach csxzor, capture:
- The exact command you ran
- The exact error message (last 30 lines)
- The output of `nvidia-smi` and `python3 --version`

That is enough to debug from a phone.

---

## FAQ

**Q: A test failed in section 2 ("Verify the install"). Which tests should I run?**
A: Run the full suite. `PYTHONPATH=. pytest tests/ -m "not slow and not gpu" -q` and send the FAIL line.

**Q: The preprocess step is slow. Is it working?**
A: Yes, ~5 minutes is normal. If it has been more than 15 minutes with no output, send csxzor the last 10 lines.

**Q: Training crashed with "CUDA out of memory".**
A: Two options:
1. Lower the batch size in `configs/default.yaml` (search for `batch_size`; default is 64, try 32 then 16).
2. Send csxzor the full error; we will lower the model size.

**Q: Can I run the training in the background and close the laptop?**
A: No. The laptop must stay awake, plugged in, and at a reasonable temperature. If you must step away, use `tmux` (see section 5) so the run survives, but the laptop itself must stay open.

**Q: Should I run on battery?**
A: No. Plug in.

**Q: The push was rejected for being too large. What now?**
A: Use the tarball fallback in section 9. Do not try to push to a different branch as a workaround.

**Q: Where is the dashboard?**
A: The Next.js dashboard (`dashboard/`) is meant to run on the dev box, not on this laptop. Skip it. csxzor will run it locally once your artifacts land.

**Q: I do not have a GitHub account. What do I do?**
A: Stop after section 8 and ask csxzor to either:
(a) add you as a collaborator — you create an account at github.com, share the username, and csxzor invites you; or
(b) generate a personal access token and send it to you over a secure channel (Signal, not email).

---

## Cheat sheet (one-line per section)

```bash
# 0. Prereqs
nvidia-smi && python3 --version   # GPU present, Python 3.10-3.14

# 1. Clone
git clone https://github.com/csxzor-devcs/sih.git && cd sih

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install "torch>=2.1,<3.0" --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -m "not slow and not gpu" -q   # 108 passed

# 3. Dataset — download 8 CSVs from https://www.unb.ca/cic/datasets/ids-2017.html
#    into data/raw/

# 4. Preprocess
python -m src.data.preprocess --raw data/raw --out artifacts/processed

# 5. Train (3-5h)
for s in 0 1 2 3 4; do python scripts/train.py --config configs/default.yaml --seed $s --epochs 30 --device cuda --out artifacts/checkpoints/seed_$s; done

# 6. Eval
for s in 0 1 2 3 4; do python scripts/eval.py --ckpt artifacts/checkpoints/seed_$s/model.pt --out artifacts/eval/seed_$s.json; done

# 7. Ablations (1-2h)
for n in V1_per_bin_encoder V4_window_size V6_no_histograms V7_horizons V8a_transition_validity V8b_downstream_utility V10_lambda_sweep V11_packet_features_off; do python scripts/ablate.py --name $n --out artifacts/ablations/${n}.json; done

# 8. Precompute
python scripts/precompute.py --ckpt artifacts/checkpoints/seed_0/model.pt --out artifacts/demo/predictions.parquet

# 9. Push back
git checkout -b training-results/$(date +%Y%m%d)
git add -f artifacts/
git commit -m "training: 5-seed + 8-ablation results from RTX 4060"
git push -u origin training-results/$(date +%Y%m%d)

# 10. Report branch name + commit SHA + onset_auroc + any failures to csxzor
```
