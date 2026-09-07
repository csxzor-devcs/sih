# Training Guide — SIH 2026 PS 26153 (Network Attack Forecasting) — **Windows edition**

> **You are about to train a machine learning model on a Windows laptop with an NVIDIA RTX 4060 (8 GB VRAM).** This guide assumes you are running Windows 10 or 11, opening **PowerShell** (not Command Prompt, not Git Bash), and that you have never seen this codebase before. Follow the steps in order. **If a step fails, stop and send csxzor the exact error message — do not improvise a fix.**

> **Heads up:** this is a *Windows* guide. The Linux equivalent is at `docs/TRAINING-linux.md` if you ever move to a Linux box. Do not mix the two — paths, command syntax, and PyTorch URLs differ.

---

## What you are doing, in one paragraph

A teammate (csxzor) built a learned model that forecasts the onset and class of network attacks from passive traffic observations. The model is a GRU + latent-dynamics architecture; it is evaluated against classical ML and sequence-model baselines. The code is finished, but training the full 5-seed pipeline + 8 ablations on a CPU is too slow. We are using your laptop's RTX 4060 8 GB GPU to run the training in a few hours. At the end you will push the trained checkpoints and metrics back to GitHub for the teammate to use.

---

## 0. Prerequisites (do this once)

Open **PowerShell as Administrator**. Click Start, type `PowerShell`, right-click "Windows PowerShell", choose "Run as administrator". Confirm the User Account Control prompt.

### 0.1 Confirm PowerShell, not Command Prompt

```powershell
$PSVersionTable.PSVersion
# Expected: Major >= 5 (Windows 10 ships 5.1, Windows 11 ships 5.1+ or 7+)
```

### 0.2 Allow script execution (venv activation needs this)

```powershell
# This is the per-user policy. It only affects your account.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# When prompted, type Y and press Enter.
```

If you skip this, every later step that calls `.venv\Scripts\Activate.ps1` will fail with "running scripts is disabled on this system".

### 0.3 Confirm NVIDIA GPU and driver

```powershell
nvidia-smi
# Expected: a table showing your GPU. Driver Version should be >= 525.
# Look for "NVIDIA GeForce RTX 4060" or similar in the top right.
```

If `nvidia-smi` is not found, install the NVIDIA driver from https://www.nvidia.com/drivers (select GeForce RTX 40 series, 4060, Windows 10/11). Reboot when prompted.

### 0.4 Install Python 3.11 (use 3.11 specifically, not 3.12 or 3.13)

The pinned dependency floors in `pyproject.toml` are `python>=3.10,<3.15`. We are picking 3.11 because every wheel we need has a 3.11 build and the RTX 4060 is well-supported.

```powershell
# Option A: using winget (built into Windows 11 / recent Windows 10)
winget install Python.Python.3.11
# Accept the "Add Python to PATH" prompt at the end of the installer.

# Option B: if winget is not available
#   1. Go to https://www.python.org/downloads/release/python-3119/
#   2. Download "Windows installer (64-bit)".
#   3. Run it. **TICK "Add python.exe to PATH"** on the first screen.
#   4. Click "Install Now".
```

**Close PowerShell and open a fresh one** so the new `python` is on PATH. Then verify:

```powershell
python --version
# Expected: Python 3.11.x

pip --version
# Expected: pip 24.x from ...\python311\site-packages\pip
```

### 0.5 Install Git for Windows

```powershell
winget install Git.Git
# Accept the default options. "Git from the command line and also from 3rd-party software".
```

**Close PowerShell and open a fresh one** again. Then verify:

```powershell
git --version
# Expected: git version 2.4x.x
```

---

## 1. Get the code

Open a **regular PowerShell** (no need for admin from here on). Pick a working directory. We use `C:\work` because some Windows file paths get long and `C:\work` is short.

```powershell
mkdir C:\work
cd C:\work

git clone https://github.com/csxzor-devcs/sih.git
cd sih

# Confirm you are on the right commit
git log --oneline -1
# Expected: 5a70c6a docs: training guide for friend's RTX 4060 laptop
```

If you are not on commit `5a70c6a`, stop and tell csxzor. Do not continue on a different commit.

**Configure git so commits on this branch have your name/email** (if not already set):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## 2. Install Python dependencies

```powershell
cd C:\work\sih

# Create a virtual environment so this project does not pollute system Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Your prompt should now start with (.venv). Confirm:
python --version
# Expected: Python 3.11.x, with the venv path

# Upgrade pip inside the venv
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support. This is the only step that
# needs the special index URL. The Windows CUDA 12.1 wheel is at the
# SAME URL as Linux — pytorch.org serves both from the same index.
pip install "torch>=2.1,<3.0" --index-url https://download.pytorch.org/whl/cu121

# Install the rest of the dependencies from pyproject.toml
pip install -e ".[dev]"
```

**Verify the install** (this is the gate to step 3):

```powershell
# 1. PyTorch sees CUDA
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
# Expected output ends with: device: NVIDIA GeForce RTX 4060 (or similar)

# 2. Tests pass
$env:PYTHONPATH = "."
pytest tests/ -m "not slow and not gpu" -q
# Expected: 108 passed
```

If `cuda available: False`, the PyTorch install picked up a CPU-only build. Re-run the torch install line above (do not skip the `--index-url`).

If any test fails, stop and send csxzor the test name and error.

> **PowerShell gotcha:** PowerShell does not inherit `PYTHONPATH=.` from the shell like bash does. The `pytest` line above sets it inline with `$env:PYTHONPATH = "."` for that command. If you open a new PowerShell window, run `$env:PYTHONPATH = "."` again before any `pytest` call.

---

## 3. Get the dataset (CIC-IDS-2017)

The training pipeline expects the raw CIC-IDS-2017 PCAP-derived CSVs in a specific directory. Download:

1. Go to one of the public mirrors. The Canadian Institute for Cybersecurity hosts the originals:
   - https://www.unb.ca/cic/datasets/ids-2017.html
2. Download the eight per-day CSVs (Monday through Friday). Total ≈ 5 GB compressed.
3. Put them all in `C:\work\sih\data\raw\` so the directory looks like:
   ```
   C:\work\sih\data\raw\
   ├── Monday-WorkingHours.pcap_ISCX.csv
   ├── Tuesday-WorkingHours.pcap_ISCX.csv
   ├── Wednesday-workingHours.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

The exact filenames matter — they are matched in `src/data/aggregate.py`. Windows file names are case-insensitive, so the casing shown above is fine; the loader matches case-insensitively.

**Verify:**

```powershell
(Get-ChildItem C:\work\sih\data\raw\*.csv | Measure-Object).Count
# Expected: 8
```

---

## 4. Preprocess the data (CPU, ~5 min)

```powershell
cd C:\work\sih
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

python -m src.data.preprocess --raw data\raw --out artifacts\processed
```

You should see progress output ending with:

```
[preprocess] wrote NNNNN rows to artifacts/processed/
```

**Verify:**

```powershell
Get-ChildItem artifacts\processed\ | Format-Table Name, Length
# Expected: at least one .parquet or .csv file, ≥ 100 MB
```

If this step errors with "Unknown raw label", the CSV column names do not match what the loader expects. Send csxzor the error.

---

## 5. Train the primary model (5 seeds, GPU, ~3-5 hours total)

```powershell
cd C:\work\sih
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

# 30 epochs x 5 seeds. The full 5-seed run is the headline result.
# Each seed takes ~40-60 min on the RTX 4060.
#
# Note the `cmd /c` wrapper: the train script uses POSIX-style
# argparse flags. On Windows we run it via cmd to avoid any PowerShell
# arg-parsing quirks. If you would rather stay in PowerShell, replace
# `cmd /c python ...` with `python ...` — both work.

0..4 | ForEach-Object {
  $seed = $_
  Write-Host "=== seed $seed ===" -ForegroundColor Cyan
  cmd /c "python scripts\train.py --config configs\default.yaml --seed $seed --epochs 30 --device cuda --out artifacts\checkpoints\seed_$seed"
}
```

**Important notes:**

- **Keep the laptop plugged in.** A full run is 3-5 hours.
- **Do not close the PowerShell window.** If you must step away, leave the window open and the laptop on. The for-loop runs sequentially so closing the window stops training.
- **Watch the GPU in another window:**
  ```powershell
  # In a SECOND PowerShell window:
  while ($true) { nvidia-smi; Start-Sleep -Seconds 5; Clear-Host }
  ```
  VRAM usage should peak around 2-3 GB. If it climbs above 7 GB, press Ctrl-C in the training window and tell csxzor.
- **Disable sleep.** Windows will try to sleep the laptop after a few minutes of "inactivity". Run this in the training window before starting:
  ```powershell
  # Prevent the laptop from sleeping while plugged in
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  # To restore later: powercfg /change standby-timeout-ac 15
  ```

**Verify after each seed:**

```powershell
Get-ChildItem artifacts\checkpoints\seed_0\ | Format-Table Name, Length
# Expected: model.pt (~50 MB), config.json, metrics.json
```

---

## 6. Run evaluation (CPU, ~2 min per seed)

```powershell
cd C:\work\sih
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

0..4 | ForEach-Object {
  $seed = $_
  Write-Host "=== eval seed $seed ===" -ForegroundColor Cyan
  cmd /c "python scripts\eval.py --ckpt artifacts\checkpoints\seed_$seed\model.pt --out artifacts\eval\seed_$seed.json"
}
```

**Verify:**

```powershell
# 5 eval JSONs exist
Get-ChildItem artifacts\eval\*.json | Format-Table Name, Length
# Expected: seed_0.json, seed_1.json, ..., seed_4.json

# Inspect one
Get-Content artifacts\eval\seed_0.json
# Expected JSON keys (subset): onset_auroc, class_auroc, present_auroc, brier, ece, plus bootstrap CIs
```

If `onset_auroc` is missing or NaN, something is wrong with the eval. Send csxzor the file.

---

## 7. Run ablations (8 ablations × 1 seed each, GPU, ~1-2 hours total)

```powershell
cd C:\work\sih
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

# The core ablation set. Each trains a fresh model with one knob changed.
$ablations = @(
  "V1_per_bin_encoder",
  "V4_window_size",
  "V6_no_histograms",
  "V7_horizons",
  "V8a_transition_validity",
  "V8b_downstream_utility",
  "V10_lambda_sweep",
  "V11_packet_features_off"
)
foreach ($name in $ablations) {
  Write-Host "=== ablation $name ===" -ForegroundColor Cyan
  cmd /c "python scripts\ablate.py --name $name --out artifacts\ablations\${name}.json"
}
```

This produces 8 JSON files in `artifacts\ablations\`.

If any ablation crashes, that is fine for the headline submission — note which one failed and continue. Send csxzor the error log.

---

## 8. Precompute the demo (CPU, ~1 min)

```powershell
cd C:\work\sih
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

cmd /c "python scripts\precompute.py --ckpt artifacts\checkpoints\seed_0\model.pt --out artifacts\demo\predictions.parquet"
```

**Verify:**

```powershell
Get-ChildItem artifacts\demo\predictions.parquet | Format-Table Name, Length
# Expected: ≥ 1 MB

# Confirm the column-group invariant the dashboard depends on
python -c @'
import pandas as pd
df = pd.read_parquet('artifacts/demo/predictions.parquet')
pred_cols = [c for c in df.columns if c.startswith('pred__')]
gt_cols = [c for c in df.columns if c.startswith('gt__')]
overlap = set(pred_cols) & set(gt_cols)
assert not overlap, f'OVERLAP: {overlap}'
print(f'OK: {len(pred_cols)} pred columns, {len(gt_cols)} gt columns, no overlap')
'@
```

---

## 9. Send the artifacts back

The artifacts directory contains everything csxzor needs. We push it to GitHub as a separate branch so the main branch stays clean.

```powershell
cd C:\work\sih

# 1. Check the size. GitHub rejects pushes > 5 GB total or single files > 100 MB.
$size = (Get-ChildItem -Recurse artifacts\ | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "artifacts size: $([math]::Round($size, 2)) GB"

# 2. Create a branch to carry the artifacts
$stamp = Get-Date -Format "yyyyMMdd"
git checkout -b "training-results/$stamp"

# 3. Force-add the artifacts directory. They are large but the repo
#    has no LFS configured; we accept the size for the hackathon.
git add -f artifacts/
git status --short | Select-Object -First 20

# 4. Commit
git commit -m "training: 5-seed + 8-ablation results from RTX 4060"

# 5. Push (you need write access to csxzor-devcs/sih).
#    If you do not have a GitHub account linked to this repo, ask csxzor
#    to add you as a collaborator (Settings -> Collaborators), or to
#    generate a personal access token and hand it to you. Do NOT
#    hardcode a token in any file.
git push -u origin "training-results/$stamp"

# 6. Report back to csxzor
Write-Host "Pushed branch: training-results/$stamp"
Write-Host "Tell csxzor: artifacts are at the tip of this branch"
```

**If `git push` fails with "403 Permission denied":** you do not have push access. Send csxzor:
- The size of `artifacts\` (the script above prints it in GB)
- The output of `git log --oneline -1`
- Whether you have a GitHub account

csxzor will either add you as a collaborator or set up a token-based push.

**If `git push` fails with "this exceeds GitHub's file size limit"** (>100 MB for a single file, >5 GB total): zip the artifacts and use a different transfer:

```powershell
cd C:\work\sih

# Use 7-Zip if installed; otherwise Windows ships tar in PowerShell 5.1+.
Compress-Archive -Path artifacts\ -DestinationPath artifacts.zip
# Or with tar (PowerShell 5.1+):
tar -czf artifacts.tar.gz artifacts\

Get-ChildItem artifacts.zip, artifacts.tar.gz | Format-Table Name, Length
# Send via Google Drive, WeTransfer, or scp to csxzor
```

---

## 10. What to send csxzor in chat

After the push succeeds, send one message with:

1. The branch name: `training-results/YYYYMMDD`
2. The final commit SHA: `git rev-parse HEAD` (run in the repo)
3. The size of `artifacts\` in GB
4. The `onset_auroc` number from `artifacts\eval\seed_0.json` (just that one number — open the file in Notepad and copy it)
5. Any ablation that failed (or "all 8 ablations passed")
6. Any deviation from this guide

If you hit a step that does not work and you cannot reach csxzor, capture:
- The exact command you ran (copy from PowerShell with `Select-String`)
- The exact error message (last 30 lines)
- The output of `nvidia-smi` and `python --version`

That is enough to debug from a phone.

---

## FAQ

**Q: A test failed in section 2 ("Verify the install").**
A: Run the full suite: `$env:PYTHONPATH = "."; pytest tests/ -m "not slow and not gpu" -q` and send the FAIL line. Do not skip the `PYTHONPATH` set — pytest will not find `src\` without it on Windows.

**Q: The "Activate.ps1" command says "running scripts is disabled on this system".**
A: Go back to section 0.2 and run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. This is the #1 PowerShell gotcha.

**Q: I closed PowerShell by accident. Can I restart?**
A: Yes. Re-open PowerShell, `cd C:\work\sih`, run `.\.venv\Scripts\Activate.ps1`, set `$env:PYTHONPATH = "."`, and resume. The for-loop above runs seeds sequentially, so if you were on seed 2, just edit the loop to start at 2.

**Q: The preprocess step is slow. Is it working?**
A: Yes, ~5 minutes is normal. If it has been more than 15 minutes with no output, send csxzor the last 10 lines.

**Q: Training crashed with "CUDA out of memory".**
A: Two options:
1. Lower the batch size in `configs\default.yaml` (search for `batch_size`; default is 64, try 32 then 16).
2. Send csxzor the full error; we will lower the model size.

**Q: The laptop went to sleep and the training stopped. How do I prevent that?**
A: Run the two `powercfg /change` lines from section 5 before starting training. They disable sleep on AC power only — battery behavior is unchanged.

**Q: Can I run training in the background and use the laptop for other things?**
A: Yes, but keep PowerShell open. You can use a second PowerShell window for other work. Do NOT close the training window. Do NOT run a game or another GPU workload on the same laptop — VRAM is shared.

**Q: The push was rejected for being too large. What now?**
A: Use the zip fallback in section 9. Do not try to push to a different branch as a workaround.

**Q: Where is the dashboard?**
A: The Next.js dashboard (`dashboard\`) is meant to run on the dev box, not on this laptop. Skip it. csxzor will run it locally once your artifacts land.

**Q: I do not have a GitHub account. What do I do?**
A: Stop after section 8 and ask csxzor to either:
(a) add you as a collaborator — you create an account at github.com, share the username, and csxzor invites you; or
(b) generate a personal access token and send it to you over a secure channel (Signal, not email).

**Q: Git for Windows complains about long paths. Help.**
A: This is rare on a fresh `C:\work\sih` path, but if you see "Filename too long":
```powershell
git config --global core.longpaths true
# And enable Windows long-path support (admin PowerShell):
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

**Q: My CSV filenames are slightly different from the list in section 3. Will it work?**
A: Probably not. The loader in `src\data\aggregate.py` matches exact substrings. Rename your files to match the names in section 3 verbatim (case-insensitive is fine on Windows).

---

## Cheat sheet (one-line per section)

```powershell
# 0. Prereqs (admin PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
nvidia-smi                                     # GPU present
python --version                               # Python 3.11.x
git --version                                  # git 2.4x

# 1. Clone
mkdir C:\work; cd C:\work
git clone https://github.com/csxzor-devcs/sih.git; cd sih

# 2. Install
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "torch>=2.1,<3.0" --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
$env:PYTHONPATH = "."
pytest tests/ -m "not slow and not gpu" -q    # 108 passed

# 3. Dataset — download 8 CSVs from https://www.unb.ca/cic/datasets/ids-2017.html
#    into data\raw\

# 4. Preprocess
$env:PYTHONPATH = "."
python -m src.data.preprocess --raw data\raw --out artifacts\processed

# 5. Train (3-5h) — disable sleep first
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
0..4 | ForEach-Object { cmd /c "python scripts\train.py --config configs\default.yaml --seed $_ --epochs 30 --device cuda --out artifacts\checkpoints\seed_$_" }

# 6. Eval
0..4 | ForEach-Object { cmd /c "python scripts\eval.py --ckpt artifacts\checkpoints\seed_$_\model.pt --out artifacts\eval\seed_$_.json" }

# 7. Ablations (1-2h)
"V1_per_bin_encoder","V4_window_size","V6_no_histograms","V7_horizons","V8a_transition_validity","V8b_downstream_utility","V10_lambda_sweep","V11_packet_features_off" | ForEach-Object { cmd /c "python scripts\ablate.py --name $_ --out artifacts\ablations\$_.json" }

# 8. Precompute
cmd /c "python scripts\precompute.py --ckpt artifacts\checkpoints\seed_0\model.pt --out artifacts\demo\predictions.parquet"

# 9. Push back
$stamp = Get-Date -Format "yyyyMMdd"
git checkout -b "training-results/$stamp"
git add -f artifacts/
git commit -m "training: 5-seed + 8-ablation results from RTX 4060"
git push -u origin "training-results/$stamp"

# 10. Report branch name + commit SHA + onset_auroc + any failures to csxzor
```
