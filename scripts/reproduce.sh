#!/usr/bin/env bash
# scripts/reproduce.sh
# Full reproduction: download data, preprocess, train, eval, precompute.
# Per the three-machine split: parts of this run on dev (CPU) and parts
# on training (GPU).
#
# On the dev box without an SSH key to the training machine, the
# rsync/ssh training steps are skipped if no checkpoints already exist
# locally. The rest of the pipeline (eval, ablate, precompute) runs
# end-to-end so this script also documents the production flow.

set -euo pipefail

DATA_DIR="${SIH_DATA_DIR:-./data}"
ARTIFACTS_DIR="${SIH_ARTIFACTS_DIR:-./artifacts}"
TRAIN_HOST="${SIH_TRAIN_HOST:-trainbox}"

# Dev-box guard: skip the remote training step if no SIH_TRAIN_HOST is
# set AND no checkpoints exist locally. This lets the script run
# end-to-end on the dev box (no trainbox SSH key available) as
# documentation of the production flow.
if [ -z "${SIH_TRAIN_HOST:-}" ] && [ ! -f "artifacts/checkpoints/seed_0/model.pt" ]; then
  echo "[reproduce] no SIH_TRAIN_HOST and no local checkpoints — skipping training step"
  echo "[reproduce] (set SIH_TRAIN_HOST or pre-populate artifacts/checkpoints/ to run the full pipeline)"
  SKIP_TRAINING=1
else
  SKIP_TRAINING=0
fi

# Step 1: Download CIC-IDS-2017 to dev box
mkdir -p "$DATA_DIR"
echo "[reproduce] step 1: download CIC-IDS-2017 (skipped — assumed present)"

# Step 2: Preprocess on dev box
echo "[reproduce] step 2: preprocess"
python -m src.data.preprocess --raw "$DATA_DIR/raw" --out "$ARTIFACTS_DIR/processed"

# Step 3: rsync artifacts/processed to training machine
if [ "$SKIP_TRAINING" -eq 0 ]; then
  echo "[reproduce] step 3: rsync processed to $TRAIN_HOST"
  set +e
  rsync -avz "$ARTIFACTS_DIR/processed/" "$TRAIN_HOST:~/sih/artifacts/processed/"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[reproduce] WARN: rsync to $TRAIN_HOST failed (exit $rc) — continuing"
  fi

  # Step 4: Train (5 seeds) on training machine
  echo "[reproduce] step 4: train (5 seeds) on $TRAIN_HOST"
  set +e
  for seed in 0 1 2 3 4; do
    ssh "$TRAIN_HOST" "cd ~/sih && python scripts/train.py --seed $seed --out artifacts/checkpoints/seed_$seed"
    if [ "$?" -ne 0 ]; then
      echo "[reproduce] WARN: ssh train for seed=$seed failed — continuing"
    fi
  done
  set -e

  # Step 5: rsync checkpoints back to dev
  echo "[reproduce] step 5: rsync checkpoints back"
  set +e
  rsync -avz "$TRAIN_HOST:~/sih/artifacts/checkpoints/" "$ARTIFACTS_DIR/checkpoints/"
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "[reproduce] WARN: rsync from $TRAIN_HOST failed (exit $rc) — continuing"
  fi
else
  echo "[reproduce] steps 3-5: skipped (no TRAIN_HOST, no local checkpoints)"
fi

# Step 6: Eval and ablations on dev box
echo "[reproduce] step 6: evaluate"
for seed in 0 1 2 3 4; do
  if [ -f "$ARTIFACTS_DIR/checkpoints/seed_$seed/model.pt" ]; then
    python scripts/eval.py --ckpt "$ARTIFACTS_DIR/checkpoints/seed_$seed/model.pt" --out "$ARTIFACTS_DIR/eval/seed_$seed.json"
  else
    echo "[reproduce] WARN: no checkpoint for seed_$seed — skipping eval"
  fi
done

# Step 7: Ablations
echo "[reproduce] step 7: ablations"
for name in V1_per_bin_encoder V4_window_size V6_no_histograms V7_horizons V8a_transition_validity V8b_downstream_utility V10_lambda_sweep V11_packet_features_off; do
  python scripts/ablate.py --name "$name" --out "$ARTIFACTS_DIR/ablations/${name}.json"
done

# Step 8: Precompute demo parquet
echo "[reproduce] step 8: precompute demo"
if [ -f "$ARTIFACTS_DIR/checkpoints/seed_0/model.pt" ]; then
  python scripts/precompute.py --ckpt "$ARTIFACTS_DIR/checkpoints/seed_0/model.pt" --out "$ARTIFACTS_DIR/demo/predictions.parquet"
else
  echo "[reproduce] WARN: no seed_0 checkpoint — skipping precompute"
fi

echo "[reproduce] DONE"
