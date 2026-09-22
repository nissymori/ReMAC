#!/usr/bin/env bash
# Phase 3 (priority): the actor-epsilon sweep on the FIXED dense Reacher.
#
# The paper sweeps eps in {1e-8, 1e-2, 1e-1, 1} for every M on every main task
# (App. C.2, Figs. 8-15).  Reacher's runs have to be redone because remax_ac was
# silently using episode_length=1000 there.  eps=1e-8 is already covered by
# sh/reacher_final.sh, so this script runs the other three values.
#
# Usage:  bash sh/reacher_eps.sh <GPU_ID> <LR> "<M:EPS> <M:EPS> ..."
set -euo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


GPU=$1
LR=$2
JOBS=$3

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

# Wait until this GPU is free (Phase 2 may still be running on it).
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU")" ]; do
  echo "[GPU $GPU] busy, waiting..."; sleep 60
done

total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=7
wandb_project="brax-remax-reacher-sparse"

for job in $JOBS; do
  m=${job%%:*}
  eps=${job##*:}
  if [ "$m" -eq 8 ]; then B=16; else B=8; fi
  echo "===== dense Reacher  M=$m  eps=$eps  lr=$LR  (GPU $GPU) ====="
  python train.py \
    --config configs/brax/reacher.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id=$seed \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    --set remax_m=$m --set remax_num_samples=$B --set actor_epsilon=$eps \
    --set learning_rate=$LR \
    --wandb --wandb-project=$wandb_project \
    --wandb-run-name "dense-remac-m${m}-eps${eps}"
done
echo "===== GPU $GPU DONE (eps sweep) ====="
