#!/usr/bin/env bash
# Phase 1: re-tune ReMAC's learning rate on Reacher, now that episode_length is
# fixed to 50 for every algorithm (it was silently 1000 for remax_ac before).
#
# Protocol is the paper's (App. C.1) and sh/tune/tune_remax.sh:
#   lr in {1e-4, 2e-4, 3e-4, 5e-4, 1e-3},  3 seeds,  eps = 1e-8,
#   3M steps,  B = 16 if M = 8 else 8.
#
# One GPU per M, five learning rates run sequentially on it.
#
# Usage:  bash sh/reacher_tune.sh <M> <GPU_ID>
set -euo pipefail

m=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

total_timesteps=3000000
eval_freq=100000            # as in sh/tune/tune_remax.sh
num_seeds=3
seed=1                      # tuning seeds; the final runs use a different seed
actor_epsilon=1e-8
wandb_project="tune-remax-reacher-ep50"

if [ "$m" -eq 8 ]; then B=16; else B=8; fi

for lr in 0.0001 0.0002 0.0003 0.0005 0.001; do
  echo "===== M=$m  lr=$lr  (GPU $GPU) ====="
  python train.py \
    --config configs/brax/reacher.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id=$seed \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    --set remax_m=$m --set remax_num_samples=$B --set actor_epsilon=$actor_epsilon \
    --set learning_rate=$lr \
    --wandb --wandb-project=$wandb_project \
    --wandb-run-name "tune-reacher-m${m}-lr${lr}"
done
echo "===== DONE: M=$m ====="
