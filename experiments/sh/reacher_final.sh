#!/usr/bin/env bash
# Phase 2: dense Reacher and sparse Reacher with the re-tuned learning rate.
#
# Both configs are identical except for the reward:
#   configs/brax/reacher.yaml         r_t = -||fingertip - target|| - ||a||^2
#   ../experiments/configs/sparse_reacher.yaml  r_t = 1{ ||fingertip - target|| < 0.05 }
# episode_length = 50 for every algorithm in both (the bug that made remax_ac
# run 1000-step episodes on Reacher is fixed).
#
# Seeds differ from the tuning phase (tuning used seed_id=1).
#
# Usage:  bash sh/reacher_final.sh <ENV> <ALGO_TAG> <GPU_ID> <LR>
#   ENV      in {dense, sparse}
#   ALGO_TAG in {sac, m1, m2, m4, m8}     (sac ignores LR: it keeps its own tuned lr)
set -euo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"
export PYTHONPATH="$(cd ../experiments && pwd)${PYTHONPATH:+:$PYTHONPATH}"


ENV=$1
TAG=$2
GPU=$3
LR=${4:-}

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=7                      # <-- different from the tuning seed (1)
actor_epsilon=1e-8
wandb_project="brax-remax-reacher-sparse"

case "$ENV" in
  dense)  CFG=configs/brax/reacher.yaml ;;
  sparse) CFG=../experiments/configs/sparse_reacher.yaml ;;
  *) echo "unknown env: $ENV" >&2; exit 1 ;;
esac

COMMON="--config $CFG --num-seeds=$num_seeds --seed_id=$seed \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        --wandb --wandb-project=$wandb_project"

case "$TAG" in
  sac)
    python train.py $COMMON --algorithm sac --wandb-run-name "${ENV}-sac"
    ;;
  m*)
    m=${TAG#m}
    if [ "$m" -eq 8 ]; then B=16; else B=8; fi
    [ -n "$LR" ] || { echo "ReMAC needs a learning rate (arg 4)" >&2; exit 1; }
    python train.py $COMMON --algorithm remax_ac \
      --set remax_m=$m --set remax_num_samples=$B --set actor_epsilon=$actor_epsilon \
      --set learning_rate=$LR \
      --wandb-run-name "${ENV}-remac-m${m}-lr${LR}"
    ;;
  *) echo "unknown tag: $TAG" >&2; exit 1 ;;
esac
