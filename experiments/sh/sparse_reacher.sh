#!/usr/bin/env bash
# Sparse-reward Reacher: SAC vs ReMAC (M = 1, 2, 4, 8).
#
# Reward is replaced by r_t = 1{ d_t < 0.05 } (JaxGCRL convention); physics,
# observations, episode length and every hyperparameter are identical to the
# dense Reacher, so the reward is the only difference.
#
# Usage:  bash sh/sparse_reacher.sh <ALGO_TAG> <GPU_ID>
#   ALGO_TAG in {sac, m1, m2, m4, m8}
set -euo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"
export PYTHONPATH="$(cd ../experiments && pwd)${PYTHONPATH:+:$PYTHONPATH}"


TAG=$1
GPU=$2

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 5 jobs share 4 GPUs
export MPLBACKEND=Agg                        # headless: plt.show() is a no-op

total_timesteps=3000000     # same budget as the paper's main runs
eval_freq=30000
num_seeds=10
seed=1
actor_epsilon=1e-8          # Adam's default, as in the main results
wandb_project="brax-remax-sparse-reacher"

CFG=../experiments/configs/sparse_reacher.yaml
COMMON="--config $CFG --num-seeds=$num_seeds --seed_id=$seed \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        --wandb --wandb-project=$wandb_project"

case "$TAG" in
  sac)
    python train.py $COMMON --algorithm sac --wandb-run-name "sparse_reacher-sac"
    ;;
  m*)
    m=${TAG#m}
    if [ "$m" -eq 8 ]; then B=16; else B=8; fi   # paper's B(M) schedule
    python train.py $COMMON --algorithm remax_ac \
      --set remax_m=$m --set remax_num_samples=$B --set actor_epsilon=$actor_epsilon \
      --wandb-run-name "sparse_reacher-remac-m$m"
    ;;
  *)
    echo "unknown tag: $TAG" >&2; exit 1
    ;;
esac
