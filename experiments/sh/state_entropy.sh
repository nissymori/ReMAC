#!/usr/bin/env bash
# Visited-state entropy: at EVERY evaluation, dumps a fixed-size sample of the RAW
# observations visited by the eval policy.  Pooling them over a run gives a uniform-
# in-training-time sample of the states the agent actually visited, which is what the
# metric is about -- it is not a property of the converged policy, so the final-value
# rule of App. C.1 does not apply.  Raw observations are stored so that the k-NN
# entropy can be estimated post hoc with ONE normalization per environment shared by
# every method (agent-specific running statistics would put each agent in a different
# coordinate system).
#   analysis: python ref/knn_state_entropy.py --states-dir <dir> --k 4 --k-check
set -uo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"
export PYTHONPATH="$(cd ../experiments && pwd)${PYTHONPATH:+:$PYTHONPATH}"


GPU="0"
JOBS=""                          # "env:method" pairs, e.g. "ant:sac ant:remac_m1"
OUT="eval_states"
SEED_ID=2
NUM_SEEDS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)  GPU="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out)  OUT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false"
export MPLBACKEND=Agg

total_timesteps=3000000
eval_freq=30000
remax_num_samples=16
wandb_project="remac-state-entropy"
flags="--collect-eval-states --eval-states-dir=$OUT --eval-states-num-envs 64 \
       --eval-states-keep 500 --eval-states-frac 1.0"

for job in $JOBS; do
  env="${job%%:*}"; method="${job#*:}"
  if [ "$method" = "sac" ]; then algo=sac; extra=""; else
    algo=remax_ac; m="${method#remac_m}"
    extra="--set remax_m=$m --set remax_num_samples=$remax_num_samples"
  fi
  echo ">>> [state-entropy] $method env=$env gpu=$GPU"
  python train.py --config configs/brax/$env.yaml --algorithm $algo \
    --num-seeds=$NUM_SEEDS --seed_id=$SEED_ID \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    $extra $flags --wandb --wandb-project=$wandb_project \
    --wandb-run-name="${env}_${method}_g${SEED_ID}" \
    || echo "!!! FAILED [state-entropy] $method env=$env"
done
echo "=== state_entropy.sh done (gpu=$GPU) ==="
