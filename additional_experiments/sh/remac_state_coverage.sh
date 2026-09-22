#!/usr/bin/env bash
# ReMAC rebuttal experiment (4): state-space coverage.
# ReMAC m in {1,2,4} and SAC, on halfcheetah/ant/hopper, logging coverage/* .
# wandb project: remac-coverage-report
set -uo pipefail   # not -e: one failed run must not abort the whole sweep
# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


# --- defaults ---
GPU="0"
M=""            # empty => run m=1,2,4; otherwise a single m
RUN_REMAC=1
RUN_SAC=1
SEED_OVERRIDE=""
ENV_OVERRIDE=""

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   GPU="$2"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift 1 ;;
    --m)     M="$2"; RUN_SAC=0; shift 2 ;;
    --m=*)   M="${1#*=}"; RUN_SAC=0; shift 1 ;;
    --seeds)   SEED_OVERRIDE="$2"; shift 2 ;;   # e.g. --seeds "5 6"
    --seeds=*) SEED_OVERRIDE="${1#*=}"; shift 1 ;;
    --envs)    ENV_OVERRIDE="$2"; shift 2 ;;    # e.g. --envs "ant hopper"
    --envs=*)  ENV_OVERRIDE="${1#*=}"; shift 1 ;;
    --sac)     RUN_REMAC=0; RUN_SAC=1; shift 1 ;;
    --no-sac)  RUN_SAC=0; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--gpu N] [--m M] [--seeds \"5 6\"] [--envs \"ant hopper\"] [--sac] [--no-sac]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --- environment ---
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false"
export MPLBACKEND=Agg

# --- common settings ---
total_timesteps=3000000
eval_freq=30000
num_seeds=2                 # num_seeds=10 OOMs; split into 5 runs of 2 seeds
seed_ids="2 3 4 5 6"        # seed_id 0,1 were used for tuning -> start at 2
remax_num_samples=16
envs="halfcheetah ant hopper"
wandb_project="remac-coverage-report"
cov_flags="--state-coverage --coverage-num-envs 64 --coverage-bins 16 --coverage-proj-dims 2 --coverage-range 3.0"

if [ -n "$SEED_OVERRIDE" ]; then seed_ids="$SEED_OVERRIDE"; fi
if [ -n "$ENV_OVERRIDE" ]; then envs="$ENV_OVERRIDE"; fi

run_remac () {
  local m="$1"
  for seed in $seed_ids; do
    for env in $envs; do
      echo ">>> [coverage] remac m=$m env=$env seed_id=$seed gpu=$GPU"
      python train.py --config configs/brax/$env.yaml --algorithm remax_ac \
        --num-seeds=$num_seeds --seed_id=$seed \
        --set remax_m=$m --set remax_num_samples=$remax_num_samples \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        $cov_flags --wandb --wandb-project=$wandb_project \
        --wandb-run-name="remac_m${m}_${env}_seed${seed}" \
        || echo "!!! FAILED [coverage] remac m=$m env=$env seed_id=$seed"
    done
  done
}

run_sac () {
  for seed in $seed_ids; do
    for env in $envs; do
      echo ">>> [coverage] sac env=$env seed_id=$seed gpu=$GPU"
      python train.py --config configs/brax/$env.yaml --algorithm sac \
        --num-seeds=$num_seeds --seed_id=$seed \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        $cov_flags --wandb --wandb-project=$wandb_project \
        --wandb-run-name="sac_${env}_seed${seed}" \
        || echo "!!! FAILED [coverage] sac env=$env seed_id=$seed"
    done
  done
}

if [ "$RUN_REMAC" -eq 1 ]; then
  if [ -n "$M" ]; then
    run_remac "$M"
  else
    for m in 1 2 4; do run_remac "$m"; done
  fi
fi

if [ "$RUN_SAC" -eq 1 ]; then
  run_sac
fi

echo "=== remac_state_coverage.sh done (gpu=$GPU) ==="
