#!/usr/bin/env bash
# ReMAC rebuttal experiment (1)+(2)+(3): ReMAC with a plain-SGD actor.
# Carries the dampening (actor_update_norm vs actor_grad_norm) and
# entropy-via-sigma (policy_std, sigma_grad) diagnostics for free via train/* .
# ReMAC m in {1,2,4} on halfcheetah/ant/hopper.
# wandb project: remac-sgd-report
#
# NOTE: SGD has no epsilon -- `actor_epsilon` is ignored when
# actor_optimizer=sgd. The Adam epsilon sweep exists only as the *comparison*
# side of reviewer point (1) ("comparable to Adam under some choice of
# epsilon?"), and that Adam data was already collected in the `remac-report`
# project. So the Adam sweep is OPT-IN via --adam-sweep and is off by default.
# SAC baselines likewise already exist; opt in with --sac.
set -uo pipefail   # not -e: one failed run must not abort the whole sweep
# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


# --- defaults ---
GPU="0"
M=""            # empty => run m=1,2,4; otherwise a single m
RUN_REMAC=1
RUN_ADAM=0      # Adam eps sweep: already run elsewhere, opt in with --adam-sweep
RUN_SAC=0       # SAC baseline: already run elsewhere, opt in with --sac
SEED_OVERRIDE=""

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)   GPU="$2"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift 1 ;;
    --m)     M="$2"; shift 2 ;;
    --m=*)   M="${1#*=}"; shift 1 ;;
    --seeds)   SEED_OVERRIDE="$2"; shift 2 ;;   # e.g. --seeds "6"
    --seeds=*) SEED_OVERRIDE="${1#*=}"; shift 1 ;;
    --adam-sweep) RUN_ADAM=1; shift 1 ;;
    --sac)     RUN_REMAC=0; RUN_SAC=1; shift 1 ;;
    --no-sac)  RUN_SAC=0; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--gpu N] [--m M] [--seeds \"2 3\"] [--adam-sweep] [--sac]"
      echo "  default: ReMAC+SGD only, m=1,2,4, seed_id 2..6, 3 envs (45 runs)"
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
adam_eps_sweep="1e-8 1e-2 1e-1 1 10 100"
wandb_project="remac-sgd-report"

if [ -n "$SEED_OVERRIDE" ]; then seed_ids="$SEED_OVERRIDE"; fi

run_remac_sgd () {
  local m="$1"
  for seed in $seed_ids; do
    for env in $envs; do
      echo ">>> [sgd] remac SGD m=$m env=$env seed_id=$seed gpu=$GPU"
      python train.py --config configs/brax/$env.yaml --algorithm remax_ac \
        --num-seeds=$num_seeds --seed_id=$seed \
        --set remax_m=$m --set remax_num_samples=$remax_num_samples \
        --set actor_optimizer=sgd \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        --wandb --wandb-project=$wandb_project \
        --wandb-run-name="remac_sgd_m${m}_${env}_seed${seed}" \
        || echo "!!! FAILED [sgd] remac SGD m=$m env=$env seed_id=$seed"
    done
  done
}

run_remac_adam () {
  local m="$1"
  for eps in $adam_eps_sweep; do
    for seed in $seed_ids; do
      for env in $envs; do
        echo ">>> [sgd] remac Adam eps=$eps m=$m env=$env seed_id=$seed gpu=$GPU"
        python train.py --config configs/brax/$env.yaml --algorithm remax_ac \
          --num-seeds=$num_seeds --seed_id=$seed \
          --set remax_m=$m --set remax_num_samples=$remax_num_samples \
          --set actor_optimizer=adam --set actor_epsilon=$eps \
          --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
          --wandb --wandb-project=$wandb_project \
          --wandb-run-name="remac_adam_eps${eps}_m${m}_${env}_seed${seed}" \
          || echo "!!! FAILED [sgd] remac Adam eps=$eps m=$m env=$env seed_id=$seed"
      done
    done
  done
}

run_sac () {
  for seed in $seed_ids; do
    for env in $envs; do
      echo ">>> [sgd] sac env=$env seed_id=$seed gpu=$GPU"
      python train.py --config configs/brax/$env.yaml --algorithm sac \
        --num-seeds=$num_seeds --seed_id=$seed \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
        --wandb --wandb-project=$wandb_project \
        --wandb-run-name="sac_${env}_seed${seed}" \
        || echo "!!! FAILED [sgd] sac env=$env seed_id=$seed"
    done
  done
}

if [ "$RUN_REMAC" -eq 1 ]; then
  if [ -n "$M" ]; then
    run_remac_sgd "$M"
    [ "$RUN_ADAM" -eq 1 ] && run_remac_adam "$M"
  else
    for m in 1 2 4; do
      run_remac_sgd "$m"
      [ "$RUN_ADAM" -eq 1 ] && run_remac_adam "$m"
    done
  fi
fi

if [ "$RUN_SAC" -eq 1 ]; then
  run_sac
fi

echo "=== remac_sgd.sh done (gpu=$GPU) ==="
