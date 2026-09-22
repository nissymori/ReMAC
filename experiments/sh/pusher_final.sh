#!/usr/bin/env bash
# Phase 2 for Pusher: the main-experiment runs at Adam's default eps = 1e-8.
#
# Protocol is the paper's (App. C.1), as in experiments/sh/main_remac.sh and sh/reacher_final.sh:
#   3M steps, eval every 30k, 10 seeds, eps = 1e-8, the re-tuned lr of Phase 1,
#   B = 16 if M = 8 else 8.  SAC and PPO keep their own rejax-tuned lr.
# The 10 seeds are vmapped in one process; Pusher (obs = 23, d = 7) fits them on
# an 11 GB card.  The seed differs from the tuning phase (which used seed_id=1).
#
# Usage:  bash sh/pusher_final.sh <GPU_ID> "<TAG> <TAG> ..."
#   TAG in {sac, ppo, m1, m2, m4, m8}
#   e.g.  bash sh/pusher_final.sh 0 "m1 m2"
#   ReMAC tags need the tuned lr in $PUSHER_LR.
#   -> logs/pusher/final_<TAG>.log
set -uo pipefail   # not -e: one failed run must not abort the sweep

cd "$(dirname "$0")/../../brax"   # every path below is relative to brax/
source "$(dirname "$0")/../lib_runlog.sh"

GPU=$1
TAGS=$2

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

env=pusher
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=7                      # <-- different from the tuning seed (1)
actor_epsilon=1e-8
RUNDIR=logs/pusher

COMMON="--config configs/brax/$env.yaml --num-seeds=$num_seeds --seed_id=$seed \
        --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq"

for TAG in $TAGS; do
  out="$RUNDIR/final_${TAG}.log"
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> [pusher-final] skip (done): $out"; continue
  fi
  case "$TAG" in
    sac|ppo)
      # The baselines keep their own rejax-tuned lr; record it so the log is
      # self-describing (ref/parse_brax_logs.py reads the header, not the config).
      blr=$(python -c "import yaml,sys;print(yaml.safe_load(open('configs/brax/$env.yaml'))['$TAG']['learning_rate'])")
      echo ">>> [pusher-final] $TAG lr=$blr seed_id=$seed gpu=$GPU -> $out"
      write_config_header "$out" \
        "algo=$TAG env=$env lr=$blr seed_id=$seed num_seeds=$num_seeds"
      python -u train.py $COMMON --algorithm "$TAG" >> "$out" 2>&1 \
        && echo "===== RUN COMPLETE" >> "$out" \
        || echo "!!! FAILED [pusher-final] $TAG (see $out)"
      ;;
    m*)
      m=${TAG#m}
      B=$(b_for_m "$m")
      lr=$(lr_for $env) || exit 1
      echo ">>> [pusher-final] remac m=$m B=$B lr=$lr seed_id=$seed gpu=$GPU -> $out"
      write_config_header "$out" \
        "algo=remax_ac env=$env m=$m b=$B eps=$actor_epsilon lr=$lr opt=adam seed_id=$seed num_seeds=$num_seeds"
      python -u train.py $COMMON --algorithm remax_ac \
        --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon=$actor_epsilon \
        --set learning_rate="$lr" >> "$out" 2>&1 \
        && echo "===== RUN COMPLETE" >> "$out" \
        || echo "!!! FAILED [pusher-final] remac m=$m (see $out)"
      ;;
    *) echo "unknown tag: $TAG" >&2 ;;
  esac
done

echo "=== pusher_final.sh done (gpu=$GPU, tags=$TAGS) ==="
