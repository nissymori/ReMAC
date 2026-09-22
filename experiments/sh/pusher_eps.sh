#!/usr/bin/env bash
# Phase 3 for Pusher: the actor-epsilon sweep.
#
# The appendix plots the M sweep at every eps in {1e-8, 1e-2, 1e-1, 1} for every
# main task.  eps = 1e-8 is covered by sh/pusher_final.sh, so this script runs the
# other three values.  Everything else matches Phase 2: 3M steps, eval every 30k,
# 10 vmapped seeds, the re-tuned lr, B = 16 if M = 8 else 8.
#
# Usage:  bash sh/pusher_eps.sh <GPU_ID> "<M:EPS> <M:EPS> ..."
#   e.g.  bash sh/pusher_eps.sh 0 "1:1e-2 1:1e-1"
#   -> logs/pusher/eps_m<M>_eps<EPS>.log
set -uo pipefail   # not -e: one failed run must not abort the sweep

cd "$(dirname "$0")/../../brax"   # every path below is relative to brax/
source "$(dirname "$0")/../lib_runlog.sh"

GPU=$1
JOBS=$2

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

env=pusher
lr=$(lr_for $env) || exit 1
total_timesteps=3000000
eval_freq=30000
num_seeds=10
seed=7
RUNDIR=logs/pusher

for job in $JOBS; do
  m=${job%%:*}
  eps=${job##*:}
  B=$(b_for_m "$m")
  out="$RUNDIR/eps_m${m}_eps${eps}.log"
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> [pusher-eps] skip (done): $out"; continue
  fi
  echo ">>> [pusher-eps] m=$m eps=$eps B=$B lr=$lr gpu=$GPU -> $out"
  write_config_header "$out" \
    "algo=remax_ac env=$env m=$m b=$B eps=$eps lr=$lr opt=adam seed_id=$seed num_seeds=$num_seeds"
  python -u train.py \
    --config configs/brax/$env.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id=$seed \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon="$eps" \
    --set learning_rate="$lr" >> "$out" 2>&1 \
    && echo "===== RUN COMPLETE" >> "$out" \
    || echo "!!! FAILED [pusher-eps] m=$m eps=$eps (see $out)"
done

echo "=== pusher_eps.sh done (gpu=$GPU, jobs=$JOBS) ==="
