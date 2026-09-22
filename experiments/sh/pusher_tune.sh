#!/usr/bin/env bash
# Phase 1 for Pusher (the eighth main task): tune ReMAC's learning rate.
#
# Protocol is the paper's (App. C.1), identical to sh/tune/tune_remax.sh and
# sh/reacher_tune.sh:
#   lr in {1e-4, 2e-4, 3e-4, 5e-4, 1e-3},  3 seeds,  eps = 1e-8,
#   3M steps,  B = 16 if M = 8 else 8.
#
# Pusher's remax_ac config default lr (2.05e-4) is rejax's SAC value; it is
# overridden here, as every swept value is.
#
# One (M, lr) cell per invocation, so the 20 cells can be spread over the GPUs by
# experiments/queue_worker.sh without two of them appending to the same log.
#
# Usage:  bash sh/pusher_tune.sh <GPU_ID> "<M:LR> <M:LR> ..."
#   e.g.  bash sh/pusher_tune.sh 0 "1:0.0001 1:0.0002"
#   -> logs/pusher/tune_m<M>_lr<LR>.log   (read by ref/pick_pusher_lr.py)
set -uo pipefail   # not -e: one failed cell must not abort the sweep

cd "$(dirname "$0")/../../brax"   # every path below is relative to brax/
source "$(dirname "$0")/../lib_runlog.sh"

GPU=$1
JOBS=$2

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

env=pusher
total_timesteps=3000000
eval_freq=100000            # as in sh/tune/tune_remax.sh
num_seeds=3
seed=1                      # tuning seeds; the final runs use a different seed
actor_epsilon=1e-8
RUNDIR=logs/pusher

for job in $JOBS; do
  m=${job%%:*}
  lr=${job##*:}
  B=$(b_for_m "$m")
  out="$RUNDIR/tune_m${m}_lr${lr}.log"
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> [pusher-tune] skip (done): $out"; continue
  fi
  echo ">>> [pusher-tune] m=$m lr=$lr B=$B gpu=$GPU -> $out"
  write_config_header "$out" \
    "algo=remax_ac env=$env m=$m b=$B eps=$actor_epsilon lr=$lr opt=adam seed_id=$seed num_seeds=$num_seeds"
  python -u train.py \
    --config configs/brax/$env.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id=$seed \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon=$actor_epsilon \
    --set learning_rate="$lr" >> "$out" 2>&1 \
    && echo "===== RUN COMPLETE" >> "$out" \
    || echo "!!! FAILED [pusher-tune] m=$m lr=$lr (see $out)"
done

echo "=== pusher_tune.sh done (gpu=$GPU, jobs=$JOBS) ==="
