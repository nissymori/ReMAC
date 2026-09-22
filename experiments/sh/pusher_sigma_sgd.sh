#!/usr/bin/env bash
# Phase 4 for Pusher: the B = 16 runs that the two diagnostic tables need.
#
#   * Tab. 7 (the scale gradient): the fraction of sigma-increasing updates and
#     the final policy scale, with an Adam actor, B = 16 at every M.
#   * Tab. 8 (Adam actor vs SGD actor): the SGD arm, plus its Adam control --
#     which is the Adam arm below, since it shares env, lr, M, B, seeds and step
#     budget with the SGD arm and differs only in the actor optimizer.
#
# Protocol matches the existing rows of both tables (see
# additional_experiments/sh/remac_sigma_m8.sh and remac_sgd.sh): B = 16 for every M,
# the tuned lr, the config default eps = 1e-8, 3M steps, eval every 30k, 10 seeds.
# The Adam M = 8 cell duplicates sh/pusher_final.sh's m8 run (same config), so it
# is taken from there rather than re-run.
#
# Usage:  bash sh/pusher_sigma_sgd.sh <GPU_ID> "<OPT:M> <OPT:M> ..."
#   OPT in {adam, sgd}
#   e.g.  bash sh/pusher_sigma_sgd.sh 0 "adam:1 sgd:1"
#   -> logs/sigma_m8/runs/remac_m<M>_pusher_seed7.log          (adam; Tab. 7 parser)
#   -> logs/sgd/remac_sgd_m<M>_pusher_seed7.log                (sgd)
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
B=16                        # B = 16 for every M, as in the existing table rows

for job in $JOBS; do
  opt=${job%%:*}
  m=${job##*:}
  case "$opt" in
    adam) out="logs/sigma_m8/runs/remac_m${m}_${env}_seed${seed}.log" ;;
    sgd)  out="logs/sgd/remac_sgd_m${m}_${env}_seed${seed}.log" ;;
    *) echo "unknown optimizer: $opt" >&2; continue ;;
  esac
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> [pusher-$opt] skip (done): $out"; continue
  fi
  echo ">>> [pusher-$opt] m=$m B=$B lr=$lr gpu=$GPU -> $out"
  write_config_header "$out" \
    "algo=remax_ac env=$env m=$m b=$B eps=1e-8 lr=$lr opt=$opt seed_id=$seed num_seeds=$num_seeds"
  python -u train.py \
    --config configs/brax/$env.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id=$seed \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    --set remax_m="$m" --set remax_num_samples=$B \
    --set actor_optimizer="$opt" \
    --set learning_rate="$lr" >> "$out" 2>&1 \
    && echo "===== RUN COMPLETE" >> "$out" \
    || echo "!!! FAILED [pusher-$opt] m=$m (see $out)"
done

echo "=== pusher_sigma_sgd.sh done (gpu=$GPU, jobs=$JOBS) ==="
