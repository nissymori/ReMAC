#!/usr/bin/env bash
# Phase 1: the learning-rate sweep, for any task.
#
# The protocol of App. C.1: sweep lr over {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} with 3 seeds
# at eps = 1e-8, 3M steps, B = 16 if M = 8 else 8, and take the value that does well
# across every M.  analysis/pick_reacher_lr.py and analysis/pick_pusher_lr.py apply
# that selection rule to the logs this writes.
#
# The tuning seeds are seed_id = 1 with 3 vmapped seeds, deliberately disjoint from the
# seeds the reported runs use (see seeds_for in experiments/lib_runlog.sh), so tuning
# cannot leak into a reported number.
#
# Usage:  bash experiments/sh/tune.sh <ENV> <GPU> "<M:LR> <M:LR> ..."
#   e.g.  bash experiments/sh/tune.sh pusher 0 "1:1e-4 1:2e-4"
#   -> brax/logs/<ENV>/tune_m<M>_lr<LR>.log
set -uo pipefail   # not -e: one failed cell must not abort the sweep

cd "$(dirname "$0")/../../brax"
source "$(dirname "$0")/../lib_runlog.sh"

ENV=${1:?usage: $0 <ENV> <GPU> "<M:LR> ..."}
GPU=${2:?usage: $0 <ENV> <GPU> "<M:LR> ..."}
JOBS=${3:?usage: $0 <ENV> <GPU> "<M:LR> ..."}
common_env "$GPU"

TOTAL=3000000
EVAL_FREQ=100000            # coarser than the reported runs: only the final value is read
NUM_SEEDS=3
SEED=1
EPS=1e-8

for job in $JOBS; do
  m=${job%%:*}
  lr=${job##*:}
  B=$(b_for_m "$m")
  run_logged "logs/$ENV/tune_m${m}_lr${lr}.log" \
    "algo=remax_ac env=$ENV m=$m b=$B eps=$EPS lr=$lr opt=adam seed_id=$SEED num_seeds=$NUM_SEEDS" \
    -- --config "configs/brax/$ENV.yaml" --algorithm remax_ac \
       --num-seeds=$NUM_SEEDS --seed_id=$SEED \
       --set total_timesteps=$TOTAL --set eval_freq=$EVAL_FREQ \
       --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon=$EPS \
       --set learning_rate="$lr"
done
