#!/usr/bin/env bash
# The missing epsilon arm of HumanoidStandup.
#
# The appendix plots the M sweep at every eps in {1e-8, 1e-2, 1e-1, 1} for every
# main task.  HumanoidStandup was run at eps in {1e-8, 1e-1, 1} only (see
# sh/humanoid_remac.sh), so its eps = 1e-2 panel is missing; this script fills it.
#
# Protocol matches sh/humanoid_remac.sh in everything that affects training:
# lr = 1e-4 (Tab. 2), B = 16 if M = 8 else 8, 3M steps, eval every 30k, 10 seeds.
# The 10 seeds are 10 processes of ONE vmapped seed each (seed_id 2..11), not
# 5 x 2: HumanoidStandup (d = 17, obs = 376) does not fit 2 vmapped seeds with
# B = 16 on an 11 GB card.  seed_id 0 and 1 are reserved for tuning.
#
# Usage:  bash sh/humanoid_eps.sh <GPU_ID> <EPS> "<M:SEED_ID> <M:SEED_ID> ..."
#   e.g.  bash sh/humanoid_eps.sh 0 1e-2 "1:2 1:3"
#   -> logs/humanoid_eps/remac_m<M>_humanoidstandup_eps<EPS>_seed<SEED_ID>.log
set -uo pipefail   # not -e: one failed run must not abort the sweep

cd "$(dirname "$0")/.."   # brax/
source sh/lib_runlog.sh

GPU=$1
EPS=$2
JOBS=$3

export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

env=humanoidstandup
lr=$(lr_for $env) || exit 1
total_timesteps=3000000
eval_freq=30000
num_seeds=1
RUNDIR=logs/humanoid_eps

for job in $JOBS; do
  m=${job%%:*}
  seed=${job##*:}
  B=$(b_for_m "$m")
  out="$RUNDIR/remac_m${m}_${env}_eps${EPS}_seed${seed}.log"
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> [humanoid-eps] skip (done): $out"; continue
  fi
  echo ">>> [humanoid-eps] m=$m eps=$EPS lr=$lr seed_id=$seed gpu=$GPU -> $out"
  write_config_header "$out" \
    "algo=remax_ac env=$env m=$m b=$B eps=$EPS lr=$lr opt=adam seed_id=$seed num_seeds=$num_seeds"
  python -u train.py --config configs/brax/$env.yaml --algorithm remax_ac \
    --num-seeds=$num_seeds --seed_id="$seed" \
    --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon="$EPS" \
    --set learning_rate=$lr \
    --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
    >> "$out" 2>&1 \
    && echo "===== RUN COMPLETE" >> "$out" \
    || echo "!!! FAILED [humanoid-eps] m=$m eps=$EPS seed_id=$seed (see $out)"
done

echo "=== humanoid_eps.sh done (gpu=$GPU, eps=$EPS, jobs=$JOBS) ==="
