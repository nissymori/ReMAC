#!/usr/bin/env bash
# Extend Tab. 7 (the scale gradient on Brax) to M=8 and to Walker2d.
#
# Tab. 7 currently covers M in {1,2,4} on halfcheetah/ant/hopper, which leaves the
# reviewer's observation about M=8 unmeasured on the scale side.  This script adds
#
#     halfcheetah, ant, hopper : M = 8
#     walker2d                 : M = 1, 2, 4, 8
#
# Protocol matches the existing Tab. 7 runs (remac-coverage-report) in everything that
# affects training: B = 16 for every M, Adam actor with the config default eps = 1e-8,
# the tuned learning rate from the config (Tab. 2), 3M steps, eval every 30k, and
# 5 processes of 2 vmapped seeds = 10 seeds.  The only difference is that the coverage
# rollouts are not run here; they are an eval-time measurement, and the two protocols
# agree within the across-seed spread on the configurations where both exist.
#
# The sigma diagnostics (policy_std, sigma_grad) are logged by
# brax/{remax_ac,train_metrics,train}.py.  wandb is not used here: the metrics are
# read back from stdout by analysis/parse_sigma_logs.py.
#
# Usage:  ./remac_sigma_m8.sh --gpu 0 --jobs "walker2d:1 walker2d:2"
set -uo pipefail   # not -e: one failed run must not abort the sweep

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


GPU="0"
JOBS=""
SEED_OVERRIDE=""
NUM_SEEDS_OVERRIDE=""   # humanoidstandup (d=17, B=16) OOMs at 2 vmapped seeds on 11 GB

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)     GPU="$2"; shift 2 ;;
    --gpu=*)   GPU="${1#*=}"; shift 1 ;;
    --jobs)    JOBS="$2"; shift 2 ;;      # e.g. "walker2d:1 halfcheetah:8"
    --jobs=*)  JOBS="${1#*=}"; shift 1 ;;
    --seeds)   SEED_OVERRIDE="$2"; shift 2 ;;
    --seeds=*) SEED_OVERRIDE="${1#*=}"; shift 1 ;;
    --num-seeds)   NUM_SEEDS_OVERRIDE="$2"; shift 2 ;;
    --num-seeds=*) NUM_SEEDS_OVERRIDE="${1#*=}"; shift 1 ;;
    -h|--help) echo "Usage: $0 --gpu N --jobs \"env:M env:M\" [--seeds \"2 3\"]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$JOBS" ]; then echo "--jobs is required" >&2; exit 1; fi

export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false"
export MPLBACKEND=Agg

total_timesteps=3000000
eval_freq=30000
num_seeds=2                 # num_seeds=10 OOMs on an 11 GB card; 5 x 2 = 10 seeds
seed_ids="2 3 4 5 6"        # seed_id 0,1 were used for tuning
remax_num_samples=16        # B = 16 for every M, as in the existing Tab. 7 runs

# Metrics are read from stdout rather than wandb: `train_log_interval` defaults to the
# eval interval (30080 steps here), so train/{entropy,policy_std,sigma_grad} and the
# eval returns are printed ~100 times per run, which is the resolution the "mean of the
# last 10%" convention needs.  python -u keeps the log current if a run is interrupted.
RUNDIR="logs/sigma_m8/runs"

# Tuned learning rate per environment (Tab. 2).  Set explicitly rather than relying on the
# config default: for reacher (config 5e-4) and humanoidstandup (config 2.04e-4) the default
# differs from the value the paper's results were produced with, so leaving it implicit would
# make the new rows incomparable with the existing ones.  For the other environments the
# config default already equals the Tab. 2 value and this is a no-op.
lr_for () {
  case "$1" in
    halfcheetah)     echo 1e-4 ;;
    ant)             echo 2e-4 ;;
    hopper)          echo 3e-4 ;;
    walker2d)        echo 1e-4 ;;
    reacher)         echo 3e-4 ;;
    swimmer)         echo 1e-4 ;;
    humanoidstandup) echo 1e-4 ;;
    *) echo "no tuned learning rate for env '$1'" >&2; exit 1 ;;
  esac
}

if [ -n "$SEED_OVERRIDE" ]; then seed_ids="$SEED_OVERRIDE"; fi
if [ -n "$NUM_SEEDS_OVERRIDE" ]; then num_seeds="$NUM_SEEDS_OVERRIDE"; fi

mkdir -p "$RUNDIR"

for job in $JOBS; do
  env="${job%%:*}"
  m="${job##*:}"
  lr=$(lr_for "$env") || exit 1
  for seed in $seed_ids; do
    out="$RUNDIR/remac_m${m}_${env}_seed${seed}.log"
    echo ">>> [sigma] remac m=$m env=$env lr=$lr seed_id=$seed gpu=$GPU -> $out"
    python -u train.py --config configs/brax/$env.yaml --algorithm remax_ac \
      --num-seeds=$num_seeds --seed_id=$seed \
      --set remax_m=$m --set remax_num_samples=$remax_num_samples \
      --set learning_rate=$lr \
      --set total_timesteps=$total_timesteps --set eval_freq=$eval_freq \
      > "$out" 2>&1 \
      || echo "!!! FAILED [sigma] remac m=$m env=$env seed_id=$seed (see $out)"
  done
done

echo "=== remac_sigma_m8.sh done (gpu=$GPU, jobs=$JOBS) ==="
