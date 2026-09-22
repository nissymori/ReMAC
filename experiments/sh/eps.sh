#!/usr/bin/env bash
# Phase 3: the Adam-epsilon sweep, for any task.
#
# Everything matches the reported runs of final.sh -- 3M steps, eval every 30k, 10
# seeds split as seeds_for says, the tuned learning rate, B = 16 if M = 8 else 8 --
# except that eps is set per job.  These are the runs behind Figs. 7-14.  eps = 1e-8 is
# already covered by final.sh, so it is not repeated here.
#
# --lr overrides the learning rate, which is how the lr-versus-eps probes of Figs. 15
# and 16 are run; --bs overrides B at every M.
#
# Usage:  bash experiments/sh/eps.sh <ENV> <GPU> "<M:EPS> ..." [--lr LR] [--bs B]
#   e.g.  bash experiments/sh/eps.sh pusher 0 "1:1e-2 2:1e-2"
#         bash experiments/sh/eps.sh halfcheetah 0 "4:1" --lr 1e-3     # Figs. 15, 16
#   -> brax/logs/<ENV>/eps_m<M>_eps<EPS>[_b<B>][_lr<LR>].log
set -uo pipefail

cd "$(dirname "$0")/../../brax"
source "$(dirname "$0")/../lib_runlog.sh"

ENV=${1:?usage: $0 <ENV> <GPU> "<M:EPS> ..." [--lr LR] [--bs B]}
GPU=${2:?usage: $0 <ENV> <GPU> "<M:EPS> ..." [--lr LR] [--bs B]}
JOBS=${3:?usage: $0 <ENV> <GPU> "<M:EPS> ..." [--lr LR] [--bs B]}
shift 3
LR_OVERRIDE=""
BS_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --lr) LR_OVERRIDE="$2"; shift 2 ;;
    --bs) BS_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
common_env "$GPU"

TOTAL=3000000
EVAL_FREQ=30000
read -r NUM_SEEDS SEED_IDS <<< "$(seeds_for "$ENV")"

suffix=""
[ -n "$BS_OVERRIDE" ] && suffix="${suffix}_b${BS_OVERRIDE}"
[ -n "$LR_OVERRIDE" ] && suffix="${suffix}_lr${LR_OVERRIDE}"

for job in $JOBS; do
  m=${job%%:*}
  eps=${job##*:}
  B=${BS_OVERRIDE:-$(b_for_m "$m")}
  lr=${LR_OVERRIDE:-$(lr_for "$ENV")} || exit 1
  for seed in $SEED_IDS; do
    sfx=$suffix
    [ "$(echo "$SEED_IDS" | wc -w)" -gt 1 ] && sfx="${suffix}_seed${seed}"
    run_logged "logs/$ENV/eps_m${m}_eps${eps}${sfx}.log" \
      "algo=remax_ac env=$ENV m=$m b=$B eps=$eps lr=$lr opt=adam seed_id=$seed num_seeds=$NUM_SEEDS" \
      -- --config "configs/brax/$ENV.yaml" --algorithm remax_ac \
         --num-seeds="$NUM_SEEDS" --seed_id="$seed" \
         --set total_timesteps=$TOTAL --set eval_freq=$EVAL_FREQ \
         --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon="$eps" \
         --set learning_rate="$lr"
  done
done
