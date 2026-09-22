#!/usr/bin/env bash
# Phase 2: the reported runs at Adam's default eps = 1e-8, for any task.
#
# 3M steps, eval every 30k, 10 seeds split as seeds_for says for this task, the tuned
# learning rate, and B = 16 if M = 8 else 8.  These are the runs behind Figs. 3 and 4.
# SAC, PPO and TD3 keep their own rejax-tuned learning rate, so no lr is passed for
# them and the header records the one the config carries.
#
# --bs overrides B at every M, which is how the B ablation of Fig. 17 is run.
# --lr overrides the learning rate, which is how the lr probes of Figs. 15 and 16 are.
#
# Usage:  bash experiments/sh/final.sh <ENV> <GPU> "<TAG> ..." [--lr LR] [--bs B]
#   TAG in {m1, m2, m4, m8, sac, ppo, td3}
#   e.g.  bash experiments/sh/final.sh pusher 0 "m1 m2 m4 m8 sac ppo"
#         bash experiments/sh/final.sh ant 0 "m1 m2 m4" --bs 16      # Fig. 17
#   -> brax/logs/<ENV>/final_<TAG>[_b<B>][_lr<LR>].log
set -uo pipefail

cd "$(dirname "$0")/../../brax"
source "$(dirname "$0")/../lib_runlog.sh"

ENV=${1:?usage: $0 <ENV> <GPU> "<TAG> ..." [--lr LR] [--bs B]}
GPU=${2:?usage: $0 <ENV> <GPU> "<TAG> ..." [--lr LR] [--bs B]}
TAGS=${3:?usage: $0 <ENV> <GPU> "<TAG> ..." [--lr LR] [--bs B]}
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
EPS=1e-8
read -r NUM_SEEDS SEED_IDS <<< "$(seeds_for "$ENV")"

# A name that says what differs from the default, so an ablation cannot overwrite the
# run it is being compared against.
suffix=""
[ -n "$BS_OVERRIDE" ] && suffix="${suffix}_b${BS_OVERRIDE}"
[ -n "$LR_OVERRIDE" ] && suffix="${suffix}_lr${LR_OVERRIDE}"

for TAG in $TAGS; do
  for seed in $SEED_IDS; do
    # one log per process; several processes only when the seeds are split
    sfx=$suffix
    [ "$(echo "$SEED_IDS" | wc -w)" -gt 1 ] && sfx="${suffix}_seed${seed}"

    case "$TAG" in
      sac|ppo|td3)
        lr=$(python -c "import yaml;print(yaml.safe_load(open('configs/brax/$ENV.yaml'))['$TAG']['learning_rate'])")
        run_logged "logs/$ENV/final_${TAG}${sfx}.log" \
          "algo=$TAG env=$ENV lr=$lr seed_id=$seed num_seeds=$NUM_SEEDS" \
          -- --config "configs/brax/$ENV.yaml" --algorithm "$TAG" \
             --num-seeds="$NUM_SEEDS" --seed_id="$seed" \
             --set total_timesteps=$TOTAL --set eval_freq=$EVAL_FREQ
        ;;
      m*)
        m=${TAG#m}
        B=${BS_OVERRIDE:-$(b_for_m "$m")}
        lr=${LR_OVERRIDE:-$(lr_for "$ENV")} || exit 1
        run_logged "logs/$ENV/final_${TAG}${sfx}.log" \
          "algo=remax_ac env=$ENV m=$m b=$B eps=$EPS lr=$lr opt=adam seed_id=$seed num_seeds=$NUM_SEEDS" \
          -- --config "configs/brax/$ENV.yaml" --algorithm remax_ac \
             --num-seeds="$NUM_SEEDS" --seed_id="$seed" \
             --set total_timesteps=$TOTAL --set eval_freq=$EVAL_FREQ \
             --set remax_m="$m" --set remax_num_samples="$B" --set actor_epsilon=$EPS \
             --set learning_rate="$lr"
        ;;
      *) echo "unknown tag: $TAG" >&2 ;;
    esac
  done
done
