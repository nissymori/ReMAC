#!/usr/bin/env bash
# Emit the job list for a phase, one command per line, for experiments/queue_worker.sh.
#
# The three phase scripts take one task and one job list, which keeps them small but
# means the full matrix has to be written out somewhere.  Here is that somewhere, so
# the set of runs behind each figure is one grep away instead of spread over a dozen
# per-task scripts.
#
#   tune    the learning-rate sweep, M x 5 learning rates            -> Tab. 2
#   final   the reported runs at eps = 1e-8, ReMAC + the baselines   -> Figs. 3, 4
#   eps     the epsilon sweep at eps != 1e-8                         -> Figs. 7-14
#   lr      the lr-versus-epsilon probes, M = 4 only                 -> Figs. 15, 16
#   bs      the B ablation, B = 16 at every M                        -> Fig. 17
#
# Usage:  bash experiments/sh/make_queue.sh <PHASE> [ENV ...] > brax/logs/queue/jobs.txt
#   e.g.  bash experiments/sh/make_queue.sh final
#         bash experiments/sh/make_queue.sh eps pusher humanoidstandup
#
# Then, from the repository root, one worker per GPU:
#   for g in 0 1 2 3; do bash experiments/queue_worker.sh $g brax/logs/queue/jobs.txt & done
set -euo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


PHASE=${1:?usage: $0 <tune|final|eps|lr|bs> [ENV ...]}
shift || true

# the eight main tasks, in the order the figures show them
ALL_ENVS="ant halfcheetah hopper walker2d reacher swimmer humanoidstandup pusher"
ENVS=${*:-$ALL_ENVS}

MS="1 2 4 8"
LRS="1e-4 2e-4 3e-4 5e-4 1e-3"
EPSILONS="1e-2 1e-1 1"          # eps = 1e-8 is the `final` phase
# Figs. 15 and 16 compare lr against eps on the three tasks whose tuned lr is 1e-4,
# which is what makes the comparison readable.
LR_PROBE_ENVS="halfcheetah swimmer walker2d"
LR_PROBE="1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3"

for env in $ENVS; do
  case "$PHASE" in
    tune)
      for m in $MS; do for lr in $LRS; do
        echo "bash experiments/sh/tune.sh $env \$GPU \"$m:$lr\""
      done; done
      ;;
    final)
      for tag in m1 m2 m4 m8 sac ppo; do
        echo "bash experiments/sh/final.sh $env \$GPU \"$tag\""
      done
      ;;
    eps)
      for eps in $EPSILONS; do for m in $MS; do
        echo "bash experiments/sh/eps.sh $env \$GPU \"$m:$eps\""
      done; done
      ;;
    lr)
      case " $LR_PROBE_ENVS " in *" $env "*) ;; *) continue ;; esac
      for lr in $LR_PROBE; do
        echo "bash experiments/sh/eps.sh   $env \$GPU \"4:1\" --lr $lr"
        echo "bash experiments/sh/final.sh $env \$GPU \"m4\"  --lr $lr"
      done
      ;;
    bs)
      for tag in m1 m2 m4 m8; do
        echo "bash experiments/sh/final.sh $env \$GPU \"$tag\" --bs 16"
      done
      ;;
    *) echo "unknown phase: $PHASE" >&2; exit 1 ;;
  esac
done
