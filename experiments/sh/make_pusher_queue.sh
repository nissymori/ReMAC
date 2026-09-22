#!/usr/bin/env bash
# Emit the Pusher job list for everything after the learning-rate sweep, one command
# per line, for experiments/queue_worker.sh to consume.
#
# Ordered by what it unblocks, because the queue is drained front to back:
#   1. the main runs at eps = 1e-8  -> Figs. 3 and 4, and Tab. 2
#   2. the B = 16 Adam and SGD runs -> Tabs. 7 and 8
#   3. the epsilon sweep            -> Figs. 7-14
#
# The Adam M = 8 cell of Tab. 7 is not listed: it is the same configuration as the
# main M = 8 run (B = 16, eps = 1e-8, tuned lr), so it is read from that run instead
# of being trained twice.
#
# Usage:  bash sh/make_pusher_queue.sh <LR> > logs/queue/pusher.txt
set -euo pipefail

# Run from anywhere: every path below is relative to brax/.
cd "$(dirname "$0")/../../brax"


LR="${1:?usage: $0 <learning rate>   (see analysis/pick_pusher_lr.py)}"

# 1. main runs: ReMAC at each M, plus the SAC and PPO baselines
for tag in m1 m2 m4 m8 sac ppo; do
  echo "PUSHER_LR=$LR bash experiments/sh/pusher_final.sh \$GPU \"$tag\""
done

# 2. the diagnostics tables: B = 16 at every M, Adam and SGD actors
for m in 1 2 4; do
  echo "PUSHER_LR=$LR bash experiments/sh/pusher_sigma_sgd.sh \$GPU \"adam:$m\""
done
for m in 1 2 4; do
  echo "PUSHER_LR=$LR bash experiments/sh/pusher_sigma_sgd.sh \$GPU \"sgd:$m\""
done

# 3. the epsilon sweep (eps = 1e-8 is already covered by the main runs)
for eps in 1e-2 1e-1 1; do
  for m in 1 2 4 8; do
    echo "PUSHER_LR=$LR bash experiments/sh/pusher_eps.sh \$GPU \"$m:$eps\""
  done
done
