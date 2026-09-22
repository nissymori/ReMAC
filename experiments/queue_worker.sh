#!/usr/bin/env bash
# Opportunistic worker: pop one job per line off a queue file and run it, until
# the queue is empty.  One worker per GPU runs in parallel, so a GPU that
# finishes a short job starts the next one immediately instead of idling until
# every other GPU is done.
#
# Every job line is a shell command; $GPU is exported so the job can use it.
#
# Usage:  bash experiments/queue_worker.sh <GPU_ID> <QUEUE_FILE>
set -uo pipefail

cd "$(dirname "$0")/.."   # repository root

GPU=$1
QUEUE=$2
LOCK="${QUEUE}.lock"

export GPU

touch "$LOCK"
while true; do
  job=$(flock "$LOCK" -c "head -1 '$QUEUE'; sed -i '1d' '$QUEUE'")
  [ -z "$job" ] && { echo "[GPU $GPU] queue empty, exiting"; break; }
  echo "===== [GPU $GPU] $job ====="
  eval "$job"
done
echo "===== GPU $GPU DONE ($QUEUE) ====="
