#!/usr/bin/env bash
# Shared helpers for the log-only sweeps (pusher_*.sh, humanoid_eps.sh).
#
# wandb is unavailable on this machine, so every metric is read back from the
# stdout that train.py prints (the same fallback ref/parse_sigma_logs.py already
# uses).  Two things make that parseable:
#
#   * `python -u`, so the log stays current and is not lost if a run is killed;
#   * a `#CONFIG ...` header line written before the run, so the parser does not
#     have to recover the configuration from the file name.
#
# train_log_interval defaults to the eval interval, so train/{entropy,policy_std,
# sigma_grad} and the eval returns are printed at the same steps, ~100 times per
# 3M-step run -- the resolution the "mean of the last 10%" rule needs (App. C.1).

# write_config_header <logfile> <key=value>...
write_config_header () {
  local out="$1"; shift
  mkdir -p "$(dirname "$out")"
  printf '#CONFIG %s\n' "$*" > "$out"
}

# b_for_m <M>  ->  the paper's per-M action-sample batch size (App. C.2).
b_for_m () {
  if [ "$1" -eq 8 ]; then echo 16; else echo 8; fi
}

# lr_for <env>  ->  the tuned learning rate of Tab. 2.
# Set explicitly rather than relying on the config default, so that a run is
# comparable with the reported ones even if a config is edited.  Pusher's value can
# be overridden with $PUSHER_LR, which is how the tuning sweep's own output is fed
# back in before the config is updated (see analysis/pick_pusher_lr.py).
lr_for () {
  case "$1" in
    halfcheetah)     echo 1e-4 ;;
    ant)             echo 2e-4 ;;
    hopper)          echo 3e-4 ;;
    walker2d)        echo 1e-4 ;;
    reacher)         echo 3e-4 ;;
    swimmer)         echo 1e-4 ;;
    humanoidstandup) echo 1e-4 ;;
    pusher)          echo "${PUSHER_LR:-2e-4}" ;;
    *) echo "no tuned learning rate for env '$1'" >&2; return 1 ;;
  esac
}
