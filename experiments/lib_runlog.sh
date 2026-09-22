#!/usr/bin/env bash
# Shared by every run script in experiments/sh/.
#
# wandb is not used: every metric is read back from the stdout that train.py prints,
# by analysis/parse_brax_logs.py.  Two things make that parseable:
#
#   * `python -u`, so the log stays current and is not lost if a run is killed;
#   * a `#CONFIG ...` header written before the run, so the configuration of a run is
#     recoverable from its log alone and the parser does not have to infer it from the
#     file name.
#
# train_log_interval defaults to the eval interval, so train/{entropy,policy_std,
# sigma_grad} and the eval returns are printed at the same steps, ~100 times per
# 3M-step run -- the resolution the "mean of the last 10%" rule needs (App. C.1).
#
# Everything that varies per environment lives in the three lookups below, so the run
# scripts themselves are environment-agnostic.

# ---------------------------------------------------------------------------
# lr_for <env>  ->  the tuned learning rate of Tab. 2.
#
# Set explicitly rather than read from the config, so a run stays comparable with the
# reported ones even if a config is edited.  These are the values the config files also
# carry; analysis/pick_{reacher,pusher}_lr.py is how they were chosen.
lr_for () {
  case "$1" in
    halfcheetah)     echo 1e-4 ;;
    ant)             echo 2e-4 ;;
    hopper)          echo 3e-4 ;;
    walker2d)        echo 1e-4 ;;
    reacher)         echo 3e-4 ;;
    swimmer)         echo 1e-4 ;;
    humanoidstandup) echo 1e-4 ;;
    pusher)          echo 2e-4 ;;
    sparse_reacher)  echo 3e-4 ;;   # the dense Reacher's value; same dynamics
    *) echo "no tuned learning rate for env '$1'" >&2; return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# b_for_m <M>  ->  the action-sample batch size (App. C.2).
#
# B = 8, raised to 16 at the largest retry budget, where the estimator is noisiest
# because M approaches B.
b_for_m () {
  if [ "$1" -eq 8 ]; then echo 16; else echo 8; fi
}

# ---------------------------------------------------------------------------
# seeds_for <env>  ->  "<num_seeds> <seed_id> <seed_id> ..."
#
# A configuration is always 10 seeds.  How they are split across processes depends on
# what fits on an 11 GB card, since the seeds are vmapped:
#
#   humanoidstandup   d = 17, obs = 376   10 processes of 1
#   pusher, reacher   small observations   1 process of 10
#   everything else                        5 processes of 2
#
# A unique seed is (seed_id, vmap index), so every split gives n = 10.  seed_id 0 and 1
# are reserved for hyperparameter tuning and never appear in a reported run, so that
# tuning cannot leak into the results.
seeds_for () {
  case "$1" in
    humanoidstandup) echo "1 2 3 4 5 6 7 8 9 10 11" ;;
    pusher|reacher|sparse_reacher) echo "10 7" ;;
    *) echo "2 2 3 4 5 6" ;;
  esac
}

# ---------------------------------------------------------------------------
# write_config_header <logfile> <key=value>...
write_config_header () {
  local out="$1"; shift
  mkdir -p "$(dirname "$out")"
  printf '#CONFIG %s\n' "$*" > "$out"
}

# ---------------------------------------------------------------------------
# run_logged <logfile> <config-header> -- <train.py args>...
#
# Skips a run that already carries the completion marker, so an interrupted sweep can
# be restarted by rerunning the same command.  With DRY_RUN=1 it prints the command it
# would run and touches nothing, which is what the tests in the README use to check
# that a refactor did not change any invocation.
run_logged () {
  local out="$1"; shift
  local header="$1"; shift
  [ "$1" = "--" ] && shift

  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%s\n' "$out | $header | python -u train.py $*"
    return 0
  fi
  if grep -q "^===== RUN COMPLETE" "$out" 2>/dev/null; then
    echo ">>> skip (done): $out"; return 0
  fi
  echo ">>> $out"
  write_config_header "$out" "$header"
  if python -u train.py "$@" >> "$out" 2>&1; then
    echo "===== RUN COMPLETE" >> "$out"
  else
    echo "!!! FAILED: $out" >&2
  fi
}

# ---------------------------------------------------------------------------
# common_env <GPU>  ->  the environment every run wants.
common_env () {
  export CUDA_VISIBLE_DEVICES="$1"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export MPLBACKEND=Agg
}
