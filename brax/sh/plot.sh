#!/usr/bin/env bash
# Regenerates the plots from docs/continuous_remax_tmlr2026.pdf using plot.py,
# with the changes requested on 2026-05-22:
#
#   - Main figures (Fig 3 & 4): per-m B selection m in {1,2,4} -> B=8, m=8 -> B=16
#     (the original design), instead of pooling B=8+B=16 as the unfiltered code did.
#   - Baselines: SAC only.  PPO and TD3 are excluded (--no-other-baselines).
#   - NEW: B (=remax_num_samples) ablation, B=8 vs B=16 per m.
#   - NEW: combined lr sweep with eps=1 (lr raised) and eps=1e-8 (lr lowered).
#     Curves are coloured by eps family (eps=1 -> reds, eps=1e-8 -> blues, lr by
#     shade) and the lr legend is split into two eps tiers (eps=1 on top).
#     Produced both per-env (lr-sweep-grid, return+entropy in one figure) and as
#     a 3-env side-by-side figure with a shared legend (lr-sweep-grid-multi),
#     which writes return and entropy as SEPARATE figures.
#   - NEW: a MAIN return figure with SAC + PPO baselines (TD3 still excluded).
#
# Note: the appendix per-M eps sweeps keep m=8 at B=16, because the eps != 1e-8
# runs for m=8 only exist at B=16 (so all four eps stay on the same B within a
# figure).  Only the *main* cross-m figure switches m=8 to B=8.
#
# Toy-experiment figures (PDF Fig 1, 2, 5) are not produced here — they come
# from the toy/ scripts, not from wandb data.
#
# Outputs land in brax/fig/.  A single pickle cache is shared across every plot
# so the wandb fetch only runs once.

set -euo pipefail

cd "$(dirname "$0")/.."   # brax/

PROJECT="brax-remax-ac-report"
CACHE="cache/${PROJECT}.pkl"
OUT_DIR="fig"
COMMON="--project ${PROJECT} --cache ${CACHE} --out-dir ${OUT_DIR}"

mkdir -p "$(dirname "${CACHE}")" "${OUT_DIR}"

# ============================================================
# Paper Fig 3 (return) & Fig 4 (entropy)  -- MAIN
#   6 envs (2x3), m in {1,2,4,8}, eps=1e-8, default lr, SAC only.
#   Per-m B selection (original design): m in {1,2,4} -> B=8, m=8 -> B=16.
# ============================================================
python plot.py 2x3 ${COMMON} \
    --metrics eval/return,train/entropy \
    --ms 1,2,4,8 --epsilons 1e-8 \
    --bs-per-m 1:8,2:8,4:8,8:16 \
    --include-sac --no-other-baselines

# ============================================================
# MAIN return with SAC + PPO baselines (return only, main only).
#   Same ReMAC selection as above; baselines = SAC and PPO (TD3 excluded).
# ============================================================
python plot.py 2x3 ${COMMON} \
    --metrics eval/return \
    --ms 1,2,4,8 --epsilons 1e-8 \
    --bs-per-m 1:8,2:8,4:8,8:16 \
    --include-sac --baselines SAC,PPO

# ============================================================
# Paper Fig 6-9 (entropy) & Fig 10-13 (return) -- APPENDIX
#   per-M eps sweep: 6 envs (2x3), fixed m, eps in {1e-8,1e-2,1e-1,1.0},
#   default lr, SAC only.
#     m in {1,2,4}: B=8.   m=8: B=16 (eps != 1e-8 only exist at B=16).
# ============================================================
for m in 1 2 4; do
    python plot.py 2x3 ${COMMON} \
        --metrics train/entropy,eval/return \
        --ms "${m}" --epsilons 1e-8,1e-2,1e-1,1 \
        --bs 8 \
        --include-sac --no-other-baselines --include-eps-label
done

python plot.py 2x3 ${COMMON} \
    --metrics train/entropy,eval/return \
    --ms 8 --epsilons 1e-8,1e-2,1e-1,1 \
    --bs 16 \
    --include-sac --no-other-baselines --include-eps-label

# ============================================================
# Paper Fig 14
#   lr sweep at m=4, eps=1.0 for halfcheetah / swimmer / walker2d
#   (return + entropy, single env).  Default lr is included so the red
#   reference line (max return / min entropy at default lr, eps=1e-8) lines up.
# ============================================================
for env in halfcheetah swimmer walker2d; do
    python plot.py lr-sweep ${COMMON} \
        --env "${env}" --m 4 --epsilon 1.0 \
        --lrs 1e-4,3e-4,5e-4,1e-3
done

# ============================================================
# NEW (request 5): combined lr sweep, eps=1 (lr raised) on the top row and
#   eps=1e-8 (lr lowered) on the bottom row, return + entropy together.
#   m=4 is the only m with raised-lr runs at eps=1.  Each row only shows the
#   lrs that exist for that eps (eps=1: 1e-4..1e-3; eps=1e-8: 1e-5..1e-4).
# ============================================================
for env in halfcheetah swimmer walker2d; do
    python plot.py lr-sweep-grid ${COMMON} \
        --env "${env}" --m 4 \
        --epsilons 1,1e-8 \
        --lrs 1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3
done

# Combined version: the three envs side by side in one figure with a single
# shared (eps-tiered) legend.  (Per-env figures above are kept too.)
python plot.py lr-sweep-grid-multi ${COMMON} \
    --envs halfcheetah,swimmer,walker2d --m 4 \
    --epsilons 1,1e-8 \
    --lrs 1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3

# ============================================================
# NEW (request 3): B (=remax_num_samples) ablation, B=8 vs B=16 per m.
#   1 row x 4 cols (m in {1,2,4,8}), default lr, eps=1e-8, return + entropy.
# ============================================================
for env in ant halfcheetah hopper walker2d reacher swimmer; do
    python plot.py b-ablation ${COMMON} \
        --env "${env}" --epsilon 1e-8 \
        --ms 1,2,4,8 --bs 8,16 \
        --metric eval/return

    python plot.py b-ablation ${COMMON} \
        --env "${env}" --epsilon 1e-8 \
        --ms 1,2,4,8 --bs 8,16 \
        --metric train/entropy
done

echo "All plots written to ${OUT_DIR}/"
