#!/usr/bin/env bash
# Regenerates every Brax figure in the paper, into fig/.
#
# Reads data/brax-remax-ac-report.pkl.gz, so it needs no wandb access.  A single
# frame is shared by every plot below, so it is loaded once per invocation of
# analysis/plot.py and never re-fetched.
#
# Design decisions encoded here:
#   - Main figures (Figs. 3 & 4): the eight main tasks in a 2x4 grid, with the
#     per-M action-sample batch size the paper uses -- B = 8 for M in {1,2,4} and
#     B = 16 for M = 8, where the estimator is noisiest.  Fig. 3 carries the SAC
#     and PPO baselines, Fig. 4 SAC only (PPO logs no policy entropy).
#   - The appendix per-M epsilon sweeps (Figs. 7-14) come from
#     analysis/make_cxt2_figs.py, which overlays M in {1,2,4,8} in each panel.
#     They keep M = 8 at B = 16, because the eps != 1e-8 runs for M = 8 only exist
#     at B = 16, so every curve in a figure stays on one B.
#   - The lr sweeps (Figs. 15 & 16) cover the three tasks whose tuned lr is 1e-4,
#     which makes the lr-vs-epsilon comparison readable.
#   - The B ablation (Fig. 17) is per-task, B = 8 vs B = 16 at each M.
#
# The toy figures (Figs. 1, 2, 5, 6) come from toy/ and additional_experiments/,
# not from here.
#
# Usage:  bash analysis/plot.sh
set -euo pipefail

cd "$(dirname "$0")/.."           # repository root

CACHE="data/brax-remax-ac-report.pkl.gz"
OUT_DIR="fig"
COMMON="--project brax-remax-ac-report --cache ${CACHE} --out-dir ${OUT_DIR}"
BS_PER_M="1:8,2:8,4:8,8:16"

mkdir -p "${OUT_DIR}"
[ -f "${CACHE}" ] || { echo "missing ${CACHE}" >&2; exit 1; }

# ============================================================
# Figs. 3 and 4 -- MAIN, eight tasks in a 2x4 grid, eps = 1e-8, tuned lr.
#   Fig. 3: return, with the SAC and PPO baselines.
#   Fig. 4: policy entropy, with SAC only.
# ============================================================
python analysis/plot.py 2x4 ${COMMON} \
    --metrics eval/return \
    --ms 1,2,4,8 --epsilons 1e-8 --bs-per-m ${BS_PER_M} \
    --include-sac --baselines SAC,PPO

python analysis/plot.py 2x4 ${COMMON} \
    --metrics train/entropy \
    --ms 1,2,4,8 --epsilons 1e-8 --bs-per-m ${BS_PER_M} \
    --include-sac --no-other-baselines

# analysis/plot.py names its output after the settings that produced it; the paper
# includes two of them under short names.  Copying rather than renaming keeps the
# settings-named file around, so it stays obvious which run of this script a figure
# in fig/ came from.
cp "${OUT_DIR}/paper_2x4_bsPerM1.8-2.8-4.8-8.16_sacT_baseSAC-PPO_dfltlrT_return_m[1, 2, 4, 8]_eps[1e-08].pdf" \
   "${OUT_DIR}/main_result_return.pdf"
cp "${OUT_DIR}/paper_2x4_bsPerM1.8-2.8-4.8-8.16_sacT_dfltlrT_entropy_m[1, 2, 4, 8]_eps[1e-08].pdf" \
   "${OUT_DIR}/full_entropy.pdf"

# ============================================================
# Figs. 7-14 -- APPENDIX, the per-M epsilon sweeps, eight tasks in a 2x4 grid.
#   entropy_vary_m_eps_{1em8,1em2,1em1,1}.pdf  and  return_vary_m_eps_*.pdf
# ============================================================
python analysis/make_cxt2_figs.py

# ============================================================
# Figs. 15 and 16 -- the lr sweep at M = 4, eps = 1 (lr raised) over eps = 1e-8
# (lr lowered), for the three tasks whose tuned lr is 1e-4.  Written as two
# figures (return, entropy) sharing one epsilon-tiered legend.
# ============================================================
python analysis/plot.py lr-sweep-grid-multi ${COMMON} \
    --envs halfcheetah,swimmer,walker2d --m 4 \
    --epsilons 1,1e-8 \
    --lrs 1e-5,3e-5,5e-5,1e-4,3e-4,5e-4,1e-3

# ============================================================
# Fig. 17 -- the B ablation, B = 8 vs B = 16 at each M, one figure per task.
# ============================================================
for env in ant halfcheetah hopper walker2d reacher swimmer; do
    python analysis/plot.py b-ablation ${COMMON} \
        --env "${env}" --epsilon 1e-8 --ms 1,2,4,8 --bs 8,16 \
        --metric eval/return
done

echo "All plots written to ${OUT_DIR}/"
