"""Pick ReMAC's learning rate on Pusher, the eighth main task.

Reads the tuning logs written by brax/sh/pusher_tune.sh and applies the paper's
protocol (App. C.1): sweep lr in {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} with 3 seeds and
"select a value that performed consistently well across environments and M".
Here there is a single environment, so the selection is across M, exactly as in
analysis/pick_reacher_lr.py: rank the learning rates by their *worst* M (so the chosen
lr is not one that is great at one M and broken at another), and break ties by the
mean over M.

Unlike the Reacher tuning, each (M, lr) cell is its own log file (so the 20 cells
could be spread over the four GPUs), and each file carries a `#CONFIG` header, so
M and lr are read from the header rather than from an in-log banner.

The selected value is written to data/PUSHER_LR, which is what the plotting
scripts and brax/sh/lib_runlog.sh read, so the number is never transcribed by hand.

    python analysis/pick_pusher_lr.py            # report + write data/PUSHER_LR
    python analysis/pick_pusher_lr.py --no-write # report only
"""

import argparse
import glob
import os
import re
from collections import defaultdict

LOG_GLOB = "brax/logs/pusher/tune_m*_lr*.log"
OUT_FILE = "data/PUSHER_LR"
MS = [1, 2, 4, 8]
LRS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001]

CONFIG = re.compile(r"^#CONFIG\s+(.*)$")
EVAL = re.compile(
    r"^Seed (\d+) Step (\d+), Mean episode length: [\d.eE+-]+, "
    r"Mean return: (-?[\d.eE+-]+)"
)
DONE = re.compile(r"^===== RUN COMPLETE")


def parse_config(line):
    return dict(kv.split("=", 1) for kv in line.split() if "=" in kv)


def parse_log(path):
    """-> (M, lr, complete, {vmap_seed: [(step, return), ...]})."""
    cfg, complete = None, False
    seeds = defaultdict(list)
    with open(path, errors="ignore") as fh:
        for line in fh:
            if cfg is None and (c := CONFIG.match(line)):
                cfg = parse_config(c.group(1))
            elif (e := EVAL.match(line)):
                seeds[int(e.group(1))].append((int(e.group(2)), float(e.group(3))))
            elif DONE.match(line):
                complete = True
    if cfg is None:
        return None
    return int(cfg["m"]), float(cfg["lr"]), complete, seeds


def summarize(seeds, last_frac):
    """-> (mean final return over seeds, mean over the last `last_frac`, n seeds).

    The primary statistic is the last logged evaluation, which is the rule
    analysis/pick_reacher_lr.py already applies; the trailing-window mean is printed
    alongside it as a robustness check (App. C.1's final-value convention).
    """
    last, window = [], []
    for pts in seeds.values():
        if not pts:
            continue
        pts = sorted(pts)
        last.append(pts[-1][1])
        cut = pts[-1][0] * (1.0 - last_frac)
        tail = [r for s, r in pts if s >= cut]
        window.append(sum(tail) / len(tail))
    if not last:
        return None, None, 0
    return sum(last) / len(last), sum(window) / len(window), len(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=LOG_GLOB)
    ap.add_argument("--last-frac", type=float, default=0.1)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cells, tail_cells, incomplete, finished = {}, {}, [], set()
    for path in sorted(glob.glob(args.glob)):
        parsed = parse_log(path)
        if parsed is None:
            continue
        m, lr, complete, seeds = parsed
        mean_last, mean_tail, n = summarize(seeds, args.last_frac)
        if mean_last is None:
            incomplete.append((m, lr, "no evaluations yet"))
            continue
        if complete:
            finished.add((m, lr))
        else:
            incomplete.append((m, lr, f"still running ({n} seeds, "
                                      f"{max(len(v) for v in seeds.values())} evals)"))
        cells[(m, lr)] = mean_last
        tail_cells[(m, lr)] = mean_tail

    if not cells:
        raise SystemExit(f"no evaluations found in {args.glob} yet")

    for title, table in (("final evaluation", cells),
                         (f"mean over the last {args.last_frac:.0%}", tail_cells)):
        print(f"\n== {title} ==")
        print(f"{'lr':>8s} | " + " ".join(f"{'M=' + str(m):>10s}" for m in MS)
              + " |     worst       mean")
        print("-" * 74)
        for lr in LRS:
            vals = [table.get((m, lr)) for m in MS]
            got = [v for v in vals if v is not None]
            cells_s = " ".join(f"{v:10.2f}" if v is not None else f"{'--':>10s}"
                               for v in vals)
            if len(got) == len(MS):
                print(f"{lr:8.4f} | {cells_s} | {min(got):9.2f} {sum(got)/len(got):10.2f}")
            else:
                print(f"{lr:8.4f} | {cells_s} |  (incomplete: {len(got)}/{len(MS)} M)")

    if incomplete:
        print("\nnot finished yet:")
        for m, lr, why in sorted(incomplete):
            print(f"  M={m} lr={lr:g}: {why}")

    complete_lrs = {lr: (min(v), sum(v) / len(v))
                    for lr in LRS
                    if len(v := [cells[(m, lr)] for m in MS
                                 if (m, lr) in finished]) == len(MS)}
    if len(complete_lrs) < len(LRS):
        print(f"\ntuning still running -- {len(finished)}/{len(MS) * len(LRS)} cells have "
              f"finished ({len(cells)} have partial data); rerun when all "
              f"{len(MS) * len(LRS)} are finished")
        return

    best = max(complete_lrs, key=lambda lr: (complete_lrs[lr][0], complete_lrs[lr][1]))
    print(f"\nselected lr = {best:g}  (best worst-case over M: {complete_lrs[best][0]:.2f}, "
          f"mean over M: {complete_lrs[best][1]:.2f})")

    if not args.no_write:
        os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
        with open(OUT_FILE, "w") as fh:
            fh.write(f"{best:g}\n")
        print(f"wrote {OUT_FILE}")
    print("\nnext:  cd brax && "
          f"PUSHER_LR={best:g} bash sh/pusher_final.sh <gpu> \"m1 m2 m4 m8 sac ppo\"")


if __name__ == "__main__":
    main()
