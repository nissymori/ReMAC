"""Pick ReMAC's learning rate on the fixed Reacher (episode_length = 50).

Reads the tuning logs written by experiments/sh/tune.sh and applies the paper's
protocol (App. C.1): sweep lr in {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} with 3 seeds and
"select a value that performed consistently well across environments and M".
Here there is a single environment, so the selection is across M: we rank the
learning rates by their *worst* M (so the chosen lr is not one that is great at
one M and broken at another), and break ties by the mean over M.

    python analysis/pick_reacher_lr.py
"""

import glob
import os
import re
from collections import defaultdict

LOG_GLOB = "brax/logs/tune_m*.log"
MS = [1, 2, 4, 8]
LRS = [0.0001, 0.0002, 0.0003, 0.0005, 0.001]

HDR = re.compile(r"^===== M=(\d+)\s+lr=([\d.e-]+)")
EVAL = re.compile(r"^Seed (\d+) Step (\d+), Mean episode length: ([\d.]+), Mean return: (-?[\d.]+)")


def parse():
    """final return per (M, lr): mean over seeds of the last logged eval."""
    runs = defaultdict(lambda: defaultdict(list))  # (M, lr) -> seed -> [(step, ret)]
    for path in sorted(glob.glob(LOG_GLOB)):
        key = None
        for line in open(path, errors="ignore"):
            if (h := HDR.match(line)):
                key = (int(h.group(1)), float(h.group(2)))
            elif key and (e := EVAL.match(line)):
                runs[key][int(e.group(1))].append((int(e.group(2)), float(e.group(4))))
    out = {}
    for key, seeds in runs.items():
        finals = [max(v, key=lambda t: t[0])[1] for v in seeds.values() if v]
        if finals:
            out[key] = (sum(finals) / len(finals), len(finals))
    return out


def main():
    res = parse()
    if not res:
        print(f"no completed evals found in {LOG_GLOB} yet")
        return

    print(f"{'lr':>8s} | " + " ".join(f"{'M=' + str(m):>9s}" for m in MS) + " |    worst      mean")
    print("-" * 68)
    table = {}
    for lr in LRS:
        vals = [res.get((m, lr), (None, 0))[0] for m in MS]
        cells = " ".join(f"{v:9.2f}" if v is not None else f"{'--':>9s}" for v in vals)
        got = [v for v in vals if v is not None]
        if len(got) == len(MS):
            table[lr] = (min(got), sum(got) / len(got))
            print(f"{lr:8.4f} | {cells} | {min(got):8.2f} {sum(got)/len(got):9.2f}")
        else:
            print(f"{lr:8.4f} | {cells} |  (incomplete: {len(got)}/{len(MS)} M done)")

    if len(table) < len(LRS):
        print("\ntuning still running -- rerun when all 20 cells are filled")
        return

    best = max(table, key=lambda lr: (table[lr][0], table[lr][1]))
    print(f"\nselected lr = {best}  (best worst-case over M: {table[best][0]:.2f}, "
          f"mean over M: {table[best][1]:.2f})")
    print(f"\nnext:  bash experiments/sh/final.sh <dense|sparse> <sac|m1|m2|m4|m8> <gpu> {best}")


if __name__ == "__main__":
    main()
