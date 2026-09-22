"""Visited-state entropy from the raw eval states dumped by train.py.

The quantity is the entropy of the states a run actually visited over the whole
of training, not a property of the converged policy: each evaluation contributes
the same number of states, so pooling a run's evaluations gives a uniform-in-
training-time sample of its visited-state distribution.  One k-NN estimate is
therefore computed per seed, from all of that seed's pooled states, and the
final-value rule of App. C.1 does not apply.

Per environment:
  1. Keep only the evaluation steps present for EVERY run, so that every seed of
     every method contributes the same number of states on the same training-time
     grid.
  2. Build ONE reference mean/std from a balanced pool: the same number of states
     is drawn from every (method, seed), so no method or seed dominates it.
     Dimensions whose pooled variance is numerically zero are dropped for all.
  3. Standardize every seed's pooled states with that single reference mean/std.
  4. Estimate the differential entropy with the MEPOL / RE3 k-NN estimator (k=4),
     then report mean +/- s.d. over seeds.

Entropies are comparable only within an environment (N and d are fixed there).

Usage:
    python additional_experiments/knn_state_entropy.py --states-dir <dir> [--k 4] [--k-check]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "code", "ReMAC", "brax"))
from state_entropy import knn_entropy  # noqa: E402

ZERO_VAR_TOL = 1e-8
POOL_PER_SEED = 2000  # states each (method, seed) contributes to the reference stats
SUBSAMPLE_SEED = 0

# <env>_<method>_g<seed_id>_seed<idx>_step<t>.npz ; a seed is (seed_id, vmap index).
FNAME = re.compile(
    r"(?P<env>[a-z0-9]+)_(?P<method>sac|remac_m\d+)_g(?P<gid>\d+)"
    r"_seed(?P<idx>\d+)_step(?P<step>\d+)\.npz"
)


def load(states_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(states_dir, "*.npz"))):
        m = FNAME.match(os.path.basename(path))
        if m is None:
            continue
        key = (m["env"], m["method"], f'{m["gid"]}-{m["idx"]}')
        runs.setdefault(key, {})[int(m["step"])] = path
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--k-check", action="store_true",
                    help="Also print k=3,5 to confirm the ordering is unchanged.")
    args = ap.parse_args()

    runs = load(args.states_dir)
    if not runs:
        raise SystemExit(f"no eval-state files under {args.states_dir}")
    ks = [3, args.k, 5] if args.k_check else [args.k]

    for env in sorted({e for e, _, _ in runs}):
        env_runs = {k: v for k, v in runs.items() if k[0] == env}

        # Common training-time grid: the evaluation steps present in every run.
        common = set.intersection(*(set(v) for v in env_runs.values()))
        dropped = max(len(v) for v in env_runs.values()) - len(common)
        steps = sorted(common)

        pooled = {
            key: np.concatenate([np.load(evals[s])["obs"] for s in steps], 0)
            for key, evals in sorted(env_runs.items())
        }
        sizes = {len(v) for v in pooled.values()}
        if len(sizes) != 1:
            raise SystemExit(f"{env}: pooled N differs across runs: {sorted(sizes)}")
        n = sizes.pop()

        # One reference mean/std per environment, from a balanced pool.
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        ref = np.concatenate(
            [v[rng.choice(len(v), size=POOL_PER_SEED, replace=False)]
             for v in pooled.values()], 0)
        mean, std = ref.mean(0), ref.std(0)
        keep = std > ZERO_VAR_TOL
        d = int(keep.sum())

        note = f", {dropped} evaluation(s) dropped to keep the grid common" if dropped else ""
        print(f"\n% --- {env}: N={n} pooled states per seed "
              f"({len(steps)} evaluations x {n // len(steps)}), d={d} "
              f"({int((~keep).sum())} zero-variance dims dropped), k={args.k}{note}")

        per_method = {}
        for (_e, method, _seed), x in pooled.items():
            z = (x - mean)[:, keep] / std[keep]
            for k in ks:
                per_method.setdefault((method, k), []).append(knn_entropy(z, k=k))

        for k in ks:
            tag = "" if k == args.k else f"  [k={k} check]"
            for method in ["sac"] + [f"remac_m{m}" for m in (1, 2, 4, 8)]:
                v = per_method.get((method, k))
                if v is None:
                    continue
                v = np.asarray(v)
                print(f"        {method:10s} & ${v.mean():.2f} \\pm {v.std(ddof=1):.2f}$ "
                      f"\\\\  % n={len(v)} seeds{tag}")


if __name__ == "__main__":
    main()
