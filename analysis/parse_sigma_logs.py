"""Aggregate the Tab. 7 extension runs from their stdout logs.

The extension sweep (experiments/sh/remac_sigma_m8.sh) writes one log per
process to brax/logs/sigma_m8/runs/remac_m{M}_{env}_seed{SEED}.log.  wandb was unavailable,
so the metrics are read back from the two line formats train.py prints:

    [remax_ac] seed=0 step=30080 actor_loss=... entropy=... actor_grad_norm=...
               actor_update_norm=... policy_std=... sigma_grad=...
    Seed 0 Step 30080, Mean episode length: ..., Mean return: ...

Each process runs `num_seeds=2` vmapped seeds (printed as seed=0 and seed=1), and the five
processes use seed_id 2..6, so one configuration is 10 seeds.  Following App. C.1, the
final value of a seed is the mean over the last 10% of its logged points, and we report the
mean +/- s.d. across seeds.

Also reports the two quantities the scale-gradient discussion needs:
  * the fraction of logged updates whose sigma_grad is negative (i.e. that push sigma up),
    computed over the whole run as in Tab. 7;
  * the decomposition of the logged entropy into the pre-squash Gaussian term
    d/2 * log(2 pi e sigma^2) and the remaining tanh-Jacobian term.

Usage:  python analysis/parse_sigma_logs.py [--runs DIR] [--last-frac 0.1]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNS = "brax/logs/sigma_m8/runs"

# Action dimension per Brax task, used for the per-dimension and decomposition columns.
ACTION_DIM = {
    "ant": 8, "halfcheetah": 6, "hopper": 3, "walker2d": 6,
    "reacher": 2, "swimmer": 2, "humanoidstandup": 17,
    "pusher": 7,
}

FNAME_RE = re.compile(r"remac_m(?P<m>\d+)_(?P<env>[a-z0-9]+)_seed(?P<seed_id>\d+)\.log$")
TRAIN_RE = re.compile(
    r"^\[remax_ac\]\s+seed=(?P<seed>\d+)\s+step=(?P<step>\d+)\s+.*?"
    r"entropy=(?P<entropy>-?[\d.eE+-]+)\s+.*?"
    r"policy_std=(?P<policy_std>-?[\d.eE+-]+)\s+"
    r"sigma_grad=(?P<sigma_grad>-?[\d.eE+-]+)"
)
EVAL_RE = re.compile(
    r"^Seed (?P<seed>\d+) Step (?P<step>\d+), Mean episode length: [\d.eE+-]+, "
    r"Mean return: (?P<ret>-?[\d.eE+-]+)"
)


def parse_log(path):
    """Return (train_rows, eval_rows) for one process log."""
    meta = FNAME_RE.search(path.name)
    if meta is None:
        return [], []
    base = {"env": meta["env"], "remax_m": int(meta["m"]), "seed_id": int(meta["seed_id"])}
    train, ev = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = TRAIN_RE.match(line)
            if m:
                train.append({**base, "vmap_seed": int(m["seed"]), "step": int(m["step"]),
                              "entropy": float(m["entropy"]),
                              "policy_std": float(m["policy_std"]),
                              "sigma_grad": float(m["sigma_grad"])})
                continue
            m = EVAL_RE.match(line)
            if m:
                ev.append({**base, "vmap_seed": int(m["seed"]), "step": int(m["step"]),
                           "ret": float(m["ret"])})
    return train, ev


def final_per_seed(df, col, last_frac):
    """Mean over the last `last_frac` of each seed's logged points (App. C.1)."""
    out = {}
    for key, g in df.groupby(["env", "remax_m", "seed_id", "vmap_seed"]):
        g = g.sort_values("step")
        cut = g["step"].max() * (1.0 - last_frac)
        out[key] = g.loc[g["step"] >= cut, col].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=DEFAULT_RUNS)
    ap.add_argument("--last-frac", type=float, default=0.1)
    args = ap.parse_args()

    paths = sorted(Path(args.runs).glob("remac_m*_seed*.log"))
    if not paths:
        raise SystemExit(f"no logs under {args.runs}")

    train_rows, eval_rows = [], []
    for p in paths:
        t, e = parse_log(p)
        train_rows += t
        eval_rows += e
    train = pd.DataFrame(train_rows)
    ev = pd.DataFrame(eval_rows)
    if train.empty:
        raise SystemExit("no train-metric lines parsed; check the log format")

    fin = {c: final_per_seed(train, c, args.last_frac) for c in
           ("entropy", "policy_std")}
    fin_ret = final_per_seed(ev, "ret", args.last_frac) if not ev.empty else {}

    # Fraction of logged updates that push sigma up, over the whole run (Tab. 7 convention).
    frac_up = (train.assign(up=train["sigma_grad"] < 0)
               .groupby(["env", "remax_m", "seed_id", "vmap_seed"])["up"].mean().to_dict())

    per_seed = defaultdict(dict)
    for key in fin["entropy"]:
        per_seed[key] = {
            "entropy": fin["entropy"][key],
            "policy_std": fin["policy_std"][key],
            "sigma_up_frac": 100.0 * frac_up.get(key, np.nan),
            "ret": fin_ret.get(key, np.nan),
        }

    rows = []
    for (env, m, seed_id, vmap_seed), v in per_seed.items():
        rows.append({"env": env, "remax_m": m, "seed_id": seed_id,
                     "vmap_seed": vmap_seed, **v})
    seeds = pd.DataFrame(rows)

    c = 0.5 * np.log(2 * np.pi * np.e)
    agg = seeds.groupby(["env", "remax_m"]).agg(
        n=("entropy", "size"),
        entropy=("entropy", "mean"), entropy_sd=("entropy", "std"),
        ret=("ret", "mean"), ret_sd=("ret", "std"),
        policy_std=("policy_std", "mean"), policy_std_sd=("policy_std", "std"),
        sigma_up=("sigma_up_frac", "mean"), sigma_up_sd=("sigma_up_frac", "std"),
    ).reset_index()
    d = agg["env"].map(ACTION_DIM)
    agg["scale_term"] = d * (c + np.log(agg["policy_std"]))
    agg["tanh_term"] = agg["entropy"] - agg["scale_term"]

    pd.set_option("display.width", 200)
    print(agg.round(3).to_string(index=False))
    print()
    print("scale_term = d/2*log(2 pi e sigma^2) using the mean sigma; by Jensen this "
          "overestimates it, so the tanh term is at most as negative as shown.")
    incomplete = agg[agg["n"] < 10]
    if not incomplete.empty:
        print()
        print("INCOMPLETE (fewer than 10 seeds):")
        print(incomplete[["env", "remax_m", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
