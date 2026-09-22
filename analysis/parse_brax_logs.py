"""Turn the stdout run logs into rows of the report DataFrame's schema.

Every figure and table in the paper is drawn from
``data/brax-remax-ac-report.pkl.gz``, a long-format pandas frame that
``fetch_wandb_data_with_seeds()`` (in analysis/plot.py) originally built from wandb.
The wandb API key on this machine is no longer accepted, so the camera-ready runs
(Pusher, and HumanoidStandup's missing eps = 1e-2 arm) are launched with
``python -u`` into per-run log files instead, and this module converts those logs
into rows of exactly that schema.  The merged frame is a drop-in replacement for
the pickle, so plot.py, make_cxt2_figs.py and make_tables.py keep working unchanged.

Log format (written by brax/train.py plus experiments/lib_runlog.sh)
----------------------------------------------------------------
One header line, then two interleaved metric line formats::

    #CONFIG algo=remax_ac env=pusher m=4 b=8 eps=1e-8 lr=0.0001 opt=adam seed_id=7 num_seeds=10
    [remax_ac] seed=0 step=30080 actor_loss=-1.2 entropy=-5.6 actor_grad_norm=0.1 \
        critic_grad_norm=0.2 model_grad_norm=0.3 actor_update_norm=0.4 policy_std=0.5 sigma_grad=-0.6
    Seed 0 Step 30080, Mean episode length: 100.0, Mean return: -123.456
    ===== RUN COMPLETE

``seed=N`` / ``Seed N`` is the vmap index inside the process, not the process's
``seed_id``.  A unique seed is therefore ``(seed_id, vmap index)``, which is why
the frame identifies a seed by ``(run_id, seed_id)``: this module gives each log
file its own synthetic ``run_id`` and writes the vmap index into ``seed_id`` as
``"seed_<i>"``, matching what the wandb fetch produced.  10 seeds per
configuration either way -- Pusher runs 1 process x 10 vmapped seeds,
HumanoidStandup 10 processes x 1.

Schema notes that matter
------------------------
``_prepare_plot_df()`` recovers M and epsilon from the ``algo_label`` STRING, not
from the ``remax_m``/``actor_epsilon`` columns, so the label has to be rebuilt
byte-for-byte as fetch_wandb_data_with_seeds() wrote it::

    ReMAC (M=4, eps=1e-08, lr=0.0001)        SAC (lr=0.000204951)

``make_cxt2_figs.remac()`` filters on the columns instead, so both have to agree.

Usage
-----
    python analysis/parse_brax_logs.py --dry-run --logs 'brax/logs/pusher/final_*.log'
    python analysis/parse_brax_logs.py \\
        --base data/brax-remax-ac-report.pkl.gz \\
        --out  data/brax-remax-ac-camera-ready.pkl.gz \\
        --logs 'brax/logs/pusher/final_*.log' \\
               'brax/logs/pusher/eps_*.log' \\
               'brax/logs/humanoid_eps/*.log'
"""

import argparse
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- log grammar
CONFIG_RE = re.compile(r"^#CONFIG\s+(.*)$")
DONE_RE = re.compile(r"^===== RUN COMPLETE")
# The train line is printed by train.py's train_log_callback. Its keys vary with
# the algorithm and with which diagnostics are compiled in, so they are scraped
# as key=value pairs rather than matched positionally.
TRAIN_RE = re.compile(r"^\[(?P<algo>[a-z_0-9]+)\]\s+seed=(?P<seed>\d+)\s+step=(?P<step>\d+)\s+(?P<kv>.*)$")
KV_RE = re.compile(r"([a-z_]+)=(-?[\d.eE+-]+|nan|inf|-inf)")
EVAL_RE = re.compile(
    r"^Seed (?P<seed>\d+) Step (?P<step>\d+), "
    r"Mean episode length: (?P<len>-?[\d.eE+-]+), "
    r"Mean return: (?P<ret>-?[\d.eE+-]+)"
)

# train-line key -> frame column.  Anything not listed is ignored.
TRAIN_COLS = {
    "actor_loss": "train/actor_loss",
    "critic_loss": "train/critic_loss",
    "entropy": "train/entropy",
    "actor_grad_norm": "train/actor_grad_norm",
    "critic_grad_norm": "train/critic_grad_norm",
    "model_grad_norm": "train/model_grad_norm",
    "actor_update_norm": "train/actor_update_norm",
    "policy_std": "train/policy_std",
    "sigma_grad": "train/sigma_grad",
}

ALGO_TAG = {"sac": "SAC", "ppo": "PPO", "td3": "TD3"}


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def _label(cfg):
    """Rebuild algo_label exactly as fetch_wandb_data_with_seeds() wrote it."""
    lr = _to_float(cfg.get("lr"))
    if cfg["algo"] == "remax_ac":
        return (f"ReMAC (M={int(float(cfg['m']))}, "
                f"eps={_to_float(cfg['eps']):g}, lr={lr:g})")
    return f"{ALGO_TAG[cfg['algo']]} (lr={lr:g})"


def _run_id(cfg, path):
    """A stable synthetic run_id, one per log file.

    The wandb ids were 8-character slugs; these are the same width and carry a
    "log" prefix so a merged frame shows at a glance which rows came from a log.
    """
    key = "|".join(str(cfg.get(k)) for k in
                   ("algo", "env", "m", "b", "eps", "lr", "opt", "seed_id"))
    key += "|" + os.path.basename(path)
    return "log" + hashlib.sha1(key.encode()).hexdigest()[:8]


def parse_log(path):
    """-> (config dict, tidy DataFrame indexed by (_step, seed_id)) or None."""
    cfg, complete = None, False
    train_rows, eval_rows = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            if cfg is None:
                c = CONFIG_RE.match(line)
                if c:
                    cfg = dict(kv.split("=", 1) for kv in c.group(1).split() if "=" in kv)
                continue
            t = TRAIN_RE.match(line)
            if t:
                row = {"_step": int(t["step"]), "seed_id": f"seed_{int(t['seed'])}"}
                for k, v in KV_RE.findall(t["kv"]):
                    if k in TRAIN_COLS:
                        row[TRAIN_COLS[k]] = _to_float(v)
                train_rows.append(row)
                continue
            e = EVAL_RE.match(line)
            if e:
                eval_rows.append({
                    "_step": int(e["step"]),
                    "seed_id": f"seed_{int(e['seed'])}",
                    "eval/episode_length": _to_float(e["len"]),
                    "eval/return": _to_float(e["ret"]),
                })
                continue
            if DONE_RE.match(line):
                complete = True

    if cfg is None:
        return None
    if not train_rows and not eval_rows:
        return cfg, complete, pd.DataFrame(columns=["_step", "seed_id"])

    tr = (pd.DataFrame(train_rows).groupby(["_step", "seed_id"], as_index=False).last()
          if train_rows else pd.DataFrame(columns=["_step", "seed_id"]))
    ev = (pd.DataFrame(eval_rows).groupby(["_step", "seed_id"], as_index=False).last()
          if eval_rows else pd.DataFrame(columns=["_step", "seed_id"]))
    # Outer join: a step may carry only an evaluation (step 0) or only train metrics.
    df = pd.merge(ev, tr, on=["_step", "seed_id"], how="outer")
    return cfg, complete, df


def logs_to_frame(paths, *, require_complete=False, verbose=True):
    """Parse every log into one frame in the report pickle's schema."""
    frames, skipped = [], []
    for path in paths:
        parsed = parse_log(path)
        if parsed is None:
            skipped.append((path, "no #CONFIG header"))
            continue
        cfg, complete, df = parsed
        if df.empty:
            skipped.append((path, "no metric lines"))
            continue
        if require_complete and not complete:
            skipped.append((path, "no '===== RUN COMPLETE' marker"))
            continue

        is_remax = cfg["algo"] == "remax_ac"
        df = df.copy()
        df["env"] = f"brax/{cfg['env']}"
        df["algorithm"] = cfg["algo"]
        df["algo_label"] = _label(cfg)
        df["remax_m"] = float(cfg["m"]) if is_remax else np.nan
        df["actor_epsilon"] = _to_float(cfg.get("eps")) if is_remax else np.nan
        df["remax_num_samples"] = float(cfg["b"]) if is_remax else np.nan
        df["learning_rate"] = _to_float(cfg.get("lr"))
        # Not a wandb column; carried so Tab. 8 can separate the two actor optimizers.
        df["actor_optimizer"] = cfg.get("opt", "adam" if is_remax else np.nan)
        df["cfg_seed_id"] = float(cfg["seed_id"])
        df["num_seeds"] = float(cfg["num_seeds"])
        df["run_id"] = _run_id(cfg, path)
        df["run_name"] = os.path.splitext(os.path.basename(path))[0]
        df["complete"] = complete
        frames.append(df)
        if verbose:
            n_seeds = df["seed_id"].nunique()
            print(f"  {os.path.basename(path):48s} {len(df):5d} rows  "
                  f"{n_seeds:2d} vmap seed(s)  "
                  f"max step {int(df['_step'].max()):>8d}"
                  f"{'' if complete else '   [INCOMPLETE]'}")

    if skipped and verbose:
        print("\nskipped:")
        for path, why in skipped:
            print(f"  {os.path.basename(path):48s} {why}")
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["_step"] = pd.to_numeric(out["_step"], errors="coerce")
    out = out.dropna(subset=["_step"])
    out["_step"] = out["_step"].astype(int)
    for c in ("learning_rate", "actor_epsilon", "remax_num_samples"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def seed_census(df):
    """Seeds per configuration -- the check that every cell really has n = 10."""
    if df.empty:
        return df
    keys = ["env", "algorithm", "actor_optimizer", "remax_m", "actor_epsilon",
            "remax_num_samples", "learning_rate"]
    keys = [k for k in keys if k in df.columns]
    g = df.groupby(keys, dropna=False)
    return pd.DataFrame({
        "seeds": g.apply(lambda s: s.groupby(["run_id", "seed_id"]).ngroups,
                         include_groups=False),
        "max_step": g["_step"].max(),
        "complete": g["complete"].all() if "complete" in df.columns else True,
    }).reset_index()


def drop_incomplete_configs(df, min_seeds=10, verbose=True):
    """Drop configurations that do not yet have `min_seeds` seeds.

    Every configuration the paper reports is n = 10.  A configuration that is still
    filling up would otherwise be drawn as a curve like any other -- with a narrower
    error band, or, worse, as the only M present in a panel, which reads as a result
    rather than as a run that has not finished.
    """
    if df.empty:
        return df
    keys = [k for k in ("env", "algorithm", "actor_optimizer", "remax_m",
                        "actor_epsilon", "remax_num_samples", "learning_rate")
            if k in df.columns]
    counts = (df.groupby(keys, dropna=False)
                .apply(lambda g: g.groupby(["run_id", "seed_id"]).ngroups,
                       include_groups=False)
                .rename("seeds").reset_index())
    short = counts[counts["seeds"] < min_seeds]
    if short.empty:
        return df
    keep = df.merge(counts[counts["seeds"] >= min_seeds][keys], on=keys, how="inner")
    if verbose:
        print(f"\ndropped {len(df) - len(keep)} rows from "
              f"{len(short)} configuration(s) with fewer than {min_seeds} seeds:")
        for _, r in short.iterrows():
            desc = ", ".join(f"{k}={r[k]}" for k in keys if str(r[k]) != "nan")
            print(f"  {int(r['seeds']):2d} seeds  {desc}")
    return keep


def merge_into(base_pkl, paths, out_pkl, *, require_complete=False, min_seeds=None):
    base = pd.read_pickle(base_pkl)
    print(f"base: {base.shape[0]} rows, {base.shape[1]} cols  ({base_pkl})")
    print("parsing logs:")
    new = logs_to_frame(paths, require_complete=require_complete)
    if new.empty:
        raise SystemExit("no parseable logs -- nothing to merge")

    print("\nseeds per configuration (want 10):")
    census = seed_census(new)
    print(census.to_string(index=False))
    short = census[census["seeds"] != 10]
    if not short.empty:
        print(f"\nWARNING: {len(short)} configuration(s) do not have 10 seeds")

    new = new.drop(columns=["complete"])
    if min_seeds:
        new = drop_incomplete_configs(new, min_seeds)
    merged = pd.concat([base, new], ignore_index=True)
    if out_pkl:
        merged.to_pickle(out_pkl)
        print(f"\nwrote {out_pkl}: {merged.shape[0]} rows "
              f"(+{merged.shape[0] - base.shape[0]}), {merged.shape[1]} cols")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="data/brax-remax-ac-report.pkl.gz")
    ap.add_argument("--out", default="data/brax-remax-ac-camera-ready.pkl.gz")
    ap.add_argument("--logs", nargs="+", required=True,
                    help="glob(s) of run logs, e.g. 'brax/logs/pusher/*.log'")
    ap.add_argument("--require-complete", action="store_true",
                    help="skip runs without a '===== RUN COMPLETE' marker")
    ap.add_argument("--min-seeds", type=int, default=None,
                    help="drop configurations with fewer than this many seeds "
                         "(use 10 for a figure-ready frame)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and report, but do not write the merged pickle")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no logs matched {args.logs}")
    merge_into(args.base, paths, None if args.dry_run else args.out,
               require_complete=args.require_complete, min_seeds=args.min_seeds)


if __name__ == "__main__":
    main()
