"""Emit the LaTeX table bodies for the appendix, straight from the data.

Numbers quoted in prose cannot be checked against a plot, so every quantity the
appendix claims is tabulated instead.  This script prints the rows so they are
transcribed from the data rather than by hand.

Conventions (stated in the paper, App. C.1):

  * a *seed* is one training run.  A configuration is run as 5 processes x 2 vmapped
    seeds; vmap only batches independent runs, so the unit of analysis is the seed and
    n = 10 per configuration.
  * the *final* value of a run is the mean over its last 10% of training.
  * we report mean +/- s.d. across seeds and make no claim of statistical significance:
    the tables report averages, and the s.d. shows the spread.

    python analysis/make_tables.py
"""

import os
import pickle

import numpy as np
import pandas as pd

MAIN = os.environ.get("REMAC_DATA", "data/brax-remax-ac-report.pkl.gz")
REB = "data/remac_rebuttal_data.pkl"
ENVS = ["halfcheetah", "ant", "hopper"]
PRETTY = {"halfcheetah": "HalfCheetah", "ant": "Ant", "hopper": "Hopper"}
B_PER_M = {1: 8, 2: 8, 4: 8, 8: 16}
SEEDS = ["seed_0", "seed_1"]


def fmt(x, sd, prec=0):
    return f"${x:.{prec}f} \\pm {sd:.{prec}f}$"


# ---------------------------------------------------------------- HumanoidStandup
def humanoidstandup():
    d = pd.read_pickle(MAIN)
    d["env_s"] = d["env"].str.replace("brax/", "", regex=False)
    h = d[d.env_s == "humanoidstandup"]

    def finals(sub, metric="eval/return"):
        sub = sub[sub[metric].notna()]
        late = sub[sub["_step"] >= 0.9 * sub["_step"].max()]
        return late.groupby(["run_id", "seed_id"])[metric].mean().to_numpy()

    names = ["PPO", "SAC"] + [f"ReMAC ($M={m}$)" for m in [1, 2, 4, 8]]
    groups = [finals(h[h.algorithm == "ppo"]), finals(h[h.algorithm == "sac"])]
    for m in [1, 2, 4, 8]:
        groups.append(finals(h[(h.algorithm == "remax_ac") & (h.remax_m == m)
                               & (h.remax_num_samples == B_PER_M[m])
                               & np.isclose(h.actor_epsilon, 1e-8)]))
    print("% --- Table: HumanoidStandup return (n=10 seeds)")
    for nm, g in zip(names, groups):
        print(f"        {nm} & {fmt(g.mean(), g.std(ddof=1))} \\\\")
    print()


def humanoid_epsilon():
    """HumanoidStandup entropy and return across Adam's epsilon (n=10)."""
    d = pd.read_pickle(MAIN)
    d["env_s"] = d["env"].str.replace("brax/", "", regex=False)
    h = d[d.env_s == "humanoidstandup"]
    # HumanoidStandup was originally run at eps in {1e-8, 1e-1, 1}; the missing 1e-2
    # arm was added for the camera-ready version, so include it once it is present.
    epss = [e for e in (1e-8, 1e-2, 1e-1, 1.0)
            if np.isclose(h["actor_epsilon"].dropna().to_numpy()[:, None], e,
                          atol=0).any()]
    print(f"% (epsilons found in the data: {[f'{e:g}' for e in epss]})")

    def finals(sub, metric):
        sub = sub[sub[metric].notna()]
        late = sub[sub["_step"] >= 0.9 * sub["_step"].max()]
        return late.groupby(["run_id", "seed_id"])[metric].mean().to_numpy()

    def cell(m, e, metric, dec):
        g = finals(h[(h.algorithm == "remax_ac") & (h.remax_m == m)
                     & (h.remax_num_samples == B_PER_M[m])
                     & np.isclose(h.actor_epsilon, e, atol=0)], metric)
        return f"${g.mean():.{dec}f} \\pm {g.std(ddof=1):.{dec}f}$"

    print("% --- Table: HumanoidStandup policy entropy across epsilon (n=10)")
    for m in [1, 2, 4, 8]:
        row = " & ".join(cell(m, e, "train/entropy", 0) for e in epss)
        print(f"        ReMAC ($M={m}$) & {row} \\\\")
    print("% --- return across epsilon (for the text)")
    for m in [1, 2, 4, 8]:
        row = " & ".join(cell(m, e, "eval/return", 0) for e in epss)
        print(f"        ReMAC ($M={m}$) & {row} \\\\")
    print()


# ---------------------------------------------------------------- rebuttal pickle
def per_seed(runs, metric, *, late=True, frac=0.1):
    out = []
    for r in runs:
        for s in SEEDS:
            key = f"{s}/{metric}"
            if key not in r["curves"]:
                continue
            c = np.array(r["curves"][key], float)
            if np.all(np.isnan(c)):
                continue
            out.append(np.nanmean(c[int((1 - frac) * len(c)):]) if late else np.nanmean(c))
    return np.array(out)


def sel(runs, env, m=None, opt=None, algo="remax_ac"):
    return [r for r in runs
            if r.get("env") == env
            and (m is None or r.get("remax_m") == m)
            and (opt is None or r.get("actor_optimizer") == opt)
            and (algo is None or r.get("algorithm") == algo)]


def damping(runs):
    print("% --- Table: gradient / update norm, ratio M=4 over M=1 (no p-values)")
    for opt in ["adam", "sgd"]:
        name = "Adam" if opt == "adam" else "SGD"
        for metric, sym in [("train/actor_grad_norm",
                             "$\\|\\nabla_\\theta L^M_{\\text{actor}}\\|$"),
                            ("train/actor_update_norm", "$\\|\\Delta\\theta\\|$")]:
            cells = []
            for env in ENVS:
                v1 = per_seed(sel(runs, env, 1, opt), metric)
                v4 = per_seed(sel(runs, env, 4, opt), metric)
                cells.append(f"${v4.mean()/v1.mean():.2f}$")
            print(f"        {name} & {sym} & " + " & ".join(cells) + " \\\\")
    print()


def sigma(runs):
    """Scale-gradient proxy, summarised by the *fraction of updates that push sigma up*.

    The run-mean of train/sigma_grad is a poor summary: it is heavy-tailed, and the runs
    have already separated in sigma before the first logged evaluation, so the mean is
    dominated by the post-separation tail where each M sits at a different sigma.  The
    fraction of logged updates whose scale gradient is negative (i.e. that push the
    pre-squash scale up) is computed per seed and is monotone in M in every environment.
    """
    print("% --- Table: fraction of sigma-increasing updates, and final policy scale, Adam actor")

    def sci(x):
        e = int(np.floor(np.log10(abs(x))))
        return f"${x/10**e:.1f}\\!\\times\\!10^{{{e}}}$"

    def frac_up(rs):
        """Per-seed fraction of logged updates with a negative (sigma-increasing) scale gradient."""
        out = []
        for r in rs:
            for s in SEEDS:
                k = f"{s}/train/sigma_grad"
                if k not in r["curves"]:
                    continue
                c = np.array(r["curves"][k], float)
                if np.all(np.isnan(c)):
                    continue
                out.append(np.nanmean(c < 0))
        return np.array(out)

    for env in ENVS:
        fu = [frac_up(sel(runs, env, m, "adam")) for m in [1, 2, 4]]
        ps = [per_seed(sel(runs, env, m, "adam"), "train/policy_std").mean() for m in [1, 2, 4]]
        a = " & ".join(f"${v.mean()*100:.1f} \\pm {v.std(ddof=1)*100:.1f}$" for v in fu)
        b = " & ".join(f"${v:.3f}$" if v >= 1e-2 else sci(v) for v in ps)
        print(f"        {PRETTY[env]} & {a} & {b} \\\\")
    print()


def sgd_return(runs):
    print("% --- Table: return, Adam actor vs SGD actor ")
    for env in ENVS:
        for m in [1, 2, 4]:
            a = per_seed(sel(runs, env, m, "adam"), "eval/return")
            s = per_seed(sel(runs, env, m, "sgd"), "eval/return")
            lead = PRETTY[env] if m == 1 else ""
            print(f"        {lead} & ${m}$ & {fmt(a.mean(), a.std(ddof=1))} "
                  f"& {fmt(s.mean(), s.std(ddof=1))} \\\\")
    print()


def coverage(runs):
    METRICS = [("coverage/marginal_frac", "marginal"),
               ("coverage/joint_frac", "joint"),
               ("coverage/state_entropy_norm", "state entropy")]
    print("% --- Table: state coverage ")
    for env in ENVS:
        for k, (metric, label) in enumerate(METRICS):
            groups = [per_seed([r for r in runs if r.get("env") == env
                                and r.get("algorithm") == "sac"], metric)]
            groups += [per_seed(sel(runs, env, m, "adam"), metric) for m in [1, 2, 4]]
            lead = PRETTY[env] if k == 0 else ""
            cells = [fmt(g.mean(), g.std(ddof=1), prec=3) for g in groups]
            print(f"        {lead} & {label} & " + " & ".join(cells) + " \\\\")
    print()


if __name__ == "__main__":
    humanoidstandup()
    humanoid_epsilon()      # Tab. 5; defined but not wired in before
    runs = pickle.load(open(REB, "rb"))["runs"]
    damping(runs)
    sigma(runs)
    sgd_return(runs)
    coverage(runs)
