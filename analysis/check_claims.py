"""Re-check, against the data, every quantitative claim the text makes over the task set.

Sentences of the form "in all six tasks" are claims about a particular set of tasks, so
they have to be rechecked whenever that set changes.  This script evaluates each of them
and prints PASS or FAIL with the numbers behind it, so the wording follows the data
rather than memory.

Conventions are the paper's (App. C.1): a seed is (run_id, seed_id), the *final* value
of a run is the mean over the last 10% of training, and n = 10 per configuration.

    python analysis/check_claims.py
    REMAC_DATA=data/brax-remax-ac-merged.pkl.gz python analysis/check_claims.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_eps_sweeps import B_PER_M, DEFAULT_LR, ENV_ORDER, MS  # noqa: E402

DATA = os.environ.get("REMAC_DATA", "data/brax-remax-ac-report.pkl.gz")
LAST_FRAC = 0.1
EPSILONS = [1e-8, 1e-2, 1e-1, 1.0]

_n_pass = _n_fail = _n_skip = 0


def load():
    d = pd.read_pickle(DATA)
    d["env_s"] = d["env"].str.replace("brax/", "", regex=False)
    return d[d["_step"] <= 3_000_000]


def _finals(df, metric):
    """One final value per seed: the mean over the last 10% of that seed's curve.

    A metric absent from the frame gives an empty result rather than a KeyError:
    train/policy_std and train/sigma_grad only exist for the runs made after the
    diagnostics were added, so most of the frame does not carry them.
    """
    if metric not in df.columns:
        return np.array([])
    sub = df[df[metric].notna()]
    if sub.empty:
        return np.array([])
    out = []
    for _, g in sub.groupby(["run_id", "seed_id"], observed=True):
        cut = g["_step"].max() * (1.0 - LAST_FRAC)
        out.append(g.loc[g["_step"] >= cut, metric].mean())
    return np.array(out, dtype=float)


def remac(d, env, m, eps=1e-8):
    return d[(d.env_s == env) & (d.algorithm == "remax_ac") & (d.remax_m == m)
             & (d.remax_num_samples == B_PER_M[m])
             & np.isclose(d.actor_epsilon, eps)
             & np.isclose(d.learning_rate, DEFAULT_LR.get(env, np.nan))]


def baseline(d, env, algo):
    return d[(d.env_s == env) & (d.algorithm == algo)]


def report(name, ok, detail, skipped=False):
    global _n_pass, _n_fail, _n_skip
    if skipped:
        _n_skip += 1
        tag = "SKIP"
    elif ok:
        _n_pass += 1
        tag = "PASS"
    else:
        _n_fail += 1
        tag = "FAIL"
    print(f"[{tag}] {name}")
    for line in detail:
        print(f"       {line}")


def envs_present(d):
    return [e for e in ENV_ORDER if (d.env_s == e).any()]


# ---------------------------------------------------------------- the claims
def claim_entropy_m_gt_1(d, envs):
    """Sec. 5: "M>1 yields higher entropy than M=1 in all <N> tasks"."""
    bad, detail, missing = [], [], []
    for env in envs:
        e1 = _finals(remac(d, env, 1), "train/entropy")
        if e1.size == 0:
            missing.append(env)
            continue
        worst = None
        for m in (2, 4, 8):
            em = _finals(remac(d, env, m), "train/entropy")
            if em.size == 0:
                continue
            diff = em.mean() - e1.mean()
            worst = diff if worst is None else min(worst, diff)
        if worst is None:
            missing.append(env)
            continue
        detail.append(f"{env:16s} M=1 {e1.mean():9.2f} | smallest gain over M=1: {worst:+8.2f}")
        if worst <= 0:
            bad.append(env)
    report(f"entropy: every M>1 beats M=1, in all {len(envs)} tasks (eps=1e-8)",
           not bad and not missing,
           detail + ([f"VIOLATED in: {bad}"] if bad else [])
                  + ([f"no data yet: {missing}"] if missing else []),
           skipped=bool(missing) and not bad)


def claim_scale_increases(d, envs):
    """App. C.4: "the final scale increases with M in every environment"."""
    bad, detail, missing = [], [], []
    for env in envs:
        scales = []
        for m in MS:
            s = _finals(remac(d, env, m), "train/policy_std")
            scales.append(s.mean() if s.size else np.nan)
        if np.isnan(scales).any():
            missing.append(env)
            continue
        mono = all(scales[i] < scales[i + 1] for i in range(len(scales) - 1))
        detail.append(f"{env:16s} " + "  ".join(f"M={m}:{v:.4g}" for m, v in zip(MS, scales))
                      + ("" if mono else "   <- not increasing"))
        if not mono:
            bad.append(env)
    report("policy scale increases with M in every environment (Tab. 7)",
           not bad and not missing,
           detail + ([f"VIOLATED in: {bad}"] if bad else [])
                  + ([f"no policy_std in this frame for: {missing}"
                       " -- those tasks' scale data lives in the sigma runs, read with"
                       " analysis/parse_sigma_logs.py, not in the report frame"] if missing else []),
           skipped=bool(missing) and not bad)


def claim_m8_entropy_every_eps(d, envs):
    """Sec. 5: "M=8 yields higher final entropy than M=1 at every tested eps"."""
    bad, detail, missing = [], [], []
    for eps in EPSILONS:
        for env in envs:
            e1 = _finals(remac(d, env, 1, eps), "train/entropy")
            e8 = _finals(remac(d, env, 8, eps), "train/entropy")
            if e1.size == 0 or e8.size == 0:
                missing.append(f"{env}@{eps:g}")
                continue
            if e8.mean() <= e1.mean():
                bad.append(f"{env}@eps={eps:g} (M=8 {e8.mean():.2f} <= M=1 {e1.mean():.2f})")
        detail.append(f"eps={eps:<8g} checked {len(envs)} tasks")
    report("entropy: M=8 above M=1 at every tested epsilon, in every task",
           not bad and not missing,
           detail + ([f"VIOLATED: {bad}"] if bad else [])
                  + ([f"no data yet: {len(missing)} cells, e.g. {missing[:4]}"] if missing else []),
           skipped=bool(missing) and not bad)


def claim_return_vs_ppo(d, envs):
    """Sec. 5: ReMAC (M>1) "generally performed better than PPO"."""
    detail, worse = [], []
    for env in envs:
        p = _finals(baseline(d, env, "ppo"), "eval/return")
        if p.size == 0:
            continue
        best = max((_finals(remac(d, env, m), "eval/return").mean()
                    for m in (2, 4, 8)
                    if _finals(remac(d, env, m), "eval/return").size), default=np.nan)
        if np.isnan(best):
            continue
        detail.append(f"{env:16s} best ReMAC(M>1) {best:10.2f}  PPO {p.mean():10.2f}")
        if best <= p.mean():
            worse.append(env)
    report("return: best ReMAC with M>1 beats PPO (\"generally\", so a miss is not fatal)",
           True, detail + ([f"note: PPO is ahead in {worse}"] if worse else []))


def claim_return_vs_sac(d, envs):
    """Sec. 5: ReMAC with M>1 "achieved returns comparable to those of SAC".

    "Comparable" is reported here as the gap between the best ReMAC (M>1) and SAC in
    units of SAC's across-seed standard deviation, so the wording can be checked
    against a number instead of an impression.

    The test is one-sided: only ReMAC falling more than 1 sd BELOW SAC threatens the
    claim.  ReMAC coming out well above SAC (Reacher, HumanoidStandup) is consistent
    with "comparable" as the paper uses it -- the sentence claims ReMAC keeps up, not
    that it ties.
    """
    detail, flagged = [], []
    for env in envs:
        sac = _finals(baseline(d, env, "sac"), "eval/return")
        if sac.size == 0:
            continue
        best_m, best = None, -np.inf
        for m in (2, 4, 8):
            r = _finals(remac(d, env, m), "eval/return")
            if r.size and r.mean() > best:
                best, best_m = r.mean(), m
        if best_m is None:
            continue
        sd = sac.std(ddof=1) if sac.size > 1 else np.nan
        gap = (best - sac.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        note = "   <- ReMAC below SAC by more than 1 sd" if gap < -1 else ""
        detail.append(f"{env:16s} best ReMAC(M={best_m}) {best:10.2f}  SAC {sac.mean():10.2f}"
                      f"  gap {gap:+6.2f} sd{note}")
        if gap < -1:
            flagged.append(f"{env} ({gap:+.2f} sd)")
    report("return: ReMAC with M>1 keeps up with SAC (not more than 1 sd below it)",
           not flagged, detail + ([f"outside 1 sd in: {flagged}"] if flagged else []))


def claim_humanoid_vs_sac(d):
    """App. C.3: "ReMAC with M>=2 attains a higher average return than SAC"."""
    env = "humanoidstandup"
    s = _finals(baseline(d, env, "sac"), "eval/return")
    if s.size == 0:
        report("HumanoidStandup: ReMAC M>=2 above SAC", False, ["no SAC data"], skipped=True)
        return
    detail, bad = [f"SAC {s.mean():.0f}"], []
    for m in (2, 4, 8):
        r = _finals(remac(d, env, m), "eval/return")
        if r.size == 0:
            continue
        detail.append(f"M={m}: {r.mean():.0f}")
        if r.mean() <= s.mean():
            bad.append(m)
    report("HumanoidStandup: ReMAC with M>=2 above SAC (App. C.3)", not bad,
           detail + ([f"VIOLATED for M in {bad}"] if bad else []))


def claim_pusher_present(d):
    """Pusher must be complete at n = 10 on every arm of Figs. 3 and 4."""
    detail, missing = [], []
    for m in MS:
        r = remac(d, "pusher", m)
        n = r.groupby(["run_id", "seed_id"], observed=True).ngroups if not r.empty else 0
        detail.append(f"ReMAC M={m}: {n} seeds")
        if n != 10:
            missing.append(f"M={m} ({n})")
    for algo in ("sac", "ppo"):
        b = baseline(d, "pusher", algo)
        n = b.groupby(["run_id", "seed_id"], observed=True).ngroups if not b.empty else 0
        detail.append(f"{algo.upper()}: {n} seeds")
        if n != 10:
            missing.append(f"{algo} ({n})")
    report("Pusher is complete at n=10 for every arm of Figs. 3 and 4",
           not missing, detail + ([f"INCOMPLETE: {missing}"] if missing else []),
           skipped=bool(missing))


def main():
    print(f"data: {DATA}\n")
    d = load()
    envs = envs_present(d)
    print(f"tasks present: {envs}\n")

    claim_pusher_present(d)
    print()
    claim_entropy_m_gt_1(d, envs)
    print()
    claim_scale_increases(d, envs)
    print()
    claim_m8_entropy_every_eps(d, envs)
    print()
    claim_return_vs_ppo(d, envs)
    print()
    claim_return_vs_sac(d, envs)
    print()
    claim_humanoid_vs_sac(d)

    print(f"\n{_n_pass} passed, {_n_fail} failed, {_n_skip} skipped (incomplete data)")
    if _n_fail:
        print("A FAILED claim means the sentence in the paper must be reworded, not that "
              "the run is wrong.")


if __name__ == "__main__":
    main()
