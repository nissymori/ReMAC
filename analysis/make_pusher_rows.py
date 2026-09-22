"""Emit the Pusher rows of Tabs. 2, 7 and 8, straight from the run logs.

Pusher is the eighth main task, added for the camera-ready version.  Its runs were
logged to files rather than to wandb (see analysis/parse_brax_logs.py), so its table rows
come from the logs instead of from the report pickle that analysis/make_tables.py reads.
The conventions are the ones stated in App. C.1 and reused here unchanged:

  * a *seed* is one training run; a configuration is n = 10 seeds, here 10 vmapped
    seeds in one process, so a seed is (run_id, seed_id);
  * the *final* value of a run is the mean over the last 10% of training;
  * we report mean +/- s.d. across seeds.

Tab. 7's "sigma-increasing updates" is the fraction of logged updates whose scale
gradient is negative, computed per seed over the *whole* run (not just the tail) --
the run mean of the gradient itself is dominated by the phase after the different M
have already separated in sigma, which is why the paper summarizes it this way.

    python analysis/make_pusher_rows.py
    python analysis/make_pusher_rows.py --check   # also re-derive Tab. 7 with the
                                             # independent parser, as a cross-check
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_brax_logs import logs_to_frame  # noqa: E402

BRAX = "brax"
LR_FILE = "data/PUSHER_LR"
MS = [1, 2, 4, 8]
SGD_MS = [1, 2, 4]          # Tab. 8 covers M = 1, 2, 4
LAST_FRAC = 0.1


def _load(pattern, require_complete=True):
    """Parse the logs matching `pattern`.

    require_complete defaults to True: a "final value" is the mean over the last 10%
    of *training*, so a run that has not reached 3M steps would contribute the last
    10% of however far it got -- a number that looks finished but is not.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()
    df = logs_to_frame(paths, require_complete=require_complete, verbose=False)
    n_total = len(paths)
    n_used = df["run_name"].nunique() if not df.empty else 0
    if n_used < n_total:
        print(f"%   NOTE: {n_total - n_used} of {n_total} logs matching "
              f"{os.path.basename(pattern)} are still running and were excluded")
    return df


def _per_seed_final(df, metric):
    """Final value per seed: the mean over the last `LAST_FRAC` of that seed's curve."""
    out = []
    sub = df[df[metric].notna()]
    for _, g in sub.groupby(["run_id", "seed_id"], observed=True):
        cut = g["_step"].max() * (1.0 - LAST_FRAC)
        out.append(g.loc[g["_step"] >= cut, metric].mean())
    return np.array(out, dtype=float)


def _per_seed_frac_negative(df, metric):
    """Per-seed fraction of logged updates with a negative (sigma-increasing) value."""
    out = []
    sub = df[df[metric].notna()]
    for _, g in sub.groupby(["run_id", "seed_id"], observed=True):
        out.append(float((g[metric] < 0).mean()))
    return np.array(out, dtype=float)


def _sci(x):
    """Match the paper's formatting: 3 decimals, or scientific below 1e-2."""
    if not np.isfinite(x):
        return "---"
    if abs(x) >= 1e-2:
        return f"${x:.3f}$"
    e = int(np.floor(np.log10(abs(x))))
    return f"${x/10**e:.1f}\\!\\times\\!10^{{{e}}}$"


def _pm(v, prec=0):
    if v.size == 0:
        return "---"
    return f"${v.mean():.{prec}f} \\pm {v.std(ddof=1):.{prec}f}$"


def table2():
    print("% --- Tab. 2: the selected learning rate for Pusher")
    if not os.path.exists(LR_FILE):
        print(f"%   {LR_FILE} does not exist yet (run analysis/pick_pusher_lr.py)")
        return
    lr = float(open(LR_FILE).read().strip())
    exp = int(np.floor(np.log10(lr)))
    mant = lr / 10 ** exp
    tex = (f"$10^{{{exp}}}$" if abs(mant - 1) < 1e-9
           else f"${mant:g}\\times10^{{{exp}}}$")
    print(f"        \\fix{{cr}}{{Pusher}} & \\fix{{cr}}{{{tex}}} \\\\")
    print()


def table7(d):
    """Pusher's row of the scale-gradient table: sigma-increasing %, and final sigma."""
    print("% --- Tab. 7: Pusher (B=16 at every M, Adam actor, n=10 seeds)")
    if d.empty:
        print("%   no adam B=16 logs yet")
        return
    fracs, scales, ns = [], [], []
    for m in MS:
        g = d[(d.remax_m == m) & (d.actor_optimizer == "adam")]
        fr = 100.0 * _per_seed_frac_negative(g, "train/sigma_grad")
        sc = _per_seed_final(g, "train/policy_std")
        fracs.append(_pm(fr, 1) if fr.size else "---")
        scales.append(_sci(sc.mean()) if sc.size else "---")
        ns.append(fr.size)
    print(f"        Pusher & {' & '.join(fracs)} & {' & '.join(scales)} \\\\")
    print(f"%   seeds per M: {dict(zip(MS, ns))}")
    print()


def table8(adam, sgd):
    """Pusher's rows of the Adam-actor vs SGD-actor table."""
    print("% --- Tab. 8: Pusher, Adam actor vs SGD actor (n=10 seeds)")
    if adam.empty and sgd.empty:
        print("%   no logs yet")
        return
    for i, m in enumerate(SGD_MS):
        a = _per_seed_final(adam[adam.remax_m == m], "eval/return") if not adam.empty else np.array([])
        s = _per_seed_final(sgd[sgd.remax_m == m], "eval/return") if not sgd.empty else np.array([])
        lead = "Pusher" if i == 0 else ""
        print(f"        {lead} & ${m}$ & {_pm(a, 1)} & {_pm(s, 1)} \\\\"
              f"   % n = {a.size} / {s.size}")
    print()


def cross_check():
    """Re-derive Tab. 7's Pusher row with analysis/parse_sigma_logs.py, the other parser.

    The two read the same files through completely different code paths (one keys off
    the `#CONFIG` header, the other off the file name), so agreement is a real check.
    """
    import subprocess
    print("% --- cross-check: analysis/parse_sigma_logs.py on the same logs")
    r = subprocess.run([sys.executable, "analysis/parse_sigma_logs.py"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "pusher" in line or line.lstrip().startswith("env"):
            print("%   " + line)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="also re-derive Tab. 7 with analysis/parse_sigma_logs.py")
    args = ap.parse_args()

    adam = _load(f"{BRAX}/logs/sigma_m8/runs/remac_m*_pusher_seed*.log")
    sgd = _load(f"{BRAX}/logs/sgd/remac_sgd_m*_pusher_seed*.log")
    # The Adam M=8 cell is the main run (same config: B=16, eps=1e-8, tuned lr).
    main_m8 = _load(f"{BRAX}/logs/pusher/final_m8.log")
    if not main_m8.empty and (adam.empty or not (adam.remax_m == 8).any()):
        adam = pd.concat([adam, main_m8], ignore_index=True)

    table2()
    table7(adam)
    table8(adam, sgd)
    if args.check:
        cross_check()


if __name__ == "__main__":
    main()
