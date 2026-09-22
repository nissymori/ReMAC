"""The per-epsilon M sweeps: Figs. 7-14, drawn from the report frame.

One figure per epsilon, with M = 1, 2, 4, 8 overlaid in every environment panel, so
the effect of the retry budget is directly readable at a fixed optimizer setting.
Colour is therefore free to carry M, and reuses the assignment of Figs. 3 and 4
(M = 1, 2, 4, 8 -> blue, orange, green, red).  A distinct line style and marker per M
is layered on top, giving three redundant channels, so the four curves stay separable
in a small panel and in greyscale.  Each panel holds four curves, not sixteen.

    -> fig/{entropy,return}_vary_m_eps_{1em8,1em2,1em1,1}.pdf

Style is imported from plot.py, so these are drawn with exactly the same rcParams,
M -> colour map and baseline styles as Figs. 3 and 4.

Run:  python analysis/make_eps_sweeps.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import (  # noqa: E402  (shared styling: keeps us consistent with Figs. 3-4)
    BASELINE_STYLES,
    M_TO_TAB10_INDEX,
    _paper_rcparams,
)

DATA = os.environ.get("REMAC_DATA", "data/brax-remax-ac-report.pkl.gz")
OUT = "fig"

DEFAULT_LR = {
    "halfcheetah": 1e-4, "ant": 2e-4, "hopper": 3e-4,
    "reacher": 3e-4, "swimmer": 1e-4, "walker2d": 1e-4, "humanoidstandup": 1e-4,
    "pusher": 2e-4,
}
# The paper's design: M <= 4 -> B = 8, M = 8 -> B = 16 (for eps != 1e-8 the M=8 runs
# only exist at B=16, so this keeps every curve in a figure on a consistent B).
B_PER_M = {1: 8, 2: 8, 4: 8, 8: 16}
MS = [1, 2, 4, 8]
EPSILONS = [1e-8, 1e-2, 1e-1, 1.0]
EPS_LABEL = {1e-8: r"$\epsilon=10^{-8}$", 1e-2: r"$\epsilon=10^{-2}$",
             1e-1: r"$\epsilon=10^{-1}$", 1.0: r"$\epsilon=1$"}
EPS_TAG = {1e-8: "1em8", 1e-2: "1em2", 1e-1: "1em1", 1.0: "1"}
# M -> (linestyle, marker): redundant channels on top of the paper's M -> hue assignment.
M_LINE = {1: ("-", "o"), 2: ("--", "s"), 4: ("-.", "^"), 8: (":", "D")}

# The eight main tasks, in the order the figures show them.
ENV_ORDER = ["ant", "halfcheetah", "hopper", "walker2d", "reacher", "swimmer",
             "humanoidstandup", "pusher"]
NROWS, NCOLS = 2, 4
PRETTY = {"ant": "ant", "halfcheetah": "halfcheetah", "hopper": "hopper",
          "walker2d": "walker2d", "reacher": "reacher", "swimmer": "swimmer",
          "humanoidstandup": "humanoidstandup", "pusher": "pusher"}
TAB10 = plt.get_cmap("tab10")
M_COLOR = {m: TAB10(M_TO_TAB10_INDEX[m] % 10) for m in MS}
LW = 2.2
ALPHA = 0.80


def load():
    d = pd.read_pickle(DATA)
    d["env_s"] = d["env"].str.replace("brax/", "", regex=False)
    return d


def curve(df, metric):
    """Mean +/- standard error over the 10 seeds; one seed = (run_id, seed_id)."""
    df = df[df[metric].notna()]
    df = df[df["_step"] <= 3_000_000]
    if df.empty:
        return None, None, None
    per_seed = df.groupby(["_step", "run_id", "seed_id"])[metric].mean().reset_index()
    g = per_seed.groupby("_step")[metric]
    steps = np.array(sorted(per_seed["_step"].unique()))
    mean = g.mean().reindex(steps).to_numpy()
    n = g.count().reindex(steps).to_numpy()
    se = g.std(ddof=1).reindex(steps).to_numpy() / np.sqrt(np.maximum(n, 1))
    return steps, mean, se


def remac(d, env, m, eps):
    return d[(d.env_s == env) & (d.algorithm == "remax_ac") & (d.remax_m == m)
             & (d.remax_num_samples == B_PER_M[m])
             & np.isclose(d.actor_epsilon, eps)
             & np.isclose(d.learning_rate, DEFAULT_LR[env])]


def draw(ax, x, mu, se, color, ls, marker, lw=LW, alpha=ALPHA):
    ax.plot(x, mu, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
            marker=marker, markersize=4.5,
            markevery=max(1, len(x) // 7) if marker else None)
    ax.fill_between(x, mu - se, mu + se, color=color, alpha=min(0.18, 0.45 * alpha),
                    linewidth=0.0)


def style_axis(ax, env, metric):
    ax.set_title(PRETTY[env])
    ax.set_xlabel("step")
    ax.set_ylabel("entropy" if metric == "train/entropy" else "return")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


# ------------------------------------------- HumanoidStandup on its own
def fig_humanoidstandup(d, eps=1e-8):
    """Same visual language as Figs. 3 and 4: colour = M, solid lines, SAC/PPO baselines."""
    env = "humanoidstandup"
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    for ax, metric in zip(axes, ["eval/return", "train/entropy"]):
        for kind in ["SAC", "PPO"]:
            algo = kind.lower()
            x, mu, se = curve(d[(d.env_s == env) & (d.algorithm == algo)], metric)
            if x is None:      # PPO logs no entropy
                continue
            st = BASELINE_STYLES[kind]
            draw(ax, x, mu, se, st.color, st.linestyle, None, lw=st.linewidth,
                 alpha=st.alpha)
        for m in MS:
            x, mu, se = curve(remac(d, env, m, eps), metric)
            if x is None:
                continue
            draw(ax, x, mu, se, M_COLOR[m], "-", None)
        style_axis(ax, env, metric)

    handles = [
        Line2D([0], [0], color=BASELINE_STYLES["SAC"].color, lw=4.5,
               ls=BASELINE_STYLES["SAC"].linestyle, label="SAC"),
        Line2D([0], [0], color=BASELINE_STYLES["PPO"].color, lw=4.5,
               ls=BASELINE_STYLES["PPO"].linestyle, label="PPO"),
    ] + [Line2D([0], [0], color=M_COLOR[m], lw=4.5, ls="-", label=rf"ReMAC ($M={m}$)")
         for m in MS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = f"{OUT}/humanoidstandup.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ------------------------------------------------------------------ Figs. 7-14
# One figure per epsilon, with M = 1, 2, 4, 8 overlaid in every environment panel, so the
# effect of the retry budget is directly readable at a fixed optimizer setting.  Colour is
# therefore free to carry M, and it reuses the paper's assignment from Figs. 3 and 4
# (M = 1, 2, 4, 8 -> blue, orange, green, red).  A distinct line style and marker per M is
# layered on top, giving three redundant channels, so the four curves remain separable in a
# small panel and in greyscale.  Each panel holds four curves, not sixteen.
def fig_m_for_eps(d, metric, eps):
    """One figure per epsilon, with all four M overlaid (paper hue + linestyle + marker)."""
    # per-panel aspect ratio of the original 2x3 figure, widened to NCOLS columns
    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(12.5 * NCOLS / 3.0, 5.9 * NROWS / 2.0))
    axes = axes.ravel()
    incomplete = []

    for ax, env in zip(axes, ENV_ORDER):
        if metric == "train/entropy":
            x, mu, se = curve(d[(d.env_s == env) & (d.algorithm == "sac")], metric)
            if x is not None:
                st = BASELINE_STYLES["SAC"]
                draw(ax, x, mu, se, st.color, st.linestyle, None, lw=st.linewidth,
                     alpha=st.alpha)
        n_drawn = 0
        for m in MS:
            x, mu, se = curve(remac(d, env, m, eps), metric)
            if x is None:
                continue
            ls, mk = M_LINE[m]
            draw(ax, x, mu, se, M_COLOR[m], ls, mk, alpha=0.95)
            n_drawn += 1
        # A panel is kept only if it has at least one ReMAC curve.  The SAC baseline
        # alone does not count: the figure is about how M behaves at a fixed epsilon,
        # so a panel showing only SAC would read as a result rather than as an
        # environment that was not run at this epsilon.
        if not n_drawn:
            ax.clear()
            ax.axis("off")
            continue
        # A panel with some but not all of M is a partially finished sweep.  The legend
        # still lists every M, so the missing curve reads as an absent result; say so
        # loudly rather than shipping the figure silently.
        if n_drawn < len(MS):
            incomplete.append(f"{env} ({n_drawn}/{len(MS)} M)")
        style_axis(ax, env, metric)
    for ax in axes[len(ENV_ORDER):]:
        ax.axis("off")

    handles = ([Line2D([0], [0], color=BASELINE_STYLES["SAC"].color, lw=4.5,
                       ls=BASELINE_STYLES["SAC"].linestyle, label="SAC")]
               if metric == "train/entropy" else [])
    handles += [Line2D([0], [0], color=M_COLOR[m], lw=3.2, ls=M_LINE[m][0],
                       marker=M_LINE[m][1], markersize=6, label=rf"ReMAC ($M={m}$)")
                for m in MS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(EPS_LABEL[eps], y=1.0, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    if incomplete:
        print(f"  WARNING: partially finished panels: {', '.join(incomplete)}")

    kind = "entropy" if metric == "train/entropy" else "return"
    out = f"{OUT}/{kind}_vary_m_eps_{EPS_TAG[eps]}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    _paper_rcparams()
    os.makedirs(OUT, exist_ok=True)
    d = load()
    fig_humanoidstandup(d)
    for eps in EPSILONS:                       # Figs. 8-11
        fig_m_for_eps(d, "train/entropy", eps)
    for eps in EPSILONS:                       # Figs. 12-15
        fig_m_for_eps(d, "eval/return", eps)


if __name__ == "__main__":
    main()
