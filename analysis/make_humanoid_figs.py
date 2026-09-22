"""HumanoidStandup entropy/return curves, in the same visual language as the other tasks.

HumanoidStandup is currently reported only as tables (Tabs. 4 and 5), while every other
task is also shown as a learning curve.  This script closes that gap without touching any
figure already included in the paper: it writes only new filenames.

Style (rcParams, M -> hue, baseline styles) is imported from plot.py, and the M encoding
(hue + linestyle + marker) is the one make_cxt2_figs.py uses for the epsilon sweeps, so the
panels read exactly like Figs. 3, 4 and 7-14.

Outputs
-------
fig/humanoidstandup.pdf                      return + entropy at Adam's default eps=1e-8,
                                             with the SAC and PPO baselines (matches Figs. 3, 4)
fig/entropy_humanoidstandup_vary_eps.pdf     entropy, one panel per eps (matches Figs. 7-10)
fig/return_humanoidstandup_vary_eps.pdf      return,  one panel per eps (matches Figs. 11-14)

Note: HumanoidStandup was originally run at eps in {1e-8, 1e-1, 1} only.  The missing
eps=1e-2 arm was added for the camera-ready version (experiments/sh/humanoid_eps.sh), so the
sweep now covers all four values, as it does for every other task.

Run:  python analysis/make_humanoid_figs.py
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import (  # noqa: E402  (shared styling: keeps us consistent with Figs. 3-4)
    BASELINE_STYLES,
    _paper_rcparams,
)
from make_cxt2_figs import (  # noqa: E402  (shared data handling: same pkl, same filters)
    EPS_LABEL,
    MS,
    M_COLOR,
    M_LINE,
    OUT,
    curve,
    draw,
    load,
    remac,
)

ENV = "humanoidstandup"
# All four epsilons, since the eps=1e-2 arm was filled in for the camera-ready version.
HUMANOID_EPSILONS = [1e-8, 1e-2, 1e-1, 1.0]


def _style(ax, metric, title):
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("entropy" if metric == "train/entropy" else "return")
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def fig_default_eps(d, eps=1e-8):
    """Return and entropy at Adam's default eps, with SAC/PPO -- the Figs. 3 and 4 encoding.

    Colour carries M and every ReMAC line is solid, exactly as in the main-text figures;
    no markers, since a single environment leaves the four curves easy to separate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))

    for ax, metric in zip(axes, ["eval/return", "train/entropy"]):
        for kind in ["SAC", "PPO"]:
            x, mu, se = curve(d[(d.env_s == ENV) & (d.algorithm == kind.lower())], metric)
            if x is None:      # PPO logs no entropy
                continue
            st = BASELINE_STYLES[kind]
            draw(ax, x, mu, se, st.color, st.linestyle, None, lw=st.linewidth, alpha=st.alpha)
        for m in MS:
            x, mu, se = curve(remac(d, ENV, m, eps), metric)
            if x is None:
                continue
            draw(ax, x, mu, se, M_COLOR[m], "-", None)
        _style(ax, metric, ENV)

    handles = [
        Line2D([0], [0], color=BASELINE_STYLES[k].color, lw=4.5,
               ls=BASELINE_STYLES[k].linestyle, label=k) for k in ["SAC", "PPO"]
    ] + [Line2D([0], [0], color=M_COLOR[m], lw=4.5, ls="-", label=rf"ReMAC ($M={m}$)")
         for m in MS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    out = f"{OUT}/{ENV}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_vary_eps(d, metric):
    """One panel per eps, with M = 1, 2, 4, 8 overlaid -- the Figs. 7-14 encoding.

    Same three redundant channels per M (paper hue + linestyle + marker) as the six-task
    sweeps, so a panel here is directly comparable with a panel there.
    """
    n = len(HUMANOID_EPSILONS)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.0))
    axes = [axes] if n == 1 else list(axes)

    for ax, eps in zip(axes, HUMANOID_EPSILONS):
        if metric == "train/entropy":
            x, mu, se = curve(d[(d.env_s == ENV) & (d.algorithm == "sac")], metric)
            if x is not None:
                st = BASELINE_STYLES["SAC"]
                draw(ax, x, mu, se, st.color, st.linestyle, None, lw=st.linewidth,
                     alpha=st.alpha)
        for m in MS:
            x, mu, se = curve(remac(d, ENV, m, eps), metric)
            if x is None:
                continue
            ls, mk = M_LINE[m]
            draw(ax, x, mu, se, M_COLOR[m], ls, mk, alpha=0.95)
        _style(ax, metric, EPS_LABEL[eps])

    handles = ([Line2D([0], [0], color=BASELINE_STYLES["SAC"].color, lw=4.5,
                       ls=BASELINE_STYLES["SAC"].linestyle, label="SAC")]
               if metric == "train/entropy" else [])
    handles += [Line2D([0], [0], color=M_COLOR[m], lw=3.2, ls=M_LINE[m][0],
                       marker=M_LINE[m][1], markersize=6, label=rf"ReMAC ($M={m}$)")
                for m in MS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(ENV, y=1.02, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    kind = "entropy" if metric == "train/entropy" else "return"
    out = f"{OUT}/{kind}_{ENV}_vary_eps.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    _paper_rcparams()
    os.makedirs(OUT, exist_ok=True)
    d = load()
    fig_default_eps(d)
    fig_vary_eps(d, "train/entropy")
    fig_vary_eps(d, "eval/return")


if __name__ == "__main__":
    main()
