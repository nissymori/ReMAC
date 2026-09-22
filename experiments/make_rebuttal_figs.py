"""Rebuttal figures for the TMLR ReMAC submission.

Self-contained: reads data/remac_rebuttal_data.pkl and writes 4 PDFs to fig/.

    python experiments/make_rebuttal_figs.py

Figures
    fig/rebuttal_coverage.pdf    state-space coverage, ReMAC (M=1,2,4) vs SAC   (reviewer Lak5)
    fig/rebuttal_sgd.pdf         ReMAC with a plain-SGD actor vs the Adam actor (reviewer cxT2)
    fig/rebuttal_damping.pdf     gradient damping: ||g|| and ||dtheta|| vs M     (reviewer cxT2)
    fig/rebuttal_sigma_grad.pdf  entropy-increase effect: sigma_grad, policy_std (reviewer cxT2)

Style follows analysis/plot.py (serif/Times, Type-42 fonts, mean +- s.e. shading, per-env
subplot grid), but the sequential colormap is replaced by the Okabe-Ito colorblind-safe
palette with distinct line styles and markers (reviewer cxT2 asked for this).
"""

import os
import pickle
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKL = os.path.join(HERE, "data", "remac_rebuttal_data.pkl")
FIGDIR = os.path.join(ROOT, "fig")

ENVS: Tuple[str, ...] = ("halfcheetah", "ant", "hopper")
MS: Tuple[int, ...] = (1, 2, 4)

# --- M -> colour exactly as in the paper's Figs. 3 and 4 (plot.py: tab10 indexed by
# --- DEFAULT_M_COLOR_ORDER), so these figures match the existing ones.  A distinct
# --- marker and linestyle per M is layered on top, which keeps the curves separable
# --- in greyscale and for colour-vision-deficient readers.
M_COLOR: Dict[int, str] = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c"}  # blue / orange / green
M_MARKER: Dict[int, str] = {1: "o", 2: "s", 4: "^"}
M_LS: Dict[int, str] = {1: "-", 2: "--", 4: "-."}
SAC_COLOR = "#000000"
SAC_MARKER = "D"
# optimizer encoded by line style when both arms are on the same axes
OPT_LS: Dict[str, str] = {"adam": "-", "sgd": ":"}


def _rcparams() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 12,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.5,
        }
    )


def load() -> dict:
    with open(PKL, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- curve helpers
def select(runs: List[dict], **kw) -> List[dict]:
    out = runs
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def mean_se(runs: List[dict], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Mean +- standard error across runs (n=5 processes; each is the mean of its 2 vmapped seeds)."""
    key = f"mean/{metric}"
    keep = [r for r in runs if key in r["curves"]]
    if not keep:
        raise KeyError(f"no runs carry {key}")
    n = min(len(r["curves"][key]) for r in keep)
    y = np.stack([np.asarray(r["curves"][key], dtype=float)[:n] for r in keep])
    x = np.asarray(keep[0]["steps"][key], dtype=float)[:n]
    m = np.nanmean(y, axis=0)
    se = np.nanstd(y, axis=0, ddof=1) / np.sqrt(y.shape[0])
    return x, m, se, y.shape[0]


def band(ax, x, m, se, color, ls, marker, label, lw=2.0, alpha=0.95):
    ax.plot(
        x, m, color=color, linestyle=ls, linewidth=lw, alpha=alpha, label=label,
        marker=marker, markevery=max(1, len(x) // 8), markersize=4.5, markeredgewidth=0.0,
    )
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, linewidth=0.0)


def finish(ax, xlabel="environment steps", ylabel=None, title=None):
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def save(fig, name: str):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}  ({os.path.getsize(path)/1024:.1f} kB)")


# ================================================================ 1) coverage
def fig_coverage(d: dict):
    runs = [r for r in d["runs"] if "coverage" in r["project"]]
    metrics = [
        ("coverage/marginal_frac", "marginal coverage"),
        ("coverage/joint_frac", "joint coverage (dims 1-2)"),
        ("coverage/state_entropy_norm", "normalised state entropy"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(13.0, 12.0))

    # rows 0-2: learning curves of each coverage metric
    for i, (metric, mlabel) in enumerate(metrics):
        for j, env in enumerate(ENVS):
            ax = axes[i, j]
            sac = select(runs, env=env, algorithm="sac")
            x, m, se, n = mean_se(sac, metric)
            band(ax, x, m, se, SAC_COLOR, "-", SAC_MARKER, f"SAC (n={n})", lw=2.4)
            for mm in MS:
                rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm)
                x, m, se, n = mean_se(rs, metric)
                band(ax, x, m, se, M_COLOR[mm], M_LS[mm], M_MARKER[mm], f"ReMAC $M={mm}$")
            finish(ax, ylabel=mlabel if j == 0 else None, title=env if i == 0 else None)

    # row 3: grouped bar chart of final values (mean of last 10% of the curve), n=5 runs, +- s.d.
    methods = [("SAC", None), ("ReMAC $M=1$", 1), ("ReMAC $M=2$", 2), ("ReMAC $M=4$", 4)]
    t = d["tidy"]
    t = t[t["project"].str.contains("coverage")]
    width = 0.2
    for j, env in enumerate(ENVS):
        ax = axes[3, j]
        xs = np.arange(len(metrics))
        for k, (label, mm) in enumerate(methods):
            sub = t[t["env"] == env]
            sub = sub[sub["algorithm"] == "sac"] if mm is None else sub[
                (sub["algorithm"] == "remax_ac") & (sub["remax_m"] == mm)
            ]
            # aggregate to run level (n=5) to match the reported summary table
            means, sds = [], []
            for metric, _ in metrics:
                v = sub.groupby("run_name")[f"final/{metric}"].mean().to_numpy()
                means.append(v.mean())
                sds.append(v.std(ddof=1))
            color = SAC_COLOR if mm is None else M_COLOR[mm]
            ax.bar(
                xs + (k - 1.5) * width, means, width, yerr=sds, capsize=3,
                color=color, alpha=0.55 if mm is None else 0.85,
                edgecolor=color, linewidth=1.2, hatch={0: "", 1: "", 2: "//", 3: "xx"}[k],
                label=label, error_kw={"elinewidth": 1.0},
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(["marginal", "joint", "entropy"], rotation=0)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("final value (mean of last 10%)\n$\\pm$ s.d. over $n=5$ runs")

    handles = [Line2D([0], [0], color=SAC_COLOR, lw=3, ls="-", marker=SAC_MARKER, label="SAC")]
    handles += [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls=M_LS[m], marker=M_MARKER[m], label=rf"ReMAC $M={m}$")
        for m in MS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "State-space coverage: ReMAC vs SAC (curves: mean $\\pm$ s.e. over $n=5$ runs / 10 seeds).\n"
        "Coverage saturates on these dense-reward tasks; ReMAC does not increase it.",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save(fig, "rebuttal_coverage.pdf")


# ================================================================ 2) SGD vs Adam
def fig_sgd(d: dict):
    runs = d["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for j, env in enumerate(ENVS):
        ax = axes[j]
        for mm in MS:
            for opt in ("adam", "sgd"):
                rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm, actor_optimizer=opt)
                x, m, se, n = mean_se(rs, "eval/return")
                band(
                    ax, x, m, se, M_COLOR[mm], OPT_LS[opt], M_MARKER[mm],
                    rf"$M={mm}$ ({opt})", lw=2.2 if opt == "adam" else 2.4,
                    alpha=0.95 if opt == "adam" else 0.85,
                )
        finish(ax, ylabel="eval return" if j == 0 else None, title=env)

    handles = [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls="-", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (Adam)")
        for m in MS
    ] + [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls=":", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (SGD)")
        for m in MS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(
        "ReMAC with a plain-SGD actor optimiser (dotted) vs the Adam actor (solid). "
        "Mean $\\pm$ s.e. over $n=5$ runs (10 seeds).\n"
        "Same seeds, envs, $M$ and learning rate; learning rates were tuned for Adam and reused unchanged for SGD.",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()
    save(fig, "rebuttal_sgd.pdf")


# ================================================================ 3) damping
def fig_damping(d: dict):
    runs = d["runs"]
    rows = [
        ("train/actor_grad_norm", "actor gradient norm $\\|g\\|$"),
        ("train/actor_update_norm", "actor update norm $\\|\\Delta\\theta\\|$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    for i, (metric, ylabel) in enumerate(rows):
        for j, env in enumerate(ENVS):
            ax = axes[i, j]
            for mm in MS:
                for opt in ("adam", "sgd"):
                    rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm, actor_optimizer=opt)
                    x, m, se, n = mean_se(rs, metric)
                    band(
                        ax, x, m, se, M_COLOR[mm], OPT_LS[opt], M_MARKER[mm],
                        None, lw=2.2, alpha=0.95 if opt == "adam" else 0.85,
                    )
            ax.set_yscale("log")
            finish(ax, ylabel=ylabel if j == 0 else None, title=env if i == 0 else None)

    handles = [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls="-", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (Adam)")
        for m in MS
    ] + [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls=":", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (SGD)")
        for m in MS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09))
    # No suptitle: the LaTeX caption carries the description.
    fig.tight_layout()
    save(fig, "rebuttal_damping.pdf")


# ================================================================ 4) sigma grad
def _smooth(y: np.ndarray, w: int = 9) -> np.ndarray:
    """Centred moving average; the raw per-eval sigma_grad flips sign at every eval point."""
    if w <= 1:
        return y
    k = np.ones(w) / w
    pad = w // 2
    ypad = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(ypad, k, mode="valid")[: len(y)]


def fig_sigma(d: dict):
    runs = d["runs"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 10.6))
    for j, env in enumerate(ENVS):
        # --- row 0: sigma_grad, smoothed (negative => the update pushes sigma / entropy UP)
        ax = axes[0, j]
        for mm in MS:
            for opt in ("adam", "sgd"):
                rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm, actor_optimizer=opt)
                x, m, se, n = mean_se(rs, "train/sigma_grad")
                band(ax, x, _smooth(m), _smooth(se), M_COLOR[mm], OPT_LS[opt], M_MARKER[mm], None,
                     lw=2.2, alpha=0.95 if opt == "adam" else 0.85)
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=0)
        ax.set_yscale("symlog", linthresh=1e-6)
        finish(
            ax,
            ylabel="$\\sigma$-gradient (smoothed)\n$\\partial\\mathcal{L}_\\pi/\\partial b_{\\log\\sigma}$"
            if j == 0 else None,
            title=env,
        )

        # --- row 1: run-mean sigma_grad, mean +- s.e. over n=5 runs (the quantitative version of row 0)
        ax = axes[1, j]
        width = 0.35
        for k, opt in enumerate(("adam", "sgd")):
            for i, mm in enumerate(MS):
                rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm, actor_optimizer=opt)
                v = np.array([np.mean(r["curves"]["mean/train/sigma_grad"]) for r in rs])
                ax.bar(
                    i + (k - 0.5) * width, v.mean(), width,
                    yerr=v.std(ddof=1) / np.sqrt(len(v)), capsize=3,
                    color=M_COLOR[mm], alpha=0.9 if opt == "adam" else 0.45,
                    edgecolor=M_COLOR[mm], linewidth=1.2, hatch="" if opt == "adam" else "///",
                    error_kw={"elinewidth": 1.0},
                )
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=0)
        ax.set_yscale("symlog", linthresh=1e-7)
        ax.set_xticks(range(len(MS)))
        ax.set_xticklabels([rf"$M={m}$" for m in MS])
        ax.set_xlabel("Adam (solid) / SGD (hatched)")
        ax.grid(True, axis="y", alpha=0.25)
        if j == 0:
            ax.set_ylabel("mean $\\sigma$-gradient over training\n$\\pm$ s.e. ($n=5$ runs)")

        # --- row 2: realised policy_std
        ax = axes[2, j]
        for mm in MS:
            for opt in ("adam", "sgd"):
                rs = select(runs, env=env, algorithm="remax_ac", remax_m=mm, actor_optimizer=opt)
                x, m, se, n = mean_se(rs, "train/policy_std")
                band(ax, x, m, se, M_COLOR[mm], OPT_LS[opt], M_MARKER[mm], None, lw=2.2,
                     alpha=0.95 if opt == "adam" else 0.85)
        ax.set_yscale("log")
        finish(ax, ylabel="policy $\\sigma$" if j == 0 else None)

    handles = [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls="-", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (Adam)")
        for m in MS
    ] + [
        Line2D([0], [0], color=M_COLOR[m], lw=3, ls=":", marker=M_MARKER[m], label=rf"ReMAC $M={m}$ (SGD)")
        for m in MS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        "Entropy-increase effect. Row 1: $\\sigma$-gradient (symlog, grey line $=0$, moving average over 9 evals; "
        "the raw signal flips sign between evals).\nThe loss is $-$objective, so negative $\\Rightarrow$ the update pushes "
        "$\\sigma$ up. Row 2: its mean over training. Row 3: the realised policy $\\sigma$.\n"
        "Larger $M$ gives a monotonically more negative $\\sigma$-gradient and a policy $\\sigma$ orders of magnitude larger.",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    save(fig, "rebuttal_sigma_grad.pdf")


def main():
    _rcparams()
    d = load()
    fig_coverage(d)
    fig_sgd(d)
    fig_damping(d)
    fig_sigma(d)


if __name__ == "__main__":
    main()
