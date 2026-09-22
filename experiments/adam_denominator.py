"""Isolating Adam's denominator on the ReMax toy problem (rebuttal, Reviewer 22h8).

Reviewer 22h8 objects that sweeping Adam's numerical-stability constant eps does not
isolate adaptive normalization, because the denominator is present for every eps.
This script adds the ablation that does isolate it: EMA-only, i.e. Adam with the
denominator deleted,

    theta_{t+1} = theta_t - alpha * m_hat_t                            (EMA-only)
    theta_{t+1} = theta_t - alpha * m_hat_t / (sqrt(v_hat_t) + eps)    (Adam)
    theta_{t+1} = theta_t - alpha * g_t                                (SGD)

EMA-only keeps beta1 and the bias correction and differs from Adam *only* in the
denominator, so any difference between the two is attributable to adaptive
normalization -- and not to momentum or to the bias-correction terms.

Toy problem (same as the paper): a ~ N(mu, sigma^2), r(a) = -a^2, RP gradient of the
ReMax objective with batch size B, retry budget M, started at (mu, sigma) = (-1.5, 1).

Run from the repository root:
    python experiments/adam_denominator.py --save fig/adam_denominator.pdf
"""

import argparse
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ReMAC"))
from toy.remax import compute_batch_remax  # noqa: E402

jax.config.update("jax_enable_x64", True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def reparam_objective(mu, sigma, key, batch_size, m):
    xi = jax.random.normal(key, shape=(batch_size,))
    actions = mu + sigma * xi
    return compute_batch_remax(-(actions**2), m)


@partial(jax.jit, static_argnames=("batch_size", "m"))
def reparam_grad_remax(mu, sigma, key, batch_size, m):
    dmu, dsigma = jax.grad(reparam_objective, argnums=(0, 1))(mu, sigma, key, batch_size, m)
    return jnp.array([dmu, dsigma])


def simulate(kind, *, lr, eps, steps, m, batch_size, seed, beta1=0.9, beta2=0.999,
             start=(-1.5, 1.0)):
    """kind in {"adam", "ema", "sgd"}.  Returns the distance-to-optimum trajectory."""
    key = jax.random.PRNGKey(seed)
    params = np.array(start, dtype=np.float64)
    mom = np.zeros(2)
    vel = np.zeros(2)
    dist = [float(np.linalg.norm(params))]

    for t in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        g = np.array(reparam_grad_remax(params[0], params[1], subkey, batch_size, m),
                     dtype=np.float64)

        if kind == "sgd":
            delta = lr * g
        else:
            mom = beta1 * mom + (1 - beta1) * g
            m_hat = mom / (1 - beta1**t)
            if kind == "ema":
                # denominator deleted: bias-corrected momentum, no adaptive scaling
                delta = lr * m_hat
            elif kind == "adam":
                vel = beta2 * vel + (1 - beta2) * (g**2)
                v_hat = vel / (1 - beta2**t)
                delta = lr * m_hat / (np.sqrt(v_hat) + eps)
            else:
                raise ValueError(kind)

        params = params + delta           # ascent: g is the gradient of the objective
        params[1] = max(params[1], 1e-4)  # sigma > 0
        dist.append(float(np.linalg.norm(params)))

    return np.array(dist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--save", type=str, default="fig/adam_denominator.pdf")
    args = p.parse_args()

    # (label, kind, eps, color, linestyle)
    arms = [
        (r"Adam $\epsilon=10^{-8}$", "adam", 1e-8, "#0072B2", "-"),
        (r"Adam $\epsilon=10^{-1}$", "adam", 1e-1, "#009E73", "--"),
        (r"Adam $\epsilon=1$", "adam", 1.0, "#D55E00", "-."),
        ("EMA-only (no denominator)", "ema", None, "#CC79A7", "-"),
        ("SGD", "sgd", None, "0.45", ":"),
    ]

    fig, axes = plt.subplots(1, len(args.ms), figsize=(5.0 * len(args.ms), 4.6), squeeze=False)
    axes = axes[0]

    summary = {}
    for ax, m in zip(axes, args.ms):
        for label, kind, eps, color, ls in arms:
            runs = np.stack([
                simulate(kind, lr=args.lr, eps=eps, steps=args.steps, m=m,
                         batch_size=args.batch_size, seed=s)
                for s in range(args.seeds)
            ])
            mean = runs.mean(axis=0)
            se = runs.std(axis=0) / np.sqrt(args.seeds)
            it = np.arange(len(mean))
            ax.plot(it, mean, color=color, linestyle=ls, linewidth=2.6, label=label,
                    marker={"-": "o", "--": "s", "-.": "^", ":": "D"}[ls],
                    markersize=5, markevery=max(1, len(it) // 10), alpha=0.95)
            ax.fill_between(it, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
            summary[(m, label)] = mean[-1]

        ax.set_yscale("log")
        ax.set_xlabel("iteration", fontsize=20)
        ax.set_title(f"M={m}", fontsize=22)
        ax.grid(True, which="both", alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_box_aspect(1)

    axes[0].set_ylabel(r"$\|\theta_t-\theta^\star\|_2$", fontsize=20)
    # One shared legend below the panels: in-axes legends collide with the curves,
    # which sweep through the lower-left corner of every panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14, frameon=False, ncol=len(arms),
               loc="upper center", bbox_to_anchor=(0.5, 0.06))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    fig.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"Saved: {args.save}")

    print("\nFinal distance to the optimum (mean over "
          f"{args.seeds} seeds), lower = converged further:")
    labels = [a[0] for a in arms]
    print(f"{'':30s}" + "".join(f"{'M=' + str(m):>12s}" for m in args.ms))
    for label in labels:
        print(f"{label:30s}" + "".join(f"{summary[(m, label)]:12.2e}" for m in args.ms))

    print("\nSlowdown factor relative to M=1 (final distance ratio); "
          "larger = more M-dependent damping retained:")
    for label in labels:
        ratios = [summary[(m, label)] / summary[(args.ms[0], label)] for m in args.ms]
        print(f"{label:30s}" + "".join(f"{r:12.1f}" for r in ratios))


if __name__ == "__main__":
    main()
