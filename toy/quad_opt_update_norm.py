# remax_paths_toy.py
import argparse
import os
import re
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
})

try:
    from toy.remax import compute_batch_remax
except ModuleNotFoundError:
    from remax import compute_batch_remax

jax.config.update("jax_enable_x64", True)

# -----------------------------
# Toy env / objective
# -----------------------------
def quad(x: jnp.ndarray) -> jnp.ndarray:
    return -x**2


def sample_actions(key: jax.Array, mu: jnp.ndarray, sigma: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    eps = jax.random.normal(key, shape=(batch_size,))
    return mu + sigma * eps


def reparam_objective(mu: jnp.ndarray, sigma: jnp.ndarray, key: jax.Array, batch_size: int, m: int) -> jnp.ndarray:
    actions = sample_actions(key, mu, sigma, batch_size)
    returns = quad(actions)
    return compute_batch_remax(returns, m)


# -----------------------------
# Reparam gradient (JAX autodiff)
# -----------------------------
@partial(jax.jit, static_argnames=("batch_size", "m"))
def reparam_grad_remax(mu, sigma, key, batch_size, m):
    grad_fn = jax.grad(reparam_objective, argnums=(0, 1))
    dmu, dsigma = grad_fn(mu, sigma, key, batch_size, m)
    return jnp.array([dmu, dsigma])


# -----------------------------
# Optimizers
# -----------------------------
class SGD:
    def __init__(self, lr: float):
        self.lr = lr

    def update(self, params, grads):
        return self.lr * grads


class Adam:
    def __init__(self, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        """
        Returns:
          delta: update amount (same shape as params)
          m_hat: bias-corrected first moment
          v_hat: bias-corrected second moment
          precond: m_hat / (sqrt(v_hat) + eps)   (i.e., delta / lr)
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads**2)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        precond = m_hat / (np.sqrt(v_hat) + self.eps)
        delta = self.lr * precond
        return delta, m_hat, v_hat, precond


# -----------------------------
# Simulation
# -----------------------------
def simulate_path(
    optimizer: str,
    lr: float,
    steps: int,
    start_mu: float,
    start_sigma: float,
    batch_size: int,
    m: int,
    seed: int,
    actor_epsilon: float,
    beta1: float,
    beta2: float,
):
    key = jax.random.PRNGKey(seed)
    params = np.array([start_mu, start_sigma], dtype=np.float64)

    optimizer = optimizer.lower()
    if optimizer == "sgd":
        opt = SGD(lr)
        is_adam = False
    elif optimizer == "adam":
        opt = Adam(lr, beta1=beta1, beta2=beta2, eps=actor_epsilon)
        is_adam = True
    else:
        raise ValueError(f"Unknown optimizer: {optimizer} (use 'sgd' or 'adam')")

    mu_hist = [params[0]]
    sigma_hist = [params[1]]
    dmu_hist, dsigma_hist, dtheta_norm_hist = [], [], []
    grad_norm_hist = []

    mhat_hist = []
    vhat_hist = []
    precond_hist = []

    for _ in range(steps):
        mu, sigma = params

        key, subkey = jax.random.split(key)
        grads = reparam_grad_remax(mu, sigma, subkey, batch_size, m)
        grads_np = np.array(grads, dtype=np.float64)
        grad_norm_hist.append(float(np.linalg.norm(grads_np)))

        if is_adam:
            delta, m_hat, v_hat, precond = opt.step(params, grads_np)
            mhat_hist.append(m_hat.copy())
            vhat_hist.append(v_hat.copy())
            precond_hist.append(precond.copy())
        else:
            delta = opt.update(params, grads_np)

        new_params = params + delta
        new_params[1] = max(new_params[1], 1e-4)

        dmu = new_params[0] - params[0]
        dsigma = new_params[1] - params[1]
        dtheta_norm = float(np.sqrt(dmu**2 + dsigma**2))

        dmu_hist.append(dmu)
        dsigma_hist.append(dsigma)
        dtheta_norm_hist.append(dtheta_norm)

        params = new_params
        mu_hist.append(params[0])
        sigma_hist.append(params[1])

    out = {
        "mu": np.array(mu_hist),
        "sigma": np.array(sigma_hist),
        "dmu": np.array(dmu_hist),
        "dsigma": np.array(dsigma_hist),
        "dnorm": np.array(dtheta_norm_hist),
        "grad_norm": np.array(grad_norm_hist),
        "is_adam": is_adam,
    }
    if is_adam:
        out["mhat"] = np.array(mhat_hist)
        out["vhat"] = np.array(vhat_hist)
        out["precond"] = np.array(precond_hist)

    out["dist"] = np.sqrt(out["mu"] ** 2 + out["sigma"] ** 2)
    return out


# -----------------------------
# Parsing helpers
# -----------------------------
def _parse_pair(csv: str):
    xs = [float(x) for x in csv.split(",") if x.strip()]
    if len(xs) != 2:
        raise ValueError("Expected two comma-separated floats, e.g. -1.8,0.2")
    return (xs[0], xs[1])


def _parse_float_list(csv: str):
    toks = [t for t in re.split(r"[,\s]+", csv.strip()) if t]
    xs = [float(t) for t in toks]
    if len(xs) == 0:
        raise ValueError("Expected at least one float, e.g. 1e-8 or 1e-8,1e-6,1e-4")
    return xs


def _eps_tag(eps: float) -> str:
    return f"{eps:.0e}".replace("+", "")


def _derive_sibling_savepath(savepath: str, suffix: str) -> str:
    root, ext = os.path.splitext(savepath)
    if not ext:
        ext = ".pdf"
    return f"{root}_{suffix}{ext}"


# -----------------------------
# Plot: multi epsilon (ADAM runs) + SGD baseline (gray)
# -----------------------------
def plot_results_multi(
    results_adam,
    actor_epsilons,
    *,
    m: int,
    title: str,
    savepath: str,
    xlim=None,
    ylim=None,
    add_contours=True,
    plot_deltas=True,
    sgd_baseline=None,
    plot_legend=True,
):
    if len(results_adam) == 0:
        raise ValueError("results_adam must be non-empty")

    steps = len(results_adam[0]["dmu"])
    it_params = np.arange(steps + 1)
    it_delta = np.arange(steps)

    assert results_adam[0]["is_adam"], "plot_results_multi expects Adam results (is_adam=True)."

    fig = plt.figure(figsize=(13, 20))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.2, 1.0, 1.1, 1.0, 0.9])

    ax_path = fig.add_subplot(gs[0, 0])
    ax_param = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_norm = fig.add_subplot(gs[1, 1])
    ax_mhat = fig.add_subplot(gs[2, 0])
    ax_vhat = fig.add_subplot(gs[2, 1])
    ax_pre = fig.add_subplot(gs[3, :])
    ax_dist = fig.add_subplot(gs[4, :])

    if add_contours:
        all_mu = np.concatenate([res["mu"] for res in results_adam] + ([sgd_baseline["mu"]] if sgd_baseline else []))
        all_sigma = np.concatenate([res["sigma"] for res in results_adam] + ([sgd_baseline["sigma"]] if sgd_baseline else []))
        x_min = (np.min(all_mu) - 0.5) if xlim is None else xlim[0]
        x_max = (np.max(all_mu) + 0.5) if xlim is None else xlim[1]
        y_min = (np.min(all_sigma) - 0.3) if ylim is None else ylim[0]
        y_max = (np.max(all_sigma) + 0.3) if ylim is None else ylim[1]

        x = np.linspace(x_min, x_max, 120)
        y = np.linspace(max(0.0, y_min), y_max, 120)
        X, Y = np.meshgrid(x, y)
        Z = -(X**2 + Y**2)
        ax_path.contourf(X, Y, Z, levels=18, cmap="Greys", alpha=0.3)
        ax_path.contour(X, Y, Z, levels=18, colors="k", alpha=0.12, linewidths=0.6)

    # ---- SGD baseline ----
    if sgd_baseline is not None:
        mu_hist = sgd_baseline["mu"]
        sigma_hist = sgd_baseline["sigma"]
        dmu_hist = sgd_baseline["dmu"]
        dsigma_hist = sgd_baseline["dsigma"]
        dtheta_norm_hist = sgd_baseline["dnorm"]
        dist_hist = sgd_baseline["dist"]

        ax_path.plot(mu_hist, sigma_hist, color="0.5", linewidth=2.6, alpha=0.95, label="SGD")
        ax_path.plot(mu_hist[0], sigma_hist[0], "o", color="0.5", markersize=5, alpha=0.95)

        ax_param.plot(it_params, mu_hist, color="0.5", linewidth=2.2, alpha=0.8, label=r"$\mu_t$ (SGD)")
        ax_param.plot(it_params, sigma_hist, color="0.5", linestyle=":", linewidth=2.2, alpha=0.8, label=r"$\sigma_t$ (SGD)")

        if plot_deltas:
            ax_delta.plot(it_delta, dmu_hist, color="0.5", linewidth=2.0, alpha=0.8, label=r"$\Delta\mu_t$ (SGD)")
            ax_delta.plot(it_delta, dsigma_hist, color="0.5", linestyle=":", linewidth=2.0, alpha=0.8, label=r"$\Delta\sigma_t$ (SGD)")

        ax_norm.plot(it_delta, dtheta_norm_hist, color="0.5", linewidth=2.4, alpha=0.85, label="SGD")
        ax_dist.plot(it_params, dist_hist, color="0.5", linewidth=2.4, alpha=0.85, label="SGD")

    # ---- Adam multi-eps ----
    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "+", "v", "<", ">", "*"]

    for i, (eps, res) in enumerate(zip(actor_epsilons, results_adam)):
        color = cmap(i % 10)
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        tag = rf"Adam $\epsilon={eps:g}$"

        mu_hist = res["mu"]
        sigma_hist = res["sigma"]
        dmu_hist = res["dmu"]
        dsigma_hist = res["dsigma"]
        dtheta_norm_hist = res["dnorm"]
        grad_norm_hist = res["grad_norm"]
        dist_hist = res["dist"]

        ax_path.plot(mu_hist, sigma_hist, color=color, linestyle=ls, linewidth=2.0, alpha=0.95, label=tag)
        ax_path.plot(mu_hist[0], sigma_hist[0], marker=mk, color=color, markersize=5, alpha=0.95)

        ax_param.plot(it_params, mu_hist, color=color, linestyle=ls, linewidth=1.8, label=rf"$\mu_t$ ({tag})")
        ax_param.plot(it_params, sigma_hist, color=color, linestyle=":", linewidth=1.8, label=rf"$\sigma_t$ ({tag})")

        if plot_deltas:
            ax_delta.plot(it_delta, dmu_hist, color=color, linestyle=ls, linewidth=1.4, label=rf"$\Delta\mu_t$ ({tag})")
            ax_delta.plot(it_delta, dsigma_hist, color=color, linestyle=":", linewidth=1.4, label=rf"$\Delta\sigma_t$ ({tag})")
        else:
            ax_delta.axis("off")

        ax_norm.plot(it_delta, dtheta_norm_hist, color=color, linestyle=ls, linewidth=2.0, label=tag)
        ax_dist.plot(it_params, dist_hist, color=color, linestyle=ls, linewidth=2.0, label=tag)

        mhat = res["mhat"]
        vhat = res["vhat"]
        ax_mhat.plot(it_delta, mhat[:, 0], color=color, linestyle=ls, linewidth=1.4, label=rf"$\hat m^\mu$ ({tag})")
        ax_mhat.plot(it_delta, mhat[:, 1], color=color, linestyle=":", linewidth=1.4, label=rf"$\hat m^\sigma$ ({tag})")

        ax_vhat.plot(it_delta, vhat[:, 0], color=color, linestyle=ls, linewidth=1.4, label=rf"$\hat v^\mu$ ({tag})")
        ax_vhat.plot(it_delta, vhat[:, 1], color=color, linestyle=":", linewidth=1.4, label=rf"$\hat v^\sigma$ ({tag})")

        ax_pre.plot(it_delta, grad_norm_hist, color=color, linestyle=ls, linewidth=2.0, label=tag)

    # Decorate
    ax_path.plot(0, 0, "rx", markersize=10, label="Optimum")
    ax_path.set_xlabel(r"$\mu$")
    ax_path.set_ylabel(r"$\sigma$")
    ax_path.set_title("Optimization path")
    ax_path.grid(True, alpha=0.25)
    if xlim is not None:
        ax_path.set_xlim(xlim)
    if ylim is not None:
        ax_path.set_ylim(ylim)
    if plot_legend:
        ax_path.legend(loc="best", fontsize="small", framealpha=0.9)

    ax_param.set_xlabel("iteration")
    ax_param.set_ylabel("value")
    ax_param.set_title("Parameters")
    ax_param.grid(True, alpha=0.25)
    if plot_legend:
        ax_param.legend(fontsize="small", ncol=2)

    if plot_deltas:
        ax_delta.set_xlabel("iteration")
        ax_delta.set_ylabel("update")
        ax_delta.set_title("Updates")
        ax_delta.grid(True, alpha=0.25)
        if plot_legend:
            ax_delta.legend(fontsize="small", ncol=2)

    ax_norm.set_xlabel("iteration")
    ax_norm.set_ylabel("norm")
    ax_norm.set_title(r"$\|\Delta\theta_t\|_2$")
    ax_norm.grid(True, alpha=0.25)
    if plot_legend:
        ax_norm.legend(fontsize="small")

    ax_mhat.set_xlabel("iteration")
    ax_mhat.set_ylabel("value")
    ax_mhat.set_title(r"$\hat m_t$")
    ax_mhat.grid(True, alpha=0.25)
    if plot_legend:
        ax_mhat.legend(fontsize="small", ncol=2)

    ax_vhat.set_xlabel("iteration")
    ax_vhat.set_ylabel("value")
    ax_vhat.set_title(r"$\hat v_t$")
    ax_vhat.grid(True, alpha=0.25)
    if plot_legend:
        ax_vhat.legend(fontsize="small", ncol=2)

    ax_pre.set_xlabel("iteration")
    ax_pre.set_ylabel("norm")
    ax_pre.set_title(r"$\|\nabla_\theta J_t\|_2$")
    ax_pre.grid(True, alpha=0.25)
    if plot_legend:
        ax_pre.legend(fontsize="small", ncol=2)

    ax_dist.set_xlabel("iteration")
    ax_dist.set_ylabel(r"$\|\theta_t-\theta^\star\|_2$")
    ax_dist.set_title("Distance")
    ax_dist.grid(True, alpha=0.25)
    ax_dist.set_yscale("log")
    if plot_legend:
        ax_dist.legend(fontsize="small")

    fig.suptitle(r"$M = $" + f"{m}" + "  |  " + title, fontsize=13)
    plt.tight_layout()

    out_dir = os.path.dirname(savepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.close(fig)


# -----------------------------
# Plot: distance-only
# -----------------------------
def plot_distance_only(results_adam, actor_epsilons, *, m: int, savepath: str, sgd_baseline=None, plot_legend=True):
    fig, ax = plt.subplots(figsize=(5.7, 5))

    if sgd_baseline is not None:
        it_params = np.arange(len(sgd_baseline["dist"]))
        ax.plot(it_params, sgd_baseline["dist"], color="0.5", linewidth=2.8, alpha=0.95, label="SGD")

    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "+", "v", "<", ">", "*"]

    for i, (eps, res) in enumerate(zip(actor_epsilons, results_adam)):
        color = cmap(i % 10)
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        it_params = np.arange(len(res["dist"]))
        ax.plot(
            it_params,
            res["dist"],
            color=color,
            linestyle=ls,
            linewidth=3.0,
            marker=mk,
            markersize=5,
            markevery=max(1, len(it_params) // 12),
            label=rf"Adam $\epsilon={eps:g}$",
            alpha=0.95,
        )

    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=20)
    ax.set_ylabel(r"$\|\theta_t-\theta^\star\|_2$", fontsize=20)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax.set_title(f"M={m}", fontsize=22)
    ax.grid(True, which="both", alpha=0.25)
    if plot_legend:
        ax.legend(fontsize=14, ncol=1, frameon=False)

    fig.tight_layout()
    out_dir = os.path.dirname(savepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.close(fig)


# -----------------------------
# Plot: optimization-path-only (mu vs sigma)
# -----------------------------
def optimization_path_only(
    results_adam,
    actor_epsilons,
    *,
    m: int,
    savepath: str,
    plot_legend: bool = True,
    sgd_baseline=None,
):
    if len(results_adam) == 0:
        raise ValueError("results_adam must be non-empty")

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    # background contours
    all_mu = np.concatenate([res["mu"] for res in results_adam] + ([sgd_baseline["mu"]] if sgd_baseline else []))
    all_sigma = np.concatenate([res["sigma"] for res in results_adam] + ([sgd_baseline["sigma"]] if sgd_baseline else []))

    x_min = np.min(all_mu) - 0.6
    x_max = np.max(all_mu) + 0.6
    y_min = max(0.0, np.min(all_sigma) - 0.35)
    y_max = np.max(all_sigma) + 0.35

    x = np.linspace(x_min, x_max, 140)
    y = np.linspace(y_min, y_max, 140)
    X, Y = np.meshgrid(x, y)
    Z = -(X**2 + Y**2)
    ax.contourf(X, Y, Z, levels=18, cmap="Greys", alpha=0.25)
    ax.contour(X, Y, Z, levels=18, colors="k", alpha=0.12, linewidths=0.6)

    # SGD
    if sgd_baseline is not None:
        mu_hist = sgd_baseline["mu"]
        sigma_hist = sgd_baseline["sigma"]
        ax.plot(mu_hist, sigma_hist, color="0.5", linewidth=2.6, alpha=0.95, label="SGD")
        ax.plot(mu_hist[0], sigma_hist[0], marker="o", color="0.5", markersize=5, alpha=0.95)

    # Adam
    cmap = plt.get_cmap("tab10")
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "x", "+", "v", "<", ">", "*"]

    for i, (eps, res) in enumerate(zip(actor_epsilons, results_adam)):
        color = cmap(i % 10)
        ls = linestyles[i % len(linestyles)]
        mk = markers[i % len(markers)]
        mu_hist = res["mu"]
        sigma_hist = res["sigma"]

        ax.plot(
            mu_hist,
            sigma_hist,
            color=color,
            linestyle=ls,
            linewidth=3,
            alpha=0.95,
            label=rf"Adam $\epsilon={eps:g}$",
        )
        ax.plot(mu_hist[0], sigma_hist[0], marker=mk, color=color, markersize=5, alpha=0.95)
        ax.plot(mu_hist[-1], sigma_hist[-1], marker=".", color=color, markersize=6, alpha=0.95)

    ax.plot(0.0, 0.0, marker="x", markersize=9, color="r", label="Optimum")

    ax.set_xlabel(r"$\mu$", fontsize=18)
    ax.set_ylabel(r"$\sigma$", fontsize=18)
    ax.set_title(f"M={m}", fontsize=22)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if plot_legend:
        ax.legend(fontsize=12, frameon=False, loc="best")

    fig.tight_layout()
    out_dir = os.path.dirname(savepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {savepath}")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--start-mu", type=float, default=-1.5)
    p.add_argument("--start-sigma", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=0.01)

    # Adam knobs
    p.add_argument(
        "--actor-epsilon",
        type=str,
        default="1e-8,1e-1,1.0",
        help="Comma/space-separated list, e.g. '1e-8,1e-6,1e-4'",
    )
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)

    # SGD baseline lr
    p.add_argument("--lr-sgd", type=float, default=None, help="If set, use this lr for SGD baseline; else use --lr.")

    # plot controls
    p.add_argument("--xlim", type=str, default="")
    p.add_argument("--ylim", type=str, default="")
    p.add_argument("--save", type=str, default="", help="Main multi-panel save path (default: fig/...pdf)")
    p.add_argument("--save-distance", type=str, default="", help="Distance-only save path (default: derived)")
    p.add_argument("--save-opt-path", type=str, default="", help="Opt-path-only save path (default: derived)")
    p.add_argument("--no-deltas", action="store_true", help="Hide Δμ/Δσ subplot in multi-panel figure")

    # NEW: choose what to plot
    p.add_argument("--distance-only", action="store_true", help="Only plot distance-only figure")
    p.add_argument("--optimize-only", action="store_true", help="Only plot optimization-path-only figure")
    p.add_argument("--no-legend", action="store_true", help="Hide legends (paper)")

    args = p.parse_args()

    xlim = _parse_pair(args.xlim) if args.xlim.strip() else None
    ylim = _parse_pair(args.ylim) if args.ylim.strip() else None
    actor_epsilons = _parse_float_list(args.actor_epsilon)

    os.makedirs("fig", exist_ok=True)

    lr_sgd = args.lr if args.lr_sgd is None else args.lr_sgd
    sgd_baseline = simulate_path(
        optimizer="sgd",
        lr=lr_sgd,
        steps=args.steps,
        start_mu=args.start_mu,
        start_sigma=args.start_sigma,
        batch_size=args.batch_size,
        m=args.m,
        seed=args.seed,
        actor_epsilon=1e-8,  # ignored for SGD
        beta1=args.beta1,
        beta2=args.beta2,
    )

    results_adam = []
    for eps in actor_epsilons:
        res = simulate_path(
            optimizer="adam",
            lr=args.lr,
            steps=args.steps,
            start_mu=args.start_mu,
            start_sigma=args.start_sigma,
            batch_size=args.batch_size,
            m=args.m,
            seed=args.seed,
            actor_epsilon=eps,
            beta1=args.beta1,
            beta2=args.beta2,
        )
        results_adam.append(res)

    # title text (short, paper-ish)
    if len(actor_epsilons) == 1:
        eps_text = f"{actor_epsilons[0]:g}"
    else:
        eps_text = ", ".join([f"{eps:g}" for eps in actor_epsilons])
    title = rf"lr={args.lr:g}, eps=[{eps_text}] (SGD lr={lr_sgd:g})"

    # save paths (default: pdf)
    if args.save.strip():
        savepath = args.save.strip()
    else:
        if len(actor_epsilons) == 1:
            savepath = f"fig/remax_paths_M{args.m}_adam_eps_{actor_epsilons[0]:g}_lr_{args.lr}.pdf"
        else:
            eps_slug = "_".join([_eps_tag(eps) for eps in actor_epsilons])
            savepath = f"fig/remax_paths_M{args.m}_adam_eps_multi_{eps_slug}_lr_{args.lr}.pdf"

    distance_savepath = args.save_distance.strip() or _derive_sibling_savepath(savepath, "distance_only")
    opt_path_savepath = args.save_opt_path.strip() or _derive_sibling_savepath(savepath, "opt_path_only")

    plot_legend = (not args.no_legend)

    # dispatch
    if args.distance_only and args.optimize_only:
        plot_distance_only(
            results_adam,
            actor_epsilons=actor_epsilons,
            m=args.m,
            savepath=distance_savepath,
            sgd_baseline=sgd_baseline,
            plot_legend=plot_legend,
        )
        optimization_path_only(
            results_adam,
            actor_epsilons=actor_epsilons,
            m=args.m,
            savepath=opt_path_savepath,
            plot_legend=plot_legend,
            sgd_baseline=sgd_baseline,
        )
        return

    if args.distance_only:
        plot_distance_only(
            results_adam,
            actor_epsilons=actor_epsilons,
            m=args.m,
            savepath=distance_savepath,
            sgd_baseline=sgd_baseline,
            plot_legend=plot_legend,
        )
        return

    if args.optimize_only:
        optimization_path_only(
            results_adam,
            actor_epsilons=actor_epsilons,
            m=args.m,
            savepath=opt_path_savepath,
            plot_legend=plot_legend,
            sgd_baseline=sgd_baseline,
        )
        return

    # default: multi + distance + opt-path
    plot_results_multi(
        results_adam,
        actor_epsilons=actor_epsilons,
        m=args.m,
        title=title,
        savepath=savepath,
        xlim=xlim,
        ylim=ylim,
        add_contours=True,
        plot_deltas=(not args.no_deltas),
        sgd_baseline=sgd_baseline,
        plot_legend=plot_legend,
    )

    plot_distance_only(
        results_adam,
        actor_epsilons=actor_epsilons,
        m=args.m,
        savepath=distance_savepath,
        sgd_baseline=sgd_baseline,
        plot_legend=plot_legend,
    )

    optimization_path_only(
        results_adam,
        actor_epsilons=actor_epsilons,
        m=args.m,
        savepath=opt_path_savepath,
        plot_legend=plot_legend,
        sgd_baseline=sgd_baseline,
    )


if __name__ == "__main__":
    main()