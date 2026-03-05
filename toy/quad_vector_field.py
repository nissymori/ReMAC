import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
})

try:
    from toy.remax import compute_batch_remax
except ModuleNotFoundError:
    from remax import compute_batch_remax


def quad(x: jnp.ndarray) -> jnp.ndarray:
    return -(x**2)  # + 0.01 * jnp.sin(x)


def sample_actions(key: jax.Array, mu: jnp.ndarray, sigma: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    eps = jax.random.normal(key, shape=(batch_size,))
    return mu + sigma * eps


def reparam_objective(
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    key: jax.Array,
    batch_size: int,
    m: int,
    alpha: float
) -> jnp.ndarray:
    actions = sample_actions(key, mu, sigma, batch_size)
    returns = quad(actions)
    remax_obj = compute_batch_remax(returns, m)

    # 1次元ガウス分布のエントロピー: H = 0.5 * log(2 * pi * e * sigma^2)
    # alphaを掛けて目的関数に足し合わせる（探索ボーナス）
    entropy = 0.5 * jnp.log(2 * jnp.pi * jnp.e * (sigma ** 2))

    return remax_obj + alpha * entropy


def estimate_gradients(
    mu_grid: jnp.ndarray,
    sigma_grid: jnp.ndarray,
    batch_size: int,
    m: int,
    alpha: float,
    repeats: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    grad_fn = jax.grad(reparam_objective, argnums=(0, 1))

    mus = mu_grid.ravel()
    sigmas = sigma_grid.ravel()
    num_points = mus.shape[0]
    keys = jax.random.split(key, num_points * repeats).reshape(num_points, repeats, 2)

    def grad_for_point(mu: jnp.ndarray, sigma: jnp.ndarray, keys_for_point: jnp.ndarray) -> jnp.ndarray:
        # alpha を引数として渡す
        grads = jax.vmap(lambda k: grad_fn(mu, sigma, k, batch_size, m, alpha))(keys_for_point)
        dmu, dsigma = grads
        return jnp.stack([jnp.mean(dmu, axis=0), jnp.mean(dsigma, axis=0)], axis=0)

    grads = jax.vmap(grad_for_point)(mus, sigmas, keys)
    dmu = grads[:, 0].reshape(mu_grid.shape)
    dsigma = grads[:, 1].reshape(mu_grid.shape)
    return dmu, dsigma


def plot_gradients(
    mu_grid: jnp.ndarray,
    sigma_grid: jnp.ndarray,
    dmu: jnp.ndarray,
    dsigma: jnp.ndarray,
    ax: plt.Axes,
    normalize: bool,
    cmap: str,
    m: int,
):
    mu_np = np.asarray(mu_grid)
    sigma_np = np.asarray(sigma_grid)
    dmu_np = np.asarray(dmu)
    dsigma_np = np.asarray(dsigma)
    # normalize: まず「単位ベクトル」だけ作る（長さLは掛けない）
    magnitude = np.sqrt(dmu_np**2 + dsigma_np**2)
    eps0 = 1e-12  # 0割回避の閾値
    if normalize:
        # 単位ベクトル（magnitude=0 の点は 0 ベクトル）
        u = np.divide(dmu_np, magnitude, out=np.zeros_like(dmu_np), where=(magnitude > 0))
        v = np.divide(dsigma_np, magnitude, out=np.zeros_like(dsigma_np), where=(magnitude > 0))

        dmu_plot = u 
        dsigma_plot = v 
    else:
        dmu_plot = dmu_np
        dsigma_plot = dsigma_np

    # LogNorm はそのまま
    mag_min = float(np.nanmin(magnitude[magnitude > 0])) if np.any(magnitude > 0) else 1e-12
    mag_max = float(np.nanmax(magnitude)) if np.isfinite(np.nanmax(magnitude)) else mag_min * 10.0
    norm = LogNorm(vmin=mag_min, vmax=mag_max)

    # quiver: 長さを「axes幅」基準で固定（潰れない・等長に見える）
    q = ax.quiver(
        mu_np, sigma_np,
        dmu_plot, dsigma_plot,
        magnitude,
        cmap=cmap,
        norm=norm,
        angles="xy",
        scale_units="width",  # ← ここが肝：画面基準で長さが揃う
        scale=18,             # ← 小さいほど長い（目安 14〜24）
        pivot="tail",
        width=0.008,
        linewidths=0.6,
        headlength=4.5,
        headwidth=4.0,
        headaxislength=4.0,
    )

    ax.margins(x=0.04, y=0.06)  # 端でクリップされて短く見えるのを軽減
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\sigma$")
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_title(f"M={m}")
    return q

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize ReMax gradients for a Gaussian policy and quadratic reward (reparam only)."
    )
    parser.add_argument("--m", type=int, default=4, help="ReMax comparator count.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of actions sampled per point.")
    parser.add_argument("--repeats", type=int, default=100, help="Monte Carlo repeats per grid point.")
    parser.add_argument("--mu-min", type=float, default=-2.0)
    parser.add_argument("--mu-max", type=float, default=2.0)
    parser.add_argument("--sigma-min", type=float, default=0.1)
    parser.add_argument("--sigma-max", type=float, default=2.0)
    parser.add_argument("--mu-steps", type=int, default=13)
    parser.add_argument("--sigma-steps", type=int, default=13)

    # エントロピーボーナス用の引数を追加
    parser.add_argument("--alpha", type=float, default=0.0, help="Entropy bonus coefficient (default: 0.0)")

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize vector field arrows to unit length (color still shows |grad|).",
    )

    # Colorbar on/off (portable: works on Python 3.8+)
    parser.add_argument("--colorbar", dest="colorbar", action="store_true", help="Show colorbar.")
    parser.add_argument("--no-colorbar", dest="colorbar", action="store_false", help="Hide colorbar.")
    parser.set_defaults(colorbar=True)

    # Paper-friendly colormap choices (sequential)
    cmap_choices = [
        "cividis", "cividis_r", "viridis", "viridis_r", "plasma", "plasma_r",
        "magma", "magma_r", "inferno", "inferno_r", "Greys", "Blues", "turbo",
        "BuPu", "BuPu_r", "BuGn", "BuGn_r", "BuRd", "BuRd_r", "PuBu", "PuBu_r",
        "PuBuGn", "PuBuGn_r", "PuRd", "PuRd_r", "RdPu", "RdPu_r", "RdBu",
        "RdYlBu", "twilight",
    ]
    parser.add_argument("--cmap", type=str, default="Blues", choices=cmap_choices)

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sigma_min <= 0.0:
        raise ValueError("sigma-min must be > 0.")
    if args.sigma_max <= args.sigma_min:
        raise ValueError("sigma-max must be greater than sigma-min.")

    # A bit more readable defaults for papers
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 23,
            "axes.labelsize": 22,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    mu_vals = jnp.linspace(args.mu_min, args.mu_max, args.mu_steps)
    sigma_vals = jnp.linspace(args.sigma_min, args.sigma_max, args.sigma_steps)
    mu_grid, sigma_grid = jnp.meshgrid(mu_vals, sigma_vals)

    key = jax.random.PRNGKey(args.seed)
    dmu, dsigma = estimate_gradients(
        mu_grid,
        sigma_grid,
        batch_size=args.batch_size,
        m=args.m,
        alpha=args.alpha,  # alpha を渡す
        repeats=args.repeats,
        key=key,
    )
    if args.colorbar:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 5.0))
    q = plot_gradients(
        mu_grid,
        sigma_grid,
        dmu,
        dsigma,
        ax=ax,
        normalize=args.normalize,
        cmap=args.cmap,
        m=args.m,
    )

    if args.colorbar:
        cbar = fig.colorbar(q, ax=ax, pad=0.02)
        cbar.set_label(r"$\|\nabla\|$")

    fig.tight_layout()

    # 出力ファイル名に alpha を含めるように変更
    out = (
        f"fig/quad_gradients_m={args.m}"
        f"_alpha={args.alpha}"
        f"_norm={args.normalize}"
        f"_cmap={args.cmap}"
        f"_cb={args.colorbar}.pdf"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    main()