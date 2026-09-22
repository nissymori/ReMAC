# paper_plots.py
import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# fetch_wandb_brax.py
import re
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm


def _canon_env(env) -> str:
    s = str(env).strip()
    if s == "" or s.lower() == "none":
        return ""
    s = s.lower()
    if s.startswith("brax/"):
        return s
    # env="ant" のような場合を想定
    return f"brax/{s}"


def _get_lr_from_config(cfg: dict):
    # まずは指定通り learning_rate を優先
    for k in ["learning_rate", "lr", "actor_lr", "actor_learning_rate", "policy_lr"]:
        if k in cfg and cfg.get(k) is not None:
            try:
                return float(cfg.get(k))
            except Exception:
                pass
    return None


def fetch_wandb_data_with_seeds(
    project_name: str,
    entity: str | None = None,
    samples: int = 500,
    *,
    target_envs: tuple[str, ...] = ("ant", "hopper", "walker2d", "halfcheetah", "swimmer", "reacher"),
    base_metrics: tuple[str, ...] = ("eval/return", "train/entropy", "train/actor_grad_norm", "train/actor_update_norm"),
    eval_freq_required: int = 30000,
    skip_soft_critic: bool = True,
    label_with_lr: bool = True,
):
    """
    Returns a long-format dataframe aggregated across runs:
      columns include:
        env, algo_label, algorithm, remax_m, actor_epsilon, learning_rate, run_id, run_name,
        _step, seed_id, and metrics columns (eval/return, train/entropy, ...)
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project_name}" if entity else project_name)

    all_runs_data = []
    target_envs_lc = tuple([e.lower() for e in target_envs])
    seed_pattern = re.compile(r"seed_\d+/(.*)")

    print(f"Fetching data from {len(runs)} runs...")

    for run in tqdm(runs):
        cfg = dict(run.config or {})

        env_raw = cfg.get("env")
        env = _canon_env(env_raw)
        if env == "":
            continue
        if not any(t in env for t in target_envs_lc):
            continue

        algo = cfg.get("algorithm")
        if algo not in ("remax_ac", "sac", "ppo", "td3"):
            continue

        # config filters
        if cfg.get("eval_freq") != eval_freq_required:
            continue
        if algo == "remax_ac" and skip_soft_critic and cfg.get("soft_critic"):
            continue

        # parse ReMAC params
        remax_m = None
        actor_epsilon = None
        remax_num_samples = None
        if algo == "remax_ac":
            remax_m = cfg.get("remax_m")
            actor_epsilon = cfg.get("actor_epsilon")
            remax_num_samples = cfg.get("remax_num_samples")
            if remax_m is None or actor_epsilon is None:
                continue
            try:
                remax_m = int(remax_m)
                actor_epsilon = float(actor_epsilon)
            except Exception:
                continue
            if remax_num_samples is not None:
                try:
                    remax_num_samples = int(remax_num_samples)
                except Exception:
                    remax_num_samples = None

        lr = _get_lr_from_config(cfg)
        if lr is None:
            # learning_rate が無い run は lr sweep も default-lr filter もできないので落とすのが無難
            continue

        # label (learning_rate も含める)
        if algo == "remax_ac":
            if label_with_lr:
                label = f"ReMAC (M={remax_m}, eps={actor_epsilon:g}, lr={lr:g})"
            else:
                label = f"ReMAC (M={remax_m}, eps={actor_epsilon:g})"
        else:
            algo_tag = {"sac": "SAC", "ppo": "PPO", "td3": "TD3"}[algo]
            label = f"{algo_tag} (lr={lr:g})" if label_with_lr else algo_tag

        # --- detect available seed_* keys (light fetch, sample=1) ---
        try:
            head = run.history(samples=1, pandas=True)
        except Exception:
            continue
        if head is None or head.empty:
            continue

        available_keys = []
        for k in head.columns:
            m = seed_pattern.match(str(k))
            if not m:
                continue
            base = m.group(1)
            if base in base_metrics:
                available_keys.append(k)

        if len(available_keys) == 0:
            continue

        # --- fetch sampled history only for needed keys ---
        try:
            history = run.history(keys=["_step"] + available_keys, samples=samples, pandas=True)
        except Exception:
            continue
        if history is None or history.empty:
            continue

        melted = history.melt(id_vars=["_step"], value_vars=available_keys)
        melted[["seed_id", "metric"]] = melted["variable"].astype(str).str.extract(r"(seed_\d+)/(.*)")
        melted = melted.dropna(subset=["seed_id", "metric"])

        final_df = (
            melted.pivot_table(index=["_step", "seed_id"], columns="metric", values="value")
            .reset_index()
        )

        final_df["env"] = env
        final_df["algo_label"] = label
        final_df["algorithm"] = algo
        final_df["remax_m"] = remax_m
        final_df["actor_epsilon"] = actor_epsilon
        final_df["remax_num_samples"] = remax_num_samples
        final_df["learning_rate"] = lr
        final_df["run_id"] = run.id
        final_df["run_name"] = run.name

        all_runs_data.append(final_df)

    if not all_runs_data:
        print("データが見つかりませんでした。キー名やプロジェクト名を確認してください。")
        return pd.DataFrame()

    df = pd.concat(all_runs_data, ignore_index=True)

    # ensure numeric
    df["_step"] = pd.to_numeric(df["_step"], errors="coerce")
    df = df.dropna(subset=["_step"])
    df["_step"] = df["_step"].astype(int)
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    if "remax_num_samples" in df.columns:
        df["remax_num_samples"] = pd.to_numeric(df["remax_num_samples"], errors="coerce")

    return df


# The CLI entry point is defined at the bottom of this file.


# ============================================================
# Defaults (edit to match your runs)
# ============================================================

DEFAULT_ENV_LR: Dict[str, float] = {
    "brax/ant": 2e-4,
    "brax/halfcheetah": 1e-4,
    "brax/hopper": 3e-4,
    "brax/walker2d": 1e-4,
    "brax/reacher": 3e-4,   # re-tuned after the episode_length fix (was 5e-4)
    "brax/swimmer": 1e-4,
    "brax/humanoidstandup": 1e-4,
    "brax/pusher": 2e-4,       # tuned for the camera-ready version
}

# Environment order of the paper's main grids: the six envs of Figs. 3-4 (2x3), and the
# camera-ready eight (2x4), which appends humanoidstandup and pusher so that the first
# six keep their panel positions.
ENV_ORDER_6: Tuple[str, ...] = (
    "brax/ant",
    "brax/halfcheetah",
    "brax/hopper",
    "brax/walker2d",
    "brax/reacher",
    "brax/swimmer",
)
ENV_ORDER_8: Tuple[str, ...] = ENV_ORDER_6 + ("brax/humanoidstandup", "brax/pusher")

DEFAULT_M_COLOR_ORDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
M_TO_TAB10_INDEX = {m: i for i, m in enumerate(DEFAULT_M_COLOR_ORDER)}  # m -> 0..9

# Baseline (non-ReMAC) styles. Distinguished by linestyle so they remain readable
# in B&W. Tones chosen to stay outside the tab10 palette used for ReMAC's M.
BASELINE_KINDS: Tuple[str, ...] = ("SAC", "PPO", "TD3")


# ============================================================
# Utils
# ============================================================
_M_RE = re.compile(r"m=(\d+)", re.IGNORECASE)
_EPS_RE = re.compile(r"eps=([0-9eE.+-]+)", re.IGNORECASE)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _paper_rcparams() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 13,
            "legend.handlelength": 1.2,
            "legend.handletextpad": 0.5,
        }
    )


LEGEND_HANDLE_LW = 4.5


def _metric_short(metric: str) -> str:
    return str(metric).split("/")[-1]


def _fmt_eps(eps) -> str:
    """Format epsilon for display: '1e-8' (not '1e-08'), '1', '0.1', '0.01', ..."""
    s = f"{float(eps):g}"
    m = re.fullmatch(r"(\d+(?:\.\d+)?)[eE]([+-]?)0*(\d+)", s)
    if m:
        mant, sign, exp = m.group(1), m.group(2), m.group(3)
        sign = "-" if sign == "-" else ""
        return f"{mant}e{sign}{exp}"
    return s


def _canon_env(env) -> str:
    s = str(env).strip().lower()
    if s.startswith("brax/"):
        return s
    if s == "" or s == "none":
        return ""
    return f"brax/{s}"


def _prepare_plot_df(
    df: pd.DataFrame,
    ms: Optional[Sequence[int]],
    epsilons: Optional[Sequence[float]],
    include_sac: bool,
    *,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    # --- NEW: lr filter ---
    lrs: Optional[Sequence[float]] = None,
    lr_atol: float = 0.0,
    lr_rtol: float = 1e-12,
    # --- NEW: B (remax_num_samples) filter ---
    bs: Optional[Sequence[int]] = None,
    # --- NEW: explicit baseline subset (overrides include_other_baselines) ---
    baselines: Optional[Sequence[str]] = None,
    # ---
    atol: float = 0.0,
    rtol: float = 1e-12,
    include_other_baselines: bool = True,
) -> pd.DataFrame:
    required = {"env", "algo_label", "_step"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    if use_default_lr:
        env_lr_map = DEFAULT_ENV_LR if env_lr_map is None else env_lr_map
        if lr_key not in df.columns:
            raise ValueError(f"use_default_lr=True but df has no column '{lr_key}'")

    out = df.copy()
    out["env"] = out["env"].map(_canon_env)

    label_str = out["algo_label"].astype(str)
    out["algo"] = np.select(
        [
            label_str.str.contains("SAC", regex=False),
            label_str.str.contains("PPO", regex=False),
            label_str.str.contains("TD3", regex=False),
        ],
        ["SAC", "PPO", "TD3"],
        default="REMAX",
    )
    is_sac = out["algo"].isin(BASELINE_KINDS)  # "include_sac" historically means "include any baseline"
    out["m"] = pd.to_numeric(out["algo_label"].astype(str).str.extract(_M_RE)[0], errors="coerce")
    out["eps"] = pd.to_numeric(out["algo_label"].astype(str).str.extract(_EPS_RE)[0], errors="coerce")

    out["_step"] = pd.to_numeric(out["_step"], errors="coerce")
    out = out.dropna(subset=["_step"])
    out["_step"] = out["_step"].astype(int)

    if lr_key in out.columns:
        out[lr_key] = pd.to_numeric(out[lr_key], errors="coerce")

    keep = np.ones(len(out), dtype=bool)

    is_baseline = out["algo"].isin(BASELINE_KINDS).to_numpy()
    is_non_sac_baseline = out["algo"].isin([k for k in BASELINE_KINDS if k != "SAC"]).to_numpy()

    if not include_sac:
        keep &= ~is_baseline
    elif baselines is not None:
        allowed = {str(b).upper() for b in baselines}
        in_allowed = out["algo"].isin(allowed).to_numpy()
        keep &= (~is_baseline | in_allowed)
    elif not include_other_baselines:
        keep &= ~is_non_sac_baseline

    if ms is not None:
        ms_set = set(int(x) for x in ms)
        keep &= (is_baseline | out["m"].isin(ms_set).to_numpy())

    if epsilons is not None:
        eps_arr = np.array(list(epsilons), dtype=float)
        eps_ok = np.isclose(out["eps"].to_numpy(dtype=float), eps_arr[:, None], atol=atol, rtol=rtol).any(axis=0)
        keep &= (is_baseline | eps_ok)

    if lrs is not None:
        if lr_key not in out.columns:
            raise ValueError(f"lrs is specified but df has no column '{lr_key}'")
        lrs_arr = np.array(list(lrs), dtype=float)
        lr_vals = out[lr_key].to_numpy(dtype=float)
        lr_ok = np.isclose(lr_vals, lrs_arr[:, None], atol=lr_atol, rtol=lr_rtol).any(axis=0)
        keep &= (is_baseline | lr_ok)

    if bs is not None:
        if "remax_num_samples" not in out.columns:
            raise ValueError("bs is specified but df has no column 'remax_num_samples'")
        bs_set = {int(b) for b in bs}
        b_vals = pd.to_numeric(out["remax_num_samples"], errors="coerce")
        # Keep baselines (no B) always; keep ReMAC rows whose B is in the set.
        b_ok = b_vals.isin(bs_set).to_numpy()
        keep &= (is_baseline | b_ok)

    if use_default_lr:
        env_lr_map = DEFAULT_ENV_LR if env_lr_map is None else env_lr_map
        default_lr = out["env"].map(env_lr_map)  # NaN for unknown
        known_env = ~default_lr.isna()

        lr = out[lr_key]
        lr_ok = np.isclose(lr.to_numpy(dtype=float), default_lr.to_numpy(dtype=float), atol=0.0, rtol=rtol)
        # Don't drop baseline rows (no remax_m / actor_epsilon, but keep their own lr).
        keep &= (is_baseline | (~known_env.to_numpy()) | lr_ok)

    out = out.loc[keep].copy()
    return out


# ============================================================
# Styles
# ============================================================
@dataclass(frozen=True)
class Style:
    color: Tuple[float, float, float, float]
    linestyle: str
    alpha: float
    linewidth: float


# Concrete styles for non-ReMAC baselines.
BASELINE_STYLES: Dict[str, "Style"] = {
    "SAC": Style(color=(0.00, 0.00, 0.00, 1.0), linestyle="-",  alpha=0.95, linewidth=2.4),
    "PPO": Style(color=(0.30, 0.30, 0.30, 1.0), linestyle="--", alpha=0.95, linewidth=2.4),
    "TD3": Style(color=(0.55, 0.55, 0.55, 1.0), linestyle="-.", alpha=0.95, linewidth=2.4),
}


# 2) _build_style_maps を少しだけ変更して eps2alpha を返すようにする
def _build_style_maps(
    ms: Sequence[int],
    epsilons: Sequence[float],
    vary_eps_alpha: bool,
    *,
    alpha_fixed: float = 0.30,
    alpha_min: float = 0.40,
    alpha_max: float = 1.00,
) -> Tuple[Dict[Tuple[int, float], Style], Style, Dict[float, float]]:
    ms_sorted = sorted({int(m) for m in ms})
    eps_sorted = sorted({float(eps) for eps in epsilons})

    tab = plt.get_cmap("tab10")

    def _m_to_color(m: int):
        if m in M_TO_TAB10_INDEX:
            return tab(M_TO_TAB10_INDEX[m] % 10)
        if m > 0 and (m & (m - 1) == 0):
            idx = int(round(np.log2(m))) % 10
        else:
            idx = int(m) % 10
        return tab(idx)

    m2color = {m: _m_to_color(m) for m in ms_sorted}

    if vary_eps_alpha and len(eps_sorted) > 1:
        alphas = np.linspace(alpha_min, alpha_max, num=len(eps_sorted))
        eps2alpha = {eps: float(alphas[i]) for i, eps in enumerate(eps_sorted)}
    else:
        eps2alpha = {eps: float(alpha_fixed) for eps in eps_sorted}

    style_map: Dict[Tuple[int, float], Style] = {}
    for m in ms_sorted:
        for eps in eps_sorted:
            style_map[(m, eps)] = Style(
                color=m2color[m],
                linestyle="-",
                alpha=eps2alpha[eps],
                linewidth=2.2,
            )

    sac_style = Style(color=(0, 0, 0, 1), linestyle="-", alpha=0.95, linewidth=2.4)
    return style_map, sac_style, eps2alpha


def _build_lr_style_map(
    lrs: Sequence[float],
    *,
    lr_encode: str = "color",  # "color" | "linestyle" | "both"
    alpha_fixed: float = 0.8,
    linewidth: float = 2.2,
) -> Dict[float, Style]:
    lrs_sorted = sorted({float(x) for x in lrs})

    # Up to 4 lrs: keep the original navy/gray/yellow palette (Fig 14 unchanged).
    # More than 4 lrs: switch to an extended palette with enough *distinct* colors
    # so no two lrs share a color (e.g. the eps=1 vs eps=1e-8 lr sweeps don't clash).
    base_colors = ["#2F3E5C", "#5A5A5A", "#C9CDD3", "#D9C27A"]
    ext_colors = [
        "#2F3E5C",  # navy
        "#4C72B0",  # blue
        "#55A868",  # green
        "#8172B3",  # purple
        "#C44E52",  # red
        "#CCB974",  # yellow/olive
        "#64B5CD",  # cyan
        "#937860",  # brown
    ]
    palette = base_colors if len(lrs_sorted) <= len(base_colors) else ext_colors

    styles = ["-", "--", ":", "-."]

    lr_style_map: Dict[float, Style] = {}
    for i, lr in enumerate(lrs_sorted):
        color = (0.0, 0.0, 0.0, 1.0)
        linestyle = "-"

        if lr_encode in ("color", "both"):
            color = palette[i % len(palette)]
        if lr_encode in ("linestyle", "both"):
            linestyle = styles[i % len(styles)]

        lr_style_map[lr] = Style(
            color=color,
            linestyle=linestyle,
            alpha=alpha_fixed,
            linewidth=linewidth,
        )
    return lr_style_map


def _build_eps_lr_style_map(
    eps_sorted: Sequence[float],
    lrs_by_eps: Dict[float, Sequence[float]],
    *,
    families: Sequence[str] = ("Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"),
    alpha: float = 0.95,
    linewidth: float = 2.2,
    shade_lo: float = 0.42,
    shade_hi: float = 0.95,
    common_lrs: Sequence[float] = (),
    common_color=(0.0, 0.0, 0.0, 1.0),
) -> Dict[Tuple[float, float], Style]:
    """
    Colour each curve by *epsilon family* (eps=1 -> reds, eps=1e-8 -> blues, ...),
    using the lr's rank within that eps to pick the shade (small lr = light, large
    lr = dark).  All solid lines, so the figure reads as two calm colour groups
    rather than a busy rainbow.

    `common_lrs` are lrs that appear under more than one eps (e.g. the default lr
    shared by the eps=1 and eps=1e-8 sweeps); they get a single `common_color` in
    every tier so the reader can see it is the same lr.  Family shades are spread
    over the remaining (non-common) lrs only.

    Returns a map keyed by (eps, lr).
    """
    common_set = {float(x) for x in common_lrs}
    out: Dict[Tuple[float, float], Style] = {}
    for i, eps in enumerate(eps_sorted):
        cmap = plt.get_cmap(families[i % len(families)])
        all_lrs = sorted(float(x) for x in lrs_by_eps.get(eps, []))
        family_lrs = [lr for lr in all_lrs if lr not in common_set]
        n = len(family_lrs)
        for lr in all_lrs:
            if lr in common_set:
                color = common_color
            else:
                k = family_lrs.index(lr)
                t = shade_lo if n <= 1 else shade_lo + (shade_hi - shade_lo) * (k / (n - 1))
                color = cmap(t)
            out[(float(eps), float(lr))] = Style(
                color=color, linestyle="-", alpha=alpha, linewidth=linewidth
            )
    return out


# ============================================================
# Plot primitives
# ============================================================
def _summarize_by_step(df_sub: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    g = df_sub.groupby("_step")[metric]
    mean = g.mean()
    se = g.sem()
    x = mean.index.to_numpy(dtype=int)
    y = np.stack([mean.to_numpy(dtype=float), se.fillna(0.0).to_numpy(dtype=float)], axis=1)
    order = np.argsort(x)
    return x[order], y[order]


def _summarize_xy_by_step(
    df_sub: pd.DataFrame, x_col: str, y_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate both x and y across seeds at each _step. x is mean over seeds; y is (mean, sem)."""
    g = df_sub.groupby("_step")
    x_mean = g[x_col].mean()
    y_mean = g[y_col].mean()
    y_sem = g[y_col].sem().fillna(0.0)
    df_agg = pd.DataFrame({"x": x_mean, "y_mean": y_mean, "y_sem": y_sem}).dropna(subset=["x", "y_mean"])
    df_agg = df_agg.sort_values("x")
    x = df_agg["x"].to_numpy(dtype=float)
    y = np.stack([df_agg["y_mean"].to_numpy(dtype=float), df_agg["y_sem"].to_numpy(dtype=float)], axis=1)
    return x, y


def _add_cum_dtheta(
    df: pd.DataFrame,
    *,
    update_key: str = "train/actor_update_norm",
    out_key: str = "cum_dtheta",
) -> pd.DataFrame:
    """Per (env, algo_label, seed_id) trajectory, cumulative sum of ||Δθ|| over _step."""
    if update_key not in df.columns:
        raise ValueError(f"df missing column '{update_key}' (needed for cum_dtheta)")
    out = df.copy()
    out[update_key] = pd.to_numeric(out[update_key], errors="coerce").fillna(0.0)
    out = out.sort_values(["env", "algo_label", "seed_id", "_step"])
    out[out_key] = out.groupby(["env", "algo_label", "seed_id"])[update_key].cumsum()
    return out


def _plot_mean_se(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    style: Style,
    *,
    show_band: bool = True,
):
    ax.plot(
        x,
        y[:, 0],
        label=label,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        alpha=style.alpha,
    )
    if show_band:
        lo = y[:, 0] - y[:, 1]
        hi = y[:, 0] + y[:, 1]
        ax.fill_between(
            x,
            lo,
            hi,
            color=style.color,
            alpha=min(0.18, 0.45 * style.alpha),
            linewidth=0.0,
        )


def _dedupe_handles(handles: List[Line2D]) -> List[Line2D]:
    seen = set()
    out = []
    for h in handles:
        lab = h.get_label()
        if lab in seen:
            continue
        seen.add(lab)
        out.append(h)
    return out

# 1) 追加：eps 用の凡例ハンドル（mに依存しない）
def _legend_handles_eps(
    epsilons: Sequence[float],
    eps2alpha: Dict[float, float],
    *,
    include_sac: bool,
    sac_style: Style,
    linewidth: float = LEGEND_HANDLE_LW,
    linestyle: str = "-",
) -> List[Line2D]:
    handles: List[Line2D] = []
    if include_sac:
        handles.append(
            Line2D(
                [0], [0],
                color=sac_style.color,
                lw=LEGEND_HANDLE_LW,
                ls=sac_style.linestyle,
                label="SAC",
            )
        )

    # eps は alpha で表現（色は黒固定にして誤解を避ける）
    for eps in sorted({float(x) for x in epsilons}):
        a = float(eps2alpha[eps])
        # RGBAでalpha込みの色を作る（legendでalphaが薄く見えない環境対策）
        color = (0.0, 0.0, 0.0, a)
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                lw=linewidth,
                ls=linestyle,
                label=rf"$\epsilon={_fmt_eps(eps)}$",
            )
        )
    return handles

# 置き換え：プロットと完全に同じスタイルで凡例ハンドルを作る
def _legend_handles_remax(
    ms: Sequence[int],
    epsilons: Sequence[float],
    style_map: Dict[Tuple[int, float], Style],
    *,
    include_sac: bool,
    sac_style: Style,
    include_eps_in_label: bool,
    baselines_present: Sequence[str] = ("SAC",),
) -> List[Line2D]:
    handles: List[Line2D] = []

    if include_sac:
        for kind in BASELINE_KINDS:
            if kind not in baselines_present:
                continue
            st = BASELINE_STYLES[kind]
            handles.append(
                Line2D(
                    [0], [0],
                    color=st.color,
                    lw=LEGEND_HANDLE_LW,
                    ls=st.linestyle,
                    alpha=st.alpha,
                    label=kind,
                )
            )

    ms_sorted = sorted({int(x) for x in ms})
    eps_sorted = sorted({float(x) for x in epsilons})

    if include_eps_in_label:
        # (m, eps) 全部出す → 色・alphaが必ず一致
        for m in ms_sorted:
            for eps in eps_sorted:
                st = style_map[(m, eps)]
                handles.append(
                    Line2D(
                        [0], [0],
                        color=st.color,
                        lw=LEGEND_HANDLE_LW,
                        ls=st.linestyle,
                        alpha=st.alpha,
                        label=rf"ReMAC ($M={m}$, $\epsilon={_fmt_eps(eps)}$)",
                    )
                )
    else:
        # m だけ出す（色だけ見せたいので alpha=1 に固定）
        # eps_sorted の先頭の色を使う（色自体は eps に依存しないのでどれでも同じ）
        eps0 = eps_sorted[0] if len(eps_sorted) > 0 else 1.0
        for m in ms_sorted:
            st = style_map[(m, eps0)]
            handles.append(
                Line2D(
                    [0], [0],
                    color=st.color,
                    lw=LEGEND_HANDLE_LW,
                    ls=st.linestyle,
                    alpha=1.0,
                    label=rf"ReMAC ($M={m}$)",
                )
            )

    return handles

def _legend_handles_m(
    ms: Sequence[int],
    style_map: Dict[Tuple[int, float], Style],
    eps_for_color: float,
    include_sac: bool,
    sac_style: Style,
) -> List[Line2D]:
    handles: List[Line2D] = []
    if include_sac:
        handles.append(Line2D([0], [0], color=sac_style.color, lw=LEGEND_HANDLE_LW, ls=sac_style.linestyle, label="SAC"))
    for m in sorted(set(int(x) for x in ms)):
        st = style_map[(m, eps_for_color)]
        handles.append(Line2D([0], [0], color=st.color, lw=LEGEND_HANDLE_LW, ls="-", alpha=1.0, label=rf"ReMAC $M={m}$"))
    return _dedupe_handles(handles)


# ============================================================
# 1) envs in an nrows x ncols grid (the paper's 2x3, and the camera-ready 2x4)
# ============================================================
def plot_envs_grid(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    metrics: Sequence[str],
    *,
    nrows: int = 2,
    ncols: int = 3,
    vary_eps_alpha: bool = True,
    include_sac: bool = True,
    include_other_baselines: bool = True,
    include_eps_in_label: bool = False,
    alpha_fixed: float = 0.80,
    show_band: bool = True,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    # --- NEW ---
    lrs: Optional[Sequence[float]] = None,
    lr_rtol: float = 1e-12,
    bs: Optional[Sequence[int]] = None,
    bs_per_m: Optional[Dict[int, int]] = None,
    baselines: Optional[Sequence[str]] = None,
    # ----------
    env_order: Sequence[str] = ENV_ORDER_6,
    out_dir: str = "fig",
    filename_prefix: str = "paper_2x3",
    log_scale=False,
) -> None:
    _paper_rcparams()

    plot_df = _prepare_plot_df(
        df,
        ms=ms,
        epsilons=epsilons,
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        # --- NEW ---
        lrs=lrs,
        lr_rtol=lr_rtol,
        bs=bs,
        baselines=baselines,
        # ----------
    )

    style_map, sac_style, _ = _build_style_maps(ms, epsilons, vary_eps_alpha, alpha_fixed=alpha_fixed)
    envs_present = [e for e in env_order if e in set(plot_df["env"].unique())]
    if len(envs_present) == 0:
        raise ValueError("None of the requested envs are present.")
    n_panels = nrows * ncols
    # keep the per-panel aspect ratio of the original 2x3 figure
    figsize = (12.5 * ncols / 3.0, 5.5 * nrows / 2.0)

    for metric in metrics:
        if metric not in plot_df.columns:
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=False)
        axes = axes.ravel()

        for i, env in enumerate(envs_present[:n_panels]):
            ax = axes[i]
            df_env = plot_df[plot_df["env"] == env]

            if include_sac:
                for kind in BASELINE_KINDS:
                    df_b = df_env[df_env["algo"] == kind]
                    df_b = df_b[df_b["_step"] <= 3_000_000]
                    if len(df_b) == 0:
                        continue
                    x, y = _summarize_by_step(df_b, metric)
                    _plot_mean_se(ax, x, y, kind, BASELINE_STYLES[kind], show_band=show_band)

            for m in sorted(set(int(x) for x in ms)):
                for eps in sorted(set(float(x) for x in epsilons)):
                    mask = (
                        (df_env["algo"] == "REMAX")
                        & (df_env["m"] == m)
                        & np.isclose(df_env["eps"].to_numpy(dtype=float), eps, atol=0.0, rtol=1e-12)
                    )
                    if bs_per_m is not None and m in bs_per_m:
                        b_vals = pd.to_numeric(df_env["remax_num_samples"], errors="coerce")
                        mask = mask & (b_vals == bs_per_m[m])
                    df_re = df_env.loc[mask]
                    if len(df_re) == 0:
                        continue

                    x, y = _summarize_by_step(df_re, metric)
                    st = style_map[(m, eps)]
                    if include_eps_in_label:
                        label = rf"ReMAC $M={m}$, $\epsilon={_fmt_eps(eps)}$"
                    else:
                        label = rf"ReMAC $M={m}$"
                    _plot_mean_se(ax, x, y, label, st, show_band=show_band)

            ax.set_title(env.replace("brax/", ""))
            ax.set_xlabel("step")
            ax.set_ylabel(_metric_short(metric))
            ax.grid(True, alpha=0.25)
            if log_scale:
              ax.set_yscale("log")

        for j in range(len(envs_present[:n_panels]), n_panels):
            axes[j].axis("off")

        baselines_in_df = [k for k in BASELINE_KINDS if (plot_df["algo"] == k).any()]
        handles = _legend_handles_remax(
            ms=ms,
            epsilons=epsilons,
            style_map=style_map,
            include_sac=include_sac,
            sac_style=sac_style,
            include_eps_in_label=include_eps_in_label,
            baselines_present=baselines_in_df,
        )

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(6, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.tight_layout(rect=(0, 0.06, 1, 1))
        metric_slug = _metric_short(metric)
        out_path = os.path.join(out_dir, f"{filename_prefix}_{metric_slug}_m{list(ms)}_eps{list(epsilons)}.pdf")
        _ensure_dir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_6envs_2x3(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    metrics: Sequence[str],
    **kwargs,
) -> None:
    """The paper's 2x3 grid (Figs. 3-4): plot_envs_grid with nrows=2, ncols=3."""
    plot_envs_grid(df, epsilons, ms, metrics, nrows=2, ncols=3, **kwargs)


# ============================================================
# 2) 4 envs in a row (1x4)
# ============================================================
def plot_4envs_row(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    metrics: Sequence[str],
    *,
    vary_eps_alpha: bool = True,
    include_sac: bool = True,
    include_other_baselines: bool = True,
    include_eps_in_label: bool = False,
    alpha_fixed: float = 0.80,
    show_band: bool = True,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    # --- NEW ---
    lrs: Optional[Sequence[float]] = None,
    lr_rtol: float = 1e-12,
    # ----------
    env_order: Sequence[str] = (
        "brax/ant",
        "brax/halfcheetah",
        "brax/hopper",
        "brax/walker2d",
        "brax/reacher",
        "brax/swimmer",
    ),
    out_dir: str = "fig",
    filename_prefix: str = "paper_4",
):
    _paper_rcparams()
    envs = ["brax/ant", "brax/halfcheetah", "brax/swimmer", "brax/walker2d"]
    plot_df = _prepare_plot_df(
        df,
        ms=ms,
        epsilons=epsilons,
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        # --- NEW ---
        lrs=lrs,
        lr_rtol=lr_rtol,
        # ----------
    )

    style_map, sac_style, eps2alpha = _build_style_maps(ms, epsilons, vary_eps_alpha, alpha_fixed=alpha_fixed)
    envs_present = [e for e in envs if e in set(plot_df["env"].unique())]
    if len(envs_present) == 0:
        raise ValueError("None of the requested envs are present.")

    for metric in metrics:
        if metric not in plot_df.columns:
            continue

        fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.6), sharex=False, sharey=False)
        axes = np.asarray(axes)

        for i in range(4):
            ax = axes[i]
            if i >= len(envs_present):
                ax.axis("off")
                continue

            env = envs_present[i]
            df_env = plot_df[plot_df["env"] == env]

            if include_sac:
                for kind in BASELINE_KINDS:
                    df_b = df_env[df_env["algo"] == kind]
                    df_b = df_b[df_b["_step"] <= 3_000_000]
                    if len(df_b) == 0:
                        continue
                    x, y = _summarize_by_step(df_b, metric)
                    _plot_mean_se(ax, x, y, kind, BASELINE_STYLES[kind], show_band=show_band)

            for m in sorted(set(int(x) for x in ms)):
                for eps in sorted(set(float(x) for x in epsilons)):
                    mask = (
                        (df_env["algo"] == "REMAX")
                        & (df_env["m"] == m)
                        & np.isclose(df_env["eps"].to_numpy(dtype=float), eps, atol=0.0, rtol=1e-12)
                    )
                    df_re = df_env.loc[mask]
                    if len(df_re) == 0:
                        continue

                    x, y = _summarize_by_step(df_re, metric)
                    st = style_map[(m, eps)]
                    if include_eps_in_label:
                        label = rf"ReMAC $M={m}$, $\epsilon={_fmt_eps(eps)}$"
                    else:
                        label = rf"ReMAC $M={m}$"
                    _plot_mean_se(ax, x, y, label, st, show_band=show_band)

            ax.set_title(env.replace("brax/", ""))
            ax.set_xlabel("step")
            ax.set_ylabel(_metric_short(metric))
            ax.grid(True, alpha=0.25)

        baselines_in_df = [k for k in BASELINE_KINDS if (plot_df["algo"] == k).any()]
        handles = _legend_handles_remax(
            ms=ms,
            epsilons=epsilons,
            style_map=style_map,
            include_sac=include_sac,
            sac_style=sac_style,
            include_eps_in_label=include_eps_in_label,
            baselines_present=baselines_in_df,
        )

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(6, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.05),
        )

        fig.tight_layout(rect=(0, 0.10, 1, 1))
        metric_slug = _metric_short(metric)
        out_path = os.path.join(out_dir, f"{filename_prefix}_{metric_slug}_m{list(ms)}_eps{list(epsilons)}.pdf")
        _ensure_dir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ============================================================
# 3) env,m,eps固定で lr sweep（return + entropy）
# ============================================================
def plot_lr_sweep_return_entropy(
    df: pd.DataFrame,
    *,
    env: str,
    m: int,
    epsilon: float,
    lrs: Sequence[float] = (1e-4, 3e-4, 5e-4, 1e-3),
    lr_key: str = "learning_rate",
    lr_encode: str = "both",  # "color" | "linestyle" | "both"
    include_sac: bool = False,
    include_other_baselines: bool = True,
    show_band: bool = True,
    out_path: str = "fig/lr_sweep.pdf",
    # --- reference lines (default lr, eps=1e-8) ---
    add_default_reference: bool = True,
    reference_epsilon: float = 1e-8,
    env_lr_map: Optional[Dict[str, float]] = None,
    reference_linewidth: float = 2.2,
    reference_linestyle: str = "--",
    reference_alpha: float = 0.95,
    annotate_reference: bool = False,
) -> None:
    """
    (env, m, eps) 固定で lr sweep を描く（return + entropy）。
    追加: 同じ env, m, actor_epsilon=reference_epsilon (=1e-8), default_lr の
         - max return
         - min entropy
    を赤い横線で reference として描く。
    """
    _paper_rcparams()

    required = {"env", "algo_label", "_step", lr_key, "eval/return", "train/entropy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    env = _canon_env(env)
    env_lr_map = DEFAULT_ENV_LR if env_lr_map is None else env_lr_map
    default_lr = env_lr_map.get(env, None)

    # lr sweep を見たいので default-lr filter はOFF
    plot_df = _prepare_plot_df(
        df,
        ms=[m],
        epsilons=[epsilon],
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        use_default_lr=False,
        lr_key=lr_key,
    )
    plot_df = plot_df[plot_df["env"] == env].copy()
    if len(plot_df) == 0:
        raise ValueError(f"No rows found for env={env}, M={m}, eps={epsilon:g}")

    plot_df[lr_key] = pd.to_numeric(plot_df[lr_key], errors="coerce")

    lrs_arr = np.array(list(lrs), dtype=float)
    lr_ok = np.isclose(plot_df[lr_key].to_numpy(dtype=float), lrs_arr[:, None], atol=0.0, rtol=1e-12).any(axis=0)

    keep = lr_ok & (plot_df["algo"] == "REMAX")
    if include_sac:
        keep |= (plot_df["algo"] == "SAC")
    plot_df = plot_df.loc[keep].copy()

    lr_style = _build_lr_style_map(lrs, lr_encode=lr_encode)

    # --- compute reference values: default_lr + reference_epsilon ---
    ref_max_return = None
    ref_min_entropy = None
    if add_default_reference and (default_lr is not None):
        ref_df = _prepare_plot_df(
            df,
            ms=[m],
            epsilons=[reference_epsilon],
            include_sac=False,
            use_default_lr=False,   # ここは明示的に default_lr で絞るのでOFF
            lr_key=lr_key,
        )
        ref_df = ref_df[ref_df["env"] == env].copy()
        if len(ref_df) > 0:
            ref_df[lr_key] = pd.to_numeric(ref_df[lr_key], errors="coerce")
            mask_lr = np.isclose(ref_df[lr_key].to_numpy(dtype=float), float(default_lr), atol=0.0, rtol=1e-12)
            ref_df = ref_df.loc[(ref_df["algo"] == "REMAX") & mask_lr].copy()

        if len(ref_df) > 0:
            # use mean curve (across seeds) then take max/min over steps
            x_r, y_r = _summarize_by_step(ref_df, "eval/return")
            x_e, y_e = _summarize_by_step(ref_df, "train/entropy")
            if y_r.shape[0] > 0:
                ref_max_return = float(np.nanmax(y_r[:, 0]))
            if y_e.shape[0] > 0:
                ref_min_entropy = float(np.nanmin(y_e[:, 0]))

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), sharex=False, sharey=False)

    for ax, metric in zip(axes, ["eval/return", "train/entropy"]):
        # SAC (optional)
        if include_sac:
            df_sac = plot_df[plot_df["algo"] == "SAC"]
            df_sac = df_sac[df_sac["_step"] <= 3_000_000]
            if len(df_sac) > 0:
                x, y = _summarize_by_step(df_sac, metric)
                sac_style = Style((0, 0, 0, 1), "-", 0.95, 2.6)
                _plot_mean_se(ax, x, y, "SAC", sac_style, show_band=show_band)

        # ReMAC at different lrs
        for lr in sorted({float(x) for x in lrs}):
            mask = (plot_df["algo"] == "REMAX") & np.isclose(
                plot_df[lr_key].to_numpy(dtype=float), lr, atol=0.0, rtol=1e-12
            )
            df_lr = plot_df.loc[mask]
            if len(df_lr) == 0:
                continue
            x, y = _summarize_by_step(df_lr, metric)
            st = lr_style[lr]
            _plot_mean_se(ax, x, y, rf"lr={lr:g}", st, show_band=show_band)

        # reference lines (red)
        if metric == "eval/return" and (ref_max_return is not None):
            ax.axhline(
                ref_max_return,
                color="red",
                linestyle=reference_linestyle,
                linewidth=reference_linewidth,
                alpha=reference_alpha,
                zorder=0,
            )
            if annotate_reference:
                ax.text(
                    0.02,
                    0.05,
                    rf"ref: max return @ default lr={default_lr:g}, eps={_fmt_eps(reference_epsilon)}"
                    + f"\n= {ref_max_return:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                    va="bottom",
                    ha="left",
                )

        if metric == "train/entropy" and (ref_min_entropy is not None):
            ax.axhline(
                ref_min_entropy,
                color="red",
                linestyle=reference_linestyle,
                linewidth=reference_linewidth,
                alpha=reference_alpha,
                zorder=0,
            )
            if annotate_reference:
                ax.text(
                    0.02,
                    0.05,
                    rf"ref: min entropy @ default lr={default_lr:g}, eps={_fmt_eps(reference_epsilon)}"
                    + f"\n= {ref_min_entropy:.3g}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                    va="bottom",
                    ha="left",
                )

        #ax.set_title(_metric_short(metric), fontsize=18)
        ax.set_xlabel("step")
        ax.set_ylabel(_metric_short(metric))
        ax.grid(True, alpha=0.25)

    # shared legend (lr only)
    handles: List[Line2D] = []
    if include_sac:
        handles.append(Line2D([0], [0], color="k", lw=LEGEND_HANDLE_LW, ls="-", label="SAC"))
    for lr in sorted({float(x) for x in lrs}):
        st = lr_style[lr]
        handles.append(
            Line2D([0], [0], color=st.color, lw=LEGEND_HANDLE_LW, ls=st.linestyle, alpha=st.alpha, label=rf"lr={lr:g}")
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    #fig.suptitle(rf"{env.replace('brax/','')}: ReMAC $m={m}$, $\epsilon={epsilon:g}$", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_6envs_2x3_update_over_grad_lr(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    *,
    vary_eps_alpha: bool = True,
    include_sac: bool = True,
    include_other_baselines: bool = True,
    include_eps_in_label: bool = False,
    alpha_fixed: float = 0.80,
    show_band: bool = True,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    # --- NEW: lr filter ---
    lrs: Optional[Sequence[float]] = None,
    lr_rtol: float = 1e-12,
    # ----------------------
    env_order: Sequence[str] = (
        "brax/ant",
        "brax/halfcheetah",
        "brax/hopper",
        "brax/walker2d",
        "brax/reacher",
        "brax/swimmer",
    ),
    out_dir: str = "fig",
    filename_prefix: str = "paper_2x3_update_over_grad_lr",
    # metric keys
    grad_key: str = "train/actor_grad_norm",
    update_key: str = "train/actor_update_norm",
    out_key: str = "train/update_over_grad_lr",
) -> None:
    """
    6 envs (3 columns x 2 rows) で
      actor_update_norm / (actor_grad_norm * learning_rate)
    を描く。
    ms/eps の指定は plot_6envs_2x3 と同じ。lrs で learning rate を明示フィルタ可能。
    """
    _paper_rcparams()

    required = {"env", "algo_label", "_step", lr_key, grad_key, update_key}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    plot_df = _prepare_plot_df(
        df,
        ms=ms,
        epsilons=epsilons,
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        lrs=lrs,
        lr_rtol=lr_rtol,
    )

    # build ratio column
    lr = pd.to_numeric(plot_df[lr_key], errors="coerce")
    g = pd.to_numeric(plot_df[grad_key], errors="coerce")
    u = pd.to_numeric(plot_df[update_key], errors="coerce")
    denom = g * lr
    ratio = u / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    plot_df[out_key] = ratio

    # reuse existing 2x3 plotter (metrics は out_key だけ)
    plot_6envs_2x3(
        plot_df,
        epsilons=epsilons,
        ms=ms,
        metrics=[out_key],
        vary_eps_alpha=vary_eps_alpha,
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        include_eps_in_label=include_eps_in_label,
        alpha_fixed=alpha_fixed,
        show_band=show_band,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        lrs=lrs,
        lr_rtol=lr_rtol,
        env_order=env_order,
        out_dir=out_dir,
        filename_prefix=filename_prefix,
        log_scale=True,
    )
    # y axis をlog scaleにしたい


# ============================================================
# 6) Damping diagnostics: gradient / update / Adam normalization
# ============================================================
DAMPING_KINDS: Tuple[str, ...] = (
    "grad_norm",            # raw ||grad|| (smaller for larger M => damping)
    "update_norm",          # raw ||update||
    "update_per_lr",        # ||update|| / lr  (effective step size)
    "update_per_grad",      # ||update|| / ||grad||  (Adam amplification)
    "update_per_grad_lr",   # ||update|| / (||grad|| * lr)  (1.0 = SGD; >1 = Adam normalizes)
)


def _add_damping_metric(
    df: pd.DataFrame,
    kind: str,
    *,
    lr_key: str = "learning_rate",
    grad_key: str = "train/actor_grad_norm",
    update_key: str = "train/actor_update_norm",
) -> str:
    """Add a derived damping column to df (in-place). Returns the resulting column name."""
    if kind == "grad_norm":
        return grad_key
    if kind == "update_norm":
        return update_key

    lr = pd.to_numeric(df[lr_key], errors="coerce")
    g = pd.to_numeric(df[grad_key], errors="coerce")
    u = pd.to_numeric(df[update_key], errors="coerce")

    if kind == "update_per_lr":
        out_key = "train/update_per_lr"
        df[out_key] = u / lr
    elif kind == "update_per_grad":
        out_key = "train/update_per_grad"
        df[out_key] = (u / g).replace([np.inf, -np.inf], np.nan)
    elif kind == "update_per_grad_lr":
        out_key = "train/update_per_grad_lr"
        df[out_key] = (u / (g * lr)).replace([np.inf, -np.inf], np.nan)
    else:
        raise ValueError(f"unknown damping kind: {kind!r}. Allowed: {DAMPING_KINDS}")
    return out_key


def plot_damping_2x3(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    *,
    kind: str = "grad_norm",
    include_sac: bool = False,
    include_other_baselines: bool = True,
    include_eps_in_label: bool = False,
    alpha_fixed: float = 0.80,
    show_band: bool = True,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    lrs: Optional[Sequence[float]] = None,
    log_scale: bool = True,
    env_order: Sequence[str] = (
        "brax/ant",
        "brax/halfcheetah",
        "brax/hopper",
        "brax/walker2d",
        "brax/reacher",
        "brax/swimmer",
    ),
    out_dir: str = "fig",
    filename_prefix: str = "paper_damping",
) -> None:
    """
    Damping diagnostics across 6 envs (2x3), aggregated over seeds.

    Reads `train/actor_grad_norm` and `train/actor_update_norm` from `df`
    (must be present), combines them with `learning_rate` as needed, and
    plots one of:

      - grad_norm          : raw ||g||
      - update_norm        : raw ||u||
      - update_per_lr      : ||u|| / lr            (effective step per parameter)
      - update_per_grad    : ||u|| / ||g||         (Adam's amplification of g)
      - update_per_grad_lr : ||u|| / (||g|| * lr)  (1.0 = SGD; >1 = Adam normalizing)

    Use `ms`/`epsilons` to slice the (M, ε) configurations: e.g. ms=[1,2,4,8] +
    epsilons=[1e-8] sweeps M; ms=[4] + epsilons=[1e-8,1e-2,1e-1,1] sweeps ε.
    """
    grad_key = "train/actor_grad_norm"
    update_key = "train/actor_update_norm"
    required = {"env", "algo_label", "_step", lr_key, grad_key, update_key}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    df = df.copy()
    out_key = _add_damping_metric(df, kind, lr_key=lr_key, grad_key=grad_key, update_key=update_key)

    plot_6envs_2x3(
        df,
        ms=ms,
        epsilons=epsilons,
        metrics=[out_key],
        include_sac=include_sac,
        include_other_baselines=include_other_baselines,
        include_eps_in_label=include_eps_in_label,
        alpha_fixed=alpha_fixed,
        show_band=show_band,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        lrs=lrs,
        env_order=env_order,
        out_dir=out_dir,
        filename_prefix=f"{filename_prefix}_{kind}",
        log_scale=log_scale,
    )


# ============================================================
# 6b) Combined damping overlay: ||grad|| (solid, left) and
#     ||u||/(||grad||*lr) (dashed, right) on the same axes (twin-y).
# ============================================================
def plot_damping_combined_2x3(
    df: pd.DataFrame,
    epsilons: Sequence[float],
    ms: Sequence[int],
    *,
    alpha_fixed: float = 0.9,
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lr_key: str = "learning_rate",
    lrs: Optional[Sequence[float]] = None,
    log_scale: bool = True,
    env_order: Sequence[str] = (
        "brax/ant",
        "brax/halfcheetah",
        "brax/hopper",
        "brax/walker2d",
        "brax/reacher",
        "brax/swimmer",
    ),
    out_dir: str = "fig",
    filename_prefix: str = "paper_damping_combined",
) -> None:
    """
    For each env, overlay (solid) raw ||grad|| and (dashed) Adam normalization
    ||u||/(||grad||*lr) on twin y-axes. Color = M, shared between the two lines
    so the inverse relationship (grad shrinks ↔ Adam factor grows) is visible
    at a glance.
    """
    grad_key = "train/actor_grad_norm"
    update_key = "train/actor_update_norm"
    ratio_key = "train/update_per_grad_lr"
    required = {"env", "algo_label", "_step", lr_key, grad_key, update_key}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    _paper_rcparams()
    df = df.copy()
    _add_damping_metric(df, "update_per_grad_lr", lr_key=lr_key, grad_key=grad_key, update_key=update_key)

    plot_df = _prepare_plot_df(
        df,
        ms=ms,
        epsilons=epsilons,
        include_sac=False,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        lrs=lrs,
    )

    style_map, _, _ = _build_style_maps(ms, epsilons, vary_eps_alpha=False, alpha_fixed=alpha_fixed)

    envs_present = [e for e in env_order if e in set(plot_df["env"].unique())]
    if not envs_present:
        raise ValueError("None of the requested envs are present.")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4), sharex=False, sharey=False)
    axes = axes.ravel()

    ms_sorted = sorted({int(x) for x in ms})
    eps_sorted = sorted({float(x) for x in epsilons})

    for i, env in enumerate(envs_present[:6]):
        ax = axes[i]
        ax2 = ax.twinx()
        df_env = plot_df[plot_df["env"] == env]

        for m in ms_sorted:
            for eps in eps_sorted:
                mask = (
                    (df_env["algo"] == "REMAX")
                    & (df_env["m"] == m)
                    & np.isclose(df_env["eps"].to_numpy(dtype=float), eps, atol=0.0, rtol=1e-12)
                )
                df_sub = df_env.loc[mask]
                if len(df_sub) == 0:
                    continue
                st = style_map[(m, eps)]

                x_g, y_g = _summarize_by_step(df_sub, grad_key)
                ax.plot(x_g, y_g[:, 0], color=st.color, ls="-", lw=st.linewidth, alpha=st.alpha)

                x_r, y_r = _summarize_by_step(df_sub, ratio_key)
                ax2.plot(x_r, y_r[:, 0], color=st.color, ls="--", lw=st.linewidth, alpha=st.alpha)

        ax.set_title(env.replace("brax/", ""))
        ax.set_xlabel("step")
        ax.set_ylabel(r"$\|\nabla\|$ (solid)")
        ax2.set_ylabel(r"$\|u\| / (\|\nabla\|\cdot\mathrm{lr})$ (dashed)")
        ax.grid(True, alpha=0.25)
        if log_scale:
            ax.set_yscale("log")
            ax2.set_yscale("log")

    for j in range(len(envs_present[:6]), 6):
        axes[j].axis("off")

    handles: List[Line2D] = []
    eps0 = eps_sorted[0]
    for m in ms_sorted:
        st = style_map[(m, eps0)]
        handles.append(
            Line2D([0], [0], color=st.color, lw=LEGEND_HANDLE_LW, alpha=1.0, label=rf"ReMac $M={m}$")
        )
    handles.append(
        Line2D([0], [0], color="0.3", lw=LEGEND_HANDLE_LW, ls="-", label=r"solid: $\|\nabla\|$")
    )
    handles.append(
        Line2D([0], [0], color="0.3", lw=LEGEND_HANDLE_LW, ls="--",
               label=r"dashed: $\|u\|/(\|\nabla\|\cdot\mathrm{lr})$")
    )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(6, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path = os.path.join(
        out_dir, f"{filename_prefix}_m{list(ms)}_eps{list(epsilons)}.pdf"
    )
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 4) lr sweep at multiple epsilons (rows = eps, cols = return/entropy)
# ============================================================
def _add_eps_tiered_legend(
    fig,
    *,
    eps_sorted: Sequence[float],
    lrs_by_eps: Dict[float, Sequence[float]],
    style_lookup,
    include_sac: bool,
    x_axis: str,
    nrows: int,
) -> float:
    """
    Add one legend tier per eps, stacked at the bottom (largest eps on top).
    Each tier is titled with its eps and lists only the lrs present for that eps.
    `style_lookup(eps, lr) -> Style` provides the colour/linestyle per entry.
    Returns the figure-fraction bottom margin to reserve via tight_layout.
    """
    def _eps_handles(eps: float) -> List[Line2D]:
        hs: List[Line2D] = []
        if include_sac and x_axis == "step":
            hs.append(Line2D([0], [0], color="k", lw=LEGEND_HANDLE_LW, ls="-", label="SAC"))
        for lr in lrs_by_eps.get(eps, []):
            st = style_lookup(eps, float(lr))
            hs.append(
                Line2D([0], [0], color=st.color, lw=LEGEND_HANDLE_LW, ls=st.linestyle,
                       alpha=st.alpha, label=rf"lr={float(lr):g}")
            )
        return hs

    n_tiers = len(eps_sorted)
    tier_h = 0.12 / max(1, nrows) * 2   # scale room by figure height (nrows)
    bottom_pad = 0.02
    y_positions = [bottom_pad + (n_tiers - 1 - r) * tier_h for r in range(n_tiers)]
    for r, eps in enumerate(eps_sorted):
        hs = _eps_handles(eps)
        if not hs:
            continue
        leg = fig.legend(
            handles=hs,
            loc="lower center",
            ncol=len(hs),
            frameon=False,
            bbox_to_anchor=(0.5, y_positions[r]),
            title=rf"$\epsilon={_fmt_eps(eps)}$",
            columnspacing=1.4,
        )
        leg._legend_box.align = "center"
        fig.add_artist(leg)

    return bottom_pad + n_tiers * tier_h


def plot_lr_sweep_eps_grid(
    df: pd.DataFrame,
    *,
    env: str,
    m: int,
    epsilons: Sequence[float] = (1.0, 1e-8),
    lrs: Sequence[float] = (1e-4, 5e-5, 3e-5, 1e-5),
    lr_key: str = "learning_rate",
    lr_encode: str = "both",
    include_sac: bool = False,
    include_other_baselines: bool = True,
    show_band: bool = True,
    out_dir: str = "fig",
    filename_prefix: str = "paper_lr_sweep_eps_grid",
    x_axis: str = "step",
) -> None:
    """
    (env, m) 固定で，epsilon を変えた lr sweep を 1 枚に並べる．
    Rows = epsilon (e.g. 1, 1e-8), Cols = (eval/return, train/entropy)

    x_axis: "step" (default) or "cum_dtheta" (per-seed cumulative ||Δθ||).
    cum_dtheta absorbs (ε, lr) differences in effective step size so curves can be
    compared on equal "distance travelled in parameter space".
    """
    _paper_rcparams()
    # Slightly larger labels for this figure (~1.4x).
    _label_scale = 1.4
    plt.rcParams.update(
        {
            "font.size": 11 * _label_scale,
            "axes.titlesize": 18 * _label_scale,
            "axes.labelsize": 12 * _label_scale,
            "xtick.labelsize": 10 * _label_scale,
            "ytick.labelsize": 10 * _label_scale,
            "legend.fontsize": 13 * _label_scale,
            "legend.title_fontsize": 13 * _label_scale,
        }
    )

    if x_axis not in ("step", "cum_dtheta"):
        raise ValueError(f"x_axis must be 'step' or 'cum_dtheta', got {x_axis!r}")

    required = {"env", "algo_label", "_step", lr_key, "eval/return", "train/entropy"}
    if x_axis == "cum_dtheta":
        required = required | {"train/actor_update_norm", "seed_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    env_canon = _canon_env(env)
    eps_sorted = sorted({float(e) for e in epsilons}, reverse=True)
    nrows = len(eps_sorted)

    fig, axes = plt.subplots(nrows, 2, figsize=(7.0, 3.4 * nrows), sharex=False, sharey=False)
    if nrows == 1:
        axes = np.array([axes])

    # Pre-pass: filter each eps block once and collect the lrs present per eps.
    # Curves are coloured by epsilon family (eps=1 -> reds, eps=1e-8 -> blues),
    # with the lr picked out by shade; the lr shared across eps (the default lr)
    # gets one common colour.
    blocks: Dict[float, pd.DataFrame] = {}
    lrs_by_eps: Dict[float, List[float]] = {}
    for eps in eps_sorted:
        plot_df = _prepare_plot_df(
            df,
            ms=[m],
            epsilons=[eps],
            include_sac=include_sac,
            include_other_baselines=include_other_baselines,
            use_default_lr=False,
            lr_key=lr_key,
        )
        plot_df = plot_df[plot_df["env"] == env_canon].copy()
        if len(plot_df) == 0:
            blocks[eps] = plot_df
            lrs_by_eps[eps] = []
            continue

        plot_df[lr_key] = pd.to_numeric(plot_df[lr_key], errors="coerce")
        lrs_arr = np.array(list(lrs), dtype=float)
        lr_ok = np.isclose(plot_df[lr_key].to_numpy(dtype=float), lrs_arr[:, None], atol=0.0, rtol=1e-12).any(axis=0)
        keep = lr_ok & (plot_df["algo"] == "REMAX")
        if include_sac:
            keep |= (plot_df["algo"] == "SAC")
        plot_df = plot_df.loc[keep].copy()

        if x_axis == "cum_dtheta":
            plot_df = _add_cum_dtheta(plot_df)

        blocks[eps] = plot_df
        re_lrs = pd.to_numeric(
            plot_df.loc[plot_df["algo"] == "REMAX", lr_key], errors="coerce"
        ).dropna().unique()
        lrs_by_eps[eps] = sorted(float(x) for x in re_lrs)

    common_lrs = (
        set.intersection(*[set(lrs_by_eps[e]) for e in eps_sorted]) if eps_sorted else set()
    )
    eps_lr_style = _build_eps_lr_style_map(eps_sorted, lrs_by_eps, common_lrs=common_lrs)

    for r, eps in enumerate(eps_sorted):
        plot_df = blocks[eps]
        if len(plot_df) == 0:
            for c in range(2):
                axes[r, c].set_title(rf"$\epsilon={_fmt_eps(eps)}$ (no data)")
                axes[r, c].grid(True, alpha=0.25)
            continue

        for c, metric in enumerate(["eval/return", "train/entropy"]):
            ax = axes[r, c]
            if include_sac and x_axis == "step":
                df_sac = plot_df[plot_df["algo"] == "SAC"]
                df_sac = df_sac[df_sac["_step"] <= 3_000_000]
                if len(df_sac) > 0:
                    x, y = _summarize_by_step(df_sac, metric)
                    sac_style = Style((0, 0, 0, 1), "-", 0.95, 2.6)
                    _plot_mean_se(ax, x, y, "SAC", sac_style, show_band=show_band)

            for lr in lrs_by_eps[eps]:
                mask = (plot_df["algo"] == "REMAX") & np.isclose(
                    plot_df[lr_key].to_numpy(dtype=float), lr, atol=0.0, rtol=1e-12
                )
                df_lr = plot_df.loc[mask]
                if len(df_lr) == 0:
                    continue
                if x_axis == "step":
                    x, y = _summarize_by_step(df_lr, metric)
                else:
                    x, y = _summarize_xy_by_step(df_lr, x_col="cum_dtheta", y_col=metric)
                st = eps_lr_style[(float(eps), float(lr))]
                _plot_mean_se(ax, x, y, rf"lr={lr:g}", st, show_band=show_band)

            ax.set_xlabel("step" if x_axis == "step" else r"$\sum_t \|\Delta\theta_t\|$")
            ax.set_ylabel(_metric_short(metric))
            ax.grid(True, alpha=0.25)
            if x_axis == "cum_dtheta":
                ax.set_xscale("log")
            if c == 0:
                ax.set_title(rf"$\epsilon={_fmt_eps(eps)}$")

    # Shared eps-tiered legend (eps=1 on top, eps=1e-8 below).
    bottom = _add_eps_tiered_legend(
        fig,
        eps_sorted=eps_sorted,
        lrs_by_eps=lrs_by_eps,
        style_lookup=lambda e, lr: eps_lr_style[(float(e), float(lr))],
        include_sac=include_sac,
        x_axis=x_axis,
        nrows=nrows,
    )

    fig.tight_layout(rect=(0, bottom, 1, 1))
    env_slug = env_canon.replace("brax/", "")
    x_suffix = "" if x_axis == "step" else "_xCumDtheta"
    out_path = os.path.join(out_dir, f"{filename_prefix}_{env_slug}_m{m}{x_suffix}.pdf")
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_lr_sweep_eps_grid_multi(
    df: pd.DataFrame,
    *,
    envs: Sequence[str],
    m: int,
    epsilons: Sequence[float] = (1.0, 1e-8),
    lrs: Sequence[float] = (1e-4, 5e-5, 3e-5, 1e-5),
    metrics: Sequence[str] = ("eval/return", "train/entropy"),
    lr_key: str = "learning_rate",
    include_sac: bool = False,
    include_other_baselines: bool = True,
    show_band: bool = True,
    out_dir: str = "fig",
    filename_prefix: str = "paper_lr_sweep_eps_grid_multi",
    x_axis: str = "step",
) -> None:
    """
    Several envs side by side, ONE figure *per metric* (return / entropy plotted
    separately so the panels stay readable), with a single shared eps-tiered legend.

    Per-metric layout: rows = epsilon (e.g. 1, 1e-8); columns = envs.
    Curves are coloured by epsilon family (eps=1 -> reds, eps=1e-8 -> blues) with
    the lr picked out by shade, so the figure reads as two calm colour groups.
    Env names title each column; each eps row is labelled on the left.
    """
    _paper_rcparams()
    _label_scale = 1.4
    plt.rcParams.update(
        {
            "font.size": 11 * _label_scale,
            "axes.titlesize": 18 * _label_scale,
            "axes.labelsize": 12 * _label_scale,
            "xtick.labelsize": 10 * _label_scale,
            "ytick.labelsize": 10 * _label_scale,
            "legend.fontsize": 13 * _label_scale,
            "legend.title_fontsize": 13 * _label_scale,
        }
    )

    if x_axis not in ("step", "cum_dtheta"):
        raise ValueError(f"x_axis must be 'step' or 'cum_dtheta', got {x_axis!r}")

    metrics = list(metrics)
    required = {"env", "algo_label", "_step", lr_key} | set(metrics)
    if x_axis == "cum_dtheta":
        required = required | {"train/actor_update_norm", "seed_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    envs_canon = [_canon_env(e) for e in envs]
    eps_sorted = sorted({float(e) for e in epsilons}, reverse=True)
    nrows = len(eps_sorted)
    n_env = len(envs_canon)

    # ---- pre-pass: filter each (eps, env) block once and collect present lrs ----
    blocks: Dict[Tuple[float, str], pd.DataFrame] = {}
    lrs_by_eps: Dict[float, set] = {eps: set() for eps in eps_sorted}
    for eps in eps_sorted:
        for env_canon in envs_canon:
            plot_df = _prepare_plot_df(
                df,
                ms=[m],
                epsilons=[eps],
                include_sac=include_sac,
                include_other_baselines=include_other_baselines,
                use_default_lr=False,
                lr_key=lr_key,
            )
            plot_df = plot_df[plot_df["env"] == env_canon].copy()
            if len(plot_df) == 0:
                blocks[(eps, env_canon)] = plot_df
                continue
            plot_df[lr_key] = pd.to_numeric(plot_df[lr_key], errors="coerce")
            lrs_arr = np.array(list(lrs), dtype=float)
            lr_ok = np.isclose(
                plot_df[lr_key].to_numpy(dtype=float), lrs_arr[:, None], atol=0.0, rtol=1e-12
            ).any(axis=0)
            keep = lr_ok & (plot_df["algo"] == "REMAX")
            if include_sac:
                keep |= (plot_df["algo"] == "SAC")
            plot_df = plot_df.loc[keep].copy()
            if x_axis == "cum_dtheta":
                plot_df = _add_cum_dtheta(plot_df)
            blocks[(eps, env_canon)] = plot_df
            re_lrs = pd.to_numeric(
                plot_df.loc[plot_df["algo"] == "REMAX", lr_key], errors="coerce"
            ).dropna().unique()
            lrs_by_eps[eps].update(float(x) for x in re_lrs)

    lrs_by_eps_sorted = {eps: sorted(v) for eps, v in lrs_by_eps.items()}
    # lrs shared across eps (e.g. the default lr) get one common colour in every tier.
    common_lrs = (
        set.intersection(*[set(lrs_by_eps_sorted[e]) for e in eps_sorted])
        if eps_sorted else set()
    )
    eps_lr_style = _build_eps_lr_style_map(
        eps_sorted, lrs_by_eps_sorted, common_lrs=common_lrs
    )
    title_fs = plt.rcParams["axes.titlesize"]
    envs_slug = "-".join(e.replace("brax/", "") for e in envs_canon)
    x_suffix = "" if x_axis == "step" else "_xCumDtheta"

    # ---- one figure per metric (return / entropy separately) ----
    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, axes = plt.subplots(
            nrows, n_env, figsize=(4.2 * n_env, 3.4 * nrows), squeeze=False
        )

        for j, env_canon in enumerate(envs_canon):
            for r, eps in enumerate(eps_sorted):
                ax = axes[r, j]
                plot_df = blocks[(eps, env_canon)]
                ax.grid(True, alpha=0.25)
                if len(plot_df) == 0:
                    continue

                if include_sac and x_axis == "step":
                    df_sac = plot_df[plot_df["algo"] == "SAC"]
                    df_sac = df_sac[df_sac["_step"] <= 3_000_000]
                    if len(df_sac) > 0:
                        x, y = _summarize_by_step(df_sac, metric)
                        sac_style = Style((0, 0, 0, 1), "-", 0.95, 2.6)
                        _plot_mean_se(ax, x, y, "SAC", sac_style, show_band=show_band)

                for lr in lrs_by_eps_sorted[eps]:
                    mask = (plot_df["algo"] == "REMAX") & np.isclose(
                        plot_df[lr_key].to_numpy(dtype=float), lr, atol=0.0, rtol=1e-12
                    )
                    df_lr = plot_df.loc[mask]
                    if len(df_lr) == 0:
                        continue
                    if x_axis == "step":
                        x, y = _summarize_by_step(df_lr, metric)
                    else:
                        x, y = _summarize_xy_by_step(df_lr, x_col="cum_dtheta", y_col=metric)
                    st = eps_lr_style[(float(eps), float(lr))]
                    _plot_mean_se(ax, x, y, rf"lr={lr:g}", st, show_band=show_band)

                ax.set_xlabel("step" if x_axis == "step" else r"$\sum_t \|\Delta\theta_t\|$")
                ax.set_ylabel(_metric_short(metric))
                if x_axis == "cum_dtheta":
                    ax.set_xscale("log")

        # Shared eps-tiered legend (eps=1 reds on top, eps=1e-8 blues below).
        bottom = _add_eps_tiered_legend(
            fig,
            eps_sorted=eps_sorted,
            lrs_by_eps=lrs_by_eps_sorted,
            style_lookup=lambda e, lr: eps_lr_style[(float(e), float(lr))],
            include_sac=include_sac,
            x_axis=x_axis,
            nrows=nrows,
        )

        left_pad = 0.06
        top_pad = 0.91
        fig.tight_layout(rect=(left_pad, bottom, 1, top_pad))

        # env-name titles (one per column) and eps row labels (rotated, left).
        fig.canvas.draw()
        for j, env_canon in enumerate(envs_canon):
            p = axes[0, j].get_position()
            xc = (p.x0 + p.x1) / 2.0
            fig.text(xc, min(0.99, p.y1 + 0.03), env_canon.replace("brax/", ""),
                     ha="center", va="bottom", fontsize=title_fs)
        for r, eps in enumerate(eps_sorted):
            p = axes[r, 0].get_position()
            yc = (p.y0 + p.y1) / 2.0
            fig.text(0.012, yc, rf"$\epsilon={_fmt_eps(eps)}$",
                     ha="left", va="center", rotation=90, fontsize=title_fs)

        metric_slug = _metric_short(metric)
        out_path = os.path.join(
            out_dir, f"{filename_prefix}_{envs_slug}_m{m}_{metric_slug}{x_suffix}.pdf"
        )
        _ensure_dir(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ============================================================
# 5) B (remax_num_samples) ablation per m
# ============================================================
def _build_b_style_map(
    bs: Sequence[int],
    *,
    alpha_fixed: float = 0.9,
    linewidth: float = 2.2,
) -> Dict[int, Style]:
    bs_sorted = sorted({int(b) for b in bs})
    tab = plt.get_cmap("tab10")
    return {
        b: Style(color=tab(i % 10), linestyle="-", alpha=alpha_fixed, linewidth=linewidth)
        for i, b in enumerate(bs_sorted)
    }


def plot_b_ablation_per_m(
    df: pd.DataFrame,
    *,
    env: str,
    epsilon: float,
    ms: Sequence[int] = (1, 2, 4, 8),
    bs: Sequence[int] = (8, 16),
    lr_key: str = "learning_rate",
    use_default_lr: bool = True,
    env_lr_map: Optional[Dict[str, float]] = None,
    lrs: Optional[Sequence[float]] = None,
    show_band: bool = True,
    metric: str = "eval/return",
    out_dir: str = "fig",
    filename_prefix: str = "paper_b_ablation",
) -> None:
    """
    1 row × len(ms) cols. 各 subplot は固定 m について B (remax_num_samples) を変えた ablation．
    """
    _paper_rcparams()
    # Enlarge all text labels (titles, axis labels, ticks, legend) ~1.8x for the
    # ablation figure. Safe to mutate rcParams here: each plot.py invocation is a
    # separate process, and this runs before plt.subplots() below.
    _label_scale = 1.8
    plt.rcParams.update(
        {
            "font.size": 11 * _label_scale,
            "axes.titlesize": 18 * _label_scale,
            "axes.labelsize": 12 * _label_scale,
            "xtick.labelsize": 10 * _label_scale,
            "ytick.labelsize": 10 * _label_scale,
            "legend.fontsize": 13 * _label_scale,
        }
    )

    if "remax_num_samples" not in df.columns:
        raise ValueError(
            "df has no column 'remax_num_samples'. "
            "Re-run fetch_wandb_data_with_seeds so that remax_num_samples is included."
        )

    env_canon = _canon_env(env)

    plot_df = _prepare_plot_df(
        df,
        ms=ms,
        epsilons=[epsilon],
        include_sac=False,
        use_default_lr=use_default_lr,
        env_lr_map=env_lr_map,
        lr_key=lr_key,
        lrs=lrs,
    )
    plot_df = plot_df[plot_df["env"] == env_canon].copy()
    if len(plot_df) == 0:
        raise ValueError(f"No rows found for env={env_canon}, eps={epsilon:g}")

    plot_df["remax_num_samples"] = pd.to_numeric(plot_df["remax_num_samples"], errors="coerce")
    bs_set = {int(b) for b in bs}
    plot_df = plot_df[plot_df["remax_num_samples"].isin(bs_set)].copy()

    ms_sorted = sorted({int(x) for x in ms})
    n = len(ms_sorted)

    b_style = _build_b_style_map(bs)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6), sharex=False, sharey=False)
    axes = np.atleast_1d(axes)

    for i, m in enumerate(ms_sorted):
        ax = axes[i]
        for b in sorted(bs_set):
            mask = (
                (plot_df["algo"] == "REMAX")
                & (plot_df["m"] == m)
                & (plot_df["remax_num_samples"] == b)
            )
            df_sub = plot_df.loc[mask]
            if len(df_sub) == 0:
                continue
            x, y = _summarize_by_step(df_sub, metric)
            st = b_style[b]
            _plot_mean_se(ax, x, y, rf"$B={b}$", st, show_band=show_band)

        ax.set_title(rf"$M={m}$")
        ax.set_xlabel("step")
        if i == 0:
            ax.set_ylabel(_metric_short(metric))
        ax.grid(True, alpha=0.25)

    handles: List[Line2D] = []
    for b in sorted(bs_set):
        st = b_style[b]
        handles.append(
            Line2D([0], [0], color=st.color, lw=LEGEND_HANDLE_LW, ls=st.linestyle, alpha=st.alpha, label=rf"$B={b}$")
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    env_slug = env_canon.replace("brax/", "")
    metric_slug = _metric_short(metric)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{env_slug}_eps{epsilon:g}_{metric_slug}.pdf")
    _ensure_dir(out_path)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# CLI
# ============================================================
def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_int_map(s: str) -> Dict[int, int]:
    """Parse 'm:B' pairs, e.g. '1:8,2:8,4:8,8:16' -> {1:8,2:8,4:8,8:16}."""
    out: Dict[int, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        out[int(k)] = int(v)
    return out


def _slug_value(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, dict):
        return "-".join(f"{_slug_value(k)}.{_slug_value(val)}" for k, val in sorted(v.items()))
    if isinstance(v, (list, tuple)):
        return "-".join(_slug_value(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _make_prefix(plot_type: str, **kwargs) -> str:
    """Build a filename prefix that reflects the chosen settings."""
    parts = [plot_type]
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        if isinstance(v, bool) and not v:
            continue
        parts.append(f"{k}{_slug_value(v)}")
    return "_".join(parts)


def _load_or_fetch(project: str, entity: Optional[str], samples: int, cache: Optional[str]) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        print(f"Loading cached data from {cache}")
        return pd.read_pickle(cache)
    df = fetch_wandb_data_with_seeds(project, entity=entity, samples=samples)
    if cache:
        _ensure_dir(cache)
        df.to_pickle(cache)
        print(f"Cached data to {cache}")
    return df


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", required=True, help="wandb project name")
    p.add_argument("--entity", default=None, help="wandb entity (optional)")
    p.add_argument("--cache", default=None, help="Optional parquet cache path for fetched df")
    p.add_argument("--samples", type=int, default=500, help="wandb history samples per run")
    p.add_argument("--out-dir", default="fig", help="output directory (default: fig)")


def _add_env_grid_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by the "2x3" and "2x4" env-grid subcommands."""
    p.add_argument("--metrics", default="eval/return", help="comma-separated metric names")
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--epsilons", type=_parse_floats, default=[1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=None, help="optional explicit lr filter")
    p.add_argument("--bs", type=_parse_ints, default=None,
                   help="optional remax_num_samples (B) filter, e.g. '8' keeps only B=8 ReMAC runs")
    p.add_argument("--bs-per-m", type=_parse_int_map, default=None,
                   help="per-m B selection 'm:B,...', e.g. '1:8,2:8,4:8,8:16' (overrides --bs for listed m)")
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--baselines", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()],
                   default=None,
                   help="explicit baseline subset to include, e.g. 'SAC,PPO' (needs --include-sac; overrides --no-other-baselines)")
    p.add_argument("--include-eps-label", action="store_true")
    p.add_argument("--no-default-lr", action="store_true", help="disable env-default lr filter")
    p.add_argument("--log-scale", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot ReMAC experiments. Each plot type is a subcommand; outputs are saved under --out-dir with settings in the filename.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 2x3 / 2x4 envs (identical arguments; 2x4 adds humanoidstandup and pusher)
    for _cmd, _help in (("2x3", "6 envs in a 2x3 grid (per metric)"),
                        ("2x4", "8 envs in a 2x4 grid (per metric)")):
        p = sub.add_parser(_cmd, help=_help)
        _add_common_args(p)
        _add_env_grid_args(p)

    # 4 envs row
    p = sub.add_parser("4row", help="4 envs in a row (per metric)")
    _add_common_args(p)
    p.add_argument("--metrics", default="eval/return")
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--epsilons", type=_parse_floats, default=[1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=None)
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--include-eps-label", action="store_true")
    p.add_argument("--no-default-lr", action="store_true")

    # lr-sweep (single eps)
    p = sub.add_parser("lr-sweep", help="lr sweep for fixed env, m, epsilon (return + entropy)")
    _add_common_args(p)
    p.add_argument("--env", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--lrs", type=_parse_floats, default=[1e-4, 5e-5, 3e-5, 1e-5])
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--no-reference", action="store_true", help="disable red reference line at default-lr eps=1e-8")
    p.add_argument("--annotate-reference", action="store_true")

    # lr-sweep-grid (multi eps)
    p = sub.add_parser(
        "lr-sweep-grid",
        help="lr sweep grid: rows = epsilons, cols = return/entropy (e.g. eps=1 and eps=1e-8)",
    )
    _add_common_args(p)
    p.add_argument("--env", required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--epsilons", type=_parse_floats, default=[1.0, 1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=[1e-4, 5e-5, 3e-5, 1e-5])
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--x-axis", choices=["step", "cum_dtheta"], default="step",
                   help="x-axis: 'step' (default) or 'cum_dtheta' (per-seed cumulative ||Δθ||, absorbs effective-lr differences)")

    # lr-sweep-grid-multi (several envs side by side, shared legend)
    p = sub.add_parser(
        "lr-sweep-grid-multi",
        help="lr sweep grid for several envs side by side in one figure with a shared legend",
    )
    _add_common_args(p)
    p.add_argument("--envs", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   required=True, help="comma-separated envs, e.g. 'halfcheetah,swimmer,walker2d'")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--epsilons", type=_parse_floats, default=[1.0, 1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=[1e-4, 5e-5, 3e-5, 1e-5])
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--x-axis", choices=["step", "cum_dtheta"], default="step",
                   help="x-axis: 'step' (default) or 'cum_dtheta'")

    # b-ablation
    p = sub.add_parser("b-ablation", help="for each m, ablate over B (=remax_num_samples)")
    _add_common_args(p)
    p.add_argument("--env", required=True)
    p.add_argument("--epsilon", type=float, required=True)
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--bs", type=_parse_ints, default=[8, 16])
    p.add_argument("--metric", default="eval/return")
    p.add_argument("--lrs", type=_parse_floats, default=None)
    p.add_argument("--no-default-lr", action="store_true")

    # damping diagnostics
    p = sub.add_parser(
        "damping",
        help="6 envs 2x3 of a gradient/update damping metric (grad_norm, update_per_lr, update_per_grad_lr, ...)",
    )
    _add_common_args(p)
    p.add_argument(
        "--kind",
        default="grad_norm",
        choices=list(DAMPING_KINDS),
        help="which damping metric to plot",
    )
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--epsilons", type=_parse_floats, default=[1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=None)
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--include-eps-label", action="store_true")
    p.add_argument("--no-default-lr", action="store_true")
    p.add_argument("--no-log", action="store_true", help="disable y-axis log scale")

    # damping-combined: ||grad|| (solid) + ||u||/(||grad||*lr) (dashed) overlay
    p = sub.add_parser(
        "damping-combined",
        help="overlay raw ||grad|| (solid) and Adam normalization (dashed) on twin-y per env (2x3)",
    )
    _add_common_args(p)
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--epsilons", type=_parse_floats, default=[1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=None)
    p.add_argument("--no-default-lr", action="store_true")
    p.add_argument("--no-log", action="store_true")

    # update-over-grad-lr (legacy alias of damping --kind update_per_grad_lr)
    p = sub.add_parser("update-grad", help="6 envs 2x3 of update_norm / (grad_norm * lr)")
    _add_common_args(p)
    p.add_argument("--ms", type=_parse_ints, default=[1, 2, 4, 8])
    p.add_argument("--epsilons", type=_parse_floats, default=[1e-8])
    p.add_argument("--lrs", type=_parse_floats, default=None)
    p.add_argument("--include-sac", action="store_true")
    p.add_argument("--no-other-baselines", action="store_true",
                   help="when --include-sac is set, exclude non-SAC baselines (PPO, TD3)")
    p.add_argument("--no-default-lr", action="store_true")

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    df = _load_or_fetch(args.project, args.entity, args.samples, args.cache)

    if args.cmd in ("2x3", "2x4"):
        metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
        ncols = 3 if args.cmd == "2x3" else 4
        env_order = ENV_ORDER_6 if args.cmd == "2x3" else ENV_ORDER_8
        prefix = _make_prefix(
            f"paper_{args.cmd}",
            lrs=args.lrs,
            bs=args.bs,
            bsPerM=args.bs_per_m,
            sac=args.include_sac,
            base=args.baselines,
            epsL=args.include_eps_label,
            dfltlr=not args.no_default_lr,
            log=args.log_scale,
        )
        plot_envs_grid(
            df,
            ms=args.ms,
            epsilons=args.epsilons,
            metrics=metrics,
            nrows=2,
            ncols=ncols,
            env_order=env_order,
            lrs=args.lrs,
            bs=args.bs,
            bs_per_m=args.bs_per_m,
            baselines=args.baselines,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            include_eps_in_label=args.include_eps_label,
            use_default_lr=not args.no_default_lr,
            out_dir=args.out_dir,
            filename_prefix=prefix,
            log_scale=args.log_scale,
        )

    elif args.cmd == "4row":
        metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
        prefix = _make_prefix(
            "paper_4",
            lrs=args.lrs,
            sac=args.include_sac,
            epsL=args.include_eps_label,
            dfltlr=not args.no_default_lr,
        )
        plot_4envs_row(
            df,
            ms=args.ms,
            epsilons=args.epsilons,
            metrics=metrics,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            include_eps_in_label=args.include_eps_label,
            use_default_lr=not args.no_default_lr,
            out_dir=args.out_dir,
            filename_prefix=prefix,
        )

    elif args.cmd == "lr-sweep":
        env_slug = _canon_env(args.env).replace("brax/", "")
        prefix = _make_prefix(
            "lr_sweep",
            env=env_slug,
            m=args.m,
            eps=args.epsilon,
            lrs=args.lrs,
            sac=args.include_sac,
        )
        out_path = os.path.join(args.out_dir, f"{prefix}.pdf")
        plot_lr_sweep_return_entropy(
            df,
            env=args.env,
            m=args.m,
            epsilon=args.epsilon,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            add_default_reference=not args.no_reference,
            annotate_reference=args.annotate_reference,
            out_path=out_path,
        )

    elif args.cmd == "lr-sweep-grid":
        prefix = _make_prefix(
            "paper_lr_sweep_eps_grid",
            eps=args.epsilons,
            lrs=args.lrs,
            sac=args.include_sac,
        )
        plot_lr_sweep_eps_grid(
            df,
            env=args.env,
            m=args.m,
            epsilons=args.epsilons,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            out_dir=args.out_dir,
            filename_prefix=prefix,
            x_axis=args.x_axis,
        )

    elif args.cmd == "lr-sweep-grid-multi":
        prefix = _make_prefix(
            "paper_lr_sweep_eps_grid_multi",
            eps=args.epsilons,
            lrs=args.lrs,
            sac=args.include_sac,
        )
        plot_lr_sweep_eps_grid_multi(
            df,
            envs=args.envs,
            m=args.m,
            epsilons=args.epsilons,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            out_dir=args.out_dir,
            filename_prefix=prefix,
            x_axis=args.x_axis,
        )

    elif args.cmd == "b-ablation":
        prefix = _make_prefix(
            "paper_b_ablation",
            ms=args.ms,
            bs=args.bs,
            lrs=args.lrs,
            dfltlr=not args.no_default_lr,
        )
        plot_b_ablation_per_m(
            df,
            env=args.env,
            epsilon=args.epsilon,
            ms=args.ms,
            bs=args.bs,
            metric=args.metric,
            lrs=args.lrs,
            use_default_lr=not args.no_default_lr,
            out_dir=args.out_dir,
            filename_prefix=prefix,
        )

    elif args.cmd == "damping":
        prefix = _make_prefix(
            "paper_damping",
            ms=args.ms,
            eps=args.epsilons,
            lrs=args.lrs,
            sac=args.include_sac,
            epsL=args.include_eps_label,
            dfltlr=not args.no_default_lr,
            log=not args.no_log,
        )
        plot_damping_2x3(
            df,
            ms=args.ms,
            epsilons=args.epsilons,
            kind=args.kind,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            include_eps_in_label=args.include_eps_label,
            use_default_lr=not args.no_default_lr,
            log_scale=not args.no_log,
            out_dir=args.out_dir,
            filename_prefix=prefix,
        )

    elif args.cmd == "damping-combined":
        prefix = _make_prefix(
            "paper_damping_combined",
            ms=args.ms,
            eps=args.epsilons,
            lrs=args.lrs,
            dfltlr=not args.no_default_lr,
            log=not args.no_log,
        )
        plot_damping_combined_2x3(
            df,
            ms=args.ms,
            epsilons=args.epsilons,
            lrs=args.lrs,
            use_default_lr=not args.no_default_lr,
            log_scale=not args.no_log,
            out_dir=args.out_dir,
            filename_prefix=prefix,
        )

    elif args.cmd == "update-grad":
        prefix = _make_prefix(
            "paper_2x3_update_over_grad_lr",
            lrs=args.lrs,
            sac=args.include_sac,
            dfltlr=not args.no_default_lr,
        )
        plot_6envs_2x3_update_over_grad_lr(
            df,
            ms=args.ms,
            epsilons=args.epsilons,
            lrs=args.lrs,
            include_sac=args.include_sac,
            include_other_baselines=not args.no_other_baselines,
            use_default_lr=not args.no_default_lr,
            out_dir=args.out_dir,
            filename_prefix=prefix,
        )

    else:
        raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    _dispatch(args)