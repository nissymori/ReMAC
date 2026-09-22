import importlib
import re
import timeit

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from matplotlib import pyplot as plt

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

try:
    from rejax import get_algo as _get_rejax_algo
except ImportError:  # pragma: no cover - local algorithms can still run
    _get_rejax_algo = None

# Registers the sparse-reward Reacher under the name "sparse_reacher", so a
# config can select it with `env: brax/sparse_reacher`.
try:
    import sparse_reacher  # noqa: F401
except ImportError:  # pragma: no cover - optional
    pass


def _resolve_local_algo(algo_str):
    local_algo_map = {
        "sac": ("sac", "SAC"),
        "remax_ac": ("remax_ac", "ReMaxAC"),
        "td3": ("td3", "TD3"),
        "ppo": ("ppo", "PPO"),
    }
    if algo_str not in local_algo_map:
        return None

    module_name, class_name = local_algo_map[algo_str]
    import_candidates = (f"brax.{module_name}", module_name)
    last_error = None
    for candidate in import_candidates:
        try:
            module = importlib.import_module(candidate)
        except Exception as err:  # pragma: no cover - depends on execution mode
            last_error = err
            continue

        if hasattr(module, class_name):
            return getattr(module, class_name)

    raise ImportError(
        f"Failed to import local algorithm '{algo_str}'. Last error: {last_error}"
    )


def resolve_algo_class(algo_str):
    local_algo_cls = _resolve_local_algo(algo_str)
    if local_algo_cls is not None:
        return local_algo_cls

    if _get_rejax_algo is None:
        raise ImportError(
            f"Algorithm '{algo_str}' requires rejax, but rejax is not available."
        )
    return _get_rejax_algo(algo_str)


def _build_remax_config_from_sac(sac_config):
    remax_config = dict(sac_config)
    remax_config.setdefault("remax_num_samples", 16)
    remax_config.setdefault("remax_m", 4)
    remax_config.setdefault("actor_epsilon", 1e-8)
    remax_config.setdefault("soft_critic", False)
    return remax_config


def load_algorithm_config(config_path, algorithm):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f.read())

    if algorithm in full_config:
        return dict(full_config[algorithm])

    if algorithm == "remax_ac" and "sac" in full_config:
        return _build_remax_config_from_sac(full_config["sac"])

    available = ", ".join(sorted(full_config.keys()))
    raise KeyError(
        f"Algorithm '{algorithm}' is not defined in {config_path}. "
        f"Available entries: {available}"
    )


def _parse_override_value(value_str):
    parsed = yaml.safe_load(value_str)
    if not isinstance(parsed, str):
        return parsed

    if re.fullmatch(r"[+-]?\d+", parsed):
        return int(parsed)

    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", parsed):
        return float(parsed)

    return parsed


def _set_nested_config_value(config, key, value):
    parts = key.split(".")
    cursor = config

    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            cursor[part] = {}
            existing = cursor[part]
        elif not isinstance(existing, dict):
            raise ValueError(
                f"Cannot set nested key '{key}': '{part}' is not a mapping."
            )
        cursor = existing

    cursor[parts[-1]] = value


def apply_config_overrides(config, overrides):
    updated = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --set override '{override}'. Expected key=value format."
            )
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --set override '{override}': empty key.")
        value = _parse_override_value(raw_value.strip())
        _set_nested_config_value(updated, key, value)
    return updated


def _step_to_int(step):
    step_arr = np.asarray(step)
    if step_arr.ndim == 0:
        return int(step_arr)
    return int(step_arr.reshape(-1)[0])


def _values_per_seed(values, num_seeds):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float64)

    if num_seeds > 1 and arr.shape[0] == num_seeds:
        if arr.ndim == 1:
            return arr.astype(np.float64)
        reduce_axes = tuple(range(1, arr.ndim))
        return arr.mean(axis=reduce_axes).astype(np.float64)

    return np.asarray([float(arr.mean())], dtype=np.float64)


def _metric_scalar(value):
    return float(np.asarray(value).mean())


def _collect_mean_payload(mean_state, step, seed_idx, metric_map, num_seeds):
    metric_names = tuple(sorted(metric_map.keys()))
    state_key = (int(step), metric_names)
    step_bucket = mean_state.setdefault(state_key, {})

    for metric_name, raw_values in metric_map.items():
        seed_values = step_bucket.setdefault(metric_name, {})
        seed_values[int(seed_idx)] = _metric_scalar(raw_values)

    if any(len(seed_values) < num_seeds for seed_values in step_bucket.values()):
        return None

    payload = {}
    for metric_name, seed_values in step_bucket.items():
        values = np.asarray(list(seed_values.values()), dtype=np.float64)
        payload[f"mean/{metric_name}"] = float(values.mean())

    del mean_state[state_key]
    return payload


def _effective_eval_step_interval(algo):
    eval_freq = int(getattr(algo, "eval_freq", 0))
    if eval_freq <= 0:
        return eval_freq

    iteration_steps = int(getattr(algo, "num_envs", 1))
    if hasattr(algo, "num_steps"):
        iteration_steps *= int(getattr(algo, "num_steps"))

    if iteration_steps <= 0:
        return eval_freq

    return ((eval_freq + iteration_steps - 1) // iteration_steps) * iteration_steps


def _log_seeded_metrics(
    logger,
    step,
    metric_map,
    num_seeds,
    seed_idx=None,
    mean_state=None,
):
    if logger is None:
        return

    payload = {}
    if seed_idx is not None and seed_idx >= 0:
        for metric_name, raw_values in metric_map.items():
            payload[f"seed_{seed_idx}/{metric_name}"] = _metric_scalar(raw_values)
        logger.log(payload, step=step)
        if mean_state is not None:
            mean_payload = _collect_mean_payload(
                mean_state,
                step,
                seed_idx,
                metric_map,
                num_seeds,
            )
            if mean_payload is not None:
                logger.log(mean_payload, step=step)
        return

    for metric_name, raw_values in metric_map.items():
        per_seed_values = _values_per_seed(raw_values, num_seeds)
        if per_seed_values.size == 1:
            payload[metric_name] = float(per_seed_values[0])
        else:
            payload[f"mean/{metric_name}"] = float(per_seed_values.mean())
            for i, value in enumerate(per_seed_values):
                payload[f"seed_{i}/{metric_name}"] = float(value)

    logger.log(payload, step=step)


def _make_eval_log_callback(
    logger,
    algo_name,
    num_seeds,
    latest_train_metrics=None,
    mean_state=None,
):
    def _callback(step, seed_idx, mean_length, mean_return):
        step_i = _step_to_int(step)
        seed_i = int(np.asarray(seed_idx))
        _log_seeded_metrics(
            logger,
            step_i,
            {
                "eval/episode_length": mean_length,
                "eval/return": mean_return,
            },
            num_seeds,
            seed_idx=seed_i,
            mean_state=mean_state,
        )
        if latest_train_metrics is not None:
            train_metrics = latest_train_metrics.get(seed_i)
            if train_metrics is not None:
                _log_seeded_metrics(
                    logger,
                    step_i,
                    train_metrics,
                    num_seeds,
                    seed_idx=seed_i,
                    mean_state=mean_state,
                )

    return _callback


def _make_train_log_callback(
    algo_name,
    logger,
    num_seeds,
    latest_train_metrics=None,
    mean_state=None,
):
    def _callback(
        step,
        seed_idx,
        actor_loss,
        critic_loss,
        entropy,
        actor_grad_norm,
        critic_grad_norm,
        model_grad_norm,
        actor_update_norm=None,
        policy_std=None,
        sigma_grad=None,
    ):
        step_i = _step_to_int(step)
        seed_i = int(np.asarray(seed_idx))
        train_metrics = {
            "train/actor_loss": actor_loss,
            "train/critic_loss": critic_loss,
            "train/entropy": entropy,
            "train/actor_grad_norm": actor_grad_norm,
            "train/critic_grad_norm": critic_grad_norm,
            "train/model_grad_norm": model_grad_norm,
        }
        if actor_update_norm is not None:
            train_metrics["train/actor_update_norm"] = actor_update_norm
        if policy_std is not None:
            train_metrics["train/policy_std"] = policy_std
        if sigma_grad is not None:
            train_metrics["train/sigma_grad"] = sigma_grad

        message = (
            f"[{algo_name}] seed={seed_i} step={step_i} "
            f"actor_loss={_metric_scalar(actor_loss):.6f} "
            f"critic_loss={_metric_scalar(critic_loss):.6f} "
            f"entropy={_metric_scalar(entropy):.6f} "
            f"actor_grad_norm={_metric_scalar(actor_grad_norm):.6f} "
            f"critic_grad_norm={_metric_scalar(critic_grad_norm):.6f} "
            f"model_grad_norm={_metric_scalar(model_grad_norm):.6f}"
        )
        if actor_update_norm is not None:
            message += f" actor_update_norm={_metric_scalar(actor_update_norm):.6f}"
        if policy_std is not None:
            message += f" policy_std={_metric_scalar(policy_std):.6g}"
        if sigma_grad is not None:
            message += f" sigma_grad={_metric_scalar(sigma_grad):.6g}"
        print(message)

        if latest_train_metrics is not None:
            latest_train_metrics[seed_i] = {
                metric_name: _metric_scalar(raw_values)
                for metric_name, raw_values in train_metrics.items()
            }
            return

        _log_seeded_metrics(
            logger,
            step_i,
            train_metrics,
            num_seeds,
            seed_idx=seed_i,
            mean_state=mean_state,
        )

    return _callback


def setup_wandb(enabled, project, entity, run_name, config):
    if not enabled:
        return None

    if wandb is None:
        raise ImportError(
            "wandb logging is enabled, but wandb is not installed. Install with `pip install wandb`."
        )

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )


def _attach_train_logger(
    algo,
    algo_str,
    train_log_interval,
    logger,
    num_seeds,
    latest_train_metrics=None,
    mean_state=None,
):
    if not hasattr(algo, "train_log_interval") or not hasattr(algo, "train_log_callback"):
        return algo

    if train_log_interval is None:
        train_log_interval = _effective_eval_step_interval(algo)

    replace_kwargs = {
        "train_log_interval": int(train_log_interval),
        "train_log_callback": _make_train_log_callback(
            algo_str,
            logger,
            num_seeds,
            latest_train_metrics=latest_train_metrics,
            mean_state=mean_state,
        ),
    }
    if hasattr(algo, "train_seed_axis_name"):
        replace_kwargs["train_seed_axis_name"] = "seed"

    return algo.replace(**replace_kwargs)



def _make_eval_state_dumper(out_dir, run_tag):
    """Writes one .npz per (seed, eval step) holding raw, subsampled eval states."""
    import os

    os.makedirs(out_dir, exist_ok=True)

    def dump(global_step, seed_idx, obs):
        path = os.path.join(
            out_dir, f"{run_tag}_seed{int(seed_idx)}_step{int(global_step)}.npz"
        )
        np.savez_compressed(path, obs=np.asarray(obs, dtype=np.float32))

    return dump


def _attach_eval_state_collector(algo, eval_callback, cfg):
    """Wraps ``eval_callback`` so the last ``cfg['frac']`` of training also dumps
    the raw observations visited by the eval policy.

    Raw (un-normalized) observations are stored because the k-NN entropy estimate
    must use one normalization per environment, shared by every method and seed.
    """
    from state_entropy import collect_eval_states

    dump = _make_eval_state_dumper(cfg["out_dir"], cfg["run_tag"])
    keep = int(cfg["keep"])
    num_envs = int(cfg["num_envs"])
    start_step = int((1.0 - float(cfg["frac"])) * int(algo.total_timesteps))
    max_steps = int(algo.env_params.max_steps_in_episode)

    def wrapped(algo_, ts, rng):
        out = eval_callback(algo_, ts, rng)

        rng_roll, rng_sub = jax.random.split(jax.random.fold_in(rng, 12345))
        obs, valid = collect_eval_states(
            algo_.make_act(ts), rng_roll, algo_.env, algo_.env_params,
            num_envs=num_envs, max_steps=max_steps,
        )
        obs = obs.reshape(-1, obs.shape[-1])
        valid = valid.reshape(-1)

        # Gumbel top-k: a fixed-size uniform subsample of the valid states,
        # without replacement, with a seed fixed by rng_sub.
        g = jax.random.gumbel(rng_sub, valid.shape)
        scores = jnp.where(valid, g, -jnp.inf)
        idx = jnp.argsort(-scores)[:keep]
        sub = obs[idx]

        def maybe_dump(step, seed_idx, sub_obs, n_valid):
            if int(step) >= start_step and int(n_valid) >= keep:
                dump(step, seed_idx, sub_obs)

        jax.debug.callback(
            maybe_dump,
            ts.global_step,
            jax.lax.axis_index("seed"),
            sub,
            jnp.sum(valid),
        )
        return out

    return wrapped


def main(algo_str, config, seed_id, num_seeds, time_fit, wandb_run=None, eval_states_cfg=None):
    config = dict(config)
    train_log_interval = config.pop("train_log_interval", config.pop("log_interval", None))
    latest_train_metrics = {}
    mean_state = {}

    algo_cls = resolve_algo_class(algo_str)
    algo = algo_cls.create(**config)
    algo = _attach_train_logger(
        algo,
        algo_str,
        train_log_interval,
        wandb_run,
        num_seeds,
        latest_train_metrics=latest_train_metrics,
        mean_state=mean_state,
    )
    print(algo.config)

    old_eval_callback = algo.eval_callback

    eval_log_callback = _make_eval_log_callback(
        wandb_run,
        algo_str,
        num_seeds,
        latest_train_metrics=latest_train_metrics,
        mean_state=mean_state,
    )

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        mean_length = lengths.mean()
        mean_return = returns.mean()
        seed_idx = jax.lax.axis_index("seed")
        jax.debug.print(
            "Seed {} Step {}, Mean episode length: {}, Mean return: {}",
            seed_idx,
            ts.global_step,
            mean_length,
            mean_return,
        )
        jax.debug.callback(eval_log_callback, ts.global_step, seed_idx, mean_length, mean_return)
        return lengths, returns

    if eval_states_cfg is not None:
        eval_callback = _attach_eval_state_collector(algo, eval_callback, eval_states_cfg)

    algo = algo.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    vmap_train = jax.jit(jax.vmap(algo_cls.train, in_axes=(None, 0), axis_name="seed"))
    ts, (_, returns) = vmap_train(algo, keys)
    returns.block_until_ready()

    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * algo.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(algo, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(
            f"{time:.1f} seconds total, equalling to "
            f"{time / num_seeds:.1f} seconds per seed"
        )

    if wandb_run is not None:
        wandb_run.finish()

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gymnax/cartpole.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--time-fit",
        action="store_true",
        help="Time how long it takes to fit the agent by fitting 3 times.",
    )
    parser.add_argument(
        "--seed_id",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to roll out.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config entries (supports dotted keys). Can be passed multiple times.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="continuous-remax-brax-report",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--collect-eval-states",
        action="store_true",
        help="Dump the raw observations visited by the eval policy (for the k-NN "
             "visited-state entropy), over the last --eval-states-frac of training.",
    )
    parser.add_argument("--eval-states-dir", type=str, default="eval_states")
    parser.add_argument("--eval-states-num-envs", type=int, default=64)
    parser.add_argument("--eval-states-keep", type=int, default=4000,
                        help="States kept per (seed, evaluation); this is the N of the estimator.")
    parser.add_argument("--eval-states-frac", type=float, default=0.1,
                        help="Trailing fraction of training over which states are dumped "
                             "(matches the final-value rule of App. C.1).")

    args = parser.parse_args()
    config = load_algorithm_config(args.config, args.algorithm)
    config = apply_config_overrides(config, args.set)

    wandb_config = dict(config)
    wandb_config.update(
        {
            "algorithm": args.algorithm,
            "seed_id": args.seed_id,
            "num_seeds": args.num_seeds,
            "config_path": args.config,
        }
    )
    wandb_run = setup_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        config=wandb_config,
    )

    eval_states_cfg = None
    if args.collect_eval_states:
        eval_states_cfg = {
            "out_dir": args.eval_states_dir,
            "run_tag": args.wandb_run_name or f"{args.algorithm}_{args.seed_id}",
            "num_envs": args.eval_states_num_envs,
            "keep": args.eval_states_keep,
            "frac": args.eval_states_frac,
        }

    main(
        args.algorithm,
        config,
        args.seed_id,
        args.num_seeds,
        args.time_fit,
        wandb_run,
        eval_states_cfg=eval_states_cfg,
    )
