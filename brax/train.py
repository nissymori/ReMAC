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


def _resolve_local_algo(algo_str):
    local_algo_map = {
        "sac": ("sac", "SAC"),
        "remax_ac": ("remax_ac", "ReMaxAC"),
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


def main(algo_str, config, seed_id, num_seeds, time_fit, wandb_run=None):
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

    main(
        args.algorithm,
        config,
        args.seed_id,
        args.num_seeds,
        args.time_fit,
        wandb_run,
    )
