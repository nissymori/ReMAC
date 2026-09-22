import inspect

import jax
import jax.numpy as jnp

_METRIC_KEYS = (
    "actor_loss",
    "critic_loss",
    "entropy",
    "actor_grad_norm",
    "critic_grad_norm",
    "model_grad_norm",
    "actor_update_norm",
    "policy_std",
    "sigma_grad",
)


def zero_train_metrics(dtype=jnp.float32):
    return {k: jnp.array(0.0, dtype=dtype) for k in _METRIC_KEYS}


def add_train_metrics(lhs, rhs):
    return jax.tree_util.tree_map(lambda x, y: x + y, lhs, rhs)


def mean_train_metrics(metrics, denom):
    denom = jnp.asarray(denom, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda x: x / denom, metrics)


def entropy_from_logprob(logprob, discrete):
    if discrete:
        probs = jnp.exp(logprob)
        return -jnp.sum(probs * logprob, axis=1).mean()
    return (-logprob).mean()


def make_train_metrics(
    actor_loss,
    critic_loss,
    entropy,
    actor_grad_norm,
    critic_grad_norm,
    actor_update_norm=0.0,
    policy_std=0.0,
    sigma_grad=0.0,
):
    dt = actor_grad_norm.dtype
    actor_update_norm = jnp.asarray(actor_update_norm, dtype=dt)
    policy_std = jnp.asarray(policy_std, dtype=dt)
    sigma_grad = jnp.asarray(sigma_grad, dtype=dt)
    model_grad_norm = jnp.sqrt(actor_grad_norm**2 + critic_grad_norm**2)
    return {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy,
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "model_grad_norm": model_grad_norm,
        "actor_update_norm": actor_update_norm,
        "policy_std": policy_std,
        "sigma_grad": sigma_grad,
    }


def _callback_accepts_actor_update_norm(callback):
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        return False

    params = tuple(sig.parameters.values())
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params
    ):
        return True

    return "actor_update_norm" in sig.parameters or len(params) >= 9


def maybe_log_train_metrics(callback, interval, step, is_training, metrics, seed_idx):
    if callback is None or interval <= 0:
        return

    should_log = jnp.logical_and(is_training, step % interval == 0)
    supports_actor_update_norm = _callback_accepts_actor_update_norm(callback)

    def _emit(_):
        if supports_actor_update_norm:
            jax.debug.callback(
                callback,
                step,
                seed_idx,
                metrics["actor_loss"],
                metrics["critic_loss"],
                metrics["entropy"],
                metrics["actor_grad_norm"],
                metrics["critic_grad_norm"],
                metrics["model_grad_norm"],
                metrics["actor_update_norm"],
                metrics["policy_std"],
                metrics["sigma_grad"],
            )
        else:
            jax.debug.callback(
                callback,
                step,
                seed_idx,
                metrics["actor_loss"],
                metrics["critic_loss"],
                metrics["entropy"],
                metrics["actor_grad_norm"],
                metrics["critic_grad_norm"],
                metrics["model_grad_norm"],
            )

    jax.lax.cond(should_log, _emit, lambda _: None, operand=None)
