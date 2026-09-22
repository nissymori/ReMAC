"""Visited-state entropy: raw eval-state collection + k-NN differential entropy.

Two separable pieces:

1. ``collect_eval_states`` rolls the current eval policy out and returns the
   *raw* (un-normalized) observations it visited, together with a validity mask
   that is False for post-termination padding.  Raw observations are required
   because the entropy estimate must use one normalization per environment that
   is shared by every method and seed; an agent-specific running mean/std would
   put each agent in a different coordinate system.

2. ``knn_entropy`` is the k-nearest-neighbor differential-entropy estimator used
   by MEPOL (Mutti et al., AAAI 2021) and RE3 (Seo et al., ICML 2021):

       H_k = -(1/N) sum_i log( k / (N V_i) ) + log k - psi(k),
       V_i = pi^{d/2} / Gamma(d/2 + 1) * R_i^d,

   with R_i the distance from x_i to its k-th nearest neighbor (i itself
   excluded).  Everything is done in log space via ``gammaln`` so that R_i^d
   never overflows in high d.

This module does no normalization itself; the caller supplies already-
standardized points.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import digamma, gammaln

# Distances of exactly zero occur only for duplicated states (e.g. a frozen
# policy at a fixed point).  They are floored rather than dropped so that N is
# identical across methods.
MIN_RADIUS = 1e-12


def _rollout_single(act, env, env_params, rng, max_steps):
    """One fixed-length rollout; returns the raw observations it visited."""
    rng_reset, rng_eval = jax.random.split(rng)
    obs0, env_state0 = env.reset(rng_reset, env_params)

    def step(carry, _):
        rng, env_state, last_obs, done = carry
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        action = act(last_obs, rng_act)
        obs, env_state_next, _reward, done_step, _info = env.step(
            rng_step, env_state, action, env_params
        )
        valid = jnp.logical_not(done)
        new_done = jnp.logical_or(done, done_step)
        return (rng, env_state_next, obs, new_done), (last_obs, valid)

    carry0 = (rng_eval, env_state0, obs0, jnp.array(False))
    _, (obs_traj, valid) = jax.lax.scan(step, carry0, None, length=max_steps)
    return obs_traj, valid


def collect_eval_states(act, rng, env, env_params, num_envs=64, max_steps=None):
    """Vectorized eval rollouts returning raw observations and a validity mask.

    ``act`` is the policy ``(raw_obs, rng) -> action``; it applies the agent's
    own observation normalization internally, so what is returned here is the
    raw observation.
    """
    if max_steps is None:
        max_steps = int(env_params.max_steps_in_episode)
    seeds = jax.random.split(rng, num_envs)
    obs_traj, valid = jax.vmap(
        _rollout_single, in_axes=(None, None, None, 0, None)
    )(act, env, env_params, seeds, max_steps)
    return obs_traj, valid  # (num_envs, max_steps, obs_dim), (num_envs, max_steps)


def knn_entropy(x, k=4, min_radius=MIN_RADIUS):
    """MEPOL / RE3 k-NN differential-entropy estimate of the distribution of ``x``.

    Args:
        x: (N, d) points, already standardized with the shared reference stats.
        k: neighbor rank (k=4 in MEPOL's high-dimensional continuous-control runs).

    Returns:
        Scalar estimate, in nats.  Values from different d or N are not
        comparable to one another.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    if n <= k:
        raise ValueError(f"need N > k, got N={n}, k={k}")

    # k-th nearest neighbor distance, excluding the point itself.
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(x).query(x, k=k + 1, workers=-1)
    r = np.maximum(dist[:, k], min_radius)

    # log V_i = (d/2) log pi - gammaln(d/2 + 1) + d log R_i
    log_v = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0) + d * np.log(r)
    # H = -(1/N) sum log(k / (N V_i)) + log k - psi(k)
    #   = -(log k - log N - mean(log V_i)) + log k - psi(k)
    h = -(np.log(k) - np.log(n) - log_v.mean()) + np.log(k) - digamma(k)
    return float(h)
