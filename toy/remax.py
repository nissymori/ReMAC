import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln



def comb_float(n, k):
    """Stable binomial coefficient comb(n, k) supporting vectorized JAX inputs."""
    n_arr = jnp.asarray(n, dtype=jnp.float64)
    k_arr = jnp.asarray(k, dtype=jnp.float64)
    valid = n_arr >= k_arr
    safe_n = jnp.where(valid, n_arr, 0.0)
    safe_k = jnp.where(valid, k_arr, 0.0)
    log_comb = gammaln(safe_n + 1.0) - gammaln(safe_k + 1.0) - gammaln(safe_n - safe_k + 1.0)
    return jnp.where(valid, jnp.exp(log_comb), 0.0)


def compute_batch_remax(returns: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    The unbiased estimator of the ReMax objective.
    """
    B = returns.shape[0]
    if B < m:
        raise ValueError(f"Batch size {B} must be greater than or equal to {m}.")

    sorted_returns = jnp.sort(returns)
    indices = jnp.arange(B)

    coeffs = comb_float(indices, m - 1)
    total_combs = comb_float(B, m)

    weights = coeffs / jnp.where(total_combs == 0, 1.0, total_combs)
    remax_est = jnp.sum(weights * sorted_returns)
    return remax_est


def compute_batch_remax(returns: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    The unbiased estimator of the ReMax objective.

    Args:
        returns: Array of shape (B,) containing the returns for the batch.
        m: number of comparisons to make (k in the paper).
    Returns:
        Scalar jnp.ndarray representing the unbiased estimate of the ReMax objective.
        (Note: While the docstring mentions shape (B,), the estimator defines a
         single scalar value for the batch. This returns a scalar.)
    """
    B = returns.shape[0]
    if B < m:
        raise ValueError(f"Batch size {B} must be greater than or equal to {m}.")

    sorted_returns = jnp.sort(returns)

    indices = jnp.arange(B)

    coeffs = comb_float(indices, m - 1)
    total_combs = comb_float(B, m)

    weights = coeffs / jnp.where(total_combs == 0, 1.0, total_combs)
    remax_est = jnp.sum(weights * sorted_returns)
    return remax_est


def compute_batch_ei(returns: jnp.ndarray, m: int) -> jnp.ndarray:
    """
    The unbiased estimator of the expected improvement scores for batch EI with without-replacement comparators.
    Args:
        returns: Array of shape (B,) containing the returns for the batch.
        m: number of comparisons to make.
    Returns:
        Array of shape (B,) containing the expected improvement scores.
    """
    b = returns.shape[0]
    if b < m:
        raise ValueError(f"Batch size {b} must be greater than or equal to {m}.")

    order = jnp.argsort(returns)
    sorted_returns = returns[order]
    inv_order = jnp.argsort(order)

    diff = jnp.maximum(sorted_returns[:, None] - sorted_returns[None, :], 0.0)
    w_num = comb_float(jnp.arange(b, dtype=returns.dtype), m - 2)
    denom = comb_float(b - 1, m - 1)
    w = jnp.where(denom > 0.0, w_num / denom, jnp.zeros_like(w_num))
    w = w.astype(returns.dtype)
    s_sorted = diff @ w
    return s_sorted[inv_order]


def l2o_weight_matrix_sorted(b: int, m: int, dtype) -> jnp.ndarray:
    """Weight matrix W^{L2O} for sorted rows/cols (comparators without replacement)."""
    i_idx = jnp.arange(b).reshape(b, 1)
    ell = jnp.arange(b).reshape(1, b)
    below_adj = (ell - jnp.where(ell > i_idx, 1, 0)).astype(dtype)
    w_raw = comb_float(below_adj, m - 2)
    denom = comb_float(b - 2, m - 1)
    w = jnp.where(denom > 0.0, w_raw / denom, jnp.zeros_like(w_raw))
    w = w.astype(dtype)
    w = jnp.where(jnp.eye(b, dtype=bool), 0.0, w)
    return w


def compute_l2o_baseline(returns: jnp.ndarray, m: int) -> jnp.ndarray:
    """Leave-two-out EI baseline b_{-i}^{L2O} computed in O(B^2)."""
    b = returns.shape[0]
    if b < m + 1:
        raise ValueError(f"Batch size {b} must be greater than or equal to {m + 1}.")

    order = jnp.argsort(returns)
    sorted_returns = returns[order]
    inv_order = jnp.argsort(order)

    diff = jnp.maximum(sorted_returns[:, None] - sorted_returns[None, :], 0.0)  # (B,B)
    col_sums = jnp.sum(diff, axis=0)  # (B,)
    overline_m = (col_sums[None, :] - diff) / float(b - 1)  # (B,B)
    weights = l2o_weight_matrix_sorted(b, m, returns.dtype)  # (B,B)
    b_sorted = jnp.sum(overline_m * weights, axis=1)  # (B,)
    return b_sorted[inv_order]