"""Core mathematical routines, solvers, and metrics for Dependent KMM.

This module implements:
- Covariance generators (RBF, Matérn, Gibbs).
- Exact Random Fourier Features for Gaussian and Matérn processes.
- Kernel Mean Matching (KMM) quadratic programming via CVXOPT.
- Closed-form effective sample size formulas (Input neff and GP neff).
- Empirical variogram length-scale estimation.
- Confidence interval construction, Coverage, and Winkler Score evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cvxopt
import cvxopt.solvers
import numpy as np
import scipy.optimize

cvxopt.solvers.options["show_progress"] = False


# =====================================================================
# 1. Covariance Kernels & Generators
# =====================================================================

def rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """Computes Gaussian RBF Gram matrix k(x_i, y_j) = exp(-||x_i - y_j||^2 / (2 * sigma^2))."""
    diff = x[:, None] - y[None, :]
    return np.exp(-(diff**2) / (2.0 * bandwidth**2))


def se_cholesky(n: int, ell_x: float, marginal_sd: float = 1.0, jitter: float = 1e-9) -> np.ndarray:
    """Computes lower Cholesky factor for stationary Squared-Exponential covariance."""
    idx = np.arange(n, dtype=float)
    d = idx[:, None] - idx[None, :]
    cov = marginal_sd**2 * np.exp(-(d**2) / (2.0 * ell_x**2))
    cov.flat[:: n + 1] += jitter
    return np.linalg.cholesky(cov)


def matern_cholesky(n: int, ell_x: float, marginal_sd: float = 1.0, nu: float = np.inf, jitter: float = 1e-8) -> np.ndarray:
    """Computes lower Cholesky factor for stationary Matérn-nu covariance."""
    idx = np.arange(n, dtype=float)
    d = np.abs(idx[:, None] - idx[None, :])

    if np.isinf(nu):
        cov = np.exp(-(d**2) / (2.0 * ell_x**2))
    elif nu == 2.5:
        s5 = math.sqrt(5.0) * d / ell_x
        cov = (1.0 + s5 + (5.0 * d**2) / (3.0 * ell_x**2)) * np.exp(-s5)
    elif nu == 1.5:
        s3 = math.sqrt(3.0) * d / ell_x
        cov = (1.0 + s3) * np.exp(-s3)
    elif nu == 0.5:
        cov = np.exp(-d / ell_x)
    else:
        raise ValueError(f"Unsupported Matérn parameter nu: {nu}")

    cov = marginal_sd**2 * cov
    cov.flat[:: n + 1] += jitter
    return np.linalg.cholesky(cov)


def gibbs_nonstat_cholesky(n: int, profile: str, marginal_sd: float = 1.0, jitter: float = 1e-8) -> np.ndarray:
    """Computes Cholesky factor for non-stationary Gibbs kernel with time-varying length-scale."""
    t = np.arange(n, dtype=float)

    if profile == "Stationary":
        L = np.full(n, 2.5)
    elif profile == "Abrupt Switch":
        L = np.where(t < n / 2, 0.5, 4.5)
    elif profile == "Smooth Wave":
        L = 2.5 + 2.0 * np.sin(2.0 * math.pi * t / n)
    else:
        raise ValueError(f"Unknown non-stationary profile: {profile}")

    L_i = L[:, None]
    L_j = L[None, :]
    prefactor = np.sqrt((2.0 * L_i * L_j) / (L_i**2 + L_j**2))
    d_sq = (t[:, None] - t[None, :]) ** 2
    cov = marginal_sd**2 * prefactor * np.exp(-d_sq / (L_i**2 + L_j**2))
    cov.flat[:: n + 1] += jitter
    return np.linalg.cholesky(cov)


def generate_gaussian_paths(
    rng: np.random.Generator,
    n_reps: int,
    n: int,
    ell_x: float,
    mean: float = 0.0,
    marginal_sd: float = 1.0,
) -> np.ndarray:
    """Generates continuous stationary Gaussian trajectories."""
    chol = se_cholesky(n, ell_x, marginal_sd)
    return mean + rng.standard_normal((n_reps, n)) @ chol.T


# =====================================================================
# 2. Generative Surrogates & Response Functions
# =====================================================================

class GPResponse:
    """RBF Gaussian Process response surface via Random Fourier Features."""

    def __init__(self, rng_or_seed: Any, h: float = 1.5, n_features: int = 512):
        rng = np.random.default_rng(rng_or_seed) if isinstance(rng_or_seed, int) else rng_or_seed
        self.h = h
        self.omega = rng.normal(0.0, 1.0 / h, size=n_features)
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
        self.weights = rng.normal(size=n_features)
        self.scale = math.sqrt(2.0 / n_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_flat = x.flatten()
        features = np.cos(x_flat[:, None] * self.omega[None, :] + self.phase[None, :])
        raw = self.scale * (features @ self.weights)
        out = 0.5 + 0.4 * np.tanh(0.9 * x_flat + raw)
        return out.reshape(orig_shape)


class MaternGPResponse:
    """Matérn-nu GP response surface via exact Student-t Random Fourier Features."""

    def __init__(self, rng: np.random.Generator, h: float = 1.5, nu: float = np.inf, n_features: int = 512):
        self.h = h
        self.nu = nu

        if np.isinf(nu):
            self.omega = rng.normal(0.0, 1.0 / h, size=n_features)
        else:
            df = 2.0 * nu
            t_samples = rng.standard_t(df, size=n_features)
            self.omega = (math.sqrt(df) / h) * t_samples

        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
        self.weights = rng.normal(size=n_features)
        self.scale = math.sqrt(2.0 / n_features)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_flat = x.flatten()
        features = np.cos(x_flat[:, None] * self.omega[None, :] + self.phase[None, :])
        raw = self.scale * (features @ self.weights)
        out = 0.5 + 0.4 * np.tanh(0.9 * x_flat + raw)
        return out.reshape(orig_shape)


# =====================================================================
# 3. KMM Quadratic Programming & Weights
# =====================================================================

def clipped_kmm_weights(
    x_source: np.ndarray,
    x_target: np.ndarray,
    bandwidth: float = 1.0,
    clip: float = 15.0,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Solves the KMM Quadratic Program with bounded weights [0, B] and sum constraint sum(w) = n."""
    n, m = x_source.shape[0], x_target.shape[0]
    K_ss = rbf_kernel(x_source, x_source, bandwidth) + ridge * np.eye(n)
    K_st = rbf_kernel(x_source, x_target, bandwidth)
    kappa = (n / m) * np.sum(K_st, axis=1)

    P = cvxopt.matrix(K_ss)
    q = cvxopt.matrix(-kappa)
    G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
    h = cvxopt.matrix(np.hstack((np.zeros(n), np.full(n, clip))))
    A = cvxopt.matrix(np.ones((1, n)))
    b = cvxopt.matrix(np.array([float(n)]))

    try:
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        w = np.array(sol["x"]).flatten()
    except Exception:
        w = np.ones(n, dtype=float)
    return np.clip(w, 0.0, clip)


def get_oracle_density_ratio_weights(
    x: np.ndarray,
    shift: float,
    marginal_sd: float = 1.0,
    clip: float = 15.0,
) -> np.ndarray:
    """Evaluates analytical density ratio w*(x) = p_te(x) / p_tr(x) under Gaussian shift."""
    w = np.exp((shift * x - 0.5 * shift**2) / (marginal_sd**2))
    w = np.clip(w, 0.0, clip)
    w = (w / np.sum(w)) * len(x)
    return w


# =====================================================================
# 4. Effective Sample Size Formulas
# =====================================================================

def neff_input(n: int, ell_x: float) -> float:
    """Calculates effective sample size based strictly on raw input autocorrelation."""
    lags = np.arange(1, n, dtype=float)
    rho = np.exp(-(lags**2) / (2.0 * ell_x**2))
    denom = 1.0 + 2.0 * float(np.sum((1.0 - lags / n) * rho))
    return n / denom


def gp_formula_neff(
    n: int,
    h: float,
    ell_x: float,
    tau: float = 1.0,
    sigma0: float = 1.0,
    d: int = 1,
) -> float:
    """Calculates GP-induced effective sample size via analytical autocorrelation formula (Equation 5 & 6)."""
    marginal_var = tau**2 * sigma0**2
    a_h = 2.0 * marginal_var / (h**2)
    q_h = (1.0 + a_h) ** (-d / 2.0)
    lags = np.arange(1, n, dtype=float)
    r_k = np.exp(-(lags**2) / (2.0 * ell_x**2))
    rho = ((1.0 + a_h * (1.0 - r_k)) ** (-d / 2.0) - q_h) / (1.0 - q_h)
    denom = 1.0 + 2.0 * float(np.sum((1.0 - lags / n) * rho))
    return n / denom


def estimate_h(x: np.ndarray, z: np.ndarray) -> float:
    """Estimates functional length-scale h via empirical variogram regression."""
    var_z = float(np.var(z, ddof=1))
    if var_z < 1e-8:
        return 1.5

    idx = np.argsort(x)
    xs, zs = x[idx], z[idx]
    if len(xs) > 350:
        step = len(xs) // 350
        xs, zs = xs[::step][:350], zs[::step][:350]

    d_sq = (xs[:, None] - xs[None, :]) ** 2
    v = (zs[:, None] - zs[None, :]) ** 2
    i, j = np.triu_indices(len(xs), k=1)
    d_sq_tri, v_tri = d_sq[i, j], v[i, j]

    def loss(h_val: np.ndarray) -> float:
        h_v = h_val[0]
        pred_v = 2.0 * var_z * (1.0 - np.exp(-d_sq_tri / (2.0 * h_v**2)))
        return float(np.mean(((v_tri - pred_v) / (2.0 * var_z)) ** 2))

    best_loss, best_h = float("inf"), 1.5
    for h0 in [0.8, 1.5, 3.0]:
        res = scipy.optimize.minimize(loss, x0=[h0], bounds=[(0.2, 15.0)], method="L-BFGS-B")
        if res.fun < best_loss:
            best_loss, best_h = res.fun, float(res.x[0])
    return best_h


# =====================================================================
# 5. Inference, Confidence Intervals & Evaluation Metrics
# =====================================================================

def compute_winkler_score(lower: float, upper: float, truth: float, alpha: float = 0.05) -> float:
    """Evaluates the Winkler Interval Score (Equation 57)."""
    width = upper - lower
    if truth < lower:
        return width + (2.0 / alpha) * (lower - truth)
    elif truth > upper:
        return width + (2.0 / alpha) * (truth - upper)
    return width


def evaluate_ci(
    y_values: np.ndarray,
    weights: np.ndarray,
    neff: float,
    truth: float,
    z_score: float = 1.95996,
    alpha: float = 0.05,
    clip_B: float = 15.0,
) -> Dict[str, float]:
    """Evaluates Wald confidence interval and returns estimation metrics."""
    estimate = float(np.mean(weights * y_values))
    abs_error = abs(estimate - truth)

    var = float(np.var(weights * (y_values - estimate), ddof=1))
    half_width = z_score * math.sqrt(max(var, 1e-14) / max(neff, 1.0))
    lower, upper = estimate - half_width, estimate + half_width
    covered = float(lower <= truth <= upper)
    winkler = compute_winkler_score(lower, upper, truth, alpha=alpha)

    # Theorem 1 conservative theoretical radius
    strict_radius = clip_B * math.sqrt(math.log(6.0 / alpha) / (2.0 * max(neff, 1.0)))

    return {
        "estimate": estimate,
        "abs_error": abs_error,
        "half_width": half_width,
        "covered": covered,
        "winkler": winkler,
        "strict_radius": strict_radius,
    }


def block_bootstrap_ci(
    y_values: np.ndarray,
    weights: np.ndarray,
    truth: float,
    block_len: int = 6,
    n_boot: int = 200,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Evaluates Moving-Block Bootstrap confidence interval."""
    n = len(y_values)
    weighted_y = weights * y_values
    estimate = float(np.mean(weighted_y))
    abs_error = abs(estimate - truth)

    n_blocks = int(math.ceil(n / block_len))
    boot_means = []
    for _ in range(n_boot):
        start_indices = np.random.randint(0, n - block_len + 1, size=n_blocks)
        boot_sample = np.concatenate([weighted_y[i : i + block_len] for i in start_indices])[:n]
        boot_means.append(float(np.mean(boot_sample)))

    lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    half_width = (upper - lower) / 2.0
    covered = float(lower <= truth <= upper)
    winkler = compute_winkler_score(lower, upper, truth, alpha=alpha)

    return {
        "estimate": estimate,
        "abs_error": abs_error,
        "half_width": half_width,
        "covered": covered,
        "winkler": winkler,
        "strict_radius": 0.0,
    }


def select_best_valid_method(
    results: Dict[str, Dict[str, List[float]]],
    n_reps: int = 300,
    nominal_cov: float = 0.95,
) -> Optional[str]:
    """Identifies the practical method with minimal half-width among those achieving valid coverage."""
    # Binomial 2-sigma threshold
    threshold = nominal_cov - 1.95996 * math.sqrt(nominal_cov * (1.0 - nominal_cov) / n_reps)
    best_method = None
    min_radius = float("inf")

    for method, records in results.items():
        if "Oracle" in method or "Strict" in method:
            continue
        cov = float(np.mean(records["cov"]))
        rad = float(np.mean(records["hw"]))
        if cov >= threshold and rad < min_radius:
            min_radius = rad
            best_method = method
    return best_method