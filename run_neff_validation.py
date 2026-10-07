"""Validate the GP-induced effective sample size formula for dependent KMM.

Reproduces Figures from the paper:
- Figure 1: Validation of GP-induced neff on matched GP realizations (fig1_neff_validation_gp.pdf)
- Figure 3: Validation on misspecified deterministic harmonic surface (fig3_neff_validation_deterministic.pdf)
- Figure 4: 2D ratio heatmap over (ell_x, h) for GP models (fig4_heatmap_gp.pdf)
- Figure 5: 2D ratio heatmap over (ell_x, omega) for deterministic models (fig5_heatmap_deterministic.pdf)

Usage:
    python run_neff_validation.py [--force]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.optimize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    MATPLOTLIB_ERROR = exc


def se_input_cholesky(n: int, ell_x: float, marginal_sd: float, jitter: float) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    d = idx[:, None] - idx[None, :]
    cov = marginal_sd**2 * np.exp(-(d**2) / (2.0 * ell_x**2))
    cov.flat[:: n + 1] += jitter
    return np.linalg.cholesky(cov)


def gp_formula(
    n: int,
    h: float,
    ell_x: float,
    tau: float,
    sigma0: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Calculate finite-n GP effective sample size prediction."""
    marginal_var = tau**2 * sigma0**2
    a_h = 2.0 * marginal_var / (h**2)
    q_h = (1.0 + a_h) ** -0.5
    lags = np.arange(1, n, dtype=float)
    r_k = np.exp(-(lags**2) / (2.0 * ell_x**2))
    rho = ((1.0 + a_h * (1.0 - r_k)) ** -0.5 - q_h) / (1.0 - q_h)
    weights = 1.0 - lags / n
    denom = 1.0 + 2.0 * float(np.sum(weights * rho))
    neff = n / denom
    gamma0 = 1.0 - q_h
    return lags.astype(int), rho, neff, gamma0, q_h


def estimate_h(x: np.ndarray, z: np.ndarray) -> float:
    """Empirically estimate GP length-scale h by fitting a variogram."""
    var_z = float(np.var(z, ddof=1))
    if var_z < 1e-12:
        return 1.0
    
    idx = np.argsort(x)
    xs = x[idx]
    zs = z[idx]
    
    # Subsample to keep variogram fitting fast
    if len(xs) > 400:
        step = len(xs) // 400
        xs = xs[::step][:400]
        zs = zs[::step][:400]
        
    d_sq = (xs[:, None] - xs[None, :])**2
    v = (zs[:, None] - zs[None, :])**2
    
    # Extract upper triangle (unique pairs)
    i, j = np.triu_indices(len(xs), k=1)
    d_sq_tri = d_sq[i, j]
    v_tri = v[i, j]
    
    def loss(h_val):
        h_val = h_val[0]
        pred_v = 2.0 * var_z * (1.0 - np.exp(-d_sq_tri / (2.0 * h_val**2)))
        return np.mean(((v_tri - pred_v) / (2.0 * var_z))**2)
    
    best_loss = float('inf')
    best_h = 1.0
    for h0 in [0.5, 1.0, 2.0, 5.0]:
        res = scipy.optimize.minimize(loss, x0=[h0], bounds=[(1e-2, 50.0)], method='L-BFGS-B')
        if res.fun < best_loss:
            best_loss = res.fun
            best_h = res.x[0]
            
    return float(best_h)


def make_gp_func(rng: np.random.Generator, h: float, n_features: int) -> callable:
    omega = rng.normal(0.0, 1.0 / h, size=n_features)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
    weights = rng.normal(size=n_features)
    scale = math.sqrt(2.0 / n_features)
    
    def f(x: np.ndarray) -> np.ndarray:
        features = np.cos(x[:, None] * omega[None, :] + phase[None, :])
        return scale * (features @ weights)
    return f


def make_det_func(omega: float) -> callable:
    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(omega * x) + 0.5 * np.cos(2.0 * omega * x)
    return f


def run_scenario(
    f0_func: callable,
    g_func: callable,
    n: int,
    ell_x: float,
    tau: float,
    sigma0: float,
    rng: np.random.Generator,
    n_reps: int
) -> Dict[str, object]:
    marginal_sd = tau * sigma0
    
    # 1. Estimate h_Y and h_L using an independent sequence
    x_val = rng.standard_normal(2000) * marginal_sd
    y_val = f0_func(x_val)
    l_val = (y_val - g_func(x_val))**2
    
    h_Y = estimate_h(x_val, y_val)
    h_L = estimate_h(x_val, l_val)
    
    var_marg_Y = float(np.var(y_val, ddof=1))
    var_marg_L = float(np.var(l_val, ddof=1))
    
    # 2. Simulate dependent sequences
    chol = se_input_cholesky(n, ell_x, marginal_sd, jitter=1e-10)
    x_paths = rng.standard_normal((n_reps, n)) @ chol.T
    
    x_flat = x_paths.reshape(-1)
    y_flat = f0_func(x_flat)
    l_flat = (y_flat - g_func(x_flat))**2
    
    y_paths = y_flat.reshape(n_reps, n)
    l_paths = l_flat.reshape(n_reps, n)
    
    # Compute sample means
    y_means = np.mean(y_paths, axis=1)
    l_means = np.mean(l_paths, axis=1)
    
    # Empirical variances of sample means
    var_mean_Y = float(np.var(y_means, ddof=1))
    var_mean_L = float(np.var(l_means, ddof=1))
    
    # Empirical n_eff
    neff_emp_Y = var_marg_Y / max(var_mean_Y, 1e-12)
    neff_emp_L = var_marg_L / max(var_mean_L, 1e-12)
    
    # Predicted n_eff
    _, _, neff_pred_Y, _, _ = gp_formula(n, h_Y, ell_x, tau, sigma0)
    _, _, neff_pred_L, _, _ = gp_formula(n, h_L, ell_x, tau, sigma0)
    
    return {
        "ell_x": ell_x,
        "h_Y_est": h_Y,
        "h_L_est": h_L,
        "neff_emp_Y": neff_emp_Y,
        "neff_pred_Y": float(neff_pred_Y),
        "neff_emp_L": neff_emp_L,
        "neff_pred_L": float(neff_pred_L),
    }


def plot_single_scatter(out_path: Path, rows: List[Dict], title: str) -> None:
    """Plot and save an individual scatter plot as a separate PDF."""
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)

    ax.scatter([r["neff_pred_Y"] for r in rows], [r["neff_emp_Y"] for r in rows], 
               c='C0', marker='o', alpha=0.7, label='Target (Y)')
    ax.scatter([r["neff_pred_L"] for r in rows], [r["neff_emp_L"] for r in rows], 
               c='C1', marker='x', alpha=0.7, label='Risk (Loss)')

    max_val = max(
        max([r["neff_pred_Y"] for r in rows] + [r["neff_emp_Y"] for r in rows]),
        max([r["neff_pred_L"] for r in rows] + [r["neff_emp_L"] for r in rows])
    ) * 1.05

    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_title(title)
    ax.set_xlabel(r"Predicted $n_{\mathrm{eff}}^{\mathrm{GP}}$")
    ax.set_ylabel(r"Empirical $n_{\mathrm{eff}}$ (Var. of sample mean)")
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_heatmaps(out_dir: Path, rows: List[Dict], model_name: str, y_grid_name: str, filename: str) -> None:
    if plt is None:
        return
        
    ells = sorted(list(set(r["ell_x"] for r in rows)))
    params = sorted(list(set(r["true_param"] for r in rows)))
    
    ratio_Y = np.full((len(ells), len(params)), np.nan)
    ratio_L = np.full((len(ells), len(params)), np.nan)
    
    for i, ell in enumerate(ells):
        for j, p in enumerate(params):
            sub = [r for r in rows if r["ell_x"] == ell and r["true_param"] == p]
            if sub:
                ratio_Y[i, j] = np.mean([r["neff_emp_Y"] / r["neff_pred_Y"] for r in sub])
                ratio_L[i, j] = np.mean([r["neff_emp_L"] / r["neff_pred_L"] for r in sub])
                
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    
    for ax, vals, title in zip(axes, [ratio_Y, ratio_L], ["Target (Y)", "Risk (Loss)"]):
        im = ax.imshow(vals, origin="lower", aspect="auto", cmap="viridis", vmin=0.8, vmax=1.2)
        ax.set_xticks(np.arange(len(params)), [f"{p:g}" for p in params])
        ax.set_yticks(np.arange(len(ells)), [f"{ell:g}" for ell in ells])
        ax.set_title(title)
        
        for i_e in range(vals.shape[0]):
            for j_p in range(vals.shape[1]):
                if not np.isnan(vals[i_e, j_p]):
                    ax.text(j_p, i_e, f"{vals[i_e, j_p]:.2f}", ha="center", va="center", 
                            color="white" if abs(vals[i_e, j_p] - 1.0) > 0.15 else "black", fontsize=9)
                            
        fig.colorbar(im, ax=ax, label=r"Empirical / Predicted $n_{\mathrm{eff}}$")
    
    fig.supxlabel(y_grid_name)
    fig.supylabel(r"Input dependence $\ell_x$")
    fig.suptitle(f"Effective Sample Size Ratio: {model_name}")
    
    fig.savefig(out_dir / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GP neff formula validation (Figures 1, 3, 4, 5).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "figures",
        help="Directory where output PDFs are saved (default: results/figures)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if target figure files already exist.",
    )
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--n-reps", type=int, default=800)
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sigma0", type=float, default=1.0)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Список ключевых файлов, строго соответствующих проекту
    target_figures = [
        out_dir / "fig1_neff_validation_gp.pdf",
        out_dir / "fig3_neff_validation_deterministic.pdf",
        out_dir / "fig4_heatmap_gp.pdf",
        out_dir / "fig5_heatmap_deterministic.pdf",
    ]

    if all(p.exists() for p in target_figures) and not args.force:
        
        return

    rng = np.random.default_rng(args.seed)

    h_grid = [0.5, 1.0, 2.0, 4.0]
    ell_x_grid = [0.5, 1.0, 2.0, 4.0]
    
    g_func = lambda x: np.full_like(x, 0.2)
    
    gp_rows = []
    print("Running Model 1 (GP Realizations)...")
    for ell_x in ell_x_grid:
        for h in h_grid:
            for rep_gp in range(3):
                print(f" GP: ell_x={ell_x}, h={h}, realization={rep_gp}")
                f0_func = make_gp_func(rng, h, args.n_features)
                res = run_scenario(f0_func, g_func, args.n, ell_x, args.tau, args.sigma0, rng, args.n_reps)
                res["model"] = "GP"
                res["true_param"] = h
                res["realization"] = rep_gp
                gp_rows.append(res)
                
    omega_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    det_rows = []
    print("\nRunning Model 2 (Deterministic Sine/Cosine)...")
    for ell_x in ell_x_grid:
        for omega in omega_grid:
            print(f" Det: ell_x={ell_x}, omega={omega}")
            f0_func = make_det_func(omega)
            res = run_scenario(f0_func, g_func, args.n, ell_x, args.tau, args.sigma0, rng, args.n_reps)
            res["model"] = "Deterministic"
            res["true_param"] = omega
            res["realization"] = 0
            det_rows.append(res)

    print("\nGenerating and saving figures...")
    # Figure 1: GP Realizations (валидация на GP)
    plot_single_scatter(out_dir / "fig1_neff_validation_gp.pdf", gp_rows, "GP")

    # Figure 3: Deterministic Misspecification (валидация на гармонической функции)
    plot_single_scatter(out_dir / "fig3_neff_validation_deterministic.pdf", det_rows, "Deterministic")

    # Figure 4: Тепловая карта для GP
    plot_heatmaps(out_dir, gp_rows, "GP", r"GP True Length-scale $h$", filename="fig4_heatmap_gp.pdf")

    # Figure 5: Тепловая карта для детерминированной модели
    plot_heatmaps(out_dir, det_rows, "Deterministic", r"Sine Frequency $\omega$", filename="fig5_heatmap_deterministic.pdf")
    
    print(f"Done! Results and figures saved to {out_dir}")


if __name__ == "__main__":
    main()