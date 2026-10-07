"""Main Benchmark Experiment for Dependent KMM (Table 2 & Table 7).

Evaluates Target Mean (Y) and Target Risk (L) across all baseline methods:
- Unweighted (ignores shift)
- Nominal n (standard i.i.d. KMM)
- Block-Bootstrap (resampling baseline)
- Input neff (raw covariate autocorrelation correction)
- GP neff [Ours] (smoothness-aware GP correction)
- Strict Bound (Theorem 1 non-asymptotic concentration)
- Oracle IW (importance weighting with true density ratio)
- Oracle i.i.d. (independent target evaluation)

Usage:
    python run_main_benchmark.py [--n_reps 2200] [--seed 42] [--force]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
from typing import Dict, List

import numpy as np

from core import (
    GPResponse,
    block_bootstrap_ci,
    clipped_kmm_weights,
    compute_winkler_score,
    estimate_h,
    evaluate_ci,
    generate_gaussian_paths,
    get_oracle_density_ratio_weights,
    gp_formula_neff,
    neff_input,
    select_best_valid_method,
)


def run_benchmark(
    n: int = 96,
    ell_x: float = 3.0,
    h_nominal: float = 1.5,
    clip_B: float = 15.0,
    shift: float = 0.85,
    noise_sd: float = 0.04,
    marginal_sd: float = 1.0,
    n_reps: int = 30000,
    seed: int = 12345,
    force: bool = False,
) -> None:
    out_dir = Path(__file__).resolve().parent / "results" / "tables"
    csv_path = out_dir / "table7_main_benchmark.csv"

    if csv_path.exists() and not force:
        return

    start_time = time.time()
    rng = np.random.default_rng(seed)

    # 1. Define invariant latent response and predictor
    resp_func = GPResponse(rng_or_seed=seed, h=h_nominal)
    pred_func = lambda x: 0.65 * resp_func(x) + 0.15

    # 2. Ground-truth target expectations via large-scale Monte Carlo
    rng_truth = np.random.default_rng(9999)
    x_truth = shift + marginal_sd * rng_truth.standard_normal(150000)
    y_truth = np.clip(resp_func(x_truth) + noise_sd * rng_truth.standard_normal(150000), 0.0, 1.0)
    loss_truth = np.clip((pred_func(x_truth) - y_truth) ** 2, 0.0, 1.0)
    theta_Y = float(np.mean(y_truth))
    theta_Risk = float(np.mean(loss_truth))

    # 3. Pre-calibrate functional length-scales via variogram regression
    rng_calib = np.random.default_rng(8888)
    x_cal = marginal_sd * rng_calib.standard_normal(2500)
    y_cal = resp_func(x_cal)
    loss_cal = (pred_func(x_cal) - y_cal) ** 2
    h_Y_est = estimate_h(x_cal, y_cal)
    h_Risk_est = estimate_h(x_cal, loss_cal)

    # 4. Effective sample sizes
    n_eff_in = neff_input(n, ell_x)
    n_eff_gp_Y = gp_formula_neff(n, h_Y_est, ell_x, tau=marginal_sd, sigma0=1.0)
    n_eff_gp_Risk = gp_formula_neff(n, h_Risk_est, ell_x, tau=marginal_sd, sigma0=1.0)

    # 5. Pre-generate source trajectories and target batches
    x_source_all = generate_gaussian_paths(rng, n_reps, n, ell_x, mean=0.0, marginal_sd=marginal_sd)
    x_target_all = shift + marginal_sd * rng.standard_normal((n_reps, n))

    methods = [
        "Unweighted",
        "Nominal n",
        "Block-Bootstrap",
        "Input neff",
        "GP neff (Ours)",
        "Strict Thm 1",
        "Oracle IW",
        "Oracle i.i.d.",
    ]

    metrics = {
        est: {m: {"err": [], "hw": [], "cov": [], "ws": []} for m in methods}
        for est in ["Target_Y", "Risk"]
    }

    # 6. Monte Carlo Evaluation Loop
    for r in range(n_reps):
        xs = x_source_all[r]
        xt = x_target_all[r]

        # Dependent source observations
        ys = np.clip(resp_func(xs) + noise_sd * rng.standard_normal(n), 0.0, 1.0)
        ls = np.clip((pred_func(xs) - ys) ** 2, 0.0, 1.0)

        # Independent draws for Oracle i.i.d.
        xs_iid = marginal_sd * rng.standard_normal(n)
        xt_iid = shift + marginal_sd * rng.standard_normal(n)
        ys_iid = np.clip(resp_func(xs_iid) + noise_sd * rng.standard_normal(n), 0.0, 1.0)
        ls_iid = np.clip((pred_func(xs_iid) - ys_iid) ** 2, 0.0, 1.0)

        # Optimization and reweighting
        w_kmm = clipped_kmm_weights(xs, xt, bandwidth=1.0, clip=clip_B)
        w_oracle = get_oracle_density_ratio_weights(xs, shift=shift, marginal_sd=marginal_sd, clip=clip_B)
        w_iid_kmm = clipped_kmm_weights(xs_iid, xt_iid, bandwidth=1.0, clip=clip_B)
        w_un = np.ones(n, dtype=float)

        for estimand, vals, truth, gp_neff_val, iid_vals in [
            ("Target_Y", ys, theta_Y, n_eff_gp_Y, ys_iid),
            ("Risk", ls, theta_Risk, n_eff_gp_Risk, ls_iid),
        ]:
            # Unweighted
            res_un = evaluate_ci(vals, w_un, float(n), truth, clip_B=clip_B)
            # Nominal n
            res_nom = evaluate_ci(vals, w_kmm, float(n), truth, clip_B=clip_B)
            # Block-Bootstrap
            res_boot = block_bootstrap_ci(vals, w_kmm, truth, block_len=6, n_boot=200)
            # Input neff
            res_in = evaluate_ci(vals, w_kmm, n_eff_in, truth, clip_B=clip_B)
            # GP neff [Ours]
            res_gp = evaluate_ci(vals, w_kmm, gp_neff_val, truth, clip_B=clip_B)

            # Strict Theorem 1 bound
            tr = res_gp["strict_radius"]
            cov_s = float(res_gp["abs_error"] <= tr)
            low_s, up_s = res_gp["estimate"] - tr, res_gp["estimate"] + tr
            ws_s = compute_winkler_score(low_s, up_s, truth)
            res_strict = {
                "abs_error": res_gp["abs_error"],
                "half_width": tr,
                "covered": cov_s,
                "winkler": ws_s,
            }

            # Oracle IW
            res_oiw = evaluate_ci(vals, w_oracle, gp_neff_val, truth, clip_B=clip_B)
            # Oracle i.i.d.
            res_oiid = evaluate_ci(iid_vals, w_iid_kmm, float(n), truth, clip_B=clip_B)

            res_map = {
                "Unweighted": res_un,
                "Nominal n": res_nom,
                "Block-Bootstrap": res_boot,
                "Input neff": res_in,
                "GP neff (Ours)": res_gp,
                "Strict Thm 1": res_strict,
                "Oracle IW": res_oiw,
                "Oracle i.i.d.": res_oiid,
            }

            for m in methods:
                metrics[estimand][m]["err"].append(res_map[m]["abs_error"])
                metrics[estimand][m]["hw"].append(res_map[m]["half_width"])
                metrics[estimand][m]["cov"].append(res_map[m]["covered"])
                metrics[estimand][m]["ws"].append(res_map[m]["winkler"])

    # 7. Select best practical valid method
    best_Y = select_best_valid_method(metrics["Target_Y"], n_reps=n_reps)
    best_Risk = select_best_valid_method(metrics["Risk"], n_reps=n_reps)
    cov_threshold = 0.95 - 1.95996 * math.sqrt(0.95 * 0.05 / n_reps)

    # 8. Render Results Tables
    print("\n" + "=" * 115)
    print("Table 7: Full Benchmark Results including Point Estimation and Winkler Score")
    print(f"Configurations: n={n}, ell_x={ell_x}, h={h_nominal}, B={clip_B}, Repetitions={n_reps}")
    print("Asterisk (*) indicates significant undercoverage (< 0.925). Bold highlights best valid practical method.")
    print("=" * 115)
    print(
        f"| {'Method':<18} | {'MAE (Y)':<8} | {'Cov (Y)':<8} | {'Rad (Y)':<8} | {'WS (Y)':<8} | "
        f"{'MAE (L)':<8} | {'Cov (L)':<8} | {'Rad (L)':<8} | {'WS (L)':<8} |"
    )
    print(
        "| :----------------- | :------- | :------- | :------- | :------- | "
        ":------- | :------- | :------- | :------- |"
    )

    for m in methods:
        y_mae = float(np.mean(metrics["Target_Y"][m]["err"]))
        y_cov = float(np.mean(metrics["Target_Y"][m]["cov"]))
        y_rad = float(np.mean(metrics["Target_Y"][m]["hw"]))
        y_ws = float(np.mean(metrics["Target_Y"][m]["ws"]))

        l_mae = float(np.mean(metrics["Risk"][m]["err"]))
        l_cov = float(np.mean(metrics["Risk"][m]["cov"]))
        l_rad = float(np.mean(metrics["Risk"][m]["hw"]))
        l_ws = float(np.mean(metrics["Risk"][m]["ws"]))

        y_flag = "*" if y_cov < cov_threshold else ""
        l_flag = "*" if l_cov < cov_threshold else ""

        y_cov_str = f"{y_cov:.3f}{y_flag}"
        y_rad_str = f"{y_rad:.4f}"
        y_ws_str = f"{y_ws:.3f}"

        l_cov_str = f"{l_cov:.3f}{l_flag}"
        l_rad_str = f"{l_rad:.4f}"
        l_ws_str = f"{l_ws:.3f}"

        if m == best_Y:
            y_cov_str = f"**{y_cov_str}**"
            y_rad_str = f"**{y_rad_str}**"
            y_ws_str = f"**{y_ws_str}**"

        if m == best_Risk:
            l_cov_str = f"**{l_cov_str}**"
            l_rad_str = f"**{l_rad_str}**"
            l_ws_str = f"**{l_ws_str}**"

        print(
            f"| {m:<18} | {y_mae:<8.4f} | {y_cov_str:<8} | {y_rad_str:<8} | {y_ws_str:<8} | "
            f"{l_mae:<8.4f} | {l_cov_str:<8} | {l_rad_str:<8} | {l_ws_str:<8} |"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Method,MAE_Y,Coverage_Y,Radius_Y,WS_Y,MAE_Risk,Coverage_Risk,Radius_Risk,WS_Risk\n")
        for m in methods:
            y_mae = np.mean(metrics["Target_Y"][m]["err"])
            y_cov = np.mean(metrics["Target_Y"][m]["cov"])
            y_rad = np.mean(metrics["Target_Y"][m]["hw"])
            y_ws = np.mean(metrics["Target_Y"][m]["ws"])
            l_mae = np.mean(metrics["Risk"][m]["err"])
            l_cov = np.mean(metrics["Risk"][m]["cov"])
            l_rad = np.mean(metrics["Risk"][m]["hw"])
            l_ws = np.mean(metrics["Risk"][m]["ws"])
            y_f = "*" if y_cov < cov_threshold else ""
            l_f = "*" if l_cov < cov_threshold else ""
            f.write(f"{m},{y_mae:.4f},{y_cov:.3f}{y_f},{y_rad:.4f},{y_ws:.3f},{l_mae:.4f},{l_cov:.3f}{l_f},{l_rad:.4f},{l_ws:.3f}\n")
    print(f"Artifacts successfully written to {csv_path}")
    
    print("=" * 115)
    print(f"Benchmark finished in {time.time() - start_time:.2f} seconds.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dependent KMM Main Benchmark (Table 2 & Table 7).")
    parser.add_argument("--n_reps", type=int, default=30000, help="Number of Monte Carlo trials (default: 300).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if table7_main_benchmark.csv already exists.",
    )
    args = parser.parse_args()

    run_benchmark(n_reps=args.n_reps, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()