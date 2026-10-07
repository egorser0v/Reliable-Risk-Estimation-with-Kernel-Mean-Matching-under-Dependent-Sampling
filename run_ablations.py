"""Hyperparameter and Structural Ablation Sweeps for Dependent KMM.

Reproduces Tables 8, 9, 10, and 11 from the Supplementary Materials:
- Table 8:  Sample size sweep (n in {32, 48, 64, 96, 128, 192, 256})
- Table 9:  Input persistence length (ell_x in {1.5, 2.0, 3.0, 4.0, 5.0, 6.0})
- Table 10: Functional smoothness length-scale (h in {1.0, 1.25, 1.5, 1.75, 2.0, 3.0})
- Table 11: Weight clipping threshold (B in {5.0, 7.5, 10.0, 15.0, 20.0, 30.0})

Usage:
    python run_ablations.py [--param {all,n,ell_x,h,clip_B}] [--n_reps 30000] [--seed 42] [--force]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core import (
    GPResponse,
    clipped_kmm_weights,
    compute_winkler_score,
    estimate_h,
    evaluate_ci,
    generate_gaussian_paths,
    gp_formula_neff,
    neff_input,
    select_best_valid_method,
)


TABLE_METADATA = {
    "n": {
        "title": "Table 8: Sample-Size Scaling of Target Risk Intervals",
        "caption": r"\textbf{Sample-size scaling of Target Risk intervals ($n \in \{32, \dots, 256\}$).} Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage). Asterisks ($^*$) indicate significant empirical undercoverage below the nominal 0.95 level ($< 0.925$). Strict Thm 1 is excluded from best-method highlighting due to worst-case conservatism.",
        "label": "tab:n_scaling_risk",
        "param_latex": r"\bm{n}",
    },
    "ell_x": {
        "title": "Table 9: Input Dependence Ablation (ell_x) on Target Risk",
        "caption": r"\textbf{Hyperparameter Ablation: Input Dependence ($\ell_x \in [1.5, 6.0]$) on Target Risk.} Asterisks ($^*$) indicate significant empirical undercoverage below the nominal 0.95 level ($< 0.925$). Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage). Strict Thm 1 is excluded from best-method highlighting due to worst-case conservatism.",
        "label": "tab:ablation-ellx",
        "param_latex": r"\bm{\ell_x}",
    },
    "h": {
        "title": "Table 10: Functional Smoothness Ablation (h) on Target Risk",
        "caption": r"\textbf{Hyperparameter Ablation: Functional Smoothness ($h \in [1.0, 3.0]$) on Target Risk.} Asterisks ($^*$) indicate significant empirical undercoverage below the nominal 0.95 level ($< 0.925$). Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage).",
        "label": "tab:ablation-h",
        "param_latex": r"\bm{h}",
    },
    "clip_B": {
        "title": "Table 11: Clipping Bound Ablation (B) on Target Risk",
        "caption": r"\textbf{Hyperparameter Ablation: Clipping Bound ($B \in [5.0, 30.0]$) on Target Risk.} Asterisks ($^*$) indicate significant empirical undercoverage below the nominal 0.95 level ($< 0.925$). Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage).",
        "label": "tab:ablation-B",
        "param_latex": r"\bm{B}",
    },
}


def method_to_latex(method: str) -> str:
    """Formats method name for LaTeX tables."""
    mapping = {
        "Unweighted": "Unweighted",
        "Nominal n": r"Nominal $n$",
        "Input neff": r"Input $n_{\mathrm{eff}}$",
        "GP neff (Ours)": r"GP $n_{\mathrm{eff}}$ (Ours)",
        "Strict Thm 1": "Strict Thm 1",
    }
    return mapping.get(method, method)


def export_tables(
    param_name: str,
    collected_rows: List[Dict[str, Any]],
    output_dir: str = "results/tables",
) -> None:
    """Saves benchmark ablation results to both CSV and LaTeX formats."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_file = out_path / f"table_ablation_{param_name}.csv"
    tex_file = out_path / f"table_ablation_{param_name}.tex"

    # 1. Export CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["param_val", "method", "coverage", "radius", "winkler_score", "neff", "is_valid", "is_best"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in collected_rows:
            writer.writerow(row)

    # 2. Export LaTeX
    meta = TABLE_METADATA.get(param_name, {
        "caption": f"Ablation sweep on {param_name}.",
        "label": f"tab:ablation_{param_name}",
        "param_latex": rf"\bm{{{param_name}}}",
    })

    # Group rows by param_val
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for r in collected_rows:
        grouped.setdefault(r["param_val"], []).append(r)

    tex_lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        f"\\caption{{{meta['caption']}}}",
        f"\\label{{{meta['label']}}}",
        r"\vspace{2mm}",
        r"\resizebox{0.78\textwidth}{!}{%",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        f"{meta['param_latex']} & \\textbf{{Method}} & \\textbf{{Coverage}} & \\textbf{{Radius}} & \\textbf{{Winkler Score}} $\\downarrow$ & $\\bm{{n_{{\\mathrm{{eff}}}}}}$ \\\\",
        r"\midrule",
    ]

    for val, rows in grouped.items():
        n_rows = len(rows)
        # Format param display
        if isinstance(val, float) and val.is_integer():
            val_str = str(int(val))
        elif param_name == "h":
            val_str = f"{val:.2f}"
        else:
            val_str = f"{val}"

        for i, row in enumerate(rows):
            m_tex = method_to_latex(row["method"])
            cov_val = f"{row['coverage']:.3f}"
            cov_flag = r"$^*$" if not row["is_valid"] else "    "
            rad_val = f"{row['radius']:.4f}"
            ws_val = f"{row['winkler_score']:.4f}"
            neff_val = f"{row['neff']:.1f}"

            if row["is_best"]:
                m_tex = rf"\textbf{{{m_tex}}}"
                cov_str = rf"\textbf{{{cov_val}}}"
                rad_str = rf"\textbf{{{rad_val}}}"
                ws_str = rf"\textbf{{{ws_val}}}"
                neff_str = rf"\textbf{{{neff_val}}}"
            else:
                cov_str = f"{cov_val}{cov_flag}"
                rad_str = rad_val
                ws_str = ws_val
                neff_str = neff_val

            if i == 0:
                prefix = rf"\multirow{{{n_rows}}}{{*}}{{{val_str}}}"
            else:
                prefix = ""

            tex_lines.append(f"{prefix:<8} & {m_tex:<38} & {cov_str:<12} & {rad_str:<10} & {ws_str:<10} & {neff_str} \\\\")

        tex_lines.append(r"\midrule" if val != list(grouped.keys())[-1] else r"\bottomrule")

    tex_lines.extend([
        r"\end{tabular}}",
        r"\end{table*}",
        "",
    ])

    with open(tex_file, mode="w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))

    print(f"\n[SAVED] CSV Table:   {csv_file}")
    print(f"[SAVED] LaTeX Table: {tex_file}")


def run_ablation_sweep(
    param_name: str,
    grid_values: List[float],
    base_cfg: Dict[str, Any],
    master_seed: int = 12345,
    n_reps: int = 300000,
    force: bool = False,
    output_dir: str = "results/tables",
) -> None:
    csv_file = Path(output_dir) / f"table_ablation_{param_name}.csv"
    tex_file = Path(output_dir) / f"table_ablation_{param_name}.tex"

    if csv_file.exists() and tex_file.exists() and not force:
        return

    start_time = time.time()
    cov_threshold = 0.95 - 1.95996 * math.sqrt(0.95 * 0.05 / n_reps)

    # Base canonical generative model
    canonical_resp_func = GPResponse(rng_or_seed=master_seed, h=float(base_cfg["h"]))

    rng_truth = np.random.default_rng(9999)
    x_truth = base_cfg["shift"] + rng_truth.standard_normal(150000)
    y_truth = np.clip(canonical_resp_func(x_truth) + base_cfg["noise_sd"] * rng_truth.standard_normal(150000), 0.0, 1.0)
    canonical_pred_func = lambda x: 0.65 * canonical_resp_func(x) + 0.15
    loss_truth = np.clip((canonical_pred_func(x_truth) - y_truth) ** 2, 0.0, 1.0)
    canonical_theta_risk = float(np.mean(loss_truth))

    rng_calib = np.random.default_rng(8888)
    x_cal = rng_calib.standard_normal(2500)
    loss_cal = (0.65 * canonical_resp_func(x_cal) + 0.15 - canonical_resp_func(x_cal)) ** 2
    canonical_h_risk = estimate_h(x_cal, loss_cal)

    # For clipping ablation B, fix trajectory draws across B to ensure paired comparison
    if param_name == "clip_B":
        rng_fixed = np.random.default_rng(master_seed)
        fixed_xs = generate_gaussian_paths(
            rng_fixed, n_reps, base_cfg["n"], base_cfg["ell_x"], mean=0.0, marginal_sd=1.0
        )
        fixed_xt = base_cfg["shift"] + rng_fixed.standard_normal((n_reps, base_cfg["n"]))
        fixed_ys = np.clip(canonical_resp_func(fixed_xs) + base_cfg["noise_sd"] * rng_fixed.standard_normal((n_reps, base_cfg["n"])), 0.0, 1.0)
        fixed_losses = np.clip((canonical_pred_func(fixed_xs) - fixed_ys) ** 2, 0.0, 1.0)

    title_map = {k: v["title"] for k, v in TABLE_METADATA.items()}

    print("\n" + "=" * 105)
    print(title_map.get(param_name, f"Ablation on {param_name}"))
    print("Asterisk (*) denotes significant undercoverage (< 0.925). Bold denotes best valid practical method.")
    print("=" * 105)
    print(f"| {param_name:<8} | {'Method':<18} | {'Coverage':<10} | {'Radius':<10} | {'Winkler Score':<15} | {'neff':<8} |")
    print(f"| :{'-'*7} | :{'-'*17} | :{'-'*9} | :{'-'*9} | :{'-'*14} | :{'-'*7} |")

    methods = ["Unweighted", "Nominal n", "Input neff", "GP neff (Ours)"]
    collected_rows: List[Dict[str, Any]] = []

    for val in grid_values:
        cfg = base_cfg.copy()
        cfg[param_name] = val
        n_val = int(cfg["n"])
        ell_x_val = float(cfg["ell_x"])
        h_val = float(cfg["h"])
        clip_b_val = float(cfg["clip_B"])

        if param_name == "h":
            resp_func = GPResponse(rng_or_seed=master_seed + int(h_val * 100), h=h_val)
            pred_func = lambda x: 0.65 * resp_func(x) + 0.15
            xt_h = cfg["shift"] + rng_truth.standard_normal(150000)
            yt_h = np.clip(resp_func(xt_h) + cfg["noise_sd"] * rng_truth.standard_normal(150000), 0.0, 1.0)
            theta_risk = float(np.mean((pred_func(xt_h) - yt_h) ** 2))
            x_cal_h = rng_calib.standard_normal(2500)
            h_risk = estimate_h(x_cal_h, (pred_func(x_cal_h) - resp_func(x_cal_h)) ** 2)
        else:
            resp_func = canonical_resp_func
            pred_func = canonical_pred_func
            theta_risk = canonical_theta_risk
            h_risk = canonical_h_risk

        neff_in_val = neff_input(n_val, ell_x_val)
        neff_gp_val = gp_formula_neff(n_val, h_risk, ell_x_val)

        if param_name == "clip_B":
            xs_paths, xt_paths, ls_paths = fixed_xs, fixed_xt, fixed_losses
        else:
            rng_data = np.random.default_rng(master_seed + int(val * 10))
            xs_paths = generate_gaussian_paths(rng_data, n_reps, n_val, ell_x_val, mean=0.0, marginal_sd=1.0)
            xt_paths = cfg["shift"] + rng_data.standard_normal((n_reps, n_val))
            ys_paths = np.clip(resp_func(xs_paths) + cfg["noise_sd"] * rng_data.standard_normal((n_reps, n_val)), 0.0, 1.0)
            ls_paths = np.clip((pred_func(xs_paths) - ys_paths) ** 2, 0.0, 1.0)

        records: Dict[str, Dict[str, List[float]]] = {
            m: {"hw": [], "cov": [], "ws": []} for m in methods
        }
        strict_records: Dict[str, List[float]] = {"hw": [], "cov": [], "ws": []}

        for r in range(n_reps):
            xs, xt, ls = xs_paths[r], xt_paths[r], ls_paths[r]
            w_kmm = clipped_kmm_weights(xs, xt, bandwidth=1.0, clip=clip_b_val)
            w_un = np.ones(n_val, dtype=float)

            res_un = evaluate_ci(ls, w_un, float(n_val), theta_risk, clip_B=clip_b_val)
            res_nom = evaluate_ci(ls, w_kmm, float(n_val), theta_risk, clip_B=clip_b_val)
            res_in = evaluate_ci(ls, w_kmm, neff_in_val, theta_risk, clip_B=clip_b_val)
            res_gp = evaluate_ci(ls, w_kmm, neff_gp_val, theta_risk, clip_B=clip_b_val)

            records["Unweighted"]["hw"].append(res_un["half_width"])
            records["Unweighted"]["cov"].append(res_un["covered"])
            records["Unweighted"]["ws"].append(res_un["winkler"])

            records["Nominal n"]["hw"].append(res_nom["half_width"])
            records["Nominal n"]["cov"].append(res_nom["covered"])
            records["Nominal n"]["ws"].append(res_nom["winkler"])

            records["Input neff"]["hw"].append(res_in["half_width"])
            records["Input neff"]["cov"].append(res_in["covered"])
            records["Input neff"]["ws"].append(res_in["winkler"])

            records["GP neff (Ours)"]["hw"].append(res_gp["half_width"])
            records["GP neff (Ours)"]["cov"].append(res_gp["covered"])
            records["GP neff (Ours)"]["ws"].append(res_gp["winkler"])

            # Strict Theorem 1 tracker
            tr = res_gp["strict_radius"]
            cov_s = float(res_gp["abs_error"] <= tr)
            strict_records["hw"].append(tr)
            strict_records["cov"].append(cov_s)
            strict_records["ws"].append(compute_winkler_score(res_gp["estimate"] - tr, res_gp["estimate"] + tr, theta_risk))

        best_method = select_best_valid_method(records, n_reps=n_reps)

        for idx, m in enumerate(methods):
            cov = float(np.mean(records[m]["cov"]))
            rad = float(np.mean(records[m]["hw"]))
            ws = float(np.mean(records[m]["ws"]))

            flag = "*" if cov < cov_threshold else ""
            cov_str = f"{cov:.3f}{flag}"
            rad_str = f"{rad:.4f}"
            ws_str = f"{ws:.4f}"

            if m in ["Unweighted", "Nominal n"]:
                neff_float = float(n_val)
            elif m == "Input neff":
                neff_float = neff_in_val
            else:
                neff_float = neff_gp_val
            neff_str = f"{neff_float:.1f}"

            is_best = (m == best_method)
            is_valid = (cov >= cov_threshold)

            collected_rows.append({
                "param_val": val,
                "method": m,
                "coverage": round(cov, 3),
                "radius": round(rad, 4),
                "winkler_score": round(ws, 4),
                "neff": round(neff_float, 1),
                "is_valid": is_valid,
                "is_best": is_best,
            })

            if is_best:
                cov_str = f"**{cov_str}**"
                rad_str = f"**{rad_str}**"
                ws_str = f"**{ws_str}**"

            param_disp = f"{val}" if idx == 0 else ""
            print(f"| {param_disp:<8} | {m:<18} | {cov_str:<10} | {rad_str:<10} | {ws_str:<15} | {neff_str:<8} |")

        # Table 9 in paper explicitly reports Strict Bound at ell_x = 2.0
        if param_name == "ell_x" and math.isclose(val, 2.0):
            cov_s = float(np.mean(strict_records['cov']))
            rad_s = float(np.mean(strict_records['hw']))
            ws_s = float(np.mean(strict_records['ws']))
            print(f"| {'':<8} | {'Strict Thm 1':<18} | {cov_s:.3f}{' ':<7} | {rad_s:.4f}{' ':<4} | {ws_s:.4f}{' ':<9} | {neff_gp_val:.1f}    |")
            collected_rows.append({
                "param_val": val,
                "method": "Strict Thm 1",
                "coverage": round(cov_s, 3),
                "radius": round(rad_s, 4),
                "winkler_score": round(ws_s, 4),
                "neff": round(neff_gp_val, 1),
                "is_valid": True,
                "is_best": False,
            })

    export_tables(param_name, collected_rows, output_dir=output_dir)
    print("=" * 105)
    print(f"Ablation for {param_name} completed in {time.time() - start_time:.2f} seconds.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hyperparameter Ablations for Dependent KMM.")
    parser.add_argument(
        "--param",
        type=str,
        default="all",
        choices=["all", "n", "ell_x", "h", "clip_B"],
        help="Ablation parameter to evaluate (default: all).",
    )
    parser.add_argument("--n_reps", type=int, default=300000, help="Number of Monte Carlo trials (default: 300).")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42).")
    parser.add_argument("--force", action="store_true", help="Force recalculation and overwrite existing tables.")
    parser.add_argument("--tables_dir", type=str, default="results/tables", help="Output directory for tables.")
    args = parser.parse_args()

    base_config = {
        "n": 96,
        "ell_x": 3.0,
        "h": 1.5,
        "clip_B": 15.0,
        "shift": 0.85,
        "noise_sd": 0.04,
    }

    grids = {
        "n": [32, 48, 64, 96, 128, 192, 256],
        "ell_x": [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        "h": [1.0, 1.25, 1.5, 1.75, 2.0, 3.0],
        "clip_B": [5.0, 7.5, 10.0, 15.0, 20.0, 30.0],
    }

    if args.param == "all":
        for p in ["n", "ell_x", "h", "clip_B"]:
            run_ablation_sweep(p, grids[p], base_config, master_seed=args.seed, n_reps=args.n_reps, force=args.force, output_dir=args.tables_dir)
    else:
        run_ablation_sweep(args.param, grids[args.param], base_config, master_seed=args.seed, n_reps=args.n_reps, force=args.force, output_dir=args.tables_dir)


if __name__ == "__main__":
    main()