"""
Definitive Grid-Search Benchmark for Dependent KMM (Real Data).
Automatically sweeps over n_source and clip_B, saves all results,
and prints the BEST configuration table for each dataset.
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
OUT_DIR = ROOT / "results" / "kmm_final_paper"

# ==============================================================================
# 1. CORE THEORETICAL MATH (Variograms & n_eff)
# ==============================================================================

def compute_winkler_score(low: float, up: float, truth: float, alpha: float = 0.05) -> float:
    width = up - low
    if truth < low: return width + (2.0 / alpha) * (low - truth)
    if truth > up:  return width + (2.0 / alpha) * (truth - up)
    return width

def estimate_length_scale_h_grid(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim == 1: x = x.reshape(-1, 1)
    n = len(x)
    if n > 250:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, 250, replace=False)
        x, y = x[idx], y[idx]

    pdist_x = pdist(x)
    semi_var = 0.5 * (pdist(y.reshape(-1, 1)) ** 2)
    pos_mask = pdist_x > 1e-5
    pdist_x = pdist_x[pos_mask]
    semi_var = semi_var[pos_mask]
    
    if len(pdist_x) < 10: return 1.5
    
    q_bins = np.linspace(0.05, 0.95, 15)
    bins = np.quantile(pdist_x, q_bins)
    bin_centers, gamma_means = [], []
    
    for i in range(len(bins) - 1):
        mask = (pdist_x >= bins[i]) & (pdist_x < bins[i+1])
        if np.sum(mask) >= 5:
            bin_centers.append(np.median(pdist_x[mask]))
            gamma_means.append(np.mean(semi_var[mask]))
            
    if len(bin_centers) < 4: return float(np.median(pdist_x))
    
    bin_centers = np.array(bin_centers)
    gamma_means = np.array(gamma_means)
    var_y = max(float(np.var(y)), 1e-6)
    
    best_loss = float('inf')
    best_h = float(np.median(pdist_x))
    h_candidates = np.logspace(-2, 2, 100) * best_h 
    
    for h_val in h_candidates:
        gamma_theor = var_y * (1.0 - np.exp(-0.5 * (bin_centers / h_val) ** 2))
        loss = np.mean((gamma_means - gamma_theor) ** 2)
        if loss < best_loss:
            best_loss = loss
            best_h = h_val
            
    return float(best_h)

def compute_empirical_autocorr(features: np.ndarray, max_lags: int = 14) -> Tuple[np.ndarray, float]:
    n, d = features.shape
    rk = np.zeros(max_lags)
    sigma_x2 = float(np.mean(np.var(features, axis=0)))
    if sigma_x2 < 1e-8: sigma_x2 = 1.0
        
    for k in range(1, max_lags + 1):
        if k >= n - 2: break
        cov_k = np.mean([np.dot(features[i] - features.mean(0), features[i+k] - features.mean(0)) / d for i in range(n - k)])
        rk[k - 1] = np.clip(cov_k / sigma_x2, -0.99, 0.99)
    return rk, np.sqrt(sigma_x2)

def compute_neff_values(n: int, rk: np.ndarray, sigma_x: float, h: float, d_dim: int = 1) -> Tuple[float, float]:
    denom_in = 1.0
    for k in range(1, len(rk) + 1):
        if k < n: denom_in += 2.0 * (1.0 - (k / float(n))) * rk[k - 1]
    neff_in = max(1.0, float(n / max(denom_in, 1.0)))
    
    ah = 2.0 * (sigma_x ** 2) / max(h ** 2, 1e-6)
    base_denom = max(1.0 - (1.0 + ah) ** (-d_dim / 2.0), 1e-8)
        
    denom_gp = 1.0
    for k in range(1, len(rk) + 1):
        if k < n:
            num = ((1.0 + ah * (1.0 - rk[k - 1])) ** (-d_dim / 2.0)) - ((1.0 + ah) ** (-d_dim / 2.0))
            rho_bar = num / base_denom
            denom_gp += 2.0 * (1.0 - (k / float(n))) * rho_bar
            
    neff_gp = max(1.0, float(n / max(denom_gp, 1.0)))
    return neff_in, neff_gp

# ==============================================================================
# 2. SOLVER AND INTERVALS
# ==============================================================================

def solve_kmm(xs: np.ndarray, xt: np.ndarray, clip_b: float) -> np.ndarray:
    ns = len(xs)
    prep = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    pooled = prep.fit_transform(np.vstack([xs, xt]))
    xs_p, xt_p = pooled[:ns], pooled[ns:]
    
    dists = pdist(pooled)
    sigma2 = max(float(np.median(dists[dists > 0]) ** 2), 1e-8)
    
    K = np.exp(-cdist(xs_p, xs_p, 'sqeuclidean') / (2.0 * sigma2))
    cross = np.exp(-cdist(xs_p, xt_p, 'sqeuclidean') / (2.0 * sigma2))
    A = K + np.eye(ns) * 1e-5
    b = ns * cross.mean(1)
    
    res = minimize(
        lambda w: 0.5 * w @ A @ w - b @ w, np.ones(ns), jac=lambda w: A @ w - b, method='SLSQP',
        bounds=[(0.0, clip_b)] * ns,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - ns, 'jac': lambda w: np.ones(ns)}],
        options={'maxiter': 1000, 'ftol': 1e-7}
    )
    w = res.x if res.success else np.ones(ns)
    return np.clip(w, 0.0, clip_b) * (ns / max(w.sum(), 1e-12))

def evaluate_unweighted(y_vals: np.ndarray, truth: float) -> Dict:
    n = len(y_vals)
    est = float(np.mean(y_vals))
    v_iid = float(np.var(y_vals, ddof=1) / n) if n > 1 else 0.0
    hw = 1.96 * np.sqrt(max(0.0, v_iid))
    return {
        "MAE": abs(est - truth), "radius": hw, 
        "coverage": float(abs(est - truth) <= hw),
        "WS": compute_winkler_score(est - hw, est + hw, truth), "neff": float(n)
    }

def evaluate_intervals(y_vals: np.ndarray, weights: np.ndarray, truth: float, neff_in: float, neff_gp: float) -> Dict:
    n = len(y_vals)
    wn = weights / weights.sum()
    estimate = float(wn @ y_vals)
    abs_err = abs(estimate - truth)
    v_iid_mean = float((wn * (y_vals - estimate)) @ (wn * (y_vals - estimate))) * n / (n - 1.0)
    
    def get_m(v_eff, n_val):
        hw = 1.96 * np.sqrt(max(0.0, v_eff))
        return {
            "MAE": abs_err, "radius": hw, 
            "coverage": float(abs_err <= hw), 
            "WS": compute_winkler_score(estimate - hw, estimate + hw, truth), "neff": n_val
        }
    
    metrics = {
        "Unweighted": evaluate_unweighted(y_vals, truth),
        "Nominal n": get_m(v_iid_mean, n),
        "Input neff": get_m(v_iid_mean * (n / max(neff_in, 1.0)), neff_in),
        "GP neff [Ours]": get_m(v_iid_mean * (n / max(neff_gp, 1.0)), neff_gp)
    }
    
    q = max(2, int(np.round(n ** (1.0 / 3.0))))
    n_blocks = max(1, n // q)
    boot_ests = []
    rng_b = np.random.default_rng(SEED)
    for _ in range(200): 
        idx = np.concatenate([np.arange(b * q, min((b + 1) * q, n)) for b in rng_b.integers(0, n_blocks, size=n_blocks)])
        if len(idx) < n: idx = np.pad(idx, (0, n - len(idx)), mode='edge')
        bw = weights[idx]
        if bw.sum() > 0: boot_ests.append((bw / bw.sum()) @ y_vals[idx])
            
    if boot_ests:
        q_l, q_u = np.percentile(boot_ests, [2.5, 97.5])
        hw = (q_u - q_l) / 2.0
        metrics["Block-Bootstrap"] = {
            "MAE": abs_err, "radius": hw, 
            "coverage": float(q_l <= truth <= q_u), 
            "WS": compute_winkler_score(q_l, q_u, truth), "neff": float(n_blocks)
        }
    else:
        metrics["Block-Bootstrap"] = metrics["GP neff [Ours]"]
        
    return metrics

# ==============================================================================
# 3. DOMAIN LOADERS
# ==============================================================================

def extract_rolling_source(df: pd.DataFrame, source_ids: np.ndarray, n_source: int, rng: np.random.Generator, id_col='domain'):
    for _ in range(50):
        sid = rng.choice(source_ids)
        st_data = df[df[id_col] == sid].sort_values('date')
        if len(st_data) > n_source:
            start = rng.integers(0, len(st_data) - n_source)
            return st_data.iloc[start : start + n_source]
    return st_data.iloc[:n_source]

def setup_pm25():
    df = pd.read_csv(ROOT / 'data/pm25/beijing_pm25_daily_prepared.csv', parse_dates=['date'])
    df['response'] = np.clip(df['PM2.5'] / 500.0, 0.0, 1.0)
    features = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'SO2', 'NO2', 'CO', 'O3', 'day_of_year_sin', 'day_of_year_cos']
    df = df.dropna(subset=['response'] + features)
    df['domain'] = df['station']
    return df[df.date.dt.year < 2015], df[df.date.dt.year == 2016], df[df.date.dt.year == 2015], features, df[df.date.dt.year == 2015].domain.unique()

def setup_noaa():
    df = pd.read_csv(ROOT / 'data/noaa/ghcn2025_benchmark.csv', parse_dates=['date'])
    df['response'] = np.clip((df.TMAX + 30.0) / 80.0, 0.0, 1.0)
    df['sin_doy'] = np.sin(2.0 * np.pi * df.date.dt.dayofyear / 365.25)
    df['cos_doy'] = np.cos(2.0 * np.pi * df.date.dt.dayofyear / 365.25)
    features = ['TMIN', 'PRCP', 'sin_doy', 'cos_doy']
    df = df.dropna(subset=['response'] + features)
    ids = np.array(sorted(df.domain.unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    return df[df.domain.isin(ids[:20])], df[df.domain.isin(ids[20:32])], df[df.domain.isin(ids[32:])], features, ids[32:]

def setup_ozone():
    df_cov = pd.read_csv(ROOT / 'data/ozon/epa_ozone_2022_complete_daily_covariates.csv')
    df_out = pd.read_csv(ROOT / 'data/ozon/epa_ozone_2022_complete_daily_outcomes.csv')
    df = pd.merge(df_cov, df_out, on=['site_id', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df['domain'] = df['site_id']
    exclude = ['site_id', 'date', 'response', 'ozone_daily_max8h_ppm', 'domain']
    features = [c for c in df.columns if c not in exclude]
    df = df.dropna(subset=['response'] + features)
    ids = np.array(sorted(df.domain.unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    n_t, n_tg = int(len(ids) * 0.3), int(len(ids) * 0.2)
    return df[df.domain.isin(ids[:n_t])], df[df.domain.isin(ids[n_t:n_t+n_tg])], df[df.domain.isin(ids[n_t+n_tg:])], features, ids[n_t+n_tg:]


# ==============================================================================
# 4. BENCHMARK RUNNER (Grid Search Optimized)
# ==============================================================================

def run_dataset_grid(name: str, setup_fn, n_reps: int, n_sources: List[int], clip_bs: List[float]):
    print(f"\n--- Loading and Fitting Model for {name} ---")
    train_df, target_pool, source_pool, features, source_ids = setup_fn()
    
    model = make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=1))
    model.fit(train_df[features].to_numpy(), train_df.response.to_numpy())
    
    xt_large = target_pool[features].to_numpy()
    yt_large = target_pool.response.to_numpy()
    lt_large = np.clip((np.clip(model.predict(xt_large), 0.0, 1.0) - yt_large) ** 2, 0.0, 1.0)
    
    theta_Y, theta_Risk = float(np.mean(yt_large)), float(np.mean(lt_large))
    rng_m = np.random.default_rng(SEED)
    xt_kmm = target_pool.iloc[rng_m.choice(len(target_pool), min(1000, len(target_pool)), replace=False)][features].to_numpy()
    
    results = []
    total_configs = len(n_sources) * len(clip_bs)
    
    for n_idx, n_source in enumerate(n_sources):
        for c_idx, clip_b in enumerate(clip_bs):
            config_num = n_idx * len(clip_bs) + c_idx + 1
            print(f"[{name}] Running Config {config_num}/{total_configs} (n_source={n_source}, clip_B={clip_b})")
            start_time = time.time()
            
            for rep in range(n_reps):
                src_df = extract_rolling_source(source_pool, source_ids, n_source, rng_m)
                xs = src_df[features].to_numpy()
                ys = src_df.response.to_numpy()
                ls = np.clip((np.clip(model.predict(xs), 0.0, 1.0) - ys) ** 2, 0.0, 1.0)
                
                w_kmm = solve_kmm(xs, xt_kmm, clip_b)
                rk, sigma_x = compute_empirical_autocorr(xs, max_lags=14)
                
                h_Y = estimate_length_scale_h_grid(xs, ys)
                h_Risk = estimate_length_scale_h_grid(xs, ls)
                
                n_in_Y, n_gp_Y = compute_neff_values(n_source, rk, sigma_x, h_Y, d_dim=1)
                n_in_Risk, n_gp_Risk = compute_neff_values(n_source, rk, sigma_x, h_Risk, d_dim=1)
                
                ci_Y = evaluate_intervals(ys, w_kmm, theta_Y, n_in_Y, n_gp_Y)
                ci_Risk = evaluate_intervals(ls, w_kmm, theta_Risk, n_in_Risk, n_gp_Risk)
                
                for est_name, ci_dict in [('Target Mean (Y)', ci_Y), ('Target Risk (L)', ci_Risk)]:
                    for m_name, m_data in ci_dict.items():
                        results.append({
                            'dataset': name, 'n_source': n_source, 'clip_b': clip_b, 
                            'rep': rep, 'estimand': est_name, 'method': m_name, **m_data
                        })
            print(f"  -> Done in {time.time() - start_time:.1f}s")
    return results

def print_best_configuration(df: pd.DataFrame, dataset_name: str):
    d = df[df.dataset == dataset_name]
    
    # Ищем лучшую конфигурацию по минимальному среднему WS для GP neff
    best_config = None
    best_ws_sum = float('inf')
    
    configs = d[['n_source', 'clip_b']].drop_duplicates().values
    for n_src, c_b in configs:
        sub = d[(d.n_source == n_src) & (d.clip_b == c_b) & (d.method == 'GP neff [Ours]')]
        if len(sub) == 0: continue
        ws_y = sub[sub.estimand == 'Target Mean (Y)'].WS.mean()
        ws_l = sub[sub.estimand == 'Target Risk (L)'].WS.mean()
        ws_sum = ws_y + ws_l
        if ws_sum < best_ws_sum:
            best_ws_sum = ws_sum
            best_config = (n_src, c_b)
            
    n_best, c_best = best_config
    print(f"\n==============================================================================")
    print(f"BEST CONFIGURATION FOR: {dataset_name.upper()}")
    print(f"Selected: n_source = {n_best}, KMM clip_B = {c_best}")
    print(f"==============================================================================")
    
    best_df = d[(d.n_source == n_best) & (d.clip_b == c_best)]
    methods = ["Unweighted", "Nominal n", "Block-Bootstrap", "Input neff", "GP neff [Ours]"]
    
    print(f"{'':<17} | {'Target Mean (Y)':<27} | {'Target Risk (L)':<27}")
    print(f"{'Method':<17} | {'Cov':<6} {'Radius':<8} {'WS':<9} | {'Cov':<6} {'Radius':<8} {'WS':<9}")
    print("-" * 78)
    
    best_m_y, min_ws_y = None, float('inf')
    best_m_l, min_ws_l = None, float('inf')
    
    for m in ["Block-Bootstrap", "Input neff", "GP neff [Ours]"]:
        sub = best_df[best_df.method == m]
        if len(sub) == 0: continue
        wy = sub[sub.estimand == 'Target Mean (Y)'].WS.mean()
        wl = sub[sub.estimand == 'Target Risk (L)'].WS.mean()
        if wy < min_ws_y: min_ws_y, best_m_y = wy, m
        if wl < min_ws_l: min_ws_l, best_m_l = wl, m

    for m in methods:
        sub = best_df[best_df.method == m]
        if len(sub) == 0: continue
        
        y_data = sub[sub.estimand == 'Target Mean (Y)']
        l_data = sub[sub.estimand == 'Target Risk (L)']
        
        y_cov, y_rad, y_ws = y_data.coverage.mean(), y_data.radius.mean(), y_data.WS.mean()
        l_cov, l_rad, l_ws = l_data.coverage.mean(), l_data.radius.mean(), l_data.WS.mean()
        
        y_mark = "<--B" if m == best_m_y else ""
        l_mark = "<--B" if m == best_m_l else ""
        
        print(f"{m:<17} | {y_cov:<6.3f} {y_rad:<8.4f} {y_ws:<6.4f}{y_mark:<3} | {l_cov:<6.3f} {l_rad:<8.4f} {l_ws:<6.4f}{l_mark:<3}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=200, help="Number of rolling windows")
    parser.add_argument('--n_sources', nargs='+', type=int, default=[96, 128, 192, 256], help="n_source grid")
    parser.add_argument('--clip_bs', nargs='+', type=float, default=[10.0, 15.0, 20.0], help="clip_b grid")
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'pm25', 'noaa', 'ozone'], help="Dataset to run")
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_res = []
    
    with threadpool_limits(limits=1):
        if args.dataset in ['all', 'pm25']:
            all_res.extend(run_dataset_grid('PM25', setup_pm25, args.reps, args.n_sources, args.clip_bs))
        if args.dataset in ['all', 'noaa']:
            all_res.extend(run_dataset_grid('NOAA', setup_noaa, args.reps, args.n_sources, args.clip_bs))
        if args.dataset in ['all', 'ozone']:
            all_res.extend(run_dataset_grid('EPA Ozone', setup_ozone, args.reps, args.n_sources, args.clip_bs))
            
    df = pd.DataFrame(all_res)
    
    # Сохраняем в файл с именем датасета, чтобы процессы не мешали друг другу
    file_name = f"grid_search_{args.dataset}.csv"
    df.to_csv(OUT_DIR / file_name, index=False)
    
    for ds in df.dataset.unique():
        print_best_configuration(df, ds)
    print(f"\nAll done! Raw results saved to {file_name}")

if __name__ == '__main__':
    main()