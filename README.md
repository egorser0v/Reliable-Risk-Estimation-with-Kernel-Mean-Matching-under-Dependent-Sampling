
# Reliable Risk Estimation with Kernel Mean Matching under Dependent Sampling

This directory contains the complete source code for reproducing the synthetic experimental results, ablation studies, and real-data benchmarks reported in the paper:

> **Reliable Risk Estimation with Kernel Mean Matching under Dependent Sampling**  
> *Anonymous Authors*  
> *Under review at AISTATS 2027*

--- 

## 1. Directory Structure

```text
.
├── core.py                   # Core mathematical primitives, KMM solver, neff formulas, metrics
├── run_main_benchmark.py     # Reproduces Table 2 (Main Text) and Table 7 (Appendix G.2)
├── run_ablations.py          # Reproduces Tables 8, 9, 10, 11 (Appendix G.5: n, ell_x, h, B)
├── run_kernel_ablation.py    # Reproduces Table 12 (Appendix G.6: Matérn roughness stress test)
├── run_nonstationarity.py    # Reproduces Table 13 (Appendix G.7: Non-stationary Gibbs kernel)
├── experiments_real/         # Real-data experiments directory
│   ├── kmm_three_domain_benchmark.py # Three-domain real-data benchmark on PM2.5, NOAA, and TNBC
│   ├── run_paper_benchmark.py        # Grid-search benchmark for real data
│   └── verify_kmm_three_domain.py    # Numerical feasibility and test verification script
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation and reproduction guide
```

---

## 2. Installation & Prerequisites

The code requires **Python 3.8+** and standard scientific computing packages.

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## 3. Mapping of Scripts to Paper Tables

| Table | Script | Description |
| :--- | :--- | :--- |
| **Table 2** / **Table 7** | `python run_main_benchmark.py` | Full benchmark on Target Mean ($Y$) and Target Risk ($L$) across all baseline methods. |
| **Table 8** | `python run_ablations.py --param n` | Sample-size scaling ($n \in \{32, 48, 64, 96, 128, 192, 256\}$). |
| **Table 9** | `python run_ablations.py --param ell_x` | Input dependence ablation ($\ell_x \in \{1.5, 2.0, 3.0, 4.0, 5.0, 6.0\}$). |
| **Table 10** | `python run_ablations.py --param h` | Functional smoothness ablation ($h \in \{1.00, 1.25, 1.50, 1.75, 2.00, 3.00\}$). |
| **Table 11** | `python run_ablations.py --param clip_B` | Weight clipping bound ablation ($B \in \{5.0, 7.5, 10.0, 15.0, 20.0, 30.0\}$). |
| **Table 12** | `python run_kernel_ablation.py` | Roughness stress-test across Matérn-$\nu$ processes ($\nu \in \{\infty, 2.5, 1.5, 0.5\}$). |
| **Table 13** | `python run_nonstationarity.py` | Robustness to non-stationary covariates with time-varying $\ell_x(t)$ (Gibbs kernel). |
| **Real Data** | `python experiments_real/kmm_three_domain_benchmark.py` | Real-data benchmark across PM2.5, NOAA, and TNBC datasets. |

---

## 4. Replication Commands (Synthetic)

All scripts are pre-configured with the exact random seeds, ground-truth integration sample sizes, and calibration hyperparameters described in Appendix G.1.

### 4.1 Main Benchmark (Table 2 & Table 7)
Evaluates 8 estimators: `Unweighted`, `Nominal n`, `Block-Bootstrap`, `Input neff`, `GP neff (Ours)`, `Strict Thm 1`, `Oracle IW`, and `Oracle i.i.d.` on both estimands:
```bash
python run_main_benchmark.py --n_reps 300 --seed 42
```

### 4.2 Hyperparameter Ablations (Tables 8, 9, 10, 11)
To run all four ablation sweeps sequentially:
```bash
python run_ablations.py --param all --n_reps 300 --seed 42
```
To run an individual parameter ablation:
```bash
python run_ablations.py --param n        # Table 8 (Sample size)
python run_ablations.py --param ell_x    # Table 9 (Input dependence)
python run_ablations.py --param h        # Table 10 (Functional smoothness)
python run_ablations.py --param clip_B   # Table 11 (Clipping bound)
```

### 4.3 Kernel Roughness Stress Test (Table 12)
Evaluates KMM when inputs and responses follow rough Matérn processes while the GP $n_{\mathrm{eff}}$ formula deliberately assumes an RBF geometry:
```bash
python run_kernel_ablation.py --n_reps 300 --seed 20260702
```

### 4.4 Non-Stationary Covariates (Table 13)
Evaluates KMM over non-stationary Gibbs trajectories under `Stationary`, `Abrupt Switch`, and `Smooth Wave` regimes:
```bash
python run_nonstationarity.py --n_reps 300 --seed 202611
```

*Tip: For a quick verification run, append `--n_reps 50` to any script.*

---

## 5. Real-Data Experiments

The real-data evaluation consists of predicting two estimands (bounded target response mean and fixed-model target risk) across domains: **PM2.5**, **NOAA**, and **TNBC**. These are empirical stress tests that use real-world dependent sampling distributions. Target labels are exclusively used for scoring metrics.

### 5.1 Three-Domain Benchmark
To run the evaluation across all three datasets:
```bash
python experiments_real/kmm_three_domain_benchmark.py --dataset all
```
*Note: Ensure datasets are downloaded to `./data/`. For NOAA, set `KMM_NOAA_ROOT` if rebuilding the cache. Outputs are stored in `results/kmm_three_domain` and will not overwrite historical runs.*

You can also run a specific dataset or limit the number of repetitions (params specified inside the file):
```bash
python experiments_real/kmm_three_domain_benchmark.py --dataset PM25 --limit 10
```

### 5.2 Real-Data Grid Search & Baselines
The repository also includes parallelized grid-search benchmarks to find optimal $n_{\mathrm{source}}$ and clipping bounds dynamically:
```bash
python experiments_real/run_paper_benchmark.py --dataset all --n_jobs 10
```

### 5.3 Test Verification
To test numerical feasibility, check shared kernel validity, and verify absence of target-label leakage:
```bash
python experiments_real/verify_kmm_three_domain.py
```

---

## 6. Mathematical Formulations Implemented in `core.py`

1. **Kernel Mean Matching QP (Eq. 2 & Eq. 56):**
   $$\min_{\mathbf{w} \in [0, B]^n} \frac{1}{2} \mathbf{w}^\top (\mathbf{K}_{ss} + \lambda \mathbf{I}) \mathbf{w} - \boldsymbol{\kappa}^\top \mathbf{w} \quad \text{s.t.} \quad \sum_{i=1}^n w_i = n$$
   solved using the interior-point solver in `cvxopt`.

2. **GP-Induced Effective Sample Size (Eq. 5 & Eq. 6):**
   $$n_{\mathrm{eff}}^{\mathrm{GP}}(h) = \frac{n}{1 + 2 \sum_{k=1}^{n-1} \left(1 - \frac{k}{n}\right) \bar{\rho}_u(k)},$$
   where $\bar{\rho}_u(k) = \frac{(1 + a_h(1 - r_k))^{-d/2} - (1 + a_h)^{-d/2}}{1 - (1 + a_h)^{-d/2}}$ and $a_h = \frac{2\sigma_x^2}{h^2}$.

3. **GP-Calibrated Practical Confidence Radius (Eq. 7):**
   $$\hat{r}_{\mathrm{eff}} = 1.96 \sqrt{\frac{\widehat{\mathrm{Var}}\left[\hat{w}_i (Y_i^{\mathrm{tr}} - \hat{R}_{\mathrm{KMM}}(g))\right]}{n_{\mathrm{eff}}^{\mathrm{GP}}(h_\ell)}}.$$

4. **Winkler Interval Score (Eq. 57):**
   $$\mathrm{WS} = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U), \quad \alpha = 0.05.$$

5. **Statistical Validity Criterion:**
   For nominal $95\%$ intervals over $N = 300$ repetitions, empirical coverage is considered statistically valid if it meets or exceeds the binomial two-sigma lower bound:
   $$0.95 - 1.96 \sqrt{\frac{0.95 \times 0.05}{300}} \approx 0.925.$$
   Any coverage falling below $0.925$ is automatically flagged with an asterisk (`*`).
```