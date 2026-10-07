# -*- coding: utf-8 -*-


# ============================================================
# CALIFORNIA HOUSING: KMM OVERLAP DIAGNOSTIC
# ============================================================

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 20260624

N_SPLITS = 40
N_SOURCE = 256
N_TARGET = 256

TARGET_RADIUS = 1.15
SOURCE_BUFFER = 0.35

KMM_BANDWIDTH = 2.0
KMM_B = 10.0
KMM_RIDGE = 1e-4

BLOCK_LON_BINS = 5
BLOCK_LAT_BINS = 5

N_BOOTSTRAP = 300
ALPHA = 0.05

OUTPUT_DIR = Path("/content/california_housing_overlap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. RBF KERNEL
# ============================================================

def rbf_kernel(x, y, bandwidth):
    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )

    return np.exp(
        -d2 / (2.0 * bandwidth**2)
    )


# ============================================================
# 3. CONSTRAINED KMM
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth=2.0,
    B=10.0,
    ridge=1e-4,
):
    """
    Solve

        min_w  1/2 w^T K w - kappa^T w

    subject to

        0 <= w_i <= B
        sum_i w_i = n_source

    Thus mean(w) = 1.
    """

    n = x_source.shape[0]
    m = x_target.shape[0]

    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    K = 0.5 * (K + K.T)
    K = K + ridge * np.eye(n)

    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    kappa = (
        n / m
    ) * np.sum(
        K_st,
        axis=1
    )

    def objective(w):
        return (
            0.5 * w @ K @ w
            - kappa @ w
        )

    def gradient(w):
        return K @ w - kappa

    constraint = {
        "type": "eq",
        "fun": lambda w: np.sum(w) - n,
        "jac": lambda w: np.ones_like(w),
    }

    bounds = [
        (0.0, B)
        for _ in range(n)
    ]

    w0 = np.ones(
        n,
        dtype=float
    )

    result = minimize(
        objective,
        w0,
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "KMM optimization failed: "
            + result.message
        )

    w = np.asarray(
        result.x,
        dtype=float
    )

    # Numerical checks
    if np.min(w) < -1e-5:
        raise RuntimeError(
            f"Negative KMM weight: {np.min(w)}"
        )

    if np.max(w) > B + 1e-4:
        raise RuntimeError(
            f"KMM weight exceeded B: {np.max(w)}"
        )

    if abs(np.sum(w) - n) > 1e-3:
        raise RuntimeError(
            "KMM weight sum constraint violated."
        )

    return w


# ============================================================
# 4. OVERLAP DIAGNOSTICS
# ============================================================

def overlap_diagnostics(weights, B):
    """
    Weight ESS:

        ESS_w = (sum w)^2 / sum(w^2)

    Lower ESS means stronger weight concentration.
    """

    w = np.asarray(
        weights,
        dtype=float
    )

    denominator = np.sum(
        w**2
    )

    if denominator <= 1e-14:
        ess = np.nan
    else:
        ess = (
            np.sum(w)**2
            / denominator
        )

    return {
        "weight_ess": float(ess),

        "ess_fraction": float(
            ess / len(w)
        ),

        "max_weight": float(
            np.max(w)
        ),

        "weight_sd": float(
            np.std(w)
        ),

        "frac_at_B": float(
            np.mean(
                w >= B - 1e-4
            )
        ),

        "frac_near_zero": float(
            np.mean(
                w <= 1e-4
            )
        ),
    }


# ============================================================
# 5. SPATIAL BLOCKS
# ============================================================

def spatial_block_ids(
    coords,
    n_lon_bins,
    n_lat_bins,
):
    lon = coords[:, 0]
    lat = coords[:, 1]

    lon_edges = np.quantile(
        lon,
        np.linspace(
            0,
            1,
            n_lon_bins + 1
        )
    )

    lat_edges = np.quantile(
        lat,
        np.linspace(
            0,
            1,
            n_lat_bins + 1
        )
    )

    # avoid edge numerical issues
    lon_edges[0] -= 1e-9
    lon_edges[-1] += 1e-9

    lat_edges[0] -= 1e-9
    lat_edges[-1] += 1e-9

    lon_id = np.digitize(
        lon,
        lon_edges[1:-1]
    )

    lat_id = np.digitize(
        lat,
        lat_edges[1:-1]
    )

    return (
        lon_id * n_lat_bins
        + lat_id
    )


# ============================================================
# 6. SPATIAL BLOCK BOOTSTRAP
# ============================================================

def block_bootstrap_half_width(
    rng,
    values,
    weights,
    block_ids,
    n_boot=300,
    alpha=0.05,
):
    """
    Resample spatial blocks.

    The point estimator remains

        mu_hat = (1/n) sum_i w_i y_i

    throughout.
    """

    unique_blocks = np.unique(
        block_ids
    )

    block_to_idx = {
        block: np.flatnonzero(
            block_ids == block
        )
        for block in unique_blocks
    }

    estimates = np.empty(
        n_boot,
        dtype=float
    )

    n = len(values)

    for b in range(n_boot):

        selected_blocks = rng.choice(
            unique_blocks,
            size=len(unique_blocks),
            replace=True
        )

        idx = np.concatenate([
            block_to_idx[block]
            for block
            in selected_blocks
        ])

        estimates[b] = (
            np.sum(
                weights[idx]
                * values[idx]
            )
            / n
        )

    lo, hi = np.quantile(
        estimates,
        [
            alpha / 2,
            1 - alpha / 2
        ]
    )

    half_width = (
        hi - lo
    ) / 2

    return (
        float(half_width),
        int(len(unique_blocks))
    )


# ============================================================
# 7. LOAD CALIFORNIA HOUSING
# ============================================================

print("\nLoading California Housing...")

data = fetch_california_housing(
    as_frame=True
)

df = data.frame.copy()

raw_y = df[
    "MedHouseVal"
].to_numpy(float)

# bounded outcome in [0,1]
y = (
    raw_y - raw_y.min()
) / (
    raw_y.max()
    - raw_y.min()
)

df["Target01"] = y

feature_cols = list(
    data.feature_names
)

features = df[
    feature_cols
].to_numpy(float)

coords = df[
    [
        "Longitude",
        "Latitude"
    ]
].to_numpy(float)

print(
    "Rows:",
    len(df)
)

print(
    "Features:",
    len(feature_cols)
)


# ============================================================
# 8. SCALE COVARIATES
# ============================================================

# We preserve the scaling convention of the original
# California experiment for comparability.
scaler = StandardScaler().fit(
    features
)

features_scaled = scaler.transform(
    features
)


# ============================================================
# 9. GENERATE CANDIDATE TARGET CENTERS
# ============================================================

candidate_indices = rng.choice(
    len(df),
    size=min(
        N_SPLITS * 10,
        len(df)
    ),
    replace=False
)

candidate_centers = coords[
    candidate_indices
]


# ============================================================
# 10. RUN EXPERIMENT
# ============================================================

rows = []

split_id = 0

print(
    "\nRunning spatial splits...\n"
)

for center in candidate_centers:

    # geographic distance in coordinate degrees
    dist = np.sqrt(
        np.sum(
            (
                coords
                - center[None, :]
            ) ** 2,
            axis=1
        )
    )

    # Target = local spatial window
    target_pool = np.flatnonzero(
        dist <= TARGET_RADIUS
    )

    # Source must be separated from target
    source_pool = np.flatnonzero(
        dist >= (
            TARGET_RADIUS
            + SOURCE_BUFFER
        )
    )

    if (
        len(target_pool) < N_TARGET
        or
        len(source_pool) < N_SOURCE
    ):
        continue

    target_idx = rng.choice(
        target_pool,
        size=N_TARGET,
        replace=False
    )

    # Prefer source observations from a buffered ring
    ring = np.flatnonzero(
        (
            dist
            >= TARGET_RADIUS
            + SOURCE_BUFFER
        )
        &
        (
            dist
            <= TARGET_RADIUS
            + 2.75
        )
    )

    if len(ring) >= N_SOURCE:
        source_candidates = ring
    else:
        source_candidates = source_pool

    source_idx = rng.choice(
        source_candidates,
        size=N_SOURCE,
        replace=False
    )

    # --------------------------------------------------------
    # Source / target data
    # --------------------------------------------------------

    x_s = features_scaled[
        source_idx
    ]

    x_t = features_scaled[
        target_idx
    ]

    y_s = y[
        source_idx
    ]

    y_t = y[
        target_idx
    ]

    coords_s = coords[
        source_idx
    ]

    # observed finite target-window mean
    target_truth = float(
        np.mean(y_t)
    )

    # --------------------------------------------------------
    # KMM
    # --------------------------------------------------------

    try:
        w_kmm = constrained_kmm_weights(
            x_s,
            x_t,
            bandwidth=KMM_BANDWIDTH,
            B=KMM_B,
            ridge=KMM_RIDGE,
        )

    except Exception as exc:
        print(
            f"Skipping candidate: "
            f"KMM failed ({exc})"
        )
        continue

    w_un = np.ones(
        N_SOURCE
    )

    # --------------------------------------------------------
    # Overlap diagnostics
    # --------------------------------------------------------

    diag = overlap_diagnostics(
        w_kmm,
        KMM_B
    )

    # --------------------------------------------------------
    # Point estimates
    # --------------------------------------------------------

    estimate_un = float(
        np.mean(y_s)
    )

    estimate_kmm = float(
        np.mean(
            w_kmm * y_s
        )
    )

    error_un = (
        estimate_un
        - target_truth
    )

    error_kmm = (
        estimate_kmm
        - target_truth
    )

    # --------------------------------------------------------
    # iid intervals
    # --------------------------------------------------------

    z = (
        w_kmm
        * y_s
    )

    iid_hw_un = (
        1.959963984540054
        * math.sqrt(
            np.var(
                y_s,
                ddof=1
            )
            / N_SOURCE
        )
    )

    iid_hw_kmm = (
        1.959963984540054
        * math.sqrt(
            np.var(
                z,
                ddof=1
            )
            / N_SOURCE
        )
    )

    # --------------------------------------------------------
    # Spatial block bootstrap
    # --------------------------------------------------------

    block_ids = spatial_block_ids(
        coords_s,
        BLOCK_LON_BINS,
        BLOCK_LAT_BINS
    )

    block_hw_un, n_blocks = (
        block_bootstrap_half_width(
            rng,
            y_s,
            w_un,
            block_ids,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA,
        )
    )

    block_hw_kmm, _ = (
        block_bootstrap_half_width(
            rng,
            y_s,
            w_kmm,
            block_ids,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA,
        )
    )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    un_iid_covered = int(
        abs(error_un)
        <= iid_hw_un
    )

    un_block_covered = int(
        abs(error_un)
        <= block_hw_un
    )

    kmm_iid_covered = int(
        abs(error_kmm)
        <= iid_hw_kmm
    )

    kmm_block_covered = int(
        abs(error_kmm)
        <= block_hw_kmm
    )

    # --------------------------------------------------------
    # Spatial separation diagnostics
    # --------------------------------------------------------

    source_center = np.mean(
        coords_s,
        axis=0
    )

    target_coords = coords[
        target_idx
    ]

    target_center = np.mean(
        target_coords,
        axis=0
    )

    centroid_distance = float(
        np.linalg.norm(
            source_center
            - target_center
        )
    )

    min_source_target_distance = float(
        np.min(
            pairwise_distances(
                coords_s,
                target_coords
            )
        )
    )

    # --------------------------------------------------------
    # Save one row per split
    # --------------------------------------------------------

    rows.append({
        "split": split_id,

        "target_center_lon":
            float(center[0]),

        "target_center_lat":
            float(center[1]),

        "centroid_distance":
            centroid_distance,

        "min_source_target_distance":
            min_source_target_distance,

        "target_truth":
            target_truth,

        # point estimates
        "unweighted_estimate":
            estimate_un,

        "kmm_estimate":
            estimate_kmm,

        "unweighted_error":
            error_un,

        "kmm_error":
            error_kmm,

        "unweighted_abs_error":
            abs(error_un),

        "kmm_abs_error":
            abs(error_kmm),

        # iid uncertainty
        "unweighted_iid_half_width":
            iid_hw_un,

        "kmm_iid_half_width":
            iid_hw_kmm,

        "unweighted_iid_covered":
            un_iid_covered,

        "kmm_iid_covered":
            kmm_iid_covered,

        # block uncertainty
        "unweighted_block_half_width":
            block_hw_un,

        "kmm_block_half_width":
            block_hw_kmm,

        "unweighted_block_covered":
            un_block_covered,

        "kmm_block_covered":
            kmm_block_covered,

        "n_spatial_blocks":
            n_blocks,

        # overlap diagnostics
        "weight_ess":
            diag["weight_ess"],

        "weight_ess_fraction":
            diag["ess_fraction"],

        "max_weight":
            diag["max_weight"],

        "weight_sd":
            diag["weight_sd"],

        "frac_at_B":
            diag["frac_at_B"],

        "frac_near_zero":
            diag["frac_near_zero"],
    })

    print(
        f"split {split_id:02d} | "
        f"ESS={diag['weight_ess']:6.1f}/{N_SOURCE} | "
        f"max w={diag['max_weight']:5.2f} | "
        f"at B={diag['frac_at_B']:.3f} | "
        f"KMM error={abs(error_kmm):.4f}"
    )

    split_id += 1

    if split_id >= N_SPLITS:
        break


# ============================================================
# 11. CHECK NUMBER OF SPLITS
# ============================================================

if len(rows) == 0:
    raise RuntimeError(
        "No valid spatial splits were produced."
    )

if len(rows) < N_SPLITS:
    print(
        f"\nWARNING: only {len(rows)} "
        f"valid splits were produced."
    )


# ============================================================
# 12. PER-SPLIT DATAFRAME
# ============================================================

replicates = pd.DataFrame(
    rows
)

replicates_path = (
    OUTPUT_DIR
    / "real_spatial_replicates.csv"
)

replicates.to_csv(
    replicates_path,
    index=False
)


# ============================================================
# 13. SUMMARY TABLE
# ============================================================

summary_rows = []


def add_summary(
    method,
    estimate_col,
    error_col,
    iid_hw_col,
    iid_cov_col,
    block_hw_col,
    block_cov_col,
):
    summary_rows.append({
        "method": method,

        "n_splits":
            len(replicates),

        "bias":
            replicates[
                error_col
            ].mean(),

        "mae":
            replicates[
                error_col
            ].abs().mean(),

        "rmse":
            np.sqrt(
                np.mean(
                    replicates[
                        error_col
                    ] ** 2
                )
            ),

        "iid_coverage":
            replicates[
                iid_cov_col
            ].mean(),

        "block_coverage":
            replicates[
                block_cov_col
            ].mean(),

        "iid_half_width":
            replicates[
                iid_hw_col
            ].mean(),

        "block_half_width":
            replicates[
                block_hw_col
            ].mean(),

        "mean_weight_ess":
            (
                N_SOURCE
                if method == "Unweighted"
                else
                replicates[
                    "weight_ess"
                ].mean()
            ),

        "median_weight_ess":
            (
                N_SOURCE
                if method == "Unweighted"
                else
                replicates[
                    "weight_ess"
                ].median()
            ),

        "mean_ess_fraction":
            (
                1.0
                if method == "Unweighted"
                else
                replicates[
                    "weight_ess_fraction"
                ].mean()
            ),

        "mean_max_weight":
            (
                1.0
                if method == "Unweighted"
                else
                replicates[
                    "max_weight"
                ].mean()
            ),

        "mean_frac_at_B":
            (
                0.0
                if method == "Unweighted"
                else
                replicates[
                    "frac_at_B"
                ].mean()
            ),

        "mean_frac_near_zero":
            (
                0.0
                if method == "Unweighted"
                else
                replicates[
                    "frac_near_zero"
                ].mean()
            ),
    })


add_summary(
    method="Unweighted",

    estimate_col=
        "unweighted_estimate",

    error_col=
        "unweighted_error",

    iid_hw_col=
        "unweighted_iid_half_width",

    iid_cov_col=
        "unweighted_iid_covered",

    block_hw_col=
        "unweighted_block_half_width",

    block_cov_col=
        "unweighted_block_covered",
)


add_summary(
    method="KMM",

    estimate_col=
        "kmm_estimate",

    error_col=
        "kmm_error",

    iid_hw_col=
        "kmm_iid_half_width",

    iid_cov_col=
        "kmm_iid_covered",

    block_hw_col=
        "kmm_block_half_width",

    block_cov_col=
        "kmm_block_covered",
)


summary = pd.DataFrame(
    summary_rows
)

summary_path = (
    OUTPUT_DIR
    / "real_spatial_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 14. CORRELATIONS:
#     DO OVERLAP DIAGNOSTICS TRACK ERROR?
# ============================================================

diagnostic_correlations = pd.DataFrame({
    "diagnostic": [
        "weight_ess",
        "weight_ess_fraction",
        "max_weight",
        "frac_at_B",
        "frac_near_zero",
        "centroid_distance",
    ],

    "corr_with_KMM_abs_error": [
        replicates[
            "weight_ess"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),

        replicates[
            "weight_ess_fraction"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),

        replicates[
            "max_weight"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),

        replicates[
            "frac_at_B"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),

        replicates[
            "frac_near_zero"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),

        replicates[
            "centroid_distance"
        ].corr(
            replicates[
                "kmm_abs_error"
            ]
        ),
    ],
})

diagnostic_correlations.to_csv(
    OUTPUT_DIR
    / "overlap_error_correlations.csv",
    index=False
)


# ============================================================
# 15. FIGURE: ESS VS ERROR
# ============================================================

plt.figure(
    figsize=(6, 4.5)
)

plt.scatter(
    replicates[
        "weight_ess"
    ],
    replicates[
        "kmm_abs_error"
    ],
    alpha=0.75
)

plt.xlabel(
    "KMM weight ESS"
)

plt.ylabel(
    "Absolute target-mean error"
)

plt.title(
    "California Housing: overlap vs error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "weight_ess_vs_error.png",
    dpi=180
)

plt.show()


# ============================================================
# 16. FIGURE: MAX WEIGHT VS ERROR
# ============================================================

plt.figure(
    figsize=(6, 4.5)
)

plt.scatter(
    replicates[
        "max_weight"
    ],
    replicates[
        "kmm_abs_error"
    ],
    alpha=0.75
)

plt.xlabel(
    "Maximum KMM weight"
)

plt.ylabel(
    "Absolute target-mean error"
)

plt.title(
    "California Housing: weight concentration vs error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "max_weight_vs_error.png",
    dpi=180
)

plt.show()


# ============================================================
# 17. FIGURE: iid VS BLOCK COVERAGE
# ============================================================

coverage_plot = pd.DataFrame({
    "method": [
        "Unweighted iid",
        "Unweighted block",
        "KMM iid",
        "KMM block",
    ],

    "coverage": [
        replicates[
            "unweighted_iid_covered"
        ].mean(),

        replicates[
            "unweighted_block_covered"
        ].mean(),

        replicates[
            "kmm_iid_covered"
        ].mean(),

        replicates[
            "kmm_block_covered"
        ].mean(),
    ],
})

plt.figure(
    figsize=(7, 4.5)
)

plt.bar(
    coverage_plot[
        "method"
    ],
    coverage_plot[
        "coverage"
    ]
)

plt.axhline(
    0.95,
    linestyle="--"
)

plt.ylim(
    0,
    1.05
)

plt.ylabel(
    "Coverage"
)

plt.xticks(
    rotation=20
)

plt.title(
    "California Housing interval coverage"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "coverage.png",
    dpi=180
)

plt.show()


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

display(
    summary.round(4)
)

print("\n")
print("=" * 80)
print("OVERLAP DIAGNOSTICS VS KMM ERROR")
print("=" * 80)

display(
    diagnostic_correlations.round(4)
)

print("\n")
print("=" * 80)
print("FIRST 10 SPLITS")
print("=" * 80)

display(
    replicates[
        [
            "split",
            "kmm_abs_error",
            "weight_ess",
            "weight_ess_fraction",
            "max_weight",
            "frac_at_B",
            "frac_near_zero",
            "kmm_iid_covered",
            "kmm_block_covered",
        ]
    ].head(10).round(4)
)


# ============================================================
# 19. VERIFY FILES
# ============================================================

print("\nSaved files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        " -",
        file
    )


print("\nDONE.")
print(
    "Please send these two files:"
)

print(
    OUTPUT_DIR
    / "real_spatial_summary.csv"
)

print(
    OUTPUT_DIR
    / "real_spatial_replicates.csv"
)

# ============================================================
# SEMI-SYNTHETIC CALIFORNIA HOUSING
# Controlled overlap stress test for KMM
#
# 

# ============================================================

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 20261005

N_SOURCE = 256
N_TARGET = 256

# Number of independent repetitions at each overlap level
N_REPS = 25

# Larger buffer = source pushed farther away from target.
# These are geographic degrees in the California coordinates.
OVERLAP_LEVELS = {
    "good": 0.00,
    "mild": 0.25,
    "moderate": 0.60,
    "poor": 1.00,
    "severe": 1.50,
}

TARGET_RADIUS = 1.00

# We use a finite outer ring so increasing BUFFER actually
# changes the source distribution rather than just enlarging
# the eligible source set.
SOURCE_RING_WIDTH = 1.25

KMM_BANDWIDTH = 2.0
KMM_B = 10.0
KMM_RIDGE = 1e-4

BLOCK_LON_BINS = 5
BLOCK_LAT_BINS = 5

N_BOOTSTRAP = 250
ALPHA = 0.05


NOISE_SD = 0.08

OUTPUT_DIR = Path(
    "/content/california_semisynthetic_overlap"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

master_rng = np.random.default_rng(
    SEED
)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. RBF KERNEL
# ============================================================

def rbf_kernel(x, y, bandwidth):

    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )

    return np.exp(
        -d2 / (
            2.0 * bandwidth**2
        )
    )


# ============================================================
# 3. CONSTRAINED KMM
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth=2.0,
    B=10.0,
    ridge=1e-4,
):

    n = x_source.shape[0]
    m = x_target.shape[0]

    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    K = (
        0.5 * (K + K.T)
        + ridge * np.eye(n)
    )

    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    kappa = (
        n / m
    ) * np.sum(
        K_st,
        axis=1
    )

    def objective(w):
        return (
            0.5 * w @ K @ w
            - kappa @ w
        )

    def gradient(w):
        return (
            K @ w
            - kappa
        )

    constraint = {
        "type": "eq",
        "fun":
            lambda w:
            np.sum(w) - n,

        "jac":
            lambda w:
            np.ones_like(w),
    }

    bounds = [
        (0.0, B)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        np.ones(n),
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    w = np.asarray(
        result.x,
        dtype=float
    )

    if np.max(w) > B + 1e-4:
        raise RuntimeError(
            "KMM upper bound violated"
        )

    if np.min(w) < -1e-5:
        raise RuntimeError(
            "Negative KMM weight"
        )

    if abs(
        np.sum(w) - n
    ) > 1e-3:
        raise RuntimeError(
            "KMM sum constraint violated"
        )

    return w


# ============================================================
# 4. WEIGHT / OVERLAP DIAGNOSTICS
# ============================================================

def weight_diagnostics(w, B):

    w = np.asarray(
        w,
        dtype=float
    )

    ess = (
        np.sum(w)**2
        / np.sum(w**2)
    )

    return {
        "weight_ess":
            float(ess),

        "ess_fraction":
            float(
                ess / len(w)
            ),

        "max_weight":
            float(
                np.max(w)
            ),

        "weight_sd":
            float(
                np.std(w)
            ),

        "frac_at_B":
            float(
                np.mean(
                    w >= B - 1e-4
                )
            ),

        "frac_near_zero":
            float(
                np.mean(
                    w <= 1e-4
                )
            ),
    }


# ============================================================
# 5. SIMPLE FEATURE-MMD DIAGNOSTIC
# ============================================================

def empirical_mmd2(
    x_source,
    x_target,
    bandwidth=2.0,
):

    Kss = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    Ktt = rbf_kernel(
        x_target,
        x_target,
        bandwidth
    )

    Kst = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    return float(
        np.mean(Kss)
        + np.mean(Ktt)
        - 2.0 * np.mean(Kst)
    )


# ============================================================
# 6. SPATIAL BLOCK IDS
# ============================================================

def spatial_block_ids(
    coords,
    n_lon_bins,
    n_lat_bins,
):

    lon = coords[:, 0]
    lat = coords[:, 1]

    lon_edges = np.quantile(
        lon,
        np.linspace(
            0,
            1,
            n_lon_bins + 1
        )
    )

    lat_edges = np.quantile(
        lat,
        np.linspace(
            0,
            1,
            n_lat_bins + 1
        )
    )

    lon_edges[0] -= 1e-9
    lon_edges[-1] += 1e-9

    lat_edges[0] -= 1e-9
    lat_edges[-1] += 1e-9

    lon_id = np.digitize(
        lon,
        lon_edges[1:-1]
    )

    lat_id = np.digitize(
        lat,
        lat_edges[1:-1]
    )

    return (
        lon_id * n_lat_bins
        + lat_id
    )


# ============================================================
# 7. BLOCK BOOTSTRAP
# ============================================================

def block_bootstrap_half_width(
    rng,
    values,
    weights,
    block_ids,
    n_boot=250,
    alpha=0.05,
):

    unique_blocks = np.unique(
        block_ids
    )

    block_to_idx = {
        block:
        np.flatnonzero(
            block_ids == block
        )
        for block
        in unique_blocks
    }

    n = len(values)

    estimates = np.empty(
        n_boot
    )

    for b in range(n_boot):

        chosen = rng.choice(
            unique_blocks,
            size=len(unique_blocks),
            replace=True
        )

        idx = np.concatenate([
            block_to_idx[c]
            for c in chosen
        ])

        estimates[b] = (
            np.sum(
                weights[idx]
                * values[idx]
            )
            / n
        )

    lo, hi = np.quantile(
        estimates,
        [
            alpha / 2,
            1 - alpha / 2
        ]
    )

    return float(
        (hi - lo) / 2
    )


# ============================================================
# 8. LOAD REAL CALIFORNIA X + COORDINATES
# ============================================================

print("\nLoading California Housing...")

data = fetch_california_housing(
    as_frame=True
)

df = data.frame.copy()

feature_cols = list(
    data.feature_names
)

X_raw = df[
    feature_cols
].to_numpy(float)

coords = df[
    [
        "Longitude",
        "Latitude"
    ]
].to_numpy(float)

print(
    "Rows:",
    len(df)
)

print(
    "Features:",
    feature_cols
)


# ============================================================
# 9. STANDARDIZE X
#
# Target covariates are allowed in KMM.
# No target LABELS are used for fitting.
# ============================================================

scaler = StandardScaler().fit(
    X_raw
)

X = scaler.transform(
    X_raw
)


# ============================================================
# 10. SEMI-SYNTHETIC OUTCOME FUNCTION
#
# Same f(X) for every location and every overlap level.
#
# Coordinates are already among California features:
# Latitude and Longitude.
#
# This nonlinear function deliberately depends on several
# real covariates but is fixed globally.
# ============================================================

feature_index = {
    name: i
    for i, name
    in enumerate(feature_cols)
}

i_income = feature_index[
    "MedInc"
]

i_age = feature_index[
    "HouseAge"
]

i_rooms = feature_index[
    "AveRooms"
]

i_occup = feature_index[
    "AveOccup"
]

i_lat = feature_index[
    "Latitude"
]

i_lon = feature_index[
    "Longitude"
]


def outcome_mean_function(X):

    income = X[:, i_income]
    age = X[:, i_age]
    rooms = X[:, i_rooms]
    occup = X[:, i_occup]
    lat = X[:, i_lat]
    lon = X[:, i_lon]

    eta = (
        0.80 * income
        + 0.30 * np.sin(
            1.2 * age
        )
        + 0.25 * np.tanh(
            rooms
        )
        - 0.18 * np.tanh(
            occup
        )
        + 0.30 * np.sin(
            0.8 * lat
        )
        + 0.25 * np.cos(
            0.8 * lon
        )
        + 0.15 * income * rooms
    )

    # logistic map:
    # conditional mean is strictly in (0,1)
    mu = (
        1.0
        / (
            1.0
            + np.exp(-eta)
        )
    )

    return mu


MU = outcome_mean_function(
    X
)

print(
    "\nSemi-synthetic conditional mean:"
)

print(
    "min =",
    round(
        float(MU.min()),
        4
    ),
    "| mean =",
    round(
        float(MU.mean()),
        4
    ),
    "| max =",
    round(
        float(MU.max()),
        4
    )
)


# ============================================================
# 11. SPATIALLY CORRELATED NOISE
#
# We generate noise as a smooth random Fourier field:
#
#   epsilon(s)
#       = sum_j a_j cos(k_j^T s + phi_j)
#
# Coefficients are redrawn for every repetition.
#
# The field has zero mean over repeated realizations,
# so:
#
#       E[Y | X] = f(X)
#
# while observations within a realization are spatially
# dependent.
# ============================================================

coords_std = StandardScaler().fit_transform(
    coords
)


def generate_spatial_noise(
    coords_standardized,
    rng,
    noise_sd=0.08,
    n_components=30,
):

    # lower frequencies -> smoother spatial dependence
    frequencies = rng.normal(
        loc=0.0,
        scale=0.70,
        size=(
            n_components,
            2
        )
    )

    phases = rng.uniform(
        0,
        2 * np.pi,
        size=n_components
    )

    amplitudes = rng.normal(
        0,
        1,
        size=n_components
    )

    projection = (
        coords_standardized
        @ frequencies.T
    )

    field = np.sum(
        amplitudes[None, :]
        * np.cos(
            projection
            + phases[None, :]
        ),
        axis=1
    )

    field = (
        field
        - np.mean(field)
    )

    sd = np.std(
        field
    )

    if sd > 1e-12:
        field = (
            field
            / sd
        )

    return (
        noise_sd
        * field
    )


# ============================================================
# 12. SELECT TARGET ANCHORS
#
# ============================================================

def pools_for_anchor(
    center,
    buffer,
):

    dist = np.sqrt(
        np.sum(
            (
                coords
                - center[None, :]
            )**2,
            axis=1
        )
    )

    target_pool = np.flatnonzero(
        dist <= TARGET_RADIUS
    )

    inner = (
        TARGET_RADIUS
        + buffer
    )

    outer = (
        inner
        + SOURCE_RING_WIDTH
    )

    source_pool = np.flatnonzero(
        (dist >= inner)
        &
        (dist <= outer)
    )

    return (
        target_pool,
        source_pool,
        dist
    )


candidate_idx = master_rng.choice(
    len(df),
    size=min(
        3000,
        len(df)
    ),
    replace=False
)

valid_anchor_indices = []

max_buffer = max(
    OVERLAP_LEVELS.values()
)

for idx in candidate_idx:

    center = coords[idx]

    target_pool, source_pool, _ = (
        pools_for_anchor(
            center,
            max_buffer
        )
    )

    if (
        len(target_pool) >= N_TARGET
        and
        len(source_pool) >= N_SOURCE
    ):
        valid_anchor_indices.append(
            idx
        )

print(
    "\nValid anchors supporting all levels:",
    len(valid_anchor_indices)
)

if len(valid_anchor_indices) < N_REPS:
    raise RuntimeError(
        "Not enough anchors support all overlap levels. "
        "Reduce N_REPS or the severe buffer."
    )

# Use the same anchor for all overlap levels within a replicate
anchor_indices = master_rng.choice(
    valid_anchor_indices,
    size=N_REPS,
    replace=False
)


# ============================================================
# 13. RUN STRESS TEST
# ============================================================

rows = []

print(
    "\nRunning semi-synthetic overlap stress test...\n"
)

for rep, anchor_idx in enumerate(
    anchor_indices
):

    center = coords[
        anchor_idx
    ]

    # One spatial-noise realization per replicate.
    # All overlap levels within the replicate see the same
    # underlying spatial field.
    rep_rng = np.random.default_rng(
        SEED + 10000 + rep
    )

    epsilon = generate_spatial_noise(
        coords_std,
        rep_rng,
        noise_sd=NOISE_SD
    )

    Y = (
        MU
        + epsilon
    )

    for level_order, (
        level_name,
        buffer
    ) in enumerate(
        OVERLAP_LEVELS.items()
    ):

        level_rng = np.random.default_rng(
            SEED
            + rep * 100
            + level_order
        )

        target_pool, source_pool, dist = (
            pools_for_anchor(
                center,
                buffer
            )
        )

        if (
            len(target_pool) < N_TARGET
            or
            len(source_pool) < N_SOURCE
        ):
            print(
                f"skip rep={rep}, "
                f"level={level_name}: "
                "insufficient observations"
            )
            continue

        # Same target sample across levels for this replicate
        target_rng = np.random.default_rng(
            SEED
            + 500000
            + rep
        )

        target_idx = target_rng.choice(
            target_pool,
            size=N_TARGET,
            replace=False
        )

        source_idx = level_rng.choice(
            source_pool,
            size=N_SOURCE,
            replace=False
        )

        Xs = X[
            source_idx
        ]

        Xt = X[
            target_idx
        ]

        ys = Y[
            source_idx
        ]

        coords_s = coords[
            source_idx
        ]

        coords_t = coords[
            target_idx
        ]

        # ----------------------------------------------------
        # KNOWN TARGET ESTIMAND
        #
        # We use the target conditional-mean average:
        #
        #   R_T = 1/m sum f(X_t)
        #
        # not the realized noisy target labels.
        #
        # Therefore target noise does not contaminate the
        # reference value.
        # ----------------------------------------------------

        truth = float(
            np.mean(
                MU[target_idx]
            )
        )

        # ----------------------------------------------------
        # KMM
        # ----------------------------------------------------

        try:
            w = constrained_kmm_weights(
                Xs,
                Xt,
                bandwidth=KMM_BANDWIDTH,
                B=KMM_B,
                ridge=KMM_RIDGE,
            )

        except Exception as exc:

            print(
                f"KMM failed: "
                f"rep={rep}, "
                f"level={level_name}: "
                f"{exc}"
            )

            continue

        diag = weight_diagnostics(
            w,
            KMM_B
        )

        # ----------------------------------------------------
        # SHIFT DIAGNOSTICS
        # ----------------------------------------------------

        mmd2 = empirical_mmd2(
            Xs,
            Xt,
            bandwidth=KMM_BANDWIDTH
        )

        centroid_distance = float(
            np.linalg.norm(
                np.mean(
                    coords_s,
                    axis=0
                )
                -
                np.mean(
                    coords_t,
                    axis=0
                )
            )
        )

        # ----------------------------------------------------
        # POINT ESTIMATORS
        # ----------------------------------------------------

        estimate_un = float(
            np.mean(ys)
        )

        estimate_kmm = float(
            np.mean(
                w * ys
            )
        )

        error_un = (
            estimate_un
            - truth
        )

        error_kmm = (
            estimate_kmm
            - truth
        )

        # ----------------------------------------------------
        # IID CI
        # ----------------------------------------------------

        z_un = ys

        z_kmm = (
            w * ys
        )

        iid_hw_un = (
            1.959963984540054
            * np.sqrt(
                np.var(
                    z_un,
                    ddof=1
                )
                / N_SOURCE
            )
        )

        iid_hw_kmm = (
            1.959963984540054
            * np.sqrt(
                np.var(
                    z_kmm,
                    ddof=1
                )
                / N_SOURCE
            )
        )

        # ----------------------------------------------------
        # SPATIAL BLOCK CI
        # ----------------------------------------------------

        block_ids = spatial_block_ids(
            coords_s,
            BLOCK_LON_BINS,
            BLOCK_LAT_BINS
        )

        block_hw_un = (
            block_bootstrap_half_width(
                level_rng,
                ys,
                np.ones(
                    N_SOURCE
                ),
                block_ids,
                n_boot=N_BOOTSTRAP,
                alpha=ALPHA,
            )
        )

        block_hw_kmm = (
            block_bootstrap_half_width(
                level_rng,
                ys,
                w,
                block_ids,
                n_boot=N_BOOTSTRAP,
                alpha=ALPHA,
            )
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        rows.append({
            "rep":
                rep,

            "level":
                level_name,

            "level_order":
                level_order,

            "buffer":
                buffer,

            "target_center_lon":
                float(center[0]),

            "target_center_lat":
                float(center[1]),

            "centroid_distance":
                centroid_distance,

            "feature_mmd2":
                mmd2,

            "truth":
                truth,

            "unweighted_estimate":
                estimate_un,

            "kmm_estimate":
                estimate_kmm,

            "unweighted_error":
                error_un,

            "kmm_error":
                error_kmm,

            "unweighted_abs_error":
                abs(error_un),

            "kmm_abs_error":
                abs(error_kmm),

            "unweighted_iid_half_width":
                iid_hw_un,

            "kmm_iid_half_width":
                iid_hw_kmm,

            "unweighted_block_half_width":
                block_hw_un,

            "kmm_block_half_width":
                block_hw_kmm,

            "unweighted_iid_covered":
                int(
                    abs(error_un)
                    <= iid_hw_un
                ),

            "kmm_iid_covered":
                int(
                    abs(error_kmm)
                    <= iid_hw_kmm
                ),

            "unweighted_block_covered":
                int(
                    abs(error_un)
                    <= block_hw_un
                ),

            "kmm_block_covered":
                int(
                    abs(error_kmm)
                    <= block_hw_kmm
                ),

            "weight_ess":
                diag[
                    "weight_ess"
                ],

            "weight_ess_fraction":
                diag[
                    "ess_fraction"
                ],

            "max_weight":
                diag[
                    "max_weight"
                ],

            "weight_sd":
                diag[
                    "weight_sd"
                ],

            "frac_at_B":
                diag[
                    "frac_at_B"
                ],

            "frac_near_zero":
                diag[
                    "frac_near_zero"
                ],
        })

        print(
            f"rep {rep:02d} | "
            f"{level_name:8s} | "
            f"buffer={buffer:4.2f} | "
            f"ESS={diag['weight_ess']:6.1f} | "
            f"MMD²={mmd2:.3f} | "
            f"KMM err={abs(error_kmm):.4f}"
        )


# ============================================================
# 14. RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)

if len(results) == 0:
    raise RuntimeError(
        "No valid results."
    )

results = results.sort_values(
    [
        "level_order",
        "rep"
    ]
).reset_index(
    drop=True
)

results.to_csv(
    OUTPUT_DIR
    / "semisynthetic_replicates.csv",
    index=False
)


# ============================================================
# 15. SUMMARY BY OVERLAP LEVEL
# ============================================================

summary = (
    results
    .groupby(
        [
            "level_order",
            "level",
            "buffer"
        ],
        as_index=False
    )
    .agg(
        n_reps=(
            "rep",
            "count"
        ),

        mean_centroid_distance=(
            "centroid_distance",
            "mean"
        ),

        mean_mmd2=(
            "feature_mmd2",
            "mean"
        ),

        mean_weight_ess=(
            "weight_ess",
            "mean"
        ),

        median_weight_ess=(
            "weight_ess",
            "median"
        ),

        mean_ess_fraction=(
            "weight_ess_fraction",
            "mean"
        ),

        mean_max_weight=(
            "max_weight",
            "mean"
        ),

        mean_frac_at_B=(
            "frac_at_B",
            "mean"
        ),

        mean_frac_near_zero=(
            "frac_near_zero",
            "mean"
        ),

        unweighted_bias=(
            "unweighted_error",
            "mean"
        ),

        kmm_bias=(
            "kmm_error",
            "mean"
        ),

        unweighted_mae=(
            "unweighted_abs_error",
            "mean"
        ),

        kmm_mae=(
            "kmm_abs_error",
            "mean"
        ),

        unweighted_iid_coverage=(
            "unweighted_iid_covered",
            "mean"
        ),

        kmm_iid_coverage=(
            "kmm_iid_covered",
            "mean"
        ),

        unweighted_block_coverage=(
            "unweighted_block_covered",
            "mean"
        ),

        kmm_block_coverage=(
            "kmm_block_covered",
            "mean"
        ),

        mean_kmm_iid_half_width=(
            "kmm_iid_half_width",
            "mean"
        ),

        mean_kmm_block_half_width=(
            "kmm_block_half_width",
            "mean"
        ),
    )
)

summary.to_csv(
    OUTPUT_DIR
    / "semisynthetic_summary.csv",
    index=False
)


# ============================================================
# 16. WITHIN-REPLICATE CHANGE FROM GOOD OVERLAP
#
# This is useful because every replicate uses the same target
# anchor and same noise field across overlap levels.
# ============================================================

good = (
    results[
        results[
            "level"
        ] == "good"
    ][
        [
            "rep",
            "weight_ess",
            "kmm_abs_error",
            "feature_mmd2"
        ]
    ]
    .rename(
        columns={
            "weight_ess":
                "good_ess",

            "kmm_abs_error":
                "good_error",

            "feature_mmd2":
                "good_mmd2",
        }
    )
)

paired = results.merge(
    good,
    on="rep",
    how="left"
)

paired[
    "delta_ess_from_good"
] = (
    paired[
        "weight_ess"
    ]
    -
    paired[
        "good_ess"
    ]
)

paired[
    "delta_error_from_good"
] = (
    paired[
        "kmm_abs_error"
    ]
    -
    paired[
        "good_error"
    ]
)

paired[
    "delta_mmd2_from_good"
] = (
    paired[
        "feature_mmd2"
    ]
    -
    paired[
        "good_mmd2"
    ]
)

paired.to_csv(
    OUTPUT_DIR
    / "semisynthetic_paired.csv",
    index=False
)


# ============================================================
# 17. PRINT SUMMARY
# ============================================================

pd.set_option(
    "display.max_columns",
    50
)

pd.set_option(
    "display.width",
    200
)

print("\n")
print("=" * 100)
print("SEMI-SYNTHETIC SUMMARY")
print("=" * 100)

display(
    summary.round(4)
)


# ============================================================
# 18. FIGURE 1:
# OVERLAP SEVERITY -> ESS
# ============================================================

plot_summary = (
    summary
    .sort_values(
        "level_order"
    )
)

plt.figure(
    figsize=(7, 4.5)
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "mean_weight_ess"
    ],
    marker="o"
)

plt.axhline(
    N_SOURCE,
    linestyle="--"
)

plt.xlabel(
    "Overlap severity"
)

plt.ylabel(
    "Mean KMM weight ESS"
)

plt.title(
    "Semi-synthetic California: overlap vs weight ESS"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "01_overlap_vs_ess.png",
    dpi=180
)

plt.show()


# ============================================================
# 19. FIGURE 2:
# OVERLAP SEVERITY -> FEATURE MMD
# ============================================================

plt.figure(
    figsize=(7, 4.5)
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "mean_mmd2"
    ],
    marker="o"
)

plt.xlabel(
    "Overlap severity"
)

plt.ylabel(
    "Mean feature MMD²"
)

plt.title(
    "Semi-synthetic California: source-target discrepancy"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "02_overlap_vs_mmd.png",
    dpi=180
)

plt.show()


# ============================================================
# 20. FIGURE 3:
# OVERLAP SEVERITY -> ERROR
# ============================================================

plt.figure(
    figsize=(7, 4.5)
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "unweighted_mae"
    ],
    marker="o",
    label="Unweighted"
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "kmm_mae"
    ],
    marker="o",
    label="KMM"
)

plt.xlabel(
    "Overlap severity"
)

plt.ylabel(
    "Mean absolute error"
)

plt.title(
    "Semi-synthetic California: overlap vs estimation error"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "03_overlap_vs_mae.png",
    dpi=180
)

plt.show()


# ============================================================
# 21. FIGURE 4:
# KMM COVERAGE
# ============================================================

plt.figure(
    figsize=(7, 4.5)
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "kmm_iid_coverage"
    ],
    marker="o",
    label="iid"
)

plt.plot(
    plot_summary[
        "level"
    ],
    plot_summary[
        "kmm_block_coverage"
    ],
    marker="o",
    label="spatial block"
)

plt.axhline(
    0.95,
    linestyle="--",
    label="nominal 95%"
)

plt.ylim(
    0,
    1.05
)

plt.xlabel(
    "Overlap severity"
)

plt.ylabel(
    "Coverage"
)

plt.title(
    "Semi-synthetic California: KMM interval coverage"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_overlap_vs_coverage.png",
    dpi=180
)

plt.show()


# ============================================================
# 22. FIGURE 5:
# ESS DIRECTLY VS KMM ERROR
# ============================================================

plt.figure(
    figsize=(7, 4.5)
)

plt.scatter(
    results[
        "weight_ess"
    ],
    results[
        "kmm_abs_error"
    ],
    alpha=0.65
)

plt.xlabel(
    "KMM weight ESS"
)

plt.ylabel(
    "KMM absolute error"
)

plt.title(
    "Semi-synthetic California: weight concentration vs error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "05_ess_vs_error.png",
    dpi=180
)

plt.show()


# ============================================================
# 23. CORRELATIONS
# ============================================================

corr_table = pd.DataFrame({
    "comparison": [
        "ESS vs KMM abs error",
        "MMD2 vs KMM abs error",
        "buffer vs ESS",
        "buffer vs KMM abs error",
        "MMD2 vs ESS",
    ],

    "pearson_r": [
        results[
            "weight_ess"
        ].corr(
            results[
                "kmm_abs_error"
            ]
        ),

        results[
            "feature_mmd2"
        ].corr(
            results[
                "kmm_abs_error"
            ]
        ),

        results[
            "buffer"
        ].corr(
            results[
                "weight_ess"
            ]
        ),

        results[
            "buffer"
        ].corr(
            results[
                "kmm_abs_error"
            ]
        ),

        results[
            "feature_mmd2"
        ].corr(
            results[
                "weight_ess"
            ]
        ),
    ]
})

corr_table.to_csv(
    OUTPUT_DIR
    / "semisynthetic_correlations.csv",
    index=False
)

print("\n")
print("=" * 100)
print("CORRELATIONS")
print("=" * 100)

display(
    corr_table.round(4)
)


# ============================================================
# 24. FILES
# ============================================================

print("\nSaved files:\n")

for path in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        " -",
        path
    )

print("\nDONE.")

print(
    "\nPlease send me these two files:"
)

print(
    OUTPUT_DIR
    / "semisynthetic_summary.csv"
)

print(
    OUTPUT_DIR
    / "semisynthetic_replicates.csv"
)

# ============================================================
# CONTROLLED SEMI-SYNTHETIC CALIFORNIA OVERLAP EXPERIMENT
#
# Real California covariates + real coordinates
# Known P(Y|X)
# Known source/target selection mechanism
# Known oracle density ratio
# Spatially dependent residuals
#
# Compare:
#   1. Unweighted
#   2. Oracle importance weighting
#   3. KMM
#
# Main question:
#
# overlap deteriorates
#       ->
# true/KMM weight ESS falls
#       ->
# estimation becomes unstable
#       ->
# dependence-aware intervals help with variance,
# but cannot repair overlap-induced instability.
# ============================================================


import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 20261005

N_SOURCE = 256
N_TARGET = 256

# 30 gives 180 total experiments.
# If Colab is too slow, change to 20.
N_REPS = 30


# ------------------------------------------------------------
# gamma controls overlap.
#
# gamma = 0:
# source and target selection probabilities are identical.
#
# larger gamma:
# increasingly different source/target distributions.
# ------------------------------------------------------------

GAMMAS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
]


KMM_BANDWIDTH = 2.0
KMM_B = 10.0
KMM_RIDGE = 1e-4


BLOCK_LON_BINS = 5
BLOCK_LAT_BINS = 5

N_BOOTSTRAP = 250

ALPHA = 0.05


# Strength of spatially correlated residual noise
NOISE_SD = 0.08


OUTPUT_DIR = Path(
    "/content/california_controlled_overlap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


master_rng = np.random.default_rng(
    SEED
)


print(
    "Output directory:",
    OUTPUT_DIR
)


# ============================================================
# 2. LOAD CALIFORNIA
# ============================================================

print(
    "\nLoading California Housing..."
)


data = fetch_california_housing(
    as_frame=True
)


df = data.frame.copy()


feature_cols = list(
    data.feature_names
)


X_raw = df[
    feature_cols
].to_numpy(
    float
)


coords = df[
    [
        "Longitude",
        "Latitude"
    ]
].to_numpy(
    float
)


print(
    "Rows:",
    len(df)
)


print(
    "Features:",
    feature_cols
)


# ============================================================
# 3. STANDARDIZE COVARIATES
# ============================================================

scaler = StandardScaler().fit(
    X_raw
)


X = scaler.transform(
    X_raw
)


coords_std = StandardScaler().fit_transform(
    coords
)


feature_index = {
    name: i
    for i, name
    in enumerate(
        feature_cols
    )
}


# ============================================================
# 4. KNOWN OUTCOME MECHANISM
#
# SAME f(X) FOR SOURCE AND TARGET.
#
# Therefore:
#
#     P(Y | X, S) = P(Y | X, T)
#
# by construction.
# ============================================================

i_income = feature_index[
    "MedInc"
]

i_age = feature_index[
    "HouseAge"
]

i_rooms = feature_index[
    "AveRooms"
]

i_occup = feature_index[
    "AveOccup"
]

i_lat = feature_index[
    "Latitude"
]

i_lon = feature_index[
    "Longitude"
]


def outcome_mean_function(X):

    income = X[
        :,
        i_income
    ]

    age = X[
        :,
        i_age
    ]

    rooms = X[
        :,
        i_rooms
    ]

    occup = X[
        :,
        i_occup
    ]

    lat = X[
        :,
        i_lat
    ]

    lon = X[
        :,
        i_lon
    ]


    eta = (
        0.75 * income

        + 0.25 * np.sin(
            age
        )

        + 0.20 * np.tanh(
            rooms
        )

        - 0.15 * np.tanh(
            occup
        )

        + 0.20 * np.sin(
            lat
        )

        + 0.20 * np.cos(
            lon
        )

        + 0.12
        * income
        * rooms
    )


    return expit(
        eta
    )


MU = outcome_mean_function(
    X
)


print(
    "\nKnown conditional mean:"
)

print(
    "min =",
    round(
        float(
            MU.min()
        ),
        4
    ),

    "| mean =",
    round(
        float(
            MU.mean()
        ),
        4
    ),

    "| max =",
    round(
        float(
            MU.max()
        ),
        4
    )
)


# ============================================================
# 5. OVERLAP SCORE g(X)
#
# IMPORTANT:
#
# We deliberately use a LOW-DIMENSIONAL score.
#
# This lets gamma control the source-target density ratio
# smoothly instead of making every regime catastrophically
# different.
# ============================================================

income = X[
    :,
    i_income
]

lat = X[
    :,
    i_lat
]

rooms = X[
    :,
    i_rooms
]


g_raw = (
    0.75 * income
    + 0.45 * lat
    + 0.25 * rooms
)


# Standardize the shift score
g = (
    g_raw
    - np.mean(
        g_raw
    )
) / np.std(
    g_raw
)


# Mild clipping avoids a few extreme California observations
# dominating the experiment.
g = np.clip(
    g,
    -3.0,
    3.0
)


print(
    "\nOverlap score:"
)

print(
    "mean =",
    round(
        float(
            np.mean(g)
        ),
        4
    ),

    "| sd =",
    round(
        float(
            np.std(g)
        ),
        4
    ),

    "| min =",
    round(
        float(
            np.min(g)
        ),
        4
    ),

    "| max =",
    round(
        float(
            np.max(g)
        ),
        4
    )
)


# ============================================================
# 6. SOURCE / TARGET SAMPLING PROBABILITIES
#
# We use symmetric exponential tilting:
#
#     q_T(x) ∝ exp(+ gamma*g(x)/2)
#
#     q_S(x) ∝ exp(- gamma*g(x)/2)
#
# Therefore the density ratio is known:
#
#     q_T(x) / q_S(x)
#       = constant * exp(gamma*g(x))
#
# This gives us an ORACLE importance weight.
# ============================================================

def sampling_probabilities(
    gamma
):

    log_target = (
        +0.5
        * gamma
        * g
    )

    log_source = (
        -0.5
        * gamma
        * g
    )


    # numerical stabilization
    log_target = (
        log_target
        - np.max(
            log_target
        )
    )

    log_source = (
        log_source
        - np.max(
            log_source
        )
    )


    p_target = np.exp(
        log_target
    )

    p_source = np.exp(
        log_source
    )


    p_target = (
        p_target
        / np.sum(
            p_target
        )
    )

    p_source = (
        p_source
        / np.sum(
            p_source
        )
    )


    return (
        p_source,
        p_target
    )


# ============================================================
# 7. ORACLE IMPORTANCE WEIGHTS
#
# For source observations:
#
#     w*(x) = q_T(x) / q_S(x)
#
# We normalize the sampled weights to mean 1 because our
# estimator is:
#
#     (1/n) sum_i w_i Y_i
# ============================================================

def oracle_weights(
    source_idx,
    p_source,
    p_target
):

    w = (
        p_target[
            source_idx
        ]
        /
        p_source[
            source_idx
        ]
    )


    w = (
        w
        / np.mean(
            w
        )
    )


    return w


# ============================================================
# 8. RBF KERNEL
# ============================================================

def rbf_kernel(
    x,
    y,
    bandwidth
):

    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )


    return np.exp(
        -d2
        /
        (
            2.0
            * bandwidth**2
        )
    )


# ============================================================
# 9. CONSTRAINED KMM
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth=2.0,
    B=10.0,
    ridge=1e-4,
):

    n = x_source.shape[0]

    m = x_target.shape[0]


    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )


    K = (
        0.5
        * (
            K
            + K.T
        )
        + ridge
        * np.eye(
            n
        )
    )


    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )


    kappa = (
        n
        / m
    ) * np.sum(
        K_st,
        axis=1
    )


    def objective(
        w
    ):

        return (
            0.5
            * w
            @ K
            @ w

            - kappa
            @ w
        )


    def gradient(
        w
    ):

        return (
            K
            @ w
            - kappa
        )


    constraint = {

        "type":
            "eq",

        "fun":
            lambda w:
            np.sum(
                w
            ) - n,

        "jac":
            lambda w:
            np.ones_like(
                w
            ),
    }


    bounds = [
        (
            0.0,
            B
        )

        for _ in range(
            n
        )
    ]


    result = minimize(

        objective,

        np.ones(
            n
        ),

        jac=gradient,

        bounds=bounds,

        constraints=constraint,

        method="SLSQP",

        options={
            "maxiter":
                1000,

            "ftol":
                1e-8,

            "disp":
                False,
        },
    )


    if not result.success:

        raise RuntimeError(
            result.message
        )


    w = np.asarray(
        result.x,
        dtype=float
    )


    return w


# ============================================================
# 10. WEIGHT DIAGNOSTICS
# ============================================================

def weight_diagnostics(
    w,
    B=None
):

    w = np.asarray(
        w,
        dtype=float
    )


    ess = (
        np.sum(
            w
        )**2
        /
        np.sum(
            w**2
        )
    )


    output = {

        "ess":
            float(
                ess
            ),

        "ess_fraction":
            float(
                ess
                / len(
                    w
                )
            ),

        "max_weight":
            float(
                np.max(
                    w
                )
            ),

        "weight_sd":
            float(
                np.std(
                    w
                )
            ),

        "frac_near_zero":
            float(
                np.mean(
                    w
                    <= 1e-4
                )
            ),
    }


    if B is None:

        output[
            "frac_at_B"
        ] = np.nan

    else:

        output[
            "frac_at_B"
        ] = float(
            np.mean(
                w
                >= B
                - 1e-4
            )
        )


    return output


# ============================================================
# 11. FEATURE MMD
# ============================================================

def empirical_mmd2(
    Xs,
    Xt,
    bandwidth=2.0
):

    Kss = rbf_kernel(
        Xs,
        Xs,
        bandwidth
    )

    Ktt = rbf_kernel(
        Xt,
        Xt,
        bandwidth
    )

    Kst = rbf_kernel(
        Xs,
        Xt,
        bandwidth
    )


    return float(
        np.mean(
            Kss
        )
        + np.mean(
            Ktt
        )
        - 2.0
        * np.mean(
            Kst
        )
    )


# ============================================================
# 12. SPATIAL NOISE
#
# Zero-mean spatial random field.
#
# Across repeated realizations:
#
#     E[epsilon(s)] = 0
#
# hence
#
#     E[Y | X] = f(X)
#
# remains known.
# ============================================================

def generate_spatial_noise(
    rng,
    noise_sd=0.08,
    n_components=30
):

    frequencies = rng.normal(
        0,
        0.70,
        size=(
            n_components,
            2
        )
    )


    phases = rng.uniform(
        0,
        2 * np.pi,
        size=n_components
    )


    amplitudes = rng.normal(
        0,
        1,
        size=n_components
    )


    projection = (
        coords_std
        @ frequencies.T
    )


    field = np.sum(

        amplitudes[
            None,
            :
        ]

        * np.cos(
            projection
            + phases[
                None,
                :
            ]
        ),

        axis=1
    )


    field = (
        field
        - np.mean(
            field
        )
    )


    sd = np.std(
        field
    )


    if sd > 1e-12:

        field = (
            field
            / sd
        )


    return (
        noise_sd
        * field
    )


# ============================================================
# 13. SPATIAL BLOCKS
# ============================================================

def spatial_block_ids(
    coordinates,
    n_lon_bins=5,
    n_lat_bins=5
):

    lon = coordinates[
        :,
        0
    ]

    lat = coordinates[
        :,
        1
    ]


    lon_edges = np.quantile(
        lon,
        np.linspace(
            0,
            1,
            n_lon_bins + 1
        )
    )


    lat_edges = np.quantile(
        lat,
        np.linspace(
            0,
            1,
            n_lat_bins + 1
        )
    )


    lon_edges[
        0
    ] -= 1e-9

    lon_edges[
        -1
    ] += 1e-9


    lat_edges[
        0
    ] -= 1e-9

    lat_edges[
        -1
    ] += 1e-9


    lon_id = np.digitize(
        lon,
        lon_edges[
            1:-1
        ]
    )


    lat_id = np.digitize(
        lat,
        lat_edges[
            1:-1
        ]
    )


    return (
        lon_id
        * n_lat_bins
        + lat_id
    )


# ============================================================
# 14. BLOCK BOOTSTRAP
# ============================================================

def block_bootstrap_half_width(
    rng,
    values,
    weights,
    block_ids,
    n_boot=250,
    alpha=0.05
):

    unique_blocks = np.unique(
        block_ids
    )


    mapping = {

        b:
        np.flatnonzero(
            block_ids
            == b
        )

        for b
        in unique_blocks
    }


    estimates = np.empty(
        n_boot
    )


    n = len(
        values
    )


    for k in range(
        n_boot
    ):

        selected = rng.choice(
            unique_blocks,
            size=len(
                unique_blocks
            ),
            replace=True
        )


        idx = np.concatenate(
            [
                mapping[
                    b
                ]

                for b
                in selected
            ]
        )


        estimates[
            k
        ] = (
            np.sum(
                weights[
                    idx
                ]
                * values[
                    idx
                ]
            )
            / n
        )


    lo, hi = np.quantile(
        estimates,
        [
            alpha / 2,
            1 - alpha / 2
        ]
    )


    return float(
        (
            hi - lo
        )
        / 2
    )


# ============================================================
# 15. IID HALF WIDTH
# ============================================================

def iid_half_width(
    y,
    w
):

    z = (
        w
        * y
    )


    return float(

        1.959963984540054

        * np.sqrt(

            np.var(
                z,
                ddof=1
            )

            / len(
                z
            )
        )
    )


# ============================================================
# 16. RUN EXPERIMENT
# ============================================================

rows = []


print(
    "\nRunning controlled overlap experiment...\n"
)


for rep in range(
    N_REPS
):

    # --------------------------------------------------------
    # One spatial field per replicate.
    #
    # All gamma levels in a replicate share the same field.
    # --------------------------------------------------------

    noise_rng = np.random.default_rng(
        SEED
        + 100000
        + rep
    )


    epsilon = generate_spatial_noise(
        noise_rng,
        noise_sd=NOISE_SD
    )


    Y = (
        MU
        + epsilon
    )


    for gamma_index, gamma in enumerate(
        GAMMAS
    ):

        sampling_rng = np.random.default_rng(
            SEED
            + rep * 1000
            + gamma_index
        )


        p_source, p_target = (
            sampling_probabilities(
                gamma
            )
        )


        # ----------------------------------------------------
        # Draw source and target independently
        # from their known distributions.
        # ----------------------------------------------------

        source_idx = sampling_rng.choice(
            len(
                df
            ),
            size=N_SOURCE,
            replace=False,
            p=p_source
        )


        # prevent exact same observation appearing
        # simultaneously in source and target
        available = np.ones(
            len(
                df
            ),
            dtype=bool
        )

        available[
            source_idx
        ] = False


        p_target_available = (
            p_target.copy()
        )


        p_target_available[
            ~available
        ] = 0.0


        p_target_available = (
            p_target_available
            / np.sum(
                p_target_available
            )
        )


        target_idx = sampling_rng.choice(
            len(
                df
            ),
            size=N_TARGET,
            replace=False,
            p=p_target_available
        )


        Xs = X[
            source_idx
        ]

        Xt = X[
            target_idx
        ]


        ys = Y[
            source_idx
        ]


        coords_s = coords[
            source_idx
        ]


        # ----------------------------------------------------
        # TARGET TRUTH
        #
        # We use the expected target-population mean under the
        # KNOWN target selection distribution:
        #
        #     mu_T = sum_x q_T(x) f(x)
        #
        # This is better than using the realized target sample
        # mean because it removes target Monte Carlo noise from
        # the reference value.
        # ----------------------------------------------------

        truth = float(
            np.sum(
                p_target
                * MU
            )
        )


        # ----------------------------------------------------
        # UNWEIGHTED
        # ----------------------------------------------------

        w_un = np.ones(
            N_SOURCE
        )


        # ----------------------------------------------------
        # ORACLE IMPORTANCE WEIGHTS
        # ----------------------------------------------------

        w_oracle = oracle_weights(
            source_idx,
            p_source,
            p_target
        )


        # ----------------------------------------------------
        # KMM
        # ----------------------------------------------------

        try:

            w_kmm = constrained_kmm_weights(

                Xs,
                Xt,

                bandwidth=
                    KMM_BANDWIDTH,

                B=
                    KMM_B,

                ridge=
                    KMM_RIDGE,
            )

        except Exception as exc:

            print(
                f"KMM failed: "
                f"rep={rep}, "
                f"gamma={gamma}: "
                f"{exc}"
            )

            continue


        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        oracle_diag = weight_diagnostics(
            w_oracle
        )


        kmm_diag = weight_diagnostics(
            w_kmm,
            B=KMM_B
        )


        mmd2 = empirical_mmd2(
            Xs,
            Xt,
            bandwidth=
                KMM_BANDWIDTH
        )


        # ----------------------------------------------------
        # POINT ESTIMATES
        # ----------------------------------------------------

        est_un = float(
            np.mean(
                ys
            )
        )


        est_oracle = float(
            np.mean(
                w_oracle
                * ys
            )
        )


        est_kmm = float(
            np.mean(
                w_kmm
                * ys
            )
        )


        error_un = (
            est_un
            - truth
        )


        error_oracle = (
            est_oracle
            - truth
        )


        error_kmm = (
            est_kmm
            - truth
        )


        # ----------------------------------------------------
        # IID INTERVALS
        # ----------------------------------------------------

        iid_un = iid_half_width(
            ys,
            w_un
        )


        iid_oracle = iid_half_width(
            ys,
            w_oracle
        )


        iid_kmm = iid_half_width(
            ys,
            w_kmm
        )


        # ----------------------------------------------------
        # SPATIAL BLOCK INTERVALS
        # ----------------------------------------------------

        blocks = spatial_block_ids(
            coords_s,
            BLOCK_LON_BINS,
            BLOCK_LAT_BINS
        )


        bootstrap_rng = np.random.default_rng(
            SEED
            + 500000
            + rep * 100
            + gamma_index
        )


        block_un = block_bootstrap_half_width(
            bootstrap_rng,
            ys,
            w_un,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )


        block_oracle = block_bootstrap_half_width(
            bootstrap_rng,
            ys,
            w_oracle,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )


        block_kmm = block_bootstrap_half_width(
            bootstrap_rng,
            ys,
            w_kmm,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )


        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        rows.append({

            "rep":
                rep,

            "gamma":
                gamma,

            "gamma_index":
                gamma_index,

            "truth":
                truth,

            "feature_mmd2":
                mmd2,


            # -----------------------------------------------
            # estimates
            # -----------------------------------------------

            "unweighted_estimate":
                est_un,

            "oracle_estimate":
                est_oracle,

            "kmm_estimate":
                est_kmm,


            # -----------------------------------------------
            # errors
            # -----------------------------------------------

            "unweighted_error":
                error_un,

            "oracle_error":
                error_oracle,

            "kmm_error":
                error_kmm,


            "unweighted_abs_error":
                abs(
                    error_un
                ),

            "oracle_abs_error":
                abs(
                    error_oracle
                ),

            "kmm_abs_error":
                abs(
                    error_kmm
                ),


            # -----------------------------------------------
            # overlap diagnostics
            # -----------------------------------------------

            "oracle_weight_ess":
                oracle_diag[
                    "ess"
                ],

            "oracle_ess_fraction":
                oracle_diag[
                    "ess_fraction"
                ],

            "oracle_max_weight":
                oracle_diag[
                    "max_weight"
                ],


            "kmm_weight_ess":
                kmm_diag[
                    "ess"
                ],

            "kmm_ess_fraction":
                kmm_diag[
                    "ess_fraction"
                ],

            "kmm_max_weight":
                kmm_diag[
                    "max_weight"
                ],

            "kmm_frac_at_B":
                kmm_diag[
                    "frac_at_B"
                ],

            "kmm_frac_near_zero":
                kmm_diag[
                    "frac_near_zero"
                ],


            # -----------------------------------------------
            # iid CI
            # -----------------------------------------------

            "unweighted_iid_half_width":
                iid_un,

            "oracle_iid_half_width":
                iid_oracle,

            "kmm_iid_half_width":
                iid_kmm,


            "unweighted_iid_covered":
                int(
                    abs(
                        error_un
                    )
                    <= iid_un
                ),

            "oracle_iid_covered":
                int(
                    abs(
                        error_oracle
                    )
                    <= iid_oracle
                ),

            "kmm_iid_covered":
                int(
                    abs(
                        error_kmm
                    )
                    <= iid_kmm
                ),


            # -----------------------------------------------
            # spatial block CI
            # -----------------------------------------------

            "unweighted_block_half_width":
                block_un,

            "oracle_block_half_width":
                block_oracle,

            "kmm_block_half_width":
                block_kmm,


            "unweighted_block_covered":
                int(
                    abs(
                        error_un
                    )
                    <= block_un
                ),

            "oracle_block_covered":
                int(
                    abs(
                        error_oracle
                    )
                    <= block_oracle
                ),

            "kmm_block_covered":
                int(
                    abs(
                        error_kmm
                    )
                    <= block_kmm
                ),
        })


        print(

            f"rep {rep:02d} | "

            f"gamma={gamma:3.1f} | "

            f"oracle ESS="
            f"{oracle_diag['ess']:6.1f} | "

            f"KMM ESS="
            f"{kmm_diag['ess']:6.1f} | "

            f"KMM max="
            f"{kmm_diag['max_weight']:5.2f} | "

            f"KMM err="
            f"{abs(error_kmm):.4f}"
        )


# ============================================================
# 17. DATAFRAME
# ============================================================

results = pd.DataFrame(
    rows
)


if len(
    results
) == 0:

    raise RuntimeError(
        "No successful experiments."
    )


results = results.sort_values(
    [
        "gamma",
        "rep"
    ]
).reset_index(
    drop=True
)


results.to_csv(

    OUTPUT_DIR
    / "controlled_overlap_replicates.csv",

    index=False
)


# ============================================================
# 18. SUMMARY
# ============================================================

summary = (

    results

    .groupby(
        "gamma",
        as_index=False
    )

    .agg(

        n_reps=(
            "rep",
            "count"
        ),


        mean_mmd2=(
            "feature_mmd2",
            "mean"
        ),


        # -----------------------------------------------
        # oracle overlap
        # -----------------------------------------------

        oracle_ess=(
            "oracle_weight_ess",
            "mean"
        ),

        oracle_ess_fraction=(
            "oracle_ess_fraction",
            "mean"
        ),

        oracle_max_weight=(
            "oracle_max_weight",
            "mean"
        ),


        # -----------------------------------------------
        # KMM overlap
        # -----------------------------------------------

        kmm_ess=(
            "kmm_weight_ess",
            "mean"
        ),

        kmm_ess_fraction=(
            "kmm_ess_fraction",
            "mean"
        ),

        kmm_max_weight=(
            "kmm_max_weight",
            "mean"
        ),

        kmm_frac_at_B=(
            "kmm_frac_at_B",
            "mean"
        ),

        kmm_frac_near_zero=(
            "kmm_frac_near_zero",
            "mean"
        ),


        # -----------------------------------------------
        # MAE
        # -----------------------------------------------

        unweighted_mae=(
            "unweighted_abs_error",
            "mean"
        ),

        oracle_mae=(
            "oracle_abs_error",
            "mean"
        ),

        kmm_mae=(
            "kmm_abs_error",
            "mean"
        ),


        # -----------------------------------------------
        # bias
        # -----------------------------------------------

        unweighted_bias=(
            "unweighted_error",
            "mean"
        ),

        oracle_bias=(
            "oracle_error",
            "mean"
        ),

        kmm_bias=(
            "kmm_error",
            "mean"
        ),


        # -----------------------------------------------
        # iid coverage
        # -----------------------------------------------

        unweighted_iid_coverage=(
            "unweighted_iid_covered",
            "mean"
        ),

        oracle_iid_coverage=(
            "oracle_iid_covered",
            "mean"
        ),

        kmm_iid_coverage=(
            "kmm_iid_covered",
            "mean"
        ),


        # -----------------------------------------------
        # block coverage
        # -----------------------------------------------

        unweighted_block_coverage=(
            "unweighted_block_covered",
            "mean"
        ),

        oracle_block_coverage=(
            "oracle_block_covered",
            "mean"
        ),

        kmm_block_coverage=(
            "kmm_block_covered",
            "mean"
        ),


        # -----------------------------------------------
        # interval width
        # -----------------------------------------------

        kmm_iid_half_width=(
            "kmm_iid_half_width",
            "mean"
        ),

        kmm_block_half_width=(
            "kmm_block_half_width",
            "mean"
        ),

        oracle_iid_half_width=(
            "oracle_iid_half_width",
            "mean"
        ),

        oracle_block_half_width=(
            "oracle_block_half_width",
            "mean"
        ),
    )
)


summary.to_csv(

    OUTPUT_DIR
    / "controlled_overlap_summary.csv",

    index=False
)


# ============================================================
# 19. PRINT SUMMARY
# ============================================================

pd.set_option(
    "display.max_columns",
    100
)


pd.set_option(
    "display.width",
    220
)


print(
    "\n"
)


print(
    "=" * 120
)


print(
    "CONTROLLED OVERLAP SUMMARY"
)


print(
    "=" * 120
)


display(
    summary.round(
        4
    )
)


# ============================================================
# 20. FIGURE:
# TRUE ESS + KMM ESS
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "oracle_ess"
    ],
    marker="o",
    label="Oracle density-ratio ESS"
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "kmm_ess"
    ],
    marker="o",
    label="KMM ESS"
)


plt.axhline(
    N_SOURCE,
    linestyle="--"
)


plt.xlabel(
    "Overlap severity γ"
)


plt.ylabel(
    "Weight ESS"
)


plt.title(
    "Controlled overlap: effective sample size"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "01_gamma_vs_ess.png",
    dpi=180
)


plt.show()


# ============================================================
# 21. FIGURE:
# MMD
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "mean_mmd2"
    ],
    marker="o"
)


plt.xlabel(
    "Overlap severity γ"
)


plt.ylabel(
    "Feature MMD²"
)


plt.title(
    "Controlled overlap: source-target discrepancy"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "02_gamma_vs_mmd.png",
    dpi=180
)


plt.show()


# ============================================================
# 22. FIGURE:
# MAE
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "unweighted_mae"
    ],
    marker="o",
    label="Unweighted"
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "oracle_mae"
    ],
    marker="o",
    label="Oracle IW"
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "kmm_mae"
    ],
    marker="o",
    label="KMM"
)


plt.xlabel(
    "Overlap severity γ"
)


plt.ylabel(
    "Mean absolute error"
)


plt.title(
    "Controlled overlap: estimation error"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "03_gamma_vs_mae.png",
    dpi=180
)


plt.show()


# ============================================================
# 23. FIGURE:
# KMM COVERAGE
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "kmm_iid_coverage"
    ],
    marker="o",
    label="KMM iid"
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "kmm_block_coverage"
    ],
    marker="o",
    label="KMM spatial block"
)


plt.axhline(
    0.95,
    linestyle="--",
    label="95% nominal"
)


plt.ylim(
    0,
    1.05
)


plt.xlabel(
    "Overlap severity γ"
)


plt.ylabel(
    "Coverage"
)


plt.title(
    "Controlled overlap: KMM interval coverage"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "04_gamma_vs_kmm_coverage.png",
    dpi=180
)


plt.show()


# ============================================================
# 24. FIGURE:
# ORACLE COVERAGE
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "oracle_iid_coverage"
    ],
    marker="o",
    label="Oracle IW iid"
)


plt.plot(
    summary[
        "gamma"
    ],
    summary[
        "oracle_block_coverage"
    ],
    marker="o",
    label="Oracle IW spatial block"
)


plt.axhline(
    0.95,
    linestyle="--",
    label="95% nominal"
)


plt.ylim(
    0,
    1.05
)


plt.xlabel(
    "Overlap severity γ"
)


plt.ylabel(
    "Coverage"
)


plt.title(
    "Controlled overlap: oracle-IW interval coverage"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "05_gamma_vs_oracle_coverage.png",
    dpi=180
)


plt.show()


# ============================================================
# 25. FIGURE:
# ESS VS KMM ERROR
# ============================================================

plt.figure(
    figsize=(
        7,
        4.5
    )
)


plt.scatter(
    results[
        "kmm_weight_ess"
    ],
    results[
        "kmm_abs_error"
    ],
    alpha=0.6
)


plt.xlabel(
    "KMM weight ESS"
)


plt.ylabel(
    "KMM absolute error"
)


plt.title(
    "Controlled overlap: ESS vs KMM error"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_DIR
    / "06_ess_vs_kmm_error.png",
    dpi=180
)


plt.show()


# ============================================================
# 26. CORRELATIONS
# ============================================================

corr = pd.DataFrame({

    "comparison": [

        "gamma vs oracle ESS",

        "gamma vs KMM ESS",

        "gamma vs MMD2",

        "gamma vs oracle abs error",

        "gamma vs KMM abs error",

        "KMM ESS vs KMM abs error",

        "oracle ESS vs oracle abs error",
    ],


    "pearson_r": [

        results[
            "gamma"
        ].corr(
            results[
                "oracle_weight_ess"
            ]
        ),


        results[
            "gamma"
        ].corr(
            results[
                "kmm_weight_ess"
            ]
        ),


        results[
            "gamma"
        ].corr(
            results[
                "feature_mmd2"
            ]
        ),


        results[
            "gamma"
        ].corr(
            results[
                "oracle_abs_error"
            ]
        ),


        results[
            "gamma"
        ].corr(
            results[
                "kmm_abs_error"
            ]
        ),


        results[
            "kmm_weight_ess"
        ].corr(
            results[
                "kmm_abs_error"
            ]
        ),


        results[
            "oracle_weight_ess"
        ].corr(
            results[
                "oracle_abs_error"
            ]
        ),
    ]
})


corr.to_csv(

    OUTPUT_DIR
    / "controlled_overlap_correlations.csv",

    index=False
)


print(
    "\n"
)


print(
    "=" * 120
)


print(
    "CORRELATIONS"
)


print(
    "=" * 120
)


display(
    corr.round(
        4
    )
)


# ============================================================
# 27. SIMPLE CHECKS
# ============================================================

print(
    "\n"
)


print(
    "=" * 120
)


print(
    "SANITY CHECKS"
)


print(
    "=" * 120
)


gamma_zero = results[
    results[
        "gamma"
    ] == 0
]


print(
    "gamma=0 oracle ESS mean:",
    round(
        gamma_zero[
            "oracle_weight_ess"
        ].mean(),
        2
    )
)


print(
    "gamma=0 oracle max weight mean:",
    round(
        gamma_zero[
            "oracle_max_weight"
        ].mean(),
        3
    )
)


print(
    "gamma=0 KMM ESS mean:",
    round(
        gamma_zero[
            "kmm_weight_ess"
        ].mean(),
        2
    )
)


print(
    "\nExpected:"
)


print(
    " - oracle ESS at gamma=0 should be exactly ~256"
)


print(
    " - oracle max weight at gamma=0 should be ~1"
)


print(
    " - MMD should increase with gamma"
)


print(
    " - oracle ESS should decrease strongly with gamma"
)


print(
    " - KMM ESS should generally decrease with gamma"
)


# ============================================================
# 28. LIST FILES
# ============================================================

print(
    "\nSaved files:\n"
)


for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        " -",
        path
    )


print(
    "\nDONE."
)


print(
    "\nPlease send me:"
)


print(
    OUTPUT_DIR
    / "controlled_overlap_summary.csv"
)


print(
    OUTPUT_DIR
    / "controlled_overlap_replicates.csv"
)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# California controlled-overlap experiment
# Summary from 30 repetitions per gamma
# ============================================================

df = pd.DataFrame({
    "gamma": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],

    "oracle_ess": [
        256.0,
        198.7821,
        89.4149,
        38.5277,
        20.2614,
        9.4365
    ],

    "kmm_ess": [
        74.9505,
        60.5136,
        46.2312,
        35.8647,
        31.7946,
        28.0171
    ],

    "unweighted_mae": [
        0.0110,
        0.0683,
        0.1372,
        0.2090,
        0.2827,
        0.4110
    ],

    "oracle_mae": [
        0.0110,
        0.0156,
        0.0285,
        0.0609,
        0.0887,
        0.1430
    ],

    "kmm_mae": [
        0.0073,
        0.0104,
        0.0231,
        0.0542,
        0.1247,
        0.2869
    ]
})


# ============================================================
# AISTATS-friendly plotting
# ============================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig, axes = plt.subplots(
    1, 2,
    figsize=(6.8, 2.45),
    constrained_layout=True
)


# ------------------------------------------------------------
# Panel A: weight ESS
# ------------------------------------------------------------

ax = axes[0]

ax.plot(
    df["gamma"],
    df["oracle_ess"],
    marker="o",
    linewidth=1.6,
    markersize=4,
    label="Oracle IW"
)

ax.plot(
    df["gamma"],
    df["kmm_ess"],
    marker="s",
    linewidth=1.6,
    markersize=4,
    label="KMM"
)

ax.axhline(
    256,
    linestyle=":",
    linewidth=1,
    label="Nominal $n$"
)

ax.set_xlabel(r"Overlap severity $\gamma$")
ax.set_ylabel("Weight ESS")
ax.set_title("(a) Effective sample size")
ax.set_xticks(df["gamma"])
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.legend(frameon=False)


# ------------------------------------------------------------
# Panel B: target-mean error
# ------------------------------------------------------------

ax = axes[1]

ax.plot(
    df["gamma"],
    df["unweighted_mae"],
    marker="^",
    linewidth=1.6,
    markersize=4,
    label="Unweighted"
)

ax.plot(
    df["gamma"],
    df["oracle_mae"],
    marker="o",
    linewidth=1.6,
    markersize=4,
    label="Oracle IW"
)

ax.plot(
    df["gamma"],
    df["kmm_mae"],
    marker="s",
    linewidth=1.6,
    markersize=4,
    label="KMM"
)

ax.set_xlabel(r"Overlap severity $\gamma$")
ax.set_ylabel("Target-mean MAE")
ax.set_title("(b) Estimation error")
ax.set_xticks(df["gamma"])
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.legend(frameon=False)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "california_overlap_stress.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "california_overlap_stress.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()









# ============================================================
# CONTROLLED SEMI-SYNTHETIC CALIFORNIA OVERLAP EXPERIMENT
# SANITY VERSION BEFORE FINAL RUN
#
# Real California covariates + coordinates
# Known bounded P(Y|X)
# Exact source/target exponential tilting
# Known oracle density ratio
# Bounded spatially dependent residuals
#
# IMPORTANT:
# This run is intentionally:
#   N_REPS = 10
#   GAMMAS = [0.0]
#
# It also runs a gamma=0 KMM bandwidth/ridge sanity ablation.
# Do NOT change to 100 reps yet.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 20261005

N_SOURCE = 256
N_TARGET = 256

# ------------------------------------------------------------
# SANITY RUN ONLY
# ------------------------------------------------------------

N_REPS = 10

GAMMAS = [0.0]


# ------------------------------------------------------------
# Current KMM parameters.
# We will decide whether to change these AFTER the ablation.
# ------------------------------------------------------------

KMM_BANDWIDTH = 2.0
KMM_B = 10.0
KMM_RIDGE = 1e-4


# ------------------------------------------------------------
# Spatial block bootstrap
# ------------------------------------------------------------

BLOCK_LON_BINS = 5
BLOCK_LAT_BINS = 5

N_BOOTSTRAP = 250

ALPHA = 0.05


# ------------------------------------------------------------
# Bounded spatial noise
# MU lies in [0.20, 0.80]
# epsilon lies in [-0.12, 0.12]
#
# Therefore:
# Y lies in [0.08, 0.92].
# ------------------------------------------------------------

NOISE_SCALE = 0.12


OUTPUT_DIR = Path(
    "/content/california_controlled_overlap_sanity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. LOAD CALIFORNIA HOUSING
# ============================================================

print("\nLoading California Housing...")

data = fetch_california_housing(
    as_frame=True
)

df = data.frame.copy()

feature_cols = list(
    data.feature_names
)

X_raw = df[
    feature_cols
].to_numpy(float)

coords = df[
    ["Longitude", "Latitude"]
].to_numpy(float)

print("Rows:", len(df))
print("Features:", feature_cols)


# ============================================================
# 3. STANDARDIZE COVARIATES
# ============================================================

scaler = StandardScaler().fit(
    X_raw
)

X = scaler.transform(
    X_raw
)

coords_std = StandardScaler().fit_transform(
    coords
)

feature_index = {
    name: i
    for i, name in enumerate(feature_cols)
}


# ============================================================
# 4. KNOWN BOUNDED OUTCOME MECHANISM
#
# Same conditional mean for source and target.
#
# MU in [0.20, 0.80].
# ============================================================

i_income = feature_index["MedInc"]
i_age = feature_index["HouseAge"]
i_rooms = feature_index["AveRooms"]
i_occup = feature_index["AveOccup"]
i_lat = feature_index["Latitude"]
i_lon = feature_index["Longitude"]


def outcome_mean_function(X):

    income = X[:, i_income]
    age = X[:, i_age]
    rooms = X[:, i_rooms]
    occup = X[:, i_occup]
    lat = X[:, i_lat]
    lon = X[:, i_lon]

    eta = (
        0.75 * income
        + 0.25 * np.sin(age)
        + 0.20 * np.tanh(rooms)
        - 0.15 * np.tanh(occup)
        + 0.20 * np.sin(lat)
        + 0.20 * np.cos(lon)
        + 0.12 * income * rooms
    )

    # Bounded away from 0 and 1 so that bounded spatial
    # residuals cannot push Y outside [0,1].
    return (
        0.20
        + 0.60 * expit(eta)
    )


MU = outcome_mean_function(
    X
)


print("\nKnown conditional mean:")
print(
    "min =",
    round(float(MU.min()), 4),
    "| mean =",
    round(float(MU.mean()), 4),
    "| max =",
    round(float(MU.max()), 4)
)


# ============================================================
# 5. OVERLAP SCORE g(X)
# ============================================================

income = X[:, i_income]
lat = X[:, i_lat]
rooms = X[:, i_rooms]

g_raw = (
    0.75 * income
    + 0.45 * lat
    + 0.25 * rooms
)

g = (
    g_raw - np.mean(g_raw)
) / np.std(g_raw)

# Avoid domination by a few extreme observations.
g = np.clip(
    g,
    -3.0,
    3.0
)


print("\nOverlap score:")
print(
    "mean =",
    round(float(np.mean(g)), 4),
    "| sd =",
    round(float(np.std(g)), 4),
    "| min =",
    round(float(np.min(g)), 4),
    "| max =",
    round(float(np.max(g)), 4)
)


# ============================================================
# 6. SOURCE / TARGET SAMPLING DISTRIBUTIONS
#
# q_T(x) proportional to exp(+gamma*g(x)/2)
# q_S(x) proportional to exp(-gamma*g(x)/2)
#
# Hence:
#
# q_T(x) / q_S(x)
#     proportional to exp(gamma*g(x))
#
# We sample WITH replacement so these distributions correspond
# directly to the implemented sampling mechanism.
# ============================================================

def sampling_probabilities(gamma):

    log_target = (
        +0.5 * gamma * g
    )

    log_source = (
        -0.5 * gamma * g
    )

    # Numerical stabilization
    log_target = (
        log_target
        - np.max(log_target)
    )

    log_source = (
        log_source
        - np.max(log_source)
    )

    p_target = np.exp(
        log_target
    )

    p_source = np.exp(
        log_source
    )

    p_target = (
        p_target
        / np.sum(p_target)
    )

    p_source = (
        p_source
        / np.sum(p_source)
    )

    return (
        p_source,
        p_target
    )


# ============================================================
# 7. ORACLE IMPORTANCE WEIGHTS
#
# Exact discrete population density ratio:
#
#       w*(x) = q_T(x) / q_S(x)
#
# We then normalize sampled weights to mean 1.
# The point estimator below is explicitly self-normalized.
# ============================================================

def oracle_weights(
    source_idx,
    p_source,
    p_target
):

    w = (
        p_target[source_idx]
        / p_source[source_idx]
    )

    w = (
        w
        / np.mean(w)
    )

    return w


# ============================================================
# 8. RBF KERNEL
# ============================================================

def rbf_kernel(
    x,
    y,
    bandwidth
):

    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )

    return np.exp(
        -d2
        / (2.0 * bandwidth**2)
    )


# ============================================================
# 9. CONSTRAINED KMM
#
# Exact constraints:
#
#       0 <= w_i <= B
#       sum_i w_i = n
#
# No post-hoc clipping / renormalization.
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth=2.0,
    B=10.0,
    ridge=1e-4
):

    n = x_source.shape[0]
    m = x_target.shape[0]

    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    K = (
        0.5 * (K + K.T)
        + ridge * np.eye(n)
    )

    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    kappa = (
        n / m
    ) * np.sum(
        K_st,
        axis=1
    )

    def objective(w):

        return (
            0.5 * w @ K @ w
            - kappa @ w
        )

    def gradient(w):

        return (
            K @ w
            - kappa
        )

    constraint = {
        "type": "eq",
        "fun": lambda w:
            np.sum(w) - n,
        "jac": lambda w:
            np.ones_like(w),
    }

    bounds = [
        (0.0, B)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        np.ones(n),
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    w = np.asarray(
        result.x,
        dtype=float
    )

    # Numerical safety checks
    if abs(np.sum(w) - n) > 1e-4:
        raise RuntimeError(
            "KMM equality constraint violated."
        )

    if np.min(w) < -1e-5:
        raise RuntimeError(
            "Negative KMM weight."
        )

    if np.max(w) > B + 1e-4:
        raise RuntimeError(
            "KMM upper bound violated."
        )

    return w


# ============================================================
# 10. WEIGHT DIAGNOSTICS
# ============================================================

def weight_diagnostics(
    w,
    B=None
):

    w = np.asarray(
        w,
        dtype=float
    )

    ess = (
        np.sum(w)**2
        / np.sum(w**2)
    )

    output = {
        "ess":
            float(ess),

        "ess_fraction":
            float(ess / len(w)),

        "max_weight":
            float(np.max(w)),

        "weight_sd":
            float(np.std(w)),

        "frac_near_zero":
            float(
                np.mean(
                    w <= 1e-4
                )
            ),
    }

    if B is None:

        output["frac_at_B"] = np.nan

    else:

        output["frac_at_B"] = float(
            np.mean(
                w >= B - 1e-4
            )
        )

    return output


# ============================================================
# 11. FEATURE MMD
# ============================================================

def empirical_mmd2(
    Xs,
    Xt,
    bandwidth=2.0
):

    Kss = rbf_kernel(
        Xs,
        Xs,
        bandwidth
    )

    Ktt = rbf_kernel(
        Xt,
        Xt,
        bandwidth
    )

    Kst = rbf_kernel(
        Xs,
        Xt,
        bandwidth
    )

    return float(
        np.mean(Kss)
        + np.mean(Ktt)
        - 2.0 * np.mean(Kst)
    )


# ============================================================
# 12. BOUNDED SPATIALLY CORRELATED NOISE
#
# Random Fourier construction:
#
# z(s) =
#   [cos(omega*s)' a + sin(omega*s)' b] / sqrt(2K)
#
# a,b are zero-mean symmetric Gaussian coefficients.
#
# epsilon(s) = scale * tanh(z(s))
#
# Hence epsilon is bounded and has zero expectation over
# repeated random-field realizations.
# ============================================================

def generate_spatial_noise(
    rng,
    noise_scale=0.12,
    n_components=30
):

    frequencies = rng.normal(
        0.0,
        0.70,
        size=(n_components, 2)
    )

    a = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    b = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    projection = (
        coords_std
        @ frequencies.T
    )

    field = (
        np.cos(projection) @ a
        + np.sin(projection) @ b
    ) / np.sqrt(
        2.0 * n_components
    )

    epsilon = (
        noise_scale
        * np.tanh(field)
    )

    return epsilon


# ============================================================
# 13. SELF-NORMALIZED WEIGHTED MEAN
# ============================================================

def weighted_mean(
    y,
    w
):

    y = np.asarray(
        y,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    denom = np.sum(w)

    if denom <= 1e-12:
        return np.nan

    return float(
        np.sum(w * y)
        / denom
    )


# ============================================================
# 14. IID HALF-WIDTH
#
# For self-normalized weighted mean:
#
# psi_i = w_i (Y_i - mu_hat)
#
# SE =
# sd(psi_i) /
# [sqrt(n) * mean(w)]
# ============================================================

def iid_half_width(
    y,
    w,
    alpha=0.05
):

    y = np.asarray(
        y,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    estimate = weighted_mean(
        y,
        w
    )

    mean_w = np.mean(w)

    if (
        not np.isfinite(estimate)
        or mean_w <= 1e-12
    ):
        return np.nan

    psi = (
        w
        * (y - estimate)
    )

    se = (
        np.std(
            psi,
            ddof=1
        )
        / (
            np.sqrt(len(y))
            * mean_w
        )
    )

    # 95% normal critical value.
    # alpha kept as an argument for consistency;
    # experiment uses alpha=0.05.
    if abs(alpha - 0.05) > 1e-12:
        from scipy.stats import norm
        zcrit = norm.ppf(
            1.0 - alpha / 2.0
        )
    else:
        zcrit = 1.959963984540054

    return float(
        zcrit * se
    )


# ============================================================
# 15. SPATIAL BLOCKS
# ============================================================

def spatial_block_ids(
    coordinates,
    n_lon_bins=5,
    n_lat_bins=5
):

    lon = coordinates[:, 0]
    lat = coordinates[:, 1]

    lon_edges = np.quantile(
        lon,
        np.linspace(
            0,
            1,
            n_lon_bins + 1
        )
    )

    lat_edges = np.quantile(
        lat,
        np.linspace(
            0,
            1,
            n_lat_bins + 1
        )
    )

    lon_edges[0] -= 1e-9
    lon_edges[-1] += 1e-9

    lat_edges[0] -= 1e-9
    lat_edges[-1] += 1e-9

    lon_id = np.digitize(
        lon,
        lon_edges[1:-1]
    )

    lat_id = np.digitize(
        lat,
        lat_edges[1:-1]
    )

    return (
        lon_id * n_lat_bins
        + lat_id
    )


# ============================================================
# 16. SPATIAL BLOCK BOOTSTRAP
#
# IMPORTANT FIX:
#
# We do NOT divide by original n.
#
# For every resampled set of blocks we recompute:
#
#       sum(w_i Y_i) / sum(w_i)
#
# This remains valid when resampled blocks have unequal sizes.
#
# We hold fitted weights fixed, so this is conditional on
# estimated weights and does not include KMM fitting uncertainty.
# ============================================================

def block_bootstrap_half_width(
    rng,
    values,
    weights,
    block_ids,
    n_boot=250,
    alpha=0.05
):

    values = np.asarray(
        values,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    block_ids = np.asarray(
        block_ids
    )

    unique_blocks = np.unique(
        block_ids
    )

    mapping = {
        b: np.flatnonzero(
            block_ids == b
        )
        for b in unique_blocks
    }

    estimates = np.empty(
        n_boot
    )

    for k in range(n_boot):

        selected = rng.choice(
            unique_blocks,
            size=len(unique_blocks),
            replace=True
        )

        idx = np.concatenate([
            mapping[b]
            for b in selected
        ])

        y_boot = values[idx]
        w_boot = weights[idx]

        estimates[k] = weighted_mean(
            y_boot,
            w_boot
        )

    estimates = estimates[
        np.isfinite(estimates)
    ]

    if len(estimates) < 10:
        return np.nan

    lo, hi = np.quantile(
        estimates,
        [
            alpha / 2,
            1 - alpha / 2
        ]
    )

    return float(
        (hi - lo) / 2.0
    )


# ============================================================
# 17. RUN SMALL GAMMA=0 EXPERIMENT
# ============================================================

rows = []

print(
    "\nRunning gamma=0 sanity experiment...\n"
)


for rep in range(N_REPS):

    # --------------------------------------------------------
    # Generate one bounded spatial field per replicate.
    # --------------------------------------------------------

    noise_rng = np.random.default_rng(
        SEED
        + 100000
        + rep
    )

    epsilon = generate_spatial_noise(
        noise_rng,
        noise_scale=NOISE_SCALE
    )

    Y = (
        MU
        + epsilon
    )

    # Boundedness should hold by construction.
    if (
        np.min(Y) < -1e-12
        or np.max(Y) > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "Y left [0,1]."
        )

    for gamma_index, gamma in enumerate(
        GAMMAS
    ):

        sampling_rng = np.random.default_rng(
            SEED
            + rep * 1000
            + gamma_index
        )

        p_source, p_target = (
            sampling_probabilities(
                gamma
            )
        )

        # ----------------------------------------------------
        # Independent draws WITH replacement.
        # ----------------------------------------------------

        source_idx = sampling_rng.choice(
            len(df),
            size=N_SOURCE,
            replace=True,
            p=p_source
        )

        target_idx = sampling_rng.choice(
            len(df),
            size=N_TARGET,
            replace=True,
            p=p_target
        )

        Xs = X[source_idx]
        Xt = X[target_idx]

        ys = Y[source_idx]

        coords_s = coords[
            source_idx
        ]

        # ----------------------------------------------------
        # Population target truth:
        #
        # E_{q_T}[MU(X)]
        #
        # Spatial residual has zero expectation over field
        # realizations, so this is the known target estimand.
        # ----------------------------------------------------

        truth = float(
            np.sum(
                p_target * MU
            )
        )

        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        w_un = np.ones(
            N_SOURCE
        )

        w_oracle = oracle_weights(
            source_idx,
            p_source,
            p_target
        )

        try:

            w_kmm = constrained_kmm_weights(
                Xs,
                Xt,
                bandwidth=KMM_BANDWIDTH,
                B=KMM_B,
                ridge=KMM_RIDGE
            )

        except Exception as exc:

            print(
                f"KMM failed: "
                f"rep={rep}, "
                f"gamma={gamma}: "
                f"{exc}"
            )

            continue

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        oracle_diag = weight_diagnostics(
            w_oracle
        )

        kmm_diag = weight_diagnostics(
            w_kmm,
            B=KMM_B
        )

        mmd2 = empirical_mmd2(
            Xs,
            Xt,
            bandwidth=KMM_BANDWIDTH
        )

        # ----------------------------------------------------
        # POINT ESTIMATES
        # ----------------------------------------------------

        est_un = weighted_mean(
            ys,
            w_un
        )

        est_oracle = weighted_mean(
            ys,
            w_oracle
        )

        est_kmm = weighted_mean(
            ys,
            w_kmm
        )

        error_un = (
            est_un - truth
        )

        error_oracle = (
            est_oracle - truth
        )

        error_kmm = (
            est_kmm - truth
        )

        # ----------------------------------------------------
        # IID INTERVALS
        # ----------------------------------------------------

        iid_un = iid_half_width(
            ys,
            w_un,
            alpha=ALPHA
        )

        iid_oracle = iid_half_width(
            ys,
            w_oracle,
            alpha=ALPHA
        )

        iid_kmm = iid_half_width(
            ys,
            w_kmm,
            alpha=ALPHA
        )

        # ----------------------------------------------------
        # SPATIAL BLOCK INTERVALS
        # ----------------------------------------------------

        blocks = spatial_block_ids(
            coords_s,
            BLOCK_LON_BINS,
            BLOCK_LAT_BINS
        )

        # Separate RNGs make bootstrap results independent
        # across estimators while remaining reproducible.

        bootstrap_rng_un = np.random.default_rng(
            SEED
            + 500000
            + rep * 1000
            + 10
        )

        bootstrap_rng_oracle = np.random.default_rng(
            SEED
            + 500000
            + rep * 1000
            + 20
        )

        bootstrap_rng_kmm = np.random.default_rng(
            SEED
            + 500000
            + rep * 1000
            + 30
        )

        block_un = block_bootstrap_half_width(
            bootstrap_rng_un,
            ys,
            w_un,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )

        block_oracle = block_bootstrap_half_width(
            bootstrap_rng_oracle,
            ys,
            w_oracle,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )

        block_kmm = block_bootstrap_half_width(
            bootstrap_rng_kmm,
            ys,
            w_kmm,
            blocks,
            n_boot=N_BOOTSTRAP,
            alpha=ALPHA
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        rows.append({

            "rep": rep,
            "gamma": gamma,

            "truth": truth,

            "y_min": float(
                np.min(Y)
            ),

            "y_max": float(
                np.max(Y)
            ),

            "feature_mmd2": mmd2,

            # Estimates
            "unweighted_estimate":
                est_un,

            "oracle_estimate":
                est_oracle,

            "kmm_estimate":
                est_kmm,

            # Errors
            "unweighted_error":
                error_un,

            "oracle_error":
                error_oracle,

            "kmm_error":
                error_kmm,

            "unweighted_abs_error":
                abs(error_un),

            "oracle_abs_error":
                abs(error_oracle),

            "kmm_abs_error":
                abs(error_kmm),

            # Oracle weights
            "oracle_weight_ess":
                oracle_diag["ess"],

            "oracle_ess_fraction":
                oracle_diag[
                    "ess_fraction"
                ],

            "oracle_max_weight":
                oracle_diag[
                    "max_weight"
                ],

            # KMM weights
            "kmm_weight_ess":
                kmm_diag["ess"],

            "kmm_ess_fraction":
                kmm_diag[
                    "ess_fraction"
                ],

            "kmm_max_weight":
                kmm_diag[
                    "max_weight"
                ],

            "kmm_frac_at_B":
                kmm_diag[
                    "frac_at_B"
                ],

            "kmm_frac_near_zero":
                kmm_diag[
                    "frac_near_zero"
                ],

            # IID
            "unweighted_iid_half_width":
                iid_un,

            "oracle_iid_half_width":
                iid_oracle,

            "kmm_iid_half_width":
                iid_kmm,

            "unweighted_iid_covered":
                int(
                    abs(error_un)
                    <= iid_un
                ),

            "oracle_iid_covered":
                int(
                    abs(error_oracle)
                    <= iid_oracle
                ),

            "kmm_iid_covered":
                int(
                    abs(error_kmm)
                    <= iid_kmm
                ),

            # Block
            "unweighted_block_half_width":
                block_un,

            "oracle_block_half_width":
                block_oracle,

            "kmm_block_half_width":
                block_kmm,

            "unweighted_block_covered":
                int(
                    abs(error_un)
                    <= block_un
                ),

            "oracle_block_covered":
                int(
                    abs(error_oracle)
                    <= block_oracle
                ),

            "kmm_block_covered":
                int(
                    abs(error_kmm)
                    <= block_kmm
                ),
        })

        print(
            f"rep {rep:02d} | "
            f"oracle ESS="
            f"{oracle_diag['ess']:6.1f} | "
            f"KMM ESS="
            f"{kmm_diag['ess']:6.1f} | "
            f"KMM max="
            f"{kmm_diag['max_weight']:5.2f} | "
            f"KMM |err|="
            f"{abs(error_kmm):.4f}"
        )


# ============================================================
# 18. SMALL-RUN SUMMARY
# ============================================================

results = pd.DataFrame(
    rows
)

if len(results) == 0:
    raise RuntimeError(
        "No successful experiments."
    )


results.to_csv(
    OUTPUT_DIR
    / "gamma0_sanity_replicates.csv",
    index=False
)


print("\n")
print("=" * 100)
print("GAMMA=0 SANITY SUMMARY")
print("=" * 100)


gamma0_summary = pd.DataFrame({

    "quantity": [

        "Oracle weight ESS",
        "KMM weight ESS",

        "Oracle max weight",
        "KMM max weight",

        "KMM frac at B",
        "KMM frac near zero",

        "Unweighted MAE",
        "Oracle MAE",
        "KMM MAE",

        "Unweighted iid coverage",
        "Oracle iid coverage",
        "KMM iid coverage",

        "Unweighted block coverage",
        "Oracle block coverage",
        "KMM block coverage",

        "Unweighted iid HW",
        "Oracle iid HW",
        "KMM iid HW",

        "Unweighted block HW",
        "Oracle block HW",
        "KMM block HW",

        "Minimum Y",
        "Maximum Y",
    ],

    "value": [

        results[
            "oracle_weight_ess"
        ].mean(),

        results[
            "kmm_weight_ess"
        ].mean(),

        results[
            "oracle_max_weight"
        ].mean(),

        results[
            "kmm_max_weight"
        ].mean(),

        results[
            "kmm_frac_at_B"
        ].mean(),

        results[
            "kmm_frac_near_zero"
        ].mean(),

        results[
            "unweighted_abs_error"
        ].mean(),

        results[
            "oracle_abs_error"
        ].mean(),

        results[
            "kmm_abs_error"
        ].mean(),

        results[
            "unweighted_iid_covered"
        ].mean(),

        results[
            "oracle_iid_covered"
        ].mean(),

        results[
            "kmm_iid_covered"
        ].mean(),

        results[
            "unweighted_block_covered"
        ].mean(),

        results[
            "oracle_block_covered"
        ].mean(),

        results[
            "kmm_block_covered"
        ].mean(),

        results[
            "unweighted_iid_half_width"
        ].mean(),

        results[
            "oracle_iid_half_width"
        ].mean(),

        results[
            "kmm_iid_half_width"
        ].mean(),

        results[
            "unweighted_block_half_width"
        ].mean(),

        results[
            "oracle_block_half_width"
        ].mean(),

        results[
            "kmm_block_half_width"
        ].mean(),

        results[
            "y_min"
        ].min(),

        results[
            "y_max"
        ].max(),
    ]
})


display(
    gamma0_summary.round(5)
)


gamma0_summary.to_csv(
    OUTPUT_DIR
    / "gamma0_sanity_summary.csv",
    index=False
)


# ============================================================
# 19. KMM GAMMA=0 BANDWIDTH x RIDGE ABLATION
#
# IMPORTANT:
# We use NO outcomes here.
#
# Selection can therefore be based only on:
#
#   - balance (weighted MMD)
#   - weight ESS
#   - max weight
#   - fraction at B / near zero
#
# This avoids tuning on target outcomes.
# ============================================================

print("\n")
print("=" * 100)
print("KMM BANDWIDTH x RIDGE SANITY ABLATION")
print("=" * 100)


bandwidths = [
    1.0,
    2.0,
    4.0,
    8.0
]

ridges = [
    1e-4,
    1e-3,
    1e-2
]

N_SANITY_REPS = 10

sanity_rows = []

p_source_0, p_target_0 = (
    sampling_probabilities(
        0.0
    )
)


for rep in range(
    N_SANITY_REPS
):

    rng = np.random.default_rng(
        SEED
        + 900000
        + rep
    )

    source_idx = rng.choice(
        len(df),
        size=N_SOURCE,
        replace=True,
        p=p_source_0
    )

    target_idx = rng.choice(
        len(df),
        size=N_TARGET,
        replace=True,
        p=p_target_0
    )

    Xs = X[source_idx]
    Xt = X[target_idx]

    for bandwidth in bandwidths:

        # Compute kernels once for each bandwidth.
        Kss_raw = rbf_kernel(
            Xs,
            Xs,
            bandwidth
        )

        Ktt = rbf_kernel(
            Xt,
            Xt,
            bandwidth
        )

        Kst = rbf_kernel(
            Xs,
            Xt,
            bandwidth
        )

        mmd_before = (
            np.mean(Kss_raw)
            + np.mean(Ktt)
            - 2.0
            * np.mean(Kst)
        )

        for ridge in ridges:

            try:

                w = constrained_kmm_weights(
                    Xs,
                    Xt,
                    bandwidth=bandwidth,
                    B=KMM_B,
                    ridge=ridge
                )

                diag = weight_diagnostics(
                    w,
                    B=KMM_B
                )

                # Normalize weights to sum to one for
                # weighted empirical kernel mean.
                wn = (
                    w
                    / np.sum(w)
                )

                # Weighted biased empirical MMD^2.
                mmd_after = (
                    wn
                    @ Kss_raw
                    @ wn

                    + np.mean(Ktt)

                    - 2.0
                    * np.mean(
                        wn @ Kst
                    )
                )

                # Tiny negative values can occur due to
                # numerical precision.
                mmd_after = float(
                    max(
                        mmd_after,
                        0.0
                    )
                )

                sanity_rows.append({

                    "rep":
                        rep,

                    "bandwidth":
                        bandwidth,

                    "ridge":
                        ridge,

                    "ess":
                        diag["ess"],

                    "ess_fraction":
                        diag[
                            "ess_fraction"
                        ],

                    "max_weight":
                        diag[
                            "max_weight"
                        ],

                    "frac_at_B":
                        diag[
                            "frac_at_B"
                        ],

                    "frac_near_zero":
                        diag[
                            "frac_near_zero"
                        ],

                    "mmd_before":
                        mmd_before,

                    "mmd_after":
                        mmd_after,

                    "mmd_reduction":
                        (
                            mmd_before
                            - mmd_after
                        ),
                })

            except Exception as exc:

                print(
                    "FAILED |",
                    "rep =", rep,
                    "| bandwidth =",
                    bandwidth,
                    "| ridge =",
                    ridge,
                    "|",
                    exc
                )


sanity = pd.DataFrame(
    sanity_rows
)


sanity.to_csv(
    OUTPUT_DIR
    / "gamma0_kmm_sanity_replicates.csv",
    index=False
)


sanity_summary = (
    sanity
    .groupby(
        [
            "bandwidth",
            "ridge"
        ],
        as_index=False
    )
    .agg(

        mean_ess=(
            "ess",
            "mean"
        ),

        median_ess=(
            "ess",
            "median"
        ),

        mean_ess_fraction=(
            "ess_fraction",
            "mean"
        ),

        mean_max_weight=(
            "max_weight",
            "mean"
        ),

        mean_frac_at_B=(
            "frac_at_B",
            "mean"
        ),

        mean_frac_near_zero=(
            "frac_near_zero",
            "mean"
        ),

        mean_mmd_before=(
            "mmd_before",
            "mean"
        ),

        mean_mmd_after=(
            "mmd_after",
            "mean"
        ),

        mean_mmd_reduction=(
            "mmd_reduction",
            "mean"
        ),
    )
)


sanity_summary = (
    sanity_summary
    .sort_values(
        [
            "bandwidth",
            "ridge"
        ]
    )
    .reset_index(
        drop=True
    )
)


print("\nAblation summary:\n")

display(
    sanity_summary.round(6)
)


sanity_summary.to_csv(
    OUTPUT_DIR
    / "gamma0_kmm_sanity.csv",
    index=False
)


# ============================================================
# 20. QUICK VISUAL SANITY CHECK
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 4.5)
)

for ridge in ridges:

    temp = sanity_summary[
        sanity_summary[
            "ridge"
        ] == ridge
    ]

    ax.plot(
        temp["bandwidth"],
        temp["mean_ess"],
        marker="o",
        label=f"ridge={ridge:g}"
    )

ax.axhline(
    N_SOURCE,
    linestyle="--",
    label="Nominal n"
)

ax.set_xlabel(
    "KMM bandwidth"
)

ax.set_ylabel(
    "Mean weight ESS at gamma=0"
)

ax.set_title(
    "KMM no-shift sanity check"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "gamma0_bandwidth_vs_ess.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FINAL CHECKS
# ============================================================

print("\n")
print("=" * 100)
print("FINAL SANITY CHECKS")
print("=" * 100)


print(
    "\nOracle ESS at gamma=0:",
    round(
        results[
            "oracle_weight_ess"
        ].mean(),
        3
    ),
    "(expected 256)"
)


print(
    "Oracle max weight at gamma=0:",
    round(
        results[
            "oracle_max_weight"
        ].mean(),
        5
    ),
    "(expected 1)"
)


print(
    "Current KMM ESS at gamma=0:",
    round(
        results[
            "kmm_weight_ess"
        ].mean(),
        3
    )
)


print(
    "\nObserved Y range:",
    round(
        results[
            "y_min"
        ].min(),
        4
    ),
    "to",
    round(
        results[
            "y_max"
        ].max(),
        4
    ),
    "(must remain inside [0,1])"
)


print(
    "\nIMPORTANT:"
)

print(
    "Do NOT run the final 100-repetition experiment yet."
)

print(
    "Send gamma0_kmm_sanity.csv / the displayed "
    "ablation table back to ChatGPT first."
)


# ============================================================
# 22. LIST SAVED FILES
# ============================================================

print("\nSaved files:\n")

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        " -",
        path
    )


print("\nDONE.")

# ============================================================
# FINAL CALIFORNIA CONTROLLED-OVERLAP EXPERIMENT
#
# Semi-synthetic experiment on real California covariates
# ------------------------------------------------------------
# - real X and geographic coordinates
# - known common P(Y|X)
# - bounded spatially dependent residuals
# - exact source/target exponential tilting
# - known oracle density ratio
# - Unweighted vs self-normalized Oracle IW vs KMM
#
# FINAL CONFIGURATION:
#   100 repetitions
#   gamma = {0, .5, 1, 1.5, 2, 3}
#   KMM bandwidth = 8
#   ridge = 0.01
#   B = 10
#
# h and ridge were selected using a gamma=0 sanity ablation
# based ONLY on covariate balance and weight stability,
# without using target outcomes.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 20261005

N_SOURCE = 256
N_TARGET = 256

N_REPS = 100

GAMMAS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0
]

# ------------------------------------------------------------
# Final KMM configuration
# Selected from gamma=0 balance/stability sanity check.
# ------------------------------------------------------------

KMM_BANDWIDTH = 8.0
KMM_B = 10.0
KMM_RIDGE = 1e-2


# ------------------------------------------------------------
# Spatial block bootstrap
# ------------------------------------------------------------

BLOCK_LON_BINS = 5
BLOCK_LAT_BINS = 5

N_BOOTSTRAP = 250
ALPHA = 0.05


# ------------------------------------------------------------
# Outcome construction
#
# MU      in [0.20, 0.80]
# epsilon in [-0.12, 0.12]
# Y       in [0.08, 0.92]
# ------------------------------------------------------------

NOISE_SCALE = 0.12


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "/content/california_controlled_overlap_final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Output directory:", OUTPUT_DIR)


# ============================================================
# 2. LOAD CALIFORNIA HOUSING
# ============================================================

print("\nLoading California Housing...")

data = fetch_california_housing(
    as_frame=True
)

df = data.frame.copy()

feature_cols = list(
    data.feature_names
)

X_raw = df[
    feature_cols
].to_numpy(float)

coords = df[
    ["Longitude", "Latitude"]
].to_numpy(float)

print("Rows:", len(df))
print("Features:", feature_cols)


# ============================================================
# 3. STANDARDIZE COVARIATES
# ============================================================

scaler = StandardScaler().fit(
    X_raw
)

X = scaler.transform(
    X_raw
)

coords_std = StandardScaler().fit_transform(
    coords
)

feature_index = {
    name: i
    for i, name in enumerate(feature_cols)
}


# ============================================================
# 4. KNOWN BOUNDED CONDITIONAL MEAN
# ============================================================

i_income = feature_index["MedInc"]
i_age = feature_index["HouseAge"]
i_rooms = feature_index["AveRooms"]
i_occup = feature_index["AveOccup"]
i_lat = feature_index["Latitude"]
i_lon = feature_index["Longitude"]


def outcome_mean_function(X):

    income = X[:, i_income]
    age = X[:, i_age]
    rooms = X[:, i_rooms]
    occup = X[:, i_occup]
    lat = X[:, i_lat]
    lon = X[:, i_lon]

    eta = (
        0.75 * income
        + 0.25 * np.sin(age)
        + 0.20 * np.tanh(rooms)
        - 0.15 * np.tanh(occup)
        + 0.20 * np.sin(lat)
        + 0.20 * np.cos(lon)
        + 0.12 * income * rooms
    )

    return (
        0.20
        + 0.60 * expit(eta)
    )


MU = outcome_mean_function(X)

print("\nConditional mean range:")
print(
    round(float(MU.min()), 4),
    "to",
    round(float(MU.max()), 4)
)


# ============================================================
# 5. OVERLAP SCORE g(X)
# ============================================================

income = X[:, i_income]
lat = X[:, i_lat]
rooms = X[:, i_rooms]

g_raw = (
    0.75 * income
    + 0.45 * lat
    + 0.25 * rooms
)

g = (
    g_raw - np.mean(g_raw)
) / np.std(g_raw)

g = np.clip(
    g,
    -3.0,
    3.0
)

print("\nOverlap score range:")
print(
    round(float(g.min()), 3),
    "to",
    round(float(g.max()), 3)
)


# ============================================================
# 6. SOURCE / TARGET DISTRIBUTIONS
#
# qS(x) proportional to q0(x) exp(-gamma*g/2)
# qT(x) proportional to q0(x) exp(+gamma*g/2)
#
# Thus qT/qS is proportional to exp(gamma*g).
# ============================================================

def sampling_probabilities(gamma):

    log_target = (
        +0.5 * gamma * g
    )

    log_source = (
        -0.5 * gamma * g
    )

    log_target -= np.max(
        log_target
    )

    log_source -= np.max(
        log_source
    )

    p_target = np.exp(
        log_target
    )

    p_source = np.exp(
        log_source
    )

    p_target /= np.sum(
        p_target
    )

    p_source /= np.sum(
        p_source
    )

    return (
        p_source,
        p_target
    )


# ============================================================
# 7. ORACLE IMPORTANCE WEIGHTS
# ============================================================

def oracle_weights(
    source_idx,
    p_source,
    p_target
):

    w = (
        p_target[source_idx]
        / p_source[source_idx]
    )

    # Normalization does not change the self-normalized
    # estimator but makes diagnostics easier to interpret.
    w = (
        w
        / np.mean(w)
    )

    return w


# ============================================================
# 8. RBF KERNEL
# ============================================================

def rbf_kernel(
    x,
    y,
    bandwidth
):

    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )

    return np.exp(
        -d2
        / (2.0 * bandwidth**2)
    )


# ============================================================
# 9. CONSTRAINED KMM
#
# min 0.5 w'Kw - kappa'w
#
# subject to:
#     0 <= w_i <= B
#     sum(w_i) = n
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth=8.0,
    B=10.0,
    ridge=1e-2
):

    n = x_source.shape[0]
    m = x_target.shape[0]

    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    K = (
        0.5 * (K + K.T)
        + ridge * np.eye(n)
    )

    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    kappa = (
        n / m
    ) * np.sum(
        K_st,
        axis=1
    )

    def objective(w):

        return (
            0.5 * w @ K @ w
            - kappa @ w
        )

    def gradient(w):

        return (
            K @ w
            - kappa
        )

    constraint = {
        "type": "eq",
        "fun":
            lambda w:
                np.sum(w) - n,
        "jac":
            lambda w:
                np.ones_like(w),
    }

    bounds = [
        (0.0, B)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        np.ones(n),
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    w = np.asarray(
        result.x,
        dtype=float
    )

    if (
        abs(np.sum(w) - n)
        > 1e-4
    ):
        raise RuntimeError(
            "KMM sum constraint violated."
        )

    if np.min(w) < -1e-5:
        raise RuntimeError(
            "Negative KMM weight."
        )

    if (
        np.max(w)
        > B + 1e-4
    ):
        raise RuntimeError(
            "KMM upper bound violated."
        )

    return w


# ============================================================
# 10. WEIGHT DIAGNOSTICS
# ============================================================

def weight_diagnostics(
    w,
    B=None
):

    w = np.asarray(
        w,
        dtype=float
    )

    ess = (
        np.sum(w)**2
        / np.sum(w**2)
    )

    out = {

        "ess":
            float(ess),

        "ess_fraction":
            float(
                ess / len(w)
            ),

        "max_weight":
            float(
                np.max(w)
            ),

        "weight_sd":
            float(
                np.std(w)
            ),

        "frac_near_zero":
            float(
                np.mean(
                    w <= 1e-4
                )
            ),
    }

    if B is None:

        out["frac_at_B"] = np.nan

    else:

        out["frac_at_B"] = float(
            np.mean(
                w >= B - 1e-4
            )
        )

    return out


# ============================================================
# 11. EMPIRICAL MMD^2
# ============================================================

def empirical_mmd2(
    Xs,
    Xt,
    bandwidth
):

    Kss = rbf_kernel(
        Xs,
        Xs,
        bandwidth
    )

    Ktt = rbf_kernel(
        Xt,
        Xt,
        bandwidth
    )

    Kst = rbf_kernel(
        Xs,
        Xt,
        bandwidth
    )

    value = (
        np.mean(Kss)
        + np.mean(Ktt)
        - 2.0 * np.mean(Kst)
    )

    return float(
        max(value, 0.0)
    )


# ============================================================
# 12. BOUNDED SPATIAL RANDOM FIELD
# ============================================================

def generate_spatial_noise(
    rng,
    noise_scale=0.12,
    n_components=30
):

    frequencies = rng.normal(
        0.0,
        0.70,
        size=(n_components, 2)
    )

    a = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    b = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    projection = (
        coords_std
        @ frequencies.T
    )

    field = (
        np.cos(projection) @ a
        + np.sin(projection) @ b
    ) / np.sqrt(
        2.0 * n_components
    )

    return (
        noise_scale
        * np.tanh(field)
    )


# ============================================================
# 13. SELF-NORMALIZED WEIGHTED MEAN
# ============================================================

def weighted_mean(
    y,
    w
):

    y = np.asarray(
        y,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    denom = np.sum(w)

    if denom <= 1e-12:
        return np.nan

    return float(
        np.sum(w * y)
        / denom
    )


# ============================================================
# 14. IID HALF-WIDTH
#
# Influence:
#
# psi_i = w_i (Y_i - mu_hat)
#
# SE =
# sd(psi) / [sqrt(n) mean(w)]
# ============================================================

def iid_half_width(
    y,
    w,
    alpha=0.05
):

    y = np.asarray(
        y,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )

    estimate = weighted_mean(
        y,
        w
    )

    mean_w = np.mean(w)

    if mean_w <= 1e-12:
        return np.nan

    psi = (
        w
        * (y - estimate)
    )

    se = (
        np.std(
            psi,
            ddof=1
        )
        / (
            np.sqrt(len(y))
            * mean_w
        )
    )

    # Experiment uses 95% intervals.
    zcrit = 1.959963984540054

    return float(
        zcrit * se
    )


# ============================================================
# 15. SPATIAL BLOCK IDS
# ============================================================

def spatial_block_ids(
    coordinates,
    n_lon_bins=5,
    n_lat_bins=5
):

    lon = coordinates[:, 0]
    lat = coordinates[:, 1]

    lon_edges = np.quantile(
        lon,
        np.linspace(
            0,
            1,
            n_lon_bins + 1
        )
    )

    lat_edges = np.quantile(
        lat,
        np.linspace(
            0,
            1,
            n_lat_bins + 1
        )
    )

    lon_edges[0] -= 1e-9
    lon_edges[-1] += 1e-9

    lat_edges[0] -= 1e-9
    lat_edges[-1] += 1e-9

    lon_id = np.digitize(
        lon,
        lon_edges[1:-1]
    )

    lat_id = np.digitize(
        lat,
        lat_edges[1:-1]
    )

    return (
        lon_id * n_lat_bins
        + lat_id
    )


# ============================================================
# 16. SPATIAL BLOCK BOOTSTRAP
#
# Fixed fitted weights.
#
# Each bootstrap estimate is recomputed as:
#
#     sum(wY) / sum(w)
#
# rather than dividing by the original n.
# ============================================================

def block_bootstrap_half_width(
    rng,
    values,
    weights,
    block_ids,
    n_boot=250,
    alpha=0.05
):

    values = np.asarray(
        values,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    block_ids = np.asarray(
        block_ids
    )

    unique_blocks = np.unique(
        block_ids
    )

    mapping = {
        b: np.flatnonzero(
            block_ids == b
        )
        for b in unique_blocks
    }

    estimates = np.empty(
        n_boot
    )

    for k in range(n_boot):

        selected = rng.choice(
            unique_blocks,
            size=len(unique_blocks),
            replace=True
        )

        idx = np.concatenate([
            mapping[b]
            for b in selected
        ])

        estimates[k] = weighted_mean(
            values[idx],
            weights[idx]
        )

    estimates = estimates[
        np.isfinite(estimates)
    ]

    if len(estimates) < 10:
        return np.nan

    lo, hi = np.quantile(
        estimates,
        [
            alpha / 2,
            1 - alpha / 2
        ]
    )

    return float(
        (hi - lo) / 2.0
    )


# ============================================================
# 17. MAIN EXPERIMENT
# ============================================================

rows = []

total_runs = (
    N_REPS
    * len(GAMMAS)
)

run_counter = 0


print("\n")
print("=" * 90)
print("RUNNING FINAL EXPERIMENT")
print("=" * 90)
print(
    f"{N_REPS} repetitions x "
    f"{len(GAMMAS)} gamma levels = "
    f"{total_runs} runs\n"
)


for rep in range(
    N_REPS
):

    # --------------------------------------------------------
    # One spatial random field per repetition.
    #
    # The same realization is used across gamma values within
    # a repetition, making overlap comparisons less noisy.
    # --------------------------------------------------------

    noise_rng = np.random.default_rng(
        SEED
        + 100000
        + rep
    )

    epsilon = generate_spatial_noise(
        noise_rng,
        noise_scale=NOISE_SCALE
    )

    Y = (
        MU
        + epsilon
    )

    if (
        np.min(Y) < -1e-12
        or np.max(Y) > 1.0 + 1e-12
    ):
        raise RuntimeError(
            "Outcome left [0,1]."
        )

    for gamma_index, gamma in enumerate(
        GAMMAS
    ):

        run_counter += 1

        # ----------------------------------------------------
        # Sampling distributions
        # ----------------------------------------------------

        p_source, p_target = (
            sampling_probabilities(
                gamma
            )
        )

        sampling_rng = np.random.default_rng(
            SEED
            + rep * 1000
            + gamma_index
        )

        # Independent source and target samples
        # drawn with replacement.
        source_idx = sampling_rng.choice(
            len(df),
            size=N_SOURCE,
            replace=True,
            p=p_source
        )

        target_idx = sampling_rng.choice(
            len(df),
            size=N_TARGET,
            replace=True,
            p=p_target
        )

        Xs = X[source_idx]
        Xt = X[target_idx]

        ys = Y[source_idx]

        coords_s = coords[
            source_idx
        ]

        # ----------------------------------------------------
        # Known population target estimand
        #
        # Spatial residual has zero expectation across
        # realizations, hence target truth is E_qT[MU(X)].
        # ----------------------------------------------------

        truth = float(
            np.sum(
                p_target
                * MU
            )
        )

        # ----------------------------------------------------
        # WEIGHTS
        # ----------------------------------------------------

        w_un = np.ones(
            N_SOURCE
        )

        w_oracle = oracle_weights(
            source_idx,
            p_source,
            p_target
        )

        try:

            w_kmm = constrained_kmm_weights(
                Xs,
                Xt,
                bandwidth=KMM_BANDWIDTH,
                B=KMM_B,
                ridge=KMM_RIDGE
            )

        except Exception as exc:

            print(
                f"FAILED KMM | "
                f"rep={rep} | "
                f"gamma={gamma} | "
                f"{exc}"
            )

            continue

        # ----------------------------------------------------
        # OVERLAP / WEIGHT DIAGNOSTICS
        # ----------------------------------------------------

        mmd2 = empirical_mmd2(
            Xs,
            Xt,
            KMM_BANDWIDTH
        )

        oracle_diag = weight_diagnostics(
            w_oracle
        )

        kmm_diag = weight_diagnostics(
            w_kmm,
            B=KMM_B
        )

        # ----------------------------------------------------
        # POINT ESTIMATES
        # ----------------------------------------------------

        est_un = weighted_mean(
            ys,
            w_un
        )

        est_oracle = weighted_mean(
            ys,
            w_oracle
        )

        est_kmm = weighted_mean(
            ys,
            w_kmm
        )

        error_un = (
            est_un - truth
        )

        error_oracle = (
            est_oracle - truth
        )

        error_kmm = (
            est_kmm - truth
        )

        # ----------------------------------------------------
        # IID INTERVALS
        # ----------------------------------------------------

        iid_un = iid_half_width(
            ys,
            w_un,
            ALPHA
        )

        iid_oracle = iid_half_width(
            ys,
            w_oracle,
            ALPHA
        )

        iid_kmm = iid_half_width(
            ys,
            w_kmm,
            ALPHA
        )

        # ----------------------------------------------------
        # SPATIAL BLOCK BOOTSTRAP
        # ----------------------------------------------------

        blocks = spatial_block_ids(
            coords_s,
            BLOCK_LON_BINS,
            BLOCK_LAT_BINS
        )

        boot_base = (
            SEED
            + 5000000
            + rep * 10000
            + gamma_index * 100
        )

        block_un = block_bootstrap_half_width(
            np.random.default_rng(
                boot_base + 1
            ),
            ys,
            w_un,
            blocks,
            N_BOOTSTRAP,
            ALPHA
        )

        block_oracle = block_bootstrap_half_width(
            np.random.default_rng(
                boot_base + 2
            ),
            ys,
            w_oracle,
            blocks,
            N_BOOTSTRAP,
            ALPHA
        )

        block_kmm = block_bootstrap_half_width(
            np.random.default_rng(
                boot_base + 3
            ),
            ys,
            w_kmm,
            blocks,
            N_BOOTSTRAP,
            ALPHA
        )

        # ----------------------------------------------------
        # SAVE REPLICATE
        # ----------------------------------------------------

        rows.append({

            "rep":
                rep,

            "gamma":
                gamma,

            "truth":
                truth,

            "feature_mmd2":
                mmd2,

            # -----------------------------------------------
            # Point estimates
            # -----------------------------------------------

            "unweighted_estimate":
                est_un,

            "oracle_estimate":
                est_oracle,

            "kmm_estimate":
                est_kmm,

            # -----------------------------------------------
            # Signed errors
            # -----------------------------------------------

            "unweighted_error":
                error_un,

            "oracle_error":
                error_oracle,

            "kmm_error":
                error_kmm,

            # -----------------------------------------------
            # Absolute errors
            # -----------------------------------------------

            "unweighted_abs_error":
                abs(error_un),

            "oracle_abs_error":
                abs(error_oracle),

            "kmm_abs_error":
                abs(error_kmm),

            # -----------------------------------------------
            # Oracle weight diagnostics
            # -----------------------------------------------

            "oracle_weight_ess":
                oracle_diag["ess"],

            "oracle_ess_fraction":
                oracle_diag[
                    "ess_fraction"
                ],

            "oracle_max_weight":
                oracle_diag[
                    "max_weight"
                ],

            # -----------------------------------------------
            # KMM weight diagnostics
            # -----------------------------------------------

            "kmm_weight_ess":
                kmm_diag["ess"],

            "kmm_ess_fraction":
                kmm_diag[
                    "ess_fraction"
                ],

            "kmm_max_weight":
                kmm_diag[
                    "max_weight"
                ],

            "kmm_frac_at_B":
                kmm_diag[
                    "frac_at_B"
                ],

            "kmm_frac_near_zero":
                kmm_diag[
                    "frac_near_zero"
                ],

            # -----------------------------------------------
            # IID uncertainty
            # -----------------------------------------------

            "unweighted_iid_half_width":
                iid_un,

            "oracle_iid_half_width":
                iid_oracle,

            "kmm_iid_half_width":
                iid_kmm,

            "unweighted_iid_covered":
                int(
                    abs(error_un)
                    <= iid_un
                ),

            "oracle_iid_covered":
                int(
                    abs(error_oracle)
                    <= iid_oracle
                ),

            "kmm_iid_covered":
                int(
                    abs(error_kmm)
                    <= iid_kmm
                ),

            # -----------------------------------------------
            # Block uncertainty
            # -----------------------------------------------

            "unweighted_block_half_width":
                block_un,

            "oracle_block_half_width":
                block_oracle,

            "kmm_block_half_width":
                block_kmm,

            "unweighted_block_covered":
                int(
                    abs(error_un)
                    <= block_un
                ),

            "oracle_block_covered":
                int(
                    abs(error_oracle)
                    <= block_oracle
                ),

            "kmm_block_covered":
                int(
                    abs(error_kmm)
                    <= block_kmm
                ),

            # -----------------------------------------------
            # Boundedness diagnostics
            # -----------------------------------------------

            "y_min":
                float(
                    np.min(Y)
                ),

            "y_max":
                float(
                    np.max(Y)
                ),
        })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            run_counter % 10 == 0
            or run_counter == total_runs
        ):

            print(
                f"{run_counter:3d}/"
                f"{total_runs} | "
                f"rep={rep:02d} | "
                f"gamma={gamma:.1f} | "
                f"oracle ESS="
                f"{oracle_diag['ess']:.1f} | "
                f"KMM ESS="
                f"{kmm_diag['ess']:.1f} | "
                f"KMM |err|="
                f"{abs(error_kmm):.4f}"
            )


# ============================================================
# 18. REPLICATE DATA
# ============================================================

results = pd.DataFrame(
    rows
)

if len(results) == 0:
    raise RuntimeError(
        "No successful runs."
    )


replicate_path = (
    OUTPUT_DIR
    / "california_final_replicates.csv"
)

results.to_csv(
    replicate_path,
    index=False
)


print("\nSuccessful runs:", len(results))
print("Expected runs:", total_runs)


# ============================================================
# 19. SUMMARY BY GAMMA
# ============================================================

summary_rows = []


for gamma in GAMMAS:

    temp = results[
        results["gamma"] == gamma
    ].copy()

    n_success = len(temp)

    if n_success == 0:
        continue

    summary_rows.append({

        "gamma":
            gamma,

        "n_reps":
            n_success,

        # -----------------------------------------------
        # Shift diagnostic
        # -----------------------------------------------

        "mean_mmd2":
            temp[
                "feature_mmd2"
            ].mean(),

        "se_mmd2":
            temp[
                "feature_mmd2"
            ].std(ddof=1)
            / np.sqrt(n_success),

        # -----------------------------------------------
        # Oracle weights
        # -----------------------------------------------

        "oracle_ess":
            temp[
                "oracle_weight_ess"
            ].mean(),

        "oracle_ess_se":
            temp[
                "oracle_weight_ess"
            ].std(ddof=1)
            / np.sqrt(n_success),

        "oracle_ess_fraction":
            temp[
                "oracle_ess_fraction"
            ].mean(),

        "oracle_max_weight":
            temp[
                "oracle_max_weight"
            ].mean(),

        # -----------------------------------------------
        # KMM weights
        # -----------------------------------------------

        "kmm_ess":
            temp[
                "kmm_weight_ess"
            ].mean(),

        "kmm_ess_se":
            temp[
                "kmm_weight_ess"
            ].std(ddof=1)
            / np.sqrt(n_success),

        "kmm_ess_fraction":
            temp[
                "kmm_ess_fraction"
            ].mean(),

        "kmm_max_weight":
            temp[
                "kmm_max_weight"
            ].mean(),

        "kmm_frac_at_B":
            temp[
                "kmm_frac_at_B"
            ].mean(),

        "kmm_frac_near_zero":
            temp[
                "kmm_frac_near_zero"
            ].mean(),

        # -----------------------------------------------
        # Bias
        # -----------------------------------------------

        "unweighted_bias":
            temp[
                "unweighted_error"
            ].mean(),

        "oracle_bias":
            temp[
                "oracle_error"
            ].mean(),

        "kmm_bias":
            temp[
                "kmm_error"
            ].mean(),

        # -----------------------------------------------
        # MAE
        # -----------------------------------------------

        "unweighted_mae":
            temp[
                "unweighted_abs_error"
            ].mean(),

        "oracle_mae":
            temp[
                "oracle_abs_error"
            ].mean(),

        "kmm_mae":
            temp[
                "kmm_abs_error"
            ].mean(),

        # SE of replicate absolute error
        "unweighted_mae_se":
            temp[
                "unweighted_abs_error"
            ].std(ddof=1)
            / np.sqrt(n_success),

        "oracle_mae_se":
            temp[
                "oracle_abs_error"
            ].std(ddof=1)
            / np.sqrt(n_success),

        "kmm_mae_se":
            temp[
                "kmm_abs_error"
            ].std(ddof=1)
            / np.sqrt(n_success),

        # -----------------------------------------------
        # IID coverage
        # -----------------------------------------------

        "unweighted_iid_coverage":
            temp[
                "unweighted_iid_covered"
            ].mean(),

        "oracle_iid_coverage":
            temp[
                "oracle_iid_covered"
            ].mean(),

        "kmm_iid_coverage":
            temp[
                "kmm_iid_covered"
            ].mean(),

        # -----------------------------------------------
        # Block coverage
        # -----------------------------------------------

        "unweighted_block_coverage":
            temp[
                "unweighted_block_covered"
            ].mean(),

        "oracle_block_coverage":
            temp[
                "oracle_block_covered"
            ].mean(),

        "kmm_block_coverage":
            temp[
                "kmm_block_covered"
            ].mean(),

        # -----------------------------------------------
        # IID half-width
        # -----------------------------------------------

        "unweighted_iid_hw":
            temp[
                "unweighted_iid_half_width"
            ].mean(),

        "oracle_iid_hw":
            temp[
                "oracle_iid_half_width"
            ].mean(),

        "kmm_iid_hw":
            temp[
                "kmm_iid_half_width"
            ].mean(),

        # -----------------------------------------------
        # Block half-width
        # -----------------------------------------------

        "unweighted_block_hw":
            temp[
                "unweighted_block_half_width"
            ].mean(),

        "oracle_block_hw":
            temp[
                "oracle_block_half_width"
            ].mean(),

        "kmm_block_hw":
            temp[
                "kmm_block_half_width"
            ].mean(),
    })


summary = pd.DataFrame(
    summary_rows
)


summary_path = (
    OUTPUT_DIR
    / "california_final_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 20. CORRELATIONS ACROSS ALL REPLICATES
# ============================================================

def safe_spearman(
    x,
    y
):

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    rho, p = spearmanr(
        np.asarray(x)[mask],
        np.asarray(y)[mask]
    )

    return (
        float(rho),
        float(p)
    )


correlation_specs = [

    (
        "gamma_vs_mmd2",
        results["gamma"],
        results["feature_mmd2"]
    ),

    (
        "gamma_vs_oracle_ess",
        results["gamma"],
        results["oracle_weight_ess"]
    ),

    (
        "gamma_vs_kmm_ess",
        results["gamma"],
        results["kmm_weight_ess"]
    ),

    (
        "gamma_vs_oracle_abs_error",
        results["gamma"],
        results["oracle_abs_error"]
    ),

    (
        "gamma_vs_kmm_abs_error",
        results["gamma"],
        results["kmm_abs_error"]
    ),

    (
        "oracle_ess_vs_oracle_abs_error",
        results["oracle_weight_ess"],
        results["oracle_abs_error"]
    ),

    (
        "kmm_ess_vs_kmm_abs_error",
        results["kmm_weight_ess"],
        results["kmm_abs_error"]
    ),
]


corr_rows = []


for (
    name,
    x,
    y
) in correlation_specs:

    rho, p = safe_spearman(
        x,
        y
    )

    corr_rows.append({

        "comparison":
            name,

        "spearman_rho":
            rho,

        "p_value":
            p,
    })


correlations = pd.DataFrame(
    corr_rows
)


correlation_path = (
    OUTPUT_DIR
    / "california_correlations.csv"
)

correlations.to_csv(
    correlation_path,
    index=False
)


# ============================================================
# 21. DISPLAY COMPACT MAIN RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("FINAL CALIFORNIA SUMMARY")
print("=" * 100)


compact_cols = [

    "gamma",
    "n_reps",
    "mean_mmd2",

    "oracle_ess",
    "kmm_ess",

    "oracle_max_weight",
    "kmm_max_weight",

    "unweighted_mae",
    "oracle_mae",
    "kmm_mae",
]


display(
    summary[
        compact_cols
    ].round(4)
)


print("\n")
print("=" * 100)
print("COVERAGE / HALF-WIDTH")
print("=" * 100)


coverage_cols = [

    "gamma",

    "unweighted_iid_coverage",
    "oracle_iid_coverage",
    "kmm_iid_coverage",

    "unweighted_block_coverage",
    "oracle_block_coverage",
    "kmm_block_coverage",

    "unweighted_iid_hw",
    "oracle_iid_hw",
    "kmm_iid_hw",

    "unweighted_block_hw",
    "oracle_block_hw",
    "kmm_block_hw",
]


display(
    summary[
        coverage_cols
    ].round(4)
)


print("\n")
print("=" * 100)
print("SPEARMAN CORRELATIONS")
print("=" * 100)


display(
    correlations.round(5)
)


# ============================================================
# 22. FINAL MAIN-TEXT FIGURE
#
# Panel A:
#   overlap severity -> weight ESS
#
# Panel B:
#   overlap severity -> target-mean MAE
#
# Error bars = +/- one SE across repetitions.
# ============================================================

plt.rcParams.update({

    "font.size": 8,

    "axes.labelsize": 8,

    "axes.titlesize": 9,

    "legend.fontsize": 7,

    "xtick.labelsize": 7,

    "ytick.labelsize": 7,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,
})


fig, axes = plt.subplots(
    1,
    2,
    figsize=(6.8, 2.45),
    constrained_layout=True
)


# ------------------------------------------------------------
# Panel A: Weight ESS
# ------------------------------------------------------------

ax = axes[0]


ax.errorbar(
    summary["gamma"],
    summary["oracle_ess"],
    yerr=summary["oracle_ess_se"],
    marker="o",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    label="Oracle IW"
)


ax.errorbar(
    summary["gamma"],
    summary["kmm_ess"],
    yerr=summary["kmm_ess_se"],
    marker="s",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    label="KMM"
)


ax.axhline(
    N_SOURCE,
    linestyle=":",
    linewidth=1,
    label="Nominal $n$"
)


ax.set_xlabel(
    r"Overlap severity $\gamma$"
)

ax.set_ylabel(
    "Weight ESS"
)

ax.set_title(
    "(a) Weight concentration"
)

ax.set_xticks(
    GAMMAS
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    alpha=0.2
)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# Panel B: MAE
# ------------------------------------------------------------

ax = axes[1]


ax.errorbar(
    summary["gamma"],
    summary["unweighted_mae"],
    yerr=summary["unweighted_mae_se"],
    marker="^",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    label="Unweighted"
)


ax.errorbar(
    summary["gamma"],
    summary["oracle_mae"],
    yerr=summary["oracle_mae_se"],
    marker="o",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    label="Oracle IW"
)


ax.errorbar(
    summary["gamma"],
    summary["kmm_mae"],
    yerr=summary["kmm_mae_se"],
    marker="s",
    linewidth=1.5,
    markersize=4,
    capsize=2,
    label="KMM"
)


ax.set_xlabel(
    r"Overlap severity $\gamma$"
)

ax.set_ylabel(
    "Target-mean MAE"
)

ax.set_title(
    "(b) Estimation error"
)

ax.set_xticks(
    GAMMAS
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    alpha=0.2
)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# Save figure
# ------------------------------------------------------------

figure_pdf = (
    OUTPUT_DIR
    / "california_overlap_stress.pdf"
)

figure_png = (
    OUTPUT_DIR
    / "california_overlap_stress.png"
)


plt.savefig(
    figure_pdf,
    bbox_inches="tight"
)

plt.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 23. SUPPLEMENT TABLE CSV
#
# A deliberately compact table for the paper.
# ============================================================

supplement_table = summary[[

    "gamma",

    "mean_mmd2",

    "oracle_ess",

    "kmm_ess",

    "oracle_max_weight",

    "kmm_max_weight",

    "kmm_frac_at_B",

    "unweighted_mae",

    "oracle_mae",

    "kmm_mae",

]].copy()


supplement_table.to_csv(
    OUTPUT_DIR
    / "california_supplement_table.csv",
    index=False
)


# ============================================================
# 24. AUTOMATIC SANITY CHECKS
# ============================================================

print("\n")
print("=" * 100)
print("FINAL SANITY CHECKS")
print("=" * 100)


gamma0 = summary.loc[
    summary["gamma"] == 0.0
].iloc[0]


print(
    "\nAt gamma=0:"
)

print(
    "Oracle ESS =",
    round(
        gamma0[
            "oracle_ess"
        ],
        2
    ),
    "/",
    N_SOURCE
)

print(
    "KMM ESS =",
    round(
        gamma0[
            "kmm_ess"
        ],
        2
    ),
    "/",
    N_SOURCE
)

print(
    "Unweighted MAE =",
    round(
        gamma0[
            "unweighted_mae"
        ],
        5
    )
)

print(
    "Oracle MAE =",
    round(
        gamma0[
            "oracle_mae"
        ],
        5
    )
)

print(
    "KMM MAE =",
    round(
        gamma0[
            "kmm_mae"
        ],
        5
    )
)


print(
    "\nOutcome range over all runs:"
)

print(
    round(
        results[
            "y_min"
        ].min(),
        4
    ),
    "to",
    round(
        results[
            "y_max"
        ].max(),
        4
    )
)


print(
    "\nKMM bound diagnostics:"
)

print(
    "Largest observed KMM weight =",
    round(
        results[
            "kmm_max_weight"
        ].max(),
        4
    )
)

print(
    "Maximum fraction at B =",
    round(
        results[
            "kmm_frac_at_B"
        ].max(),
        4
    )
)


# ============================================================
# 25. SAVE README WITH CONFIGURATION
# ============================================================

readme = f"""
FINAL CALIFORNIA CONTROLLED-OVERLAP EXPERIMENT

Seed: {SEED}

N source: {N_SOURCE}
N target: {N_TARGET}
Repetitions per gamma: {N_REPS}

Gamma:
{GAMMAS}

KMM:
bandwidth = {KMM_BANDWIDTH}
ridge = {KMM_RIDGE}
B = {KMM_B}

The bandwidth/ridge configuration was selected using a separate
gamma=0 sanity ablation based only on covariate balance and
weight stability, without using target outcomes.

Outcome:
MU = 0.20 + 0.60 * expit(eta)
bounded spatial residual scale = {NOISE_SCALE}

Sampling:
source and target sampled independently with replacement from
the corresponding exponentially tilted empirical distributions.

Oracle:
self-normalized exact discrete population importance ratio.

Intervals:
IID weighted influence approximation uses
w_i * (Y_i - estimate).

Spatial block bootstrap recomputes the self-normalized weighted
ratio within each resampled set of blocks.

IMPORTANT:
The block bootstrap holds fitted weights fixed and therefore
does not propagate KMM weight-estimation uncertainty.
"""


with open(
    OUTPUT_DIR / "README.txt",
    "w"
) as f:

    f.write(
        readme
    )


# ============================================================
# 26. ZIP EVERYTHING
# ============================================================

import shutil


zip_path = shutil.make_archive(
    "/content/california_controlled_overlap_final",
    "zip",
    OUTPUT_DIR
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)


print("\nSaved:")

print(
    "\n1.",
    replicate_path
)

print(
    "2.",
    summary_path
)

print(
    "3.",
    correlation_path
)

print(
    "4.",
    figure_pdf
)

print(
    "5.",
    figure_png
)

print(
    "6.",
    OUTPUT_DIR
    / "california_supplement_table.csv"
)

print(
    "7.",
    OUTPUT_DIR
    / "README.txt"
)

print(
    "\nZIP:",
    zip_path
)


print(
    "\nWhen finished, send me:"
)

print(
    "  - california_final_summary.csv"
)

print(
    "  - california_correlations.csv"
)

print(
    "  - california_overlap_stress.png"
)

print(
    "\nThen we will replace the provisional California "
    "numbers/text in the paper with the final results."
)

# ============================================================
# CALIFORNIA: TARGETED B-SWEEP
#
# Purpose:
#   Examine overlap x KMM weight-bound trade-off.
#
#   gamma controls source-target overlap.
#   B controls the admissible KMM weight magnitude.
#
# We record:
#   - population B*(gamma) = max_x qT(x)/qS(x)
#   - B / B*
#   - pre-weighting MMD
#   - post-KMM MMD
#   - post-oracle MMD
#   - weight ESS
#   - bias
#   - SD
#   - MAE
#   - RMSE
#
# NO target outcomes are used for fitting/tuning.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 20261006

N_SOURCE = 256
N_TARGET = 256

N_REPS = 50

GAMMAS = [
    0.0,
    1.0,
    2.0,
    3.0
]

B_VALUES = [
    5.0,
    10.0,
    20.0,
    50.0,
    100.0
]

KMM_BANDWIDTH = 8.0
KMM_RIDGE = 1e-2

NOISE_SCALE = 0.12

OUTPUT_DIR = Path(
    "/content/california_B_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Output:", OUTPUT_DIR)


# ============================================================
# 2. DATA
# ============================================================

data = fetch_california_housing(
    as_frame=True
)

df = data.frame.copy()

feature_cols = list(
    data.feature_names
)

X_raw = df[
    feature_cols
].to_numpy(float)

coords = df[
    ["Longitude", "Latitude"]
].to_numpy(float)


scaler = StandardScaler().fit(
    X_raw
)

X = scaler.transform(
    X_raw
)

coords_std = StandardScaler().fit_transform(
    coords
)

feature_index = {
    name: i
    for i, name in enumerate(feature_cols)
}


# ============================================================
# 3. KNOWN OUTCOME MECHANISM
# ============================================================

i_income = feature_index["MedInc"]
i_age = feature_index["HouseAge"]
i_rooms = feature_index["AveRooms"]
i_occup = feature_index["AveOccup"]
i_lat = feature_index["Latitude"]
i_lon = feature_index["Longitude"]


def outcome_mean_function(X):

    income = X[:, i_income]
    age = X[:, i_age]
    rooms = X[:, i_rooms]
    occup = X[:, i_occup]
    lat = X[:, i_lat]
    lon = X[:, i_lon]

    eta = (
        0.75 * income
        + 0.25 * np.sin(age)
        + 0.20 * np.tanh(rooms)
        - 0.15 * np.tanh(occup)
        + 0.20 * np.sin(lat)
        + 0.20 * np.cos(lon)
        + 0.12 * income * rooms
    )

    return (
        0.20
        + 0.60 * expit(eta)
    )


MU = outcome_mean_function(X)


# ============================================================
# 4. OVERLAP SCORE
# ============================================================

g_raw = (
    0.75 * X[:, i_income]
    + 0.45 * X[:, i_lat]
    + 0.25 * X[:, i_rooms]
)

g = (
    g_raw - np.mean(g_raw)
) / np.std(g_raw)

g = np.clip(
    g,
    -3.0,
    3.0
)


# ============================================================
# 5. SOURCE/TARGET DISTRIBUTIONS
# ============================================================

def sampling_probabilities(gamma):

    log_target = (
        +0.5 * gamma * g
    )

    log_source = (
        -0.5 * gamma * g
    )

    log_target -= np.max(
        log_target
    )

    log_source -= np.max(
        log_source
    )

    p_target = np.exp(
        log_target
    )

    p_source = np.exp(
        log_source
    )

    p_target /= np.sum(
        p_target
    )

    p_source /= np.sum(
        p_source
    )

    return (
        p_source,
        p_target
    )


# ============================================================
# 6. POPULATION B*(gamma)
#
# IMPORTANT:
# This is computed over ALL 20,640 empirical support points:
#
#       B* = max_x qT(x)/qS(x)
#
# It is NOT sampled max oracle weight.
# ============================================================

population_info = {}


for gamma in GAMMAS:

    pS, pT = sampling_probabilities(
        gamma
    )

    ratio = (
        pT / pS
    )

    population_info[gamma] = {

        "p_source":
            pS,

        "p_target":
            pT,

        "ratio":
            ratio,

        "B_star":
            float(
                np.max(ratio)
            ),

        "ratio_min":
            float(
                np.min(ratio)
            ),

        "target_truth":
            float(
                np.sum(
                    pT * MU
                )
            ),
    }


print("\nPopulation overlap quantities:")

for gamma in GAMMAS:

    info = population_info[
        gamma
    ]

    print(
        f"gamma={gamma:.1f} | "
        f"B*={info['B_star']:.3f} | "
        f"ratio min={info['ratio_min']:.6f}"
    )


# ============================================================
# 7. KERNEL
# ============================================================

def rbf_kernel(
    x,
    y,
    bandwidth
):

    d2 = pairwise_distances(
        x,
        y,
        metric="sqeuclidean"
    )

    return np.exp(
        -d2
        / (2.0 * bandwidth**2)
    )


# ============================================================
# 8. KMM
# ============================================================

def constrained_kmm_weights(
    x_source,
    x_target,
    bandwidth,
    B,
    ridge
):

    n = len(x_source)
    m = len(x_target)

    K = rbf_kernel(
        x_source,
        x_source,
        bandwidth
    )

    K = (
        0.5 * (K + K.T)
        + ridge * np.eye(n)
    )

    K_st = rbf_kernel(
        x_source,
        x_target,
        bandwidth
    )

    kappa = (
        n / m
    ) * np.sum(
        K_st,
        axis=1
    )

    def objective(w):

        return (
            0.5 * w @ K @ w
            - kappa @ w
        )

    def gradient(w):

        return (
            K @ w
            - kappa
        )

    constraint = {

        "type": "eq",

        "fun":
            lambda w:
                np.sum(w) - n,

        "jac":
            lambda w:
                np.ones_like(w),
    }

    bounds = [
        (0.0, B)
        for _ in range(n)
    ]

    result = minimize(
        objective,
        np.ones(n),
        jac=gradient,
        bounds=bounds,
        constraints=constraint,
        method="SLSQP",
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    if not result.success:

        raise RuntimeError(
            result.message
        )

    return np.asarray(
        result.x,
        dtype=float
    )


# ============================================================
# 9. WEIGHT ESS
# ============================================================

def weight_ess(w):

    w = np.asarray(
        w,
        dtype=float
    )

    return float(
        np.sum(w)**2
        / np.sum(w**2)
    )


# ============================================================
# 10. WEIGHTED MEAN
# ============================================================

def weighted_mean(
    y,
    w
):

    return float(
        np.sum(w * y)
        / np.sum(w)
    )


# ============================================================
# 11. SPATIAL FIELD
# ============================================================

def generate_spatial_noise(
    rng,
    noise_scale=0.12,
    n_components=30
):

    frequencies = rng.normal(
        0.0,
        0.70,
        size=(n_components, 2)
    )

    a = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    b = rng.normal(
        0.0,
        1.0,
        size=n_components
    )

    projection = (
        coords_std
        @ frequencies.T
    )

    field = (
        np.cos(projection) @ a
        + np.sin(projection) @ b
    ) / np.sqrt(
        2.0 * n_components
    )

    return (
        noise_scale
        * np.tanh(field)
    )


# ============================================================
# 12. MMD DIAGNOSTICS
#
# Biased empirical MMD^2.
#
# Source weights are normalized to sum to one.
# Target empirical weights are uniform.
# ============================================================

def weighted_mmd2_from_kernels(
    Kss,
    Ktt,
    Kst,
    source_weights
):

    w = np.asarray(
        source_weights,
        dtype=float
    )

    w = (
        w / np.sum(w)
    )

    m = Ktt.shape[0]

    target_weights = (
        np.ones(m) / m
    )

    value = (
        w @ Kss @ w
        + target_weights
          @ Ktt
          @ target_weights
        - 2.0
          * w
          @ Kst
          @ target_weights
    )

    return float(
        max(value, 0.0)
    )


# ============================================================
# 13. RUN EXPERIMENT
# ============================================================

rows = []

total = (
    N_REPS
    * len(GAMMAS)
    * len(B_VALUES)
)

counter = 0


print("\nRunning B-sweep...")
print("Total KMM fits:", total)


for rep in range(N_REPS):

    # One field per repetition, shared across gamma/B.
    noise_rng = np.random.default_rng(
        SEED
        + 100000
        + rep
    )

    epsilon = generate_spatial_noise(
        noise_rng,
        NOISE_SCALE
    )

    Y = (
        MU + epsilon
    )

    for gamma_index, gamma in enumerate(
        GAMMAS
    ):

        info = population_info[
            gamma
        ]

        p_source = info[
            "p_source"
        ]

        p_target = info[
            "p_target"
        ]

        B_star = info[
            "B_star"
        ]

        truth = info[
            "target_truth"
        ]

        rng = np.random.default_rng(
            SEED
            + rep * 1000
            + gamma_index
        )

        # Same source/target sample is reused for every B
        # within a given (rep, gamma), so comparisons across B
        # are paired.
        source_idx = rng.choice(
            len(df),
            size=N_SOURCE,
            replace=True,
            p=p_source
        )

        target_idx = rng.choice(
            len(df),
            size=N_TARGET,
            replace=True,
            p=p_target
        )

        Xs = X[
            source_idx
        ]

        Xt = X[
            target_idx
        ]

        ys = Y[
            source_idx
        ]

        # ----------------------------------------------------
        # Kernel matrices once per rep/gamma.
        # ----------------------------------------------------

        Kss = rbf_kernel(
            Xs,
            Xs,
            KMM_BANDWIDTH
        )

        Ktt = rbf_kernel(
            Xt,
            Xt,
            KMM_BANDWIDTH
        )

        Kst = rbf_kernel(
            Xs,
            Xt,
            KMM_BANDWIDTH
        )

        # ----------------------------------------------------
        # Unweighted diagnostics
        # ----------------------------------------------------

        w_un = np.ones(
            N_SOURCE
        )

        pre_mmd2 = (
            weighted_mmd2_from_kernels(
                Kss,
                Ktt,
                Kst,
                w_un
            )
        )

        est_un = weighted_mean(
            ys,
            w_un
        )

        error_un = (
            est_un - truth
        )

        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        w_oracle = (
            p_target[source_idx]
            / p_source[source_idx]
        )

        # Normalization does not change self-normalized
        # estimator/MMD but makes sampled diagnostics readable.
        w_oracle = (
            w_oracle
            / np.mean(w_oracle)
        )

        oracle_mmd2 = (
            weighted_mmd2_from_kernels(
                Kss,
                Ktt,
                Kst,
                w_oracle
            )
        )

        est_oracle = weighted_mean(
            ys,
            w_oracle
        )

        error_oracle = (
            est_oracle
            - truth
        )

        oracle_ess = weight_ess(
            w_oracle
        )

        oracle_max = float(
            np.max(w_oracle)
        )

        # ----------------------------------------------------
        # KMM B sweep
        # ----------------------------------------------------

        for B in B_VALUES:

            counter += 1

            try:

                w_kmm = constrained_kmm_weights(
                    Xs,
                    Xt,
                    bandwidth=KMM_BANDWIDTH,
                    B=B,
                    ridge=KMM_RIDGE
                )

            except Exception as exc:

                print(
                    "FAILED:",
                    "rep", rep,
                    "gamma", gamma,
                    "B", B,
                    exc
                )

                continue

            kmm_mmd2 = (
                weighted_mmd2_from_kernels(
                    Kss,
                    Ktt,
                    Kst,
                    w_kmm
                )
            )

            est_kmm = weighted_mean(
                ys,
                w_kmm
            )

            error_kmm = (
                est_kmm
                - truth
            )

            kmm_ess = weight_ess(
                w_kmm
            )

            kmm_max = float(
                np.max(w_kmm)
            )

            frac_at_B = float(
                np.mean(
                    w_kmm
                    >= B - 1e-4
                )
            )

            rows.append({

                "rep":
                    rep,

                "gamma":
                    gamma,

                "B":
                    B,

                "B_star":
                    B_star,

                "B_over_Bstar":
                    B / B_star,

                # Raw source-target discrepancy
                "pre_mmd2":
                    pre_mmd2,

                # After exact oracle weighting
                "oracle_mmd2":
                    oracle_mmd2,

                # After KMM
                "kmm_mmd2":
                    kmm_mmd2,

                "mmd_reduction_fraction":
                    (
                        1.0
                        - kmm_mmd2
                        / pre_mmd2
                        if pre_mmd2 > 0
                        else np.nan
                    ),

                # Oracle diagnostics
                "oracle_ess":
                    oracle_ess,

                "oracle_max_weight":
                    oracle_max,

                "oracle_error":
                    error_oracle,

                # Unweighted
                "unweighted_error":
                    error_un,

                # KMM
                "kmm_error":
                    error_kmm,

                "kmm_abs_error":
                    abs(
                        error_kmm
                    ),

                "kmm_ess":
                    kmm_ess,

                "kmm_max_weight":
                    kmm_max,

                "kmm_frac_at_B":
                    frac_at_B,
            })

            if (
                counter % 50 == 0
                or counter == total
            ):

                print(
                    f"{counter}/{total} | "
                    f"gamma={gamma:.1f} | "
                    f"B={B:g} | "
                    f"B/B*="
                    f"{B/B_star:.3f} | "
                    f"ESS={kmm_ess:.1f} | "
                    f"MMD2={kmm_mmd2:.6f}"
                )


# ============================================================
# 14. REPLICATE RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)

results.to_csv(
    OUTPUT_DIR
    / "california_B_sweep_replicates.csv",
    index=False
)


# ============================================================
# 15. SUMMARY
#
# Bias = mean signed error.
#
# SD = standard deviation of estimates/errors across reps.
#
# RMSE = sqrt(mean(error^2)).
# ============================================================

summary_rows = []


for gamma in GAMMAS:

    for B in B_VALUES:

        temp = results[
            (results["gamma"] == gamma)
            &
            (results["B"] == B)
        ]

        if len(temp) == 0:
            continue

        kmm_error = temp[
            "kmm_error"
        ].to_numpy()

        un_error = temp[
            "unweighted_error"
        ].to_numpy()

        oracle_error = temp[
            "oracle_error"
        ].to_numpy()

        summary_rows.append({

            "gamma":
                gamma,

            "B":
                B,

            "B_star":
                temp[
                    "B_star"
                ].iloc[0],

            "B_over_Bstar":
                temp[
                    "B_over_Bstar"
                ].iloc[0],

            "n_reps":
                len(temp),

            # -----------------------------------------------
            # MMD
            # -----------------------------------------------

            "pre_mmd2":
                temp[
                    "pre_mmd2"
                ].mean(),

            "oracle_mmd2":
                temp[
                    "oracle_mmd2"
                ].mean(),

            "kmm_mmd2":
                temp[
                    "kmm_mmd2"
                ].mean(),

            # -----------------------------------------------
            # Weight diagnostics
            # -----------------------------------------------

            "oracle_ess":
                temp[
                    "oracle_ess"
                ].mean(),

            "kmm_ess":
                temp[
                    "kmm_ess"
                ].mean(),

            "oracle_max_weight":
                temp[
                    "oracle_max_weight"
                ].mean(),

            "kmm_max_weight":
                temp[
                    "kmm_max_weight"
                ].mean(),

            "kmm_frac_at_B":
                temp[
                    "kmm_frac_at_B"
                ].mean(),

            # -----------------------------------------------
            # UNWEIGHTED decomposition
            # -----------------------------------------------

            "unweighted_bias":
                np.mean(
                    un_error
                ),

            "unweighted_sd":
                np.std(
                    un_error,
                    ddof=1
                ),

            "unweighted_mae":
                np.mean(
                    np.abs(
                        un_error
                    )
                ),

            "unweighted_rmse":
                np.sqrt(
                    np.mean(
                        un_error**2
                    )
                ),

            # -----------------------------------------------
            # ORACLE decomposition
            # -----------------------------------------------

            "oracle_bias":
                np.mean(
                    oracle_error
                ),

            "oracle_sd":
                np.std(
                    oracle_error,
                    ddof=1
                ),

            "oracle_mae":
                np.mean(
                    np.abs(
                        oracle_error
                    )
                ),

            "oracle_rmse":
                np.sqrt(
                    np.mean(
                        oracle_error**2
                    )
                ),

            # -----------------------------------------------
            # KMM decomposition
            # -----------------------------------------------

            "kmm_bias":
                np.mean(
                    kmm_error
                ),

            "kmm_sd":
                np.std(
                    kmm_error,
                    ddof=1
                ),

            "kmm_mae":
                np.mean(
                    np.abs(
                        kmm_error
                    )
                ),

            "kmm_rmse":
                np.sqrt(
                    np.mean(
                        kmm_error**2
                    )
                ),
        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    OUTPUT_DIR
    / "california_B_sweep_summary.csv",
    index=False
)


# ============================================================
# 16. DISPLAY IMPORTANT TABLE
# ============================================================

display_cols = [

    "gamma",
    "B",
    "B_star",
    "B_over_Bstar",

    "kmm_mmd2",
    "kmm_ess",
    "kmm_frac_at_B",

    "kmm_bias",
    "kmm_sd",
    "kmm_mae",
    "kmm_rmse",
]


print("\n")
print("=" * 110)
print("KMM B-SWEEP SUMMARY")
print("=" * 110)


display(
    summary[
        display_cols
    ].round(5)
)


# ============================================================
# 17. ORACLE / UNWEIGHTED REFERENCE
#
# These repeat across B, so show once per gamma.
# ============================================================

reference = (
    summary
    .sort_values(
        ["gamma", "B"]
    )
    .groupby(
        "gamma",
        as_index=False
    )
    .first()
)


reference_cols = [

    "gamma",
    "B_star",

    "pre_mmd2",
    "oracle_mmd2",

    "oracle_ess",

    "unweighted_bias",
    "unweighted_sd",
    "unweighted_rmse",

    "oracle_bias",
    "oracle_sd",
    "oracle_rmse",
]


print("\n")
print("=" * 110)
print("UNWEIGHTED / ORACLE REFERENCE")
print("=" * 110)


display(
    reference[
        reference_cols
    ].round(5)
)


reference[
    reference_cols
].to_csv(
    OUTPUT_DIR
    / "california_B_sweep_reference.csv",
    index=False
)


# ============================================================
# 18. FIGURE 1:
# B/B* -> KMM RMSE
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.3)
)


for gamma in GAMMAS:

    temp = summary[
        summary[
            "gamma"
        ] == gamma
    ].sort_values(
        "B_over_Bstar"
    )

    ax.plot(
        temp[
            "B_over_Bstar"
        ],
        temp[
            "kmm_rmse"
        ],
        marker="o",
        label=rf"$\gamma={gamma:g}$"
    )


ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1
)

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"$B/B^*(\gamma)$"
)

ax.set_ylabel(
    "KMM RMSE"
)

ax.set_title(
    "Weight bound versus estimation error"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "B_ratio_vs_rmse.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR
    / "B_ratio_vs_rmse.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 19. FIGURE 2:
# B -> MMD and ESS
#
# Two separate figures to keep them readable.
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.3)
)


for gamma in GAMMAS:

    temp = summary[
        summary[
            "gamma"
        ] == gamma
    ]

    ax.plot(
        temp["B"],
        temp["kmm_mmd2"],
        marker="o",
        label=rf"$\gamma={gamma:g}$"
    )


ax.set_xscale(
    "log"
)

ax.set_yscale(
    "log"
)

ax.set_xlabel(
    r"KMM bound $B$"
)

ax.set_ylabel(
    r"Post-weighting MMD$^2$"
)

ax.set_title(
    "Residual covariate imbalance"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "B_vs_postMMD.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR
    / "B_vs_postMMD.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. FIGURE 3:
# B -> ESS
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.3)
)


for gamma in GAMMAS:

    temp = summary[
        summary[
            "gamma"
        ] == gamma
    ]

    ax.plot(
        temp["B"],
        temp["kmm_ess"],
        marker="o",
        label=rf"$\gamma={gamma:g}$"
    )


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"KMM bound $B$"
)

ax.set_ylabel(
    "KMM weight ESS"
)

ax.set_title(
    "Weight stability"
)

ax.legend(
    frameon=False
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "B_vs_ESS.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR
    / "B_vs_ESS.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. FIGURE 4:
# B -> bias / SD
#
# This directly shows the bias-variance trade-off.
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 4.3)
)


for gamma in GAMMAS:

    temp = summary[
        summary[
            "gamma"
        ] == gamma
    ]

    ax.plot(
        temp["B"],
        np.abs(
            temp["kmm_bias"]
        ),
        marker="o",
        label=rf"$|\mathrm{{Bias}}|$, $\gamma={gamma:g}$"
    )


ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"KMM bound $B$"
)

ax.set_ylabel(
    "Absolute bias"
)

ax.set_title(
    "Residual bias under bounded weighting"
)

ax.legend(
    frameon=False,
    fontsize=7
)

ax.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "B_vs_bias.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR
    / "B_vs_bias.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. SAVE README
# ============================================================

readme = f"""
CALIFORNIA TARGETED B-SWEEP

Purpose:
Study the interaction between overlap and the KMM weight bound.

N source = {N_SOURCE}
N target = {N_TARGET}
N reps = {N_REPS}

gamma = {GAMMAS}
B = {B_VALUES}

KMM bandwidth = {KMM_BANDWIDTH}
KMM ridge = {KMM_RIDGE}

B_star is the population maximum density ratio
max_x qT(x)/qS(x) over all California empirical support points.

Post-weighting MMD uses the same RBF kernel as KMM.

Bias = mean signed estimation error.
SD = standard deviation of estimation error across repetitions.
RMSE = sqrt(mean(error^2)).

Source and target samples are paired across B within each
(rep, gamma), so B comparisons use the same data realization.
"""


with open(
    OUTPUT_DIR / "README.txt",
    "w"
) as f:

    f.write(readme)


# ============================================================
# 23. ZIP
# ============================================================

import shutil


zip_path = shutil.make_archive(
    "/content/california_B_sweep",
    "zip",
    OUTPUT_DIR
)


print("\nDONE.")
print("ZIP:", zip_path)

print("\nPlease send me:")
print("1. california_B_sweep_summary.csv")
print("2. california_B_sweep_reference.csv")
print("3. B_ratio_vs_rmse.png")
print("4. B_vs_postMMD.png")