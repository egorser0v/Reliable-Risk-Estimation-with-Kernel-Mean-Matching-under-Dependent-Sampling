# Three-domain real-data benchmark

Two estimands: bounded target response mean and target squared-error risk of a frozen random forest. All target outcomes are used only for scoring. No target labels select weights, predictors, windows, kernel bandwidth, or uncertainty parameters.

## Tables (normalized units)

### mean

dataset           method      MAE  Coverage  Half_width        ESS  Tasks  Domains
   NOAA       Unweighted 0.050319  0.198582    0.015678  90.070922    141       12
   NOAA      Logistic IW 0.038866  0.276596    0.020917  50.282630    141       12
   NOAA              KMM 0.037899  0.546099    0.035196  14.259430    141       12
   NOAA KMM + dependence 0.037899  0.673759    0.045777   8.757641    141       12
   PM25       Unweighted 0.020848  0.687500    0.025597  91.104167     48       12
   PM25      Logistic IW 0.026964  0.770833    0.037416  44.465131     48       12
   PM25              KMM 0.015463  0.979167    0.054019  18.060311     48       12
   PM25 KMM + dependence 0.015463  0.979167    0.058492  15.463926     48       12
   TNBC       Unweighted 0.039582  0.282051    0.013999 256.000000     78       26
   TNBC      Logistic IW 0.040549  0.435897    0.029148  22.301396     78       26
   TNBC              KMM 0.038687  0.512821    0.033658  29.341200     78       26
   TNBC KMM + dependence 0.038687  0.525641    0.035921  20.831505     78       26

### risk

dataset           method      MAE  Coverage  Half_width        ESS  Tasks  Domains
   NOAA       Unweighted 0.002086  0.333333    0.000794  90.070922    141       12
   NOAA      Logistic IW 0.002051  0.432624    0.001049  50.282630    141       12
   NOAA              KMM 0.002128  0.574468    0.001815  14.259430    141       12
   NOAA KMM + dependence 0.002128  0.581560    0.001918  12.875141    141       12
   PM25       Unweighted 0.002140  0.583333    0.002626  91.104167     48       12
   PM25      Logistic IW 0.002050  0.854167    0.003680  44.465131     48       12
   PM25              KMM 0.002437  0.937500    0.004852  18.060311     48       12
   PM25 KMM + dependence 0.002437  0.958333    0.004964  17.200179     48       12
   TNBC       Unweighted 0.021290  0.243590    0.008381 256.000000     78       26
   TNBC      Logistic IW 0.022774  0.384615    0.018989  22.301396     78       26
   TNBC              KMM 0.021115  0.538462    0.019755  29.341200     78       26
   TNBC KMM + dependence 0.021115  0.551282    0.021339  25.800656     78       26

## Results by dataset

NOAA, mean: KMM MAE change versus unweighted = +24.7% improvement; KMM iid coverage 54.6% -> dependence correction 67.4%; half-width 0.035196 -> 0.045777.
PM25, mean: KMM MAE change versus unweighted = +25.8% improvement; KMM iid coverage 97.9% -> dependence correction 97.9%; half-width 0.054019 -> 0.058492.
TNBC, mean: KMM MAE change versus unweighted = +2.3% improvement; KMM iid coverage 51.3% -> dependence correction 52.6%; half-width 0.033658 -> 0.035921.
NOAA, risk: KMM MAE change versus unweighted = -2.0% improvement; KMM iid coverage 57.4% -> dependence correction 58.2%; half-width 0.001815 -> 0.001918.
PM25, risk: KMM MAE change versus unweighted = -13.9% improvement; KMM iid coverage 93.8% -> dependence correction 95.8%; half-width 0.004852 -> 0.004964.
TNBC, risk: KMM MAE change versus unweighted = +0.8% improvement; KMM iid coverage 53.8% -> dependence correction 55.1%; half-width 0.019755 -> 0.021339.

Positive percentages above mean lower point-estimation error; negative percentages mean deterioration. These descriptive differences are not significance tests. The experiment does not establish universal point-estimation superiority or nominal calibration.

## Interpretation and limitations

These are empirical stress tests, not finite-sample theorem verification. The intervals estimate source variation conditional on fitted weights and fixed target covariates; weight-fitting and target-population sampling uncertainty are not fully represented. The reference is the observed finite target-window mean/risk, not an independently known population expectation.
Source temporal covariance uses Bartlett HAC at actual calendar lags through 14 days. TNBC uses a PSD Gaussian spatial covariance taper with bandwidth five times the median within-window nearest-neighbor distance. Corrected variance is floored at iid variance; this is an explicit empirical conservative correction, not the GP-derived ESS formula.
Tasks share stations/patients and windows; task counts are not independent replicate counts. Any inference across tasks should resample whole domains. No selection based on coverage or target errors is performed.
TNBC is a processed public secondary release of Keren et al. data (https://doi.org/10.6084/m9.figshare.26068006.v2). The X matrix is already standardized and contains negative marker values. The endpoint is explicitly the positive part of processed Ki67, upper-clipped at the training-patient 99th percentile and divided by that percentile; it is not raw concentration or clinical proliferation risk. Ki67 is excluded from predictors; derived cell-type/niche labels and spatial embeddings are excluded. Upstream cohort-level preprocessing was not refit and may carry batch effects or use held-out distributions; source-only preprocessing claims refer to this experiment, not the original release. Predictor training patients are disjoint from every evaluation patient.
NOAA compares individual station-day sequences, with training stations disjoint from source and target stations; source and target station pools are disjoint. PM2.5 trains on 2013-2014 and evaluates 2015 -> 2016 by station and quarter. Neither naturally occurring shift guarantees conditional invariance.
One pooled covariate-only scaler and median bandwidth are used for all KMM Gram matrices. Weights obey 0<=w<=10 and sum(w)=n. Logistic ratios are upper-capped and self-normalized for estimation. Zero weights and upper-bound weights are reported separately.
Historical PM2.5/NOAA numbers are not pooled with this corrected protocol. This run replaces neither historical files nor manuscript text. See task CSVs and dataset protocol JSON for complete diagnostics and provenance.

## Reproduction

`python experiments/kmm_three_domain_benchmark.py --dataset all`
Set KMM_NOAA_ROOT to the directory containing coverage.csv, candidate_metadata.csv and station gzip files if rebuilding the NOAA cache. Run prepare_tnbc_benchmark.py to download TNBC. Requirements: numpy, pandas, scipy, scikit-learn, threadpoolctl, h5py. Seed: 20261005