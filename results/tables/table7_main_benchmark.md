# Table 7: Full Benchmark Results (Target Mean Y and Target Risk L)

Configurations: $n=96$, $\ell_x=3.0$, $h=1.5$, $B=15.0$, Repetitions $= 300$, Seed $= 42$.  
*Asterisk ($^*$) indicates significant undercoverage ($< 0.925$). Bold highlights best valid practical method.*

| Method | MAE (Y) | Coverage (Y) | Radius (Y) | WS (Y) ↓ | MAE (Risk) | Coverage (Risk) | Radius (Risk) | WS (Risk) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unweighted | 0.1138 | 0.017* | 0.0362 | 3.182 | 0.0048 | 0.080* | 0.0019 | 0.119 |
| Nominal $n$ | 0.0196 | 0.917* | 0.0626 | 0.134 | 0.0025 | 0.853* | 0.0055 | 0.024 |
| Block-Bootstrap | 0.0196 | 1.000 | 0.4538 | 0.908 | 0.0025 | 0.967 | 0.0113 | **0.026** |
| Input $n_{\mathrm{eff}}$ | 0.0196 | 1.000 | 0.1696 | 0.339 | 0.0025 | 0.987 | 0.0150 | 0.031 |
| **GP $n_{\mathrm{eff}}$ (Ours)** | 0.0196 | 1.000 | **0.1380** | **0.276** | 0.0025 | 0.970 | 0.0120 | **0.026** |
| Strict Thm 1 | 0.0196 | 1.000 | 5.2189 | 10.438 | 0.0025 | 1.000 | 5.1220 | 10.244 |
| Oracle IW | 0.0334 | 0.477* | 0.0275 | 0.590 | 0.0027 | 0.627* | 0.0033 | 0.033 |
| Oracle i.i.d. | 0.0115 | 0.903* | 0.0257 | 0.072 | 0.0010 | 0.917* | 0.0024 | 0.006 |
