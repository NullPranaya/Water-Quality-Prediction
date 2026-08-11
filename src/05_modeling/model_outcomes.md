# Model Outcomes

Held-out test-set results for every water-quality model trained in
`src/05_modeling/`. Generated 2026-08-11 from the executed notebooks.

## Setup (shared by all models)

- **Dataset:** `data/03c_merge_tertiary/epa-full.csv` (48,251 rows × 318 columns)
- **Predictors:** 30 environmental / spatial / temporal features (location, PRISM
  climate, ISU weather, streamflow, soil, land cover, nutrient loading, plus
  day-of-year seasonality and observation year). Other water-quality `_value`
  columns are **excluded** as predictors.
- **Split:** `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped on
  **`MonitoringLocationIdentifier`** — 20% of the *stations* that measured a
  target are held out whole, and none of their rows are seen in training.
- **Preprocessing:** `SimpleImputer(median)` on every model; `StandardScaler`
  added for linear regression only.
- **Rows per target** vary because each target keeps only its own non-null,
  in-range measurements (see the `N test` column). The test *row* share drifts
  from 20% because station volume is heavy-tailed.

### Why the split changed

The previous results on this page used `train_test_split(test_size=0.2)` on
rows. That left **99.1–99.7% of test rows at a station that was also in the
training set** (`src/04_eda/outputs/split_leakage_check.csv`). Latitude and
longitude are model inputs with roughly one distinct value per station, so a
tree could identify the station from its coordinates and recall its typical
level — and the scores measured recall as much as prediction. The EDA's
red flags 1–2 (`src/04_eda/eda-summary.md`) called it, and predicted that
Specific Conductance and Total Dissolved Solids would fall hardest. They did.

Every number below now answers one question: **how well does this model predict
at a monitoring station it has never seen?** The pre-split numbers are preserved
in `model_metrics_random_split.csv` for comparison and should not be quoted as
performance.

### Metrics

- **R²** — coefficient of determination on the test set (higher is better; 1.0 is perfect).
- **RMSE** — root mean squared error, in the target's own units (lower is better).
- **MAE** — mean absolute error, in the target's own units.
- **Error Rate** — symmetric mean absolute percentage error / sMAPE, as a percent.
  For zero-inflated targets it is **a function of the zero fraction, not of model
  quality**: its per-row term saturates at 200% whenever `y = 0`, so Nitrite's
  84% zeros alone force a floor near 168%. Read MAE and RMSE instead for those.
- **Persistence R²** — the memorization bar: "this station's next value equals
  its previous value", a model with **no features at all**, scored on the same
  held-out rows. Because the split is grouped by station, every observation a
  test station made is in the test set, so the baseline is free to use the
  site's own history — which the model never saw.
- **Margin** — model R² minus persistence R², computed on the identical subset
  of rows (those that have a previous observation at the same station). A
  negative margin means 30 environmental predictors buy less than repeating the
  last reading.

Models compared: **Linear Regression**
(`linear_regression/multiple_linear_regression.ipynb`), **Random Forest**
(`random_forest/random_forest.ipynb`), **Gradient Boosting** —
HistGradientBoostingRegressor (`gradient_boosting/gradient_boosting.ipynb`).
Each notebook saves its 12 fitted pipelines as `<prefix>_<target>.pkl` into its
own folder (`lr_*`, `rf_*`, `gb_*`) and writes its own rows of
`model_metrics.csv`.

---

## Linear Regression

| Target | N test | N stations | R² | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | 6,092 | 200 | 0.8947 | 2.8596 | 2.2369 | 30.85 | 0.640 | +0.258 |
| Total Dissolved Solids | 3,902 | 117 | 0.4211 | 95.6835 | 76.4187 | 24.51 | 0.811 | -0.390 |
| Dissolved Oxygen | 6,640 | 182 | 0.3877 | 2.1020 | 1.5180 | 17.56 | 0.325 | +0.077 |
| Specific Conductance | 4,617 | 97 | 0.2195 | 179.4197 | 98.2960 | 18.36 | 0.856 | -0.630 |
| Nitrate + Nitrite | 1,045 | 74 | 0.2176 | 3.9983 | 2.9153 | 79.01 | 0.190 | +0.016 |
| Nitrate | 2,719 | 56 | 0.1719 | 5.2115 | 3.2675 | 111.47 | 0.398 | -0.225 |
| pH | 5,673 | 222 | 0.1707 | 0.5770 | 0.4346 | 5.52 | -0.060 | +0.237 |
| Total Suspended Solids | 2,435 | 111 | 0.0416 | 204.1877 | 80.9271 | 124.07 | -0.993 | +1.025 |
| E. coli | 3,831 | 87 | 0.0319 | 9,922.6941 | 2,372.3393 | 145.40 | -0.741 | +0.774 |
| Total Phosphorus | 1,294 | 91 | 0.0228 | 0.4714 | 0.1958 | 66.70 | 0.319 | -0.306 |
| Turbidity | 3,581 | 169 | 0.0124 | 87.0211 | 34.4120 | 112.87 | -0.788 | +0.807 |
| Nitrite | 2,271 | 46 | -0.0033 | 0.1227 | 0.0382 | 178.89 | -0.690 | +0.686 |

## Random Forest

| Target | N test | N stations | R² | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | 6,092 | 200 | 0.9459 | 2.0509 | 1.5129 | 21.41 | 0.640 | +0.308 |
| Specific Conductance | 4,617 | 97 | 0.6578 | 118.8071 | 80.2480 | 15.74 | 0.856 | -0.198 |
| Total Dissolved Solids | 3,902 | 117 | 0.5329 | 85.9496 | 67.1473 | 21.41 | 0.811 | -0.277 |
| Nitrate + Nitrite | 1,045 | 74 | 0.4966 | 3.2071 | 2.2456 | 65.68 | 0.190 | +0.268 |
| Dissolved Oxygen | 6,640 | 182 | 0.4828 | 1.9319 | 1.3020 | 15.56 | 0.325 | +0.166 |
| Nitrate | 2,719 | 56 | 0.4607 | 4.2056 | 2.5126 | 102.72 | 0.398 | +0.059 |
| pH | 5,673 | 222 | 0.4103 | 0.4865 | 0.3484 | 4.43 | -0.060 | +0.475 |
| Total Suspended Solids | 2,435 | 111 | 0.0707 | 201.0664 | 63.4836 | 95.70 | -0.993 | +1.096 |
| E. coli | 3,831 | 87 | 0.0486 | 9,836.9524 | 2,180.3220 | 123.58 | -0.741 | +0.789 |
| Turbidity | 3,581 | 169 | 0.0331 | 86.1073 | 31.5456 | 89.08 | -0.788 | +0.822 |
| Nitrite | 2,271 | 46 | 0.0160 | 0.1215 | 0.0414 | 175.27 | -0.690 | +0.707 |
| Total Phosphorus | 1,294 | 91 | 0.0002 | 0.4768 | 0.2102 | 67.19 | 0.319 | -0.317 |

## Gradient Boosting (HistGradientBoostingRegressor)

| Target | N test | N stations | R² | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | 6,092 | 200 | 0.9395 | 2.1685 | 1.6219 | 23.05 | 0.640 | +0.301 |
| Nitrate + Nitrite | 1,045 | 74 | 0.4998 | 3.1969 | 2.1952 | 66.76 | 0.190 | +0.268 |
| Dissolved Oxygen | 6,640 | 182 | 0.4668 | 1.9616 | 1.4071 | 16.67 | 0.325 | +0.149 |
| Total Dissolved Solids | 3,902 | 117 | 0.4382 | 94.2563 | 69.2902 | 22.53 | 0.811 | -0.376 |
| Nitrate | 2,719 | 56 | 0.4155 | 4.3784 | 2.6519 | 105.29 | 0.398 | +0.014 |
| pH | 5,673 | 222 | 0.3517 | 0.5101 | 0.3683 | 4.69 | -0.060 | +0.414 |
| Specific Conductance | 4,617 | 97 | 0.1846 | 183.3914 | 138.1098 | 28.19 | 0.856 | -0.698 |
| E. coli | 3,831 | 87 | 0.1157 | 9,483.7225 | 2,058.3660 | 127.24 | -0.741 | +0.857 |
| Total Phosphorus | 1,294 | 91 | 0.0266 | 0.4705 | 0.2024 | 66.99 | 0.319 | -0.216 |
| Total Suspended Solids | 2,435 | 111 | -0.0357 | 212.2647 | 72.7747 | 101.61 | -0.993 | +0.953 |
| Turbidity | 3,581 | 169 | -0.0986 | 91.7846 | 35.9358 | 94.59 | -0.788 | +0.692 |
| Nitrite | 2,271 | 46 | -0.1216 | 0.1297 | 0.0461 | 180.37 | -0.690 | +0.569 |

---

## Cross-model comparison (test R², unseen stations)

Best model per target in **bold**.

| Target | N test | N stations | Linear | Random Forest | Gradient Boosting | Best |
|---|--:|--:|--:|--:|--:|:--|
| Water Temperature | 6,092 | 200 | 0.8947 | **0.9459** | 0.9395 | Random Forest |
| Specific Conductance | 4,617 | 97 | 0.2195 | **0.6578** | 0.1846 | Random Forest |
| Total Dissolved Solids | 3,902 | 117 | 0.4211 | **0.5329** | 0.4382 | Random Forest |
| Nitrate + Nitrite | 1,045 | 74 | 0.2176 | 0.4966 | **0.4998** | Gradient Boosting |
| Dissolved Oxygen | 6,640 | 182 | 0.3877 | **0.4828** | 0.4668 | Random Forest |
| Nitrate | 2,719 | 56 | 0.1719 | **0.4607** | 0.4155 | Random Forest |
| pH | 5,673 | 222 | 0.1707 | **0.4103** | 0.3517 | Random Forest |
| E. coli | 3,831 | 87 | 0.0319 | 0.0486 | **0.1157** | Gradient Boosting |
| Total Suspended Solids | 2,435 | 111 | 0.0416 | **0.0707** | -0.0357 | Random Forest |
| Turbidity | 3,581 | 169 | 0.0124 | **0.0331** | -0.0986 | Random Forest |
| Total Phosphorus | 1,294 | 91 | 0.0228 | 0.0002 | **0.0266** | Gradient Boosting |
| Nitrite | 2,271 | 46 | -0.0033 | **0.0160** | -0.1216 | Random Forest |

## What the honest split cost (best model per target)

`R² before` is the same model family under the old random row split.

| Target | Best model | R² before | R² now | Change |
|---|:--|--:|--:|--:|
| Water Temperature | Random Forest | 0.9517 | 0.9459 | -0.006 |
| Specific Conductance | Random Forest | 0.8864 | 0.6578 | -0.229 |
| Total Dissolved Solids | Random Forest | 0.8243 | 0.5329 | -0.291 |
| Nitrate + Nitrite | Gradient Boosting | 0.7062 | 0.4998 | -0.206 |
| Dissolved Oxygen | Random Forest | 0.5962 | 0.4828 | -0.113 |
| Nitrate | Random Forest | 0.6742 | 0.4607 | -0.214 |
| pH | Random Forest | 0.5261 | 0.4103 | -0.116 |
| E. coli | Gradient Boosting | 0.2816 | 0.1157 | -0.166 |
| Total Suspended Solids | Random Forest | 0.3913 | 0.0707 | -0.321 |
| Turbidity | Random Forest | 0.2750 | 0.0331 | -0.242 |
| Total Phosphorus | Gradient Boosting | 0.3437 | 0.0266 | -0.317 |
| Nitrite | Random Forest | 0.0863 | 0.0160 | -0.070 |

Mean R² across the twelve targets:

| Family | before | now | change |
|---|--:|--:|--:|
| Linear Regression | 0.2485 | 0.2157 | -0.033 |
| Random Forest | 0.5381 | 0.3463 | -0.192 |
| Gradient Boosting | 0.5465 | 0.2652 | -0.281 |

## Against the memorization baseline

Persistence — repeat the station's previous value — beside the best model for each target, on the identical rows. `Revisit gap` is the median number of days between the two visits: a target resampled the next day is far easier to persist than one resampled a month later.

| Target | Revisit gap (days) | Pairs | Persistence R² | Best model R² | Margin |
|---|--:|--:|--:|--:|--:|
| Total Dissolved Solids | 28 | 3,785 | 0.811 | 0.533 | -0.277 |
| Total Phosphorus | 33 | 1,203 | 0.319 | 0.103 | -0.216 |
| Specific Conductance | 1 | 4,520 | 0.856 | 0.658 | -0.198 |
| Nitrate | 15 | 2,663 | 0.398 | 0.458 | +0.059 |
| Dissolved Oxygen | 20 | 6,458 | 0.325 | 0.491 | +0.166 |
| Nitrate + Nitrite | 33 | 971 | 0.190 | 0.458 | +0.268 |
| Water Temperature | 27 | 5,892 | 0.640 | 0.948 | +0.308 |
| pH | 28 | 5,451 | -0.060 | 0.414 | +0.475 |
| Nitrite | 15 | 2,225 | -0.690 | 0.017 | +0.707 |
| Turbidity | 29 | 3,412 | -0.788 | 0.034 | +0.822 |
| E. coli | 27 | 3,744 | -0.741 | 0.116 | +0.857 |
| Total Suspended Solids | 31 | 2,324 | -0.993 | 0.103 | +1.096 |

**9 of 12 targets beat persistence; 3 do not** —
Specific Conductance, Total Dissolved Solids, Total Phosphorus. For those three, the honest summary is that a site's last
reading is a better forecast than the model, and the correct next step is to
hand the model that reading explicitly as a `y_prev` / `days_since_prev` feature
rather than leaving it to be approximated from coordinates.

A large positive margin is not automatically a triumph: Total Suspended Solids,
Turbidity, *E. coli* and Nitrite clear the bar by 0.57–1.10 R² only because
persistence is *catastrophically* bad on them (R² of −0.99 to −0.69 — worse than
predicting the mean). Beating a negative baseline with an R² of 0.02–0.12 is
still a model that explains almost nothing.

## Takeaways

- **The scores fell, and that is the point.** Mean R² dropped from
  0.55 to 0.27
  for gradient boosting and 0.54 to
  0.35 for random forest. The models did not get
  worse; the measurement got honest.
- **The station-dominated targets collapsed exactly as predicted.** Specific
  Conductance 0.89 →
  0.18 (GB) and Total
  Dissolved Solids 0.83 →
  0.44. Their old
  scores were station recall, and persistence R² of 0.86 / 0.81 confirms that
  most of what there is to know about these two is *which site you are standing
  at*.
- **Water Temperature is the one target that survives intact**
  (0.946 RF, down only
  0.006).
  It is the one target with a real physical mechanism in the feature set
  (ρ = +0.85 with `prism_tmin_c`, both between *and* within station), and it
  beats persistence by 0.31.
- **The leaderboard flipped.** Random forest now wins
  9 of 12 targets against gradient
  boosting's 3 (it was 3 vs 9 under
  the leaky split). Boosting's early stopping validates on a random slice of the
  training stations, so it tunes its stopping point against the very leakage the
  split removes — Specific Conductance is the clearest casualty
  (0.66 RF vs
  0.18 GB). Random
  forest is the more robust default on unseen stations.
- **Tree ensembles still beat the linear baseline**, but by much less: the mean
  gap narrowed from 0.30
  to 0.13 R². Most of
  the trees' old advantage was their superior ability to memorise a station.
- **Eight targets now sit below R² = 0.5 for every family.** On unseen stations,
  the current 30-feature design predicts water temperature well, dissolved
  oxygen / pH / the nitrate group moderately, and the skewed
  nutrient-and-bacteria block essentially not at all.
- **The remaining EDA fixes are now the interesting ones** — §3.3 and §4.3 of
  `src/04_eda/eda-summary.md`: `y_prev`/`days_since_prev` features, a hurdle
  model for the zero-inflated nitrogen species, `log10` targets for the skewed
  block, and dropping `pct_row_crops` (an exact sum of `pct_corn + pct_soybean`,
  which leaves the linear design matrix singular). None of them were worth
  measuring against a leaky split.

## Reproducing

```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/linear_regression/multiple_linear_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/random_forest/random_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/gradient_boosting/gradient_boosting.ipynb
```

Each notebook rewrites its own 12 `.pkl` files and its own rows of
`src/05_modeling/model_metrics.csv`; the other two families' rows are left
untouched, so the notebooks may be run in any order.
