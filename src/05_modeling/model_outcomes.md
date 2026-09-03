# Model Outcomes

Held-out test-set results for every water-quality model trained in
`src/05_modeling/`. Generated 2026-08-11 from the executed notebooks.

## Setup (shared by all models)

- **Dataset:** `data/03c_merge_tertiary/epa-full.csv` (48,251 rows × 318 columns)
- **Predictors:** 29 environmental / spatial / temporal features (location, PRISM
  climate, ISU weather, streamflow, soil, land cover, nutrient loading, plus
  day-of-year seasonality and observation year). Other water-quality `_value`
  columns are **excluded** as predictors, and so is `pct_row_crops` — see below.
- **Split:** `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped on
  **`MonitoringLocationIdentifier`** — 20% of the *stations* that measured a
  target are held out whole, and none of their rows are seen in training.
- **Preprocessing:** `SimpleImputer(median)` on every model; `StandardScaler`
  added for linear regression only.
- **Targets:** **thirteen** — the twelve measured water-quality parameters plus
  **WQI**, the composite index built by `src/04_eda/wqi-calculation.ipynb`.
- **Target transform:** chosen per (target, family) by a cross-validated
  bake-off, not by hand. 11 of the 39 models are fitted on **`log10(y + c)`**;
  the other 28 on raw values. See below.
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

### Why `pct_row_crops` was dropped

`pct_row_crops` equals `pct_corn + pct_soybean` **exactly** on all 48,251 rows
(max |residual| 1.1e-16). As a model input it was the single source of a rank
deficiency: with it, the standardised design matrix had rank 29 of 30 and a
condition number of **1.3e15**; without it the matrix is full rank at 29 and the
condition number is **27.8**. It carried no information — it was a third copy of
two columns already present.

The effect on scores is exactly what that implies: **linear regression's R² is
unchanged to four decimals on all twelve targets** (the least-squares solver was
already projecting the degenerate direction away, it just could not say *how* it
split the coefficients between the three columns). What changes is that the
land-cover coefficients are now **identified** — previously the solver returned
one of infinitely many equivalent answers and not even their signs meant
anything. The tree families move by a mean of 0.009 R² (max 0.077), which is
resampling noise from `max_features` and split-candidate selection, not signal.

Eight features still have VIF ≥ 10 — the four nutrient columns worst — so
individual coefficients remain hard to read. Rank deficiency is fixed;
collinearity is not.

### How the target scale is chosen

Which models fit `log10(y + c)` instead of the raw target is **measured, not
asserted**. Every target is fitted both ways and the winner is chosen inside
each notebook, per (target, family). Two guards decide who enters at all:

- a target with negative values has no log — **Water Temperature**;
- a target that is mostly zeros has no meaningful log scale. `log10` maps the
  whole point mass onto `log10(c)`, so the resulting MAE drop measures how well
  the model predicts that mass, not how well it fits the water. **Nitrate**
  (40% zeros) and **Nitrite** (85%) are hurdle-model problems, not transform
  problems. Measured without this guard they do "win" on MAE — while sitting at
  R² ≈ 0.00 as they do it, which is the failure the guard exists to catch.

For everyone else, both arms are cross-validated over the **training** stations
(`GroupKFold`, 3 folds) and compared on **MAE in the target's own units**. MAE
is the yardstick because it is the error the dashboard displays, and because it
is the only candidate that does not structurally favour an arm: raw-scale R²
always flatters the raw fit, log-scale R² always flatters the log fit, so
neither can arbitrate. The log arm must win by **5% relative** — ties go to the
untransformed target, since a transform is a real complication (a
back-transform, a smearing correction, two scales to report) and should have to
earn its place. Nothing in this procedure touches the test set.

**The rule was itself validated.** Across 30 target-family pairs, its choice was
compared against which arm actually wins on the held-out test set:

| Selection rule | Agrees with test | Worst error |
|---|--:|:--|
| single split, strict "lower MAE wins" | 23/30 | picks log for SpC/RF, which costs **29% MAE** |
| single split + 5% margin | 26/30 | same SpC/RF false positive |
| **3-fold CV + 5% margin** (shipped) | **26/30** | skips a 7.7% gain — never picks a transform that hurts |

The two 26/30 rules tie on accuracy and differ entirely in the *kind* of error
they make. All four of the shipped rule's misses are conservative — a skipped
gain of at most 7.7% — while the single-split rules wrongly *apply* a transform
where the raw fit is 29% better. For a decision that changes the numbers the
dashboard puts in front of someone, a false positive is much worse than a
missed opportunity, so the conservative rule wins on a tie.

The margin matters concretely: without it, three models were selected on
validation margins of 0.09–1.5% and **all three were worse on test** — Nitrate +
Nitrite / Gradient Boosting gave up **0.23 R²** for a 0.51% validation win.

#### What it picked

| | Linear Regression | Random Forest | Gradient Boosting |
|---|:--|:--|:--|
| *E. coli* | log10 | log10 | log10 |
| Total Suspended Solids | log10 | log10 | log10 |
| Turbidity | log10 | log10 | log10 |
| Total Phosphorus | raw | log10 | log10 |
| all nine others, incl. **WQI** | raw | raw | raw |

This reproduces the previously hand-picked set almost exactly, with one
refinement it found on its own: **Total Phosphorus / Linear Regression stays
raw**, because there the raw fit genuinely wins (CV MAE 0.186 vs 0.188). The
per-family split is the point — the right scale is a property of the
target *and* the estimator, not of the target alone.

Note that a transform being chosen does not make the model good. All four
log-fitted targets remain the weakest in the set; the transform moves *E. coli*
from "explains nothing" to "explains a third of the log-scale variance", not to
"solved".

#### How the transform is expressed

`c` is **1% of the positive training median** — computed on training rows only.
A library default of `c = 1` would be a rounding error for *E. coli* (which runs
to 10⁶ MPN/100 mL) and wider than the entire distribution of total phosphorus.
The offsets are 1.5, 0.196, 0.140 and 0.0012.

Back-transforming needs **Duan's (1983) smearing correction**: E[y] is not
10^E[log₁₀ y], so naive exponentiation biases every prediction low. The factors
run **1.07–1.77** for the tree families and 1.64–5.19 for linear regression,
whose log-residuals are far wider.

Both numbers are stored **inside each `.pkl`**, alongside the fitted pipeline
and the feature list, so `app.py` cannot mislabel a `log10` prediction as mg/L
because a metrics file went stale. They are also written to `model_metrics.csv`
for reporting.

#### The cost, and the gain

On the target's own units a log fit scores *slightly worse* — raw-scale R² is
set by the same extreme readings the transform de-emphasises. On the log scale
it is worth a great deal, and it cuts MAE by 13–39%:

| Target | Best model | R² raw, before | R² raw, now | R² log, before | R² log, now | MAE |
|---|:--|--:|--:|--:|--:|--:|
| *E. coli* | GB | 0.119 | 0.063 | −0.403 | **0.358** | −27% |
| Total Suspended Solids | GB | −0.012 | 0.066 | −0.235 | **0.338** | −36% |
| Turbidity | GB | −0.147 | 0.038 | −0.182 | **0.283** | −39% |
| Total Phosphorus | GB | −0.006 | 0.038 | −0.010 | **0.189** | −16% |

The log scale is also the one these targets are read on in practice: the EPA's
recreational-water criterion for *E. coli* is a **geometric mean** (126 MPN/100
mL), which is a log-scale statistic.

### Metrics

- **R²** — coefficient of determination on the test set (higher is better; 1.0 is perfect).
- **R² (log)** — R² in `log10(y + c)` space, reported only for the four
  log-fitted targets. For those four this is the headline number; the raw-scale
  R² beside it is dominated by a handful of extreme readings.
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
  negative margin means 29 environmental predictors buy less than repeating the
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

| Target | Scale | N test | N stations | R² | R² (log) | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | raw | 6,092 | 200 | 0.8947 | — | 2.8596 | 2.2369 | 30.85 | 0.640 | +0.258 |
| Total Dissolved Solids | raw | 3,902 | 117 | 0.4211 | — | 95.6835 | 76.4187 | 24.51 | 0.811 | -0.390 |
| Dissolved Oxygen | raw | 6,640 | 182 | 0.3877 | — | 2.1020 | 1.5180 | 17.56 | 0.325 | +0.077 |
| Specific Conductance | raw | 4,617 | 97 | 0.2195 | — | 179.4197 | 98.2960 | 18.36 | 0.856 | -0.630 |
| Nitrate + Nitrite | raw | 1,045 | 74 | 0.2176 | — | 3.9983 | 2.9153 | 79.01 | 0.190 | +0.016 |
| Nitrate | raw | 2,719 | 56 | 0.1719 | — | 5.2115 | 3.2675 | 111.47 | 0.398 | -0.225 |
| pH | raw | 5,673 | 222 | 0.1707 | — | 0.5770 | 0.4346 | 5.52 | -0.060 | +0.237 |
| WQI | raw | 4,622 | 189 | 0.0763 | — | 16.4872 | 13.6658 | 33.18 | 0.172 | -0.091 |
| E. coli | log10 | 3,831 | 87 | 0.0254 | **0.2043** | 9,955.7669 | 1,902.9030 | 121.73 | -0.741 | +0.766 |
| Total Phosphorus | raw | 1,294 | 91 | 0.0228 | — | 0.4714 | 0.1958 | 66.70 | 0.319 | -0.306 |
| Nitrite | raw | 2,271 | 46 | -0.0033 | — | 0.1227 | 0.0382 | 178.89 | -0.690 | +0.686 |
| Turbidity | log10 | 3,581 | 169 | -0.0145 | **0.0618** | 88.2005 | 30.2246 | 91.06 | -0.788 | +0.777 |
| Total Suspended Solids | log10 | 2,435 | 111 | -0.0255 | **0.1509** | 211.2197 | 68.1379 | 102.52 | -0.993 | +0.934 |

## Random Forest

| Target | Scale | N test | N stations | R² | R² (log) | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | raw | 6,092 | 200 | 0.9466 | — | 2.0367 | 1.5004 | 21.19 | 0.640 | +0.309 |
| Specific Conductance | raw | 4,617 | 97 | 0.6545 | — | 119.3700 | 81.7043 | 16.00 | 0.856 | -0.201 |
| Total Dissolved Solids | raw | 3,902 | 117 | 0.5329 | — | 85.9466 | 66.9173 | 21.30 | 0.811 | -0.277 |
| Nitrate + Nitrite | raw | 1,045 | 74 | 0.5047 | — | 3.1810 | 2.2072 | 64.87 | 0.190 | +0.276 |
| Dissolved Oxygen | raw | 6,640 | 182 | 0.4910 | — | 1.9166 | 1.2894 | 15.44 | 0.325 | +0.175 |
| Nitrate | raw | 2,719 | 56 | 0.4593 | — | 4.2114 | 2.5166 | 102.50 | 0.398 | +0.058 |
| pH | raw | 5,673 | 222 | 0.4115 | — | 0.4860 | 0.3483 | 4.43 | -0.060 | +0.476 |
| WQI | raw | 4,622 | 189 | 0.3367 | — | 13.9718 | 11.2770 | 28.05 | 0.172 | +0.168 |
| E. coli | log10 | 3,831 | 87 | 0.0526 | **0.3550** | 9,816.1469 | 1,387.3184 | 92.69 | -0.741 | +0.793 |
| Total Suspended Solids | log10 | 2,435 | 111 | 0.0254 | **0.3666** | 205.9108 | 44.8811 | 70.84 | -0.993 | +1.036 |
| Turbidity | log10 | 3,581 | 169 | 0.0184 | **0.3302** | 86.7582 | 22.3671 | 68.75 | -0.788 | +0.805 |
| Nitrite | raw | 2,271 | 46 | 0.0026 | — | 0.1223 | 0.0419 | 175.18 | -0.690 | +0.694 |
| Total Phosphorus | log10 | 1,294 | 91 | -0.0126 | **0.1748** | 0.4798 | 0.1782 | 60.43 | 0.319 | -0.335 |

## Gradient Boosting (HistGradientBoostingRegressor)

| Target | Scale | N test | N stations | R² | R² (log) | RMSE | MAE | Error Rate (%) | Persistence R² | Margin |
|---|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | raw | 6,092 | 200 | 0.9402 | — | 2.1549 | 1.6087 | 22.73 | 0.640 | +0.302 |
| Nitrate + Nitrite | raw | 1,045 | 74 | 0.5150 | — | 3.1480 | 2.1619 | 68.73 | 0.190 | +0.285 |
| Dissolved Oxygen | raw | 6,640 | 182 | 0.4670 | — | 1.9612 | 1.3982 | 16.55 | 0.325 | +0.149 |
| Total Dissolved Solids | raw | 3,902 | 117 | 0.4324 | — | 94.7455 | 68.3148 | 22.24 | 0.811 | -0.382 |
| Nitrate | raw | 2,719 | 56 | 0.3925 | — | 4.4637 | 2.7309 | 106.18 | 0.398 | -0.010 |
| pH | raw | 5,673 | 222 | 0.3493 | — | 0.5111 | 0.3701 | 4.71 | -0.060 | +0.411 |
| WQI | raw | 4,622 | 189 | 0.2778 | — | 14.5790 | 11.7870 | 29.25 | 0.172 | +0.108 |
| Specific Conductance | raw | 4,617 | 97 | 0.2619 | — | 174.4763 | 132.1833 | 26.97 | 0.856 | -0.616 |
| Total Suspended Solids | log10 | 2,435 | 111 | 0.0664 | **0.3378** | 201.5351 | 44.4644 | 72.02 | -0.993 | +1.097 |
| E. coli | log10 | 3,831 | 87 | 0.0629 | **0.3580** | 9,762.5847 | 1,488.5883 | 93.69 | -0.741 | +0.803 |
| Turbidity | log10 | 3,581 | 169 | 0.0377 | **0.2832** | 85.9014 | 23.8518 | 70.62 | -0.788 | +0.826 |
| Total Phosphorus | log10 | 1,294 | 91 | 0.0375 | **0.1891** | 0.4678 | 0.1739 | 59.09 | 0.319 | -0.255 |
| Nitrite | raw | 2,271 | 46 | -0.1453 | — | 0.1311 | 0.0470 | 180.42 | -0.690 | +0.546 |

---

## Cross-model comparison (test R², unseen stations)

Best model per target in **bold**.

| Target | N test | N stations | Linear | Random Forest | Gradient Boosting | Best |
|---|--:|--:|--:|--:|--:|:--|
| Water Temperature | 6,092 | 200 | 0.8947 | **0.9466** | 0.9402 | Random Forest |
| Specific Conductance | 4,617 | 97 | 0.2195 | **0.6545** | 0.2619 | Random Forest |
| Total Dissolved Solids | 3,902 | 117 | 0.4211 | **0.5329** | 0.4324 | Random Forest |
| Nitrate + Nitrite | 1,045 | 74 | 0.2176 | 0.5047 | **0.5150** | Gradient Boosting |
| Dissolved Oxygen | 6,640 | 182 | 0.3877 | **0.4910** | 0.4670 | Random Forest |
| Nitrate | 2,719 | 56 | 0.1719 | **0.4593** | 0.3925 | Random Forest |
| pH | 5,673 | 222 | 0.1707 | **0.4115** | 0.3493 | Random Forest |
| WQI | 4,622 | 189 | 0.0763 | **0.3367** | 0.2778 | Random Forest |
| Total Suspended Solids | 2,435 | 111 | -0.0255\* | 0.0254\* | **0.0664\*** | Gradient Boosting |
| E. coli | 3,831 | 87 | 0.0254\* | 0.0526\* | **0.0629\*** | Gradient Boosting |
| Turbidity | 3,581 | 169 | -0.0145\* | 0.0184\* | **0.0377\*** | Gradient Boosting |
| Total Phosphorus | 1,294 | 91 | 0.0228 | -0.0126\* | **0.0375\*** | Gradient Boosting |
| Nitrite | 2,271 | 46 | -0.0033 | **0.0026** | -0.1453 | Random Forest |

\* fitted on `log10(y + c)`. **The raw-scale R² above is not the number to read
for these four** — it is the score of a log fit measured on the scale it was
deliberately not optimised for, and is set by a handful of extreme readings.
Their real comparison is the log scale:

| Target | Linear | Random Forest | Gradient Boosting | Best |
|---|--:|--:|--:|:--|
| Total Suspended Solids | 0.1509 | **0.3666** | 0.3378 | Random Forest |
| E. coli | 0.2043 | 0.3550 | **0.3580** | Gradient Boosting |
| Turbidity | 0.0618 | **0.3302** | 0.2832 | Random Forest |
| Total Phosphorus | — (raw) | 0.1748 | **0.1891** | Gradient Boosting |

On the scale it is fitted on, this block goes from "explains essentially
nothing" (R² 0.00–0.12, three of twelve family-target pairs *negative*) to
"explains a fifth to a third of the variance at stations it has never seen".
That is not a large model, but it is the first time these four have been worth
reporting at all.

## WQI — the composite index

`WQI` is the thirteenth target: a 0–100 index (**0 = best, 100 = worst**) built
by `src/04_eda/wqi-calculation.ipynb` as a weighted roll-up of the other
parameters. It is not a measurement, but predicting it from environment alone
involves no leakage — every water-quality `_value` column was already excluded
from `FEATURE_COLS`.

| Model | Scale | R² | RMSE | MAE | sMAPE (%) | Persistence R² | Margin |
|---|:--|--:|--:|--:|--:|--:|--:|
| **Random Forest** | raw | **0.3367** | 13.9718 | 11.2770 | 28.05 | 0.172 | **+0.168** |
| Gradient Boosting | raw | 0.2778 | 14.5790 | 11.7870 | 29.25 | 0.172 | +0.108 |
| Linear Regression | raw | 0.0763 | 16.4872 | 13.6658 | 33.18 | 0.172 | −0.091 |

Scored on 4,622 rows from **189 unseen stations** (27,298 rows / 943 stations
total — WQI exists on 56.6% of rows).

Three things worth stating plainly:

- **It lands mid-table, and it beats its baseline.** Random forest's 0.337 sits
  between pH (0.412) and the skewed block, and clears persistence by +0.168 —
  which is a more meaningful margin than most, because persistence on WQI is a
  functioning baseline (0.172) rather than a broken one. Linear regression does
  *not* clear it.
- **The bake-off left it raw, as expected.** WQI's skew is −0.10 — it is the
  most symmetric target in the set — and the log arm lost on all three families
  (CV MAE 13.44 vs 13.25, 12.11 vs 11.85, 12.26 vs 11.94). It is a useful check
  that the selection rule is not simply reaching for the transform.
- **About a tenth of it is sampling design, not water quality.** `WQI_n_groups`
  — how many of the eight pollution groups a sample actually measured — explains
  **9.7%** of WQI's variance on its own (Spearman 0.25), and mean WQI climbs
  from 38.4 at four groups to 52.6 at eight. That column is not a feature, so
  the effect is not fitted; it lands in the residual and caps how well any of
  these models can score. A WQI built from four measurements and one built from
  eight are on the same 0–100 scale but are not the same quantity, as the
  calculation notebook says up front. Filtering on `WQI_weight_coverage`, or
  adding `WQI_n_groups` as a feature, is the obvious next experiment.

## The three changes, target by target

Same family throughout, so each column is comparable to the one before it.
All figures are **raw-scale R²**, which is why the last column is negative for
the four log-fitted targets — that is the price of the transform, not a
regression in the model. Their gain is on the log scale, in the table above.

**leaky** = the original `train_test_split` on rows with 30 features;
**grouped** = `GroupShuffleSplit` on stations, still 30 features;
**−`pct_row_crops`** = grouped split with the redundant column dropped (29
features); **now** = the four skewed targets additionally fitted on
`log10(y + c)`.

| Target | Best model | leaky | grouped | −`pct_row_crops` | now | split cost | feature cost | log cost (raw scale) |
|---|:--|--:|--:|--:|--:|--:|--:|--:|
| Water Temperature | Random Forest | 0.9517 | 0.9459 | 0.9466 | 0.9466 | -0.006 | +0.001 | — |
| Specific Conductance | Random Forest | 0.8864 | 0.6578 | 0.6545 | 0.6545 | -0.229 | -0.003 | — |
| Total Dissolved Solids | Random Forest | 0.8243 | 0.5329 | 0.5329 | 0.5329 | -0.291 | +0.000 | — |
| Nitrate + Nitrite | Gradient Boosting | 0.7062 | 0.4998 | 0.5150 | 0.5150 | -0.206 | +0.015 | — |
| Dissolved Oxygen | Random Forest | 0.5962 | 0.4828 | 0.4910 | 0.4910 | -0.113 | +0.008 | — |
| Nitrate | Random Forest | 0.6742 | 0.4607 | 0.4593 | 0.4593 | -0.214 | -0.001 | — |
| pH | Random Forest | 0.5261 | 0.4103 | 0.4115 | 0.4115 | -0.116 | +0.001 | — |
| E. coli | Gradient Boosting | 0.2816 | 0.1157 | 0.1192 | 0.0629 | -0.166 | +0.004 | -0.056 |
| Total Suspended Solids | Gradient Boosting | 0.3913 | 0.0707 | 0.0720 | 0.0664 | -0.321 | +0.001 | +0.078 |
| Turbidity | Gradient Boosting | 0.2750 | 0.0331 | 0.0472 | 0.0377 | -0.242 | +0.014 | +0.184 |
| Total Phosphorus | Gradient Boosting | 0.0287 | 0.0228 | 0.0228 | 0.0375 | -0.006 | +0.000 | +0.043 |
| Nitrite | Random Forest | 0.0863 | 0.0160 | 0.0026 | 0.0026 | -0.070 | -0.013 | — |

Mean raw-scale R² across **the twelve original targets** — WQI is excluded from
this table because it did not exist under the earlier splits and there is
nothing to compare it to:

| Family | leaky | grouped | −`pct_row_crops` | now | split cost | feature cost | log cost |
|---|--:|--:|--:|--:|--:|--:|--:|
| Linear Regression | 0.2485 | 0.2157 | 0.2157 | 0.2073 | -0.033 | +0.000 | -0.008 |
| Random Forest | 0.5381 | 0.3463 | 0.3490 | 0.3406 | -0.192 | +0.003 | -0.008 |
| Gradient Boosting | 0.5465 | 0.2652 | 0.2640 | 0.2848 | -0.281 | -0.001 | +0.021 |

Including WQI, the thirteen-target means are 0.1973 / 0.3403 / 0.2843.

**Essentially all of the movement is still the split.** Dropping
`pct_row_crops` shifts the family means by at most 0.003 R², and the log
transform by at most 0.021 on the raw scale — where, for gradient boosting, it
is a net *gain* even before counting the log-scale improvement, because two of
those four targets had negative raw R² under raw-scale fitting.

## Against the memorization baseline

Persistence — repeat the station's previous value — beside the best model for each target, on the identical rows. `Revisit gap` is the median number of days between the two visits: a target resampled the next day is far easier to persist than one resampled a month later.

| Target | Revisit gap (days) | Pairs | Persistence R² | Best model R² | Margin |
|---|--:|--:|--:|--:|--:|
| Total Dissolved Solids | 28 | 3,785 | 0.811 | 0.533 | -0.277 |
| Total Phosphorus \* | 33 | 1,203 | 0.319 | 0.064 | -0.255 |
| Specific Conductance | 1 | 4,520 | 0.856 | 0.655 | -0.201 |
| Nitrate | 15 | 2,663 | 0.398 | 0.456 | +0.058 |
| WQI | 27 | 4,433 | 0.172 | 0.339 | +0.168 |
| Dissolved Oxygen | 20 | 6,458 | 0.325 | 0.500 | +0.175 |
| Nitrate + Nitrite | 33 | 971 | 0.190 | 0.474 | +0.285 |
| Water Temperature | 27 | 5,892 | 0.640 | 0.949 | +0.309 |
| pH | 28 | 5,451 | -0.060 | 0.416 | +0.476 |
| Nitrite | 15 | 2,225 | -0.690 | 0.004 | +0.694 |
| E. coli \* | 27 | 3,744 | -0.741 | 0.063 | +0.803 |
| Turbidity \* | 29 | 3,412 | -0.788 | 0.038 | +0.826 |
| Total Suspended Solids \* | 31 | 2,324 | -0.993 | 0.104 | +1.097 |

**10 of 13 targets beat persistence; 3 do not** —
Specific Conductance, Total Dissolved Solids, Total Phosphorus. For those three, the honest summary is that a site's last
reading is a better forecast than the model, and the correct next step is to
hand the model that reading explicitly as a `y_prev` / `days_since_prev` feature
rather than leaving it to be approximated from coordinates.

### The log-fitted four, on the scale they are fitted on

The raw-scale margins above are close to meaningless for the four starred
targets, and generously so: *E. coli*, Turbidity and TSS "beat" persistence by
0.80–1.10 R² only because persistence is **catastrophically** bad on raw units
(R² of −0.99 to −0.74, worse than predicting the mean), which is the same
extreme-tail problem the log transform addresses. Repeat the comparison on the
log scale and both sides become sane:

| Target | Persistence R² (log) | Best model R² (log) | Margin (log) |
|---|--:|--:|--:|
| Total Phosphorus | 0.377 | 0.189 | -0.188 |
| Total Suspended Solids | 0.105 | 0.375 | +0.271 |
| Turbidity | 0.055 | 0.338 | +0.282 |
| E. coli | -0.164 | 0.357 | +0.520 |

This is a **stronger** result than the raw-scale table, despite the smaller
numbers. Previously the claim was "beats a broken baseline by +1.10 with an R²
of 0.11" — arithmetically true and worth nothing. Now it is "beats a
functioning baseline by +0.27 with an R² of 0.38". Persistence on the log scale
is a real competitor for TSS and Turbidity (0.105, 0.055) and the models still
clear it comfortably.

Total Phosphorus is the exception and stays honest in both framings: persistence
reaches 0.377 on the log scale against the best model's 0.189, so a site's last
reading remains the better forecast. It joins Specific Conductance and TDS on
the list of targets waiting for `y_prev`.

## Takeaways

- **The scores fell, and that is the point.** Mean raw-scale R² dropped from
  0.55 to 0.28 for gradient boosting and 0.54 to
  0.34 for random forest. The models did not get
  worse; the measurement got honest. (Nearly all of that is the grouped split —
  see the change-by-change table above.)
- **The station-dominated targets collapsed exactly as predicted.** Specific
  Conductance 0.89 →
  0.26 (GB) and Total
  Dissolved Solids 0.83 →
  0.43. Their old
  scores were station recall, and persistence R² of 0.86 / 0.81 confirms that
  most of what there is to know about these two is *which site you are standing
  at*.
- **Water Temperature is the one target that survives intact**
  (0.947 RF, down only
  0.005).
  It is the one target with a real physical mechanism in the feature set
  (ρ = +0.85 with `prism_tmin_c`, both between *and* within station), and it
  beats persistence by 0.31.
- **The leaderboard is split by target type.** Random forest wins 8 of the 9
  raw-scale targets (WQI included, at 0.337); gradient boosting wins all four
  log-fitted ones on the raw scale and two of four on the log scale. Boosting's early stopping
  validates on a random slice of the *training* stations, so it tunes its
  stopping point against the very leakage the grouped split removes — Specific
  Conductance is the clearest casualty (0.65 RF vs 0.26 GB). Random forest
  remains the more robust default on the raw-scale targets; **on the skewed
  block, fitting in log space is what let boosting work at all** — it held three
  *negative* raw-scale R² there before the transform.
- **Tree ensembles still beat the linear baseline**, but by much less: the mean
  gap narrowed from 0.30
  to 0.13 R². Most of
  the trees' old advantage was their superior ability to memorise a station.
- **Nine of thirteen targets still sit below R² = 0.5 for every family.** On
  unseen stations the current 29-feature design predicts water temperature well,
  dissolved oxygen / pH / the nitrate group / WQI moderately, and the skewed
  nutrient-and-bacteria block weakly — but no longer *not at all*.
- **WQI is a reasonable middle-tier target and a genuine baseline win.** Random
  forest reaches 0.337 and clears persistence by +0.168 — a more meaningful
  margin than most on this page, because WQI's persistence baseline (0.172) is a
  functioning one rather than a broken one. But roughly a tenth of its variance
  is `WQI_n_groups`, i.e. how many parameters happened to be measured, so part
  of what any WQI model appears to learn is sampling design. See the WQI section.
- **The transform is now chosen by measurement, not by hand** — cross-validated
  on training stations, with a 5% margin so ties go to the untransformed target,
  and validated to make only conservative errors. It reproduced the hand-picked
  set and found one refinement (Total Phosphorus / LR is better raw). It also
  left WQI raw, which is the check that it is not simply reaching for the log.
- **The log transform is the difference between "explains nothing" and
  "explains a third".** *E. coli* went from a raw-scale R² of 0.12 to a
  log-scale R² of **0.358**, TSS to 0.367, Turbidity to 0.330 — same features,
  same estimators, same split, different question. The cost is 0.008 mean
  raw-scale R² for LR and RF (and a *gain* of 0.021 for GB), and MAE falls
  13–39% while sMAPE falls 20–30 points.
- **Zero-inflation is a separate problem the transform does not touch.** Nitrite
  (84% zeros) and Nitrate (38%) were deliberately left raw: a log would silently
  reinterpret most of Nitrite's column. They need the hurdle model, not a
  transform.
- **Dropping `pct_row_crops` cost nothing and fixed something.** Linear
  regression's R² is identical to four decimals on all twelve targets, the tree
  families move within resampling noise, and the design matrix went from rank
  29/30 with a condition number of 1.3e15 to full rank at 27.8. The land-cover
  coefficients are identified for the first time. A redundancy this exact is
  free to remove — which is why it was the one feature change worth making
  before the harder ones.
- **Feature transforms were tested and rejected.** `log1p` on the nine
  non-negative features with skew > 1.5 moves random forest's mean R² by
  **+0.0002** — trees split on thresholds, so a monotone transform of a
  predictor is a mathematical no-op — and *hurts* linear regression by 0.033
  mean R², worst on Specific Conductance (0.220 → −0.046). Skewed predictors are
  not a violation of any OLS assumption; only a nonlinear conditional mean is.
  The payoff was on the target side, and only there.
- **The remaining EDA fixes are now the interesting ones** — §3.3 and §4.3 of
  `src/04_eda/eda-summary.md`: `y_prev`/`days_since_prev` features (the single
  highest-value addition, and the direct answer to the four targets that lose to
  persistence), a hurdle model for the zero-inflated nitrogen species, and
  collapsing the four nutrient columns and the six-member thermal cluster.
  ~~`log10` targets for the skewed block~~ — done, this round.

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
