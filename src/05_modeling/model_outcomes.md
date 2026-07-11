# Model Outcomes

Held-out test-set results for every water-quality model trained in
`src/05_modeling/`. Generated 2026-07-10 from the executed notebooks.

## Setup (shared by all models)

- **Dataset:** `data/03c_merge_tertiary/epa-full.csv` (48,251 rows × 315 columns)
- **Predictors:** 30 environmental / spatial / temporal features (location, PRISM
  climate, ISU weather, streamflow, soil, land cover, nutrient loading, plus
  day-of-year seasonality and observation year). Other water-quality `_value`
  columns are **excluded** as predictors.
- **Split:** 80% train / 20% test, `random_state=42`.
- **Preprocessing:** `SimpleImputer(median)` on every model; `StandardScaler`
  added for linear regression only.
- **Rows per target** vary because each target keeps only its own non-null,
  in-range measurements (see the `N test` column).

### Metrics

- **R²** — coefficient of determination on the test set (higher is better; 1.0 is perfect).
- **RMSE** — root mean squared error, in the target's own units (lower is better).
- **Error Rate** — symmetric mean absolute percentage error / sMAPE, as a percent
  (lower is better; bounded and robust when the true value is near zero).

Models compared: **Linear Regression**
(`linear_regression/multiple_linear_regression.ipynb`), **Random Forest**
(`random_forest/random_forest.ipynb`), **Gradient Boosting** —
HistGradientBoostingRegressor (`gradient_boosting/gradient_boosting.ipynb`).
Each notebook's final cell saves its 12 fitted pipelines as `<prefix>_<target>.pkl`
into its own folder (`lr_*`, `rf_*`, `gb_*`).

---

## Linear Regression

| Target | N test | R² | RMSE | Error Rate (%) |
|---|--:|--:|--:|--:|
| Water Temperature | 6,941 | 0.8856 | 2.8287 | 27.77 |
| Total Dissolved Solids | 3,603 | 0.3802 | 98.9665 | 22.80 |
| Nitrate | 2,470 | 0.3811 | 4.1861 | 115.69 |
| Dissolved Oxygen | 6,365 | 0.3185 | 2.1934 | 18.23 |
| Specific Conductance | 3,214 | 0.3002 | 166.9870 | 20.80 |
| Nitrate + Nitrite | 932 | 0.2209 | 3.6504 | 80.68 |
| pH | 6,471 | 0.2094 | 0.5890 | 5.60 |
| E. coli | 3,180 | 0.1114 | 7263.2518 | 147.49 |
| Total Suspended Solids | 2,905 | 0.0734 | 202.0689 | 118.55 |
| Turbidity | 4,232 | 0.0597 | 117.2464 | 103.10 |
| Total Phosphorus | 1,178 | 0.0287 | 0.4292 | 74.04 |
| Nitrite | 2,323 | 0.0133 | 0.1437 | 187.12 |

## Random Forest

| Target | N test | R² | RMSE | Error Rate (%) |
|---|--:|--:|--:|--:|
| Water Temperature | 6,941 | 0.9517 | 1.8374 | 17.48 |
| Specific Conductance | 3,214 | 0.8864 | 67.2773 | 6.96 |
| Total Dissolved Solids | 3,603 | 0.8243 | 52.6977 | 9.05 |
| Nitrate | 2,470 | 0.6742 | 3.0372 | 103.22 |
| Nitrate + Nitrite | 932 | 0.6563 | 2.4244 | 59.00 |
| Dissolved Oxygen | 6,365 | 0.5962 | 1.6884 | 13.27 |
| pH | 6,471 | 0.5261 | 0.4560 | 3.85 |
| Total Suspended Solids | 2,905 | 0.3913 | 163.7680 | 78.83 |
| Total Phosphorus | 1,178 | 0.3298 | 0.3566 | 57.04 |
| Turbidity | 4,232 | 0.2750 | 102.9504 | 63.98 |
| E. coli | 3,180 | 0.2601 | 6627.6175 | 115.81 |
| Nitrite | 2,323 | 0.0863 | 0.1383 | 175.49 |

## Gradient Boosting (HistGradientBoostingRegressor)

| Target | N test | R² | RMSE | Error Rate (%) |
|---|--:|--:|--:|--:|
| Water Temperature | 6,941 | 0.9523 | 1.8270 | 18.44 |
| Specific Conductance | 3,214 | 0.8876 | 66.9074 | 7.65 |
| Total Dissolved Solids | 3,603 | 0.8312 | 51.6429 | 9.48 |
| Nitrate + Nitrite | 932 | 0.7062 | 2.2415 | 58.28 |
| Nitrate | 2,470 | 0.6863 | 2.9802 | 105.51 |
| Dissolved Oxygen | 6,365 | 0.5858 | 1.7099 | 13.89 |
| pH | 6,471 | 0.5153 | 0.4612 | 4.01 |
| Total Suspended Solids | 2,905 | 0.4290 | 158.6156 | 92.41 |
| Total Phosphorus | 1,178 | 0.3437 | 0.3528 | 61.88 |
| Turbidity | 4,232 | 0.2866 | 102.1268 | 74.70 |
| E. coli | 3,180 | 0.2816 | 6530.7604 | 124.85 |
| Nitrite | 2,323 | 0.0520 | 0.1409 | 182.72 |

---

## Cross-model comparison (test R²)

Best model per target in **bold**.

| Target | N test | Linear | Random Forest | Gradient Boosting | Best |
|---|--:|--:|--:|--:|:--|
| Water Temperature | 6,941 | 0.8856 | 0.9517 | **0.9523** | Gradient Boosting |
| Specific Conductance | 3,214 | 0.3002 | 0.8864 | **0.8876** | Gradient Boosting |
| Total Dissolved Solids | 3,603 | 0.3802 | 0.8243 | **0.8312** | Gradient Boosting |
| Nitrate + Nitrite | 932 | 0.2209 | 0.6563 | **0.7062** | Gradient Boosting |
| Nitrate | 2,470 | 0.3811 | 0.6742 | **0.6863** | Gradient Boosting |
| Dissolved Oxygen | 6,365 | 0.3185 | **0.5962** | 0.5858 | Random Forest |
| pH | 6,471 | 0.2094 | **0.5261** | 0.5153 | Random Forest |
| Total Suspended Solids | 2,905 | 0.0734 | 0.3913 | **0.4290** | Gradient Boosting |
| Total Phosphorus | 1,178 | 0.0287 | 0.3298 | **0.3437** | Gradient Boosting |
| Turbidity | 4,232 | 0.0597 | 0.2750 | **0.2866** | Gradient Boosting |
| E. coli | 3,180 | 0.1114 | 0.2601 | **0.2816** | Gradient Boosting |
| Nitrite | 2,323 | 0.0133 | **0.0863** | 0.0520 | Random Forest |

## Takeaways

- **Tree ensembles dominate.** Random forest and gradient boosting beat the
  linear baseline on all 12 targets, often by a wide margin (e.g. specific
  conductance 0.30 → 0.89, total dissolved solids 0.38 → 0.83).
- **Gradient boosting wins most targets (9 of 12)**; random forest takes
  dissolved oxygen, pH, and nitrite. The two ensembles are close throughout —
  the choice is effectively a tie, with gradient boosting using far less memory.
- **Strong models:** water temperature (R² ≈ 0.95), specific conductance and
  total dissolved solids (R² ≈ 0.83–0.89) are well predicted from environmental
  drivers alone.
- **Hard targets:** the heavily right-skewed nutrient/bacteria measurements —
  nitrite, turbidity, *E. coli*, total suspended solids, total phosphorus —
  stay low (R² ≤ 0.43) with high error rates on the raw scale. A `log1p` target
  transform is the natural next step for these.
- **Error Rate caveat:** sMAPE is inflated for low-magnitude, spiky targets
  (nitrate, nitrite, *E. coli* all exceed 100%) because small absolute errors
  are large relative to near-zero readings. Read it alongside R² and RMSE, not
  on its own.

## Reproducing

```bash
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/linear_regression/multiple_linear_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/random_forest/random_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/gradient_boosting/gradient_boosting.ipynb
```
