# EDA Summary — `epa-full.csv`

Consolidated findings from the three exploratory notebooks in `src/04_eda/`:

| Notebook | Scope |
|---|---|
| [`univariate-analysis.ipynb`](univariate-analysis.ipynb) | One variable at a time — units, distributions, missingness, censoring, temporal coverage |
| [`bivariate-analysis.ipynb`](bivariate-analysis.ipynb) | One pair at a time — shape, effect size, influence, cluster-robust reliability, autocorrelation in time and space |
| [`multivariate-analysis.ipynb`](multivariate-analysis.ipynb) | The input space as a whole — collinearity, VIF, PCA, mutual information, between/within-station decomposition |

**Input:** `data/03c_merge_tertiary/epa-full.csv` — 48,251 rows × 315 columns, 1,345 monitoring
stations, 555 HUC-12 watersheds, all 99 Iowa counties, 21 organizations, 2015-01-02 → 2025-12-25.
Supporting tables are in `src/04_eda/outputs/` (`bv_` = bivariate, `mv_` = multivariate).

---

## 1. Key findings

### 1.1 The table itself is sound

The merge did its job. These were checked and cleared:

- **Grain is exact.** Every `(MonitoringLocationIdentifier, ActivityStartDateTime)` key appears
  exactly once. No duplicate rows, no null station ids or timestamps.
- **Units are unambiguous.** All 12 targets report in exactly one unit each, with no value ever
  missing its unit. There is no hidden unit mixture masquerading as a bimodal distribution.
- **No physically impossible values.** Temperature −1 to 37 °C, pH 4 to 11.8, DO 0 to 24.6 mg/L,
  no negative concentrations anywhere. The 83 sub-zero temperatures are plausible winter readings.
- **Scope holds.** `StateCode` is constant at 19 throughout; the CDL land-cover `year` matches the
  observation year on 100% of rows.
- **Measurement reproducibility is excellent.** Two samples at the same station on the same day at
  different times correlate at **0.94–0.98** for the six well-sampled physical targets. Whatever
  limits model performance, it is not instrument noise.

### 1.2 The table is mostly ballast

155 of the 315 columns come from three NASS agricultural blocks (`chemapp__` 98, `livestock__` 29,
`cropyield__` 28), and those are also the emptiest — mean completeness 27%, 19%, 30%. **45 columns
are constant**, of which 21 are sparse NASS series carrying a single state-level value and therefore
zero within-Iowa variance. Nine more columns are populated on <5% of rows. The 30 columns the models
actually use are the well-populated ones; the rest is context that has never entered a model.

### 1.3 Targets split into two distributional families

| Family | Targets | Character |
|---|---|---|
| Bounded, near-symmetric | Water Temperature, pH, Dissolved Oxygen | \|skew\| ≤ 0.8, usable on a raw scale |
| Right-skewed, log-normal-ish | everything else | skew up to **84.9** (E. coli), 29.6 (TSS), 13.5 (Nitrite), 12.2 (Turbidity) |

For the second family the mean is not a summary — E. coli's mean is **1,845** against a median of
**160**. Coverage also spans 7×: Water Temperature has 34,705 rows (71.9%), Nitrate + Nitrite only
4,659 (9.7%) and Total Phosphorus 5,886 (12.2%).

### 1.4 Reporting artefacts are pervasive

- **Zero-inflation in the nitrogen species:** Nitrite is **84.1% zeros**, Nitrate **38.3%**. These
  are almost certainly non-detects recorded as 0, not true absences. Nitrite's interquartile range
  is exactly 0.
- **Coarse rounding:** 86% of Nitrate values are whole numbers; pH is exactly 8.0 on **16.3%** of
  observations and DO exactly 8.0 on 12.7%. pH is not measured to whole units in a lab — this is
  rounding upstream, and it puts a floor on achievable RMSE regardless of model.
- **E. coli censoring:** 1,700 values (10.7%) exceed the 2,419.6 MPN/100 mL IDEXX Colilert ceiling.
  Most are round numbers consistent with dilution-scaled re-counts, but the maximum —
  **2,420,000**, ~1,000× the ceiling, 100× the 99th percentile and 4× the next-largest value — is
  not a plausible count.

### 1.5 Signal is weak, and what there is of it is spatial

- **Only one target has a feature above \|ρ\| = 0.5**: Water Temperature ← `prism_tmin_c` at
  **+0.849**, a genuine physical mechanism. Nitrate ← `pct_corn` at +0.489 is second. The other ten
  targets top out between **0.209 and 0.442**.
- Mean \|between-station ρ\| is **0.225** against mean \|within-station ρ\| of **0.120**, and
  **145 of 360 pairs (40%)** have a between term more than twice the within term.
- Whole blocks are purely cross-sectional. **Nutrients: between 0.26, within 0.02. Land cover:
  0.15 vs 0.04.** Nitrate ← `pct_corn` is **+0.714 between stations and −0.001 within one**.
  Sites in corn country have more nitrate; a given site's nitrate on a given day has nothing to do
  with its corn share. **Streamflow is the only block that inverts** (between 0.14, within 0.28).
- **32 pairs flip sign** between the two contrasts (Simpson's paradox). Specific Conductance ←
  discharge is +0.150 between but **−0.364 within** — the dilution effect only appears once you
  stop comparing different rivers.

### 1.6 A correlation coefficient is the wrong summary for most of this data

The shape census over all 360 target × feature pairs:

| Shape | Count | Usable by a single feature? |
|---|---|---|
| irregular | 102 | no |
| flat | 70 | no |
| too few levels | 40 | no |
| hump (∩) | 69 | yes, but ρ reports the average of two opposite slopes |
| valley (∪) | 46 | same |
| monotone ↑ | 19 | yes |
| monotone ↓ | 12 | yes |
| threshold | 2 | yes |

**Only 33 of 360 pairs (9%) are cleanly monotone; 212 (59%) carry no shape a single feature could
be used on.** Two pairs make the case:

- **Water Temperature ← `doy`**: ρ = **+0.24**, conditional median running **1.1 °C → 24.7 °C**
  (1.82 IQRs — among the fifteen largest effects in the table).
- **Specific Conductance ← `LongitudeMeasure`**: ρ = **+0.0007**, conditional median running
  **327 → 703 µS/cm** (2.17 IQRs — the third-largest effect measured).

Neither coefficient is wrong; both are useless. Ranking pairs by correlation and by effect size
agree at only ρ = +0.56.

**`Nitrate ← pct_corn` is a cliff, not a slope:** conditional median 0 mg/L across the four lowest
deciles of corn share, 2 mg/L across the middle five, **10 mg/L in the top decile**. Reported as
ρ = +0.49 it reads as a gradual gradient.

### 1.7 The 30 features are far fewer than 30 facts

- **21 feature pairs exceed \|ρ\| = 0.70, five exceed 0.95**: manure N ↔ manure P **0.987**,
  fertiliser N ↔ fertiliser P **0.977**, `pct_corn` ↔ `pct_row_crops` **0.970**, `prism_tmin_c`
  ↔ `isu_min_feel_c` **0.955** (two agencies measuring the same air).
- Clustering at \|ρ\| ≥ 0.65 collapses 30 columns into **19 groups**, dominated by a six-member
  thermal cluster — `prism_tmax_c`, `prism_tmin_c`, `prism_tdmean_c`, `isu_max_feel_c`,
  `isu_min_feel_c` and `doy_cos` are statistically one variable.
- **The correlation matrix is singular — rank 29 of 30.** The null direction is
  `+0.795·pct_row_crops − 0.502·pct_corn − 0.340·pct_soybean`, i.e. the exact identity
  `pct_row_crops = pct_corn + pct_soybean` (residual 0 on all 48,251 rows, max \|residual\| 1.1e-16).
- **Eight features have VIF ≥ 10**, the four nutrient columns worst at **61.3, 57.1, 55.8, 54.6**.
- PCA: **PC1 carries 19.7%** where independent columns would give 3.3%, and 15 components reach 90%
  where 27 would be needed.

### 1.8 Two targets are one target

**Specific Conductance ↔ Total Dissolved Solids correlate at ρ = 0.984** on 9,285 shared rows — TDS
is conventionally *derived* from conductance. Total Suspended Solids ↔ Turbidity follow at **0.873**
(two measurements of one physical property). Beyond those the structure is loose.

Co-measurement is uneven: the median target pair shares 6,228 rows, but **8 of 144 pairs share
fewer than 200**, and two pairs — Nitrite with Nitrate + Nitrite, and Nitrite with Total Phosphorus
— share **zero rows**. They are measured by programmes that never overlap.

### 1.9 The sampling design, not the water, drives much of the variance

- **Summer-weighted:** 45.4% of observations fall June–August against 9.6% December–February.
  Winter predictions rest on a tenth of the data.
- **Heavy-tailed station volume:** the median station has **7** observations, the busiest has
  **2,878**, and the top 10% of stations account for **58%** of all rows (top 25% → 86%).
- **Programme identity beats chemistry.** `OrganizationIdentifier` explains **ε² = 0.459** of
  Nitrite's rank variance and 0.332 of Specific Conductance's — against ρ² = 0.125 for Nitrite's
  best actual feature. Organization is 100% station-constant and entangled with geography
  (Cramér's V = **0.502** against county).
- **Whether a target is measured at all swings 27–65 percentage points across years.** Dissolved
  Oxygen is present on 31.6% of rows in its worst year and 96.9% in its best; Nitrite 2.7 → 54.2.
  The rows a model trains on are a sample of Iowa's *monitoring programmes*, not of Iowa's water.
- `season` is the one categorical that varies **within** a station (72.2% of stations sampled in
  more than one season) — and it is the largest categorical effect in the table,
  **ε² = 0.647 on Water Temperature**.

### 1.10 Rows are not independent, and it costs an order of magnitude

The station-cluster design effect has a **median of 18.3 and a maximum of 165**. The median pair is
computed on 15,850 rows worth about **1,006 independent observations**. The damage concentrates
exactly where the between-station signal lives — soil ×47.8, location ×31.0, land cover ×29.4,
nutrients ×28.1, streamflow ×27.7 — against climate ×8.9, weather ×6.1, temporal ×6.0.

Of 360 pairs: **324 clear p < 0.05 treating rows as independent, 247 survive the clustered standard
error, 239 survive Benjamini–Hochberg, and only 30 have \|ρ\| ≥ 0.30.**

### 1.11 Persistence and space

**Deseasonalised temporal autocorrelation** splits the targets cleanly. Specific Conductance and TDS
barely decay at all — two visits 2–4 years apart still correlate at **0.77 and 0.75**. Water
Temperature falls from 0.63 to **0.14**, Dissolved Oxygen from 0.45 to 0.19. A target whose
deseasonalised autocorrelation stays high at multi-year lags is a target that is mostly a property
of the site.

**Same-day between-station correlation by distance** (`bv_spatial_autocorrelation.csv`):

| Target | 0–5 km | 20–40 km | 80–160 km |
|---|---|---|---|
| Water Temperature | 0.90 | 0.85 | 0.84 |
| Total Dissolved Solids | 0.78 | 0.69 | 0.07 |
| Nitrate | 0.69 | 0.29 | 0.12 |
| Specific Conductance | 0.67 | 0.61 | −0.10 |
| pH | 0.61 | 0.34 | 0.10 |
| Total Phosphorus | 0.51 | 0.32 | 0.01 |
| Nitrite | 0.34 | 0.03 | −0.59 |

The median station has another within **1.3 km**, but the median pixel of the interpolated map
surface is **8.7 km** from the nearest station, the 95th percentile is **43 km**, and **15% of the
rendered map is more than 20 km from any observation**.

---

## 2. Red flags

Consolidated and deduplicated across the three notebooks, ordered by severity.

| # | Red flag | Severity | Where | Source |
|---|---|---|---|---|
| 1 | Random train/test split leaks stations — every reported R² is optimistic | **critical** | training notebooks | UV 1, MV 1, BV 1 |
| 2 | Model skill comes from station identity, not from measured relationships | **critical** | training notebooks | MV 1 |
| 3 | A zero-feature persistence baseline nearly matches the two best models | **critical** | training notebooks | BV 2 |
| 4 | The dashboard serves weather frozen at the station's last visit | **critical** | `app.py` | UV 2, MV 3 |
| 5 | `pct_row_crops` is an exact sum of two other features — design matrix is singular | **critical** | `FEATURE_COLS` | MV 2 |
| 6 | `error_rate` (sMAPE) is a function of the zero fraction, not model quality | **high** | `model_metrics.csv` | UV 3 |
| 7 | Nitrite (84% zeros) is modelled as a continuous variable | **high** | data + training | UV 4 |
| 8 | Median imputation fills gaps with a systematically different subpopulation | **high** | training notebooks, `app.py` | BV 4 |
| 9 | The map interpolates far past the distance most analytes stay correlated over | **high** | `app.py` | BV 3 |
| 10 | 59% of target–feature pairs carry no usable shape | **high** | `FEATURE_COLS` | BV 5 |
| 11 | The nutrient block is four columns of about two facts, all between-station | **high** | `FEATURE_COLS` | MV 4 |
| 12 | E. coli's tail mixes real dilution counts with a scale artefact | medium | data | UV 5, BV |
| 13 | Nutrient-budget features are up to 8 years stale | medium | data | UV 6 |
| 14 | Specific Conductance and TDS are one target modelled twice (ρ = 0.984) | medium | targets | MV 5 |
| 15 | Half the thermal signal is the calendar, and the calendar is already a feature | medium | interpretation | MV 6 |
| 16 | Programme identity explains more of some targets than any feature does | medium | data | BV 6 |
| 17 | Three target–target correlations rest on fewer than 2,000 shared rows | medium | data | MV 7, BV |
| 18 | `ProviderName` duplicates `OrganizationIdentifier` exactly (V = 1.000) | low | data | BV 7 |

### The critical five, in detail

**1 & 2. Station leakage, and what fills the performance gap.**
`train_test_split(test_size=0.2, random_state=42)` with no grouping puts **99.1–99.7% of test rows
at a station that is also in the training set** (`split_leakage_check.csv`). Latitude and longitude
are model inputs with ~1 unique value per station (1,324 and 1,326 distinct values across 1,345
stations), so a tree can identify the station from coordinates and recall its typical level.

The measurement that proves it: for Specific Conductance the best single feature explains
**ρ² = 0.084** while the reported random-forest R² is **0.886** — a gap of 0.80 that no combination
of these weak, mutually-redundant correlations can plausibly fill. Mutual information names the
culprit: `Specific Conductance ← LongitudeMeasure` has **\|ρ\| = 0.001 but an MI-implied
r_equiv of 0.94**. Longitude is not a mechanism for conductance; it is an *address*. The rank
correlation between "how well a feature identifies the station" and "how much MI it carries" is
**+0.582**, and the eight highest-MI features all have between-station variance share ≥ 0.987.

Best-single-feature ρ² vs. reported RF R² (`mv_correlation_vs_model_r2.csv`), worst gaps first:

| Target | best ρ² | RF R² | gap |
|---|---|---|---|
| Specific Conductance | 0.084 | 0.886 | **+0.802** |
| Total Dissolved Solids | 0.124 | 0.824 | **+0.701** |
| Nitrate + Nitrite | 0.100 | 0.656 | +0.556 |
| pH | 0.090 | 0.526 | +0.436 |
| Nitrate | 0.239 | 0.674 | +0.435 |
| Water Temperature | 0.721 | 0.952 | +0.231 |

Water Temperature — the one target with a real mechanism in the feature set — has the *smallest*
gap. **The models score highest exactly where the correlation structure supports it least.**

**Fix:** re-split with `GroupShuffleSplit` / `GroupKFold` grouped on `MonitoringLocationIdentifier`
and re-report. Expect Specific Conductance and TDS to fall hardest. If the intended use is "predict
at this known station on a future date", a temporal split is the honest test instead — the current
split answers neither question.

**3. A model with no features nearly matches the two best targets.**
"Predict this station's next value with its previous value" (`bv_persistence_baseline.csv`):

| Target | persistence R² | best model R² | margin | median revisit gap |
|---|---|---|---|---|
| Specific Conductance | **0.857** | 0.888 | **+0.030** | 1 day |
| Total Dissolved Solids | **0.709** | 0.831 | **+0.122** | 29 days |
| Total Phosphorus | 0.167 | 0.344 | +0.176 | 33 days |
| Water Temperature | 0.679 | 0.952 | +0.274 | 20 days |
| Nitrate | 0.403 | 0.686 | +0.283 | 15 days |
| Dissolved Oxygen | 0.192 | 0.596 | +0.405 | 21 days |
| E. coli | −3.764 | 0.282 | +4.046 | 26 days |

Conductance's 1-day median gap flatters its baseline; TDS's 29-day gap does not. **Fix:** publish
the persistence baseline beside every reported R², and treat "beats persistence" as the bar for
claiming a target is *predicted* rather than *recalled*.

**4. The dashboard serves weather from the wrong date.**
79% of `prism_tmax_c`'s variance and 79% of `isu_max_feel_c`'s is *within* station — these are
per-observation measurements, not station attributes. `app.py` freezes all 26 base features at the
station's last visit and pairs them with fresh `doy`/`obs_year` from the user's chosen date. **83%
of frozen records come from April–October** (35% from August–September alone), with a **median age
of 1,410 days**. Worse, `groupby.last()` skips NaNs per column, so **16.4% of station rows are
composites** stitched from more than one visit — a combination of conditions that never co-occurred.

Compounding it, six features have **no** within-station variation at all
(`feature_variance_decomposition.csv`), so for a fixed station the features carrying the most
correlation are constant. **The date control effectively moves `doy`/`obs_year` and nothing else.**

**Fix:** either supply date-matched climate/weather at inference (forecast or climatology for the
chosen date), or drop the per-observation weather features from the served set and state explicitly
that the model is climatological.

**5. `pct_row_crops` makes the feature matrix singular.**
It equals `pct_corn + pct_soybean` exactly on all 48,251 rows. The correlation matrix has rank 29
and a genuinely zero eigenvalue; the condition number is infinite. Linear-regression coefficients on
those three columns are **not identified** — the solver returns one of infinitely many equivalent
answers, and any interpretation (including sign) is meaningless. Tree models are unaffected in
accuracy but split importance arbitrarily across the trio.

---

## 3. Best type of predictive model

### 3.1 What the current evidence says

Mean R² across the 12 targets, from `src/05_modeling/model_metrics.csv`:

| Family | mean R² | targets won |
|---|---|---|
| **Gradient Boosting** (`HistGradientBoostingRegressor`) | **0.547** | 9 of 12 |
| Random Forest | 0.538 | 3 of 12 (DO, pH, Nitrite) |
| Linear Regression | 0.249 | 0 |

**Gradient boosting is the right default family**, with random forest a near-tie and a reasonable
ensemble partner. The evidence for this is not just the leaderboard — it is structural:

- **59% of target–feature pairs are flat, irregular, or humped**, and 115 are single-peaked. Trees
  handle non-monotone shapes natively; a linear model cannot see them at all without a hand-built
  basis expansion.
- **`Nitrate ← pct_corn` is a threshold** (0 → 2 → 10 mg/L across deciles). A split-based learner
  represents that in one node; a linear coefficient smears it across the whole range.
- **Skew is harmless for trees** — they are invariant to monotone transforms of a predictor —
  but `StandardScaler` centres and scales without fixing a skew of 84.9. Linear regression is
  least well specified exactly where this data lives.
- **Collinearity does not hurt tree accuracy.** With rank deficiency, a condition number of ∞ and
  eight features at VIF ≥ 10, the linear family is fitting an ill-conditioned system.
- **Imputation is internal.** Both tree families tolerate the 50%-present discharge column;
  `HistGradientBoostingRegressor` handles NaN natively, without any imputer at all.

Keep linear regression only as an interpretable baseline — and once `pct_row_crops` is dropped, its
coefficients become identified for the first time.

### 3.2 But model family is the second-order decision

The three notebooks converge from three directions — univariate split leakage, multivariate
between/within decomposition, bivariate persistence baseline — on the same conclusion: **the gap
between families (0.009 R² between GB and RF) is an order of magnitude smaller than the gap between
the current split and an honest one.** No choice of estimator fixes a metric that is measuring
recall. Fix the split first; then compare families on numbers that mean something.

### 3.3 Per-target formulations

Once the split is honest, one estimator does not fit all twelve targets:

| Targets | Recommended formulation | Why |
|---|---|---|
| Water Temperature | GB on the current features | The one genuine mechanism (ρ = +0.849 with `prism_tmin_c`, both between **and** within station), spatially stationary (0.84 at 320 km), and beats persistence by 0.27 |
| Specific Conductance, TDS | Hierarchical / station-effect model, or **persistence + anomaly** | Deseasonalised autocorrelation of 0.77 / 0.75 at multi-year lags: these are station constants plus a small residual. Model the station level explicitly instead of letting a tree memorise it from coordinates. Report as **one** target (ρ = 0.984) |
| Nitrite, Nitrate | **Hurdle / two-part**: classify P(detect), then regress magnitude given detection | 84.1% and 38.3% zeros. Nitrite's R² of 0.01–0.09 is the correct answer to the wrong question |
| E. coli, TSS, Turbidity, Total Phosphorus | GB on **log10(y)** (after resolving censoring) | Skews 84.9 / 29.6 / 12.2 / 9.2; squared-error loss on the raw scale is fitting the tail |
| Dissolved Oxygen, pH | GB, raw scale | Bounded and near-symmetric; the ~37% integer-valued rounding puts a hard floor on achievable RMSE |
| Nitrate + Nitrite | GB, raw scale | Moderate skew (2.05), adequate behaviour already |

**Two structural upgrades worth more than any estimator swap:**

1. **Add the previous observation at the same station as a feature.** Persistence alone reaches
   R² = 0.857 for Specific Conductance and 0.709 for TDS. That information currently reaches the
   model only as something a tree can approximate by memorising coordinates. Handing it over
   explicitly — as `y_prev` and `days_since_prev` — converts leakage into a legitimate, honest,
   documentable feature, and makes the temporal-split framing natural.
2. **Model within-station anomalies for the station-dominated targets.** Predict each site's
   deviation from its own baseline rather than its absolute level. §6 of the multivariate notebook
   makes clear this is a different and much harder problem than the one the current metrics
   describe — which is precisely why the current metrics look good.

---

## 4. Preparing the data for modeling

### 4.1 Columns to drop from `FEATURE_COLS`

Ordered by confidence. Any of these changes the feature contract, so `app.py` and all three
training notebooks must move together and all **36 `.pkl` files must be retrained**.

| Column | Reason | Confidence |
|---|---|---|
| `pct_row_crops` | Exact sum of `pct_corn + pct_soybean`; sole source of the rank deficiency; carries literally zero information | **certain** |
| `isu_snowd_in` | 99.0% zero; only 15 distinct values; MI-implied r_equiv ≈ 0.05; unbinnable against **all 12** targets | **certain** |
| `isu_snow_in` | 98.8% zero; 32 distinct values; same profile | **certain** |
| 2 of the 4 `npfert__*` / `npmanure__*` columns | VIF 61.3 / 57.1 / 55.8 / 54.6; N↔P correlate at 0.977 and 0.987; four columns spanning ~two dimensions. Keep one N and one P, or replace all four with their first two PCs | high |
| `prism_tdmean_c`, `isu_max_feel_c`, `isu_min_feel_c` | The six-member thermal cluster is statistically one variable; keep `prism_tmax_c` + `prism_tmin_c` (or a single mean) and drop the rest | high |
| `isu_avg_rh` | 83% of its target pairs are flat or irregular; mean \|ρ\| = 0.075, max 0.167 | medium |
| `prism_ppt_mm` | 64.2% zero at the daily grain; unbinnable against all 12 targets. **Replace rather than drop** — see §4.3 | medium |
| `LatitudeMeasure`, `LongitudeMeasure` | ~1 unique value per station: these *are* station identifiers, and they are the mechanism behind flags 1–2. **Decision, not a defect:** drop them if the goal is generalisation to unseen stations; keep them (and say so) if the goal is per-station prediction | conditional |

Per-feature diagnostics for this shortlist are in `outputs/bv_pair_profiles.csv` (shape class per
pair), `outputs/mv_vif.csv` and `outputs/feature_variance_decomposition.csv`.

### 4.2 Columns to drop from the table

Not model features, but dead weight that invites future mistakes:

- **21 constant NASS columns** — the only rows present all share a single state-level value
  (e.g. `chemapp__corn__herbicide__24-d_2-ehe_lb` = 225,000 lb across all 7.8% of rows that have
  it). Zero within-Iowa variance.
- **9 columns populated on <5% of rows** — barley, spring wheat, rye, angora goats.
- **`ProviderName`** — Cramér's V = **1.000** with `OrganizationIdentifier` (USGS-IA → NWIS, the
  other twenty organizations → STORET). If it is kept, document the dependency in `DATA.md`.
- **`StateCode`** — constant 19.
- **23 `_unit` columns** — each is single-valued; collapse to a metadata dictionary rather than
  carrying 23 constant columns through every read.

### 4.3 Transformations and derived features

**Target transforms**

- `log10(y + c)` for E. coli, TSS, Turbidity, Total Phosphorus — and choose `c` explicitly rather
  than letting it fall out of a library default.
- Do **not** log-transform Nitrate or Nitrite without first deciding what their zeros mean (§4.4);
  a log transform silently reinterprets 84% of Nitrite's data.
- Leave Water Temperature, pH and DO on the raw scale.

**Feature engineering**

- **Missing indicators.** `SimpleImputer(add_indicator=True)` for `streamflow_discharge_cfs`
  (50.0% present), `ksat_mean` (63.0%) and `awc_mean` (62.9%). This is not cosmetic — see §4.4.
- **Antecedent precipitation windows.** Replace the daily `prism_ppt_mm` (64.2% zero, unbinnable)
  with 3-, 7- and 30-day rolling sums. Runoff responds to accumulated rainfall, not to whether it
  happened to rain on the sampling day.
- **Expose `np_years_stale` as a feature.** `np_base_year` is only ever 2012 or 2017 and staleness
  runs 0–8 years; letting the model discount stale rows costs nothing.
- **Add `y_prev` and `days_since_prev`** per station (§3.3). The single highest-value addition.
- **Add station-mean and deviation-from-station-mean encodings** for the station-dominated targets,
  so the between/within split is explicit in the design rather than implicit in the coordinates.
- **Keep `season` available**; it is the only categorical that varies within a station (72.2% of
  stations) and the largest categorical effect in the table (ε² = 0.647 on Water Temperature).
- **Do not add `OrganizationIdentifier` or `MonitoringLocationTypeName` as features.** They explain
  a great deal (ε² up to 0.459) but are 100% station-constant and entangled with geography — adding
  them is pure leakage dressed as signal.

**Renaming**

- The `pct_*` land-cover columns are **fractions in [0, 1], not percentages** — the eight
  non-overlapping classes sum to 0.9998 at the median and `pct_corn` averages 0.263 (= 26.3%).
  Rename to `frac_*` or multiply by 100; the current name has already invited one misreading.

### 4.4 Cleaning issues to resolve before retraining

1. **E. coli's censoring rule.** 1,700 values (10.7%) exceed the 2,419.6 Colilert ceiling. The
   single 2,420,000 reading (station `ERG-RC13`) is **five of the six most-influential rows in the
   entire table** — deleting it alone moves `E. coli ← prism_ppt_mm` from r = +0.101 to +0.205. The
   training notebooks' `valid_range` of `(0, 1e6)` silently drops it but keeps 610,000; `app.py`
   applies no filter at all. Write the rule down in the notebook instead of hiding it in a range
   tuple.
2. **Nitrogen zeros are non-detects, not measurements.** Nitrite 84.1%, Nitrate 38.3%. Either
   substitute LOD/2, carry a censoring flag, or use a censored/hurdle model — but do not keep
   treating them as observed zeros.
3. **Rounding is a hard floor on RMSE.** 86% of Nitrate values are whole numbers; pH is exactly 8.0
   on 16.3% of rows. No model can resolve below the recording precision. Say so wherever pH and DO
   RMSE are quoted.
4. **Median imputation is filling gaps with the wrong subpopulation.** Where `ksat_mean` and
   `awc_mean` are missing (37% of rows), the median Specific Conductance is **390 µS/cm against 525**
   where they are present (rank-biserial **+0.42**); `streamflow_discharge_cfs`, missing on 50%,
   shows the same pattern at +0.26. Missingness is informative about the target — the one condition
   under which `SimpleImputer(strategy="median")` is not neutral. Add the indicators (§4.3).
5. **The `groupby.last()` inference row is a composite.** 16.4% of station rows in `app.py`'s
   `STATIONS` frame stitch columns from more than one visit, producing feature combinations that
   never co-occurred in reality. Take a whole row, or serve date-matched values.

### 4.5 Splitting, resampling and metrics

- **Split by station**, not by row: `GroupShuffleSplit` / `GroupKFold` on
  `MonitoringLocationIdentifier`. Alternatively split by time if the use case is forecasting at
  known stations. Report which question is being answered.
- **Stratify any resampling by station and by season** — 45.4% of observations are June–August, and
  the top 10% of stations hold 58% of rows.
- **Report R² with a station-cluster bootstrap interval** (resample stations, refit). With a median
  design effect of 18.3, an R² quoted to three decimals on ~1,000 effective observations is
  precision theatre.
- **Add `n_test_stations` beside `test_rows`** in `model_metrics.csv`. Currently `test_rows` reads
  as 932–6,941 independent observations; they are drawn from 165–751 stations, almost all of which
  are also in training.
- **Publish the persistence baseline next to every R²** — `outputs/bv_persistence_baseline.csv`
  already has it.
- **Drop `error_rate` (symmetric MAPE) for zero-inflated targets.** Its per-row term
  `2·|y − ŷ| / (|y| + |ŷ|)` saturates at 200% whenever `y = 0`, so the metric has a floor set by the
  zero fraction alone: Nitrite's 84.1% zeros force a **~168% floor** and the reported figure is
  175.5%; Nitrate's 38.3% zeros force ~77% against a reported 103%. These numbers are not
  comparable across targets and do not mean "the model is 175% wrong". Report MAE and RMSE on the
  target's own scale.
- **Mask the map beyond a per-target support radius.** Take it from
  `outputs/bv_spatial_autocorrelation.csv` — e.g. the distance at which the same-day correlation
  falls below half its 0–5 km value. For Water Temperature the cubic interpolation is defensible
  at any distance in Iowa; for Nitrate (0.29 by 20–40 km), Total Phosphorus and pH most of the
  coloured area is extrapolation presented as measurement.

---

## 5. What the EDA does not establish

Every relationship reported here is **descriptive**. Between-station contrasts are confounded by
everything that varies geographically — soil, geology, point sources, sampling programme — and
within-station contrasts are confounded by season, which the partial correlations only partly
remove. `Nitrate ← pct_corn`'s threshold at 45% corn share may be a nitrogen-loading mechanism or
may be the signature of the counties where one monitoring programme operates; nothing in these
notebooks separates them.

The open question all three converge on is whether **any** of the twelve targets can be predicted at
a station the model has never seen. A `GroupKFold` re-fit grouped on `MonitoringLocationIdentifier`
would answer it in an afternoon, and it is the prerequisite for every other number in this project.

---

## Appendix: supporting tables

All in `src/04_eda/outputs/` (gitignored — regenerate by running the notebooks).

| Prefix | Notebook | Notable files |
|---|---|---|
| *(none)* | univariate | `column_completeness.csv`, `target_univariate_stats.csv`, `target_value_pileups.csv`, `feature_variance_decomposition.csv`, `split_leakage_check.csv` |
| `bv_` | bivariate | `bv_pair_profiles.csv`, `bv_pair_reliability.csv`, `bv_persistence_baseline.csv`, `bv_spatial_autocorrelation.csv`, `bv_temporal_autocorrelation.csv`, `bv_missingness_vs_target.csv` |
| `mv_` | multivariate | `mv_vif.csv`, `mv_null_space.csv`, `mv_between_within_station.csv`, `mv_correlation_vs_model_r2.csv`, `mv_mutual_information.csv`, `mv_target_overlap.csv` |
