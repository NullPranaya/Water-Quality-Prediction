Us# Merge Plan

**Status: implemented.** This document describes how every dataset catalogued in
`DATA.md` gets combined into a single modeling table, in three tiers of merges
plus a final assembly. All stages are now built — each `P*`/`S*`/`T1` below has a
notebook at `src/03_merge/<id>_*.ipynb` and writes the listed CSV. Verified output
shapes:

| Stage | Notebook | Output | Rows | Cols |
|---|---|---|--:|--:|
| P1 | `P1_wq-daily-environment-merge.ipynb` | `data/03a_merge_primary/wq-daily-environment.csv` | 48,251 | 84 |
| P2 | `P2_station-geo-soil-merge.ipynb` | `data/03a_merge_primary/station-geo-soil.csv` | 1,666 | 28 |
| P3 | `P3_county-agriculture-merge.ipynb` | `data/03a_merge_primary/county-agriculture.csv` | 2,475 | 85 |
| P3b | `P3b_county-nutrient-asof-refresh.ipynb` | `data/03a_merge_primary/county-agriculture-asof.csv` | 1,089 | 24 |
| P4 | `P4_state-chemical-spending-merge.ipynb` | `data/03a_merge_primary/state-chemical-spending.csv` | 10 | 112 |
| P5 | `P5_huc12-landuse-bmp-merge.ipynb` | `data/03a_merge_primary/huc12-landuse-bmp.csv` | 18,854 | 18 |
| P6 | `P6_npdes-facility-merge.ipynb` | `data/03a_merge_primary/npdes-facility.csv` | 14,030 | 32 |
| P7 | `P7_census-population-merge.ipynb` | `data/03a_merge_primary/census-population.csv` | 17,877 | 6 |
| S1 | `S1_wq-geo-soil-daily-merge.ipynb` | `data/03b_merge_secondary/wq-geo-soil-daily.csv` | 48,251 | 102 |
| S2 | `S2_station-year-context-merge.ipynb` | `data/03b_merge_secondary/station-year-context.csv` | 18,326 | 217 |
| T1 | `T1_epa-full-merge.ipynb` | `data/03c_merge_tertiary/epa-full.csv` | 48,251 | 315 (318 after `src/04_eda/wqi-calculation.ipynb` appends 3 `WQI*` columns) |

## Contents

- [1. Datasets & foreign keys](#1-datasets--foreign-keys)
- [2. Primary merges — `03a_merge_primary`](#2-primary-merges--03a_merge_primary)
- [3. Secondary merges — `03b_merge_secondary`](#3-secondary-merges--03b_merge_secondary)
- [4. Tertiary merges — `03c_merge_tertiary`](#4-tertiary-merges--03c_merge_tertiary)
- [5. Full dependency graph](#5-full-dependency-graph)
- [6. Known gaps, deferred datasets, caveats](#6-known-gaps-deferred-datasets-caveats)

---

## 1. Datasets & foreign keys

All paths below are cleaned outputs (`data/tabular/02_clean/...` unless marked
`spatial`, which is `data/spatial/02_clean/...`). This is the complete
input inventory for the merge stages that follow.

| Dataset | Path | Grain | Key(s) |
|---|---|---|---|
| EPA WQ measurements | `water-quality/epa-wq-clean.csv` | 1 row / station + timestamp | **(key)** `MonitoringLocationIdentifier` + `ActivityStartDateTime` |
| EPA stations | `water-quality/epa-stations-clean.csv` | 1 row / station | **(key)** `MonitoringLocationIdentifier`; also carries `HUCEightDigitCode`, `StateCode`, `CountyCode`, lat/lon |
| ISU/IEM climate | `climate/isu-climate-clean.csv` | 1 row / airport station + day | **(key)** `station` + `day`. No shared key with WQ stations — joins via **nearest-station spatial match** on lat/lon (external ISU station-coordinate lookup, inherited from the existing `merge_epa_climate.py` logic; not itself a cataloged DATA.md table) |
| PRISM climate | `climate/prism-iowa-climate-clean.csv` | 1 row / WQ station + date | **(key)** `station_id` (= `MonitoringLocationIdentifier`) + `date` — direct join, no spatial matching needed |
| Crop yields | `agriculture/crop-yields-clean.csv` | 1 row / county + year + commodity + statistic | **(key)** `county_fips` + `year` (fine-grain also includes `ag_district_code`, `commodity_detail`, `statistic`) |
| Livestock inventory | `agriculture/livestock-inventory-clean.csv` | 1 row / county + year + commodity + statistic + band | **(key)** `county_fips` + `year` |
| Crop chemical application | `agriculture/crop-chemical-application-clean.csv` | 1 row / year + commodity + input class | **(key)** `state_fips` (`19`, IA only) + `year` |
| Fertilizer/feed spending | `agriculture/chemical-fertilizer-feed-spending-clean.csv` | 1 row / year + expense category | **(key)** `state_fips` (`19`) + `year` |
| N&P from fertilizer | `agriculture/np-fertilizer-clean.csv` | 1 row / county + year + nutrient + source | **(key)** `county_fips` + `year` (5-yr steps, 1950–2017) |
| N&P from manure | `agriculture/np-manure-clean.csv` | 1 row / county + year + animal category + nutrient | **(key)** `county_fips` + `year` (5-yr steps, 1950–2017) |
| Manure animal inventory | `agriculture/manure-animal-inventory-clean.csv` | 1 row / county + year + animal | **(key)** `county_fips` + `year` |
| Manure weight coefficients | `agriculture/manure-weight-coefficients-clean.csv` | 1 row / year + animal | `year` + `animal` — **reference/lookup table already baked into the manure figures above; not merged further** (see §6) |
| BMP — HUC-8 tidy | `bmp/iowa-nrs-bmp-huc8-clean.csv` | 1 row / HUC-8 + year + practice | **(key)** `huc8_code` + `year` |
| BMP — full tracking export | `bmp/iowa-nrs-tracking-clean.csv` | 1 row / indicator record | **(key)** `huc8` / `huc12` + `year` — richer superset of the row above; **excluded from the default path** (see §6) |
| NPDES DMRs | `npdes/npdes-dmrs-clean.csv` | 1 row / permit + outfall + parameter + period | **(key)** `npdes_id` + `fiscal_year` |
| NPDES ATTAINS | `npdes/npdes-attains-clean.csv` | 1 row / permit + assessment unit + cycle | **(key)** `npdes_id` + `assessment_unit_id` |
| NPDES catchments | `npdes/npdes-catchments-clean.csv` | 1 row / permit | **(key)** `npdes_id`; also `nhdplusid`, `wbd_huc12` |
| ECHO facilities | `npdes/echo-facilities-clean.csv` | 1 row / permit | **(key)** `npdes_id` |
| ECHO NAICS | `npdes/echo-naics-clean.csv` | 1 row / permit + code | **(key)** `npdes_id` |
| ECHO SIC | `npdes/echo-sics-clean.csv` | 1 row / permit + code | **(key)** `npdes_id` |
| USGS discharge | `streamflow/usgs-iowa-discharge-clean.csv` | 1 row / gauge + date | **(key)** `site_no` + `date` |
| USGS gauges | `streamflow/usgs-iowa-gauges-clean.csv` | 1 row / gauge | **(key)** `site_no`; also `huc8`, `county_fips`, lat/lon |
| SSURGO soil attributes | `soil/ssurgo-iowa-attributes-clean.csv` | 1 row / map unit | **(key)** `mukey` |
| CDL land use | `landuse/cdl-huc12-fractions-clean.csv` | 1 row / HUC-12 + year | **(key)** `huc12_code` + `year` |
| Census population 2010–2020 | `census/iowa-census-population-2010-2020-clean.csv` | 1 row / city + year + estimate type | **(key)** `place` + `year` — **city grain, not county**; see §6 |
| Census population 2020–2025 | `census/iowa-census-population-2020-2025-clean.csv` | same | same |
| Station → HUC-12/10/8 crosswalk (spatial) | `nhdplus/wbd-huc12-station-crosswalk-clean.csv` | 1 row / station | **(key)** `MonitoringLocationIdentifier` → `huc12_code` / `huc10_code` / `huc8_code` |
| Station → soil map unit crosswalk (spatial) | `ssurgo/ssurgo-mapunit-station-crosswalk-clean.csv` | 1 row / station | **(key)** `MonitoringLocationIdentifier` → `mukey` (~65% coverage) |

**Derived keys needed (not present verbatim in any file):**

- `county_fips` for stations = `StateCode` + `CountyCode` from `epa-stations-clean.csv`, zero-padded and concatenated to the standard 5-digit FIPS used by the agriculture tables.
- `huc8_code` from a `huc12_code` = the first 8 characters of the HUC-12 code (used to join HUC-12 land use to HUC-8 BMP data, and to join HUC-8 BMP/NPDES catchment data down to individual stations).
- `year` for the tertiary join = `YEAR(ActivityStartDate)` extracted from the WQ measurement timestamp.

---

## 2. Primary merges — `03a_merge_primary`

Preliminary merges: both (or all) inputs come straight from `02_clean` /
`spatial/02_clean`. No dependency on any other merge output.

Outputs are written to `data/03a_merge_primary/` (at the `data/` root, **not**
under `data/tabular/`); secondary/tertiary tiers write to
`data/03b_merge_secondary/` and `data/03c_merge_tertiary/`. Output filenames
carry **no `-clean`/`-merged` suffix** — the bare stage name plus `.csv`.

| # | Output | Inputs (all `02_clean`) | Join | Grain |
|---|---|---|---|---|
| **P1** | `wq-daily-environment.csv` | `epa-wq-clean.csv` + `epa-stations-clean.csv` + `prism-iowa-climate-clean.csv` + `isu-climate-clean.csv` + `usgs-iowa-discharge-clean.csv` + `usgs-iowa-gauges-clean.csv` | station id (direct) for WQ/stations/PRISM; **nearest-neighbor spatial match** on lat/lon for ISU climate and USGS discharge, keyed on date | 1 row / WQ measurement event (station + timestamp) |
| **P2** | `station-geo-soil.csv` | `epa-stations-clean.csv` + `wbd-huc12-station-crosswalk-clean.csv` (spatial) + `ssurgo-mapunit-station-crosswalk-clean.csv` (spatial) + `ssurgo-iowa-attributes-clean.csv` | `MonitoringLocationIdentifier`, then `mukey` | 1 row / station |
| **P3** | `county-agriculture.csv` | `crop-yields-clean.csv` + `livestock-inventory-clean.csv` + `np-fertilizer-clean.csv` + `np-manure-clean.csv` + `manure-animal-inventory-clean.csv` — each **pivoted wide** first (commodity/statistic/animal/nutrient values → columns) to avoid a many-rows-per-county-year fan-out | `county_fips` + `year` | 1 row / county + year |
| **P3b** | `county-agriculture-asof.csv` | `np-fertilizer-clean.csv` + `np-manure-clean.csv` + `livestock-inventory-clean.csv` (cattle/hog head counts) | derived: **backward as-of** match on `county_fips` + `year`, plus a partial manure refresh (see §6) | 1 row / county + year, **dense for 2015–2025** |
| **P4** | `state-chemical-spending.csv` | `crop-chemical-application-clean.csv` + `chemical-fertilizer-feed-spending-clean.csv`, pivoted wide | `state_fips` + `year` | 1 row / year (IA only — no spatial variation) |
| **P5** | `huc12-landuse-bmp.csv` | `cdl-huc12-fractions-clean.csv` + `iowa-nrs-bmp-huc8-clean.csv` (pivoted wide by `practice_type`) | `huc8_code = left(huc12_code, 8)` + `year` | 1 row / HUC-12 + year (BMP values are broadcast to every child HUC-12 of a HUC-8 — see §6) |
| **P6** | `npdes-facility.csv` | `npdes-dmrs-clean.csv` (**aggregated** to `npdes_id` + `fiscal_year` — e.g. total exceedances, violation count, mean `dmr_value` per parameter — to avoid the per-outfall-per-parameter fan-out) + `npdes-catchments-clean.csv` + `echo-facilities-clean.csv` + `echo-naics-clean.csv` + `echo-sics-clean.csv` + `npdes-attains-clean.csv` | `npdes_id` | 1 row / permit + fiscal year, carrying `wbd_huc12`/`huc8` for downstream spatial joins |
| **P7** | `census-population.csv` | `iowa-census-population-2010-2020-clean.csv` concatenated with `iowa-census-population-2020-2025-clean.csv` (dedup overlap year 2020) | schema union | 1 row / city + year + estimate type — **held here, not merged further** (see §6) |

---

## 3. Secondary merges — `03b_merge_secondary`

Both inputs are `03a` outputs, or one `03a` output + one `02_clean` table.

| # | Output | Inputs | Join | Grain |
|---|---|---|---|---|
| **S1** | `wq-geo-soil-daily.csv` | P1 (`03a`) + P2 (`03a`) | `MonitoringLocationIdentifier` | 1 row / measurement event, now carrying HUC-12/10/8, `mukey`, and soil attributes alongside daily climate/streamflow |
| **S2** | `station-year-context.csv` | P2 (`03a`, gives `huc12_code`/`huc8_code`/derived `county_fips` per station) + P5 (`03a`, HUC-12 land use/BMP) + P6 (`03a`, NPDES facility context aggregated to `huc8` + `fiscal_year`) + **P3** (`03a`, county **crop/livestock** — annual, already dense) + **P3b** (`03a`, county **N&P fertilizer/manure** — as-of-matched & refreshed, dense for 2015–2025) + P4 (`03a`, state chemical spending) | `huc12_code` + `year` (P5 land use); `huc8_code` + `year` (P6 NPDES); derived `county_fips` + `year` for agriculture; `year` alone for state spending | 1 row / station + year — every slow-moving/annual covariate broadcast to the station via its watershed, county, and state membership |

> **Agriculture split across P3 / P3b.** S2 draws the annual, natively-dense
> crop-yield and livestock columns from **P3**, but the N&P fertilizer/manure
> columns from **P3b** — *not* from P3's raw N&P columns, which are quinquennial
> and effectively empty inside 2015–2025 (only 2017 has a native value). Joining
> P3's raw N&P on exact `year` would null out ~91% of station-years; P3b is the
> as-of-matched, 2015–2025-dense version built for exactly this join. Pull only
> the `cropyield__*` / `livestock__*` blocks from P3 and the `npfert__*` /
> `npmanure__*` blocks (plus provenance columns) from P3b.

> **As built.** S1 keeps P1's copies of the columns it shares with P2
> (`OrganizationIdentifier`, lat/lon, `StateCode`, etc.), dropping them from the
> P2 side, so it adds only the 18 new geo/soil columns → 48,251 × 102. S2 is a
> full station × year grid over the WQ window **2015–2025** (1,666 stations ×
> 11 years = 18,326 rows × 217 cols); trailing-year nulls are expected where an
> input stops short (P4 ends 2024; P3 crop/livestock ends 2025). **P6 is
> aggregated to `huc8` + year**, not HUC-12: every station's HUC-8 contains NPDES
> facilities (100% coverage) whereas only ~62% of station HUC-12s do, so HUC-8
> gives a meaningful point-source signal for every station.

---

## 4. Tertiary merges — `03c_merge_tertiary`

Uses `03b` outputs (subsequent to secondary).

| # | Output | Inputs | Join | Grain |
|---|---|---|---|---|
| **T1** | `epa-full.csv` | S1 (`03b`, daily measurement + geo + soil + climate + streamflow) + S2 (`03b`, station + year context: agriculture, land use, BMP, NPDES proximity, chemical spending) | `MonitoringLocationIdentifier` + `YEAR(ActivityStartDate)` | 1 row / WQ measurement event — **the single final modeling table**, superseding today's `data/tabular/merged/epa-climate-merged.csv` |

`T1` is the terminal output: every WQ measurement, its full daily climate/streamflow record, its static station geography and soil, and every annual watershed/county/state contextual variable, in one CSV.

> **As built.** T1 preserves S1's grain exactly (48,251 measurement rows) and
> left-joins S2 on `MonitoringLocationIdentifier` + derived `year`, dropping the
> membership keys S2 shares with S1 (`county_fips`, `huc12_code`, `huc8_code`) so
> only new annual-context columns are added → **48,251 × 315**. `app.py` reads
> `epa-full.csv` directly via its own `FEATURE_COLS` contract (see `CLAUDE.md`);
> the legacy `epa-climate-merged.csv` is no longer used at runtime.
>
> **Post-merge (out of scope for this document).** `src/04_eda/wqi-calculation.ipynb`
> appends 3 more columns (`WQI`, `WQI_n_groups`, `WQI_weight_coverage`) to
> `epa-full.csv` in place after the merge runs, bringing it to 318 columns.
> See `DATA.md` §T1 for what they mean.

---

## 5. Full dependency graph

Read `X ◀── a + b` as "output `X` is built from inputs `a` and `b`". All
primary (`P*`) inputs are `02_clean` / `spatial/02_clean` tables; secondary and
tertiary stages consume the outputs above them.

```
Primary — data/03a_merge_primary/
  P1  wq-daily-environment.csv    ◀── epa-wq + epa-stations + prism-iowa-climate + isu-climate
                                        + usgs-iowa-discharge + usgs-iowa-gauges   (last two: nearest-neighbor spatial match)
  P2  station-geo-soil.csv        ◀── epa-stations + wbd-huc12-station-crosswalk (spatial)
                                        + ssurgo-mapunit-station-crosswalk (spatial) + ssurgo-iowa-attributes
  P3  county-agriculture.csv      ◀── crop-yields + livestock-inventory + np-fertilizer + np-manure
                                        + manure-animal-inventory   (each pivoted wide)
  P3b county-agriculture-asof.csv ◀── np-fertilizer + np-manure + livestock-inventory
                                        [backward as-of + partial manure refresh — see §6]
  P4  state-chemical-spending.csv ◀── crop-chemical-application + chemical-fertilizer-feed-spending   (pivoted wide)
  P5  huc12-landuse-bmp.csv       ◀── cdl-huc12-fractions + iowa-nrs-bmp-huc8   (huc8 = left(huc12,8))
  P6  npdes-facility.csv          ◀── npdes-dmrs (aggregated) + npdes-catchments + echo-facilities
                                        + echo-naics + echo-sics + npdes-attains
  P7  census-population.csv       ◀── iowa-census-population-2010-2020 + iowa-census-population-2020-2025
                                        [terminal — see §6, not wired into T1]

Secondary — data/03b_merge_secondary/
  S1  wq-geo-soil-daily.csv       ◀── P1 + P2
  S2  station-year-context.csv    ◀── P2 + P5 + P6 + P3[crop/livestock] + P3b[N&P] + P4

Tertiary — data/03c_merge_tertiary/
  T1  epa-full.csv                ◀── S1 + S2   [the single final modeling table]
```

---

## 6. Known gaps, deferred datasets, caveats

- **Census population has no join path to the rest of the pipeline.** Its key is
  `place` (city name), not `county_fips` or a station identifier, and no
  city→county or city→coordinate crosswalk exists anywhere in `02_clean`. `P7`
  is produced but left unintegrated; wiring it in would require either adding a
  new crosswalk dataset or accepting a lossy name-matching join, both out of
  scope for this plan.
- **BMP HUC-12/HUC-8 granularity mismatch.** `iowa-nrs-bmp-huc8-clean.csv` only
  reports at HUC-8 resolution, so `P5` broadcasts the same HUC-8 BMP values to
  every child HUC-12 — it is not a true sub-division of adoption within the
  HUC-8, just a repetition. Acceptable for now but worth flagging in any
  BMP-effect analysis.
- **Ag temporal overlap is narrow — resolved for N&P by `P3b`.**
  `np-fertilizer-clean.csv` / `np-manure-clean.csv` run in ~5-year steps from
  1950–2017, while `crop-yields-clean.csv` / `livestock-inventory-clean.csv` run
  annually 2015–2025. `P3`'s outer join is therefore sparse: inside the WQ window
  2015–2025 **only 2017 has a native N&P value**, so a straight exact-`year` join
  to station-years would leave the N&P/manure columns null for ~91% of rows.
  `P3b` fixes this by producing a 2015–2025-dense nutrient table:
  - **Backward as-of matching** — each year `Y` is anchored to the most recent
    census year `≤ min(Y, 2017)` (2015–2016 → 2012, 2018–2025 carry 2017),
    recorded in `np_base_year` / `np_years_stale`. Backward (`≤`) avoids using
    future census data to represent an earlier year.
  - **Fertilizer** → pure carry-forward (no county-resolution recent driver;
    `total` recomputed as `farm + nonfarm` because the raw `total` source is null
    at the 2012/2017 base years).
  - **Partial manure refresh** — the 2017 baseline is scaled by observed
    head-count change (`manure_2017 × head_Y / head_2017`, holding Falcone's 2017
    per-head nutrient rate fixed): **cattle** refreshed annually from
    `CATTLE, INCL CALVES` SURVEY inventory; **hogs** refreshed only at the 2022
    census step (annual county hog inventory isn't published) and carried from
    there; **poultry/other** frozen at 2017. `Total` is recomputed from the
    refreshed components. Provenance flags: `manure_cattle_refreshed`,
    `manure_hogs_refreshed`, `cattle_head_ratio`, `hog_head_ratio`.
  - **Known limitation:** no *annual* hog/CAFO trend — hogs move in a single 2022
    step, not a curve.

  The raw quinquennial N&P columns in `P3` are retained as an archival/reference
  view; `S2` should draw N&P from `P3b`, not `P3` (see §3 note).
- **`manure-weight-coefficients-clean.csv` is excluded** — it's the coefficient
  table already used upstream to produce the adjusted head counts in
  `manure-animal-inventory-clean.csv`; merging it again would double-apply the
  adjustment. It is also *not* usable to convert head counts into nutrient mass
  for `P3b`: it holds **live-weight** factors, not kg-N/kg-P excretion per head —
  which is why `P3b` ratio-scales against Falcone's own 2017 N&P figures instead.
- **`iowa-nrs-tracking-clean.csv` is excluded from the default path** in favor
  of the smaller, already-tidy `iowa-nrs-bmp-huc8-clean.csv`. The tracking
  export is a richer superset (funding source, practice-level detail,
  HUC-12-level rows) that could replace `P5`'s BMP half in a future revision if
  more granularity is needed.
- **NPDES DMRs require aggregation before joining**, since the raw grain is one
  row per permit/outfall/parameter/period — a straight join would fan out the
  final table by that many rows per permit. `P6` pre-aggregates to
  `npdes_id` + `fiscal_year`.
- **ISU climate and USGS streamflow both need nearest-neighbor spatial matching**
  in `P1`, not a direct key join. The ISU match also depends on an external
  ISU/IEM airport-station coordinate list that isn't itself a `DATA.md`-cataloged
  dataset — it's inherited from the existing `src/03_merge/merge_epa_climate.py`
  logic and should be carried forward rather than re-derived.
- **The existing `src/03_merge/` notebooks/scripts predate this plan** and, per
  the `data-path-migration-incomplete` note, still reference the old
  `data/tabular/<domain>/{raw,clean}` layout in places. They should be treated
  as superseded by the `03a`/`03b`/`03c` structure above rather than extended
  further — `epa-merge-station-and-wq.ipynb` and `merge_epa_climate.py`'s logic
  map onto `P1`/`P2` here, and `merge-epa-climate-with-agriculture.ipynb` maps
  onto `P3`/`S2`.
