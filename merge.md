Us# Merge Plan

Plan only — nothing in this document has been implemented yet. It describes how
every dataset catalogued in `DATA.md` gets combined into a single modeling table,
in three tiers of merges plus a final assembly.

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

| # | Output | Inputs (all `02_clean`) | Join | Grain |
|---|---|---|---|---|
| **P1** | `wq-daily-environment-clean.csv` | `epa-wq-clean.csv` + `epa-stations-clean.csv` + `prism-iowa-climate-clean.csv` + `isu-climate-clean.csv` + `usgs-iowa-discharge-clean.csv` + `usgs-iowa-gauges-clean.csv` | station id (direct) for WQ/stations/PRISM; **nearest-neighbor spatial match** on lat/lon for ISU climate and USGS discharge, keyed on date | 1 row / WQ measurement event (station + timestamp) |
| **P2** | `station-geo-soil-clean.csv` | `epa-stations-clean.csv` + `wbd-huc12-station-crosswalk-clean.csv` (spatial) + `ssurgo-mapunit-station-crosswalk-clean.csv` (spatial) + `ssurgo-iowa-attributes-clean.csv` | `MonitoringLocationIdentifier`, then `mukey` | 1 row / station |
| **P3** | `county-agriculture-merged.csv` | `crop-yields-clean.csv` + `livestock-inventory-clean.csv` + `np-fertilizer-clean.csv` + `np-manure-clean.csv` + `manure-animal-inventory-clean.csv` — each **pivoted wide** first (commodity/statistic/animal/nutrient values → columns) to avoid a many-rows-per-county-year fan-out | `county_fips` + `year` | 1 row / county + year |
| **P4** | `state-chemical-spending-merged.csv` | `crop-chemical-application-clean.csv` + `chemical-fertilizer-feed-spending-clean.csv`, pivoted wide | `state_fips` + `year` | 1 row / year (IA only — no spatial variation) |
| **P5** | `huc12-landuse-bmp-merged.csv` | `cdl-huc12-fractions-clean.csv` + `iowa-nrs-bmp-huc8-clean.csv` (pivoted wide by `practice_type`) | `huc8_code = left(huc12_code, 8)` + `year` | 1 row / HUC-12 + year (BMP values are broadcast to every child HUC-12 of a HUC-8 — see §6) |
| **P6** | `npdes-facility-merged.csv` | `npdes-dmrs-clean.csv` (**aggregated** to `npdes_id` + `fiscal_year` — e.g. total exceedances, violation count, mean `dmr_value` per parameter — to avoid the per-outfall-per-parameter fan-out) + `npdes-catchments-clean.csv` + `echo-facilities-clean.csv` + `echo-naics-clean.csv` + `echo-sics-clean.csv` + `npdes-attains-clean.csv` | `npdes_id` | 1 row / permit + fiscal year, carrying `wbd_huc12`/`huc8` for downstream spatial joins |
| **P7** | `census-population-merged.csv` | `iowa-census-population-2010-2020-clean.csv` concatenated with `iowa-census-population-2020-2025-clean.csv` (dedup overlap year 2020) | schema union | 1 row / city + year + estimate type — **held here, not merged further** (see §6) |

---

## 3. Secondary merges — `03b_merge_secondary`

Both inputs are `03a` outputs, or one `03a` output + one `02_clean` table.

| # | Output | Inputs | Join | Grain |
|---|---|---|---|---|
| **S1** | `wq-geo-soil-daily-clean.csv` | P1 (`03a`) + P2 (`03a`) | `MonitoringLocationIdentifier` | 1 row / measurement event, now carrying HUC-12/10/8, `mukey`, and soil attributes alongside daily climate/streamflow |
| **S2** | `station-year-context-clean.csv` | P2 (`03a`, gives `huc12_code`/`huc8_code`/derived `county_fips` per station) + P5 (`03a`, HUC-12 land use/BMP) + P6 (`03a`, NPDES facility context aggregated to `huc12`/`huc8` + `fiscal_year`) + P3 (`03a`, county agriculture) + P4 (`03a`, state chemical spending) | `huc12_code`/`huc8_code` + `year` for the watershed layers; derived `county_fips` + `year` for agriculture; `year` alone for state spending | 1 row / station + year — every slow-moving/annual covariate broadcast to the station via its watershed, county, and state membership |

---

## 4. Tertiary merges — `03c_merge_tertiary`

Uses `03b` outputs (subsequent to secondary).

| # | Output | Inputs | Join | Grain |
|---|---|---|---|---|
| **T1** | `epa-full-merged.csv` | S1 (`03b`, daily measurement + geo + soil + climate + streamflow) + S2 (`03b`, station + year context: agriculture, land use, BMP, NPDES proximity, chemical spending) | `MonitoringLocationIdentifier` + `YEAR(ActivityStartDate)` | 1 row / WQ measurement event — **the single final modeling table**, superseding today's `data/tabular/03_merged/epa-climate-merged.csv` |

`T1` is the terminal output: every WQ measurement, its full daily climate/streamflow record, its static station geography and soil, and every annual watershed/county/state contextual variable, in one CSV.

---

## 5. Full dependency graph

```
02_clean/water-quality/epa-wq-clean.csv ────────────┐
02_clean/water-quality/epa-stations-clean.csv ───────┤
02_clean/climate/prism-iowa-climate-clean.csv ───────┼──▶ P1 wq-daily-environment-clean.csv ──┐
02_clean/climate/isu-climate-clean.csv ──────────────┤                                          │
02_clean/streamflow/usgs-iowa-discharge-clean.csv ───┤                                          │
02_clean/streamflow/usgs-iowa-gauges-clean.csv ──────┘                                          │
                                                                                                  │
02_clean/water-quality/epa-stations-clean.csv ───────┐                                           │
spatial/02_clean/nhdplus/wbd-huc12-station-crosswalk ┤                                           │
spatial/02_clean/ssurgo/ssurgo-mapunit-station-x ────┼──▶ P2 station-geo-soil-clean.csv ─┬───────┼──▶ S1 wq-geo-soil-daily-clean.csv ──┐
02_clean/soil/ssurgo-iowa-attributes-clean.csv ──────┘                                   │       │                                     │
                                                                                          │                                             │
02_clean/landuse/cdl-huc12-fractions-clean.csv ──────┐                                   │                                             │
02_clean/bmp/iowa-nrs-bmp-huc8-clean.csv ────────────┴──▶ P5 huc12-landuse-bmp-merged.csv ┤                                            │
                                                                                          │                                             │
02_clean/npdes/*.csv (6 files) ──────────────────────────▶ P6 npdes-facility-merged.csv ─┤                                             │
                                                                                          │                                             ├──▶ T1 epa-full-merged.csv
02_clean/agriculture/{crop-yields,livestock,np-fert,np-manure,manure-animal-inv} ────────▶ P3 county-agriculture-merged.csv ─┤          │
                                                                                                                              │          │
02_clean/agriculture/{crop-chem-app,fert-feed-spending} ─────────────────────────────────▶ P4 state-chemical-spending-merged.csv ┤     │
                                                                                                                              │          │
                                          (P2 + P5 + P6 + P3 + P4) ──────────────────▶ S2 station-year-context-clean.csv ────┴──────────┘

02_clean/census/*.csv (2 files) ──────────────────────▶ P7 census-population-merged.csv   [terminal — see §6, not wired into T1]
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
- **Ag temporal overlap is narrow.** `np-fertilizer-clean.csv` / `np-manure-clean.csv`
  run in ~5-year steps from 1950–2017, while `crop-yields-clean.csv` /
  `livestock-inventory-clean.csv` run annually 2015–2025. `P3`'s outer join on
  `county_fips` + `year` will be sparse outside the ~2015–2017 overlap window.
- **`manure-weight-coefficients-clean.csv` is excluded** — it's the coefficient
  table already used upstream to produce the adjusted head counts in
  `manure-animal-inventory-clean.csv`; merging it again would double-apply the
  adjustment.
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
