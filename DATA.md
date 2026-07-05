# Data Dictionary

A catalog of every dataset in this repository: where it comes from, and what each
column measures. Paths reflect the current `data/tabular/01_raw` · `02_clean` ·
`03_merged` layout (raw inputs, cleaned tables, and merged modeling tables).

Units are noted where the source defines them. Identifiers used to join datasets
together are flagged as **(key)**.

## Contents

- [Raw Data Footprint](#raw-data-footprint)
- [Water Quality](#water-quality)
  - [EPA Water Quality Measurements (`epa-wq.csv`)](#epa-water-quality-measurements--epa-wqcsv)
  - [EPA Monitoring Stations (`epa-stations.csv`)](#epa-monitoring-stations--epa-stationscsv)
  - [Cleaned outputs](#cleaned-water-quality-outputs)
- [Climate](#climate)
  - [ISU / IEM Climate (`isu-climate.csv`)](#isu--iem-climate--isu-climatecsv)
  - [PRISM Climate (`prism-iowa-climate.csv`)](#prism-climate--prism-iowa-climatecsv)
  - [Cleaned outputs](#cleaned-climate-outputs)
- [Agriculture](#agriculture)
  - [USDA NASS tables](#usda-nass-tables)
  - [Cleaned USDA NASS outputs](#cleaned-usda-nass-outputs)
  - [USGS County N & P Inputs (`N-P_from_*.xlsx`)](#usgs-county-n--p-inputs--n-p_from_xlsx)
  - [Cleaned N & P outputs](#cleaned-n--p-outputs)
- [Conservation BMPs (Iowa NRS)](#conservation-bmps-iowa-nrs)
  - [Cleaned BMP outputs](#cleaned-bmp-outputs)
- [NPDES Compliance & Permits](#npdes-compliance--permits)
  - [Discharge Monitoring Reports (`NPDES_DMRS_FY*.csv`)](#discharge-monitoring-reports--npdes_dmrs_fycsv)
  - [ATTAINS Assessments (`NPDES_ATTAINS_AU_SUMMARIES.csv`)](#attains-assessments--npdes_attains_au_summariescsv)
  - [NPDES Catchments (`NPDES_CATCHMENTS.csv`)](#npdes-catchments--npdes_catchmentscsv)
  - [ECHO Facilities / NAICS / SIC](#echo-facilities--naics--sic)
  - [Cleaned NPDES outputs](#cleaned-npdes-outputs)
- [Streamflow](#streamflow)
  - [Cleaned streamflow outputs](#cleaned-streamflow-outputs)
- [Soil (SSURGO)](#soil-ssurgo)
  - [Cleaned soil output](#cleaned-soil-output)
- [Land Use (Cropland Data Layer)](#land-use-cropland-data-layer)
  - [Cleaned land use output](#cleaned-land-use-output)
- [Census / Demographics](#census--demographics)
  - [Cleaned census outputs](#cleaned-census-outputs)
- [Spatial Layers](#spatial-layers)
  - [Cleaned spatial outputs](#cleaned-spatial-outputs)
- [Imagery & Text](#imagery--text)

---

## Raw Data Footprint

Sizes of the **raw** downloaded inputs only (under `data/.../01_raw/` and the raw
`data/spatial/01_raw/` layers); cleaned/merged derivatives are excluded. Rows
exclude the header; `MB` is on-disk file size (1 MB = 1024 KB).

### Tabular

| Dataset | File | Rows | Cols | Size (MB) |
|---|---|--:|--:|--:|
| EPA Water Quality Measurements | `water-quality/epa-wq.csv` | 970,946 | 81 | 556.96 |
| EPA Monitoring Stations | `water-quality/epa-stations.csv` | 1,666 | 37 | 0.37 |
| ISU / IEM Climate | `climate/isu-climate.csv` | 221,559 | 20 | 26.07 |
| PRISM Climate | `climate/prism-iowa-climate.csv` | 4,105,024 | 6 | 228.30 |
| USDA NASS Crop Yields | `agriculture/Crop-Yields.csv` | 9,931 | 21 | 1.77 |
| USDA NASS Livestock Inventory | `agriculture/Livestock-Inventory.csv` | 21,868 | 21 | 4.69 |
| USDA NASS Crop Chemical Application | `agriculture/Crop-Chemical-Application.csv` | 530 | 21 | 0.11 |
| USDA NASS Fertilizer/Feed Spending | `agriculture/Chemical-Fertilizer-Feed-Spending.csv` | 120 | 21 | 0.02 |
| USGS County N&P from Fertilizer | `agriculture/N-P_from_fertilizer_1950-2017-july23-2020.xlsx` | 3,066 | 20 | 1.43 |
| USGS County N&P from Manure | `agriculture/N-P_from_manure_1950-2017-july23-2020.xlsx` | 3,066 | 31 | 13.44 |
| Iowa NRS Tracking — Full | `bmp/iowa-nrs-tracking.csv` | 20,428 | 37 | 3.39 |
| Iowa NRS BMP — HUC-8 | `bmp/iowa-nrs-bmp-huc8.csv` | 2,195 | 8 | 0.20 |
| NPDES DMRs (11 files, FY2015–2025) | `npdes/NPDES_DMRS_FY*.csv` | 2,786,182 | 57 | 766.42 |
| NPDES ATTAINS Assessments | `npdes/NPDES_ATTAINS_AU_SUMMARIES.csv` | 820,292 | 17 | 261.99 |
| NPDES Catchments | `npdes/NPDES_CATCHMENTS.csv` | 1,255,164 | 20 | 287.59 |
| ECHO Facilities | `npdes/echo-facilities-iowa.csv` | 2,216 | 12 | 0.26 |
| ECHO NAICS Codes | `npdes/echo-naics-iowa.csv` | 1,668 | 4 | 0.08 |
| ECHO SIC Codes | `npdes/echo-sics-iowa.csv` | 1,693 | 4 | 0.06 |
| SSURGO Soil Attributes | `soil/ssurgo-iowa-attributes.csv` | 10,572 | 10 | 1.34 |
| USGS Streamflow Discharge | `streamflow/usgs-iowa-discharge.csv` | 560,896 | 4 | 23.49 |
| USGS Streamflow Gauges | `streamflow/usgs-iowa-gauges.csv` | 703 | 7 | 0.06 |
| CDL HUC-12 Fractions | `landuse/cdl-huc12-fractions.csv` | 18,854 | 11 | 1.44 |
| Census 2010–2019 (PEP) | `census/Iowa_Census-2010-2019.xlsx` | ~951 | 13 | 0.08 |
| Census 2020–2025 (PEP) | `census/Iowa-Census-2020-2025.xlsx` | ~948 | 8 | 0.06 |
| **Tabular total** | | **~13.6M** | | **~2,179** |

> The two NPDES tables (DMRs, ATTAINS, Catchments) are national EPA extracts, not
> Iowa-only, which is why their row counts and footprint dominate. Census/N-P
> Excel files carry a few title rows above the data, so row counts are approximate.

### Spatial & non-tabular (raw)

| Dataset | Path | Files | Size (MB) |
|---|---|--:|--:|
| NHDPlus WBD HUC-12 boundaries | `spatial/01_raw/nhdplus/wbd-huc12-iowa/` | 5 (shapefile bundle) | 37.62 |
| NHDPlus GageLoc | `spatial/01_raw/nhdplus/gage-loc/` | 5 (shapefile bundle) | 20.95 |
| NHDPlus Catchment↔HUC-12 crosswalk | `spatial/01_raw/nhdplus/crosswalk/` | 1 | 69.46 |
| SSURGO soil map-unit polygons | `spatial/01_raw/ssurgo/` | 5 (shapefile bundle) | 4,580 |
| NASS CDL rasters (2015–2025) | `spatial/01_raw/cdl/cdl_iowa_*.tif` | 11 | 342.38 |
| Water-quality training images | `images/water-images/train/` | 40 (24 clean + 16 dirty) | 0.59 |
| City water summaries | `text/raw/` | 20 | 0.10 |

> The SSURGO polygon layer covers all 99 Iowa counties (2,712,435 polygons,
> 11,208 distinct map units) as of 2026-07-03. It previously covered only 5
> counties on disk — the download notebook hardcoded a single WSS export date
> that only matched those 5 counties' certification date and returned HTTP 400
> for the rest; it now fetches each county's actual date from SDA.

**Approximate raw download total: ~7.1 GB** (~2.1 GB tabular + ~4.9 GB spatial/imagery/text).

---

## Water Quality

**Source:** U.S. EPA / USGS **Water Quality Portal (WQX)** — the federated
STORET + NWIS feed. <https://www.waterqualitydata.us/>. `ProviderName`
distinguishes the originating system (`NWIS` = USGS, `STORET` = EPA/state).

### EPA Water Quality Measurements (`epa-wq.csv`)

This table records individual chemical, physical, and biological measurements collected at Iowa stream and lake monitoring stations. Each row captures a single analyte result — such as nitrate concentration, dissolved oxygen, or E. coli count — along with the sampling date, depth, method, and quality flags. It is the primary response-variable source for the project, linking ~971K observations across 23 standardized parameters to specific locations and times.

Long table, one row per individual measurement (~971K rows). Full WQX result
schema (81 columns).

| Column | Description |
|---|---|
| `OrganizationIdentifier` | Code for the organization that submitted the data (e.g. `USGS-IA`). **(key)** to the submitting agency. |
| `OrganizationFormalName` | Human-readable name of that organization. |
| `ActivityIdentifier` | Unique ID for the sampling event (the field visit that produced one or more results). |
| `ActivityTypeCode` | Type of activity, e.g. `Sample-Routine`, `Quality Control Sample-Field Blank`. Used to filter out QC samples. |
| `ActivityMediaName` | Environmental medium sampled (e.g. `Water`). |
| `ActivityMediaSubdivisionName` | Subdivision of the medium (e.g. `Surface Water`, `Groundwater`). |
| `ActivityStartDate` | Calendar date the sample was collected (`YYYY-MM-DD`). |
| `ActivityStartTime/Time` | Local clock time the sample was collected. |
| `ActivityStartTime/TimeZoneCode` | Time zone of the start time (e.g. `CDT`). |
| `ActivityEndDate` | Date the activity ended (for integrated/continuous activities). |
| `ActivityEndTime/Time` | Clock time the activity ended. |
| `ActivityEndTime/TimeZoneCode` | Time zone of the end time. |
| `ActivityRelativeDepthName` | Qualitative sampling depth (e.g. `Surface`, `Bottom`). |
| `ActivityDepthHeightMeasure/MeasureValue` | Numeric depth/height at which the sample was taken. |
| `ActivityDepthHeightMeasure/MeasureUnitCode` | Unit for the depth/height value (e.g. `m`, `ft`). |
| `ActivityDepthAltitudeReferencePointText` | Reference datum for the depth (e.g. surface, sea level). |
| `ActivityTopDepthHeightMeasure/MeasureValue` | Top of a sampled depth interval. |
| `ActivityTopDepthHeightMeasure/MeasureUnitCode` | Unit for the top-of-interval depth. |
| `ActivityBottomDepthHeightMeasure/MeasureValue` | Bottom of a sampled depth interval. |
| `ActivityBottomDepthHeightMeasure/MeasureUnitCode` | Unit for the bottom-of-interval depth. |
| `ProjectIdentifier` | ID of the monitoring project the activity belongs to. |
| `ProjectName` | Name of that monitoring project. |
| `ActivityConductingOrganizationText` | Organization that physically performed the sampling. |
| `MonitoringLocationIdentifier` | **(key)** Station ID where the sample was taken; joins to `epa-stations.csv` and to the PRISM `station_id`. |
| `MonitoringLocationName` | Name of the monitoring location. |
| `ActivityCommentText` | Free-text notes about the sampling activity. |
| `SampleAquifer` | Aquifer sampled, for groundwater activities. |
| `HydrologicCondition` | Streamflow condition at sampling (e.g. stable, falling). |
| `HydrologicEvent` | Hydrologic event context (e.g. `Routine sample`, `Storm`). |
| `ActivityLocation/LatitudeMeasure` | Inline latitude of the activity (often blank/unreliable — coordinates come from the stations table instead). |
| `ActivityLocation/LongitudeMeasure` | Inline longitude of the activity (same caveat). |
| `SampleCollectionMethod/MethodIdentifier` | Code for the sample-collection method. |
| `SampleCollectionMethod/MethodIdentifierContext` | Context/authority that defines the method code. |
| `SampleCollectionMethod/MethodName` | Name of the collection method (e.g. `Single vertical, depth integrated`). |
| `SampleCollectionMethod/MethodDescriptionText` | Free-text description of the collection method. |
| `SampleCollectionEquipmentName` | Equipment used to collect the sample. |
| `ResultIdentifier` | Unique ID for this individual result/measurement. |
| `ResultDetectionConditionText` | Detection condition (e.g. `Not Detected`, `Present Below Quantification Limit`) for censored results. |
| `MethodSpeciationName` | Chemical speciation the result is reported as (e.g. `as N`, `as NO3`). Critical for nutrient unit reconciliation. |
| `CharacteristicName` | **The measured parameter** (e.g. `Temperature, water`, `pH`, `Nitrate`, `Dissolved oxygen (DO)`). |
| `ResultSampleFractionText` | Sample fraction analyzed (e.g. `Dissolved`, `Total`). |
| `ResultMeasureValue` | **The measured value** for the characteristic. |
| `ResultMeasure/MeasureUnitCode` | Unit of the measured value (e.g. `mg/L`, `deg C`, `uS/cm`). |
| `MeasureQualifierCode` | Lab/data qualifier flag(s) attached to the result. |
| `ResultStatusIdentifier` | Processing status (e.g. `Accepted`, `Preliminary`). |
| `StatisticalBaseCode` | Statistic the value represents if summarized (e.g. mean, max). |
| `ResultValueTypeName` | Whether the value is `Actual`, `Calculated`, or `Estimated`. |
| `ResultWeightBasisText` | Weight basis for the result (e.g. wet vs dry weight). |
| `ResultTimeBasisText` | Time basis over which the result was integrated. |
| `ResultTemperatureBasisText` | Reference temperature basis for the measurement. |
| `ResultParticleSizeBasisText` | Particle-size basis, for sediment-related results. |
| `DataQuality/PrecisionValue` | Reported measurement precision. |
| `DataQuality/BiasValue` | Reported measurement bias. |
| `DataQuality/ConfidenceIntervalValue` | Confidence interval for the result. |
| `DataQuality/UpperConfidenceLimitValue` | Upper confidence limit. |
| `DataQuality/LowerConfidenceLimitValue` | Lower confidence limit. |
| `ResultCommentText` | Free-text notes about the result. |
| `USGSPCode` | USGS 5-digit parameter code identifying the measured quantity. |
| `ResultDepthHeightMeasure/MeasureValue` | Depth/height at which the result applies. |
| `ResultDepthHeightMeasure/MeasureUnitCode` | Unit for that result depth/height. |
| `ResultDepthAltitudeReferencePointText` | Reference datum for the result depth. |
| `SubjectTaxonomicName` | Taxonomic name of the organism, for biological results. |
| `SampleTissueAnatomyName` | Tissue/anatomy sampled, for tissue results. |
| `BinaryObjectFileName` | Name of an attached binary file, if any. |
| `BinaryObjectFileTypeCode` | Type of the attached binary file. |
| `ResultFileUrl` | URL to an attached result file. |
| `ResultAnalyticalMethod/MethodIdentifier` | Code for the lab analytical method. |
| `ResultAnalyticalMethod/MethodIdentifierContext` | Authority defining the analytical-method code. |
| `ResultAnalyticalMethod/MethodName` | Name of the analytical method. |
| `ResultAnalyticalMethod/MethodUrl` | URL describing the analytical method. |
| `ResultAnalyticalMethod/MethodDescriptionText` | Free-text description of the analytical method. |
| `LaboratoryName` | Lab that analyzed the sample. |
| `AnalysisStartDate` | Date lab analysis began. |
| `ResultLaboratoryCommentText` | Lab comments on the result. |
| `ResultDetectionQuantitationLimitUrl` | URL for detection/quantitation limit details. |
| `DetectionQuantitationLimitTypeName` | Type of detection/quantitation limit reported. |
| `DetectionQuantitationLimitMeasure/MeasureValue` | Numeric value of the detection/quantitation limit. |
| `DetectionQuantitationLimitMeasure/MeasureUnitCode` | Unit of the detection/quantitation limit. |
| `LabSamplePreparationUrl` | URL describing lab sample preparation. |
| `LastUpdated` | Timestamp the record was last updated in the source system. |
| `ProviderName` | Source system that provided the record (`NWIS` or `STORET`). |

### EPA Monitoring Stations (`epa-stations.csv`)

This table provides the geospatial and administrative identity of each water-quality monitoring location — its coordinates, watershed affiliation, county, and station type (stream, lake, well, etc.). It is the spatial backbone that anchors measurement records to specific places on the landscape and enables joining WQ data to climate, land-use, and soil datasets via `MonitoringLocationIdentifier` and `HUCEightDigitCode`.

Station metadata, one row per monitoring location (~1,667 stations, 37 columns).

| Column | Description |
|---|---|
| `OrganizationIdentifier` | Code for the organization that owns the station. |
| `OrganizationFormalName` | Full name of that organization. |
| `MonitoringLocationIdentifier` | **(key)** Unique station ID; joins to the measurements table. |
| `MonitoringLocationName` | Descriptive station name (e.g. `Dry Run Creek near Decorah, IA`). |
| `MonitoringLocationTypeName` | Station type (e.g. `Stream`, `Lake`, `Well`). |
| `MonitoringLocationDescriptionText` | Free-text description of the station. |
| `HUCEightDigitCode` | 8-digit Hydrologic Unit Code of the containing watershed. **(key)** to HUC-8 datasets. |
| `DrainageAreaMeasure/MeasureValue` | Total upstream drainage area at the station. |
| `DrainageAreaMeasure/MeasureUnitCode` | Unit for the drainage area (e.g. `sq mi`). |
| `ContributingDrainageAreaMeasure/MeasureValue` | Contributing (effective) drainage area. |
| `ContributingDrainageAreaMeasure/MeasureUnitCode` | Unit for the contributing drainage area. |
| `LatitudeMeasure` | Station latitude (decimal degrees). |
| `LongitudeMeasure` | Station longitude (decimal degrees). |
| `SourceMapScaleNumeric` | Scale of the map used to locate the station (e.g. `24000`). |
| `HorizontalAccuracyMeasure/MeasureValue` | Horizontal positional accuracy of the coordinates. |
| `HorizontalAccuracyMeasure/MeasureUnitCode` | Unit of the horizontal accuracy (e.g. `seconds`, `meters`). |
| `HorizontalCollectionMethodName` | How the coordinates were determined (e.g. `Interpolated from MAP.`). |
| `HorizontalCoordinateReferenceSystemDatumName` | Horizontal datum of the coordinates (e.g. `NAD83`). |
| `VerticalMeasure/MeasureValue` | Elevation of the station. |
| `VerticalMeasure/MeasureUnitCode` | Unit of the elevation. |
| `VerticalAccuracyMeasure/MeasureValue` | Vertical positional accuracy. |
| `VerticalAccuracyMeasure/MeasureUnitCode` | Unit of the vertical accuracy. |
| `VerticalCollectionMethodName` | Method used to determine elevation. |
| `VerticalCoordinateReferenceSystemDatumName` | Vertical datum (e.g. `NAVD88`). |
| `CountryCode` | Country code (`US`). |
| `StateCode` | FIPS state code (`19` = Iowa). |
| `CountyCode` | FIPS county code (within the state). |
| `AquiferName` | Named aquifer, for wells/groundwater stations. |
| `LocalAqfrName` | Local aquifer name. |
| `FormationTypeText` | Geologic formation sampled. |
| `AquiferTypeName` | Aquifer type (e.g. confined, unconfined). |
| `ConstructionDateText` | Construction date, for wells. |
| `WellDepthMeasure/MeasureValue` | Total well depth. |
| `WellDepthMeasure/MeasureUnitCode` | Unit of the well depth. |
| `WellHoleDepthMeasure/MeasureValue` | Drilled hole depth of the well. |
| `WellHoleDepthMeasure/MeasureUnitCode` | Unit of the well hole depth. |
| `ProviderName` | Source system (`NWIS` / `STORET`). |

### Cleaned water quality outputs

`data/tabular/02_clean/water-quality/epa-wq-clean.csv` — wide table, one row per
`MonitoringLocationIdentifier` + `ActivityStartDateTime`, with a
`<parameter>_value` / `<parameter>_unit` column pair for each of 23 standardized
parameters (temperature, DO, pH, nitrate, nitrite, nitrate+nitrite, ammonia,
Kjeldahl N, orthophosphate, phosphate-P, total P, chloride, sulfate, specific
conductance, TDS, TSS, turbidity, E. coli, chlorophyll a, microcystin, atrazine,
alkalinity, hardness). Values are unit-standardized and range-validated. `data/tabular/02_clean/water-quality/epa-stations-clean.csv`
is the station table reduced to the join/identity columns
(`OrganizationIdentifier`, `MonitoringLocationIdentifier`, `MonitoringLocationName`,
`MonitoringLocationTypeName`, `HUCEightDigitCode`, `LatitudeMeasure`,
`LongitudeMeasure`, `StateCode`, `CountyCode`, `ProviderName`).

---

## Climate

### ISU / IEM Climate (`isu-climate.csv`)

**Source:** Iowa State University **Iowa Environmental Mesonet (IEM)**, daily ASOS/AWOS
station summaries. <https://mesonet.agron.iastate.edu/> (request endpoint
`/cgi-bin/request/daily.py`). One row per station-day (~221K rows). Keyed on ISU
station codes that are spatially matched to WQ sites in the merge step.

This dataset captures daily meteorological conditions — temperature, precipitation, humidity, wind, and snowpack — at Iowa airport weather stations. It provides the in-situ atmospheric context that drives runoff and nutrient transport events, serving as the nearest-station climate feature for each water-quality monitoring site after a spatial match is applied in the merge step.

| Column | Description |
|---|---|
| `station` | **(key)** IEM/ISU station code (e.g. `OOA`). |
| `day` | Calendar date of the daily summary (`YYYY-MM-DD`). |
| `max_temp_f` | Daily maximum air temperature (°F). |
| `min_temp_f` | Daily minimum air temperature (°F). |
| `max_dewpoint_f` | Daily maximum dew-point temperature (°F), a measure of atmospheric moisture. |
| `min_dewpoint_f` | Daily minimum dew-point temperature (°F). |
| `precip_in` | Daily liquid precipitation (inches). Frequently missing at stations without a precip sensor. |
| `avg_wind_speed_kts` | Daily mean wind speed (knots). |
| `avg_wind_drct` | Daily mean wind direction (degrees, 0–360). |
| `min_rh` | Daily minimum relative humidity (%). |
| `avg_rh` | Daily mean relative humidity (%). |
| `max_rh` | Daily maximum relative humidity (%). |
| `snow_in` | Daily snowfall (inches). Blank typically means none. |
| `snowd_in` | Snow depth on the ground (inches). Blank typically means none. |
| `min_feel` | Daily minimum "feels-like" temperature (°F) — wind chill / heat index. |
| `avg_feel` | Daily mean "feels-like" temperature (°F). |
| `max_feel` | Daily maximum "feels-like" temperature (°F). |
| `max_wind_speed_kts` | Daily maximum sustained wind speed (knots). |
| `climo_high_f` | Climatological normal high temperature for that calendar day (°F). |
| `climo_low_f` | Climatological normal low temperature for that calendar day (°F). |

Cleaned output: [`isu-climate-clean.csv`](#cleaned-climate-outputs).

### PRISM Climate (`prism-iowa-climate.csv`)

**Source:** **PRISM Climate Group, Oregon State University** — 4 km gridded daily
climate, sampled at each EPA water-quality station coordinate. <https://prism.oregonstate.edu/>
(API `services.nacse.org/prism/data/get/us/4km/`). Complete grid of ~1,666
stations × ~2,464 days (~4.1M rows). Joins **directly** to WQ sites — `station_id`
is the EPA `MonitoringLocationIdentifier`.

This dataset provides gridded daily temperature, precipitation, and dew-point estimates extracted at the exact coordinates of each water-quality monitoring station. Because PRISM interpolates climate surfaces across the full landscape, it gives complete, gap-free climate records for every WQ site without the spatial-matching uncertainty that comes with the nearest-airport IEM data.

| Column | Description |
|---|---|
| `date` | Calendar date (`YYYY-MM-DD`). |
| `station_id` | **(key)** EPA WQX `MonitoringLocationIdentifier` the grid was sampled at. |
| `tmax` | Daily maximum air temperature (°C). |
| `tmin` | Daily minimum air temperature (°C). |
| `ppt` | Daily total precipitation (mm). |
| `tdmean` | Daily mean dew-point temperature (°C). |

Cleaned output: [`prism-iowa-climate-clean.csv`](#cleaned-climate-outputs).

### Cleaned climate outputs

Produced by the notebooks in `src/02_clean/tabular/climate/`. Both tables stay
one-row-per-station-day and are range-validated against generous physical bounds
(values outside them are nulled, not clipped); gaps are left as blanks for the
modeling step to handle rather than imputed here.

**`data/tabular/02_clean/climate/isu-climate-clean.csv`** (~221K station-day rows)
— the ISU/IEM feed, keyed `station` + `day`, every variable range-validated, all
rows retained. Schema matches the [raw table above](#isu--iem-climate--isu-climatecsv)
except that every temperature-bearing column is **converted from °F to °C and
renamed with a `_c` suffix**, matching the `prism_tmax_c`-style convention used
by the PRISM cleaner:

| Column | Description |
|---|---|
| `max_temp_c` | Daily maximum air temperature (°C). |
| `min_temp_c` | Daily minimum air temperature (°C). |
| `max_dewpoint_c` | Daily maximum dew-point temperature (°C). |
| `min_dewpoint_c` | Daily minimum dew-point temperature (°C). |
| `min_feel_c` | Daily minimum "feels-like" temperature (°C) — wind chill / heat index. |
| `avg_feel_c` | Daily mean "feels-like" temperature (°C). |
| `max_feel_c` | Daily maximum "feels-like" temperature (°C). |
| `climo_high_c` | Climatological normal high temperature for that calendar day (°C). |
| `climo_low_c` | Climatological normal low temperature for that calendar day (°C). |

All other columns (`precip_in`, `avg_wind_speed_kts`, `avg_wind_drct`, `min_rh`,
`avg_rh`, `max_rh`, `snow_in`, `snowd_in`, `max_wind_speed_kts`) are unchanged
from the raw schema.

**`data/tabular/02_clean/climate/prism-iowa-climate-clean.csv`** (~4.1M
station-day rows) — one row per `station_id` + `date`, **de-duplicated so the
key is unique** and the water-quality join can't fan out. PRISM's `-9999` nodata
sentinel is mapped to blank, and `tmin > tmax` pairs are nulled. The four raw
measurements are renamed to unit-bearing columns so units and provenance are
explicit once merged onto the WQ table:

| Column | Description |
|---|---|
| `station_id` | **(key)** EPA WQX `MonitoringLocationIdentifier` the grid was sampled at. |
| `date` | Calendar date (`YYYY-MM-DD`). |
| `prism_tmax_c` | Daily maximum air temperature (°C). |
| `prism_tmin_c` | Daily minimum air temperature (°C). |
| `prism_ppt_mm` | Daily total precipitation (mm). |
| `prism_tdmean_c` | Daily mean dew-point temperature (°C). |

---

## Agriculture

### USDA NASS tables

**Source:** **USDA National Agricultural Statistics Service (NASS) Quick Stats.**
<https://quickstats.nass.usda.gov/>. Four files
(`Crop-Yields.csv`, `Livestock-Inventory.csv`,
`Crop-Chemical-Application.csv`, `Chemical-Fertilizer-Feed-Spending.csv`) all
share the standard Quick Stats 21-column export schema; only the
`Commodity` / `Data Item` / `Domain` content differs.

These four tables collectively describe Iowa's agricultural landscape at annual resolution: what crops were grown and at what yields, how many livestock were present by species and herd size, what volumes of herbicides and fertilizers were applied to crops, and what farms spent on chemicals and feed. Together they quantify the nutrient and chemical loading pressure that translates into water-quality signals observed at downstream monitoring stations.

| Column | Description |
|---|---|
| `Program` | Data program: `CENSUS` (Census of Agriculture) or `SURVEY`. |
| `Year` | Reference year of the observation. |
| `Period` | Time period the value covers (e.g. `YEAR`, `END OF DEC`). |
| `Week Ending` | Week-ending date, for weekly survey series (usually blank here). |
| `Geo Level` | Geographic aggregation level (`STATE`, `COUNTY`, etc.). |
| `State` | State name (`IOWA`). |
| `State ANSI` | FIPS state code (`19`). |
| `Ag District` | NASS agricultural statistics district name. |
| `Ag District Code` | Numeric code for the ag district. |
| `County` | County name. |
| `County ANSI` | FIPS county code. |
| `Zip Code` | ZIP code, when the series is reported at ZIP level. |
| `Region` | Region designation, when applicable. |
| `watershed_code` | 8-digit watershed code, when the series is watershed-based. |
| `Watershed` | Watershed name, when applicable. |
| `Commodity` | The commodity measured (e.g. `CORN`, `CATTLE`, `SOYBEANS`, `CHEMICAL TOTALS`). |
| `Data Item` | Full description of the measured quantity, including unit (e.g. `CORN, GRAIN - PRODUCTION, MEASURED IN BU`). |
| `Domain` | Breakdown dimension (e.g. `TOTAL`, `CHEMICAL, FUNGICIDE`, an inventory class). |
| `Domain Category` | Specific category within the domain (e.g. a chemical name, a herd-size band). |
| `Value` | The reported numeric value (comma-formatted; unit is embedded in `Data Item`). |
| `CV (%)` | Coefficient of variation (%) — the sampling reliability of survey estimates. |

### Cleaned USDA NASS outputs

One cleaning notebook per source file lives in
`src/02_clean/tabular/agriculture/`; each writes a tidy table to
`data/tabular/02_clean/agriculture/`. **Conventions shared by all four:**

- The packed `Data Item` string is unpacked into separate `commodity_detail`,
  `statistic`, and `unit` columns (e.g. `CORN, GRAIN - YIELD, MEASURED IN BU /
  ACRE` → `CORN, GRAIN` / `YIELD` / `BU / ACRE`).
- `Value` and `CV (%)` are parsed to numbers; NASS letter suppression codes
  (`(D)`, `(Z)`, `(H)`, `(L)`, …) become **blank, never `0`**. A boolean
  `value_suppressed` flag marks rows whose value was a `(D)` disclosure
  suppression, so "censored" stays distinguishable from "not collected".
- `Year` → integer; county series carry a 5-digit `county_fips` (state+county
  FIPS, blank for `OTHER (COMBINED) COUNTIES` roll-ups), state series a 2-digit
  `state_fips` (`19` = Iowa). Empty / constant export columns (`Week Ending`,
  `Zip Code`, `Geo Level`, `State`, …) are dropped, and the row key is asserted
  unique.

#### `crop-yields-clean.csv`

County-level, 9,931 rows — one per `(program, year, county, commodity_detail,
statistic)`, 2015–2025.

| Column | Description |
|---|---|
| `program` | `SURVEY` or `CENSUS`. |
| `year` | Reference year (int). |
| `ag_district_code` | NASS ag-district code. **(key)** — distinguishes the per-district `OTHER (COMBINED) COUNTIES` roll-ups. |
| `ag_district` | Ag-district name. |
| `county` | County name (includes roll-up buckets like `OTHER (COMBINED) COUNTIES`). |
| `county_fips` | **(key)** 5-digit state+county FIPS; blank for roll-up buckets. |
| `commodity` | Coarse commodity (`CORN`, `SOYBEANS`, `HAY`, `OATS`, `WHEAT`, `BARLEY`, `RYE`). |
| `commodity_detail` | Specific commodity, e.g. `CORN, GRAIN` vs `CORN, SILAGE`. |
| `statistic` | `PRODUCTION`, `YIELD`, or `ACRES PLANTED`. |
| `unit` | Measurement unit (`BU`, `BU / ACRE`, `TONS`, `TONS / ACRE`, `LB`); blank for `ACRES PLANTED`. |
| `value` | Reported numeric value (blank where suppressed). |
| `value_suppressed` | `True` where the raw value was a `(D)` suppression code. |
| `cv_pct` | Coefficient of variation (%); blank for census/suppressed rows. |

#### `livestock-inventory-clean.csv`

County-level, 21,868 rows — one per `(program, year, period, county,
commodity_detail, statistic, herd_size_band)`, 2015–2025. **Holds two layers:**
headline totals (`domain == "TOTAL"`) and the *same operations* re-counted into
herd-size bands — the band rows must **not** be summed with the totals.

| Column | Description |
|---|---|
| `program` | `SURVEY` or `CENSUS`. |
| `year` | Reference year (int). |
| `period` | Reference point: `END OF DEC` (census) or `FIRST OF JAN` (survey). |
| `ag_district_code` / `ag_district` / `county` / `county_fips` | County keys, as in crop yields. |
| `commodity` | `CATTLE`, `HOGS`, `GOATS`, `SHEEP`. |
| `commodity_detail` | Class within species, e.g. `CATTLE, COWS, MILK`. |
| `statistic` | `INVENTORY` (head) or `OPERATIONS WITH INVENTORY` (farm counts). |
| `unit` | `HEAD` or `OPERATIONS`, set from the statistic (the raw export leaves it blank). |
| `domain` | `TOTAL` for the headline series, else the herd-size breakdown dimension. |
| `herd_size_band` | Herd-size band for breakdown rows (e.g. `1 TO 9`, `500 OR MORE`); blank for `TOTAL` rows. |
| `value` | Reported numeric value (blank where suppressed). |
| `value_suppressed` | `True` where the raw value was `(D)`. |
| `cv_pct` | Coefficient of variation (%); blank for census/suppressed rows. |

#### `crop-chemical-application-clean.csv`

State-level, 530 rows — one per `(year, commodity, input_class,
active_ingredient)`, selected years 2015–2023. Pounds of active ingredient
applied statewide; the most directly water-quality-relevant ag table.

| Column | Description |
|---|---|
| `year` | Reference year (int). |
| `state_fips` | **(key)** 2-digit state FIPS (`19`). |
| `commodity` | `CORN` or `SOYBEANS`. |
| `statistic` | `APPLICATIONS`. |
| `unit` | `LB` (pounds of active ingredient). |
| `input_class` | `HERBICIDE`, `FUNGICIDE`, `INSECTICIDE`, `OTHER`, or `FERTILIZER` (parsed from `Domain`). |
| `active_ingredient` | Active-ingredient / nutrient name (e.g. `GLYPHOSATE`, `NITROGEN`), or `TOTAL` for per-class subtotals. |
| `chemical_code` | NASS numeric chemical code; blank for fertilizer nutrients and `TOTAL` rows. |
| `value` | Pounds applied statewide (blank where suppressed — ~45% are `(D)`). |
| `value_suppressed` | `True` where the raw value was `(D)`. |

#### `chemical-fertilizer-feed-spending-clean.csv`

State-level, 120 rows — one per `(year, expense_category, unit)`, 2015–2024.
Annual statewide farm production expenses.

| Column | Description |
|---|---|
| `year` | Reference year (int). |
| `state_fips` | **(key)** 2-digit state FIPS (`19`). |
| `commodity` | Raw NASS commodity label (`CHEMICAL TOTALS`, `FERTILIZER TOTALS`, `FEED`). |
| `expense_category` | Expense category: `CHEMICAL TOTALS`; `FERTILIZER TOTALS, INCL LIME & SOIL CONDITIONERS`; `FEED`. |
| `statistic` | `EXPENSE`. |
| `unit` | Reporting basis: `$`, `$ / OPERATION`, `PCT OF OPERATIONS`, `PCT OF PRODUCTION EXPENSES`. |
| `value` | Reported numeric value. |
| `value_suppressed` | `True` where suppressed (none in the current extract). |

### USGS County N & P Inputs (`N-P_from_*.xlsx`)

**Source:** **USGS** — Falcone (2021), *Estimates of county-level nitrogen and
phosphorus from fertilizer and manure from 1950 through 2017*, USGS Open-File
Report 2020-1153, data release DOI 10.5066/P9VSQN3C.
<https://www.sciencebase.gov/catalog/item/5ebad56382ce25b51361806a>. Wide tables,
one row per U.S. county with many year-suffixed columns.

These two wide-format tables quantify the kilograms of nitrogen and phosphorus delivered to Iowa counties each year via commercial fertilizer and animal manure, respectively, from 1950 through 2017. They offer a long historical baseline of nutrient loading that complements the shorter NASS survey window, and the manure table further breaks down N and P contributions by livestock species (cattle, hogs, chickens, etc.).

`N-P_from_fertilizer_1950-2017-july23-2020.xlsx`:

| Column | Description |
|---|---|
| `STCOFIPS` | **(key)** 5-digit state+county FIPS code (zero-padded string). |
| `fips-int` | Same FIPS as an integer. |
| `CountyName` | County name. |
| `State` | State abbreviation. |
| `tot-fertN-kg-YYYY` | Total nitrogen applied as commercial fertilizer in that county for year `YYYY` (kg). One column per ~5-year Ag-Census year, 1950–2017. |
| `tot-fertP-kg-YYYY` | Total phosphorus applied as commercial fertilizer in that county for year `YYYY` (kg). One column per year. |

`N-P_from_manure_1950-2017-july23-2020.xlsx`:

| Column | Description |
|---|---|
| `STCOFIPS` | **(key)** 5-digit state+county FIPS code. |
| `fips-int` | FIPS as an integer. |
| `CountyName` | County name. |
| `State` | State abbreviation. |
| `<livestock>YYYYadj` | Adjusted head counts by animal type and year, e.g. `bcows1950adj` (beef cows), `mcows…` (milk cows), `hogs…`, `chickens…`, `broilers…`, `turkeys…`, `sheep…`, `horses…`. |
| `<Animal>_N_kg-YYYY` | Nitrogen from that animal group's manure in year `YYYY` (kg), e.g. `Cattle_N_kg-1950`, `Hogs_N_kg-1950`. |
| `<Animal>_P_kg-YYYY` | Phosphorus from that animal group's manure in year `YYYY` (kg). |

### Cleaned N & P outputs

Produced by `src/02_clean/tabular/agriculture/np-fertilizer-clean.ipynb` and `np-manure-clean.ipynb`. The wide county × year matrices are melted to long format and filtered to Iowa (`state = IA`).

**`data/tabular/02_clean/agriculture/np-fertilizer-clean.csv`** (4,356 rows) — one row per `(county_fips, year, nutrient, source)`:

| Column | Description |
|---|---|
| `state_fips` | 2-digit state FIPS (`19`). |
| `county_fips` | **(key)** 5-digit state+county FIPS. |
| `county_name` | County name. |
| `state` | State abbreviation (`IA`). |
| `year` | Census year (approximately every 5 years, 1950–2017). |
| `nutrient` | `N` (nitrogen) or `P` (phosphorus). |
| `source` | Application context: `farm`, `nonfarm`, or `total`. |
| `value_kg` | Kilograms of nutrient applied as commercial fertilizer. |

**`data/tabular/02_clean/agriculture/np-manure-clean.csv`** (14,850 rows) — one row per `(county_fips, year, animal_category, nutrient)`:

| Column | Description |
|---|---|
| `state_fips` | 2-digit state FIPS (`19`). |
| `county_fips` | **(key)** 5-digit state+county FIPS. |
| `county_name` | County name. |
| `state` | State abbreviation (`IA`). |
| `year` | Census year (approximately every 5 years, 1950–2017). |
| `animal_category` | Livestock category contributing the manure (`Cattle`, `Hogs`, `Poultry`, `Other`, `Total`). |
| `nutrient` | `N` or `P`. |
| `value_kg` | Kilograms of nutrient in manure from that animal category. |

**`data/tabular/02_clean/agriculture/manure-animal-inventory-clean.csv`** (14,850 rows) — head counts by `(county_fips, year, animal)`:

| Column | Description |
|---|---|
| `state_fips` | 2-digit state FIPS (`19`). |
| `county_fips` | **(key)** 5-digit state+county FIPS. |
| `county_name` | County name. |
| `state` | State abbreviation (`IA`). |
| `year` | Census year (approximately every 5 years, 1950–2017). |
| `animal` | Animal type (e.g. `all cattle and calves`, `beef cows`, `milk cows`, `hogs and pigs`, `broilers`, `layers`, `turkeys`, `sheep and lambs`, `horses and ponies`). |
| `head_count` | Number of animals. |
| `adjusted` | `True` when the count uses the USDA slaughter-weight adjustment coefficient. |

**`data/tabular/02_clean/agriculture/manure-weight-coefficients-clean.csv`** (90 rows) — USDA live-weight adjustment coefficients used to compute nutrient loads from raw head counts:

| Column | Description |
|---|---|
| `year` | Census year. |
| `animal` | Animal type (matching `manure-animal-inventory-clean.csv`). |
| `weight_coef` | Adjustment coefficient relative to the 1992 USDA base. |
| `usda_live_weight_avg` | USDA average live weight (lbs) for that animal and year. |

---

## Conservation BMPs (Iowa NRS)

**Source:** **Iowa Nutrient Reduction Strategy (INRS) / Iowa State University**,
distributed via the ISU GIS ArcGIS portal. <https://www.nutrientstrategy.iastate.edu/>.

These two files track the adoption and spatial footprint of conservation best-management practices (BMPs) — cover crops, bioreactors, saturated buffers, constructed wetlands, and others — intended to reduce nitrogen and phosphorus losses from Iowa farmland. They provide a direct measure of mitigation effort at the HUC-8 and HUC-12 watershed scale, making them a key control variable when modeling whether observed water-quality improvements are attributable to conservation practice adoption.

`iowa-nrs-tracking.csv` — full INRS tracking export (37 columns); a flexible
long format where each row is one indicator/practice/funding record.

| Column | Description |
|---|---|
| `measurableIndicator` | The INRS measurable indicator the row reports (e.g. a reference/baseline metric). |
| `assessment` | Assessment type/round the value belongs to (e.g. `Baseline`). |
| `referencePeriod` | Reference period for the value. |
| `year` | Year of the record. |
| `value` | The reported numeric value. |
| `unit` | Unit of the value (e.g. `tons/year`, `Number`, acres). |
| `category` | Top-level category (e.g. `Nitrogen - Nonpoint`). |
| `subcategory` | Second-level category. |
| `tertiarycategory` | Third-level category. |
| `practiceName` | Name of the conservation practice. |
| `practiceCode` | NRCS practice code (e.g. 340 for cover crops). |
| `cropRotation` | Crop rotation context for the practice. |
| `poundsPerAcre` | Application/loading rate, where relevant (lb/acre). |
| `species` | Species (e.g. cover-crop species), where relevant. |
| `n` | Count/sample size for the record. |
| `mlra` | Major Land Resource Area code. |
| `mlraName` | MLRA name. |
| `huc8` | 8-digit HUC watershed code. **(key)** |
| `huc8Name` | HUC-8 watershed name. |
| `huc12` | 12-digit HUC subwatershed code. **(key)** |
| `huc12Name` | HUC-12 subwatershed name. |
| `watershedProjectID` | ID of the associated watershed project. |
| `watershedProjectName` | Name of the watershed project. |
| `lAMonitoringBasinName` | Name of the associated load-monitoring basin. |
| `lAMonitoringBasinID` | ID of the load-monitoring basin. |
| `countyFIPS` | County FIPS code. |
| `countyName` | County name. |
| `cityName` | City name, when applicable. |
| `cityFIP` | City FIPS code. |
| `priorityBasinINRSdesignation` | Whether/how the basin is an INRS priority basin. |
| `priorityBasinINRSname` | Priority basin name. |
| `priorityBasinINRSid` | Priority basin ID. |
| `fundingType` | Type of funding for the practice. |
| `fundingCategory` | Funding category. |
| `fundingAgency` | Agency providing the funding. |
| `Order` | Display/sort order from the source. |
| `dataSource` | Provenance of the record (e.g. `INRS Baseline`). |

`iowa-nrs-bmp-huc8.csv` — tidy per-watershed practice-adoption counts.

| Column | Description |
|---|---|
| `year` | Year of the adoption count. |
| `practice_type` | Machine code for the BMP type (e.g. `bioreactor_sat_buffer`). |
| `category` | Human-readable practice category (e.g. `Bioreactors and Saturated Buffers`). |
| `assessment` | Assessment label (e.g. `HUC8 Practice Adoption`). |
| `value` | The adoption count or acreage. |
| `unit` | Unit of the value (e.g. `Number`, acres). |
| `huc8_code` | **(key)** 8-digit HUC watershed code. |
| `huc8_name` | HUC-8 watershed name. |

### Cleaned BMP outputs

Produced by `src/02_clean/tabular/bmp/iowa-nrs-bmp-huc8-clean.ipynb` and `iowa-nrs-tracking-clean.ipynb`.

**`data/tabular/02_clean/bmp/iowa-nrs-bmp-huc8-clean.csv`** (1,699 rows) — one row per `(huc8_code, year, practice_type)`. The constant `assessment` column is dropped from the raw table:

| Column | Description |
|---|---|
| `huc8_code` | **(key)** 8-digit HUC watershed code. |
| `huc8_name` | HUC-8 watershed name. |
| `year` | Year of the adoption count. |
| `practice_type` | Machine code for the BMP (`cover_crop`, `bioreactor_sat_buffer`, `crep_wetland`, `erosion_control`). |
| `category` | Human-readable practice category label (e.g. `Bioreactors and Saturated Buffers`). |
| `unit` | Unit of the value (`Acres` or `Number`). |
| `value` | Adoption acreage or count. |

**`data/tabular/02_clean/bmp/iowa-nrs-tracking-clean.csv`** (20,428 rows) — the full INRS tracking export with 11 sparse or internal columns dropped: `practiceCode`, `mlra`, `mlraName`, `watershedProjectID`, `cityName`, `cityFIP`, `priorityBasinINRSdesignation`, `priorityBasinINRSname`, `priorityBasinINRSid`, `fundingType`, `Order`. All remaining columns match the raw schema; see [raw column documentation above](#conservation-bmps-iowa-nrs).

---

## NPDES Compliance & Permits

**Source:** **U.S. EPA ECHO / ICIS-NPDES** bulk downloads
(<https://echo.epa.gov/>, file `npdes_downloads.zip`) plus **EPA ATTAINS**
(Clean Water Act §303(d)/§305(b) assessments, <https://www.epa.gov/waterdata/attains>).

### Discharge Monitoring Reports (`NPDES_DMRS_FY*.csv`)

These annual files record what each permitted facility actually discharged into Iowa waterways — pollutant concentrations and loads measured at each outfall — alongside the regulatory limits those discharges were required to meet. Each row pairs a specific effluent parameter (e.g., total nitrogen, BOD, E. coli) with the reported measurement value and flags whether a limit was exceeded, making this the point-source pollution counterpart to the diffuse agricultural loading datasets.

One file per federal fiscal year 2015–2025; identical 57-column ICIS-NPDES DMR
schema. One row per reported limit/measurement on a permit's discharge point.

| Column | Description |
|---|---|
| `ACTIVITY_ID` | Internal ICIS ID for the permit activity. |
| `EXTERNAL_PERMIT_NMBR` | **(key)** NPDES permit number (joins to ECHO `npdes_id`). |
| `VERSION_NMBR` | Permit version number. |
| `PERM_FEATURE_ID` | Internal ID of the permitted feature (outfall). |
| `PERM_FEATURE_NMBR` | Outfall number (e.g. `001`). |
| `PERM_FEATURE_TYPE_CODE` | Type of permitted feature (e.g. external outfall). |
| `LIMIT_SET_ID` | ID of the limit set the limit belongs to. |
| `LIMIT_SET_DESIGNATOR` | Designator distinguishing limit sets on a feature. |
| `LIMIT_SET_SCHEDULE_ID` | ID of the monitoring/reporting schedule for the limit set. |
| `LIMIT_ID` | Unique ID of the effluent limit. |
| `LIMIT_BEGIN_DATE` | Date the limit takes effect. |
| `LIMIT_END_DATE` | Date the limit expires. |
| `NMBR_OF_SUBMISSION` | Number of DMR submissions expected for the period. |
| `NMBR_OF_REPORT` | Number of reports expected. |
| `PARAMETER_CODE` | Code for the regulated pollutant/parameter. |
| `PARAMETER_DESC` | Name of the parameter (e.g. `Polynuclear Aromatic Hydrocarbons [PAHs]`). |
| `MONITORING_LOCATION_CODE` | Code for where monitoring occurs relative to the outfall. |
| `STAY_TYPE_CODE` | Type of administrative stay on the limit, if any. |
| `LIMIT_VALUE_ID` | Unique ID of the specific limit value. |
| `LIMIT_VALUE_TYPE_CODE` | Statistical basis of the limit value (e.g. average, maximum). |
| `LIMIT_VALUE_NMBR` | The permitted limit value, as written. |
| `LIMIT_UNIT_CODE` | Code for the limit's unit of measure. |
| `LIMIT_UNIT_DESC` | Unit of the limit value (e.g. `mg/L`). |
| `STANDARD_UNIT_CODE` | Code for the standardized unit. |
| `STANDARD_UNIT_DESC` | Standardized unit description. |
| `LIMIT_VALUE_STANDARD_UNITS` | Limit value converted to standard units (comparable across permits). |
| `STATISTICAL_BASE_CODE` | Statistical base of the limit (e.g. daily, monthly). |
| `STATISTICAL_BASE_TYPE_CODE` | Type of statistic (e.g. `MAX`, `AVG`). |
| `LIMIT_VALUE_QUALIFIER_CODE` | Qualifier on the limit (e.g. `<=`). |
| `OPTIONAL_MONITORING_FLAG` | Whether monitoring is optional (`Y`/`N`). |
| `LIMIT_SAMPLE_TYPE_CODE` | Required sample type for compliance. |
| `LIMIT_FREQ_OF_ANALYSIS_CODE` | Required analysis frequency for the limit. |
| `STAY_VALUE_NMBR` | Alternate limit value in force during a stay. |
| `LIMIT_TYPE_CODE` | Limit type (e.g. `ENF` enforceable). |
| `DMR_EVENT_ID` | ID of the DMR reporting event. |
| `MONITORING_PERIOD_END_DATE` | End date of the monitoring period being reported. |
| `DMR_SAMPLE_TYPE_CODE` | Sample type actually used in the DMR. |
| `DMR_FREQ_OF_ANALYSIS_CODE` | Analysis frequency actually used. |
| `REPORTED_EXCURSION_NMBR` | Number of reported excursions (exceedances) in the period. |
| `DMR_FORM_VALUE_ID` | ID of the reported value on the DMR form. |
| `VALUE_TYPE_CODE` | Statistical type of the reported value. |
| `DMR_VALUE_ID` | Unique ID of the reported measurement value. |
| `DMR_VALUE_NMBR` | **The reported measured discharge value.** |
| `DMR_UNIT_CODE` | Code for the reported value's unit. |
| `DMR_UNIT_DESC` | Unit of the reported value. |
| `DMR_VALUE_STANDARD_UNITS` | Reported value converted to standard units. |
| `DMR_VALUE_QUALIFIER_CODE` | Qualifier on the reported value (e.g. `=`, `<`). |
| `VALUE_RECEIVED_DATE` | Date EPA received the reported value. |
| `DAYS_LATE` | Days the DMR was submitted late. |
| `NODI_CODE` | "No Data Indicator" code explaining a missing value. |
| `EXCEEDENCE_PCT` | Percent by which the reported value exceeded its limit. |
| `NPDES_VIOLATION_ID` | ID of the associated violation, if any. |
| `VIOLATION_CODE` | Code classifying the violation. |
| `RNC_DETECTION_CODE` | Reportable Non-Compliance detection code. |
| `RNC_DETECTION_DATE` | Date the non-compliance was detected. |
| `RNC_RESOLUTION_CODE` | Code for how the non-compliance was resolved. |
| `RNC_RESOLUTION_DATE` | Date the non-compliance was resolved. |

### ATTAINS Assessments (`NPDES_ATTAINS_AU_SUMMARIES.csv`)

ATTAINS is the EPA's inventory of how well each assessed water body meets its Clean Water Act designated uses — drinking water supply, aquatic life, recreation, and fish consumption. Each row summarizes the overall condition of an assessment unit (a named river segment or lake) and lists the pollutant causes of any impairment, linking those assessments back to specific NPDES permits that discharge into or near that water body.

Water-body assessment status linked to NPDES permits.

| Column | Description |
|---|---|
| `REGISTRY_ID` | EPA Facility Registry Service ID for the linked facility. |
| `ECHO_DFR_URL` | URL to the facility's ECHO Detailed Facility Report. |
| `NPDES_ID` | **(key)** NPDES permit number linked to the assessment. |
| `REPORTINGCYCLE` | Assessment reporting cycle year. |
| `STATE` | State of the assessment unit. |
| `ASSESSMENTUNITIDENTIFIER` | **(key)** Unique ID of the assessed water body (assessment unit). |
| `AU_URL` | URL to the assessment unit's How's My Waterway report. |
| `ASSESSMENTUNITNAME` | Name of the water body (e.g. `Kenai River`). |
| `WATER_CONDITION` | Overall condition (e.g. `Good`, `Impaired`). |
| `POT_IMP_PARAMETERS` | Parameters potentially impairing the water body. |
| `E90_POT_IMP_PARAMETERS` | Potentially impairing parameters tied to permits with effluent exceedances. |
| `DRINKINGWATER_USE` | Support status of the drinking-water designated use. |
| `ECOLOGICAL_USE` | Support status of the aquatic-life/ecological use. |
| `FISHCONSUMPTION_USE` | Support status of the fish-consumption use. |
| `RECREATION_USE` | Support status of the recreation use. |
| `OTHER_USE` | Support status of other designated uses. |
| `CAUSE_GROUPS_IMPAIRED` | Groups of impairment causes for the water body. |

### NPDES Catchments (`NPDES_CATCHMENTS.csv`)

This table spatially anchors each NPDES-permitted facility to the NHDPlus stream network by recording the catchment and HUC-12 subwatershed the facility drains into. It is the spatial crosswalk that makes it possible to aggregate point-source discharge loads to the same watershed units used by the water-quality, land-use, and BMP datasets.

Links each NPDES permit to its NHDPlus catchment / HUC-12 watershed.

| Column | Description |
|---|---|
| `NPDES_ID` | **(key)** NPDES permit number. |
| `PERMIT_TYPE_CODE` | Code for the permit type (e.g. `GPC`). |
| `PERMIT_TYPE_DESC` | Permit type description (e.g. `General Permit Covered Facility`). |
| `SUB_ID` | Sub-facility identifier. |
| `LATITUDE83` | Facility latitude (NAD83). |
| `LONGITUDE83` | Facility longitude (NAD83). |
| `STATECODE` | State abbreviation. |
| `NHDPLUSID` | NHDPlus catchment identifier. **(key)** |
| `WBD_HU12` | 12-digit HUC of the containing subwatershed. **(key)** |
| `WBD_HU12NAME` | Name of that HUC-12 subwatershed. |
| `REACHCODE` | NHD reach code of the associated stream reach. |
| `GNIS_NAME` | GNIS name of the associated stream feature. |
| `CATCHMENT_HUC12` | HUC-12 assigned to the catchment. |
| `AREASQKM` | Catchment area (km²). |
| `LENGTHKM` | Length of the catchment's flowline (km). |
| `NAVIGABLE` | Whether the reach is navigable (`Y`/`N`). |
| `HEADWATER` | Whether the catchment is a headwater (`Y`/`N`). |
| `COASTAL` | Whether the catchment is coastal (`Y`/`N`). |
| `TIDAL` | Whether the reach is tidal (`Y`/`N`). |
| `ALASKAN` | Whether the catchment is in Alaska (`Y`/`N`). |

### ECHO Facilities / NAICS / SIC

**Source:** **EPA ECHO** facility metadata for Iowa NPDES permits.

These three files describe the identity, location, and industry type of every NPDES-permitted facility in Iowa. The facilities table provides the physical address, coordinates, and county for each permit, while the NAICS and SIC tables classify what each facility does (e.g., municipal wastewater treatment, food processing, livestock operations), enabling analysis of which industry sectors contribute most to point-source loads.

`echo-facilities-iowa.csv`:

| Column | Description |
|---|---|
| `facility_interest_id` | ICIS internal facility ID. |
| `npdes_id` | **(key)** NPDES permit number (joins to DMR `EXTERNAL_PERMIT_NMBR`). |
| `facility_uin` | EPA universal facility identifier (FRS Registry ID). |
| `facility_type_code` | Facility type code (e.g. `POF`, `POTW`). |
| `facility_name` | Facility name. |
| `address` | Street address. |
| `city` | City. |
| `county_fips` | County FIPS code (state-prefixed, e.g. `IA111`). |
| `zip` | ZIP code. |
| `latitude` | Facility latitude. |
| `longitude` | Facility longitude. |
| `impaired_waters` | Indicator/flag for associated impaired waters. |

`echo-naics-iowa.csv` / `echo-sics-iowa.csv` (industry classifications):

| Column | Description |
|---|---|
| `NPDES_ID` | **(key)** NPDES permit number. |
| `NAICS_CODE` / `SIC_CODE` | Industry classification code for the facility. |
| `NAICS_DESC` / `SIC_DESC` | Description of the industry code (e.g. `Water Supply and Irrigation Systems`). |
| `PRIMARY_INDICATOR_FLAG` | Whether this is the facility's primary industry code (`Y`/`N`). |

### Cleaned NPDES outputs

Produced by notebooks in `src/02_clean/tabular/npdes/`. All tables are Iowa-filtered.

**`data/tabular/02_clean/npdes/npdes-dmrs-clean.csv`** (2,786,182 rows) — the 11 annual DMR files concatenated and reduced from 57 to 26 columns. Internal IDs, redundant lookup fields, and the outer permit-activity hierarchy are dropped:

| Column | Description |
|---|---|
| `npdes_id` | **(key)** NPDES permit number. |
| `fiscal_year` | Federal fiscal year of the DMR report. |
| `perm_feature_nmbr` | Outfall number (e.g. `001`). |
| `perm_feature_type_code` | Permitted feature type code. |
| `limit_set_designator` | Distinguishes multiple limit sets on one outfall. |
| `parameter_code` | Code for the regulated parameter. |
| `parameter_desc` | Parameter name (e.g. `Nitrogen, Total`). |
| `monitoring_location_code` | Where monitoring occurs relative to the outfall. |
| `statistical_base_code` | Statistical basis of the limit (e.g. daily, monthly). |
| `monitoring_period_end_date` | End date of the reporting period. |
| `limit_begin_date` / `limit_end_date` | Effective dates of the effluent limit. |
| `limit_value_type_code` | Statistical type of the limit (e.g. `Average Monthly`, `Maximum Daily`). |
| `limit_value` | The permitted limit value. |
| `limit_value_qualifier` | Qualifier on the limit (e.g. `<=`). |
| `value_type_code` | Statistical type of the reported value. |
| `dmr_value` | **The reported measured discharge value.** |
| `dmr_value_qualifier` | Qualifier on the reported value (e.g. `=`, `<`). |
| `standard_unit_desc` | Unit of both the limit and reported values. |
| `nodi_code` | No-Data Indicator code explaining a missing value. |
| `exceedence_pct` | Percent by which the reported value exceeded its limit. |
| `violation_code` | Violation classification code, if any. |
| `days_late` | Days the DMR was submitted late. |
| `version_nmbr` | Permit version number. |
| `value_received_date` | Date EPA received the reported value. |
| `dmr_form_value_id` | Unique ID of the reported value on the DMR form. |

**`data/tabular/02_clean/npdes/npdes-attains-clean.csv`** (1,102 rows) — Iowa-filtered; `STATE` column dropped; column names lowercased and snake-cased. Note: `fishconsumption_use` (present in the raw national file) is not present in the Iowa extract:

| Column | Description |
|---|---|
| `registry_id` | EPA Facility Registry ID. |
| `echo_dfr_url` | URL to the ECHO Detailed Facility Report. |
| `npdes_id` | **(key)** NPDES permit number. |
| `reporting_cycle` | Assessment reporting cycle year. |
| `assessment_unit_id` | **(key)** Unique ID of the assessed water body. |
| `au_url` | URL to the How's My Waterway report. |
| `assessment_unit_name` | Name of the assessed water body. |
| `water_condition` | Overall condition (`Good`, `Impaired`, etc.). |
| `potential_impairment_parameters` | Parameters potentially causing impairment. |
| `e90_potential_impairment_parameters` | Potentially impairing parameters tied to effluent exceedances. |
| `drinkingwater_use` | Support status of the drinking-water designated use. |
| `ecological_use` | Support status of the aquatic-life/ecological use. |
| `recreation_use` | Support status of the recreation designated use. |
| `other_use` | Support status of other designated uses. |
| `cause_groups_impaired` | Groups of impairment causes. |

**`data/tabular/02_clean/npdes/npdes-catchments-clean.csv`** (2,432 rows) — Iowa-filtered; column names lowercased and snake-cased; `STATECODE` dropped:

| Column | Description |
|---|---|
| `npdes_id` | **(key)** NPDES permit number. |
| `sub_id` | Sub-facility identifier. |
| `permit_type_code` / `permit_type_desc` | Permit type code and description. |
| `latitude` / `longitude` | Facility coordinates (NAD83). |
| `nhdplusid` | **(key)** NHDPlus catchment identifier. |
| `wbd_huc12` | **(key)** 12-digit HUC of the containing subwatershed. |
| `wbd_huc12_name` | Name of the HUC-12 subwatershed. |
| `reachcode` | NHD reach code of the associated stream. |
| `gnis_name` | GNIS name of the stream feature. |
| `catchment_huc12` | HUC-12 of the catchment. |
| `area_sqkm` | Catchment area (km²). |
| `length_km` | Flowline length (km). |
| `navigable` / `headwater` / `coastal` / `tidal` / `alaskan` | Boolean NHDPlus feature flags. |

**`data/tabular/02_clean/npdes/echo-facilities-clean.csv`** (2,216 rows) — Iowa NPDES facilities; `impaired_waters` renamed to `impaired_303d`:

| Column | Description |
|---|---|
| `npdes_id` | **(key)** NPDES permit number. |
| `facility_interest_id` | ICIS internal facility ID. |
| `facility_uin` | EPA Facility Registry Service (FRS) universal ID. |
| `facility_name` | Facility name. |
| `facility_type_code` | Facility type code (e.g. `POTW`). |
| `address` / `city` / `zip` | Physical address. |
| `county_fips` | County FIPS code (state-prefixed, e.g. `IA111`). |
| `latitude` / `longitude` | Facility coordinates. |
| `impaired_303d` | Flag for whether the facility discharges to a §303(d)-listed impaired water body. |

**`data/tabular/02_clean/npdes/echo-naics-clean.csv`** (1,668 rows) and **`echo-sics-clean.csv`** (1,693 rows) — industry classifications normalized from the raw `UPPERCASE` schema:

| Column | Description |
|---|---|
| `npdes_id` | **(key)** NPDES permit number. |
| `code` | NAICS or SIC classification code. |
| `description` | Description of the code (e.g. `Water Supply and Irrigation Systems`). |
| `is_primary` | Whether this is the facility's primary industry classification (boolean). |

---

## Streamflow

**Source:** **USGS National Water Information System (NWIS) / Water Data for the
Nation.** <https://waterdata.usgs.gov/>.

These two files record how much water is moving through Iowa streams on a daily basis, as measured by USGS stream gauges. Streamflow is a critical covariate for water-quality prediction because it controls dilution (high flow lowers concentration even when pollutant loads are constant) and because storm-driven flow pulses flush nutrients from agricultural fields into waterways.

`usgs-iowa-discharge.csv` — daily streamflow values:

| Column | Description |
|---|---|
| `site_no` | **(key)** USGS gauge site number (joins to the gauges table). |
| `date` | Date of the daily value (timestamp, UTC). |
| `discharge_cfs` | Mean daily streamflow/discharge (cubic feet per second). |
| `discharge_cd` | USGS data-qualification code (e.g. `A` approved, `e` estimated). |

`usgs-iowa-gauges.csv` — gauge metadata:

| Column | Description |
|---|---|
| `site_no` | **(key)** USGS gauge site number. |
| `station_name` | Gauge name/description. |
| `latitude` | Gauge latitude. |
| `longitude` | Gauge longitude. |
| `drain_area_sqmi` | Upstream drainage area (square miles). |
| `huc8` | 8-digit HUC watershed code. **(key)** |
| `county_fips` | County FIPS code. |

### Cleaned streamflow outputs

Produced by `src/02_clean/tabular/streamflow/usgs-iowa-discharge-clean.ipynb` and `usgs-iowa-gauges-clean.ipynb`. Both tables retain the same columns as their raw counterparts (see [raw schema above](#streamflow)); dates are parsed to `YYYY-MM-DD`, numeric fields cast to the correct types, and rows with invalid site numbers removed.

**`data/tabular/02_clean/streamflow/usgs-iowa-discharge-clean.csv`** (556,850 rows) — daily discharge time series; schema identical to raw (`site_no`, `date`, `discharge_cfs`, `discharge_cd`).

**`data/tabular/02_clean/streamflow/usgs-iowa-gauges-clean.csv`** (703 rows) — gauge metadata; schema identical to raw (`site_no`, `station_name`, `latitude`, `longitude`, `drain_area_sqmi`, `huc8`, `county_fips`).

---

## Soil (SSURGO)

**Source:** **USDA-NRCS SSURGO** via Web Soil Survey / Soil Data Access.
<https://websoilsurvey.nrcs.usda.gov/>. `ssurgo-iowa-attributes.csv` — one row per
soil map-unit component.

This table describes the physical properties of Iowa soils at the map-unit level — drainage class, hydrologic soil group, saturated hydraulic conductivity, and available water capacity. These properties govern how quickly rainfall infiltrates or runs off into streams, making them essential for explaining spatial variation in nutrient and sediment delivery to waterways independently of land-use patterns.

| Column | Description |
|---|---|
| `mukey` | **(key)** Map-unit key; joins to the soil-polygon shapefile (`MUKEY`). |
| `muname` | Map-unit name (soil description). |
| `musym` | Map-unit symbol as shown on soil survey maps. |
| `areasymbol` | Soil survey area symbol (e.g. `IA181`, a county survey area). |
| `compname` | Component (soil series) name within the map unit. |
| `comppct_r` | Representative percent of the map unit this component occupies. |
| `hydgrp` | Hydrologic soil group (A–D) — runoff/infiltration potential. |
| `drainagecl` | Drainage class (e.g. `Well drained`, `Poorly drained`). |
| `ksat_r_mean` | Representative saturated hydraulic conductivity (Ksat), mean (µm/s). |
| `awc_r_mean` | Representative available water capacity, mean (cm water per cm soil). |

### Cleaned soil output

Produced by `src/02_clean/tabular/soil/ssurgo-iowa-attributes-clean.ipynb`. The dominant component per map unit is retained (one row per `mukey`); column names are expanded from abbreviated SSURGO codes to readable snake-case names:

**`data/tabular/02_clean/soil/ssurgo-iowa-attributes-clean.csv`** (10,572 rows):

| Column | Description |
|---|---|
| `mukey` | **(key)** SSURGO map-unit key; joins to the soil polygon shapefile. |
| `survey_area` | Soil survey area symbol (was `areasymbol`). |
| `map_unit_symbol` | Map-unit symbol as shown on soil survey maps (was `musym`). |
| `map_unit_name` | Map-unit description (was `muname`). |
| `dominant_component` | Name of the dominant soil series within the map unit (was `compname`). |
| `dominant_component_pct` | Percent of the map unit the dominant component represents (was `comppct_r`). |
| `hydrologic_group` | Hydrologic soil group A–D (was `hydgrp`). |
| `drainage_class` | Drainage class (was `drainagecl`). |
| `ksat_mean` | Representative saturated hydraulic conductivity, mean µm/s (was `ksat_r_mean`). |
| `awc_mean` | Representative available water capacity, mean cm/cm (was `awc_r_mean`). |

---

## Land Use (Cropland Data Layer)

**Source:** **USDA-NASS Cropland Data Layer (CDL)** 30 m classification, summarized
to HUC-12 watersheds. <https://nassgeodata.gmu.edu/CropScape/>. `cdl-huc12-fractions.csv`
— one row per HUC-12 per year; fractions sum across the land-cover classes.

This dataset captures what fraction of each HUC-12 watershed is covered by corn, soybeans, other crops, developed land, forest, pasture, wetland, and open water — updated annually from satellite-derived 30 m land-cover maps. Land-use composition is one of the strongest predictors of nutrient loading: watersheds with high row-crop fractions export far more nitrogen and phosphorus than forested or wetland-dominated watersheds.

| Column | Description |
|---|---|
| `year` | Year of the CDL classification (2015–2025). |
| `HUC_12` | **(key)** 12-digit HUC subwatershed code. |
| `pct_corn` | Fraction of the watershed classified as corn. |
| `pct_soybean` | Fraction classified as soybean. |
| `pct_other_crops` | Fraction in other crop types. |
| `pct_developed` | Fraction in developed/urban cover. |
| `pct_forest` | Fraction in forest. |
| `pct_pasture` | Fraction in pasture/grassland. |
| `pct_wetland` | Fraction in wetland. |
| `pct_open_water` | Fraction in open water. |
| `pct_row_crops` | Combined fraction in row crops (corn + soybean + similar). |

### Cleaned land use output

Produced by `src/02_clean/tabular/landuse/cdl-huc12-fractions-clean.ipynb`. The `HUC_12` key is renamed to `huc12_code` for consistency with the rest of the pipeline; fractions and row counts are range-validated.

**`data/tabular/02_clean/landuse/cdl-huc12-fractions-clean.csv`** (18,854 rows) — one row per `(huc12_code, year)`:

| Column | Description |
|---|---|
| `huc12_code` | **(key)** 12-digit HUC subwatershed code (was `HUC_12`). |
| `year` | CDL classification year (2015–2025). |
| `pct_corn` | Fraction of the watershed classified as corn. |
| `pct_soybean` | Fraction classified as soybean. |
| `pct_other_crops` | Fraction in other crop types. |
| `pct_developed` | Fraction in developed/urban cover. |
| `pct_forest` | Fraction in forest. |
| `pct_pasture` | Fraction in pasture/grassland. |
| `pct_wetland` | Fraction in wetland. |
| `pct_open_water` | Fraction in open water. |
| `pct_row_crops` | Combined fraction in row crops (corn + soybean + similar). |

---

## Census / Demographics

**Source:** **U.S. Census Bureau Population Estimates Program (PEP).**
<https://www.census.gov/programs-surveys/popest.html>. Two Excel workbooks of
annual county population estimates (`Iowa_Census-2010-2019.xlsx` intercensal;
`Iowa-Census-2020-2025.xlsx` vintage 2020s). These are formatted PEP tables, not
flat CSVs: a few title rows precede the data, then:

These workbooks provide annual mid-year population estimates for each Iowa county from 2010 through 2025. Population is a proxy for urban wastewater load — a county with rapidly growing cities will have increasing municipal sewage discharge even if land use is otherwise unchanged — and can help separate point-source from non-point-source contributions to water-quality trends over time.

| Column | Description |
|---|---|
| `Geographic Area` | **(key)** County (or place) name; the leading `.` prefix is the Census formatting convention. |
| `April 1, 20X0 Estimates Base` | The decennial census base population the annual series is anchored to. |
| `<year>` (one column per year) | Mid-year (July 1) population estimate for that county for each year in the workbook's range (2010–2019 or 2020–2025). |

### Cleaned census outputs

Produced by `src/02_clean/tabular/census/census-population-2010-2020-clean.ipynb` and `census-population-2020-2025-clean.ipynb`. The wide Excel tables are melted from one-column-per-year to long format.

**`data/tabular/02_clean/census/iowa-census-population-2010-2020-clean.csv`** (11,304 rows) and **`iowa-census-population-2020-2025-clean.csv`** (6,573 rows) — both share the same schema, one row per `(place, year, estimate_type)`:

| Column | Description |
|---|---|
| `place` | City name. |
| `place_type` | `city`. |
| `year` | Year of the estimate (2010–2020 or 2020–2025). |
| `estimate_type` | `estimates_base` (April 1 decennial census anchor) or `july_estimate` (mid-year annual estimate). |
| `population` | Population count. |

---

## Spatial Layers

**Source:** **USGS/EPA NHDPlus V2.1** and the **Watershed Boundary Dataset (WBD)**
(<https://www.epa.gov/waterdata/nhdplus-national-data>), plus the **NRCS SSURGO**
soil polygons. Shapefiles (`.shp` + `.dbf` attributes).

These vector layers define the geographic scaffolding that all other datasets are joined to: HUC-12 watershed polygons delineate the contributing areas used for land-use and BMP aggregation, the NHDPlus gauge-location shapefile snaps USGS stream gauges onto the stream network for accurate upstream-area calculations, and the SSURGO soil polygons provide the spatial extent to aggregate tabular soil properties to watersheds. Together they are the spatial glue connecting measurements taken at points to the landscape units used for modeling.

`wbd-huc12-iowa/WBDSnapshot_Iowa.shp` — HUC-12 watershed boundary polygons:

| Column | Description |
|---|---|
| `OBJECTID_1`, `OBJECTID` | Internal feature IDs. |
| `HUC_8` / `HUC_10` / `HUC_12` | **(key)** Nested 8/10/12-digit hydrologic unit codes for the polygon. |
| `ACRES` | Polygon area (acres). |
| `NCONTRB_A` | Non-contributing drainage area within the unit. |
| `HU_10_GNIS` / `HU_12_GNIS` | GNIS IDs for the HUC-10 / HUC-12 features. |
| `HU_10_DS` / `HU_12_DS` | Downstream hydrologic unit codes. |
| `HU_10_NAME` / `HU_12_NAME` | Names of the HUC-10 / HUC-12 units. |
| `HU_10_MOD` / `HU_12_MOD` | Modification flags for the unit boundaries. |
| `HU_10_TYPE` / `HU_12_TYPE` | Hydrologic unit type codes. |
| `META_ID` | Metadata record ID. |
| `STATES` | States the unit intersects. |
| `GlobalID` | Global unique feature identifier. |
| `SHAPE_Leng` / `Shape_Area` | Polygon perimeter and area in the layer's projection units. |
| `GAZ_ID` | Gazetteer ID. |
| `WBD_Date` | Date of the WBD snapshot. |
| `VPUID` | NHDPlus Vector Processing Unit ID. |
| `AreaHUC12` | Computed HUC-12 area. |

`gage-loc/GageLoc.shp` — stream-gauge locations indexed onto the NHD network:

| Column | Description |
|---|---|
| `COMID` | **(key)** NHDPlus common identifier of the flowline the gauge sits on. |
| `EVENTDATE` | Date the gauge event was recorded on the network. |
| `REACHCODE` | NHD reach code of the associated reach. |
| `REACHSMDAT` | Reach "snapshot" date. |
| `REACHRESOL` | Resolution of the reach (e.g. high/medium). |
| `FEATURECOM` / `FEATURECLA` | Feature COMID and classification codes. |
| `SOURCE_ORI` | Originating source agency of the gauge. |
| `SOURCE_DAT` | Source dataset of the gauge. |
| `SOURCE_FEA` | Source feature identifier (typically the USGS site number). |
| `FEATUREDET` | Feature detail/description text. |
| `Measure` | Measure (percent distance) along the reach where the gauge is located. |
| `Offset` | Offset distance from the reach to the gauge point. |
| `EventType` | Type of network event (gauge). |
| `FLComID` | COMID of the flowline the event is linked to. |

`ssurgo/iowa-mapunit-polygons.shp` — SSURGO soil map-unit polygons:

| Column | Description |
|---|---|
| `AREASYMBOL` | Soil survey area symbol. |
| `SPATIALVER` | Spatial version of the SSURGO data. |
| `MUSYM` | Map-unit symbol. |
| `MUKEY` | **(key)** Map-unit key; joins to `ssurgo-iowa-attributes.csv`. |
| `areasymb_1` | Duplicate/secondary area symbol from the merge of county surveys. |

Also present: `cdl/cdl_iowa_YYYY.tif` (2015–2025) — the raw 30 m CDL land-cover
rasters underlying the HUC-12 fractions; `nhdplus/crosswalk/` — the NHDPlus
catchment ↔ WBD HUC-12 crosswalk table linking COMIDs to HUC-12 codes.

### Cleaned spatial outputs

Produced by `src/02_clean/spatial/nhdplus/wbd-huc12-crosswalk-clean.ipynb`, which
reprojects the WBD HUC-12 polygons to WGS84 and spatially joins each EPA
monitoring station to the watershed polygon it falls within.

**`data/spatial/02_clean/nhdplus/wbd-huc12-station-crosswalk-clean.csv`**
(1,666 rows) — one row per `MonitoringLocationIdentifier`, giving the HUC-12/10/8
watershed it sits in. This is the crosswalk the future land-use/BMP merge step
uses to attach `cdl-huc12-fractions-clean.csv` and `iowa-nrs-bmp-huc8-clean.csv`
to individual stations.

| Column | Description |
|---|---|
| `MonitoringLocationIdentifier` | **(key)** EPA WQX station ID; joins to the water-quality and station tables. |
| `huc12_code` | **(key)** 12-digit HUC of the watershed containing the station; joins to `cdl-huc12-fractions-clean.csv`. |
| `huc12_name` | HUC-12 watershed name. |
| `huc10_code` | 10-digit HUC (parent of `huc12_code`). |
| `huc8_code` | **(key)** 8-digit HUC (parent of `huc10_code`); joins to `iowa-nrs-bmp-huc8-clean.csv` and matches the station table's own `HUCEightDigitCode` (verified during cleaning — 0 mismatches). |
| `huc12_acres` | Watershed polygon area (acres). |

Produced by `src/02_clean/spatial/ssurgo/mapunit-crosswalk-clean.ipynb`, which
spatially joins each EPA monitoring station to the SSURGO soil map-unit polygon
it falls within (falling back to the nearest polygon, within 500 m, for the
handful of stations that sit just outside the mapped extent — typically
in-stream/lake sampling points).

**`data/spatial/02_clean/ssurgo/ssurgo-mapunit-station-crosswalk-clean.csv`**
(1,666 rows) — one row per `MonitoringLocationIdentifier`, giving the soil map
unit it sits in. This is the crosswalk the future soil merge step uses to
attach `ssurgo-iowa-attributes-clean.csv` to individual stations. **Coverage
note:** because stations sit in or next to streams/lakes, only ~1,077 of 1,666
(65%) resolve to a `mukey` present in `ssurgo-iowa-attributes-clean.csv` — the
rest land in non-soil water map units (`map_unit_symbol` `W`/`RIVER`/`LAKE`)
that carry no soil component data. That's an expected property of station
placement, not a join defect.

| Column | Description |
|---|---|
| `MonitoringLocationIdentifier` | **(key)** EPA WQX station ID; joins to the water-quality and station tables. |
| `mukey` | **(key)** SSURGO map-unit key; joins to `ssurgo-iowa-attributes-clean.csv` (~65% coverage — see note above). |
| `map_unit_symbol` | Map-unit symbol as shown on soil survey maps; `W`/`RIVER`/`LAKE` indicate non-soil water map units. |
| `survey_area` | County soil survey area symbol (e.g. `IA181`) the polygon belongs to. |
| `match_method` | `within` (station falls inside the polygon) or `nearest` (fallback snap for the ~5 stations outside the mapped extent). |
| `match_distance_m` | Distance to the matched polygon in meters; `0` for `within` matches. |

---

## Imagery & Text

**Imagery** (`data/images/water-images/train/`) — a small training set for visual
water-quality classification: **24** images in `Clean-samples/` and **16** in
`Dirty-samples/`. The folder name is the class label; each file is a photograph of
water.

This image set supports a computer-vision sub-task: classifying whether a photograph of water looks visibly clean or polluted. With only 40 labeled photos it is far too small for standalone deep learning, but it can be used for transfer-learning experiments or to demonstrate a multimodal pipeline that fuses visual signals with the tabular sensor data.

**Text** (`data/text/raw/`) — **20** plain-text narrative summaries, one per Iowa
city (Ankeny, Bettendorf, Cedar Rapids, Coralville, Council Bluffs, Davenport, Des
Moines, Eldora, Fort Dodge, Iowa City, Iowa Falls, Johnston, Marengo, Marion,
Marshalltown, Mason City, Ottumwa, Urbandale, Waterloo, Waukee). Each file is
free-form prose covering the city's geography, history, climate, and water
context — not a columnar dataset.

These narratives provide unstructured contextual knowledge about the cities in the study area — their geographic setting, water sources, and local environmental history. They are intended for retrieval-augmented generation (RAG) or LLM-based question-answering tasks where qualitative city context enriches otherwise purely quantitative model outputs.
