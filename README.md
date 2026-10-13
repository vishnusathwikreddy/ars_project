# Forest AGBD in Betul (MP) from GEDI L4A + Sentinel-2

GEDI L4A footprint AGBD (response) + Sentinel-2 L2A spectral/vegetation variables
(predictors) -> Random Forest -> wall-to-wall AGBD map -> analysis against terrain,
forest type and disturbance history. Data processing runs in Google Earth Engine (Python API);
model selection runs locally with scikit-learn.

## Setup
```bash
pip install -r requirements.txt
earthengine authenticate --auth_mode=notebook --quiet      # prints URL + code verifier
earthengine authenticate --authorization-code=<CODE> --code-verifier=<VERIFIER>
```
Then set `GEE_PROJECT` in `config.py`.

## Run order
| Step | Script | Where | Output |
|---|---|---|---|
| 0 | `00_check_access.py` | GEE | confirms all datasets resolve for Betul, creates asset folder |
| 1 | `01_gedi_inventory.py` | GEE | filter cascade, shots by year/month, 5 km density grid, **study-area selection** |
| 2 | `02_build_training.py` | GEE | GEDI footprints + predictors -> asset, Drive CSV, `data/training_points.csv` |
| 3 | `03_train_model.py` | local | feature-set comparison, spatial CV, tuning, importance, `outputs/model_config.json` |
| 4 | `04_predict_map.py` | GEE | GEE/sklearn parity check, AGBD map (asset + GeoTIFF on Drive), totals |
| 5 | `05_drivers_analysis.py` | GEE + local | AGBD vs elevation, slope, aspect, forest type, canopy density, disturbance (`--layers-only` pre-exports the class/covariate layers; can run before step 4) |

## Method summary (defaults in `config.py`)
- **GEDI L4A** (`LARSE/GEDI/GEDI04_A_002_MONTHLY`, 2019-04 to 2025-06): `l4_quality_flag=1`, `degrade_flag=0`,
  `algorithm_run_flag=1`, sensitivity >= 0.95, **leaf-on months only (Jul-Jan)** because Betul's
  forests are deciduous and GEDI's `leaf_off_flag` is always 0 here, 0-500 Mg/ha. Footprints must lie in homogeneous WorldCover tree cover (20 m radius, to absorb
  ~10 m geolocation error) with no Hansen loss since 2019.
- **Sentinel-2 L2A** (2020-2024), Cloud Score+ masking (cs_cdf >= 0.6), two seasonal medians:
  post-monsoon leaf-on (Oct-Jan) and dry-season leaf-off (Mar-May). 10 bands + NDVI, EVI, SAVI,
  NDMI, NBR, NDRE, CIre, MTCI, IRECI, S2REP per season, seasonal differences, GLCM texture of NIR.
  Optical predictors are averaged over a 12.5 m radius circle to match the GEDI footprint.
- **Terrain** from Copernicus GLO-30 (2024 release): elevation, slope, northness, eastness, TPI (300 m).
- **Model**: Random Forest (in GEE a smaller forest, `GEE_RF_OVERRIDES`, because the full one exceeds EE's model-size limit); 5-fold **spatial block CV** (5 km blocks) because neighbouring footprints
  are autocorrelated; random CV reported only to show its optimism.
- **Forest type**: phenology proxy (deciduousness terciles) unless a forest-type map is supplied
  via `FOREST_TYPE_ASSET`; canopy-density classes follow FSI thresholds (10/40/70 %) on Hansen 2000.
- **Disturbance**: Hansen GFC v1.13 loss year, grouped 2001-2010 / 2011-2018 / 2019-2025.

## Known limitations to discuss in the report
- GEDI L4A is itself a model prediction (see `agbd_se`), so the map is "GEDI-equivalent" AGBD,
  not field-measured biomass.
- Optical data saturate in dense canopies: expect underestimation at high AGBD
  (`outputs/tables/03_accuracy_by_agbd_range_*.csv`).
- Significance tests in step 5 ignore spatial autocorrelation and overstate significance.

## Implementation notes
- In the monthly GEDI L4A rasters `lat_lowestmode`/`lon_lowestmode` are empty; step 2 positions
  each footprint with `lat/lon_lowestmode_a1` (falls back to the 25 m pixel centre).
- Step 5 materialises its class layers as `<ASSET_ROOT>/driver_layers`, because the
  Sentinel-2-derived deciduousness times out in interactive requests.
