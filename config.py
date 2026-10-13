"""
Central configuration for the Betul AGBD project.

Edit the values in the "USER SETTINGS" block first. The rest are
methodological defaults that you can tune and report in the methods section.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# USER SETTINGS (fill these in)
# ----------------------------------------------------------------------------
GEE_PROJECT = "arsproject-511412"
ASSET_ROOT = f"projects/{GEE_PROJECT}/assets/betul_agb"   # created automatically
DRIVE_FOLDER = "betul_agb"                      # Google Drive folder for exports

# Study-area boundary. By default Betul is taken from FAO GAUL level-2.
# If you have your own boundary uploaded as an asset (e.g. Survey of India
# district shapefile), set AOI_ASSET to its ID and it will be used instead.
AOI_ASSET = None
GAUL_LEVEL2 = "FAO/GAUL/2025/level2"
GAUL_STATE_FIELD, GAUL_DISTRICT_FIELD = "GAUL1_NAME", "GAUL2_NAME"
DISTRICT_NAME = "Betul"
STATE_NAME = "Madhya Pradesh"

# Optional forest-type map (e.g. FSI forest-type / Champion & Seth classes)
# uploaded as an image asset with an integer class band. If None, a
# phenology-based (deciduousness) proxy is used for forest-type stratification.
FOREST_TYPE_ASSET = None
FOREST_TYPE_BAND = "b1"

# ----------------------------------------------------------------------------
# DATASETS (Earth Engine catalogue IDs)
# ----------------------------------------------------------------------------
# NOTE: the Earth Engine catalogue currently serves GEDI L4A as Version 2.1
# (GEDI04_A_002). If a V3 asset ID is available to you, put it here; the
# band names are expected to be the same.
GEDI_L4A = "LARSE/GEDI/GEDI04_A_002_MONTHLY"
S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUDSCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
WORLDCOVER = "ESA/WorldCover/v200"              # 2021 map, 10 m
DEM = "COPERNICUS/DEM/GLO30_2024_1"             # Copernicus GLO-30, 2024 release
HANSEN = "UMD/hansen/global_forest_change_2025_v1_13"
HANSEN_LAST_YEAR = 2025

# ----------------------------------------------------------------------------
# GEDI L4A QUALITY FILTERING
# ----------------------------------------------------------------------------
GEDI_START = "2019-04-01"
GEDI_END = "2025-07-01"
GEDI_FILTER = dict(
    l4_quality_flag=1,        # recommended L4A quality flag
    degrade_flag=0,           # no degraded pointing/positioning
    algorithm_run_flag=1,
    min_sensitivity=0.95,     # L4A recommended threshold (already implied by l4_quality_flag)
    leaf_on_only=True,        # deciduous forests: keep only GEDI_LEAF_ON_MONTHS
    power_beams_only=False,   # True = only full-power beams (5, 6, 8, 11)
    night_only=False,         # True = solar_elevation < 0 (less noise)
    max_rel_se=None,          # e.g. 1.0 -> drop shots with agbd_se/agbd > 1
    agbd_min=0.0,
    agbd_max=500.0,           # Central Indian forests rarely exceed this
)
POWER_BEAMS = [5, 6, 8, 11]

# Leaf-on season for Betul's dry/moist deciduous forests: monsoon and post-monsoon
# full canopy (Jul-Jan); leaf fall runs Feb-Jun. GEDI's own leaf_off_flag is always 0
# here and leaf_on_cycle follows crop/grass phenology, so months are used instead.
GEDI_LEAF_ON_MONTHS = (7, 1)     # calendar months, wraps over the year end

# Keep only shots inside homogeneous forest: every WorldCover pixel within
# this radius of the footprint centre must be tree cover. Reduces errors from
# GEDI geolocation uncertainty (~10 m) at forest edges.
HOMOGENEOUS_FOREST_RADIUS_M = 20

# Exclude shots/pixels where Hansen detected loss in or after this year,
# because canopy then differs between the GEDI date and the S2 composite.
EXCLUDE_LOSS_FROM_YEAR = 2019

# ----------------------------------------------------------------------------
# SENTINEL-2 PREDICTORS
# ----------------------------------------------------------------------------
S2_START = "2020-01-01"
S2_END = "2024-12-31"
CLOUDSCORE_THRESHOLD = 0.60     # Cloud Score+ cs_cdf; higher = stricter
MAX_SCENE_CLOUD = 70            # scene-level pre-filter (%)
# Two seasonal composites capture the deciduous phenology of Central India:
#   leaf-on  = post-monsoon (Oct-Jan), canopy fully green, few clouds
#   leaf-off = dry season (Mar-May), deciduous species shed leaves
SEASONS = {
    "on": (10, 1),     # calendar months, wraps over the year end
    "off": (3, 5),
}
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
USE_TEXTURE = True              # GLCM texture on leaf-on NIR
TEXTURE_SIZE = 2                # GLCM neighbourhood half-size (pixels)

# Predictors are averaged over a circle matching the GEDI footprint (25 m
# diameter) so that a training sample describes the same ground area.
FOOTPRINT_RADIUS_M = 12.5

# Working projection: UTM 43N covers Betul (~77.1-78.6 E).
CRS = "EPSG:32643"
OUTPUT_SCALE = 25               # prediction map resolution (m)

# ----------------------------------------------------------------------------
# STUDY AREA SELECTION (script 01)
# ----------------------------------------------------------------------------
GRID_SIZE_M = 5000              # cell size for GEDI density mapping
MIN_SHOTS_PER_CELL = 30         # cells with fewer filtered shots are dropped
MIN_FOREST_FRACTION = 0.20      # and cells with too little forest
USE_SELECTED_STUDY_AREA = True  # False = model the whole district

# ----------------------------------------------------------------------------
# MODELLING (scripts 03-04)
# ----------------------------------------------------------------------------
SEED = 42
SPATIAL_BLOCK_M = 5000          # block size for spatial cross-validation
N_FOLDS = 5
GEE_MAX_TRAIN = 60000           # subsample for GEE smileRandomForest memory
# The step-3 forest (300 trees, leaf 5, ~1.5M nodes) exceeds Earth Engine's
# "Computed value is too large" limit, so GEE uses a smaller forest. Local spatial
# CV: 150 trees/leaf 10 R2=0.293 RMSE=47.6 vs 300/5 R2=0.296 RMSE=47.5.
GEE_RF_OVERRIDES = dict(numberOfTrees=150, minLeafPopulation=10)

# ----------------------------------------------------------------------------
# LOCAL PATHS
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _p in (DATA, OUT, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

TRAINING_CSV = DATA / "training_points.csv"
MODEL_JSON = OUT / "model_config.json"
