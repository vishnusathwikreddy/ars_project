"""
Predictor names, kept free of Earth Engine imports so the local modelling
scripts can use them. gee_common.predictor_stack() produces exactly these bands.
"""
import config as C

INDICES = ["NDVI", "EVI", "SAVI", "NDMI", "NBR", "NDRE", "CIre", "MTCI", "IRECI", "S2REP"]
TEXTURE_METRICS = ["contrast", "ent", "var", "idm"]

S2_ON = [f"{b}_on" for b in C.S2_BANDS + INDICES]
S2_OFF = [f"{b}_off" for b in C.S2_BANDS + INDICES]
DELTA = ["dNDVI", "dEVI", "dNDMI", "dNDRE"]
TEXTURE = [f"TEX_{m}_on" for m in TEXTURE_METRICS] if C.USE_TEXTURE else []
TERRAIN = ["elev", "slope", "northness", "eastness", "tpi"]

ALL_PREDICTORS = S2_ON + S2_OFF + DELTA + TEXTURE + TERRAIN

# Feature sets compared in script 03 (the project focus is Sentinel-2; terrain
# is tested as an additional covariate set).
FEATURE_SETS = {
    "S2_leafon": S2_ON + TEXTURE,
    "S2_twoseason": S2_ON + S2_OFF + DELTA + TEXTURE,
    "S2_twoseason_terrain": S2_ON + S2_OFF + DELTA + TEXTURE + TERRAIN,
}

# Non-predictor attributes attached to every training point / pixel sample,
# used for stratified analysis (script 05).
ANALYSIS_ATTRS = ["tc2000", "lossyear", "wc_class", "decid"]

# GEDI attributes kept for each footprint.
GEDI_ATTRS = ["agbd", "agbd_se", "sensitivity", "beam", "leaf_off_flag",
              "solar_elevation", "pft_class", "delta_time",
              "lat_lowestmode", "lon_lowestmode"]
