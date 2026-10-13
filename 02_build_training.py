"""
Step 2: build the training table = quality-filtered GEDI L4A footprints with
Sentinel-2 / terrain predictors and analysis attributes extracted at each one.

Outputs
  asset  <ASSET_ROOT>/training_points      (used by GEE prediction, script 04)
  Drive  <DRIVE_FOLDER>/training_points.csv
  data/training_points.csv                 (used by local modelling, script 03)

    python 02_build_training.py            # export + download
    python 02_build_training.py --download # only re-download the asset
"""
import sys

import ee
import pandas as pd

import config as C
import features as F
import gee_common as G

TRAIN_ASSET = f"{C.ASSET_ROOT}/training_points"


def gedi_points(region):
    static = G.training_mask()

    def to_points(img):
        # lat/lon_lowestmode are empty in the monthly rasters; use the algorithm-setting-1
        # geolocation instead, falling back to the 25 m pixel centre if that is missing too
        ll = ee.Image.pixelLonLat()
        coords = ee.Image.cat([
            img.select("lat_lowestmode_a1").unmask(ll.select("latitude")).rename("lat_lowestmode"),
            img.select("lon_lowestmode_a1").unmask(ll.select("longitude")).rename("lon_lowestmode"),
        ]).updateMask(img.select("agbd").mask())
        img = (img.addBands(coords, overwrite=True).select(F.GEDI_ATTRS).updateMask(static))
        pts = img.sample(region=region, projection=img.select("agbd").projection(),
                         dropNulls=True, tileScale=8, geometries=False)
        # place each sample at the true footprint centre, not the 25 m pixel centre
        return pts.map(lambda f: f.setGeometry(
            ee.Geometry.Point([f.getNumber("lon_lowestmode"), f.getNumber("lat_lowestmode")])))

    return G.gedi_quality(region).map(to_points).flatten()


def add_blocks(fc):
    """UTM coordinates and spatial block id for blocked cross-validation."""
    def f(feat):
        xy = feat.geometry().transform(C.CRS, 1).coordinates()
        x, y = ee.Number(xy.get(0)), ee.Number(xy.get(1))
        bx = x.divide(C.SPATIAL_BLOCK_M).floor()
        by = y.divide(C.SPATIAL_BLOCK_M).floor()
        return feat.set({"x_utm": x, "y_utm": y, "block": bx.multiply(100000).add(by).toInt64()})
    return fc.map(f)


def build():
    G.init()
    G.ensure_asset_folder()
    region = G.study_area()
    pts = gedi_points(region)
    stack = G.predictor_stack(region).addBands(G.analysis_layers(region))
    samples = stack.sampleRegions(collection=pts, scale=10, projection=C.CRS,
                                  tileScale=16, geometries=True)
    samples = add_blocks(samples)

    G.delete_asset_if_exists(TRAIN_ASSET)
    G.run_tasks([
        ee.batch.Export.table.toAsset(samples, "training_points_asset", TRAIN_ASSET),
        ee.batch.Export.table.toDrive(samples, "training_points_csv", folder=C.DRIVE_FOLDER,
                                      fileNamePrefix="training_points", fileFormat="CSV"),
    ])


def download():
    G.init()
    df = G.fc_to_df(ee.FeatureCollection(TRAIN_ASSET))
    # delta_time = seconds since 2018-01-01T00:00:00Z
    df["date"] = pd.to_datetime("2018-01-01") + pd.to_timedelta(df["delta_time"], unit="s")
    df.to_csv(C.TRAINING_CSV, index=False)
    print(f"{len(df):,} footprints -> {C.TRAINING_CSV}")
    print(df[["agbd", "agbd_se", "sensitivity"]].describe().round(2).to_string())
    print("Footprints per year:\n", df["date"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    if "--download" not in sys.argv:
        build()
    download()
