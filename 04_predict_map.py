"""
Step 4: wall-to-wall AGBD prediction in Earth Engine with the Random Forest
configuration selected in step 3.

  1. parity check: train the GEE RF without the spatial hold-out blocks used in
     step 3 and score those blocks (confirms GEE ~ scikit-learn accuracy)
  2. train on all footprints, predict over tree cover in the study area
  3. export the map to an asset (for step 5) and a GeoTIFF to Google Drive
  4. quick-look PNG and AGB totals

    python 04_predict_map.py            # reuses an existing map asset
    python 04_predict_map.py --force    # re-run the prediction
"""
import json
import sys
import urllib.request

import ee
import numpy as np

import config as C
import gee_common as G

TRAIN_ASSET = f"{C.ASSET_ROOT}/training_points"
MAP_ASSET = f"{C.ASSET_ROOT}/agbd_pred"


def rf(cfg):
    params = {**cfg["rf"], **C.GEE_RF_OVERRIDES}
    return ee.Classifier.smileRandomForest(**params).setOutputMode("REGRESSION")


def training_fc(feats):
    fc = ee.FeatureCollection(TRAIN_ASSET).filter(ee.Filter.notNull(feats + ["agbd"]))
    return fc.randomColumn("rnd", C.SEED)


def parity_check(cfg, fc):
    feats = cfg["features"]
    hold = ee.Filter.inList("block", cfg["holdout_blocks"])
    train = fc.filter(hold.Not())
    n = train.size()
    train = train.filter(ee.Filter.lt("rnd", ee.Number(C.GEE_MAX_TRAIN).divide(n)))
    model = rf(cfg).train(train, "agbd", feats)
    scored = fc.filter(hold).classify(model, "pred").select(["agbd", "pred"])
    df = G.fc_to_df(scored)
    y, p = df.agbd.values, df.pred.values
    rmse = np.sqrt(np.mean((p - y) ** 2))
    r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"GEE hold-out parity check: n={len(df)}  R2={r2:.3f}  RMSE={rmse:.1f} Mg/ha "
          f"(local spatial CV: R2={cfg['cv_spatial']['r2']:.3f}, RMSE={cfg['cv_spatial']['rmse']:.1f})")
    return dict(n=int(len(df)), r2=float(r2), rmse=float(rmse))


def main():
    cfg = json.loads(C.MODEL_JSON.read_text())
    feats = cfg["features"]
    G.init()
    region = G.study_area()
    fc = training_fc(feats)

    cfg["gee_rf"] = {**cfg["rf"], **C.GEE_RF_OVERRIDES}
    cfg["gee_parity"] = parity_check(cfg, fc)

    # prediction is the expensive part (~5 h); reuse an existing map unless --force
    if "--force" in sys.argv or not G.asset_exists(MAP_ASSET):
        n = fc.size()
        train = fc.filter(ee.Filter.lt("rnd", ee.Number(C.GEE_MAX_TRAIN).divide(n)))
        model = rf(cfg).train(train, "agbd", feats)
        pred = (G.predictor_stack(region).select(feats).classify(model, "agbd")
                .updateMask(G.tree_mask()).clip(region).toFloat())
        G.delete_asset_if_exists(MAP_ASSET)
        G.run_tasks([
            ee.batch.Export.image.toAsset(pred, "agbd_pred_asset", MAP_ASSET, region=region,
                                          crs=C.CRS, scale=C.OUTPUT_SCALE, maxPixels=1e11,
                                          pyramidingPolicy={"agbd": "mean"}),
        ], poll_s=60)
    else:
        print(f"Reusing existing map {MAP_ASSET} (pass --force to re-predict)")

    img = ee.Image(MAP_ASSET)
    # GeoTIFF from the asset: a pixel copy, much cheaper than re-running the prediction
    G.run_tasks([
        ee.batch.Export.image.toDrive(img, "agbd_pred_geotiff", folder=C.DRIVE_FOLDER,
                                      fileNamePrefix="betul_agbd_pred", region=region,
                                      crs=C.CRS, scale=C.OUTPUT_SCALE, maxPixels=1e11,
                                      formatOptions={"cloudOptimized": True}),
    ], poll_s=60)

    ha = ee.Image.pixelArea().divide(1e4)
    stats = ee.Image.cat([img.multiply(ha).rename("agb_Mg"),
                          img.mask().gt(0).multiply(ha).rename("forest_ha")]) \
        .reduceRegion(ee.Reducer.sum(), region, scale=C.OUTPUT_SCALE, crs=C.CRS,
                      maxPixels=1e11, tileScale=8).getInfo()
    totals = dict(mean_agbd_Mg_ha=stats["agb_Mg"] / stats["forest_ha"],
                  forest_area_ha=stats["forest_ha"],
                  total_agb_Tg=stats["agb_Mg"] / 1e6,
                  total_carbon_Tg=0.47 * stats["agb_Mg"] / 1e6)
    cfg["map_totals"] = totals
    print("Map totals:", json.dumps(totals, indent=2))
    C.MODEL_JSON.write_text(json.dumps(cfg, indent=2))

    vis = {"min": 0, "max": 200, "palette": ["#f7fcb9", "#addd8e", "#41ab5d", "#006837", "#002d1a"],
           "dimensions": 1600, "region": region.bounds(maxError=100), "format": "png"}
    url = img.getThumbURL(vis)
    urllib.request.urlretrieve(url, C.FIG / "04_agbd_map_quicklook.png")
    print(f"Saved {C.FIG / '04_agbd_map_quicklook.png'} (GeoTIFF in Drive/{C.DRIVE_FOLDER})")


if __name__ == "__main__":
    main()
