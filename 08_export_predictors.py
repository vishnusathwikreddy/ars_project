"""
Step 8: materialise the predictor stack used by the models as an Earth Engine
asset, so step 9 can download it and predict maps locally with models Earth
Engine cannot run (XGBoost, neural network).

Only tree-cover pixels in the study area are kept, at the map resolution (25 m).
This is the same computation as the step-4 prediction and takes several hours.

Output
  asset  <ASSET_ROOT>/predictor_stack

    python 08_export_predictors.py
"""
import json

import ee

import config as C
import gee_common as G

STACK_ASSET = f"{C.ASSET_ROOT}/predictor_stack"


def main():
    feats = json.loads(C.MODEL_JSON.read_text())["features"]
    G.init()
    region = G.study_area()
    stack = (G.predictor_stack(region).select(feats)
             .updateMask(G.tree_mask()).clip(region).toFloat())
    G.delete_asset_if_exists(STACK_ASSET)
    G.run_tasks([ee.batch.Export.image.toAsset(
        stack, "predictor_stack_asset", STACK_ASSET, region=region, crs=C.CRS,
        scale=C.OUTPUT_SCALE, maxPixels=1e11, pyramidingPolicy={".default": "mean"})], poll_s=120)
    print(f"Exported {len(feats)} predictors to {STACK_ASSET}")


if __name__ == "__main__":
    main()
