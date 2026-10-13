"""
Step 5: how predicted AGBD varies with elevation, slope, aspect, forest type,
canopy density and disturbance history. Observed GEDI footprints are analysed
alongside the map as an independent check of each pattern.

Outputs
  outputs/tables/05_zonal_<factor>.csv     full-map zonal statistics per class
  outputs/tables/05_tests.csv              Spearman / Kruskal-Wallis tests
  outputs/figures/05_*.png
  data/map_pixel_sample.csv

    python 05_drivers_analysis.py
    python 05_drivers_analysis.py --layers-only   # pre-export driver layers (no map needed)
"""
import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config as C
import gee_common as G

MAP_ASSET = f"{C.ASSET_ROOT}/agbd_pred"
LAYERS_ASSET = f"{C.ASSET_ROOT}/driver_layers"
SCALE = 50              # zonal-stat / sampling resolution (m)
N_SAMPLE = 30000

LOSS_LABELS = {0: "No loss", 1: "Loss 2001-2010", 2: "Loss 2011-2018",
               3: f"Loss 2019-{C.HANSEN_LAST_YEAR}"}
CANOPY_LABELS = {0: "<10%", 1: "Open (10-40%)", 2: "Mod. dense (40-70%)", 3: "Very dense (>=70%)"}
SLOPE_LABELS = {0: "0-5°", 1: "5-10°", 2: "10-20°", 3: "20-30°", 4: ">30°"}
ASPECT_LABELS = {0: "N", 1: "E", 2: "S", 3: "W"}


def class_layers(region):
    t = G.terrain()
    h = G.hansen()
    ly = h.select("lossyear")
    loss_period = (ly.gt(0).add(ly.gt(10)).add(ly.gt(18))).rename("loss_period")
    tc = h.select("treecover2000")
    canopy = tc.gte(10).add(tc.gte(40)).add(tc.gte(70)).rename("canopy_class")
    slope = t.select("slope")
    slope_cls = (slope.gte(5).add(slope.gte(10)).add(slope.gte(20)).add(slope.gte(30))).rename("slope_class")
    asp = ee.Terrain.aspect(t.select("elev"))
    aspect_cls = asp.add(45).mod(360).divide(90).floor().rename("aspect_class")
    elev_cls = t.select("elev").divide(100).floor().multiply(100).rename("elev_class")

    if C.FOREST_TYPE_ASSET:
        ftype = ee.Image(C.FOREST_TYPE_ASSET).select(C.FOREST_TYPE_BAND).rename("forest_type")
    else:
        decid = G.deciduousness(region)
        q = decid.updateMask(G.tree_mask()).reduceRegion(
            ee.Reducer.percentile([33, 67]), region, scale=100, maxPixels=1e10, tileScale=8)
        ftype = (decid.gt(ee.Number(q.get("decid_p33"))).add(decid.gt(ee.Number(q.get("decid_p67"))))
                 .add(1).rename("forest_type"))
    return ee.Image.cat([elev_cls, slope_cls, aspect_cls, canopy, ftype, loss_period]).toInt()


FTYPE_LABELS = ({1: "Low deciduousness", 2: "Medium deciduousness", 3: "High deciduousness"}
                if not C.FOREST_TYPE_ASSET else {})


def zonal(pred, cls_img, band, region):
    red = (ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
           .combine(ee.Reducer.percentile([25, 50, 75]), sharedInputs=True)
           .combine(ee.Reducer.count(), sharedInputs=True)
           .group(groupField=1, groupName=band))
    out = pred.addBands(cls_img.select(band)).reduceRegion(
        red, region, scale=SCALE, crs=C.CRS, maxPixels=1e11, tileScale=8).getInfo()
    df = pd.DataFrame(out["groups"])
    df["area_ha"] = df["count"] * SCALE * SCALE / 1e4
    return df


def boxplot(ax, df, col, labels, title):
    keys = [k for k in sorted(df[col].dropna().unique())]
    data = [df.loc[df[col] == k, "agbd"].values for k in keys]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(keys) + 1),
                  [labels.get(int(k), str(int(k))) for k in keys], rotation=30, ha="right")
    ax.set(ylabel="AGBD (Mg/ha)", title=title)


def binned(ax, df, x, bins, label, color):
    b = pd.cut(df[x], bins)
    g = df.groupby(b, observed=True)["agbd"].agg(["mean", "std", "count"])
    g = g[g["count"] >= 20]
    mid = [i.mid for i in g.index]
    ax.plot(mid, g["mean"], "o-", color=color, label=label)
    ax.fill_between(mid, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15)


def export_layers(region):
    """Materialise class + covariate layers once: the S2-derived deciduousness is too
    expensive to recompute in interactive requests (they time out)."""
    if G.asset_exists(LAYERS_ASSET):
        return
    img = ee.Image.cat([G.terrain(), G.analysis_layers(region), class_layers(region)]).toFloat()
    G.run_tasks([ee.batch.Export.image.toAsset(
        img.clip(region), "driver_layers_asset", LAYERS_ASSET, region=region, crs=C.CRS,
        scale=SCALE, maxPixels=1e11, pyramidingPolicy={".default": "mean"})], poll_s=60)


def main():
    G.init()
    region = G.study_area()
    pred = ee.Image(MAP_ASSET).rename("agbd")
    export_layers(region)
    layers = ee.Image(LAYERS_ASSET)
    cls = layers.select(["elev_class", "slope_class", "aspect_class", "canopy_class",
                         "forest_type", "loss_period"]).toInt()

    # --- full-map zonal statistics ---------------------------------------------
    for band in ["elev_class", "slope_class", "aspect_class", "canopy_class", "forest_type", "loss_period"]:
        df = zonal(pred, cls, band, region)
        df.round(2).to_csv(C.TAB / f"05_zonal_{band}.csv", index=False)
        print(f"\n{band}\n", df.round(1).to_string(index=False))

    # --- pixel sample for plots and tests ---------------------------------------
    covars = ["elev", "slope", "northness", "eastness", "tpi", "tc2000", "lossyear", "wc_class", "decid"]
    img = ee.Image.cat([pred, layers.select(covars), cls])
    smp = img.sample(region=region, scale=SCALE, projection=C.CRS, numPixels=N_SAMPLE,
                     seed=C.SEED, tileScale=8, geometries=False)
    mp = G.fc_to_df(smp)
    mp.to_csv(C.DATA / "map_pixel_sample.csv", index=False)

    # observed GEDI footprints with the same class definitions
    ob = pd.read_csv(C.TRAINING_CSV)
    ob["loss_period"] = (ob.lossyear > 0).astype(int) + (ob.lossyear > 10) + (ob.lossyear > 18)
    ob["canopy_class"] = (ob.tc2000 >= 10).astype(int) + (ob.tc2000 >= 40) + (ob.tc2000 >= 70)

    # --- tests -------------------------------------------------------------------
    rows = []
    for name, d in [("map", mp), ("gedi_obs", ob)]:
        for v in ["elev", "slope", "tpi", "northness", "tc2000", "decid"]:
            rho, p = stats.spearmanr(d[v], d["agbd"], nan_policy="omit")
            rows.append(dict(data=name, variable=v, test="spearman", stat=rho, p=p))
        for v in ["canopy_class", "loss_period"] + (["slope_class", "forest_type"] if name == "map" else []):
            groups = [g["agbd"].values for _, g in d.groupby(v) if len(g) >= 20]
            if len(groups) > 1:
                h, p = stats.kruskal(*groups)
                rows.append(dict(data=name, variable=v, test="kruskal", stat=h, p=p))
    tests = pd.DataFrame(rows)
    tests.to_csv(C.TAB / "05_tests.csv", index=False)
    print("\n", tests.round(4).to_string(index=False))

    # --- figures -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    binned(ax[0], mp, "elev", np.arange(200, 1300, 50), "Predicted map", "#2b8a3e")
    binned(ax[0], ob, "elev", np.arange(200, 1300, 50), "GEDI observed", "#1c7ed6")
    ax[0].set(xlabel="Elevation (m)", ylabel="AGBD (Mg/ha)", title="AGBD vs elevation")
    binned(ax[1], mp, "slope", np.arange(0, 45, 2.5), "Predicted map", "#2b8a3e")
    binned(ax[1], ob, "slope", np.arange(0, 45, 2.5), "GEDI observed", "#1c7ed6")
    ax[1].set(xlabel="Slope (°)", title="AGBD vs slope")
    binned(ax[2], mp, "decid", np.linspace(0, 0.8, 17), "Predicted map", "#2b8a3e")
    binned(ax[2], ob, "decid", np.linspace(0, 0.8, 17), "GEDI observed", "#1c7ed6")
    ax[2].set(xlabel="Deciduousness (NDVI on-off)/on", title="AGBD vs deciduousness")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "05_terrain_phenology.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    boxplot(ax[0, 0], mp, "loss_period", LOSS_LABELS, "Map: disturbance history")
    boxplot(ax[0, 1], mp, "canopy_class", CANOPY_LABELS, "Map: canopy density (Hansen 2000)")
    boxplot(ax[0, 2], mp, "forest_type", FTYPE_LABELS, "Map: forest type")
    boxplot(ax[1, 0], ob, "loss_period", LOSS_LABELS, "GEDI obs: disturbance history")
    boxplot(ax[1, 1], mp, "slope_class", SLOPE_LABELS, "Map: slope class")
    boxplot(ax[1, 2], mp, "aspect_class", ASPECT_LABELS, "Map: aspect")
    fig.tight_layout()
    fig.savefig(C.FIG / "05_strata_boxplots.png", dpi=200)
    plt.close(fig)

    # recovery after loss: AGBD by year of loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for d, lab, col in [(mp, "Predicted map", "#2b8a3e"), (ob, "GEDI observed", "#1c7ed6")]:
        g = d[d.lossyear > 0].groupby("lossyear")["agbd"].agg(["median", "count"])
        g = g[g["count"] >= 10]
        ax.plot(2000 + g.index, g["median"], "o-", color=col, label=lab)
        ax.axhline(d.loc[d.lossyear == 0, "agbd"].median(), color=col, ls="--", lw=1)
    ax.set(xlabel="Year of Hansen tree-cover loss", ylabel="Median AGBD (Mg/ha)",
           title="Biomass by year of past disturbance (dashed = undisturbed median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "05_loss_year.png", dpi=200)
    plt.close(fig)
    print(f"\nFigures written to {C.FIG}")


if __name__ == "__main__":
    import sys
    if "--layers-only" in sys.argv:      # can run before step 4
        G.init()
        export_layers(G.study_area())
    else:
        main()
