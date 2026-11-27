"""
Step 9: wall-to-wall AGBD maps from the RF, XGBoost and MLP models of step 7,
predicted locally (GPU) from the predictor stack exported in step 8, and a
comparison of the maps: totals, value distributions, pairwise agreement and
how each model reproduces the step-5 driver patterns.

The Earth Engine RF map of step 4 is downloaded on the same grid as a reference.

Outputs
  outputs/maps/agbd_{RF,XGB,MLP,ENS,GEE_RF}.tif     25 m GeoTIFFs (EPSG:32643)
  outputs/tables/09_map_totals.csv
  outputs/tables/09_map_agreement.csv
  outputs/tables/09_map_distribution.csv
  outputs/tables/09_zonal_by_model.csv
  outputs/figures/09_maps_models.png
  outputs/figures/09_map_histograms.png
  outputs/figures/09_drivers_by_model.png

    CUDA_VISIBLE_DEVICES=0 python 09_map_compare.py
"""
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

import ee
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import xgboost as xgb
from rasterio.transform import Affine

import config as C
import gee_common as G

STACK_ASSET = f"{C.ASSET_ROOT}/predictor_stack"
MAP_ASSET = f"{C.ASSET_ROOT}/agbd_pred"
LAYERS_ASSET = f"{C.ASSET_ROOT}/driver_layers"
MODELS_DIR = C.OUT / "models"
MAPS_DIR = C.OUT / "maps"
MAPS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TILE = 384                      # 384 x 384 x ~52 bands x 4 B ~ 30 MB per request
STRATA = ["slope_class", "forest_type", "elev_class", "loss_period", "canopy_class", "aspect_class"]
PIXEL_HA = C.OUTPUT_SCALE ** 2 / 1e4
MODELS = ["RF", "XGB", "MLP"]
COLORS = {"RF": "#2b8a3e", "XGB": "#e8590c", "MLP": "#1c7ed6", "ENS": "#495057", "GEE_RF": "#868e96"}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def load_models():
    spec = json.loads((MODELS_DIR / "final_models.json").read_text())
    import importlib
    cm = importlib.import_module("07_compare_models")      # MLP / Scaler definitions
    rf = joblib.load(MODELS_DIR / "rf.joblib")
    xg = xgb.Booster()
    xg.load_model(MODELS_DIR / "xgb.json")
    xg.set_param({"device": DEVICE})
    ck = torch.load(MODELS_DIR / "mlp.pt", map_location=DEVICE)
    p = ck["params"]
    net = cm.MLP(ck["d_in"], p["width"], p["depth"], p["dropout"]).to(DEVICE)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    sc = cm.Scaler()
    for k, v in ck["scaler"].items():
        setattr(sc, k, np.array(v, dtype=np.float32))
    return spec["features"], {
        "RF": lambda X: rf.predict(X),
        "XGB": lambda X: xg.inplace_predict(X),
        "MLP": lambda X: cm.mlp_predict((net, sc), X),
    }


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def asset_grid(asset_id):
    band = ee.data.getAsset(asset_id)["bands"][0]
    g = band["grid"]
    t = g["affineTransform"]
    return (t["translateX"], t["translateY"], t["scaleX"], t["scaleY"],
            g["dimensions"]["width"], g["dimensions"]["height"], g["crsCode"])


def fetch_tile(expr, x0, y0, sx, sy, crs, col, row, w, h):
    grid = {"dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": sx, "shearX": 0, "translateX": x0 + col * sx,
                                "shearY": 0, "scaleY": sy, "translateY": y0 + row * sy},
            "crsCode": crs}
    for attempt in range(5):
        try:
            return ee.data.computePixels({"expression": expr, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        except ee.EEException as e:     # transient quota / timeout errors
            if attempt == 4:
                raise
            print(f"    retry tile ({col},{row}): {e}")


def build_maps(feats, predict):
    x0, y0, sx, sy, W, H, crs = asset_grid(STACK_ASSET)
    print(f"Grid {W} x {H} px at {sx} m ({crs})")
    stack = ee.Image(STACK_ASSET)
    # computePixels writes 0 (not the mask) outside the data, so carry an explicit validity band
    expr = ee.Image.cat([
        stack.select(feats).unmask(0),
        stack.select(feats).mask().reduce(ee.Reducer.min()).rename("valid").unmask(0),
        ee.Image(MAP_ASSET).rename("GEE_RF").unmask(-1),
        ee.Image(LAYERS_ASSET).select(STRATA).unmask(-1),
    ]).toFloat()
    out = {n: np.full((H, W), np.nan, np.float32) for n in MODELS + ["GEE_RF"]}
    strata = {s: np.full((H, W), -1, np.int16) for s in STRATA}
    tiles = [(c, r) for r in range(0, H, TILE) for c in range(0, W, TILE)]

    def job(t):
        c, r = t
        w, h = min(TILE, W - c), min(TILE, H - r)
        return c, r, fetch_tile(expr, x0, y0, sx, sy, crs, c, r, w, h)

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for c, r, a in pool.map(job, tiles):
            done += 1
            valid = a["valid"] > 0
            sl = (slice(r, r + valid.shape[0]), slice(c, c + valid.shape[1]))
            if valid.any():
                X = np.stack([a[f][valid] for f in feats], axis=1).astype(np.float32)
                for n in MODELS:
                    tile = np.full(valid.shape, np.nan, np.float32)
                    tile[valid] = predict[n](X)
                    out[n][sl] = tile
                g = a["GEE_RF"].astype(np.float32)
                g[(g < 0) | ~valid] = np.nan
                out["GEE_RF"][sl] = g
                for s in STRATA:
                    strata[s][sl] = np.where(valid, a[s], -1).astype(np.int16)
            if done % 20 == 0 or done == len(tiles):
                print(f"  {done}/{len(tiles)} tiles", flush=True)
    out["ENS"] = np.nanmean([out[n] for n in MODELS], axis=0).astype(np.float32)
    transform = Affine(sx, 0, x0, 0, sy, y0)
    return out, strata, transform, crs


def write_tif(path, arr, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate", predictor=3, tiled=True) as dst:
        dst.write(arr, 1)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def summarise(maps, strata):
    valid = np.isfinite(maps["RF"])
    vals = {n: m[valid] for n, m in maps.items()}
    n_px = int(valid.sum())

    totals = pd.DataFrame([dict(model=n, forest_area_ha=n_px * PIXEL_HA, mean_agbd=float(np.nanmean(v)),
                                total_agb_Tg=float(np.nansum(v)) * PIXEL_HA / 1e6,
                                total_carbon_Tg=0.47 * float(np.nansum(v)) * PIXEL_HA / 1e6)
                           for n, v in vals.items()])
    q = [1, 5, 25, 50, 75, 95, 99]
    dist = pd.DataFrame([dict(model=n, **dict(zip([f"p{x}" for x in q], np.nanpercentile(v, q))),
                              share_gt_150=float(np.nanmean(v > 150)) * 100) for n, v in vals.items()])
    rows = []
    names = list(vals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ok = np.isfinite(vals[a]) & np.isfinite(vals[b])
            d = vals[b][ok] - vals[a][ok]
            rows.append(dict(map_a=a, map_b=b, mean_diff=float(d.mean()), rmsd=float(np.sqrt(np.mean(d ** 2))),
                             mad=float(np.mean(np.abs(d))), pearson_r=float(np.corrcoef(vals[a][ok], vals[b][ok])[0, 1])))
    agree = pd.DataFrame(rows)

    zrows = []
    for s in STRATA:
        cls = strata[s][valid]
        for k in np.unique(cls[cls >= 0]):
            sel = cls == k
            zrows += [dict(stratum=s, cls=int(k), share_pct=100 * float(sel.mean()), model=n,
                           mean_agbd=float(np.nanmean(v[sel]))) for n, v in vals.items()]
    zonal = pd.DataFrame(zrows)
    return totals, dist, agree, zonal


def block_mean(a, f=4):
    h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    with warnings.catch_warnings():         # all-NaN blocks -> NaN, silently
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))


def plot_maps(maps, transform):
    f = 4               # 25 m -> 100 m for display
    agg = {n: block_mean(maps[n], f) for n in ["RF", "XGB", "MLP"]}
    x0, y0 = transform.c / 1e3, transform.f / 1e3
    h, w = agg["RF"].shape
    ext = [x0, x0 + w * 0.1, y0 - h * 0.1, y0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 10.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12)
    for ax, n in zip(axes[0], ["RF", "XGB", "MLP"]):
        im = ax.imshow(agg[n], extent=ext, cmap="YlGn", vmin=0, vmax=180)
        ax.set_title(f"{n}: mean {np.nanmean(maps[n]):.1f} Mg/ha")
    fig.colorbar(im, ax=axes[0], location="bottom", shrink=0.5, aspect=40, label="AGBD (Mg/ha)", extend="max")
    for ax, n in zip(axes[1][:2], ["XGB", "MLP"]):
        d = agg[n] - agg["RF"]
        im2 = ax.imshow(d, extent=ext, cmap="RdBu", vmin=-40, vmax=40)
        ax.set_title(f"{n} − RF")
    fig.colorbar(im2, ax=axes[1][:2], location="bottom", shrink=0.7, aspect=30, label="Difference (Mg/ha)", extend="both")
    sd = np.nanstd(np.stack([agg[n] for n in ["RF", "XGB", "MLP"]]), axis=0)
    im3 = axes[1][2].imshow(sd, extent=ext, cmap="magma_r", vmin=0, vmax=30)
    axes[1][2].set_title("Between-model SD")
    fig.colorbar(im3, ax=axes[1][2], location="bottom", shrink=0.9, aspect=15, label="SD (Mg/ha)", extend="max")
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(C.FIG / "09_maps_models.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_hist(maps, gedi):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 300, 61)
    ax.hist(gedi, bins=bins, density=True, color="#ced4da", label="GEDI footprints (training)")
    for n in ["RF", "XGB", "MLP", "GEE_RF"]:
        v = maps[n][np.isfinite(maps[n])]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=COLORS[n], label=f"{n} map")
    ax.set(xlabel="AGBD (Mg/ha)", ylabel="Density", title="Distribution of mapped AGBD vs GEDI reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "09_map_histograms.png", dpi=200)
    plt.close(fig)


def plot_drivers(zonal):
    labels = {"slope_class": ("Slope class", ["0-5°", "5-10°", "10-20°", "20-30°", ">30°"]),
              "forest_type": ("Deciduousness tercile", ["Low", "Medium", "High"]),
              "loss_period": ("Disturbance", ["None", "2001-10", "2011-18", "2019-25"])}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (s, (title, ticks)) in zip(axes, labels.items()):
        z = zonal[zonal.stratum == s]
        for n in ["RF", "XGB", "MLP"]:
            g = z[z.model == n].sort_values("cls")
            ax.plot(range(len(g)), g.mean_agbd, "o-", color=COLORS[n], label=n)
        ax.set_xticks(range(len(ticks)), ticks[:len(g)], rotation=20)
        ax.set(title=title, ylabel="Mean predicted AGBD (Mg/ha)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "09_drivers_by_model.png", dpi=200)
    plt.close(fig)


def main():
    feats, predict = load_models()
    G.init()
    maps, strata, transform, crs = build_maps(feats, predict)
    for n, m in maps.items():
        write_tif(MAPS_DIR / f"agbd_{n}.tif", m, transform, crs)
    totals, dist, agree, zonal = summarise(maps, strata)
    totals.round(3).to_csv(C.TAB / "09_map_totals.csv", index=False)
    dist.round(2).to_csv(C.TAB / "09_map_distribution.csv", index=False)
    agree.round(3).to_csv(C.TAB / "09_map_agreement.csv", index=False)
    zonal.round(2).to_csv(C.TAB / "09_zonal_by_model.csv", index=False)
    print(totals.round(2).to_string(index=False))
    print(dist.round(1).to_string(index=False))
    print(agree.round(2).to_string(index=False))
    plot_maps(maps, transform)
    plot_hist(maps, pd.read_csv(C.TRAINING_CSV)["agbd"].values)
    plot_drivers(zonal)
    print(f"Maps in {MAPS_DIR}, figures in {C.FIG}")


if __name__ == "__main__":
    main()
