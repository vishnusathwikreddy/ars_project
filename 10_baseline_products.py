"""
Step 10: compare the AGBD maps of steps 4/9 with two existing biomass products,
without re-running them, over the same tree-cover pixels:

  GEDI L4B   1 km gridded mean AGBD (all land cover in the cell) from GEDI L4A
             footprints, version 2 (2019-04 to 2021-08); same source data as our
             training set, different estimator. LARSE/GEDI/GEDI04_B_002
  ESA CCI    ESA Climate Change Initiative Biomass v6, 100 m, SAR-based and
             independent of our training data (mean of the 2020-2022 maps)

Comparisons
  * area totals over the mapped tree cover (each 25 m pixel takes the value of
    the product cell it falls in)
  * cell-level agreement: our maps averaged to the CCI 100 m cells and to the
    L4B 1 km cells (only cells mostly covered by mapped tree cover)
  * point level: CCI sampled at the GEDI training footprints, against the
    out-of-fold predictions of step 7
  * GEDI L4A mean AGBD by season (leaf-on / leaf-off) and land-cover mask, to
    explain the difference between our maps and L4B (which uses all seasons
    and all land cover)

Outputs
  outputs/tables/10_product_totals.csv
  outputs/tables/10_product_agreement.csv
  outputs/tables/10_product_vs_footprints.csv
  outputs/tables/10_gedi_by_season.csv
  outputs/figures/10_product_maps.png
  outputs/figures/10_product_scatter.png

    python 10_baseline_products.py
"""
import json
import re
import warnings

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_bounds

import config as C
import gee_common as G

MAPS_DIR = C.OUT / "maps"
MAPS = ["RF", "XGB", "MLP", "ENS", "GEE_RF"]
CCI = "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB"
CCI_YEARS = [2020, 2021, 2022]
L4B = "LARSE/GEDI/GEDI04_B_002"
L4B_BANDS = ["MU", "SE", "NC", "NS", "QF", "MI", "PS"]
CCI_F = 4                       # 25 m -> 100 m
MIN_COVER = 0.8                 # cells must be >= 80% mapped tree cover
PIXEL_HA = C.OUTPUT_SCALE ** 2 / 1e4


def load_maps():
    maps = {}
    for n in MAPS:
        with rasterio.open(MAPS_DIR / f"agbd_{n}.tif") as r:
            maps[n] = r.read(1)
            transform, crs = r.transform, r.crs
    return maps, transform, crs


def compute_pixels(img, crs, x0, y0, sx, sy, w, h):
    grid = {"dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": sx, "shearX": 0, "translateX": x0,
                                "shearY": 0, "scaleY": sy, "translateY": y0}}
    # Earth Engine does not parse every EPSG code (e.g. 6933), so WKT is passed through
    grid["crsWkt" if crs.lstrip().startswith(("PROJCS", "GEOGCS")) else "crsCode"] = crs
    return ee.data.computePixels({"expression": img, "fileFormat": "NUMPY_NDARRAY", "grid": grid})


# ----------------------------------------------------------------------------
# Products
# ----------------------------------------------------------------------------
def fetch_cci(transform, shape):
    """CCI on a 100 m grid aligned with the 25 m map grid (4 x 4 map pixels per cell)."""
    h, w = -(-shape[0] // CCI_F), -(-shape[1] // CCI_F)
    col = ee.ImageCollection(CCI)
    imgs = [col.filter(ee.Filter.stringContains("system:index", f"_{y}_")).first().select("AGB")
            .rename(f"AGB_{y}") for y in CCI_YEARS]
    img = ee.Image.cat(imgs).unmask(-1).toFloat()
    a = compute_pixels(img, "EPSG:32643", transform.c, transform.f, 100, -100, w, h)
    out = {}
    for y in CCI_YEARS:
        v = a[f"AGB_{y}"].astype(np.float32)
        v[v < 0] = np.nan
        out[y] = v
    out["mean"] = np.nanmean(np.stack([out[y] for y in CCI_YEARS]), axis=0)
    return out


def fetch_l4b(transform, shape, crs):
    """L4B on its native 1 km EASE-Grid 2.0 cells covering the map extent."""
    img = ee.Image(L4B)
    proj = img.select("MU").projection().getInfo()
    if "crs" not in proj:       # returned as WKT; the last AUTHORITY is the projected CRS
        proj["crs"] = "EPSG:" + re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', proj["wkt"])[-1]
    a, _, c, _, e, f = proj["transform"]
    left, top = transform.c, transform.f
    right, bottom = left + shape[1] * transform.a, top + shape[0] * transform.e
    minx, miny, maxx, maxy = transform_bounds(crs, proj["crs"], left, bottom, right, top)
    c0, c1 = int(np.floor((minx - c) / a)), int(np.ceil((maxx - c) / a))
    r0, r1 = int(np.floor((maxy - f) / e)), int(np.ceil((miny - f) / e))
    arr = compute_pixels(img.select(L4B_BANDS).unmask(-9999).toFloat(), proj["wkt"],
                         c + c0 * a, f + r0 * e, a, e, c1 - c0, r1 - r0)
    bands = {b: arr[b].astype(np.float32) for b in L4B_BANDS}
    bands["MU"][bands["MU"] <= 0] = np.nan      # 0 = no estimate
    return bands, dict(crs=proj["crs"], x0=c + c0 * a, y0=f + r0 * e, a=a, e=e)


# ----------------------------------------------------------------------------
# Aggregation of the 25 m maps to product cells
# ----------------------------------------------------------------------------
def block_stats(a, f):
    h, w = a.shape
    H, W = -(-h // f), -(-w // f)
    pad = np.full((H * f, W * f), np.nan, np.float32)
    pad[:h, :w] = a
    b = pad.reshape(H, f, W, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3)), np.isfinite(b).sum(axis=(1, 3)) / f ** 2


def l4b_cell_index(valid, transform, crs, g):
    """Row/col of the L4B cell containing each valid 25 m pixel centre."""
    rows, cols = np.nonzero(valid)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    ex, ey = warp_transform(crs, g["crs"], xs, ys)
    return (np.floor((np.asarray(ey) - g["y0"]) / g["e"]).astype(int),
            np.floor((np.asarray(ex) - g["x0"]) / g["a"]).astype(int))


def agreement(name, ours, ref, level, n_label):
    ok = np.isfinite(ours) & np.isfinite(ref)
    d = ours[ok] - ref[ok]
    return dict(level=level, map=name, n=int(ok.sum()), unit=n_label, mean_map=float(ours[ok].mean()),
                mean_product=float(ref[ok].mean()), mean_diff=float(d.mean()),
                rmsd=float(np.sqrt(np.mean(d ** 2))), pearson_r=float(np.corrcoef(ours[ok], ref[ok])[0, 1]))


def footprint_metrics(name, y, p):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    r = p - y
    return dict(source=name, n=int(ok.sum()), r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=float(np.sqrt(np.mean(r ** 2))), bias=float(r.mean()),
                pearson_r=float(np.corrcoef(y, p)[0, 1]))


def gedi_by_season(region):
    """Shot-weighted mean L4A AGBD (same quality filter as training) by season and mask."""
    col = G.gedi_collection(region)
    proj = G.gedi_projection(region)
    m0, m1 = C.GEDI_LEAF_ON_MONTHS
    seasons = {"leaf-on": ee.Filter.Or(ee.Filter.calendarRange(m0, 12, "month"),
                                       ee.Filter.calendarRange(1, m1, "month")),
               "leaf-off": ee.Filter.calendarRange(m1 + 1, m0 - 1, "month")}
    masks = {"training mask": G.training_mask(), "all tree cover": G.tree_mask(), "all land": ee.Image(1)}
    bands = []
    for sk, sf in seasons.items():
        cm = col.filter(sf).map(lambda i: i.select("agbd").updateMask(G.gedi_quality_mask(i)))
        total, n = cm.sum(), cm.count()
        for k, mk in enumerate(masks.values()):
            bands += [total.updateMask(mk).rename(f"s_{sk}_{k}"), n.updateMask(mk).rename(f"n_{sk}_{k}")]
    st = ee.Image.cat(bands).reduceRegion(ee.Reducer.sum().unweighted(), region, crs=proj,
                                          maxPixels=1e11, tileScale=16).getInfo()
    return pd.DataFrame([dict(season=sk, mask=mn, shots=int(st[f"n_{sk}_{k}"]),
                              mean_agbd=st[f"s_{sk}_{k}"] / max(st[f"n_{sk}_{k}"], 1))
                         for sk in seasons for k, mn in enumerate(masks)])


# ----------------------------------------------------------------------------
def main():
    G.init()
    maps, transform, crs = load_maps()
    valid = np.isfinite(maps["RF"])
    print(f"{valid.sum():,} mapped tree-cover pixels ({valid.sum() * PIXEL_HA:,.0f} ha)")

    cci = fetch_cci(transform, valid.shape)
    l4b, g = fetch_l4b(transform, valid.shape, crs)
    print(f"CCI grid {cci['mean'].shape}, L4B grid {l4b['MU'].shape} ({g['crs']})")

    # --- per-pixel product values over the mapped tree cover -------------------------
    rows, cols = np.nonzero(valid)
    cci_px = {k: v[rows // CCI_F, cols // CCI_F] for k, v in cci.items()}
    lr, lc = l4b_cell_index(valid, transform, crs, g)
    inside = (lr >= 0) & (lr < l4b["MU"].shape[0]) & (lc >= 0) & (lc < l4b["MU"].shape[1])
    l4b_px = np.full(len(rows), np.nan, np.float32)
    l4b_px[inside] = l4b["MU"][lr[inside], lc[inside]]

    tot = []
    for n in MAPS:
        v = maps[n][valid]
        tot.append(dict(source=f"{n} map", coverage_pct=100.0, mean_agbd=float(np.nanmean(v)),
                        total_agb_Tg=float(np.nansum(v)) * PIXEL_HA / 1e6))
    for k in CCI_YEARS + ["mean"]:
        v = cci_px[k]
        label = f"ESA CCI {k}" if k != "mean" else f"ESA CCI {CCI_YEARS[0]}-{CCI_YEARS[-1]} mean"
        tot.append(dict(source=label, coverage_pct=100 * float(np.isfinite(v).mean()),
                        mean_agbd=float(np.nanmean(v)), total_agb_Tg=float(np.nanmean(v)) * valid.sum() * PIXEL_HA / 1e6))
    cov = np.isfinite(l4b_px)
    tot.append(dict(source="GEDI L4B", coverage_pct=100 * float(cov.mean()), mean_agbd=float(np.nanmean(l4b_px)),
                    total_agb_Tg=float(np.nanmean(l4b_px)) * valid.sum() * PIXEL_HA / 1e6))
    # our maps restricted to the pixels L4B covers, for a like-for-like mean
    for n in ["RF", "MLP"]:
        tot.append(dict(source=f"{n} map (L4B-covered pixels)", coverage_pct=100 * float(cov.mean()),
                        mean_agbd=float(np.nanmean(maps[n][valid][cov])), total_agb_Tg=np.nan))
    tot = pd.DataFrame(tot)
    tot.round(3).to_csv(C.TAB / "10_product_totals.csv", index=False)
    print(tot.round(2).to_string(index=False))

    # --- cell-level agreement -------------------------------------------------------------
    rows_out = []
    agg100 = {}
    for n in MAPS:
        m100, frac100 = block_stats(maps[n], CCI_F)
        m100[frac100 < MIN_COVER] = np.nan
        agg100[n] = m100
        rows_out.append(agreement(n, m100, cci["mean"], "ESA CCI, 100 m cells", "cells"))
    cell = pd.DataFrame({"r": lr[inside], "c": lc[inside], **{n: maps[n][valid][inside] for n in MAPS}})
    cnt = cell.groupby(["r", "c"]).size()
    frac = cnt * C.OUTPUT_SCALE ** 2 / (g["a"] * -g["e"])
    means = cell.groupby(["r", "c"])[MAPS].mean()
    keep = frac[frac >= MIN_COVER].index
    ref1k = np.array([l4b["MU"][r, c] for r, c in keep])
    agg1k = {}
    for n in MAPS:
        ours = means.loc[keep, n].values
        agg1k[n] = ours
        rows_out.append(agreement(n, ours, ref1k, "GEDI L4B, 1 km cells", "cells"))
    # sensitivity: only cells meeting the L4B precision requirement (QF = 2, mostly low AGBD)
    qf2 = np.array([l4b["QF"][r, c] == 2 for r, c in keep])
    for n in ["RF", "MLP"]:
        rows_out.append(agreement(n, np.where(qf2, agg1k[n], np.nan), ref1k,
                                  "GEDI L4B QF=2, 1 km cells", "cells"))
    agree = pd.DataFrame(rows_out)
    agree.round(3).to_csv(C.TAB / "10_product_agreement.csv", index=False)
    print(agree.round(2).to_string(index=False))

    # --- point level: CCI at the GEDI footprints vs our out-of-fold predictions -------------
    tr = pd.read_csv(C.TRAINING_CSV)
    cvp = pd.read_csv(C.DATA / "cv_predictions_models.csv")
    # cv_predictions rows are the training rows after dropna on the model features (step 7)
    feats = json.loads(C.MODEL_JSON.read_text())["features"]
    tr = tr.dropna(subset=feats + ["agbd"]).reset_index(drop=True)
    assert len(tr) == len(cvp) and np.allclose(tr.agbd.values, cvp.agbd.values)
    pc = ((tr.x_utm - transform.c) / (C.OUTPUT_SCALE * CCI_F)).astype(int).values
    pr = ((tr.y_utm - transform.f) / (-C.OUTPUT_SCALE * CCI_F)).astype(int).values
    inb = (pr >= 0) & (pr < cci["mean"].shape[0]) & (pc >= 0) & (pc < cci["mean"].shape[1])
    cci_fp = np.full(len(tr), np.nan)
    cci_fp[inb] = cci["mean"][pr[inb], pc[inb]]
    y = tr.agbd.values
    fp = [footprint_metrics("ESA CCI 2020-2022 mean (no training)", y, cci_fp)]
    fp += [footprint_metrics(f"{n} (spatial CV, out-of-fold)", y, cvp[f"pred_{n}"].values)
           for n in ["LIN", "RF", "XGB", "MLP", "ENS"] if f"pred_{n}" in cvp]
    fp = pd.DataFrame(fp)
    fp.round(3).to_csv(C.TAB / "10_product_vs_footprints.csv", index=False)
    print(fp.round(3).to_string(index=False))

    season = gedi_by_season(G.study_area())
    season.round(2).to_csv(C.TAB / "10_gedi_by_season.csv", index=False)
    print(season.round(1).to_string(index=False))

    plot_maps(maps, cci["mean"], l4b, g, transform, crs, valid)
    plot_scatter(agg100, cci["mean"], agg1k, ref1k)
    print("Done.")


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_maps(maps, cci_mean, l4b, g, transform, crs, valid):
    rf100, frac = block_stats(maps["RF"], CCI_F)
    rf100[frac == 0] = np.nan
    cci_m = cci_mean.copy()
    cci_m[frac == 0] = np.nan
    # L4B shown on the 100 m grid (each cell takes its 1 km value) inside the mapped area
    h, w = rf100.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xs = transform.c + (cc.ravel() + 0.5) * 100
    ys = transform.f - (rr.ravel() + 0.5) * 100
    ex, ey = warp_transform(crs, g["crs"], xs, ys)
    lr = np.floor((np.asarray(ey) - g["y0"]) / g["e"]).astype(int)
    lc = np.floor((np.asarray(ex) - g["x0"]) / g["a"]).astype(int)
    ok = (lr >= 0) & (lr < l4b["MU"].shape[0]) & (lc >= 0) & (lc < l4b["MU"].shape[1])
    l4b100 = np.full(h * w, np.nan, np.float32)
    l4b100[ok] = l4b["MU"][lr[ok], lc[ok]]
    l4b100 = l4b100.reshape(h, w)
    l4b100[frac == 0] = np.nan

    ext = [transform.c / 1e3, transform.c / 1e3 + w * 0.1, transform.f / 1e3 - h * 0.1, transform.f / 1e3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout="constrained")
    for ax, (title, a) in zip(axes, [("This study (RF), 100 m", rf100),
                                     (f"ESA CCI v6 {CCI_YEARS[0]}-{CCI_YEARS[-1]}, 100 m", cci_m),
                                     ("GEDI L4B, 1 km", l4b100)]):
        im = ax.imshow(a, extent=ext, cmap="YlGn", vmin=0, vmax=180)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, location="bottom", shrink=0.5, aspect=40, label="AGBD (Mg/ha)", extend="max")
    fig.savefig(C.FIG / "10_product_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(agg100, cci_mean, agg1k, ref1k):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    for ax, n in zip(axes[:2], ["RF", "MLP"]):
        ok = np.isfinite(agg100[n]) & np.isfinite(cci_mean)
        ax.hexbin(cci_mean[ok], agg100[n][ok], gridsize=50, mincnt=1, bins="log",
                  extent=(0, 250, 0, 250), cmap="viridis")
        ax.set(title=f"{n} vs ESA CCI (100 m), r={np.corrcoef(cci_mean[ok], agg100[n][ok])[0, 1]:.2f}",
               xlabel="ESA CCI AGBD (Mg/ha)", ylabel=f"{n} map AGBD (Mg/ha)")
    for ax, n in zip(axes[2:], ["RF", "MLP"]):
        ok = np.isfinite(agg1k[n]) & np.isfinite(ref1k)
        ax.scatter(ref1k[ok], agg1k[n][ok], s=6, alpha=0.5, color="#1c7ed6")
        ax.set(title=f"{n} vs GEDI L4B (1 km), r={np.corrcoef(ref1k[ok], agg1k[n][ok])[0, 1]:.2f}",
               xlabel="GEDI L4B AGBD (Mg/ha)", ylabel=f"{n} map AGBD (Mg/ha)", xlim=(0, 250), ylim=(0, 250))
    for ax in axes:
        ax.plot([0, 250], [0, 250], "r--", lw=1)
    fig.savefig(C.FIG / "10_product_scatter.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
