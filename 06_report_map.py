"""
Report map figure: the predicted AGBD asset aggregated to 100 m with a colour bar
and the district outline (the step-4 quick-look PNG has neither).

    python 06_report_map.py   ->  outputs/figures/04_agbd_map.png, data/agbd_map_100m.npy
"""
import ee
import matplotlib.pyplot as plt
import numpy as np

import config as C
import gee_common as G


G.init()
region = G.study_area()
dist = G.district()
b = dist.bounds(maxError=100).transform(C.CRS, 1).coordinates().getInfo()[0]
xs, ys = [p[0] for p in b], [p[1] for p in b]
x0, y1 = min(xs), max(ys); S = 100
w, h = int((max(xs) - x0) / S) + 1, int((y1 - min(ys)) / S) + 1
arr = ee.data.computePixels({
    "expression": ee.Image(f"{C.ASSET_ROOT}/agbd_pred").setDefaultProjection(C.CRS, None, 25)
        .reduceResolution(ee.Reducer.mean(), maxPixels=64).unmask(-1),
    "fileFormat": "NUMPY_NDARRAY",
    "grid": {"dimensions": {"width": w, "height": h},
             "affineTransform": {"scaleX": S, "shearX": 0, "translateX": x0,
                                 "shearY": 0, "scaleY": -S, "translateY": y1},
             "crsCode": C.CRS}})
a = arr["agbd"].astype(float); a[a <= 0] = np.nan
np.save(C.DATA / "agbd_map_100m.npy", a)
dgeo = dist.transform(C.CRS, 1).getInfo()
fig, ax = plt.subplots(figsize=(9, 7.5))
ext = [x0 / 1e3, (x0 + w * S) / 1e3, (y1 - h * S) / 1e3, y1 / 1e3]
im = ax.imshow(a, extent=ext, cmap="YlGn", vmin=0, vmax=180)
polys = dgeo["coordinates"] if dgeo["type"] == "MultiPolygon" else [dgeo["coordinates"]]
for poly in polys:
    r = np.array(poly[0]) / 1e3
    ax.plot(r[:, 0], r[:, 1], color="k", lw=0.8)
fig.colorbar(im, ax=ax, shrink=0.75, label="Predicted AGBD (Mg ha$^{-1}$)", extend="max")
ax.set(xlabel="Easting (km, UTM 43N)", ylabel="Northing (km, UTM 43N)",
       title="Predicted AGBD, selected study area (aggregated to 100 m)")
ax.set_aspect("equal")
fig.savefig(C.FIG / "04_agbd_map.png", dpi=250, bbox_inches="tight")
print("map px", np.isfinite(a).sum(), "mean", np.nanmean(a))
