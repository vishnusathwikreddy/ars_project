"""
Step 1: GEDI L4A inventory over Betul and selection of the final study area.

Outputs
  outputs/tables/gedi_filter_cascade.csv   shots remaining after each filter
  outputs/tables/gedi_shots_by_year.csv
  outputs/tables/gedi_shots_by_month.csv   (quality shots, before leaf-on filter)
  outputs/tables/gedi_grid.csv             per-cell shot count / forest fraction
  outputs/figures/01_gedi_grid.png
  asset  <ASSET_ROOT>/gedi_grid            grid with `selected` flag (used by later steps)

    python 01_gedi_inventory.py
"""
import ee
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PolyCollection

import config as C
import gee_common as G


def count_image(col):
    """Per-pixel number of unmasked GEDI shots (0 where none)."""
    return col.select("agbd").count().unmask(0)


def sum_counts(img, geom, proj):
    # unweighted: count each GEDI pixel once instead of by its overlap with the boundary
    return img.reduceRegion(ee.Reducer.sum().unweighted(), geom, crs=proj, maxPixels=1e11, tileScale=8)


def main():
    G.init()
    G.ensure_asset_folder()
    geom = G.district()
    proj = G.gedi_projection(geom)
    raw = G.gedi_collection(geom)
    static = G.training_mask()

    # --- filter cascade ------------------------------------------------------
    # (band name, report label, collection); EE band names cannot contain '+' or '/'
    q = G.gedi_quality(geom)
    lossyear = G.hansen().select("lossyear")
    past_loss = lossyear.gt(0).And(lossyear.lt(C.EXCLUDE_LOSS_FROM_YEAR - 2000))
    stages = [
        ("s0", "all shots", raw),
        ("s1", "+ quality/degrade/algorithm flags",
         raw.map(lambda i: i.updateMask(G.gedi_quality_mask(i, stage="flags")))),
        ("s2", f"+ sensitivity >= {C.GEDI_FILTER['min_sensitivity']}",
         raw.map(lambda i: i.updateMask(G.gedi_quality_mask(i, stage="sensitivity")))),
        ("s3", f"+ leaf-on months {C.GEDI_LEAF_ON_MONTHS}",
         G.leaf_on(raw).map(lambda i: i.updateMask(G.gedi_quality_mask(i, stage="sensitivity")))),
        ("s4", "+ AGBD range / beam / SE limits", q),
        ("s5", "+ in WorldCover tree cover", q.map(lambda i: i.updateMask(G.tree_mask()))),
        ("s6", "+ homogeneous tree cover", q.map(lambda i: i.updateMask(G.homogeneous_tree_mask()))),
        ("s7", f"+ no Hansen loss since {C.EXCLUDE_LOSS_FROM_YEAR} (training set)",
         q.map(lambda i: i.updateMask(static))),
        ("x1", f"   of which on pixels with loss 2001-{C.EXCLUDE_LOSS_FROM_YEAR - 1}",
         q.map(lambda i: i.updateMask(static.And(past_loss)))),
    ]
    imgs = [count_image(c).rename(k) for k, _, c in stages]
    cascade = sum_counts(ee.Image.cat(imgs), geom, proj).getInfo()
    df = pd.DataFrame({"stage": [lab for _, lab, _ in stages],
                       "shots": [int(cascade[k]) for k, _, _ in stages]})
    df.to_csv(C.TAB / "gedi_filter_cascade.csv", index=False)
    print("GEDI filter cascade (district):\n", df.to_string(index=False))

    # --- by year (final filter) and by month (before leaf-on filter) ----------
    final = q.map(lambda i: i.updateMask(static))
    y0, y1 = int(C.GEDI_START[:4]), int(C.GEDI_END[:4])
    yimg = ee.Image.cat([count_image(final.filter(ee.Filter.calendarRange(y, y, "year"))).rename(f"y{y}")
                         for y in range(y0, y1 + 1)])
    by_year = sum_counts(yimg, geom, proj).getInfo()
    pd.DataFrame({"year": [k[1:] for k in by_year], "shots": [int(v) for v in by_year.values()]}) \
        .to_csv(C.TAB / "gedi_shots_by_year.csv", index=False)
    print("Final shots by year:", by_year)

    sens = stages[2][2]
    mimg = ee.Image.cat([count_image(sens.filter(ee.Filter.calendarRange(m, m, "month"))).rename(f"m{m:02d}")
                         for m in range(1, 13)])
    by_month = sum_counts(mimg, geom, proj).getInfo()
    pd.DataFrame({"month": list(by_month), "shots": [int(v) for v in by_month.values()]}) \
        .to_csv(C.TAB / "gedi_shots_by_month.csv", index=False)

    # --- density grid and study-area selection -------------------------------
    grid = (geom.coveringGrid(ee.Projection(C.CRS).atScale(C.GRID_SIZE_M))
            .map(lambda f: f.setGeometry(f.geometry().intersection(geom, maxError=10)))
            .map(lambda f: f.set("area_km2", f.geometry().area(maxError=10).divide(1e6)))
            .filter(ee.Filter.gt("area_km2", 0.01)))     # drop corner-touching slivers
    shots = count_image(final).rename("shots")
    grid = shots.reduceRegions(grid, ee.Reducer.sum().unweighted().setOutputs(["shots"]),
                               crs=proj, tileScale=8)
    grid = G.tree_mask().rename("forest_frac").reduceRegions(
        grid, ee.Reducer.mean().setOutputs(["forest_frac"]), scale=30, tileScale=8)
    def flag(f):
        # border slivers can contain no pixel centre -> null statistics
        shots_n = ee.Number(ee.Algorithms.If(f.get("shots"), f.get("shots"), 0))
        forest = ee.Number(ee.Algorithms.If(f.get("forest_frac"), f.get("forest_frac"), 0))
        return f.set({
            "shots": shots_n, "forest_frac": forest,
            "selected": shots_n.gte(C.MIN_SHOTS_PER_CELL).And(forest.gte(C.MIN_FOREST_FRACTION)),
        })
    grid = grid.map(flag)
    G.delete_asset_if_exists(G.GRID_ASSET)
    G.run_tasks([ee.batch.Export.table.toAsset(grid, "gedi_grid", G.GRID_ASSET)])

    gdf = ee.data.computeFeatures({"expression": ee.FeatureCollection(G.GRID_ASSET),
                                   "fileFormat": "PANDAS_DATAFRAME"})
    gdf.drop(columns=["geo"]).to_csv(C.TAB / "gedi_grid.csv", index=False)
    sel = gdf[gdf.selected == 1]
    print(f"\nGrid: {len(gdf)} cells, {len(sel)} selected "
          f"({sel.area_km2.sum():,.0f} of {gdf.area_km2.sum():,.0f} km2), "
          f"{int(sel.shots.sum()):,} training shots inside selection")

    plot_grid(gdf)


def _rings(geo):
    t, c = geo["type"], geo["coordinates"]
    polys = c if t == "MultiPolygon" else [c] if t == "Polygon" else []
    if t == "GeometryCollection":
        return [r for g in geo["geometries"] for r in _rings(g)]
    return [p[0] for p in polys]


def plot_grid(gdf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, col, cmap, title in [
        (axes[0], "shots", "viridis", "Quality-filtered GEDI L4A shots per cell"),
        (axes[1], "forest_frac", "Greens", "WorldCover tree-cover fraction"),
    ]:
        verts, vals, edge = [], [], []
        for _, r in gdf.iterrows():
            for ring in _rings(r["geo"]):
                verts.append(ring)
                vals.append(r[col])
                edge.append("red" if r["selected"] == 1 else "none")
        pc = PolyCollection(verts, array=pd.Series(vals).values, cmap=cmap,
                            edgecolors=edge, linewidths=0.8)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(pc, ax=ax, shrink=0.8)
    fig.suptitle("Betul district: GEDI density (red outline = selected study area cells)")
    fig.tight_layout()
    fig.savefig(C.FIG / "01_gedi_grid.png", dpi=200)
    print(f"Saved {C.FIG / '01_gedi_grid.png'}")


if __name__ == "__main__":
    main()
