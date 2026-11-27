"""
Step 12 (local): checks requested in the review of the report.

  1. Spatial blocks: why the CV uses 193 blocks although 191 grid cells were selected
  2. Temporal mismatch: CV error of footprints acquired outside the Sentinel-2 period (2020-2024)
  3. Mapped area: UTM 43N scale distortion over the training footprints (Table 4 areas)
  4. Totals: mean AGBD x area reproduces the map totals; ESA CCI / L4B totals on the same area

Outputs
  outputs/tables/12_small_blocks.csv
  outputs/tables/12_error_by_year.csv
  outputs/tables/12_checks.json

    python 12_review_checks.py
"""
import json

import numpy as np
import pandas as pd

import config as C


def metrics(y, p):
    r = p - y
    return dict(n=int(len(y)), rmse=float(np.sqrt(np.mean(r ** 2))), bias=float(r.mean()),
                r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)), mean_obs=float(y.mean()))


def main():
    out = {}
    df = pd.read_csv(C.TRAINING_CSV)
    cfg = json.loads(C.MODEL_JSON.read_text())
    df = df.dropna(subset=cfg["features"] + ["agbd"]).reset_index(drop=True)

    # --- 1. blocks -----------------------------------------------------------------------
    # block id = floor(x/5000)*1e5 + floor(y/5000) on the footprint centre (script 02); the
    # selection grid (script 01) is the same 5 km UTM grid, but footprints are selected by their
    # 25 m GEDI pixel and then moved to the footprint centre, which can lie in a neighbouring cell.
    bx, by = np.floor(df.x_utm / C.SPATIAL_BLOCK_M), np.floor(df.y_utm / C.SPATIAL_BLOCK_M)
    assert ((bx * 100000 + by).astype(np.int64) == df.block).all()
    n_by_block = df.groupby("block").size()
    small = n_by_block[n_by_block < 10]
    rows = []
    for b in small.index:
        s = df[df.block == b]
        x0, y0 = (b // 100000) * C.SPATIAL_BLOCK_M, (b % 100000) * C.SPATIAL_BLOCK_M
        for _, r in s.iterrows():
            dx = min(r.x_utm - x0, x0 + C.SPATIAL_BLOCK_M - r.x_utm)
            dy = min(r.y_utm - y0, y0 + C.SPATIAL_BLOCK_M - r.y_utm)
            # neighbouring block across the nearest edge
            nx = bx_n = (b // 100000) + (-1 if r.x_utm - x0 < x0 + C.SPATIAL_BLOCK_M - r.x_utm else 1)
            ny = (b % 100000) + (-1 if r.y_utm - y0 < y0 + C.SPATIAL_BLOCK_M - r.y_utm else 1)
            nb = int(nx * 100000 + (b % 100000)) if dx < dy else int((b // 100000) * 100000 + ny)
            rows.append(dict(block=int(b), n_in_block=int(small[b]), dist_to_edge_m=float(min(dx, dy)),
                             neighbour_block=nb, n_in_neighbour=int(n_by_block.get(nb, 0))))
    sb = pd.DataFrame(rows)
    sb.round(1).to_csv(C.TAB / "12_small_blocks.csv", index=False)
    out["blocks"] = dict(n_blocks=int(df.block.nunique()), n_blocks_ge_26=int((n_by_block >= 26).sum()),
                         small_blocks={int(k): int(v) for k, v in small.items()},
                         footprints_in_small_blocks=int(small.sum()),
                         max_dist_to_edge_m=float(sb.dist_to_edge_m.max()) if len(sb) else None)
    print("Blocks:", out["blocks"])
    print(sb.round(1).to_string(index=False))

    # --- 2. temporal mismatch --------------------------------------------------------------
    cv = pd.read_csv(C.DATA / "cv_predictions.csv")
    cvm = pd.read_csv(C.DATA / "cv_predictions_models.csv")
    assert np.allclose(cv.agbd, df.agbd) and np.allclose(cvm.agbd, df.agbd)
    year = pd.to_datetime(df["date"]).dt.year
    period = np.where(year < 2020, "2019", np.where(year > 2024, "2025", "2020-2024"))
    rows = []
    for grp_name, mask in [("2019", period == "2019"), ("2020-2024", period == "2020-2024"),
                           ("2025", period == "2025"), ("outside 2020-2024", period != "2020-2024"),
                           ("all", np.ones(len(df), bool))]:
        r = dict(group=grp_name)
        r.update({f"RF_{k}": v for k, v in metrics(df.agbd.values[mask], cv.pred_cv.values[mask]).items()})
        r.update({f"MLP_{k}": v for k, v in metrics(df.agbd.values[mask], cvm.pred_MLP.values[mask]).items()
                  if k != "n" and k != "mean_obs"})
        rows.append(r)
    for y in sorted(year.unique()):
        m = (year == y).values
        r = dict(group=str(y))
        r.update({f"RF_{k}": v for k, v in metrics(df.agbd.values[m], cv.pred_cv.values[m]).items()})
        rows.append(r)
    ey = pd.DataFrame(rows)
    ey.round(3).to_csv(C.TAB / "12_error_by_year.csv", index=False)
    print("\nError by acquisition period/year:\n", ey.round(2).to_string(index=False))

    # --- 3. UTM scale distortion ------------------------------------------------------------
    # point scale factor of transverse Mercator, k = k0 * (1 + x^2 / (2 R^2) + x^4 / (24 R^4)),
    # x = (E - 500 km) / k0; areal distortion = k^2
    k0, R = 0.9996, 6_371_000.0
    x = (df.x_utm.values - 500_000.0) / k0
    k = k0 * (1 + x ** 2 / (2 * R ** 2) + x ** 4 / (24 * R ** 4))
    out["utm_areal_scale"] = dict(mean=float(np.mean(k ** 2)), min=float(np.min(k ** 2)),
                                  max=float(np.max(k ** 2)))
    print("\nUTM 43N areal scale factor at the footprints:", out["utm_areal_scale"])

    # --- 4. totals --------------------------------------------------------------------------
    tot = pd.read_csv(C.TAB / "09_map_totals.csv")
    tot["mean_x_area_Tg"] = tot.mean_agbd * tot.forest_area_ha / 1e6
    out["totals_check"] = tot[["model", "forest_area_ha", "mean_agbd", "total_agb_Tg", "mean_x_area_Tg"]] \
        .round(3).to_dict("records")
    ee_tot = cfg.get("map_totals", {})
    out["ee_totals"] = ee_tot
    if ee_tot:
        local_area = float(tot.forest_area_ha.iloc[0])
        out["area_ratio_local_to_ee"] = local_area / ee_tot["forest_area_ha"]
        out["local_area_divided_by_mean_utm_scale_ha"] = local_area / out["utm_areal_scale"]["mean"]
    print("\nTotals:", json.dumps(out["totals_check"], indent=1))
    print("EE totals:", ee_tot, "area ratio local/EE:", out.get("area_ratio_local_to_ee"),
          "local area / UTM scale:", out.get("local_area_divided_by_mean_utm_scale_ha"))
    (C.TAB / "12_checks.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
