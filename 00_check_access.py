"""
Step 0: verify Earth Engine access and that every dataset resolves for Betul.

    python 00_check_access.py
"""
import ee

import config as C
import gee_common as G


def main():
    G.init()
    print(f"Earth Engine initialised with project '{C.GEE_PROJECT}'")

    dist = G.district_fc()
    n = dist.size().getInfo()
    if n == 0:
        raise SystemExit(f"No district named {C.DISTRICT_NAME} in {C.GAUL_LEVEL2}; set AOI_ASSET.")
    area = dist.geometry().area(maxError=100).divide(1e6).getInfo()
    print(f"District boundary: {n} feature(s), {area:,.0f} km2")

    geom = dist.geometry()
    gedi = G.gedi_collection(geom)
    first, last = (ee.Date(gedi.aggregate_min("system:time_start")).format("YYYY-MM-dd"),
                   ee.Date(gedi.aggregate_max("system:time_start")).format("YYYY-MM-dd"))
    print(f"GEDI L4A ({C.GEDI_L4A}): {gedi.size().getInfo()} monthly images, "
          f"{first.getInfo()} .. {last.getInfo()}")
    print(f"  bands: {ee.Image(gedi.first()).bandNames().getInfo()[:12]} ...")

    for cand in ["LARSE/GEDI/GEDI04_A_003_MONTHLY", "LARSE/GEDI/GEDI04_A_003"]:
        if G.asset_exists(cand):
            print(f"  NOTE: GEDI L4A V3 asset found: {cand}. Consider setting GEDI_L4A to it.")

    for season in C.SEASONS:
        n = G.s2_collection(geom, C.SEASONS[season]).size().getInfo()
        print(f"Sentinel-2 L2A season '{season}' {C.SEASONS[season]}: {n} scenes")

    wc = G.worldcover()
    tree_km2 = (G.tree_mask().multiply(ee.Image.pixelArea())
                .reduceRegion(ee.Reducer.sum(), geom, scale=30, maxPixels=1e10, tileScale=4)
                .getNumber("Map").divide(1e6).getInfo())
    print(f"WorldCover ({C.WORLDCOVER}): tree cover {tree_km2:,.0f} km2 "
          f"({100 * tree_km2 / area:.1f}% of district)")

    dem_stats = G.terrain().select("elev").reduceRegion(
        ee.Reducer.minMax(), geom, scale=90, maxPixels=1e10).getInfo()
    print(f"Copernicus DEM: elevation {dem_stats['elev_min']:.0f}-{dem_stats['elev_max']:.0f} m")

    loss_km2 = (G.hansen().select("loss").multiply(ee.Image.pixelArea())
                .reduceRegion(ee.Reducer.sum(), geom, scale=30, maxPixels=1e10, tileScale=4)
                .getNumber("loss").divide(1e6).getInfo())
    print(f"Hansen GFC ({C.HANSEN}): tree-cover loss 2001-{C.HANSEN_LAST_YEAR}: {loss_km2:,.1f} km2")

    G.ensure_asset_folder()
    print(f"Asset folder OK: {C.ASSET_ROOT}")
    print("\nAll datasets accessible.")


if __name__ == "__main__":
    main()
