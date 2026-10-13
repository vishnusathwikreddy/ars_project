"""
Shared Earth Engine building blocks: initialisation, study area, GEDI
filtering, Sentinel-2 composites, predictor stack, masks and export helpers.
"""
import time

import ee

import config as C
import features as F


# ----------------------------------------------------------------------------
# Session / assets
# ----------------------------------------------------------------------------
def init():
    if C.GEE_PROJECT.startswith("YOUR-"):
        raise SystemExit("Set GEE_PROJECT in config.py to your Earth Engine Cloud project ID.")
    ee.Initialize(project=C.GEE_PROJECT)


def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def ensure_asset_folder():
    if not asset_exists(C.ASSET_ROOT):
        ee.data.createAsset({"type": "FOLDER"}, C.ASSET_ROOT)
        print(f"Created asset folder {C.ASSET_ROOT}")


def delete_asset_if_exists(asset_id):
    if asset_exists(asset_id):
        ee.data.deleteAsset(asset_id)


def run_tasks(tasks, poll_s=30):
    """Start Export tasks and block until all finish; raise if any fails."""
    for t in tasks:
        t.start()
        print(f"  started task: {t.status()['description']}")
    pending = list(tasks)
    while pending:
        time.sleep(poll_s)
        still = []
        for t in pending:
            s = t.status()
            if s["state"] in ("COMPLETED",):
                print(f"  done: {s['description']}")
            elif s["state"] in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Task {s['description']} {s['state']}: {s.get('error_message')}")
            else:
                still.append(t)
        pending = still
        if pending:
            print(f"  waiting on {len(pending)} task(s)...")


def fc_to_df(fc):
    """Download a FeatureCollection as a pandas DataFrame (paginates)."""
    df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
    return df.drop(columns=["geo"], errors="ignore")


# ----------------------------------------------------------------------------
# Study area
# ----------------------------------------------------------------------------
def district_fc():
    if C.AOI_ASSET:
        return ee.FeatureCollection(C.AOI_ASSET)
    return (ee.FeatureCollection(C.GAUL_LEVEL2)
            .filter(ee.Filter.eq(C.GAUL_STATE_FIELD, C.STATE_NAME))
            .filter(ee.Filter.eq(C.GAUL_DISTRICT_FIELD, C.DISTRICT_NAME)))


def district():
    return district_fc().geometry()


GRID_ASSET = f"{C.ASSET_ROOT}/gedi_grid"


def study_area():
    """Selected GEDI-rich cells (script 01) if available, else the district."""
    if C.USE_SELECTED_STUDY_AREA and asset_exists(GRID_ASSET):
        return (ee.FeatureCollection(GRID_ASSET)
                .filter(ee.Filter.eq("selected", 1))
                .geometry().dissolve(maxError=10))
    return district()


# ----------------------------------------------------------------------------
# GEDI L4A
# ----------------------------------------------------------------------------
def gedi_collection(region):
    return (ee.ImageCollection(C.GEDI_L4A)
            .filterBounds(region)
            .filterDate(C.GEDI_START, C.GEDI_END))


def gedi_projection(region):
    return ee.Image(gedi_collection(region).first()).select("agbd").projection()


def gedi_quality_mask(img, f=None, stage="full"):
    """
    Footprint-level quality mask. `stage` lets script 01 report how many shots
    survive each step: 'flags' -> +'sensitivity' -> 'full'. The leaf-on season
    is applied separately, at collection level (leaf_on()).
    """
    f = f or C.GEDI_FILTER
    m = (img.select("l4_quality_flag").eq(f["l4_quality_flag"])
         .And(img.select("degrade_flag").eq(f["degrade_flag"]))
         .And(img.select("algorithm_run_flag").eq(f["algorithm_run_flag"])))
    if stage == "flags":
        return m
    m = m.And(img.select("sensitivity").gte(f["min_sensitivity"]))
    if stage == "sensitivity":
        return m
    agbd = img.select("agbd")
    m = m.And(agbd.gte(f["agbd_min"])).And(agbd.lte(f["agbd_max"]))
    if f["power_beams_only"]:
        m = m.And(img.select("beam").remap(C.POWER_BEAMS, [1] * len(C.POWER_BEAMS), 0).eq(1))
    if f["night_only"]:
        m = m.And(img.select("solar_elevation").lt(0))
    if f["max_rel_se"] is not None:
        m = m.And(img.select("agbd_se").divide(agbd.max(1)).lte(f["max_rel_se"]))
    return m


def leaf_on(col):
    """Keep monthly GEDI images inside the leaf-on season (if enabled)."""
    if not C.GEDI_FILTER["leaf_on_only"]:
        return col
    m0, m1 = C.GEDI_LEAF_ON_MONTHS
    return col.filter(ee.Filter.calendarRange(m0, m1, "month"))


def gedi_quality(region):
    return leaf_on(gedi_collection(region)).map(lambda im: im.updateMask(gedi_quality_mask(im)))


# ----------------------------------------------------------------------------
# Static layers: forest mask, disturbance, terrain
# ----------------------------------------------------------------------------
def worldcover():
    return ee.ImageCollection(C.WORLDCOVER).first().select("Map")


def tree_mask():
    return worldcover().eq(10)


def homogeneous_tree_mask():
    # reproject: otherwise the focal op runs at the requested (e.g. 25 m GEDI) scale
    return (tree_mask().focalMin(radius=C.HOMOGENEOUS_FOREST_RADIUS_M,
                                 kernelType="circle", units="meters")
            .reproject(crs=C.CRS, scale=10))


def hansen():
    """Hansen GFC; lossyear/treecover2000 are masked where 0, so unmask them."""
    h = ee.Image(C.HANSEN)
    return h.addBands(h.select(["lossyear", "treecover2000", "loss"]).unmask(0), overwrite=True)


def stable_mask():
    """1 where no Hansen loss in/after EXCLUDE_LOSS_FROM_YEAR."""
    return hansen().select("lossyear").lt(C.EXCLUDE_LOSS_FROM_YEAR - 2000)


def training_mask():
    return homogeneous_tree_mask().And(stable_mask())


def terrain():
    col = ee.ImageCollection(C.DEM).select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection())
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem).multiply(3.141592653589793 / 180)
    tpi = dem.subtract(dem.focalMean(radius=300, kernelType="circle", units="meters"))
    return ee.Image.cat([
        dem.rename("elev"),
        slope.rename("slope"),
        aspect.cos().rename("northness"),
        aspect.sin().rename("eastness"),
        tpi.rename("tpi"),
    ]).toFloat().reproject(crs=C.CRS, scale=30)


# ----------------------------------------------------------------------------
# Sentinel-2
# ----------------------------------------------------------------------------
def _nd(img, a, b, name):
    return img.normalizedDifference([a, b]).rename(name)


def add_indices(img):
    v = {b: img.select(b) for b in C.S2_BANDS}
    evi = img.expression("2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": v["B8"], "R": v["B4"], "B": v["B2"]})
    savi = img.expression("1.5*(N-R)/(N+R+0.5)", {"N": v["B8"], "R": v["B4"]})
    cire = v["B7"].divide(v["B5"]).subtract(1)
    mtci = v["B6"].subtract(v["B5"]).divide(v["B5"].subtract(v["B4"]))
    ireci = v["B7"].subtract(v["B4"]).divide(v["B5"].divide(v["B6"]))
    s2rep = img.expression("705+35*((R+RE3)/2-RE1)/(RE2-RE1)",
                           {"R": v["B4"], "RE1": v["B5"], "RE2": v["B6"], "RE3": v["B7"]})
    return img.addBands([
        _nd(img, "B8", "B4", "NDVI"),
        evi.clamp(-1, 2).rename("EVI"),
        savi.rename("SAVI"),
        _nd(img, "B8", "B11", "NDMI"),
        _nd(img, "B8", "B12", "NBR"),
        _nd(img, "B8A", "B5", "NDRE"),
        cire.clamp(-1, 20).rename("CIre"),
        mtci.clamp(-10, 20).rename("MTCI"),
        ireci.clamp(-5, 10).rename("IRECI"),
        s2rep.clamp(680, 760).rename("S2REP"),
    ])


def s2_collection(region, months):
    m0, m1 = months
    return (ee.ImageCollection(C.S2_SR)
            .filterBounds(region)
            .filterDate(C.S2_START, C.S2_END)
            .filter(ee.Filter.calendarRange(m0, m1, "month"))
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", C.MAX_SCENE_CLOUD))
            .linkCollection(ee.ImageCollection(C.S2_CLOUDSCORE), ["cs_cdf"]))


def s2_season(region, season):
    def prep(img):
        clear = img.select("cs_cdf").gte(C.CLOUDSCORE_THRESHOLD)
        return img.select(C.S2_BANDS).updateMask(clear).divide(10000)

    comp = (s2_collection(region, C.SEASONS[season]).map(prep).median()
            .setDefaultProjection(crs=C.CRS, scale=10))
    comp = add_indices(comp)
    return comp.rename(comp.bandNames().map(lambda b: ee.String(b).cat(f"_{season}")))


def glcm_texture(on):
    nir = on.select("B8_on").unitScale(0, 0.6).multiply(63).clamp(0, 63).toInt()
    tex = nir.glcmTexture(size=C.TEXTURE_SIZE)
    return tex.select([f"B8_on_{m}" for m in F.TEXTURE_METRICS],
                      [f"TEX_{m}_on" for m in F.TEXTURE_METRICS])


def predictor_stack(region):
    """All predictors in features.ALL_PREDICTORS, footprint-averaged."""
    on = s2_season(region, "on")
    off = s2_season(region, "off")
    delta = ee.Image.cat([
        on.select("NDVI_on").subtract(off.select("NDVI_off")).rename("dNDVI"),
        on.select("EVI_on").subtract(off.select("EVI_off")).rename("dEVI"),
        on.select("NDMI_on").subtract(off.select("NDMI_off")).rename("dNDMI"),
        on.select("NDRE_on").subtract(off.select("NDRE_off")).rename("dNDRE"),
    ])
    parts = [on, off, delta]
    if C.USE_TEXTURE:
        parts.append(glcm_texture(on).toFloat())
    optical = (ee.Image.cat(parts)
               .focalMean(radius=C.FOOTPRINT_RADIUS_M, kernelType="circle", units="meters")
               .reproject(crs=C.CRS, scale=10))
    return ee.Image.cat([optical, terrain()]).select(F.ALL_PREDICTORS).toFloat()


def deciduousness(region):
    """(NDVI_on - NDVI_off) / NDVI_on : forest-type proxy (teak/dry vs moister)."""
    on = s2_season(region, "on").select("NDVI_on")
    off = s2_season(region, "off").select("NDVI_off")
    return on.subtract(off).divide(on.max(0.05)).rename("decid")


def analysis_layers(region):
    h = hansen()
    return ee.Image.cat([
        h.select("treecover2000").rename("tc2000"),
        h.select("lossyear").rename("lossyear"),
        worldcover().rename("wc_class"),
        deciduousness(region),
    ]).toFloat()
