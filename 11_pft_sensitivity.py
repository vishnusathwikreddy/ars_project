"""
Step 11 (local): sensitivity of the Random Forest results to the GEDI L4A plant
functional type (PFT) of the training footprints.

All training footprints lie in homogeneous WorldCover tree cover, but the L4A
algorithm assigned only ~59 % of them the deciduous broadleaf tree PFT (4); the
rest got grass (6) or crop (7, 8) PFTs from coarse MODIS land cover and were
therefore processed with other allometric models. This script refits the step-3
Random Forest on deciduous-broadleaf footprints only (same predictors, same
hyper-parameters, same spatial folds) and compares it with the full model.

Within each outer spatial fold (identical to step 3):
  FULL : RF trained on all training-fold footprints
  DBT  : RF trained on the deciduous-broadleaf training-fold footprints only
Both predict every footprint of the held-out blocks, so they can be compared
on the same test footprints.

Outputs
  outputs/tables/11_pft_summary.csv          per-PFT counts, GEDI AGBD, FULL-model CV prediction
  outputs/tables/11_accuracy.csv             FULL vs DBT on DBT test footprints (and FULL on all)
  outputs/tables/11_accuracy_by_range.csv    error by GEDI AGBD range on DBT test footprints
  outputs/tables/11_level_shift.csv          mean prediction over all footprints (proxy for the map mean)
  outputs/tables/11_slope_association.csv    Spearman rho of AGBD with slope; means by slope class
  outputs/tables/11_slope_class_pft.csv      PFT shares by slope class
  outputs/tables/11_importance_dbt.csv       permutation importance of the DBT model (step-3 hold-out fold)
  data/cv_predictions_pft.csv

    python 11_pft_sensitivity.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import config as C

DBT = 4
PFT_NAMES = {2: "Evergreen broadleaf trees", 4: "Deciduous broadleaf trees", 6: "Grasses",
             7: "Cereal crops", 8: "Broadleaf crops", 9: "Urban/built-up"}
SLOPE_BINS = [0, 5, 10, 20, 30, 90]
RANGE_BINS = [0, 25, 50, 75, 100, 150, 200, 300, np.inf]


def metrics(y, p):
    r = p - y
    rmse = float(np.sqrt(np.mean(r ** 2)))
    return dict(n=int(len(y)), r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=rmse, rrmse_pct=100 * rmse / float(y.mean()), mae=float(np.mean(np.abs(r))),
                bias=float(r.mean()), slope=float(np.polyfit(y, p, 1)[0]),
                mean_obs=float(y.mean()), mean_pred=float(p.mean()))


def main():
    cfg = json.loads(C.MODEL_JSON.read_text())
    feats, params = cfg["features"], dict(cfg["sklearn_params"])
    df = pd.read_csv(C.TRAINING_CSV).dropna(subset=feats + ["agbd"]).reset_index(drop=True)
    X, y, groups = df[feats].values, df["agbd"].values, df["block"].values
    is_dbt = (df["pft_class"] == DBT).values
    print(f"{len(df):,} footprints, {is_dbt.sum():,} deciduous broadleaf ({is_dbt.mean():.1%}); "
          f"RF params {params}")

    # --- per-PFT summary (uses the step-3 CV predictions) -------------------------------
    cv = pd.read_csv(C.DATA / "cv_predictions.csv")
    assert np.allclose(cv["agbd"].values, y), "cv_predictions.csv is not aligned with the training table"
    rows = []
    for pft, g in df.assign(pred=cv["pred_cv"].values).groupby("pft_class"):
        rows.append(dict(pft_class=pft, name=PFT_NAMES.get(pft, str(pft)), n=len(g),
                         share_pct=100 * len(g) / len(df), mean_agbd=g.agbd.mean(),
                         median_agbd=g.agbd.median(), mean_se=g.agbd_se.mean(),
                         share_gt200_pct=100 * (g.agbd > 200).mean(),
                         mean_pred_full_cv=g.pred.mean(), mean_slope_deg=g.slope.mean()))
    summ = pd.DataFrame(rows)
    n200 = (df.agbd > 200).sum()
    summ["share_of_all_gt200_pct"] = [100 * ((df.pft_class == p) & (df.agbd > 200)).sum() / n200
                                      for p in summ.pft_class]
    summ.round(2).to_csv(C.TAB / "11_pft_summary.csv", index=False)
    print("\nPer-PFT summary:\n", summ.round(1).to_string(index=False))

    # --- spatial CV: FULL vs DBT models on identical outer folds -----------------------
    pred_full, pred_dbt = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups)):
        full = RandomForestRegressor(**params).fit(X[tr], y[tr])
        tr_dbt = tr[is_dbt[tr]]
        dbt = RandomForestRegressor(**params).fit(X[tr_dbt], y[tr_dbt])
        pred_full[te], pred_dbt[te], fold_id[te] = full.predict(X[te]), dbt.predict(X[te]), f
        print(f"  fold {f}: train {len(tr):,} (DBT {len(tr_dbt):,}), test {len(te):,}", flush=True)
    print("FULL reproduces step 3:", round(metrics(y, pred_full)["r2"], 4), "vs", round(cfg["cv_spatial"]["r2"], 4))
    pd.DataFrame({"block": groups, "fold": fold_id, "pft_class": df.pft_class, "agbd": y,
                  "pred_full": pred_full, "pred_dbt": pred_dbt}).to_csv(C.DATA / "cv_predictions_pft.csv",
                                                                       index=False)

    acc = [dict(model="FULL", evaluated_on="all footprints", **metrics(y, pred_full)),
           dict(model="FULL", evaluated_on="DBT footprints", **metrics(y[is_dbt], pred_full[is_dbt])),
           dict(model="DBT", evaluated_on="DBT footprints", **metrics(y[is_dbt], pred_dbt[is_dbt])),
           dict(model="FULL", evaluated_on="non-DBT footprints", **metrics(y[~is_dbt], pred_full[~is_dbt])),
           dict(model="DBT", evaluated_on="non-DBT footprints", **metrics(y[~is_dbt], pred_dbt[~is_dbt]))]
    acc = pd.DataFrame(acc)
    acc.round(3).to_csv(C.TAB / "11_accuracy.csv", index=False)
    print("\nAccuracy:\n", acc.round(3).to_string(index=False))

    rng_rows = []
    for name, p in (("FULL", pred_full), ("DBT", pred_dbt)):
        for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
            s = is_dbt & (y >= lo) & (y < hi)
            m = metrics(y[s], p[s])
            rng_rows.append(dict(model=name, agbd_range=f"{lo}-{hi}" if np.isfinite(hi) else f">{lo}",
                                 n=m["n"], rmse=m["rmse"], bias=m["bias"]))
        s = is_dbt & (y > 200)
        rng_rows.append(dict(model=name, agbd_range=">200", n=int(s.sum()),
                             rmse=metrics(y[s], p[s])["rmse"], bias=metrics(y[s], p[s])["bias"]))
    rng = pd.DataFrame(rng_rows)
    rng.round(2).to_csv(C.TAB / "11_accuracy_by_range.csv", index=False)
    print("\nError by AGBD range (DBT test footprints):\n", rng.round(1).to_string(index=False))

    # --- level shift: mean prediction over all footprints / all DBT footprints -----------
    lev = pd.DataFrame([
        dict(over="all footprints", mean_gedi=y.mean(), mean_full=pred_full.mean(), mean_dbt=pred_dbt.mean()),
        dict(over="DBT footprints", mean_gedi=y[is_dbt].mean(), mean_full=pred_full[is_dbt].mean(),
             mean_dbt=pred_dbt[is_dbt].mean()),
        dict(over="non-DBT footprints", mean_gedi=y[~is_dbt].mean(), mean_full=pred_full[~is_dbt].mean(),
             mean_dbt=pred_dbt[~is_dbt].mean()),
    ])
    lev["ratio_dbt_to_full"] = lev.mean_dbt / lev.mean_full
    lev.round(3).to_csv(C.TAB / "11_level_shift.csv", index=False)
    print("\nLevel shift:\n", lev.round(3).to_string(index=False))

    # --- slope association -----------------------------------------------------------------
    slope = df["slope"].values
    srows = []
    for label, s in (("all", np.ones(len(y), bool)), ("DBT", is_dbt), ("grass", (df.pft_class == 6).values),
                     ("crops", df.pft_class.isin([7, 8]).values)):
        srows.append(dict(subset=label, n=int(s.sum()), what="GEDI AGBD",
                          rho=spearmanr(slope[s], y[s]).statistic))
    for name, p in (("FULL", pred_full), ("DBT", pred_dbt)):
        srows.append(dict(subset="all", n=len(y), what=f"{name} prediction (CV)",
                          rho=spearmanr(slope, p).statistic))
    sl = pd.DataFrame(srows)
    cls = pd.cut(slope, SLOPE_BINS, right=False,
                 labels=[f"{a}-{b}" if b < 90 else f">{a}" for a, b in zip(SLOPE_BINS[:-1], SLOPE_BINS[1:])])
    by_cls = (pd.DataFrame({"slope_class": cls, "gedi_all": y, "gedi_dbt": np.where(is_dbt, y, np.nan),
                            "pred_full": pred_full, "pred_dbt": pred_dbt})
              .groupby("slope_class", observed=True).mean())
    by_cls["n"] = pd.Series(cls).value_counts().reindex(by_cls.index).values
    sl.round(3).to_csv(C.TAB / "11_slope_association.csv", index=False)
    by_cls.round(2).to_csv(C.TAB / "11_slope_class_means.csv")
    shares = pd.crosstab(cls, df.pft_class.map(lambda v: PFT_NAMES.get(v, str(v))), normalize="index") * 100
    shares.round(1).to_csv(C.TAB / "11_slope_class_pft.csv")
    print("\nSlope association:\n", sl.round(3).to_string(index=False))
    print("\nMeans by slope class:\n", by_cls.round(1).to_string())
    print("\nPFT shares (%) by slope class:\n", shares.round(1).to_string())

    # --- permutation importance of the DBT model on the step-3 hold-out fold --------------
    tr, te = next(GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups))
    tr_dbt, te_dbt = tr[is_dbt[tr]], te[is_dbt[te]]
    m = RandomForestRegressor(**params).fit(X[tr_dbt], y[tr_dbt])
    pi = permutation_importance(m, X[te_dbt], y[te_dbt], n_repeats=5, random_state=C.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False)
    imp.round(4).to_csv(C.TAB / "11_importance_dbt.csv", index=False)
    print("\nDBT model, top 10 permutation importance:\n", imp.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
