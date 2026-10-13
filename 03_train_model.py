"""
Step 3 (local, no Earth Engine needed): Random Forest regression of GEDI L4A
AGBD on Sentinel-2 (+ terrain) predictors, evaluated with spatial block CV.

  1. descriptive statistics of the GEDI reference data
  2. compare feature sets (S2 leaf-on / two-season / two-season + terrain)
  3. random vs spatial CV for the chosen set (shows optimism of random CV)
  4. small hyper-parameter search under spatial CV
  5. final CV metrics, scatter/residual plots, accuracy by AGBD range
  6. permutation importance
  7. write outputs/model_config.json for GEE wall-to-wall prediction (step 4)

    python 03_train_model.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, KFold

import config as C
import features as F

BASE_PARAMS = dict(n_estimators=300, max_features=0.33, min_samples_leaf=5,
                   max_samples=0.632, n_jobs=-1, random_state=C.SEED)
PARAM_GRID = [dict(max_features=mf, min_samples_leaf=leaf)
              for mf in (0.2, 0.33, 0.5) for leaf in (1, 5, 10, 20)]
MAX_TUNE_ROWS = 40000           # subsample for the grid search only
CORR_PRUNE = 0.98               # drop near-duplicate predictors


def metrics(y, p):
    resid = p - y
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return dict(n=int(len(y)),
                r2=float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=rmse, rrmse_pct=100 * rmse / float(y.mean()),
                mae=float(np.mean(np.abs(resid))), bias=float(resid.mean()))


def cv_predict(df, feats, params, splitter, groups=None):
    pred = np.full(len(df), np.nan)
    X, y = df[feats].values, df["agbd"].values
    for tr, te in splitter.split(X, y, groups):
        m = RandomForestRegressor(**{**BASE_PARAMS, **params}).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def prune_correlated(df, feats, thr):
    corr = df[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop = [c for c in upper.columns if (upper[c] > thr).any()]
    return [f for f in feats if f not in drop], drop


def load():
    df = pd.read_csv(C.TRAINING_CSV)
    n0 = len(df)
    df = df.dropna(subset=F.ALL_PREDICTORS + ["agbd"]).reset_index(drop=True)
    print(f"Loaded {n0:,} footprints, {len(df):,} with complete predictors, "
          f"{df.block.nunique()} spatial blocks of {C.SPATIAL_BLOCK_M / 1000:.0f} km")
    return df


def describe(df):
    df[["agbd", "agbd_se", "sensitivity"]].describe().round(2).to_csv(C.TAB / "03_gedi_summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df.agbd, bins=60, color="#2b8a3e")
    ax[0].set(xlabel="GEDI L4A AGBD (Mg/ha)", ylabel="Footprints", title="Reference AGBD")
    ax[1].scatter(df.agbd, df.agbd_se, s=2, alpha=0.3)
    ax[1].set(xlabel="AGBD (Mg/ha)", ylabel="AGBD SE (Mg/ha)", title="GEDI prediction uncertainty")
    fig.tight_layout()
    fig.savefig(C.FIG / "03_gedi_agbd_distribution.png", dpi=200)
    plt.close(fig)


def plot_eval(df, pred, tag):
    y = df.agbd.values
    m = metrics(y, pred)
    lim = np.percentile(np.r_[y, pred], 99.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    hb = ax[0].hexbin(y, pred, gridsize=60, mincnt=1, bins="log", extent=(0, lim, 0, lim), cmap="viridis")
    ax[0].plot([0, lim], [0, lim], "r--", lw=1)
    ax[0].set(xlabel="GEDI L4A AGBD (Mg/ha)", ylabel="Predicted AGBD (Mg/ha)",
              title=f"Spatial {C.N_FOLDS}-fold CV  R²={m['r2']:.2f}  RMSE={m['rmse']:.1f} "
                    f"({m['rrmse_pct']:.0f}%)  bias={m['bias']:.1f}")
    fig.colorbar(hb, ax=ax[0], label="log10(count)")
    ax[1].hexbin(pred, pred - y, gridsize=60, mincnt=1, bins="log", cmap="viridis")
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set(xlabel="Predicted AGBD (Mg/ha)", ylabel="Predicted − observed (Mg/ha)", title="Residuals")
    fig.tight_layout()
    fig.savefig(C.FIG / f"03_cv_scatter_{tag}.png", dpi=200)
    plt.close(fig)

    bins = [0, 25, 50, 75, 100, 150, 200, 300, np.inf]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (y >= lo) & (y < hi)
        if s.sum() >= 10:
            r = metrics(y[s], pred[s])
            rows.append(dict(agbd_range=f"{lo}-{hi}", n=r["n"], rmse=r["rmse"], bias=r["bias"]))
    pd.DataFrame(rows).round(2).to_csv(C.TAB / f"03_accuracy_by_agbd_range_{tag}.csv", index=False)
    return m


def main():
    df = load()
    describe(df)
    groups = df.block.values
    spatial = GroupKFold(n_splits=C.N_FOLDS)

    # -- 2. compare feature sets --------------------------------------------------
    rows, sets = [], {}
    for name, feats in F.FEATURE_SETS.items():
        feats, dropped = prune_correlated(df, feats, CORR_PRUNE)
        sets[name] = feats
        p = cv_predict(df, feats, {}, spatial, groups)
        m = metrics(df.agbd.values, p)
        rows.append(dict(feature_set=name, n_features=len(feats), **m))
        print(f"[{name}] {len(feats)} features (dropped {dropped}) "
              f"R2={m['r2']:.3f} RMSE={m['rmse']:.1f} rRMSE={m['rrmse_pct']:.1f}%")
    comp = pd.DataFrame(rows).round(3)
    comp.to_csv(C.TAB / "03_feature_set_comparison.csv", index=False)
    best_set = comp.sort_values("rmse").iloc[0]["feature_set"]
    feats = sets[best_set]
    print(f"Best feature set: {best_set}")

    # -- 3. random vs spatial CV ---------------------------------------------------
    p_rand = cv_predict(df, feats, {}, KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED))
    m_rand = metrics(df.agbd.values, p_rand)
    print(f"Random CV R2={m_rand['r2']:.3f} RMSE={m_rand['rmse']:.1f} (optimistic, for comparison)")

    # -- 4. tuning -------------------------------------------------------------------
    tune = df.sample(min(len(df), MAX_TUNE_ROWS), random_state=C.SEED).reset_index(drop=True)
    trows = []
    for params in PARAM_GRID:
        p = cv_predict(tune, feats, params, spatial, tune.block.values)
        trows.append(dict(**params, **metrics(tune.agbd.values, p)))
    tuning = pd.DataFrame(trows).sort_values("rmse")
    tuning.round(3).to_csv(C.TAB / "03_tuning.csv", index=False)
    best = {k: tuning.iloc[0][k] for k in ("max_features", "min_samples_leaf")}
    best = dict(max_features=float(best["max_features"]), min_samples_leaf=int(best["min_samples_leaf"]))
    print(f"Best params: {best}")

    # -- 5. final spatial CV ---------------------------------------------------------
    p_final = cv_predict(df, feats, best, spatial, groups)
    m_final = plot_eval(df, p_final, best_set)
    df.assign(pred_cv=p_final)[["x_utm", "y_utm", "block", "agbd", "pred_cv"]] \
        .to_csv(C.DATA / "cv_predictions.csv", index=False)
    print(f"Final spatial CV: {m_final}")

    # -- 6. permutation importance on a spatial hold-out fold -------------------------
    tr, te = next(spatial.split(df, groups=groups))
    model = RandomForestRegressor(**{**BASE_PARAMS, **best}).fit(df.loc[tr, feats], df.agbd[tr])
    pi = permutation_importance(model, df.loc[te, feats], df.agbd[te], n_repeats=5,
                                random_state=C.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False)
    imp.round(4).to_csv(C.TAB / "03_permutation_importance.csv", index=False)
    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.feature, top.importance, xerr=top["std"], color="#2b8a3e")
    ax.set(xlabel="Decrease in R² when permuted", title="Permutation importance (spatial hold-out)")
    fig.tight_layout()
    fig.savefig(C.FIG / "03_permutation_importance.png", dpi=200)
    plt.close(fig)

    # -- 7. config for GEE -----------------------------------------------------------
    holdout_blocks = sorted(int(b) for b in np.unique(groups[te]))
    cfg = dict(
        feature_set=best_set, features=feats,
        rf=dict(numberOfTrees=BASE_PARAMS["n_estimators"],
                variablesPerSplit=max(1, round(best["max_features"] * len(feats))),
                minLeafPopulation=best["min_samples_leaf"],
                bagFraction=BASE_PARAMS["max_samples"], seed=C.SEED),
        sklearn_params={**BASE_PARAMS, **best},
        cv_spatial=m_final, cv_random=m_rand,
        holdout_blocks=holdout_blocks,
        n_train=int(len(df)), mean_gedi_agbd_se=float(df.agbd_se.mean()),
    )
    C.MODEL_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"Wrote {C.MODEL_JSON}")


if __name__ == "__main__":
    main()
