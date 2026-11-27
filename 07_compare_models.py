"""
Step 7 (local, GPU): compare Random Forest with two further learners on the
same GEDI training table, features and spatial blocks.

  LIN  ridge regression (linear baseline)
  RF   scikit-learn Random Forest (the step-3 model)
  XGB  gradient-boosted trees (XGBoost, CUDA)
  MLP  fully connected neural network (PyTorch, CUDA)
  ENS  mean of RF, XGB and MLP

Evaluation is *nested* spatial block CV: the outer 5 folds are exactly the
step-3 folds (GroupKFold on 5 km blocks); hyper-parameters are chosen inside
each outer training set with an inner 4-fold spatial CV, so the outer scores
are not biased by tuning. Uncertainty of the scores comes from a bootstrap
over spatial blocks.

Outputs
  outputs/tables/07_cv_metrics.csv            outer-CV metrics per model (+ 95% block-bootstrap CI)
  outputs/tables/07_cv_metrics_by_fold.csv
  outputs/tables/07_paired_differences.csv    RMSE / R2 difference vs RF (bootstrap CI)
  outputs/tables/07_accuracy_by_agbd_range.csv
  outputs/tables/07_selected_hyperparameters.csv
  outputs/tables/07_permutation_importance.csv
  outputs/figures/07_*.png
  data/cv_predictions_models.csv
  outputs/models/{rf.joblib, xgb.json, mlp.pt}   final models fit on all footprints (for step 9)

    CUDA_VISIBLE_DEVICES=0 python 07_compare_models.py
    CUDA_VISIBLE_DEVICES=0 python 07_compare_models.py --final-only   # final models only

Every outer-fold result and final model is cached in outputs/models/cache/, so an
interrupted run resumes, and the outer folds can be run as parallel processes:

    python 07_compare_models.py --fold 0   # ... --fold 4, and --final-only, in parallel
    python 07_compare_models.py            # then assemble metrics, figures, importance
"""
import itertools
import json
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import config as C

MODELS_DIR = C.OUT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CACHE_DIR = MODELS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INNER_FOLDS = 4
N_BOOT = 1000
RNG = np.random.default_rng(C.SEED)
COLORS = {"LIN": "#ae3ec9", "RF": "#2b8a3e", "XGB": "#e8590c", "MLP": "#1c7ed6", "ENS": "#495057"}

# ----------------------------------------------------------------------------
# Hyper-parameter search spaces
# ----------------------------------------------------------------------------
LIN_GRID = [dict(alpha=a) for a in (0.1, 1.0, 10.0, 100.0, 1000.0)]

RF_GRID = [dict(max_features=mf, min_samples_leaf=leaf)
           for mf in (0.2, 0.33, 0.5) for leaf in (1, 5, 10, 20)]
RF_FIXED = dict(max_samples=0.632, n_jobs=-1, random_state=C.SEED)
RF_TUNE_TREES, RF_FINAL_TREES = 150, 300

XGB_SPACE = dict(max_depth=[4, 6, 8, 10], learning_rate=[0.02, 0.05, 0.1],
                 subsample=[0.7, 0.9], colsample_bytree=[0.5, 0.8],
                 min_child_weight=[1, 5, 20], reg_lambda=[1.0, 5.0])
XGB_N_CONFIGS = 24
XGB_FIXED = dict(tree_method="hist", device=DEVICE, objective="reg:squarederror",
                 eval_metric="rmse", random_state=C.SEED)

MLP_GRID = [dict(width=w, depth=d, dropout=p, weight_decay=wd)
            for w in (256, 512) for d in (2, 3) for p in (0.1, 0.3) for wd in (1e-4, 1e-3)]
MLP_MAX_EPOCHS, MLP_PATIENCE, MLP_BATCH, MLP_LR = 400, 40, 1024, 1e-3


def sample_configs(space, n, seed):
    keys = list(space)
    allc = [dict(zip(keys, v)) for v in itertools.product(*space.values())]
    idx = np.random.default_rng(seed).choice(len(allc), size=min(n, len(allc)), replace=False)
    return [allc[i] for i in idx]


XGB_GRID = sample_configs(XGB_SPACE, XGB_N_CONFIGS, C.SEED)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(y, p):
    r = p - y
    rmse = float(np.sqrt(np.mean(r ** 2)))
    return dict(n=int(len(y)), r2=float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2)),
                rmse=rmse, rrmse_pct=100 * rmse / float(y.mean()),
                mae=float(np.mean(np.abs(r))), bias=float(r.mean()),
                # slope of predicted on observed: 1 = no range compression
                slope=float(np.polyfit(y, p, 1)[0]))


# ----------------------------------------------------------------------------
# Learners: fit(Xtr, ytr, params, Xval=None, yval=None) -> (model, info)
# ----------------------------------------------------------------------------
def lin_fit(X, y, params, *_, **__):
    sc = Scaler().fit(X)
    return (Ridge(**params).fit(sc.transform(X), y), sc), {}


def lin_predict(m, X):
    model, sc = m
    return model.predict(sc.transform(X))


def rf_fit(X, y, params, n_trees=RF_FINAL_TREES, **_):
    return RandomForestRegressor(n_estimators=n_trees, **RF_FIXED, **params).fit(X, y), {}


def rf_predict(m, X):
    return m.predict(X)


def xgb_fit(X, y, params, Xval=None, yval=None, n_rounds=None):
    if Xval is not None:     # tuning: early stopping on the inner validation fold
        m = xgb.XGBRegressor(n_estimators=4000, early_stopping_rounds=100, **XGB_FIXED, **params)
        m.fit(X, y, eval_set=[(Xval, yval)], verbose=False)
        return m, {"rounds": m.best_iteration + 1}
    m = xgb.XGBRegressor(n_estimators=n_rounds, **XGB_FIXED, **params)
    return m.fit(X, y, verbose=False), {}


def xgb_predict(m, X):
    return m.predict(X)


class MLP(torch.nn.Module):
    def __init__(self, d_in, width, depth, dropout):
        super().__init__()
        layers, h = [], d_in
        for i in range(depth):
            w = max(32, width // 2 ** i)
            layers += [torch.nn.Linear(h, w), torch.nn.BatchNorm1d(w), torch.nn.ReLU(),
                       torch.nn.Dropout(dropout)]
            h = w
        layers.append(torch.nn.Linear(h, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Scaler:
    """Clip to the training 0.5-99.5 percentiles, then standardise."""
    def fit(self, X):
        self.lo, self.hi = np.percentile(X, [0.5, 99.5], axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-6
        return self

    def transform(self, X):
        return ((np.clip(X, self.lo, self.hi) - self.mu) / self.sd).astype(np.float32)


Y_SCALE = 100.0     # train on AGBD / 100


def _recalibrate_bn(net, Xt):
    """Re-estimate BatchNorm statistics with dropout switched off. The running
    statistics collected during training see dropout-inflated activations, which
    shifts eval-mode predictions (biased the final MLP by -6.5 Mg/ha before this fix)."""
    net.eval()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None           # cumulative average over all batches
        m.train()
    with torch.no_grad():
        for i in range(0, len(Xt), MLP_BATCH):
            if len(Xt) - i >= 2:
                net(Xt[i:i + MLP_BATCH])
    for m in bns:
        m.momentum = 0.1
    net.eval()


def _train_mlp(Xs, y, params, Xv=None, yv=None, epochs=None, seed=C.SEED):
    torch.manual_seed(seed)
    net = MLP(Xs.shape[1], params["width"], params["depth"], params["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=params["weight_decay"])
    Xt = torch.tensor(Xs, device=DEVICE)
    yt = torch.tensor(y / Y_SCALE, dtype=torch.float32, device=DEVICE)
    if Xv is not None:
        Xvt = torch.tensor(Xv, device=DEVICE)
        # BN recalibration before each validation check on a fixed 8k subsample
        Xcal = Xt[torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed))[:8192].to(DEVICE)]
    n, best, best_ep, best_state, wait = len(Xt), np.inf, 0, None, 0
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for ep in range(epochs or MLP_MAX_EPOCHS):
        net.train()
        perm = torch.randperm(n, device=DEVICE, generator=g)
        for i in range(0, n, MLP_BATCH):
            idx = perm[i:i + MLP_BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if Xv is not None:
            _recalibrate_bn(net, Xcal)
            with torch.no_grad():
                rmse = float(torch.sqrt(torch.mean((net(Xvt).cpu() * Y_SCALE - torch.tensor(yv)) ** 2)))
            if rmse < best:
                best, best_ep, wait = rmse, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= MLP_PATIENCE:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    _recalibrate_bn(net, Xt)
    return net, best_ep


def mlp_fit(X, y, params, Xval=None, yval=None, n_epochs=None):
    sc = Scaler().fit(X)
    if Xval is not None:
        net, ep = _train_mlp(sc.transform(X), y, params, sc.transform(Xval), yval)
        return (net, sc), {"epochs": ep}
    net, _ = _train_mlp(sc.transform(X), y, params, epochs=n_epochs)
    return (net, sc), {}


def mlp_predict(m, X):
    net, sc = m
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 65536):
            out.append(net(torch.tensor(sc.transform(X[i:i + 65536]), device=DEVICE)).cpu().numpy())
    return np.concatenate(out) * Y_SCALE


LEARNERS = {
    "LIN": dict(fit=lin_fit, predict=lin_predict, grid=LIN_GRID),
    "RF": dict(fit=rf_fit, predict=rf_predict, grid=RF_GRID),
    "XGB": dict(fit=xgb_fit, predict=xgb_predict, grid=XGB_GRID),
    "MLP": dict(fit=mlp_fit, predict=mlp_predict, grid=MLP_GRID),
}
ENS_MEMBERS = ["RF", "XGB", "MLP"]


# ----------------------------------------------------------------------------
# Nested spatial CV
# ----------------------------------------------------------------------------
def tune(name, X, y, groups):
    """Inner spatial CV over the grid; returns best params and the fit settings
    (trees / boosting rounds / epochs) to use when refitting on all of X."""
    L = LEARNERS[name]
    inner = GroupKFold(n_splits=INNER_FOLDS)
    rows = []
    for k, params in enumerate(L["grid"]):
        preds, extras = np.full(len(y), np.nan), []
        for tr, va in inner.split(X, y, groups):
            if name == "RF":
                m, info = rf_fit(X[tr], y[tr], params, n_trees=RF_TUNE_TREES)
            else:
                m, info = L["fit"](X[tr], y[tr], params, X[va], y[va])
            preds[va] = L["predict"](m, X[va])
            extras.append(info)
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        rows.append(dict(config=k, rmse=rmse, **params,
                         **{key: float(np.mean([e[key] for e in extras])) for key in extras[0]}))
    res = pd.DataFrame(rows).sort_values("rmse")
    best = res.iloc[0]
    params = dict(L["grid"][int(best["config"])])     # original types (a DataFrame row casts ints to float)
    # refit on (k-1)/k more data than each inner fit -> scale rounds/epochs up slightly
    scale = INNER_FOLDS / (INNER_FOLDS - 1)
    fit_kw = ({"n_rounds": int(round(best["rounds"] * scale))} if name == "XGB" else
              {"n_epochs": int(round(best["epochs"] * scale))} if name == "MLP" else {})
    return params, fit_kw, float(best["rmse"])


def outer_fold(f, name, X, y, groups, tr, te):
    """Tune on the outer training set, predict the outer test blocks (cached)."""
    path = CACHE_DIR / f"cv_fold{f}_{name}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return np.array(d["pred"]), d["row"]
    t0 = time.time()
    params, fit_kw, inner_rmse = tune(name, X[tr], y[tr], groups[tr])
    m, _ = LEARNERS[name]["fit"](X[tr], y[tr], params, **fit_kw)
    pred = LEARNERS[name]["predict"](m, X[te])
    row = dict(fold=f, model=name, inner_rmse=inner_rmse, **params, **fit_kw)
    r = metrics(y[te], pred)
    print(f"  fold {f} {name}: R2={r['r2']:.3f} RMSE={r['rmse']:.1f}  "
          f"params={params} {fit_kw}  ({time.time() - t0:.0f}s)", flush=True)
    path.write_text(json.dumps({"pred": [float(v) for v in pred], "row": row}))
    return pred, row


def nested_cv(df, feats, only_fold=None):
    X = df[feats].values.astype(np.float32)
    y = df["agbd"].values.astype(np.float64)
    groups = df["block"].values
    outer = GroupKFold(n_splits=C.N_FOLDS)          # identical to step 3
    preds = {n: np.full(len(y), np.nan) for n in LEARNERS}
    fold_id = np.full(len(y), -1)
    chosen = []
    for f, (tr, te) in enumerate(outer.split(X, y, groups)):
        if only_fold is not None and f != only_fold:
            continue
        fold_id[te] = f
        for name in LEARNERS:
            preds[name][te], row = outer_fold(f, name, X, y, groups, tr, te)
            chosen.append(row)
    if only_fold is not None:
        return None
    preds["ENS"] = np.mean([preds[n] for n in ENS_MEMBERS], axis=0)
    return X, y, groups, fold_id, preds, pd.DataFrame(chosen)


# ----------------------------------------------------------------------------
# Block bootstrap
# ----------------------------------------------------------------------------
def block_bootstrap(y, preds, groups):
    blocks = np.unique(groups)
    idx_by_block = {b: np.flatnonzero(groups == b) for b in blocks}
    out = {n: {"rmse": [], "r2": []} for n in preds}
    for _ in range(N_BOOT):
        sel = np.concatenate([idx_by_block[b] for b in RNG.choice(blocks, len(blocks))])
        ys = y[sel]
        for n, p in preds.items():
            m = metrics(ys, p[sel])
            out[n]["rmse"].append(m["rmse"])
            out[n]["r2"].append(m["r2"])
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in out.items()}


# ----------------------------------------------------------------------------
# Permutation importance (same spatial hold-out fold as step 3)
# ----------------------------------------------------------------------------
def permutation_importance_model(name, model, X, y, feats, repeats=5):
    pred = LEARNERS[name]["predict"]
    base = metrics(y, pred(model, X))["r2"]
    rng = np.random.default_rng(C.SEED)
    rows = []
    for j, f in enumerate(feats):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - metrics(y, pred(model, Xp))["r2"])
        rows.append(dict(model=name, feature=f, importance=np.mean(drops), std=np.std(drops)))
    return rows


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def plot_scatter(y, preds, mets):
    names = list(preds)
    lim = np.percentile(y, 99.5)
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 4.1), sharey=True)
    for ax, n in zip(axes, names):
        ax.hexbin(y, preds[n], gridsize=55, mincnt=1, bins="log", extent=(0, lim, 0, lim), cmap="viridis")
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        m = mets[n]
        ax.set_title(f"{n}: R²={m['r2']:.2f}, RMSE={m['rmse']:.1f}, slope={m['slope']:.2f}", fontsize=10)
        ax.set_xlabel("GEDI L4A AGBD (Mg/ha)")
    axes[0].set_ylabel("Predicted AGBD (Mg/ha)")
    fig.tight_layout()
    fig.savefig(C.FIG / "07_cv_scatter_models.png", dpi=200)
    plt.close(fig)


def plot_by_range(rng_df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ranges = rng_df.agbd_range.unique()
    x = np.arange(len(ranges))
    w = 0.8 / rng_df.model.nunique()
    for i, (n, g) in enumerate(rng_df.groupby("model", sort=False)):
        ax[0].bar(x + i * w, g.rmse, w, label=n, color=COLORS[n])
        ax[1].bar(x + i * w, g.bias, w, label=n, color=COLORS[n])
    for a, lab in zip(ax, ["RMSE (Mg/ha)", "Bias, predicted − observed (Mg/ha)"]):
        a.set_xticks(x + w * (rng_df.model.nunique() - 1) / 2, ranges, rotation=30)
        a.set(xlabel="GEDI AGBD range (Mg/ha)", ylabel=lab)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "07_error_by_agbd_range.png", dpi=200)
    plt.close(fig)


def plot_bootstrap(boot):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, a, lab in [("rmse", ax[0], "RMSE (Mg/ha)"), ("r2", ax[1], "R²")]:
        data = [boot[n][k] for n in boot]
        parts = a.violinplot(data, showmedians=True)
        for body, n in zip(parts["bodies"], boot):
            body.set_facecolor(COLORS[n])
            body.set_alpha(0.6)
        a.set_xticks(range(1, len(boot) + 1), list(boot))
        a.set(ylabel=lab, title=f"Block-bootstrap distribution ({N_BOOT} resamples)")
    fig.tight_layout()
    fig.savefig(C.FIG / "07_bootstrap_scores.png", dpi=200)
    plt.close(fig)


def plot_importance(imp):
    # the linear model's values (collinear predictors) would dwarf the others; it stays in the table
    imp = imp[imp.model.isin(ENS_MEMBERS)]
    top = (imp.groupby("feature").importance.mean().sort_values(ascending=False).head(15).index[::-1])
    fig, ax = plt.subplots(figsize=(7.5, 7))
    names = imp.model.unique()
    h = 0.8 / len(names)
    yy = np.arange(len(top))
    for i, n in enumerate(names):
        g = imp[imp.model == n].set_index("feature").loc[top]
        ax.barh(yy + i * h, g.importance, h, xerr=g["std"], label=n, color=COLORS[n])
    ax.set_yticks(yy + h * (len(names) - 1) / 2, top)
    ax.set(xlabel="Decrease in R² when permuted", title="Permutation importance (spatial hold-out fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "07_permutation_importance_models.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    cfg = json.loads(C.MODEL_JSON.read_text())
    feats = cfg["features"]
    df = pd.read_csv(C.TRAINING_CSV).dropna(subset=feats + ["agbd"]).reset_index(drop=True)
    print(f"{len(df):,} footprints, {len(feats)} features, {df.block.nunique()} blocks, device={DEVICE}")

    if "--fold" in sys.argv:
        nested_cv(df, feats, only_fold=int(sys.argv[sys.argv.index("--fold") + 1]))
        return
    if "--final-only" in sys.argv:
        final_models(df[feats].values.astype(np.float32), df["agbd"].values.astype(np.float64),
                     df["block"].values, feats, importance=False)
        return

    X, y, groups, fold_id, preds, chosen = nested_cv(df, feats)
    chosen.to_csv(C.TAB / "07_selected_hyperparameters.csv", index=False)
    pd.DataFrame({"block": groups, "fold": fold_id, "agbd": y, **{f"pred_{n}": p for n, p in preds.items()}}) \
        .to_csv(C.DATA / "cv_predictions_models.csv", index=False)

    # --- metrics, CIs, paired differences ---------------------------------------
    boot = block_bootstrap(y, preds, groups)
    mets = {n: metrics(y, p) for n, p in preds.items()}
    rows = []
    for n, m in mets.items():
        lo_r, hi_r = np.percentile(boot[n]["rmse"], [2.5, 97.5])
        lo_2, hi_2 = np.percentile(boot[n]["r2"], [2.5, 97.5])
        rows.append(dict(model=n, **m, rmse_ci_lo=lo_r, rmse_ci_hi=hi_r, r2_ci_lo=lo_2, r2_ci_hi=hi_2))
    pd.DataFrame(rows).round(3).to_csv(C.TAB / "07_cv_metrics.csv", index=False)
    print("\n", pd.DataFrame(rows).round(3).to_string(index=False))

    diffs = []
    for n in preds:
        if n == "RF":
            continue
        d_rmse = boot[n]["rmse"] - boot["RF"]["rmse"]
        d_r2 = boot[n]["r2"] - boot["RF"]["r2"]
        diffs.append(dict(model=n, d_rmse=mets[n]["rmse"] - mets["RF"]["rmse"],
                          d_rmse_ci_lo=np.percentile(d_rmse, 2.5), d_rmse_ci_hi=np.percentile(d_rmse, 97.5),
                          p_rmse_not_lower=float(np.mean(d_rmse >= 0)),
                          d_r2=mets[n]["r2"] - mets["RF"]["r2"],
                          d_r2_ci_lo=np.percentile(d_r2, 2.5), d_r2_ci_hi=np.percentile(d_r2, 97.5)))
    pd.DataFrame(diffs).round(4).to_csv(C.TAB / "07_paired_differences.csv", index=False)
    print("\n", pd.DataFrame(diffs).round(3).to_string(index=False))

    by_fold = [dict(model=n, fold=f, **metrics(y[fold_id == f], p[fold_id == f]))
               for n, p in preds.items() for f in range(C.N_FOLDS)]
    pd.DataFrame(by_fold).round(3).to_csv(C.TAB / "07_cv_metrics_by_fold.csv", index=False)

    bins = [0, 25, 50, 75, 100, 150, 200, 300, np.inf]
    rng_rows = []
    for n, p in preds.items():
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (y >= lo) & (y < hi)
            m = metrics(y[s], p[s])
            rng_rows.append(dict(model=n, agbd_range=f"{lo}-{hi}" if np.isfinite(hi) else f">{lo}",
                                 n=m["n"], rmse=m["rmse"], bias=m["bias"]))
    rng_df = pd.DataFrame(rng_rows)
    rng_df.round(2).to_csv(C.TAB / "07_accuracy_by_agbd_range.csv", index=False)

    q = [5, 25, 50, 75, 95]
    dist = pd.DataFrame([dict(source="GEDI", **dict(zip([f"p{v}" for v in q], np.percentile(y, q))))] +
                        [dict(source=n, **dict(zip([f"p{v}" for v in q], np.percentile(p, q))))
                         for n, p in preds.items()])
    dist.round(1).to_csv(C.TAB / "07_prediction_distribution.csv", index=False)
    print("\n", dist.round(1).to_string(index=False))

    plot_scatter(y, preds, mets)
    plot_by_range(rng_df)
    plot_bootstrap(boot)
    final_models(X, y, groups, feats)


def save_model(name, m, params, feats):
    if name == "LIN":
        joblib.dump(m, MODELS_DIR / "lin.joblib")
    elif name == "RF":
        joblib.dump(m, MODELS_DIR / "rf.joblib", compress=3)
    elif name == "XGB":
        # save the Booster: the sklearn wrapper's save_model fails with scikit-learn >= 1.6
        m.get_booster().save_model(MODELS_DIR / "xgb.json")
    else:
        net, sc = m
        torch.save({"state_dict": net.state_dict(), "params": params, "d_in": len(feats),
                    "scaler": {k: getattr(sc, k).tolist() for k in ("lo", "hi", "mu", "sd")}},
                   MODELS_DIR / "mlp.pt")


def final_fit(name, X, y, groups, feats):
    """Re-tune on all footprints and fit the model used for mapping (cached)."""
    path = CACHE_DIR / f"final_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    params, fit_kw, inner_rmse = tune(name, X, y, groups)
    m, _ = LEARNERS[name]["fit"](X, y, params, **fit_kw)
    save_model(name, m, params, feats)
    print(f"  {name}: {params} {fit_kw}  (spatial-CV RMSE {inner_rmse:.1f})", flush=True)
    spec = {"params": params, "fit_kw": fit_kw}
    path.write_text(json.dumps(spec))
    return spec


def final_models(X, y, groups, feats, importance=True):
    # --- final models on all data: re-tune on the full set ------------------------
    print("\nFitting final models on all footprints ...")
    final = {name: final_fit(name, X, y, groups, feats) for name in LEARNERS}
    (MODELS_DIR / "final_models.json").write_text(json.dumps(
        {"features": feats, **{n: {"params": d["params"], **d["fit_kw"]} for n, d in final.items()}}, indent=2))
    if not importance:
        return

    # --- permutation importance: same hold-out fold as step 3 -----------------------
    tr, te = next(GroupKFold(n_splits=C.N_FOLDS).split(X, y, groups))
    imp = []
    for name in LEARNERS:
        params, fit_kw = final[name]["params"], final[name]["fit_kw"]
        m, _ = LEARNERS[name]["fit"](X[tr], y[tr], params, **fit_kw)
        imp += permutation_importance_model(name, m, X[te], y[te], feats)
    imp = pd.DataFrame(imp)
    imp.round(4).to_csv(C.TAB / "07_permutation_importance.csv", index=False)
    plot_importance(imp)
    print("Done.")


if __name__ == "__main__":
    main()
