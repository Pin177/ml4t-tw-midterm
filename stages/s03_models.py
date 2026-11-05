"""Stage 3: climb the model ladder on walk-forward validation folds.

Writes data/processed/scores_val.parquet, reports/ic_by_fold.csv, reports/paired_comparison.csv,
reports/shap_summary.csv, reports/conformal.csv and reports/fold_protocol.csv.
"""
import json
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from common import (H, PROC, REPORTS, SEED, SIGNALS, STATES, VAL_END, VAL_START, daily_ic, folds, load_panel, log_run, nw_mean,
                    purged_train)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ALPHAS = [10.0 ** k for k in range(-3, 6)]      # nine values, eight orders of magnitude
OPTUNA_TRIALS_PER_FOLD = 12                     # 84 in total over 7 folds, inside the guideline's 100
TRAIN_EVERY_N_DAYS = 5                          # fit on weekly cross-sections to thin overlapping labels


def add_ranks(df, features):
    """Cross-sectional ranks within one date use no other date, so they cannot leak across folds."""
    df = df.copy()
    for f in features:
        df[f] = df.groupby("date")[f].rank(pct=True) - 0.5
    df["y"] = df.groupby("date")["label"].rank(pct=True) - 0.5
    return df


def winsor_limits(train, cols):
    return {c: (train[c].quantile(0.01), train[c].quantile(0.99)) for c in cols}


def apply_limits(df, limits):
    df = df.copy()
    for c, (lo, hi) in limits.items():
        df[c] = df[c].clip(lo, hi)
    return df


def thin(df, calendar):
    keep = set(calendar[::TRAIN_EVERY_N_DAYS])
    return df[df["date"].isin(keep)]


def inner_split(train, calendar):
    """Last 252 trading days of the training window for tuning, with a purge in front of it."""
    days = pd.DatetimeIndex(sorted(train["date"].unique()))
    start = days[-252]
    pos = days.searchsorted(start)
    return train[train["date"] < days[pos - H]], train[train["date"] >= start]


def mean_ic(df, pred):
    return daily_ic(df.assign(_p=pred), "_p").mean()


def fit_ridge(fit, tune, feats):
    best = (-np.inf, None)
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(fit[feats].fillna(0), fit["y"])
        ic = mean_ic(tune, m.predict(tune[feats].fillna(0)))
        if ic > best[0]:
            best = (ic, a)
    return best[1], best[0]


def fit_lgbm(fit, tune, feats, year):
    dfit = lgb.Dataset(fit[feats], fit["y"], free_raw_data=False)

    def objective(trial):
        params = dict(objective="l1", verbosity=-1, seed=SEED, num_threads=4, deterministic=True, force_row_wise=True, learning_rate=0.03,
                      max_depth=trial.suggest_int("max_depth", 3, 4),
                      num_leaves=trial.suggest_int("num_leaves", 4, 15),
                      min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 200, 2000, log=True),
                      feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                      bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0), bagging_freq=1,
                      lambda_l2=trial.suggest_float("lambda_l2", 1e-2, 100, log=True))
        dtune = lgb.Dataset(tune[feats], tune["y"], reference=dfit, free_raw_data=False)
        m = lgb.train(params, dfit, 400, valid_sets=[dtune], callbacks=[lgb.early_stopping(30, verbose=False)])
        trial.set_user_attr("rounds", m.best_iteration or 400)
        return mean_ic(tune, m.predict(tune[feats]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + year))
    study.optimize(objective, n_trials=OPTUNA_TRIALS_PER_FOLD)
    return dict(study.best_params), study.best_trial.user_attrs["rounds"], study.best_value


def lgb_params(p):
    return dict(objective="l1", verbosity=-1, seed=SEED, num_threads=4, deterministic=True, force_row_wise=True, learning_rate=0.03, bagging_freq=1, **p)


def conformal(fit_pred, fit_y, test_pred, test_y, level=0.8):
    q = np.quantile(np.abs(fit_y - fit_pred), level)
    return q, float(np.mean(np.abs(test_y - test_pred) <= q))


def main():
    triage = pd.read_csv(REPORTS / "triage.csv")
    signals = triage.loc[triage["decision"] != "STOP", "feature"].tolist()
    print("signals going to models:", signals, flush=True)
    log_run("feature_set", "after_triage", 1, json.dumps(signals), "validation", "n_features", len(signals), "use", "triage")

    panel = load_panel()
    panel = panel[panel["date"] <= VAL_END].dropna(subset=["label"])
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel = add_ranks(panel, SIGNALS)
    feats_tree = signals + STATES

    scores, protocol, ic_rows, shaps, conf = [], [], [], [], []
    for a, b in folds():
        year = a.year
        train = purged_train(panel, a, calendar)
        test = panel[(panel["date"] >= a) & (panel["date"] <= b)].copy()
        assert train["label_end"].max() < test["date"].min(), "label overlaps the test block"
        limits = winsor_limits(train, STATES)
        train, test = apply_limits(train, limits), apply_limits(test, limits)
        tr = thin(train, calendar)
        fit, tune = inner_split(tr, calendar)

        alpha, ridge_ic = fit_ridge(fit, tune, signals)
        ridge = Ridge(alpha=alpha).fit(tr[signals].fillna(0), tr["y"])
        test["ridge"] = ridge.predict(test[signals].fillna(0))
        log_run("model_ridge", f"fold_{year}", 1, f"alpha={alpha}", f"inner {tune['date'].min().date()}..", "inner_ic", float(ridge_ic), "selected", "nested")

        params, rounds, gbm_ic = fit_lgbm(fit, tune, feats_tree, year)
        gbm = lgb.train(lgb_params(params), lgb.Dataset(tr[feats_tree], tr["y"]), max(rounds, 10))
        test["lgbm"] = gbm.predict(test[feats_tree])
        log_run("model_lgbm", f"fold_{year}", OPTUNA_TRIALS_PER_FOLD, json.dumps({**params, "rounds": rounds}), f"inner {tune['date'].min().date()}..", "inner_ic", float(gbm_ic), "selected", "nested")

        test["rule"] = test["mom_12_1"]
        z = lambda s: s.groupby(test["date"]).transform(lambda x: (x - x.mean()) / x.std())
        test["ensemble"] = (z(test["rule"]) + z(test["ridge"]) + z(test["lgbm"])) / 3
        scores.append(test[["date", "code", "rule", "ridge", "lgbm", "ensemble"]])
        protocol.append(dict(fold=year, train_start=str(tr["date"].min().date()), train_end=str(tr["date"].max().date()), last_label_end=str(tr["label_end"].max().date()),
                             test_start=str(test["date"].min().date()), test_end=str(test["date"].max().date()), train_rows=len(tr), test_rows=len(test), alpha=alpha, gbm_rounds=rounds))
        for m in ("rule", "ridge", "lgbm", "ensemble"):
            ic_rows.append(dict(fold=year, model=m, ic=daily_ic(test, m).mean()))

        sample = test.sample(min(5000, len(test)), random_state=SEED)
        sv = shap.TreeExplainer(gbm).shap_values(sample[feats_tree])
        for i, f in enumerate(feats_tree):
            rho = np.corrcoef(sample[f].fillna(0), sv[:, i])[0, 1]
            shaps.append(dict(fold=year, feature=f, mean_abs_shap=float(np.abs(sv[:, i]).mean()), direction=float(rho)))
        for name, pred_tr, pred_te in (("lgbm", gbm.predict(tune[feats_tree]), test["lgbm"].to_numpy()), ("ridge", ridge.predict(tune[signals].fillna(0)), test["ridge"].to_numpy())):
            q, cov = conformal(pred_tr, tune["y"].to_numpy(), pred_te, test["y"].to_numpy())
            conf.append(dict(fold=year, model=name, half_width=float(q), coverage=cov, below_80=cov < 0.8))
        print(protocol[-1], ic_rows[-4:], flush=True)

    s = pd.concat(scores)
    s.to_parquet(PROC / "scores_val.parquet", index=False)
    pd.DataFrame(protocol).to_csv(REPORTS / "fold_protocol.csv", index=False)
    pd.DataFrame(ic_rows).pivot(index="fold", columns="model", values="ic").round(4).to_csv(REPORTS / "ic_by_fold.csv")
    pd.DataFrame(shaps).to_csv(REPORTS / "shap_summary.csv", index=False)
    pd.DataFrame(conf).to_csv(REPORTS / "conformal.csv", index=False)

    full = panel[["date", "code", "label"]].merge(s, on=["date", "code"])
    ics = {m: daily_ic(full, m) for m in ("rule", "ridge", "lgbm", "ensemble")}
    summary = []
    for m, ic in ics.items():
        r = nw_mean(ic)
        summary.append(dict(model=m, **r, icir=ic.mean() / ic.std(), positive_folds=int(sum(x["ic"] > 0 for x in ic_rows if x["model"] == m))))
    summary = pd.DataFrame(summary)
    summary.round(5).to_csv(REPORTS / "model_summary.csv", index=False)
    pairs = []
    for up, down in (("ridge", "rule"), ("lgbm", "ridge"), ("ensemble", "lgbm")):
        d = (ics[up] - ics[down]).dropna()
        pairs.append(dict(step=f"{up} vs {down}", **nw_mean(d)))
    pairs = pd.DataFrame(pairs)
    pairs.round(5).to_csv(REPORTS / "paired_comparison.csv", index=False)

    # The ladder rule fixed in setup: go up one rung only if the paired delta's CI lies above zero
    deployed = "rule"
    if pairs.loc[0, "low"] > 0:
        deployed = "ridge"
        if pairs.loc[1, "low"] > 0:
            deployed = "lgbm"
    elif (ics["lgbm"] - ics["rule"]).dropna().pipe(nw_mean)["low"] > 0:
        deployed = "lgbm"
    (REPORTS / "deployed_model.txt").write_text(deployed, encoding="utf-8")
    for _, x in summary.iterrows():
        log_run("model_ladder", x["model"], 1, "walk-forward", f"{VAL_START.date()}..{VAL_END.date()}", "ic_mean", float(x["mean"]), "deployed" if x["model"] == deployed else "not deployed", "paired delta CI")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(summary.round(4)); print(pairs.round(4)); print("deployed:", deployed)
    print(pd.DataFrame(ic_rows).pivot(index="fold", columns="model", values="ic").round(4))
    print(pd.DataFrame(shaps).groupby("feature")[["mean_abs_shap", "direction"]].mean().round(4).sort_values("mean_abs_shap", ascending=False))
    print(pd.DataFrame(conf).round(3))


if __name__ == "__main__":
    main()
