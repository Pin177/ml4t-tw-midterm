"""Stage 2: run every candidate through the five gates on validation data only. Writes reports/triage.csv."""
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import FEATURES, H, RAW, REPORTS, SIGNALS, STATES, VAL_END, VAL_START, daily_ic, folds, load_matrix, load_panel, log_run, nw_mean

ROUND_TRIP_BP = 85  # 58.5 bp commission and tax + two half-spreads at about 13 bp, the setup's middle tier


def horizon_labels(panel):
    op = load_matrix("open")
    out = {}
    for h in (5, 63):
        lab = op.shift(-(h + 1)) / op.shift(-1) - 1
        s = lab.stack(future_stack=True)
        s.index.names = ["date", "code"]
        out[h] = s
    return out


def quintile_stats(df, f):
    d = df[["date", f, "label"]].dropna()
    d["q"] = d.groupby("date")[f].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    by = d.groupby(["date", "q"])["label"].mean().unstack()
    means = by.mean()
    return means, stats.spearmanr(means.index, means.values).statistic


def turnover(df, f):
    """One-way monthly turnover of the top quintile, from month-end decisions."""
    d = df[df["month_end"]][["date", "code", f]].dropna()
    d["top"] = d.groupby("date")[f].transform(lambda s: s.rank(pct=True) > 0.8)
    sets = d[d["top"]].groupby("date")["code"].apply(set)
    prev, out = None, []
    for s in sets:
        if prev:
            out.append(len(s - prev) / max(len(s), 1))
        prev = s
    return float(np.mean(out))


def main():
    panel = load_panel()
    val = panel[(panel["date"] >= VAL_START) & (panel["date"] <= VAL_END)].copy()
    print("validation rows", len(val), "dates", val["date"].nunique(), flush=True)
    hl = horizon_labels(panel)
    idx = pd.MultiIndex.from_frame(val[["date", "code"]])
    for h, s in hl.items():
        val[f"label_{h}"] = s.reindex(idx).to_numpy()

    rows = []
    fold_list = folds()
    for f in SIGNALS:
        ic = daily_ic(val, f)
        r = nw_mean(ic)
        p = 2 * stats.norm.sf(abs(r["mean"] / r["se"]))
        fold_ic = [ic[(ic.index >= a) & (ic.index <= b)].mean() for a, b in fold_list]
        same = np.mean(np.sign(fold_ic) == np.sign(r["mean"]))
        icir = ic.mean() / ic.std()
        # Gate 1: timing placebo. The same feature, as it stood k trading days earlier
        placebo = {}
        ordered = panel.sort_values(["code", "date"])
        for k in (5, 21):
            lagged = ordered.groupby("code")[f].shift(k)
            tmp = ordered.assign(_lag=lagged)
            tmp = tmp[(tmp["date"] >= VAL_START) & (tmp["date"] <= VAL_END)]
            placebo[k] = daily_ic(tmp, "_lag").mean()
        # Gate 3 and 4
        q_means, mono = quintile_stats(val, f)
        spread_bp = (q_means.iloc[-1] - q_means.iloc[0]) * 1e4
        to = turnover(val, f)
        cost_bp = to * ROUND_TRIP_BP
        decay = {h: daily_ic(val, f, f"label_{h}").mean() for h in hl}
        rows.append(dict(feature=f, role="signal", ic_mean=r["mean"], ic_low=r["low"], ic_high=r["high"], p_nw=p, icir=icir,
                         folds_same_sign=same, ic_5d=decay[5], ic_21d=r["mean"], ic_63d=decay[63],
                         placebo_5=placebo[5], placebo_21=placebo[21], quintile_spearman=mono, q5_q1_bp_21d=spread_bp,
                         turnover_monthly=to, cost_bp_month=cost_bp))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t["bh_reject"] = multipletests(t["p_nw"], alpha=0.05, method="fdr_bh")[0]

    def decide(x):
        """Decision rules fixed here before the table was read. Gates failed are named."""
        fails = []
        if abs(x["placebo_21"]) > 0.5 * abs(x["ic_mean"]):
            fails.append("G1 placebo keeps IC")
        if x["ic_low"] <= 0 <= x["ic_high"]:
            fails.append("G2 CI spans 0")
        if x["ic_mean"] < 0:
            fails.append("G2 wrong sign")
        if x["quintile_spearman"] < 0.8:
            fails.append("G3 not monotone")
        if x["q5_q1_bp_21d"] < 2 * x["cost_bp_month"]:
            fails.append("G4 spread under 2x cost")
        if not x["bh_reject"]:
            fails.append("G5 BH")
        wrong = x["ic_mean"] < 0 and x["ic_high"] < 0
        decision = "STOP" if wrong or ("G2 CI spans 0" in fails and "G5 BH" in fails) else ("PROCEED" if not fails else "REVISE")
        return pd.Series(dict(decision=decision, failed="; ".join(fails)))

    t = pd.concat([t, t.apply(decide, axis=1)], axis=1)

    # Redundancy: mean cross-sectional rank correlation at month ends
    me = val[val["month_end"]]
    corr = me.groupby("date")[SIGNALS].apply(lambda g: g.corr(method="spearman")).groupby(level=1).mean().reindex(index=SIGNALS, columns=SIGNALS)
    corr.round(3).to_csv(REPORTS / "feature_correlation.csv")

    # Mechanism checks on the top three features by |IC|: sector-neutral IC and volatility regimes
    sectors = json.loads((RAW / "sectors.json").read_text(encoding="utf-8"))
    val["sector"] = val["code"].map(sectors).fillna("NA")
    top3 = t.reindex(t["ic_mean"].abs().sort_values(ascending=False).index)["feature"].head(3).tolist()
    terciles = pd.qcut(val.drop_duplicates("date").set_index("date")["market_vol_pctile"], 3, labels=["low vol", "mid vol", "high vol"])
    val["regime"] = val["date"].map(terciles)
    mech = []
    for f in top3:
        neutral = val[f] - val.groupby(["date", "sector"])[f].transform("mean")
        row = dict(feature=f, ic=daily_ic(val, f).mean(), sector_neutral_ic=daily_ic(val.assign(_n=neutral), "_n").mean())
        for reg, g in val.groupby("regime", observed=True):
            row[reg] = daily_ic(g, f).mean()
        mech.append(row)
    mech = pd.DataFrame(mech)
    mech.round(4).to_csv(REPORTS / "mechanism_checks.csv", index=False)

    state = []
    for s in STATES:
        cut = pd.qcut(val.drop_duplicates("date").set_index("date")[s], 3, labels=["low", "mid", "high"])
        v = val.assign(_r=val["date"].map(cut))
        state.append(dict(feature=s, **{str(k): daily_ic(g, "mom_12_1").mean() for k, g in v.groupby("_r", observed=True)}))
    state = pd.DataFrame(state)
    state.round(4).to_csv(REPORTS / "state_variable_checks.csv", index=False)

    t.round(5).to_csv(REPORTS / "triage.csv", index=False)
    for _, x in t.iterrows():
        log_run("feature_triage", x["feature"], 1, "five gates", f"{VAL_START.date()}..{VAL_END.date()}", "ic_mean", float(x["ic_mean"]), x["decision"], x["failed"] or "all passed")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(t[["feature", "ic_mean", "ic_low", "ic_high", "icir", "folds_same_sign", "placebo_21", "quintile_spearman", "q5_q1_bp_21d", "cost_bp_month", "bh_reject", "decision"]].round(4))
    print(t[["feature", "failed"]])
    print(corr.round(2)); print(mech.round(4)); print(state.round(4))


if __name__ == "__main__":
    main()
