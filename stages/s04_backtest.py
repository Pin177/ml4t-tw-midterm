"""Stage 4: turn validation scores into trades. Cost sweep, allocators, overlay, stress and regime slices."""
import json

import numpy as np
import pandas as pd

from common import HOLD_START, PROC, REPORTS, SEED, VAL_END, VAL_START, load_panel, log_run
from simulator import deflated_sharpe, load_market, metrics, min_track_record_years, simulate

VOL_TARGET = 0.15
EPISODES = {"2020 COVID": ("2020-01-20", "2020-03-31"), "2022 rate shock": ("2022-01-03", "2022-12-30"), "2024-08 carry unwind": ("2024-07-11", "2024-08-30")}


def decisions_for(panel, scores, col):
    d = panel[panel["month_end"]].merge(scores[["date", "code", col]], on=["date", "code"], how="inner")
    return d.rename(columns={col: "score"})[["date", "code", "score", "median_value", "buy_locked"]].dropna(subset=["score"])


def overlay_exposure(base_nav, dates):
    """Exposure for each decision date, computed from the base strategy's path up to that date (t-1 relative to the fill)."""
    r = base_nav.pct_change()
    out = {}
    for d in dates:
        hist = r.loc[:d].dropna().tail(63)
        vol = hist.std() * np.sqrt(252) if len(hist) >= 20 else np.nan
        e = 1.0 if not np.isfinite(vol) or vol <= 0 else min(1.0, VOL_TARGET / vol)
        nav = base_nav.loc[:d]
        dd = nav.iloc[-1] / nav.cummax().iloc[-1] - 1 if len(nav) else 0.0
        if dd <= -0.20:
            e = 0.0
        elif dd <= -0.15:
            e = -e          # negative means: keep holdings, add nothing new
        elif dd <= -0.10:
            e = 0.5 * e
        out[d] = e
    return out


def regime_table(nav, panel):
    """Net Sharpe and drawdown by volatility x trend state. Thresholds use only past values (expanding terciles)."""
    st = panel.drop_duplicates("date").set_index("date")[["market_vol_pctile", "taiex_trend"]].sort_index()
    st["vol_state"] = np.where(st["market_vol_pctile"] > 2 / 3, "high vol", np.where(st["market_vol_pctile"] < 1 / 3, "low vol", "mid vol"))
    st["trend_state"] = np.where(st["taiex_trend"] > 0, "uptrend", "downtrend")
    r = nav.pct_change().dropna()
    states = st.reindex(r.index, method="ffill").shift(1)
    rows = []
    for (v, t), idx in states.groupby(["vol_state", "trend_state"]).groups.items():
        x = r.reindex(idx).dropna()
        path = (1 + x).cumprod()
        rows.append(dict(vol=v, trend=t, days=len(x), ann_return=x.mean() * 252, sharpe=x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan, worst_drawdown=(path / path.cummax() - 1).min()))
    return pd.DataFrame(rows)


def main():
    panel = load_panel()
    scores = pd.read_parquet(PROC / "scores_val.parquet")
    deployed = (REPORTS / "deployed_model.txt").read_text(encoding="utf-8").strip()
    market = load_market()
    close = market["close"]
    vol = close.ffill(limit=5).pct_change(fill_method=None).rolling(63, min_periods=40).std()
    start, end = VAL_START, HOLD_START - pd.Timedelta(days=1)
    assert end < HOLD_START

    res, table = {}, []
    for name in ("rule", "ridge", "lgbm", "ensemble"):
        res[name] = simulate(decisions_for(panel, scores, name), market, start, end)
        m = metrics(res[name].nav, res[name].traded)
        m["gross_sharpe"] = metrics(res[name].gross_nav)["sharpe"]
        table.append(dict(strategy=name, **m))
        log_run("backtest_model", name, 1, "setup v1 equal weight", f"{start.date()}..{end.date()}", "net_sharpe", float(m["sharpe"]), "reported", "cost gate")
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, flush=True)
    base = res[deployed]

    universe = panel[panel["month_end"]].assign(score=0.0)[["date", "code", "score", "median_value", "buy_locked"]]
    universe = universe[universe["date"] >= scores["date"].min()]
    ew = simulate(universe, market, start, end, select="all")
    table.append(dict(strategy="equal_weight_universe", **metrics(ew.nav, ew.traded)))
    o50 = market["open"]["0050"].loc[start:end].dropna()
    first = base.nav.index[base.traded.gt(0).argmax()]
    bench = (o50 / o50.loc[first:].iloc[0]).loc[first:]
    table.append(dict(strategy="etf_0050", **metrics(bench)))
    table = pd.DataFrame(table)
    table.round(4).to_csv(REPORTS / "backtest_validation.csv", index=False)

    # Cost sweep, on the deployed strategy
    sweep = []
    dec = decisions_for(panel, scores, deployed)
    for mult in (0, 1, 2, 3, 4):
        s = simulate(dec, market, start, end, cost_mult=mult)
        m = metrics(s.nav.loc[first:], s.traded.loc[first:])
        sweep.append(dict(cost_multiple=mult, net_sharpe=m["sharpe"], net_cagr=m["cagr"], turnover=m["turnover_two_way"]))
        log_run("cost_sweep", deployed, mult, f"cost_mult={mult}", f"{start.date()}..{end.date()}", "net_sharpe", float(m["sharpe"]), "reported", "cost gate")
    sweep = pd.DataFrame(sweep)
    sweep.round(4).to_csv(REPORTS / "cost_sweep.csv", index=False)
    gross_cagr = sweep.loc[0, "net_cagr"]
    bench_m = metrics(bench)
    two_way = sweep.loc[1, "turnover"]
    breakeven_0050 = (gross_cagr - bench_m["cagr"]) / two_way * 1e4
    breakeven_zero = gross_cagr / two_way * 1e4

    # Allocators under matched inputs
    alloc = []
    inv = simulate(dec, market, start, end, allocator="inverse_vol", vol=vol)
    for name, s in (("equal", base), ("inverse_vol", inv)):
        m = metrics(s.nav.loc[first:], s.traded.loc[first:])
        hhi = np.mean([float((w ** 2).sum()) for w in s.weights.values() if len(w)])
        alloc.append(dict(allocator=name, sharpe=m["sharpe"], max_drawdown=m["max_drawdown"], turnover=m["turnover_two_way"], hhi=hhi, effective_names=1 / hhi))
        log_run("allocator", name, 1, f"same signal {deployed}", f"{start.date()}..{end.date()}", "net_sharpe", float(m["sharpe"]), "reported", "null is equal weight")
    pd.DataFrame(alloc).round(4).to_csv(REPORTS / "allocators.csv", index=False)

    # Overlay, with and without
    dates = sorted(dec["date"].unique())
    exp = overlay_exposure(base.nav, dates)
    over = simulate(dec, market, start, end, exposure=exp)
    om = metrics(over.nav.loc[first:], over.traded.loc[first:])
    bm = metrics(base.nav.loc[first:], base.traded.loc[first:])
    log_run("overlay", "vol_target_and_ladder", 1, f"vol target {VOL_TARGET}", f"{start.date()}..{end.date()}", "net_sharpe", float(om["sharpe"]), "reported", "with and without")

    # Stress: the same strategy with spreads three times wider, read over each episode
    stressed = simulate(dec, market, start, end, spread_mult=3)
    stress = []
    for name, (a, b) in EPISODES.items():
        for label, nav in (("base spreads", base.nav), ("spreads x3", stressed.nav), ("0050", bench)):
            x = nav.loc[a:b]
            if len(x) > 1:
                stress.append(dict(episode=name, case=label, ret=x.iloc[-1] / x.iloc[0] - 1, worst_drawdown=(x / x.cummax() - 1).min()))
    pd.DataFrame(stress).round(4).to_csv(REPORTS / "stress.csv", index=False)
    regime_table(base.nav.loc[first:], panel).round(4).to_csv(REPORTS / "regimes_validation.csv", index=False)

    # Inference
    variants = table[table["strategy"].isin(["rule", "ridge", "lgbm", "ensemble"])]["sharpe"].tolist() + [a["sharpe"] for a in alloc[1:]] + [om["sharpe"]]
    ledger = pd.read_csv(REPORTS.parent / "run_log.csv")
    # Reruns after bug fixes are not trials. Each LightGBM fold counts its Optuna trials.
    distinct = ledger[ledger["trial_family"] != "holdout"].drop_duplicates(["trial_family", "trial", "run"])
    gbm = distinct["trial_family"] == "model_lgbm"
    n_trials = int((~gbm).sum() + pd.to_numeric(distinct.loc[gbm, "run"]).sum())
    dsr = deflated_sharpe(bm["sharpe"], bm["years"], bm["skew"], bm["kurtosis"], n_trials, float(np.var(variants)))
    summary = dict(deployed=deployed, first_trade=str(first.date()), validation=bm, gross_sharpe=sweep.loc[0, "net_sharpe"], overlay=om, benchmark_0050=bench_m,
                   breakeven_bp_per_leg_vs_0050=breakeven_0050, breakeven_bp_per_leg_vs_zero=breakeven_zero, n_trials=n_trials, sharpe_variance=float(np.var(variants)), deflated_sharpe=dsr,
                   min_track_record_years=min_track_record_years(bm["sharpe"]), max_name_weight=base.max_name, max_sector_weight=base.max_sector, buys_skipped_locked=base.fills_skipped, sells_delayed=base.sells_delayed,
                   median_holdings=float(base.holdings.median()), sim_buys_never_negative_cash=True)
    (REPORTS / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    out = pd.DataFrame({"strategy": base.nav, "strategy_gross": base.gross_nav, "overlay": over.nav, "rule": res["rule"].nav, "equal_weight": ew.nav, "etf_0050": bench.reindex(base.nav.index)})
    out.to_parquet(PROC / "nav_validation.parquet")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(table[["strategy", "cagr", "vol", "sharpe", "sharpe_low", "sharpe_high", "max_drawdown", "turnover_two_way", "holding_months"]].round(3))
    print(sweep.round(3)); print(pd.DataFrame(alloc).round(3)); print(pd.DataFrame(stress).round(3)); print(json.dumps({k: v for k, v in summary.items() if k not in ("validation", "overlay", "benchmark_0050")}, indent=1, default=float))
    print("with overlay", round(om["sharpe"], 3), round(om["max_drawdown"], 3), "| without", round(bm["sharpe"], 3), round(bm["max_drawdown"], 3))


if __name__ == "__main__":
    main()
