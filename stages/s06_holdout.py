"""Stage 6: the single holdout run. Refuses to start if the audit failed, the repo is dirty, or it has already run.

Protocol fixed in advance:
- Hyperparameters are frozen at the last validation fold's values. Nothing is tuned here.
- Annual retrain as in setup: one model from the 2024-10-31 decision, one from the first decision after 2025-10-31.
  Each trains only on labels that ended before its first decision.
- First decision 2024-10-31 (fills 2024-11-01). Last decision 2026-09-30.
"""
import json
import subprocess
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from common import (H, HOLD_END, HOLD_START, P, PROC, REPORTS, ROOT, SEED, SIGNALS, STATES, TRAIN_START, daily_ic, git_commit, load_panel, log_run, nw_mean)
from s03_models import TRAIN_EVERY_N_DAYS, add_ranks, apply_limits, lgb_params, thin, winsor_limits
from s04_backtest import decisions_for, overlay_exposure, regime_table
from simulator import deflated_sharpe, load_market, metrics, min_track_record_years, simulate

MARKER = REPORTS / "holdout_touched.json"


def guard():
    audit = json.loads((REPORTS / "pre_holdout_audit.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise SystemExit("Audit did not pass. Holdout not started.")
    if "--reproduce" in sys.argv:
        # Reproducibility test only: same frozen spec and code, no adaptive choice. Does not change the marker.
        print("Reproducing the recorded holdout run", flush=True)
        return
    if MARKER.exists():
        raise SystemExit(f"Holdout already run: {MARKER.read_text(encoding='utf-8')}")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "stages", "setup.yaml"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit(f"Commit the code first; the run must point to a commit:\n{dirty}")
    MARKER.write_text(json.dumps(dict(started=datetime.now().isoformat(timespec="seconds"), commit=git_commit())), encoding="utf-8")


def frozen():
    ledger = pd.read_csv(ROOT / "run_log.csv")
    alpha = float(ledger[(ledger["trial_family"] == "model_ridge") & (ledger["trial"] == "fold_2024")]["config"].iloc[-1].split("=")[1])
    p = json.loads(ledger[(ledger["trial_family"] == "model_lgbm") & (ledger["trial"] == "fold_2024")]["config"].iloc[-1])
    rounds = p.pop("rounds")
    signals = json.loads(ledger[ledger["trial_family"] == "feature_set"]["config"].iloc[-1])
    return alpha, p, rounds, signals


def main():
    guard()
    alpha, params, rounds, signals = frozen()
    deployed = (REPORTS / "deployed_model.txt").read_text(encoding="utf-8").strip()
    panel = load_panel(allow_holdout=True)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel = add_ranks(panel, SIGNALS)
    feats = signals + STATES
    pos0 = calendar[calendar < HOLD_START][-1]
    retrain = calendar[calendar > pd.Timestamp("2025-10-31")][0]
    scored = []
    for a, b in ((pos0, retrain - pd.Timedelta(days=1)), (retrain, HOLD_END)):
        tr = panel[(panel["date"] >= TRAIN_START) & (panel["date"] < a) & (panel["label_end"] < a)].dropna(subset=["label"])
        assert tr["label_end"].max() < a
        te = panel[(panel["date"] >= a) & (panel["date"] <= b)].copy()
        limits = winsor_limits(tr, STATES)
        tr, te = apply_limits(tr, limits), apply_limits(te, limits)
        tr = thin(tr, calendar)
        te["ridge"] = Ridge(alpha=alpha).fit(tr[signals].fillna(0), tr["y"]).predict(te[signals].fillna(0))
        te["lgbm"] = lgb.train(lgb_params(params), lgb.Dataset(tr[feats], tr["y"]), max(rounds, 10)).predict(te[feats])
        te["rule"] = te["mom_12_1"]
        z = lambda s: s.groupby(te["date"]).transform(lambda x: (x - x.mean()) / x.std())
        te["ensemble"] = (z(te["rule"]) + z(te["ridge"]) + z(te["lgbm"])) / 3
        scored.append(te[["date", "code", "label", "rule", "ridge", "lgbm", "ensemble", "market_vol_pctile", "taiex_trend"]])
    s = pd.concat(scored)
    s.to_parquet(PROC / "scores_holdout.parquet", index=False)

    ic_rows = {}
    eval_rows = s[s["date"] >= HOLD_START].dropna(subset=["label"])
    for m in ("rule", "ridge", "lgbm", "ensemble"):
        ic = daily_ic(eval_rows, m)
        ic_rows[m] = {**nw_mean(ic), "icir": float(ic.mean() / ic.std())}
        ic.to_frame("ic").to_parquet(PROC / f"ic_holdout_{m}.parquet")

    market = load_market()
    vol = market["close"].ffill(limit=5).pct_change(fill_method=None).rolling(63, min_periods=40).std()
    start, end = pos0, HOLD_END
    mpanel = panel[panel["date"] >= pos0]
    dec = decisions_for(mpanel, s, deployed)
    rule_dec = decisions_for(mpanel, s, "rule")
    base = simulate(dec, market, start, end)
    rule = simulate(rule_dec, market, start, end)
    first = base.nav.index[base.traded.gt(0).argmax()]
    universe = mpanel[mpanel["month_end"]].assign(score=0.0)[["date", "code", "score", "median_value", "buy_locked"]]
    ew = simulate(universe, market, start, end, select="all")
    o50 = market["open"]["0050"].loc[first:end].dropna()
    bench = o50 / o50.iloc[0]
    exp = overlay_exposure(base.nav, sorted(dec["date"].unique()))
    over = simulate(dec, market, start, end, exposure=exp)
    inv = simulate(dec, market, start, end, allocator="inverse_vol", vol=vol)
    gross = simulate(dec, market, start, end, cost_mult=0)
    sweep = []
    for mult in (0, 1, 2, 3, 4):
        x = gross if mult == 0 else base if mult == 1 else simulate(dec, market, start, end, cost_mult=mult)
        m = metrics(x.nav.loc[first:], x.traded.loc[first:])
        sweep.append(dict(cost_multiple=mult, net_sharpe=m["sharpe"], net_cagr=m["cagr"], turnover=m["turnover_two_way"]))
    pd.DataFrame(sweep).round(4).to_csv(REPORTS / "cost_sweep_holdout.csv", index=False)

    M = {k: metrics(v.nav.loc[first:], v.traded.loc[first:]) for k, v in dict(strategy=base, rule=rule, equal_weight=ew, overlay=over, inverse_vol=inv).items()}
    M["etf_0050"] = metrics(bench)
    gross_m = metrics(gross.nav.loc[first:], gross.traded.loc[first:])
    val = json.loads((REPORTS / "validation_summary.json").read_text(encoding="utf-8"))
    dsr = deflated_sharpe(M["strategy"]["sharpe"], M["strategy"]["years"], M["strategy"]["skew"], M["strategy"]["kurtosis"], val["n_trials"], val["sharpe_variance"])
    ic_low = ic_rows[deployed]["low"]
    passed = bool(ic_low > 0 and M["strategy"]["sharpe"] >= M["rule"]["sharpe"])
    out = dict(deployed=deployed, frozen=dict(alpha=alpha, lgbm=params, rounds=rounds, signals=signals), first_fill=str(first.date()), last_day=str(base.nav.index[-1].date()),
               ic=ic_rows, metrics=M, gross=gross_m, breakeven_bp_per_leg_vs_0050=(gross_m["cagr"] - M["etf_0050"]["cagr"]) / gross_m["turnover_two_way"] * 1e4,
               breakeven_bp_per_leg_vs_zero=gross_m["cagr"] / gross_m["turnover_two_way"] * 1e4,
               deflated_sharpe=dsr, n_trials=val["n_trials"], min_track_record_years=min_track_record_years(M["strategy"]["sharpe"]),
               confirmatory=dict(ic_low=ic_low, net_sharpe=M["strategy"]["sharpe"], baseline_net_sharpe=M["rule"]["sharpe"], passed=passed),
               max_name_weight=base.max_name, max_sector_weight=base.max_sector, buys_skipped_locked=base.fills_skipped, sells_delayed=base.sells_delayed, median_holdings=float(base.holdings.median()))
    (REPORTS / "holdout_results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.DataFrame({"strategy": base.nav, "strategy_gross": gross.nav, "overlay": over.nav, "rule": rule.nav, "equal_weight": ew.nav, "etf_0050": bench.reindex(base.nav.index)}).to_parquet(PROC / "nav_holdout.parquet")
    regime_table(base.nav.loc[first:], panel).round(4).to_csv(REPORTS / "regimes_holdout.csv", index=False)
    log_run("holdout", deployed, 1, "frozen spec", f"{first.date()}..{end.date()}", "net_sharpe", float(M["strategy"]["sharpe"]), "confirmatory pass" if passed else "confirmatory fail", "single touch")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
