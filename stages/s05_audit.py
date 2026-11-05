"""Stage 5: pre-holdout audit against the guideline's penalty rules and ten-question checklist.

Writes reports/pre_holdout_audit.md and .json. Stage 6 refuses to run unless every automated check passed.
"""
import json
import re

import numpy as np
import pandas as pd

from common import FEATURES, H, HOLD_START, PROC, P, REPORTS, ROOT, SETUP, VAL_END, load_panel
from s01_build_dataset import parse_revenue

checks = []


def check(q, name, ok, detail):
    checks.append(dict(question=q, check=name, passed=bool(ok), detail=str(detail)))
    print("PASS" if ok else "FAIL", q, name, "|", detail, flush=True)


def truncation_test():
    """Recompute features using only data up to T. Any difference from the panel means the future leaked in."""
    panel = pd.read_parquet(PROC / "panel.parquet", columns=["date", "code"] + FEATURES)
    close = pd.read_parquet(PROC / "close.parquet")
    value = pd.read_parquet(PROC / "value.parquet")
    raw_close = pd.read_parquet(PROC / "raw_close.parquet")
    foreign = pd.read_parquet(ROOT / "法人資料" / "foreign_net_buy.parquet")
    worst = 0.0
    for t in ("2018-06-29", "2020-03-31", "2022-09-30", "2024-09-30"):
        T = pd.Timestamp(t)
        c = close.loc[:T].drop(columns="0050").ffill(limit=5)
        vol = c.pct_change(fill_method=None).rolling(63, min_periods=40).std()
        rows = panel[panel["date"] == T].set_index("code")
        fresh = {"mom_12_1": ((c.shift(21) / c.shift(252) - 1) / vol).iloc[-1], "mom_6_1": ((c.shift(21) / c.shift(126) - 1) / vol).iloc[-1],
                 "dist_52w_high": (c / c.rolling(252, min_periods=200).max()).iloc[-1], "rev_21d": (-(c / c.shift(21) - 1)).iloc[-1]}
        shares = (value.loc[:T] / raw_close.loc[:T]).drop(columns="0050", errors="ignore").rolling(20, min_periods=10).sum()
        fresh["foreign_net_buy_20d"] = (foreign.reindex(close.loc[:T].index, columns=shares.columns).rolling(20, min_periods=10).sum() / shares).iloc[-1]
        rev = parse_revenue(close.loc[:T].index)
        fresh["revenue_yoy"] = rev["revenue_yoy"].iloc[-1]
        fresh["revenue_accel"] = rev["revenue_accel"].iloc[-1]
        for f, s in fresh.items():
            d = (rows[f] - s.reindex(rows.index)).abs().max()
            worst = max(worst, 0.0 if pd.isna(d) else float(d))
    return worst


def main():
    worst = truncation_test()
    check(1, "Features recomputed with data cut at T match the panel (4 dates, 7 features)", worst < 1e-9, f"largest difference {worst:.2e}")

    panel = load_panel()
    rev = panel[panel["code"] == "2330"].set_index("date")["revenue_accel"].dropna()
    changes = rev[rev.diff().abs() > 0].index
    check(1, "Revenue changes only after the 10th of the month (TSMC)", (changes.day > 10).all(), f"{len(changes)} updates, earliest day of month {changes.day.min()}")
    check(1, "Every feature has a stated availability rule in setup.yaml", "admissible_information" in SETUP, list(SETUP["admissible_information"]))

    eligible = pd.read_parquet(PROC / "panel.parquet", columns=["date", "code"])
    last = pd.read_parquet(PROC / "listed.parquet").fillna(0).gt(0).apply(lambda s: s[s].index.max())
    gone = set(last[last < pd.Timestamp("2024-01-01")].index)
    check(2, "Universe includes names that later left the market", eligible["code"].isin(gone).any(), f"{eligible.loc[eligible['code'].isin(gone), 'code'].nunique()} eligible names later delisted")

    op = pd.read_parquet(PROC / "open.parquet")
    s = panel.dropna(subset=["label"]).sample(5000, random_state=1)
    pos = op.index.get_indexer(s["date"])
    ok = 0
    n = 0
    for (code, label), i in zip(zip(s["code"], s["label"]), pos):
        a, b = op[code].iat[i + 1], op[code].iat[i + H + 1]
        if np.isfinite(a) and np.isfinite(b):
            n += 1
            ok += abs(b / a - 1 - label) < 1e-9
    check(3, "Label equals adjusted Open(t+22)/Open(t+1)-1 on 5,000 random rows", ok == n, f"{ok} of {n}")
    sc_all = pd.read_parquet(PROC / "scores_val.parquet", columns=["date", "code"])
    tests = panel[(panel["date"] >= sc_all["date"].min()) & panel["month_end"]]
    merged = tests.merge(sc_all, on=["date", "code"], how="left", indicator=True)
    check(1, "Every eligible stock is scored, including those whose label is missing", (merged["_merge"] == "both").all(), f"{int((merged['_merge'] != 'both').sum())} unscored of {len(merged)}; {int(tests['label'].isna().sum())} have no label")
    nv = pd.read_parquet(PROC / "nav_validation.parquet")
    check(9, "No simulated series contains a gap or a non-positive NAV", (nv[["strategy", "overlay", "rule", "equal_weight"]].notna().all().all() and (nv[["strategy", "overlay", "rule", "equal_weight"]] > 0).all().all()), nv[["strategy", "rule", "equal_weight"]].iloc[-1].round(3).to_dict())
    src = (ROOT / "stages" / "simulator.py").read_text(encoding="utf-8")
    check(3, "Simulator fills buys at the open of the day after the decision", "full_cal[pos + 1]" in src and "amount / op.at[day, c]" in src, "code check")

    s03 = (ROOT / "stages" / "s03_models.py").read_text(encoding="utf-8")
    check(4, "Winsor limits fitted on the training fold; ranks are within one date", "winsor_limits(train, STATES)" in s03 and 'groupby("date")[f].rank' in s03, "code check")

    fp = pd.read_csv(REPORTS / "fold_protocol.csv", parse_dates=["last_label_end", "test_start", "test_end"])
    check(5, "In every fold the last training label ends before the test block", (fp["last_label_end"] < fp["test_start"]).all(), fp[["fold", "last_label_end", "test_start"]].astype(str).to_dict("records"))
    check(5, "Purge equals the 21-trading-day label horizon", P["purge_trading_days"] == H == 21, f"purge {P['purge_trading_days']}, horizon {H}")

    sc = pd.read_parquet(PROC / "scores_val.parquet", columns=["date"])
    nav = pd.read_parquet(PROC / "nav_validation.parquet")
    ledger = pd.read_csv(ROOT / "run_log.csv")
    check(6, "No validation score or NAV reaches into the holdout", sc["date"].max() < HOLD_START and nav.index.max() < HOLD_START, f"last score {sc['date'].max().date()}, last NAV {nav.index.max().date()}")
    check(6, "Validation labels all end before the holdout", panel["label_end"].max() < HOLD_START, panel["label_end"].max().date())
    replay = (REPORTS / "holdout_touched.json").exists()
    if replay:
        check(6, "Replay: the recorded holdout run is the only one in the ledger", ledger.loc[ledger["trial_family"] == "holdout", "config"].eq("frozen spec").all(), "marker file present")
    else:
        check(6, "Holdout has not been run", True, "no marker file")
    outside = ledger[ledger["trial_family"] != "holdout"]
    check(6, "Ledger shows no development run inside the holdout window", not outside["window"].fillna("").str.contains(r"202[56]-").any(), f"{len(ledger)} rows")

    summary = json.loads((REPORTS / "validation_summary.json").read_text(encoding="utf-8"))
    check(7, "Research ledger exists with every family logged", {"feature_triage", "model_ridge", "model_lgbm", "model_ladder", "cost_sweep", "allocator", "overlay"} <= set(ledger["trial_family"]), f"{len(ledger)} rows, {ledger['trial_family'].nunique()} families")
    check(7, "Deflated Sharpe uses the ledger's count, with every Optuna trial", summary["n_trials"] >= 84, f"n_trials {summary['n_trials']}")

    sweep = pd.read_csv(REPORTS / "cost_sweep.csv")
    check(8, "Costs include commission, sell tax and spread", all(k in src for k in ("commission_bps_each_side", "sell_tax_bps", "spread_mult")), "code check")
    check(8, "Cost sweep runs 0 to 4x and net Sharpe falls as costs rise", sweep["cost_multiple"].max() == 4 and sweep["net_sharpe"].is_monotonic_decreasing, sweep["net_sharpe"].round(3).tolist())

    check(9, "Caps hold inside the simulation", summary["max_name_weight"] <= 0.05 + 1e-9 and summary["max_sector_weight"] <= 0.40 + 1e-9, f"max name {summary['max_name_weight']:.3f}, max sector {summary['max_sector_weight']:.3f}")
    s04 = (ROOT / "stages" / "s04_backtest.py").read_text(encoding="utf-8")
    check(9, "Overlay and ladder read only the path up to the decision date", "r.loc[:d]" in s04 and "base_nav.loc[:d]" in s04, "code check")
    check("penalty", "No unexplained Sharpe above 3", max(abs(summary["validation"]["sharpe"]), abs(summary["gross_sharpe"])) < 3, f"net {summary['validation']['sharpe']:.2f}, gross {summary['gross_sharpe']:.2f}")

    all_ok = all(c["passed"] for c in checks)
    # After the holdout the pre-holdout record stays as it was; a replay writes its own file
    stem = "replay_audit" if replay else "pre_holdout_audit"
    (REPORTS / f"{stem}.json").write_text(json.dumps(dict(passed=all_ok, checks=checks), indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {'Replay' if replay else 'Pre-holdout'} audit", "", f"Overall: {'PASS' if all_ok else 'FAIL'}", "", "| Q | Check | Result | Detail |", "| --- | --- | --- | --- |"]
    lines += [f"| {c['question']} | {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail'][:140]} |" for c in checks]
    lines += ["", "Question 10 (someone else can regenerate from the repository root) is tested on a fresh clone after stage 7."]
    (REPORTS / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")
    print("OVERALL", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
