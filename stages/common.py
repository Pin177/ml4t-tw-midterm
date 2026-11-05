"""Shared paths, protocol dates, the holdout guard, and small statistics helpers."""
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
DB = Path(os.environ.get("TWDB_PATH", ROOT.parent / "db" / "tw.db"))
SETUP = yaml.safe_load((ROOT / "setup.yaml").read_text(encoding="utf-8"))

P = SETUP["evaluation_protocol"]
TRAIN_START = pd.Timestamp(P["training_start"])
VAL_START = pd.Timestamp(P["validation_start"])
VAL_END = pd.Timestamp(P["validation_end"])
HOLD_START = pd.Timestamp(SETUP["holdout"]["start"])
HOLD_END = pd.Timestamp(SETUP["holdout"]["end"])
H = P["label_horizon_trading_days"]
SEED = SETUP["seed"]

SIGNALS = ["mom_12_1", "mom_6_1", "dist_52w_high", "rev_21d", "foreign_net_buy_20d",
           "trust_net_buy_20d", "revenue_yoy", "revenue_accel"]
STATES = ["market_vol_pctile", "taiex_trend"]
FEATURES = SIGNALS + STATES


def load_panel(allow_holdout=False):
    """Eligible stock-days. Without allow_holdout, nothing whose label ends in the holdout is returned."""
    df = pd.read_parquet(PROC / "panel.parquet")
    if not allow_holdout:
        df = df[df["label_end"].notna() & (df["label_end"] < HOLD_START) & (df["date"] <= VAL_END)]
    return df


def load_matrix(name, allow_holdout=False):
    """A dates x codes price matrix. Without allow_holdout it stops the day before the holdout."""
    m = pd.read_parquet(PROC / f"{name}.parquet")
    return m if allow_holdout else m[m.index < HOLD_START]


def folds():
    """Annual walk-forward test blocks over the validation period."""
    out = []
    y = VAL_START.year
    while pd.Timestamp(y, 1, 1) <= VAL_END:
        out.append((pd.Timestamp(y, 1, 1), min(pd.Timestamp(y, 12, 31), VAL_END)))
        y += 1
    return out


def purged_train(df, test_start, calendar):
    """Training rows whose 21-day label ends before the test block (purge = label horizon)."""
    pos = calendar.searchsorted(test_start)
    cutoff = calendar[max(pos - P["purge_trading_days"], 0)]
    return df[(df["date"] >= TRAIN_START) & (df["date"] < cutoff)]


def nw_mean(series, lags=None):
    """Mean of a series with a Newey-West 95% CI. Returns mean, low, high, se, n."""
    x = pd.Series(series).dropna().to_numpy()
    if len(x) < 3:
        return dict(mean=np.nan, low=np.nan, high=np.nan, se=np.nan, n=len(x))
    lags = H if lags is None else lags
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(x) - 1)})
    low, high = fit.conf_int()[0]
    return dict(mean=float(fit.params[0]), low=float(low), high=float(high), se=float(fit.bse[0]), n=len(x))


def daily_ic(df, score, label="label"):
    """Spearman IC per decision date: rank the score and the label across stocks, then correlate."""
    d = df[["date", score, label]].dropna()
    ranks = d.groupby("date")[[score, label]].rank()
    ranks["date"] = d["date"].to_numpy()
    return ranks.groupby("date").apply(lambda g: g[score].corr(g[label]) if len(g) >= 30 else np.nan, include_groups=False).dropna()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_run(trial_family, trial, run, config, window, metric_name, metric_value, decision, gate, note=""):
    """Append one row to the research ledger. Every adaptive choice passes through here."""
    with open(ROOT / "run_log.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([SETUP["strategy"], trial_family, trial, run, datetime.now().isoformat(timespec="seconds"),
                                git_commit(), "tw.db@2026-11-05", config, window, metric_name,
                                f"{metric_value:.6f}" if isinstance(metric_value, float) else metric_value, decision, gate, note])
