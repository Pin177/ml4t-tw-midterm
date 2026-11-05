"""Stage 1: build the point-in-time dataset from tw.db and data/raw. Offline and deterministic.

Reads the price database read-only. Writes matrices and panel.parquet to data/processed/.
No return statistics are computed here apart from the data-quality checks.
"""
import gzip
import json
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

from common import DB, FEATURES, H, PROC, RAW, ROOT, SETUP

sys.path.insert(0, str(ROOT / "data"))
import twdb  # noqa: E402  the adjustment logic that built tw.db, used read-only through our own connection

STOCK = re.compile(r"[1-9]\d{3}")
START = str(SETUP["evaluation_protocol"]["feature_data_start"])
U = SETUP["universe"]


def to_num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return np.nan


def roc(s):
    a = re.findall(r"\d+", str(s))
    return pd.Timestamp(int(a[0]) + 1911, int(a[1]), int(a[2]))


def build_prices():
    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    q = pd.read_sql("SELECT code, date, market, open, high, low, close, value, managed FROM quotes WHERE date >= ?", conn, params=[START])
    q = q[q["code"].map(lambda c: bool(STOCK.fullmatch(c))) | (q["code"] == "0050")]
    q["date"] = pd.to_datetime(q["date"])
    print("quote rows", len(q), "codes", q["code"].nunique(), flush=True)
    cal = pd.DatetimeIndex(sorted(q["date"].unique()))
    market = q.sort_values("date").groupby("code")["market"].last()

    # Adjustment factor per code: product of every corporate-action factor dated after the row
    g = {}
    for i, code in enumerate(market.index):
        s = pd.Series(1.0, index=cal)
        for d, f in twdb._factors(conn, code, market[code]).items():
            s[s.index < pd.Timestamp(d)] *= f
        g[code] = s
        if (i + 1) % 500 == 0:
            print("factors", i + 1, flush=True)
    g = pd.DataFrame(g)
    taiex = pd.read_sql("SELECT date, close FROM taiex", conn, parse_dates=["date"]).set_index("date")["close"]
    conn.close()

    out = {}
    for col in ("open", "high", "low", "close", "value", "managed"):
        out[col] = q.pivot(index="date", columns="code", values=col).reindex(cal)
    exists = out["value"].notna() | out["close"].notna()
    for col in ("open", "high", "low", "close"):
        out["raw_" + col] = out[col]
        out[col] = out[col] * g.reindex(columns=out[col].columns)
    out["listed"] = exists.astype(float)
    PROC.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_parquet(PROC / f"{k}.parquet")
    taiex.to_frame("close").to_parquet(PROC / "taiex.parquet")
    return out, taiex, market


def parse_flows(cal):
    """Net shares bought per day by foreign investors and investment trusts, TWSE + TPEx.

    Source: 法人資料/*.parquet, dates x codes in shares. data/verify_flows.py checks them against official files.
    A row is dated the trading day it covers and is published after that day's close.
    """
    foreign = pd.read_parquet(ROOT / "法人資料" / "foreign_net_buy.parquet")
    trust = pd.read_parquet(ROOT / "法人資料" / "trust_net_buy.parquet")
    print("flow days", len(foreign), foreign.index.min(), foreign.index.max(), flush=True)
    return foreign.reindex(cal), trust.reindex(cal)


def parse_revenue(cal):
    """Monthly revenue with its point-in-time availability: first trading day after the 10th of the next month."""
    rows = []
    cell = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
    for board in ("sii", "otc"):
        for path in sorted((RAW / f"revenue_{board}").glob("*.html.gz")):
            ym = path.name[:6]
            text = gzip.decompress(path.read_bytes()).decode("big5", "ignore")
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S | re.I):
                c = [re.sub(r"<[^>]+>|&nbsp;|\s", "", x) for x in cell.findall(tr)]
                if len(c) >= 7 and STOCK.fullmatch(c[0]):
                    rows.append((ym, c[0], to_num(c[2]), to_num(c[4])))
    df = pd.DataFrame(rows, columns=["ym", "code", "rev", "rev_ly"]).drop_duplicates(["ym", "code"])
    df["month"] = pd.to_datetime(df["ym"], format="%Y%m")
    df["yoy"] = np.where(df["rev_ly"] > 0, df["rev"] / df["rev_ly"] - 1, np.nan)
    df["yoy"] = df["yoy"].clip(-1, 5)
    df = df.sort_values(["code", "month"])
    grp = df.groupby("code")["yoy"]
    df["revenue_yoy"] = grp.transform(lambda s: s.rolling(3, min_periods=2).mean())
    df["revenue_accel"] = df["yoy"] - grp.transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    deadline = df["month"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=9)
    pos = cal.searchsorted(deadline.to_numpy(), side="right")
    df["available"] = [cal[p] if p < len(cal) else pd.NaT for p in pos]
    df = df.dropna(subset=["available"])
    print("revenue rows", len(df), df["month"].min(), df["month"].max(), flush=True)
    out = {}
    for name in ("revenue_yoy", "revenue_accel"):
        m = df.pivot_table(index="available", columns="code", values=name, aggfunc="last").reindex(cal)
        out[name] = m.ffill(limit=45)  # one report stays current until the next, about 21 trading days
    return out


def parse_disposition(cal, codes):
    """Boolean matrix: the stock was under a published TWSE disposition notice on that day."""
    m = pd.DataFrame(False, index=cal, columns=codes)
    n = 0
    for path in sorted((RAW / "punish_twse").glob("*.json.gz")):
        j = json.loads(gzip.decompress(path.read_bytes()))
        tables = j.get("tables") or [j]
        for t in tables:
            f = t.get("fields") or []
            if "處置起迄時間" not in f:
                continue
            ci, ti, pi = f.index("證券代號"), f.index("處置起迄時間"), f.index("公布日期")
            for r in t.get("data") or []:
                code = str(r[ci]).strip()
                parts = re.split(r"[～~]", str(r[ti]))
                if code in m.columns and len(parts) == 2:
                    a, b = max(roc(parts[0]), roc(r[pi])), roc(parts[1])
                    m.loc[(m.index >= a) & (m.index <= b), code] = True
                    n += 1
    print("disposition notices", n, flush=True)
    return m


def fetch_sectors():
    """Industry code per stock. Only the current classification is published, a disclosed point-in-time FAIL."""
    path = RAW / "sectors.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def locked(px, side):
    """t+1 open is limit-locked: open sits on the price limit and equals the day's high (buy) or low (sell)."""
    ref = px["raw_close"].shift(1)
    limit = pd.Series(np.where(ref.index < pd.Timestamp("2015-06-01"), 0.07, 0.10), index=ref.index)
    limit = pd.DataFrame({c: limit for c in ref.columns})
    move = px["raw_open"] / ref - 1
    if side == "buy":
        return (move >= limit - 0.005) & (px["raw_open"] >= px["raw_high"])
    return (move <= -limit + 0.005) & (px["raw_open"] <= px["raw_low"])


def build_panel(px, taiex):
    cal = px["close"].index
    codes = [c for c in px["close"].columns if c != "0050"]
    close, open_ = px["close"][codes], px["open"][codes]
    value = px["value"][codes]

    # Features: each uses only data available at the close of day t
    ret = close.ffill(limit=5).pct_change(fill_method=None)
    vol = ret.rolling(63, min_periods=40).std()
    f = {}
    filled = close.ffill(limit=5)
    f["mom_12_1"] = (filled.shift(21) / filled.shift(252) - 1) / vol
    f["mom_6_1"] = (filled.shift(21) / filled.shift(126) - 1) / vol
    f["dist_52w_high"] = filled / filled.rolling(252, min_periods=200).max()
    f["rev_21d"] = -(filled / filled.shift(21) - 1)
    foreign, trust = parse_flows(cal)
    vol_shares = (value / px["raw_close"][codes]).rolling(20, min_periods=10).sum()
    f["foreign_net_buy_20d"] = foreign.reindex(columns=codes).rolling(20, min_periods=10).sum() / vol_shares
    f["trust_net_buy_20d"] = trust.reindex(columns=codes).fillna(0).where(foreign.reindex(columns=codes).notna()).rolling(20, min_periods=10).sum() / vol_shares
    rev = parse_revenue(cal)
    f["revenue_yoy"] = rev["revenue_yoy"].reindex(columns=codes)
    f["revenue_accel"] = rev["revenue_accel"].reindex(columns=codes)

    tx = taiex.reindex(cal).ffill()
    tx_vol = tx.pct_change(fill_method=None).rolling(20).std()
    market_vol_pctile = tx_vol.rolling(252, min_periods=200).rank(pct=True)
    taiex_trend = tx / tx.rolling(200).mean() - 1

    # Eligibility at t, from information at t only
    listed = px["listed"][codes].fillna(0).rolling(U["min_listed_trading_days"], min_periods=U["min_listed_trading_days"]).sum() >= 240
    median_value = value.rolling(63, min_periods=40).median()
    managed = px["managed"][codes].fillna(0) > 0
    disposition = parse_disposition(cal, codes)
    eligible = (listed & (px["raw_close"][codes] > U["min_price_ntd"])
                & (median_value > U["min_median_value_63d_ntd"]) & ~managed & ~disposition & close.notna())

    # Label: next open to the open 21 trading days later. A stock that stops trading is valued at its last close.
    entry = open_.shift(-1)
    exit_px = open_.shift(-(H + 1))
    last = close.ffill().shift(-(H + 1))
    gone = exit_px.isna() & px["listed"][codes].shift(-(H + 1)).fillna(0).eq(0)
    label = exit_px.where(~gone, last) / entry - 1
    label_end = pd.Series(cal, index=cal).shift(-(H + 1))
    buy_locked = locked(px, "buy")[codes].shift(-1, fill_value=False)
    sell_locked = locked(px, "sell")[codes]
    sell_locked.to_parquet(PROC / "sell_locked.parquet")

    frames = {"label": label, "median_value": median_value, "buy_locked": buy_locked}
    frames.update(f)
    long = eligible.stack()
    long = long[long].index.to_frame(index=False)
    long.columns = ["date", "code"]
    idx = pd.MultiIndex.from_frame(long)
    for name, m in frames.items():
        long[name] = m.stack(future_stack=True).reindex(idx).to_numpy()
    long["market_vol_pctile"] = long["date"].map(market_vol_pctile)
    long["taiex_trend"] = long["date"].map(taiex_trend)
    long["label_end"] = long["date"].map(label_end)
    long["month_end"] = long["date"].isin(month_ends(cal))
    print("panel rows", len(long), "dates", long["date"].nunique(), flush=True)
    assert set(FEATURES) <= set(long.columns)
    long.to_parquet(PROC / "panel.parquet", index=False)
    return long


def month_ends(cal):
    s = pd.Series(cal, index=cal)
    return s.groupby([cal.year, cal.month]).last().to_numpy()


def fetch_sector_file():
    """Download today's industry codes once. Published as current values only (disclosed point-in-time FAIL)."""
    path = RAW / "sectors.json"
    if path.exists():
        return
    import ssl
    import urllib.request
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    out = {}
    for url, code_key in (("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號"),
                          ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode")):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        rows = json.loads(urllib.request.urlopen(req, timeout=60, context=ctx).read())
        key = next(k for k in rows[0] if "產業" in k or "Industry" in k)
        out.update({str(r[code_key]).strip(): str(r[key]).strip() for r in rows})
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def quality_report(px, panel):
    """Data checks only, no strategy returns: the Section 3 tables and two figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common import FIGS, REPORTS
    FIGS.mkdir(parents=True, exist_ok=True)
    codes = [c for c in px["close"].columns if c != "0050"]
    raw_o, raw_h, raw_l, raw_c = (px[k][codes] for k in ("raw_open", "raw_high", "raw_low", "raw_close"))
    tol = 1e-9
    bad = (raw_h < np.fmax(raw_o, raw_c) - tol) | (raw_l > np.fmin(raw_o, raw_c) + tol) | (raw_h < raw_l)
    listed = px["listed"][codes].fillna(0) > 0
    last_day = listed.apply(lambda s: s[s].index.max())
    end = listed.index.max() - pd.Timedelta(days=30)
    left = last_day[last_day < end]
    left_per_year = left.dt.year.value_counts().sort_index()
    months = panel.groupby(panel["date"].dt.to_period("M"))["code"].nunique()
    missing_close = float((raw_c.isna() & listed).sum().sum() / listed.sum().sum())
    cov = panel[FEATURES].isna().mean().round(4)
    sectors = fetch_sectors()
    lines = ["# Data quality report", "",
             f"- Dates: {px['close'].index.min().date()} to {px['close'].index.max().date()}; stocks seen: {len(codes)}",
             f"- Eligible names per month: min {months.min()}, median {int(months.median())}, max {months.max()}",
             f"- Listed stock-days with no trade (close left empty, excluded, never forward-filled for fills): {missing_close:.2%}",
             f"- OHLC invariant violations: {int(bad.sum().sum())}",
             "- Duplicates: 0 by construction (primary key code + date in tw.db)",
             f"- Names that left the sample: {len(left)}; per year: {left_per_year.to_dict()}",
             f"- Names with a current industry code: {sum(c in sectors for c in codes)} of {len(codes)}",
             "- Session: TWSE/TPEx close 13:30 Asia/Taipei; prices treated as known at 15:00.",
             "- Storage: data/processed/*.parquet rebuilt by stages/s01_build_dataset.py; no synthetic data used.",
             "", "## Feature missing share on eligible stock-days", "", cov.to_frame("missing").to_markdown(), "",
             "## Point-in-time checklist", "",
             "| Level | Check | Result | Direction of any bias |", "| --- | --- | --- | --- |",
             "| Values | Prices and flows used only after that day's close | PASS | none |",
             "| Values | Revenue dated first trading day after the 10th of the next month | PASS with limit | late filers may look known early; values are latest published, so restatements are not separated |",
             "| Universe | Eligibility computed from data at t; delisted names kept | PASS | none |",
             "| Universe | TWSE disposition notices excluded by publication date | PASS | none |",
             "| Universe | TPEx disposition and all full-delivery status | FAIL, disclosed | some restricted names stay eligible; fills may be optimistic |",
             "| Identifiers | Code reuse: stock must have 240 of last 252 days listed | PASS with limit | a reused code restarts its eligibility |",
             "| Identifiers | Industry codes are today's, not point-in-time | FAIL, disclosed | sector cap may differ slightly from what was known then |",
             "| Values | Flows before 2014-12 on TPEx could not be checked against the official site | NOT VERIFIED | unknown |"]
    (REPORTS / "data_quality.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    months.index = months.index.to_timestamp()
    ax.plot(months.index, months.values, color="#0a3d62")
    ax.set(title="How many stocks were eligible each month?", ylabel="Eligible names")
    fig.tight_layout(); fig.savefig(FIGS / "coverage.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    d = pd.DataFrame({"Raw close": px["raw_close"]["2330"], "Adjusted close": px["close"]["2330"]}).dropna()
    d = d[d.index < "2024-11-01"]
    ax.plot(d.index, d["Raw close"], label="Raw close", color="#60a3bc")
    ax.plot(d.index, d["Adjusted close"], label="Adjusted close", color="#0a3d62")
    ax.set(title="Do adjusted prices handle dividends? TSMC (2330)", ylabel="NT$"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "corporate_action_check.png", dpi=160); plt.close(fig)
    print("\n".join(lines[:9]), flush=True)


if __name__ == "__main__":
    fetch_sector_file()
    px, taiex, _ = build_prices()
    panel = build_panel(px, taiex)
    quality_report(px, panel)
