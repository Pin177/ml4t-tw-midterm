"""台股（上市＋上櫃）官方日線資料庫：證交所 TWSE、櫃買中心 TPEx 公開資料，不需 API key

資料來源與起始日:
- 上市日行情 MI_INDEX                 2004-02-11 起（不含權證）
- 上櫃日行情 dailyQuotes              2007-07-02 起（含次日參考價）
- 上市除權息 TWT49U                   2003-05-05 起
- 上市減資 TWTAUU                     2011-01-01 起（更早的減資只能靠品質檢查發現）
- 上市面額變更 TWTB8U
- 上櫃除權息 exDailyQ、上櫃減資 revivt
- 加權指數 MI_5MINS_HIST              1999-01 起
- 終止上市清單 suspendListing

用法:
    python twdb.py sync        # 下載缺少的資料；第一次約 5 小時，可中斷，重跑會接續
    python twdb.py rebuild     # 從 raw 快照重新解析整個資料庫（不連網）
    python twdb.py check       # 品質檢查，明細寫入 tw_qc_report.csv
    python twdb.py status

資料原則（依 ML4T 第 2 章）:
- quotes 存交易所公布的原始價格，未還原；當天沒成交就是 NULL，不補值。
- 公司行動存在 corp_actions；還原價在 load() 讀取時以「參考價 / 前收盤」計算。
  上櫃用每日行情的「次日參考價」推算，涵蓋除權息、減資、面額變更等所有事件。
- 每個回應原樣壓縮存到 raw/tw/，downloads 記錄抓取時間與雜湊值，資料庫可完全由 raw 重建。
- 股票池用當天實際有行情的證券（含日後下市者），避免存活者偏誤。
- 日行情的可得時間定為交易日 15:00（台北時間），load() 會附上 available_at 欄位。
"""
import bisect
import gzip
import hashlib
import json
import os
import random
import re
import sqlite3
import ssl
import sys
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("TWDB_PATH", BASE / "db" / "tw.db"))
RAW = BASE / "raw" / "tw"
QC_REPORT = BASE / "tw_qc_report.csv"
TWSE = "https://www.twse.com.tw/rwd/zh/"
TPEX = "https://www.tpex.org.tw/www/zh-tw/"
DELAY = 3.0                  # 每個網站兩次請求之間至少間隔秒數，避免被封鎖
TZ = ZoneInfo("Asia/Taipei")
AVAILABLE_AT = "15:00"

SSL_CTX = ssl.create_default_context()
SSL_CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT  # 證交所憑證缺 Subject Key Identifier，Python 3.13+ 嚴格模式會拒絕

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    code     TEXT NOT NULL,
    date     TEXT NOT NULL,
    market   TEXT NOT NULL,      -- TWSE / TPEx
    name     TEXT,               -- 當天的證券名稱（改名會反映在這裡）
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,               -- 原始收盤價，未還原；沒成交為 NULL
    volume   REAL,               -- 成交股數
    value    REAL,               -- 成交金額（元）
    trades   REAL,               -- 成交筆數
    ref_next REAL,               -- 次日參考價（僅上櫃）
    managed  INTEGER NOT NULL DEFAULT 0,  -- 上櫃管理股票
    PRIMARY KEY (code, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_quotes_date ON quotes(date);

CREATE TABLE IF NOT EXISTS corp_actions (
    code        TEXT NOT NULL,
    date        TEXT NOT NULL,   -- 除權息日或恢復買賣日（調整套用在這天之前的價格）
    market      TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- exright 除權息 / reduction 減資 / par 面額變更
    prev_close  REAL,            -- 事件前收盤價
    ref_price   REAL,            -- 事件後參考價
    cash_div    REAL,            -- 息值
    stock_value REAL,            -- 權值
    note        TEXT,
    PRIMARY KEY (code, date, kind)
);

CREATE TABLE IF NOT EXISTS taiex (
    date TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL
);

CREATE TABLE IF NOT EXISTS delisted (
    code TEXT NOT NULL, name TEXT, date TEXT NOT NULL, market TEXT NOT NULL,
    PRIMARY KEY (code, date)
);

CREATE TABLE IF NOT EXISTS downloads (
    source     TEXT NOT NULL,
    key        TEXT NOT NULL,
    fetched_at TEXT NOT NULL,    -- UTC
    path       TEXT,             -- raw 快照（相對於專案資料夾）
    sha256     TEXT,
    rows       INTEGER,
    final      INTEGER NOT NULL, -- 0 表示期間還沒結束，下次 sync 會重抓
    message    TEXT,
    PRIMARY KEY (source, key)
);
"""


def connect(init=True):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=60)
    if init:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
    return conn


# ---------- 解析 ----------

def num(s):
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None  # '--'、'' 等表示沒有成交


def price(s):
    """價格欄位：櫃買中心沒有成交時填 0.00，一律當成沒有價格。"""
    return num(s) or None


def roc_date(s):
    """民國日期轉 ISO：'93年07月01日'、'100/01/25'、'1130205'、' 93/02/02'"""
    parts = re.findall(r"\d+", str(s))
    if len(parts) == 1 and len(parts[0]) == 7:
        parts = [parts[0][:3], parts[0][3:5], parts[0][5:]]
    y, m, d = (int(p) for p in parts[:3])
    return date(y + 1911, m, d).isoformat()


def sec_type(code):
    rules = [
        (r"[1-9]\d{3}", "stock"),
        (r"[1-9]\d{3}[A-Z]", "preferred"),
        (r"00\d{2,4}[A-Z]?", "etf"),
        (r"01\d{3}T", "reit"),
        (r"02\d{4}", "etn"),
        (r"91\d{4}", "tdr"),
        (r"(0[3-8]\d{4}|7\d{4,5})[A-Z0-9]?", "warrant"),
    ]
    return next((t for pat, t in rules if re.fullmatch(pat, code)), "other")


def _norm(field):
    return re.sub(r"<[^>]+>|\s", "", str(field))


def table_rows(j, required):
    """回傳所有含 required 欄位的表格的列（dict），以及表格標題。"""
    for t in (j.get("tables") or []) + ([j] if "fields" in j else []):
        fields = [_norm(f) for f in t.get("fields") or []]
        if all(r in fields for r in required):
            for row in t.get("data") or []:
                yield t.get("title") or "", dict(zip(fields, row))


def parse(source, key, j):
    """把一個回應解析成 {表名: [列]}。"""
    if source == "twse_quotes":
        d = f"{key[:4]}-{key[4:6]}-{key[6:]}"
        return {"quotes": [
            (r["證券代號"].strip(), d, "TWSE", r["證券名稱"].strip(), price(r["開盤價"]), price(r["最高價"]),
             price(r["最低價"]), price(r["收盤價"]), num(r["成交股數"]), num(r["成交金額"]), num(r["成交筆數"]), None, 0)
            for _, r in table_rows(j, ["證券代號", "收盤價"])]}
    if source == "tpex_quotes":
        d = f"{key[:4]}-{key[4:6]}-{key[6:]}"
        return {"quotes": [
            (r["代號"].strip(), d, "TPEx", r["名稱"].strip(), price(r["開盤"]), price(r["最高"]), price(r["最低"]),
             price(r["收盤"]), num(r["成交股數"]), num(r["成交金額(元)"]), num(r["成交筆數"]),
             price(r.get("次日參考價")), int("管理" in title))
            for title, r in table_rows(j, ["代號", "收盤"]) if sec_type(r["代號"].strip()) != "warrant"]}
    # 較新的除權息表只有「權值+息值」，沒有分開的權值、息值欄位
    if source == "twse_exright":
        return {"corp_actions": [
            (r["股票代號"].strip(), roc_date(r["資料日期"]), "TWSE", "exright", num(r["除權息前收盤價"]),
             num(r["除權息參考價"]), num(r.get("息值")), num(r.get("權值")),
             f"{r.get('權/息', '')} 權值+息值={r.get('權值+息值', '')}")
            for _, r in table_rows(j, ["資料日期", "股票代號", "除權息參考價"])]}
    if source == "tpex_exright":
        return {"corp_actions": [
            (r["代號"].strip(), roc_date(r["除權息日期"]), "TPEx", "exright", num(r["除權息前收盤價"]),
             num(r["除權息參考價"]), num(r.get("息值")), num(r.get("權值")),
             f"{r.get('權/息', '')} 權值+息值={r.get('權值+息值', '')}")
            for _, r in table_rows(j, ["除權息日期", "代號", "除權息參考價"])]}
    if source == "twse_reduction":
        return {"corp_actions": [
            (r["股票代號"].strip(), roc_date(r["恢復買賣日期"]), "TWSE", "reduction", num(r["停止買賣前收盤價格"]),
             num(r["恢復買賣參考價"]), None, None, r.get("減資原因"))
            for _, r in table_rows(j, ["恢復買賣日期", "股票代號", "恢復買賣參考價"])]}
    if source == "tpex_reduction":
        return {"corp_actions": [
            (r["股票代號"].strip(), roc_date(r["恢復買賣日期"]), "TPEx", "reduction", num(r["最後交易日之收盤價格"]),
             num(r["減資恢復買賣開始日參考價格"]), None, None, r.get("減資原因"))
            for _, r in table_rows(j, ["恢復買賣日期", "股票代號", "減資恢復買賣開始日參考價格"])]}
    if source == "tpex_par":
        return {"corp_actions": [
            (r["證券代號"].strip(), roc_date(r["恢復買賣日期"]), "TPEx", "par", num(r["最後交易日之收盤價格"]),
             num(r["恢復買賣開始參考價"]), None, None, None)
            for _, r in table_rows(j, ["恢復買賣日期", "證券代號", "恢復買賣開始參考價"])]}
    if source == "twse_par":
        return {"corp_actions": [
            (r["股票代號"].strip(), roc_date(r["恢復買賣日期"]), "TWSE", "par", num(r["停止買賣前收盤價格"]),
             num(r["恢復買賣參考價"]), None, None, None)
            for _, r in table_rows(j, ["恢復買賣日期", "股票代號", "恢復買賣參考價"])]}
    if source == "twse_etfsplit":
        return {"corp_actions": [
            (r["ETF代號"].strip(), roc_date(r["恢復買賣日期"]), "TWSE", "split", num(r["停止買賣前收盤價格"]),
             num(r["恢復買賣參考價"]), None, None, r.get("分割(反分割)"))
            for _, r in table_rows(j, ["恢復買賣日期", "ETF代號", "恢復買賣參考價"])]}
    if source == "taiex":
        return {"taiex": [(roc_date(r[0]), num(r[1]), num(r[2]), num(r[3]), num(r[4])) for r in j.get("data") or []]}
    if source == "twse_delisted":
        return {"delisted": [(r["上市編號"].strip(), r["公司名稱"].strip(), roc_date(r["終止上市日期"]), "TWSE")
                             for _, r in table_rows(j, ["終止上市日期", "上市編號"])]}
    raise ValueError(source)


def write(conn, parsed):
    sql = {
        "quotes": "INSERT OR REPLACE INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        "corp_actions": "INSERT OR REPLACE INTO corp_actions VALUES (?,?,?,?,?,?,?,?,?)",
        "taiex": "INSERT OR REPLACE INTO taiex VALUES (?,?,?,?,?)",
        "delisted": "INSERT OR REPLACE INTO delisted VALUES (?,?,?,?)",
    }
    for table, rows in parsed.items():
        conn.executemany(sql[table], rows)
    return sum(len(r) for r in parsed.values())


# ---------- 下載工作清單 ----------

def _today():
    return datetime.now(TZ).date()


def _trading_days(start, end):
    """週一到週五，加上加權指數有開盤的週六（補行上班日）。"""
    conn = connect(init=False)
    saturdays = {d for (d,) in conn.execute("SELECT date FROM taiex")}
    conn.close()
    d = end
    while d >= start:  # 由新到舊，近期資料先到
        if d.weekday() < 5 or d.isoformat() in saturdays:
            yield d
        d -= timedelta(days=1)


def _months(start, end):
    m = date(end.year, end.month, 1)
    while m >= date(start.year, start.month, 1):
        last = (m.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        yield max(m, start), min(last, end)
        m = (m - timedelta(days=1)).replace(day=1)


def _years(start, end):
    for y in range(end.year, start.year - 1, -1):
        yield max(date(y, 1, 1), start), min(date(y, 12, 31), end)


def jobs(host):
    """回傳 (source, key, url, 期間結束日) 清單。"""
    today = _today()
    ymd, slash = (lambda d: d.strftime("%Y%m%d")), (lambda d: d.strftime("%Y%%2F%m%%2F%d"))
    out = []
    if host == "twse":
        out.append(("twse_delisted", today.isoformat(), TWSE + "company/suspendListing?response=json", today))
        out += [("taiex", ymd(a)[:6], TWSE + f"TAIEX/MI_5MINS_HIST?date={ymd(a)}&response=json", b)
                for a, b in _months(date(1999, 1, 1), today)]
        out += [("twse_exright", f"{ymd(a)}-{ymd(b)}",
                 TWSE + f"exRight/TWT49U?startDate={ymd(a)}&endDate={ymd(b)}&response=json", b)
                for a, b in _months(date(2003, 5, 5), today)]
        out += [("twse_reduction", f"{ymd(a)}-{ymd(b)}",
                 TWSE + f"reducation/TWTAUU?startDate={ymd(a)}&endDate={ymd(b)}&response=json", b)
                for a, b in _years(date(2011, 1, 1), today)]
        out += [("twse_par", f"{ymd(a)}-{ymd(b)}",
                 TWSE + f"change/TWTB8U?startDate={ymd(a)}&endDate={ymd(b)}&response=json", b)
                for a, b in _years(date(2004, 1, 1), today)]
        out += [("twse_etfsplit", f"{ymd(a)}-{ymd(b)}",
                 TWSE + f"split/TWTCAU?startDate={ymd(a)}&endDate={ymd(b)}&response=json", b)
                for a, b in _years(date(2004, 1, 1), today)]
        out += [("twse_quotes", ymd(d), TWSE + f"afterTrading/MI_INDEX?date={ymd(d)}&type=ALLBUT0999&response=json", d)
                for d in _trading_days(date(2004, 2, 11), today)]
    else:
        out += [("tpex_exright", f"{ymd(a)}-{ymd(b)}",
                 TPEX + f"bulletin/exDailyQ?startDate={slash(a)}&endDate={slash(b)}&response=json", b)
                for a, b in _months(date(2007, 1, 1), today)]
        out += [("tpex_reduction", f"{ymd(a)}-{ymd(b)}",
                 TPEX + f"bulletin/revivt?startDate={slash(a)}&endDate={slash(b)}&response=json", b)
                for a, b in _years(date(2007, 1, 1), today)]
        out += [("tpex_par", f"{ymd(a)}-{ymd(b)}",
                 TPEX + f"bulletin/pvChgRslt?startDate={slash(a)}&endDate={slash(b)}&response=json", b)
                for a, b in _years(date(2019, 1, 1), today)]
        out += [("tpex_quotes", ymd(d), TPEX + f"afterTrading/dailyQuotes?date={slash(d)}&id=&response=json", d)
                for d in _trading_days(date(2007, 7, 2), today)]
    return out


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
        body = r.read()
    return body, json.loads(body)  # 被封鎖時回傳 HTML，json 解析失敗會當成錯誤處理


def _run_host(host, log):
    conn = connect(init=False)  # 資料表已在 sync() 建立，兩個執行緒同時建表會互相鎖住
    done = {(s, k) for s, k in conn.execute("SELECT source, key FROM downloads WHERE final=1")}
    todo = [j for j in jobs(host) if (j[0], j[1]) not in done]
    log(f"[{host}] 待下載 {len(todo)} 筆")
    started, failures = time.time(), 0
    i = 0
    while i < len(todo):
        source, key, url, end = todo[i]
        t0 = time.time()
        try:
            body, j = _fetch(url)
        except Exception as e:
            failures += 1
            if failures >= 6:
                log(f"[{host}] 連續失敗 {failures} 次，停止（可能被暫時封鎖，稍後重跑 sync 會接續）: {e!r}")
                return
            wait = 60 * 2 ** (failures - 1)
            log(f"[{host}] {source} {key} 失敗: {e!r}，{wait} 秒後重試")
            time.sleep(wait)
            continue
        failures = 0
        rows = write(conn, parse(source, key, j))
        path = RAW / source / key[:4] / f"{key}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body))
        final = end < _today() or rows > 0
        conn.execute("INSERT OR REPLACE INTO downloads VALUES (?,?,?,?,?,?,?,?)",
                     (source, key, datetime.now(timezone.utc).isoformat(timespec="seconds"), str(path.relative_to(BASE)),
                      hashlib.sha256(body).hexdigest(), rows, int(final), str(j.get("stat", ""))[:200]))
        conn.commit()
        i += 1
        if i % 100 == 0 or i == len(todo):
            eta = (time.time() - started) / i * (len(todo) - i) / 3600
            log(f"[{host}] {i}/{len(todo)}，最新 {source} {key}，預估剩 {eta:.1f} 小時")
        time.sleep(max(0.0, DELAY + random.uniform(0, 1.5) - (time.time() - t0)))
    log(f"[{host}] 完成")


def _single_instance():
    """同一時間只允許一個 sync；行程結束時 Windows 會自動釋放檔案鎖。回傳 None 表示已有別的 sync 在跑。"""
    import msvcrt
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = open(DB_PATH.with_suffix(".sync.lock"), "a+")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        return None
    return f


def sync():
    """回傳 True 表示這次有執行，False 表示已有另一個 sync 在跑。"""
    guard = _single_instance()
    if guard is None:
        print("另一個 twdb sync 正在執行，略過")
        return False
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)

    connect().close()
    threads = [threading.Thread(target=_run_host, args=(h, log)) for h in ("twse", "tpex")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    guard.close()
    return True


def rebuild():
    """刪除解析後的資料表，從 raw 快照重新解析。"""
    with connect() as conn:
        for t in ("quotes", "corp_actions", "taiex", "delisted"):
            conn.execute(f"DELETE FROM {t}")
        entries = conn.execute("SELECT source, key, path FROM downloads ORDER BY source, key").fetchall()
        for n, (source, key, path) in enumerate(entries, 1):
            write(conn, parse(source, key, json.loads(gzip.decompress((BASE / path).read_bytes()))))
            if n % 500 == 0:
                print(f"{n}/{len(entries)}")
    print(f"已從 {len(entries)} 個快照重建")


def status():
    with connect() as conn:
        print(pd.read_sql("""SELECT source, COUNT(*) AS files, SUM(rows) AS rows, MIN(key) AS first, MAX(key) AS last
                             FROM downloads GROUP BY source""", conn).to_string(index=False))
        print(pd.read_sql("""SELECT market, COUNT(DISTINCT code) AS codes, MIN(date) AS first, MAX(date) AS last,
                                    COUNT(*) AS rows FROM quotes GROUP BY market""", conn).to_string(index=False))


# ---------- 讀取 ----------

def _factors(conn, code, market):
    """回傳 {生效日: 調整係數}，係數套用在生效日之前的價格。"""
    q = pd.read_sql("SELECT date, close, ref_next FROM quotes WHERE code=? ORDER BY date", conn, params=[code])
    acts = conn.execute("SELECT date, kind, ref_price / prev_close FROM corp_actions "
                        "WHERE code=? AND prev_close > 0 AND ref_price > 0", (code,)).fetchall()
    # 除權息日遇到颱風假等休市時，順延到下一個交易日，才不會和次日參考價算出的係數重複
    days = q["date"].tolist()
    acts = [(days[i] if (i := bisect.bisect_left(days, d)) < len(days) else d, k, f) for d, k, f in acts]
    out = {}
    if market == "TPEx":
        # 第 t 天公布的次日參考價，作用在下一個交易日。只用有成交的日子：
        # 沒成交時櫃買中心會依買賣報價移動參考價，那是價格變動，不是公司行動
        f = (q["ref_next"] / q["close"]).shift(1)
        f.index = q["date"]
        out = f[(f - 1).abs() > 1e-6].dropna().to_dict()
        # 前一天沒成交或停止買賣（減資）時，次日參考價抓不到事件，改用除權息表、減資表
        acts = [(d, k, f) for d, k, f in acts if d not in out]
    for d, _, f in acts:
        out[d] = out.get(d, 1.0) * f
    return out


def load(code, start=None, end=None, adjust=True):
    """讀取單一證券日線。

    adjust=True 時 OHLC 為還原價（除權息、減資、面額變更都已調整到最新價位），成交量反向調整；
    close_raw 保留原始收盤價，adj_factor 是乘上去的倍數。available_at 是這筆資料可以被使用的時間。
    """
    with connect() as conn:
        q = pd.read_sql("SELECT * FROM quotes WHERE code=? ORDER BY date", conn, params=[code])
        if q.empty:
            return q
        factors = _factors(conn, code, q["market"].iloc[-1])
    q["date"] = pd.to_datetime(q["date"])
    q = q.set_index("date")
    g = pd.Series(1.0, index=q.index)
    for d, f in factors.items():
        g[g.index < pd.Timestamp(d)] *= f
    q["adj_factor"] = g
    q["close_raw"] = q["close"]
    if adjust:
        for c in ("open", "high", "low", "close", "ref_next"):
            q[c] = q[c] * g
        q["volume"] = q["volume"] / g
    q["available_at"] = (q.index + pd.Timedelta(AVAILABLE_AT + ":00")).tz_localize(TZ)
    return q.loc[start:end]


def universe(day, types=("stock",), min_value=None, lookback=20):
    """某交易日收盤時的股票池：當天有收盤價的證券（含日後下市者）。

    min_value: 過去 lookback 個交易日（含當天）成交金額中位數下限（元），只用當天以前的資料。
    """
    with connect() as conn:
        days = [d for (d,) in conn.execute(
            "SELECT DISTINCT date FROM quotes WHERE date<=? ORDER BY date DESC LIMIT ?", (day, lookback))]
        q = pd.read_sql(f"SELECT code, date, name, market, close, value FROM quotes WHERE date IN "
                        f"({','.join('?' * len(days))})", conn, params=days)
    today = q[(q["date"] == day) & q["close"].notna()].copy()
    today = today[today["code"].map(sec_type).isin(types)]
    if min_value:
        med = q.groupby("code")["value"].median()
        today = today[today["code"].map(med) >= min_value]
    return today.set_index("code")[["name", "market", "close"]]


# ---------- 品質檢查 ----------

def check():
    """依 ML4T 資料品質報告的項目檢查，明細寫入 tw_qc_report.csv。"""
    issues, traded_no_price = [], 0
    np.seterr(invalid="ignore")
    with connect() as conn:
        codes = [c for (c,) in conn.execute("SELECT DISTINCT code FROM quotes")]
        per_day = pd.read_sql("SELECT date, market, COUNT(*) AS n FROM quotes GROUP BY date, market", conn)
    # 1. 某天的證券數突然變少，可能是下載不完整
    for m, g in per_day.groupby("market"):
        med = g["n"].rolling(21, center=True, min_periods=5).median()
        for _, r in g[g["n"] < 0.8 * med].iterrows():
            issues.append(("*", r["date"], m, "當日筆數偏少", f"{r['n']} 筆，前後中位數 {med[_]:.0f}"))
    for n, code in enumerate(codes, 1):
        t = sec_type(code)
        q = load(code)
        c, o, h, l = q["close"], q["open"], q["high"], q["low"]
        days = q.index.strftime("%Y-%m-%d")
        bad = (h < np.fmax(o, c) - 1e-9) | (l > np.fmin(o, c) + 1e-9) | (h < l)
        issues += [(code, d, t, "OHLC 矛盾", f"O={o.iloc[i]} H={h.iloc[i]} L={l.iloc[i]} C={c.iloc[i]}")
                   for i, d in enumerate(days) if bad.iloc[i]]
        # 只有鉅額、盤後定價成交時有量但沒有收盤價，屬正常，只計數不列入報告
        traded_no_price += int(((q["volume"] > 0) & c.isna()).sum())
        if t == "stock":
            # 2. 超過漲跌幅限制：2015-06-01 前 7%，之後 10%；上市前 5 日無限制
            # 中間有沒成交的日子時，參考價每天都可以再漲跌一次，所以容許範圍按間隔天數複利
            pos = pd.Series(np.arange(len(c)), index=c.index)
            gap = (pos - pos.where(c.notna()).ffill().shift()).fillna(1)
            r = c / c.ffill().shift() - 1
            limit = np.where(q.index < "2015-06-01", 0.07, 0.10) + 0.012  # 檔位進位容忍
            valid = c.notna().cumsum() > 5
            over = ((r > (1 + limit) ** gap - 1) | (r < (1 - limit) ** gap - 1)) & valid
            issues += [(code, d, t, "超過漲跌幅", f"還原後 {r.iloc[i]:+.1%}，可能缺公司行動")
                       for i, d in enumerate(days) if over.iloc[i]]
        # 3. 代號可能被重複使用：同一代號中斷超過一年後又出現，且名稱不同
        dt = q.index.to_series().diff().dt.days
        for i in np.flatnonzero(dt > 365):
            if q["name"].iloc[i] != q["name"].iloc[i - 1]:
                issues.append((code, days[i], t, "代號可能重複使用",
                               f"{q['name'].iloc[i - 1]} → {q['name'].iloc[i]}，中斷 {int(dt.iloc[i])} 天"))
        if n % 500 == 0:
            print(f"已檢查 {n}/{len(codes)}")
    report = pd.DataFrame(issues, columns=["code", "date", "type", "check", "detail"])
    report.to_csv(QC_REPORT, index=False, encoding="utf-8-sig")
    print(report.pivot_table(index="type", columns="check", values="date", aggfunc="count", fill_value=0)
          .to_string() if not report.empty else "沒有發現問題")
    print(f"\n共 {len(report)} 筆，明細見 {QC_REPORT}")
    print(f"另有 {traded_no_price} 筆有成交量但沒有收盤價（只有鉅額或盤後成交，收盤價保留空值）")
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    {"sync": sync, "rebuild": rebuild, "check": check, "status": status}.get(cmd, lambda: print(__doc__))()
