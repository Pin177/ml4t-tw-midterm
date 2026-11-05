"""Check 法人資料/ against the official TWSE T86 and TPEx dailyTrade responses on sampled days.

Writes reports/flow_verification.csv. One request per ~4 s. Needs network access.

    python data/verify_flows.py
"""
import json
import re
import ssl
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CTX = ssl.create_default_context()
CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT
STOCK = re.compile(r"[1-9]\d{3}")
TWSE_DAYS = ["2012-05-03", "2014-06-10", "2017-12-20", "2018-01-16", "2020-03-19", "2023-06-14", "2026-10-15"]
TPEX_DAYS = ["2012-05-03", "2013-06-17", "2014-12-15", "2016-06-15", "2018-06-15", "2020-03-19", "2023-06-14", "2026-10-15"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return json.loads(r.read())


def num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except ValueError:
        return np.nan


def twse_day(d):
    j = get(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={d.replace('-', '')}&selectType=ALLBUT0999&response=json")
    fields = [re.sub(r"\s", "", x) for x in j.get("fields") or []]
    if not j.get("data"):
        return None
    fi = next(i for i, x in enumerate(fields) if x.startswith(("外陸資買賣超股數", "外資買賣超股數")))
    ti = fields.index("投信買賣超股數")
    return {r[0].strip(): (num(r[fi]), num(r[ti])) for r in j["data"] if STOCK.fullmatch(r[0].strip())}


def tpex_day(d):
    for sect in ("EW", "AL"):
        j = get(f"https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect={sect}&date={d.replace('-', '/')}&id=&response=json")
        t = next((t for t in (j.get("tables") or [j]) if t.get("data")), None)
        if t is not None:
            n = len(t["data"][0])
            fi, ti = (4, 13) if n == 24 else (4, 7) if n >= 12 else (None, None)
            if fi is None:
                return None
            return {str(r[0]).strip(): (num(r[fi]), num(r[ti])) for r in t["data"] if STOCK.fullmatch(str(r[0]).strip())}
        time.sleep(4)
    return None


def main():
    foreign = pd.read_parquet(ROOT / "法人資料" / "foreign_net_buy.parquet")
    trust = pd.read_parquet(ROOT / "法人資料" / "trust_net_buy.parquet")
    rows = []
    for market, days, fn in (("TWSE", TWSE_DAYS, twse_day), ("TPEx", TPEX_DAYS, tpex_day)):
        for d in days:
            off = fn(d)
            time.sleep(4)
            ts = pd.Timestamp(d)
            if not off:
                rows.append(dict(market=market, date=d, official_rows=0, note="official endpoint returned no rows"))
                continue
            o = pd.DataFrame.from_dict(off, orient="index", columns=["foreign", "trust"])
            exact_f = (o["foreign"] - foreign.loc[ts].reindex(o.index)).abs() < 1
            exact_t = (o["trust"] - trust.loc[ts].reindex(o.index)).abs() < 1
            rows.append(dict(market=market, date=d, official_rows=len(o), foreign_exact=int(exact_f.sum()), trust_exact=int(exact_t.sum()),
                             in_file=int(foreign.loc[ts].reindex(o.index).notna().sum()), note=""))
            print(rows[-1], flush=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "flow_verification.csv", index=False)


if __name__ == "__main__":
    main()
