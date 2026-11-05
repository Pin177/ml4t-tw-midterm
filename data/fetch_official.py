"""Download official data missing from tw.db: institutional flows, monthly revenue, disposition notices.

Every response is stored untouched under data/raw/ so the dataset can be rebuilt offline.
Re-running resumes: a file that exists is not fetched again. Each site gets one request per ~3.5 s.

    python data/fetch_official.py
"""
import gzip
import os
import random
import sqlite3
import ssl
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = Path(os.environ.get("TWDB_PATH", ROOT.parent / "db" / "tw.db"))
FIRST = "2012-05-02"
LAST = "2026-11-04"
TWSE = "https://www.twse.com.tw/rwd/zh/"
TPEX = "https://www.tpex.org.tw/www/zh-tw/"
MOPS = "https://mopsov.twse.com.tw/nas/t21/"

CTX = ssl.create_default_context()
CTX.verify_flags &= ~ssl.VERIFY_X509_STRICT


def trading_days():
    c = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    days = [d for (d,) in c.execute("SELECT DISTINCT date FROM quotes WHERE date BETWEEN ? AND ? ORDER BY date DESC", (FIRST, LAST))]
    c.close()
    return days


def months():
    y, m = 2026, 9
    while (y, m) >= (2011, 1):
        yield y, m
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)


def revenue_url(board, y, m):
    roc = y - 1911
    # Before 2013 (ROC 102) the page has no foreign-company suffix
    tail = "_0.html" if roc > 101 else ".html"
    return f"{MOPS}{board}/t21sc03_{roc}_{m}{tail}"


def jobs_for(site):
    days = trading_days()
    out = []
    if site == "mops":
        for y, m in months():
            for board in ("sii", "otc"):
                out.append((RAW / f"revenue_{board}" / f"{y}{m:02d}.html.gz", revenue_url(board, y, m)))
    elif site == "twse":
        out += [(RAW / "punish_twse" / f"{y}{m:02d}.json.gz",
                 f"{TWSE}announcement/punish?startDate={y}{m:02d}01&endDate={y}{m:02d}28&response=json") for y, m in months() if y >= 2012]
        out += [(RAW / "flow_twse" / f"{d}.json.gz",
                 f"{TWSE}fund/T86?date={d.replace('-', '')}&selectType=ALLBUT0999&response=json") for d in days]
    else:
        out += [(RAW / "flow_tpex" / f"{d}.json.gz",
                 f"{TPEX}insti/dailyTrade?type=Daily&sect=EW&date={d.replace('-', '/')}&id=&response=json") for d in days]
    return out


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
        return r.read()


def run(site):
    todo = [(p, u) for p, u in jobs_for(site) if not p.exists()]
    print(f"[{site}] {len(todo)} to fetch", flush=True)
    fails = 0
    i = 0
    while i < len(todo):
        path, url = todo[i]
        t0 = time.time()
        try:
            body = fetch(url)
        except Exception as e:
            fails += 1
            if fails >= 8:
                print(f"[{site}] stopped after {fails} failures: {e!r}. Re-run to resume.", flush=True)
                return
            time.sleep(60 * 2 ** (fails - 1))
            continue
        fails = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(body))
        i += 1
        if i % 200 == 0:
            print(f"[{site}] {i}/{len(todo)} {path.name}", flush=True)
        time.sleep(max(0.0, 3.0 + random.uniform(0, 1.0) - (time.time() - t0)))
    print(f"[{site}] done", flush=True)


if __name__ == "__main__":
    threads = [threading.Thread(target=run, args=(s,)) for s in ("mops", "twse", "tpex")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("all done", date.today(), flush=True)
