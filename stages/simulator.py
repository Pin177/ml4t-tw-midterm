"""Event-style daily simulator: month-end signal, next-open fills, constraints and costs inside the loop.

Value is marked open to open. A missing open means no fill that day and the last known price for marking.
A stock that stops trading is sold at its last close, with sell costs, on the day after its last quote.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import PROC, RAW, SETUP

C = SETUP["cost_model"]
CON = SETUP["constraints"]
RULES = SETUP["score_to_trade"]
ENTRY, HOLD = 0.10, 0.20
MIN_NAMES = 100
SECTORS = json.loads((RAW / "sectors.json").read_text(encoding="utf-8"))


@dataclass
class Result:
    nav: pd.Series             # net of costs, open-to-open daily
    gross_nav: pd.Series       # before any cost
    traded: pd.Series          # value traded each day / NAV, two-way
    holdings: pd.Series        # names held per rebalance
    costs_paid: float
    max_name: float
    max_sector: float
    fills_skipped: int
    sells_delayed: int
    weights: dict


def half_spread(median_value, mult=1.0):
    return np.select([median_value > 5e8, median_value > 1e8], [5.0, 10.0], 20.0) * mult


def cap(w, max_name=CON["max_weight_per_name"], max_sector=CON["max_weight_per_sector"]):
    """Apply name and sector caps. Excess goes pro rata to uncapped names; what cannot be placed stays in cash."""
    w = w.copy()
    for _ in range(50):
        over = w > max_name + 1e-12
        if over.any():
            excess = (w[over] - max_name).sum()
            w[over] = max_name
            room = w[~over]
            if room.sum() > 0:
                w[~over] += excess * room / room.sum()
        sec = pd.Series({c: SECTORS.get(c, f"NA_{c}") for c in w.index})
        totals = w.groupby(sec).sum()
        bad = totals[totals > max_sector + 1e-12]
        for s, tot in bad.items():
            mem = sec[sec == s].index
            excess = tot - max_sector
            w[mem] *= max_sector / tot
            others = w.index.difference(mem)
            others = others[(sec[others].map(totals) < max_sector) & (w[others] < max_name)]
            if len(others) and w[others].sum() > 0:
                w[others] += excess * w[others] / w[others].sum()
        if not over.any() and bad.empty:
            break
    return w[w > 0]


def load_market():
    o = pd.read_parquet(PROC / "open.parquet")
    c = pd.read_parquet(PROC / "close.parquet")
    listed = pd.read_parquet(PROC / "listed.parquet").fillna(0) > 0
    last_listed = listed.apply(lambda s: s[s].index.max())
    sell_locked = pd.read_parquet(PROC / "sell_locked.parquet")
    return dict(open=o, close=c, last_listed=last_listed, sell_locked=sell_locked)


def simulate(decisions, market, start, end, select="score", cost_mult=1.0, spread_mult=1.0, allocator="equal",
             exposure=None, vol=None):
    """decisions: rows of date, code, score, median_value, buy_locked (month-end eligible stocks).

    exposure: {decision date: fraction invested}, from t-1 data only. select="all" holds every eligible stock.
    """
    op, cl, last_listed, slk = market["open"], market["close"], market["last_listed"], market["sell_locked"]
    cal = op.index[(op.index >= start) & (op.index <= end)]
    # Last known price for marking: today's open, else the most recent close before today
    mark = op.where(op.notna(), cl.ffill().shift(1)).reindex(cal)
    by_date = {d: g for d, g in decisions.groupby("date")}
    exec_map = {}
    full_cal = op.index
    for d in by_date:
        pos = full_cal.searchsorted(d)
        if pos + 1 < len(full_cal) and full_cal[pos + 1] in mark.index:
            exec_map[full_cal[pos + 1]] = d

    shares = {}          # code -> units of adjusted price
    cash, gross = 1.0, 1.0
    g_shares = {}
    nav_s, gnav_s, traded_s, hold_s = [], [], [], {}
    pending_sells, spread_of = set(), {}
    costs_paid, skip, delayed = 0.0, 0, 0
    max_name, max_sector = 0.0, 0.0
    weights_log = {}

    for day in cal:
        prices = mark.loc[day]
        def value(pos):
            return sum(q * prices.get(c, np.nan) if np.isfinite(prices.get(c, np.nan)) else 0.0 for c, q in pos.items())
        nav = cash + value(shares)
        traded = 0.0

        # Names gone from the market: sell at last close
        for c in list(shares):
            if day > last_listed[c]:
                px = prices.get(c)
                if not np.isfinite(px):
                    px = cl[c].loc[:day].dropna().iloc[-1] if cl[c].loc[:day].notna().any() else 0.0
                amount = shares.pop(c) * px
                fee = amount * (C["commission_bps_each_side"] + C["sell_tax_bps"] + spread_of.get(c, 20.0) * spread_mult) * cost_mult / 1e4
                cash += amount - fee
                costs_paid += fee
                traded += amount
                pending_sells.discard(c)

        # Delayed sells retry at each open
        for c in list(pending_sells):
            px = op.at[day, c] if c in op.columns else np.nan
            if c in shares and np.isfinite(px) and not bool(slk.at[day, c] if c in slk.columns else False):
                amount = shares.pop(c) * px
                fee = amount * (C["commission_bps_each_side"] + C["sell_tax_bps"] + spread_of.get(c, 20.0) * spread_mult) * cost_mult / 1e4
                cash += amount - fee
                costs_paid += fee
                traded += amount
                pending_sells.discard(c)

        if day in exec_map:
            dec = exec_map[day]
            g = by_date[dec].set_index("code")
            nav = cash + value(shares)
            exposure_now = 1.0 if exposure is None else float(exposure.get(dec, 1.0))
            halt_new = exposure_now < 0
            exposure_now = abs(exposure_now)
            if select == "all":
                target = list(g.index)
            elif len(g) < MIN_NAMES:
                target = [c for c in shares if c in g.index]
            else:
                pct = g["score"].rank(pct=True, ascending=False)
                keep = [c for c in shares if c in pct.index and pct[c] <= HOLD and c not in pending_sells]
                new = [] if halt_new else [c for c in pct[pct <= ENTRY].index if c not in shares]
                target = keep + new
            # A limit-up or untraded open: no buy this month, cash goes to the other names
            buyable = []
            for c in target:
                if c not in shares and (bool(g.at[c, "buy_locked"]) or not np.isfinite(op.at[day, c])):
                    skip += 1
                else:
                    buyable.append(c)
            target = buyable
            if target:
                if allocator == "inverse_vol" and vol is not None:
                    v = vol.loc[dec].reindex(target).replace(0, np.nan)
                    v = v.fillna(v.median()).fillna(1.0)
                    w = (1 / v) / (1 / v).sum()
                else:
                    w = pd.Series(1 / len(target), index=target)
                w = cap(w) * exposure_now
            else:
                w = pd.Series(dtype=float)
            if len(w):
                max_name = max(max_name, float(w.max()))
                sec = pd.Series({c: SECTORS.get(c, f"NA_{c}") for c in w.index})
                max_sector = max(max_sector, float(w.groupby(sec).sum().max()))
            for c in g.index.intersection(list(w.index)):
                spread_of[c] = float(half_spread(g.at[c, "median_value"]))
            # Sells and trims first
            for c in list(shares):
                px = op.at[day, c] if c in op.columns else np.nan
                tgt = w.get(c, 0.0) * nav
                if not np.isfinite(px):
                    if tgt == 0:
                        pending_sells.add(c)
                    continue
                cur = shares[c] * px
                if cur > tgt + 1e-12:
                    if bool(slk.at[day, c] if c in slk.columns else False):
                        if tgt == 0:
                            pending_sells.add(c); delayed += 1
                        continue
                    amount = cur - tgt
                    fee = amount * (C["commission_bps_each_side"] + C["sell_tax_bps"] + spread_of.get(c, 20.0) * spread_mult) * cost_mult / 1e4
                    cash += amount - fee
                    costs_paid += fee
                    traded += amount
                    shares[c] = tgt / px
                    if tgt == 0:
                        shares.pop(c)
            # Buys and top-ups, scaled to the cash on hand so that no leverage can appear
            need = {c: max(w[c] * nav - shares.get(c, 0.0) * op.at[day, c], 0.0) for c in w.index if np.isfinite(op.at[day, c])}
            total = sum(need.values())
            bp = {c: (C["commission_bps_each_side"] + spread_of.get(c, 20.0) * spread_mult) * cost_mult / 1e4 for c in need}
            all_in = sum(need[c] * (1 + bp[c]) for c in need)
            scale = min(1.0, cash / all_in) if all_in > 0 else 0.0
            for c, amount in need.items():
                amount *= scale
                if amount <= 0:
                    continue
                fee = amount * bp[c]
                cash -= amount + fee
                costs_paid += fee
                traded += amount
                shares[c] = shares.get(c, 0.0) + amount / op.at[day, c]
            hold_s[dec] = len(shares)
            weights_log[dec] = w
            # Gross track: same target, no costs, marked the same way
            g_nav = sum(q * prices.get(c, 0.0) for c, q in g_shares.items() if np.isfinite(prices.get(c, np.nan))) + gross
            g_shares = {c: w[c] * g_nav / op.at[day, c] for c in w.index if np.isfinite(op.at[day, c])}
            gross = g_nav - sum(q * op.at[day, c] for c, q in g_shares.items())

        nav_s.append(cash + value(shares))
        gnav_s.append(gross + sum(q * prices.get(c, 0.0) for c, q in g_shares.items() if np.isfinite(prices.get(c, np.nan))))
        traded_s.append(traded / max(nav_s[-1], 1e-9))

    return Result(pd.Series(nav_s, index=cal), pd.Series(gnav_s, index=cal), pd.Series(traded_s, index=cal), pd.Series(hold_s),
                  costs_paid, max_name, max_sector, skip, delayed, weights_log)


def metrics(nav, traded=None, benchmark=None):
    r = nav.pct_change().dropna()
    years = len(r) / 252
    sr = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    se = np.sqrt((1 + sr ** 2 / 2) / years)
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()
    recovered = nav.loc[trough:][nav.loc[trough:] >= nav.loc[peak]]
    out = dict(cagr=(nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1, vol=r.std() * np.sqrt(252), sharpe=sr, sharpe_low=sr - 1.96 * se, sharpe_high=sr + 1.96 * se,
               max_drawdown=dd.min(), dd_peak=str(peak.date()), dd_trough=str(trough.date()), dd_recovered=str(recovered.index[0].date()) if len(recovered) else "not recovered",
               var_95=r.quantile(0.05), cvar_95=r[r <= r.quantile(0.05)].mean(), skew=r.skew(), kurtosis=r.kurtosis() + 3, years=years)
    if traded is not None:
        out["turnover_two_way"] = traded.sum() / years
        out["holding_months"] = 24 / out["turnover_two_way"] if out["turnover_two_way"] > 0 else np.nan
    return out


def deflated_sharpe(sr_annual, years, skew, kurt, n_trials, sr_var_annual):
    """Probability that the true Sharpe is above zero after picking the best of n_trials (Bailey and Lopez de Prado)."""
    from scipy.stats import norm
    t = years * 252
    sr = sr_annual / np.sqrt(252)
    v = sr_var_annual / 252
    gamma = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))) if n_trials > 1 else 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    return float(norm.cdf((sr - sr0) * np.sqrt(t - 1) / denom))


def min_track_record_years(sr, alpha_z=1.645, power_z=0.84):
    return ((alpha_z + power_z) / sr) ** 2 if sr and sr > 0 else np.inf
