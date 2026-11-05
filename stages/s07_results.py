"""Stage 7: headline table, the six required figures, and descriptive diagnostics. Reads results only; changes nothing."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from common import FIGS, HOLD_START, PROC, REPORTS, SIGNALS, TRAIN_START, daily_ic, load_panel, nw_mean, purged_train
from s03_models import add_ranks, thin

NAVY, BLUE, GREY, GOLD, RED = "#0a3d62", "#3c8dbc", "#8a8f98", "#d9a441", "#b5473c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
VAL = json.loads((REPORTS / "validation_summary.json").read_text(encoding="utf-8"))
HOLD = json.loads((REPORTS / "holdout_results.json").read_text(encoding="utf-8"))
DEPLOYED = HOLD["deployed"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=170)
    plt.close(fig)


def fig1_equity():
    v = pd.read_parquet(PROC / "nav_validation.parquet")
    h = pd.read_parquet(PROC / "nav_holdout.parquet")
    fig, ax = plt.subplots(figsize=(8.6, 4))
    for col, label, color, lw in (("strategy", "Ridge strategy (net)", NAVY, 2), ("rule", "12-1 momentum rule (net)", GOLD, 1.3), ("etf_0050", "0050 ETF", GREY, 1.3)):
        a = v[col].dropna()
        a = a / a.iloc[0]
        b = h[col].dropna()
        b = b / b.iloc[0] * a.iloc[-1]
        ax.plot(pd.concat([a, b]), label=label, color=color, lw=lw)
    ax.axvline(HOLD_START, color=RED, ls="--", lw=1)
    ax.text(HOLD_START, ax.get_ylim()[1] * 0.97, " holdout", color=RED, va="top")
    ax.set_yscale("log")
    ax.set(title="Did the strategy beat its baselines, net of costs?", ylabel="Growth of 1 (log scale)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig1_equity_curves.png")


def fig2_rolling_ic():
    s = pd.read_parquet(PROC / "scores_val.parquet")
    panel = load_panel(allow_holdout=True)[["date", "code", "label"]]
    val = daily_ic(panel.merge(s, on=["date", "code"]), DEPLOYED)
    hold = pd.read_parquet(PROC / f"ic_holdout_{DEPLOYED}.parquet")["ic"]
    ic = pd.concat([val, hold])
    ic.to_frame("ic").to_parquet(PROC / "ic_all.parquet")
    r = ic.rolling(63, min_periods=40).mean()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(r, color=NAVY)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(HOLD_START, color=RED, ls="--", lw=1)
    ax.set(title="Is the signal stable? Rolling 3-month Spearman IC", ylabel="IC")
    save(fig, "fig2_rolling_ic.png")
    return float((r.dropna() > 0).mean())


def fig3_cost():
    v = pd.read_csv(REPORTS / "cost_sweep.csv")
    h = pd.read_csv(REPORTS / "cost_sweep_holdout.csv")
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(v))
    ax.bar(x - 0.2, v["net_sharpe"], 0.4, color=BLUE, label="Validation")
    ax.bar(x + 0.2, h["net_sharpe"], 0.4, color=NAVY, label="Holdout")
    ax.set_xticks(x, [f"{m}x" for m in v["cost_multiple"]])
    ax.set(title=f"Do costs kill it? Break-even vs zero: {VAL['breakeven_bp_per_leg_vs_zero']:.0f} / {HOLD['breakeven_bp_per_leg_vs_zero']:.0f} bp per leg", xlabel="Cost multiple (0x = gross, 1x = assumed)", ylabel="Net Sharpe")
    ax.legend(frameon=False)
    save(fig, "fig3_gross_to_net.png")


def fig4_regime():
    v = pd.read_csv(REPORTS / "regimes_validation.csv")
    h = pd.read_csv(REPORTS / "regimes_holdout.csv")
    m = v.merge(h, on=["vol", "trend"], how="outer", suffixes=("_val", "_hold"))
    m["state"] = m["vol"] + " / " + m["trend"]
    m = m.sort_values(["trend", "vol"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(m))
    ax.barh(y + 0.2, m["sharpe_val"], 0.4, color=BLUE, label="Validation")
    ax.barh(y - 0.2, m["sharpe_hold"], 0.4, color=NAVY, label="Holdout")
    ax.set_yticks(y, m["state"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Where does it fail? Net Sharpe by market state", xlabel="Net Sharpe")
    ax.legend(frameon=False)
    save(fig, "fig4_regime_slice.png")
    return m


def fig5_shap():
    """Linear SHAP for the deployed Ridge, plus coefficient stability across the annual refits."""
    alpha_by_fold = pd.read_csv(REPORTS / "fold_protocol.csv").set_index("fold")["alpha"]
    panel = load_panel(allow_holdout=True)
    calendar = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel = add_ranks(panel, SIGNALS)
    feats = HOLD["frozen"]["signals"]
    rows = []
    for year, alpha in alpha_by_fold.items():
        tr = thin(purged_train(panel, pd.Timestamp(year, 1, 1), calendar).dropna(subset=["label"]), calendar)
        m = Ridge(alpha=alpha).fit(tr[feats].fillna(0), tr["y"])
        rows.append(dict(fit=str(year), **dict(zip(feats, m.coef_))))
    a = panel[panel["date"] < HOLD_START].iloc[:0]
    pos0 = calendar[calendar < HOLD_START][-1]
    tr = thin(panel[(panel["date"] >= TRAIN_START) & (panel["date"] < pos0) & (panel["label_end"] < pos0)].dropna(subset=["label"]), calendar)
    m = Ridge(alpha=HOLD["frozen"]["alpha"]).fit(tr[feats].fillna(0), tr["y"])
    rows.append(dict(fit="holdout model", **dict(zip(feats, m.coef_))))
    coefs = pd.DataFrame(rows)
    coefs.round(6).to_csv(REPORTS / "ridge_coefficients.csv", index=False)
    x = panel[(panel["date"] >= HOLD_START)][feats].fillna(0).sample(6000, random_state=1)
    values = m.coef_ * (x - tr[feats].fillna(0).mean())
    shap.summary_plot(values.to_numpy(), x, show=False, plot_size=(7.6, 3), color_bar_label="Feature rank")
    plt.title("Does the model read as the mechanism says? Ridge SHAP, holdout")
    plt.gcf().savefig(FIGS / "fig5_shap.png", dpi=170, bbox_inches="tight")
    plt.close("all")
    return coefs


def fig6_drawdown():
    h = pd.read_parquet(PROC / "nav_holdout.parquet")["strategy"].dropna()
    v = pd.read_parquet(PROC / "nav_validation.parquet")["strategy"].dropna()
    r = pd.concat([v.pct_change().dropna(), h.pct_change().dropna()])
    nav = (1 + r).cumprod()
    dd = nav / nav.cummax() - 1
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.fill_between(dd.index, dd.to_numpy() * 100, 0, color=BLUE, alpha=0.45)
    for level, word in ((5, "watch"), (10, "halve"), (15, "halt"), (20, "unwind")):
        ax.axhline(-level, color=RED, lw=0.8, ls=":")
        ax.text(dd.index[0], -level + 0.4, f"{level}% {word}", color=RED, fontsize=8)
    ax.axvline(HOLD_START, color=RED, ls="--", lw=1)
    ax.set(title="When would the kill-switch ladder have triggered?", ylabel="Drawdown (%)")
    save(fig, "fig6_drawdown.png")


def diagnostics():
    """Why did it trail 0050? Descriptive only: nothing here feeds back into the strategy."""
    op = pd.read_parquet(PROC / "open.parquet")
    a, b = "2024-11-01", "2026-10-30"
    ret = (op.loc[:b].ffill().iloc[-1] / op.loc[a:].ffill().iloc[0] - 1)
    panel = load_panel(allow_holdout=True)
    me = panel[(panel["date"] >= pd.Timestamp("2024-10-31")) & panel["month_end"]]
    s = pd.read_parquet(PROC / "scores_holdout.parquet")
    d = me.merge(s[["date", "code", DEPLOYED]], on=["date", "code"])
    d["top"] = d.groupby("date")[DEPLOYED].rank(pct=True, ascending=False) <= 0.10
    m = d.groupby("date").agg(names=("code", "size"), held=("top", "sum"))
    has_tsmc = d[d["code"] == "2330"].set_index("date")["top"]
    close = pd.read_parquet(PROC / "close.parquet")
    r = pd.read_parquet(PROC / "nav_holdout.parquet")
    daily = r[["strategy", "etf_0050"]].dropna().pct_change().dropna()
    beta = daily["strategy"].rolling(63).cov(daily["etf_0050"]) / daily["etf_0050"].rolling(63).var()
    out = dict(tsmc_return=float(ret["2330"]), etf_0050_return=float(ret["0050"]), months_tsmc_in_top_decile=int(has_tsmc.sum()), months=int(len(has_tsmc)),
               median_universe_stock_return=float(ret[d["code"].unique()].median()), rolling_beta_mean=float(beta.mean()), rolling_beta_last=float(beta.dropna().iloc[-1]))
    return out


def headline(fold_share):
    V, H = VAL["validation"], HOLD["metrics"]["strategy"]
    icv = pd.read_csv(REPORTS / "model_summary.csv").set_index("model").loc[DEPLOYED]
    ich = HOLD["ic"][DEPLOYED]
    fold = pd.read_csv(REPORTS / "ic_by_fold.csv")
    B = HOLD["metrics"]["etf_0050"]
    bv = VAL["benchmark_0050"]
    rows = [
        ("Signal", "Daily IC [HAC 95% CI]", f"{icv['mean']:.3f} [{icv['low']:.3f}, {icv['high']:.3f}]", f"{ich['mean']:.3f} [{ich['low']:.3f}, {ich['high']:.3f}]", "--"),
        ("Signal", "ICIR; positive folds", f"{icv['icir']:.2f}; {int((fold[DEPLOYED] > 0).sum())}/7", f"{ich['icir']:.2f}; n/a", "--"),
        ("Portfolio", "Gross Sharpe", f"{VAL['gross_sharpe']:.2f}", f"{HOLD['gross']['sharpe']:.2f}", "--"),
        ("Portfolio", "Max drawdown", f"{V['max_drawdown']:.1%}", f"{H['max_drawdown']:.1%}", f"{bv['max_drawdown']:.1%} / {B['max_drawdown']:.1%}"),
        ("Portfolio", "Two-way turnover per year", f"{V['turnover_two_way']:.1f}x", f"{H['turnover_two_way']:.1f}x", "--"),
        ("Costs", "Net Sharpe [95% CI]", f"{V['sharpe']:.2f} [{V['sharpe_low']:.2f}, {V['sharpe_high']:.2f}]", f"{H['sharpe']:.2f} [{H['sharpe_low']:.2f}, {H['sharpe_high']:.2f}]", f"{bv['sharpe']:.2f} / {B['sharpe']:.2f}"),
        ("Costs", "Net CAGR", f"{V['cagr']:.1%}", f"{H['cagr']:.1%}", f"{bv['cagr']:.1%} / {B['cagr']:.1%}"),
        ("Costs", "Break-even bp per leg, vs 0050", f"{VAL['breakeven_bp_per_leg_vs_0050']:.0f}", f"{HOLD['breakeven_bp_per_leg_vs_0050']:.0f}", "--"),
        ("Overlay", "Net Sharpe; max drawdown", f"{VAL['overlay']['sharpe']:.2f}; {VAL['overlay']['max_drawdown']:.1%}", f"{HOLD['metrics']['overlay']['sharpe']:.2f}; {HOLD['metrics']['overlay']['max_drawdown']:.1%}", "--"),
        ("Inference", f"Deflated Sharpe (trials = {HOLD['n_trials']})", f"{VAL['deflated_sharpe']:.2f}", f"{HOLD['deflated_sharpe']:.2f}", "--"),
        ("Inference", "Minimum track record, years", f"{VAL['min_track_record_years']:.1f}", f"{HOLD['min_track_record_years']:.1f}", "--"),
    ]
    t = pd.DataFrame(rows, columns=["Stage", "Metric", "Validation", "Holdout", "0050 (val / hold)"])
    t.to_csv(REPORTS / "headline_table.csv", index=False)
    (REPORTS / "headline_table.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    tb = ax.table(cellText=t.values, colLabels=t.columns, loc="center", cellLoc="left", colWidths=[0.1, 0.3, 0.22, 0.22, 0.2])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.75)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#d8dde3")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif c == 3:
            cell.set_facecolor("#eaf2f8")
    fig.savefig(FIGS / "headline_table.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return t


def main():
    FIGS.mkdir(exist_ok=True, parents=True)
    fig1_equity()
    share = fig2_rolling_ic()
    fig3_cost()
    regimes = fig4_regime()
    coefs = fig5_shap()
    fig6_drawdown()
    t = headline(share)
    diag = diagnostics()
    diag["rolling_ic_positive_share"] = share
    (REPORTS / "diagnostics.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(t.to_string(index=False)); print(diag); print(coefs.round(5)); print(regimes[["state", "sharpe_val", "sharpe_hold", "days_hold", "ann_return_hold", "worst_drawdown_hold"]].round(3))
    print(pd.read_csv(REPORTS / "cost_sweep_holdout.csv").round(3))


if __name__ == "__main__":
    main()
