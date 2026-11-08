"""Build the Proposal as PDF and Word from one source. Tables read result files; prose is written by hand.

Student names and IDs live in docs/authors.local.json, which stays out of git.

    python docs/build_proposal.py
"""
import csv
import json
import os
import re
from html import unescape
from pathlib import Path

import matplotlib
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"
OUT = ROOT / "docs" / "ML4T_Midterm_Proposal.pdf"
DOCX = ROOT / "docs" / "ML4T_Midterm_Proposal.docx"
AUTHORS = ROOT / "docs" / "authors.local.json"
A = json.loads(AUTHORS.read_text(encoding="utf-8")) if AUTHORS.exists() else dict(members=[], date="", repo="")
NAVY, GREY, PALE, FILL = colors.HexColor("#0a3d62"), colors.HexColor("#c9d0d8"), colors.HexColor("#eef3f7"), colors.HexColor("#fff3bf")

ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
for name, file in (("Body", "DejaVuSans.ttf"), ("Body-Bold", "DejaVuSans-Bold.ttf"), ("Body-It", "DejaVuSans-Oblique.ttf"), ("Body-BI", "DejaVuSans-BoldOblique.ttf")):
    pdfmetrics.registerFont(TTFont(name, os.path.join(ttf, file)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-It", boldItalic="Body-BI")
pdfmetrics.registerFont(TTFont("CJK", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))

BODY = ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=15.5, spaceAfter=7, alignment=TA_LEFT)
H1 = ParagraphStyle("h1", fontName="Body-Bold", fontSize=14.5, leading=19, textColor=NAVY, spaceBefore=13, spaceAfter=7, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Body-Bold", fontSize=11, leading=15, textColor=NAVY, spaceBefore=7, spaceAfter=4, keepWithNext=1)
CAP = ParagraphStyle("cap", fontName="Body-It", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#444444"), spaceAfter=9)
CELL = ParagraphStyle("cell", fontName="Body", fontSize=8.6, leading=11.2)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Body-Bold", textColor=colors.white)
TINY = ParagraphStyle("tiny", parent=CELL, fontSize=7, leading=8.8)
LOG = ParagraphStyle("log", parent=CELL, fontSize=5.8, leading=7.2)
LOGH = ParagraphStyle("logh", parent=LOG, fontName="Body-Bold", textColor=colors.white)
TINYH = ParagraphStyle("tinyh", parent=TINY, fontName="Body-Bold", textColor=colors.white)
TITLE = ParagraphStyle("title", fontName="Body-Bold", fontSize=17, leading=22, textColor=NAVY, spaceAfter=5)
META = ParagraphStyle("meta", parent=BODY, textColor=colors.HexColor("#444444"), spaceAfter=2)
WIDTH = A4[0] - 4 * cm

story = []

# ---------- Word document, filled by the same helpers as the PDF ----------
word = Document()
_sec = word.sections[0]
_sec.page_width, _sec.page_height, _sec.orientation = Cm(21), Cm(29.7), WD_ORIENT.PORTRAIT
_sec.left_margin = _sec.right_margin = Cm(2)
_sec.top_margin = _sec.bottom_margin = Cm(2)
WORD_WIDTH_CM = 17.0
NAVY_RGB = RGBColor(0x0A, 0x3D, 0x62)


def set_font(style, size, bold=None, color=None):
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("w:ascii", "w:hAnsi", "w:cs"):
        fonts.set(qn(key), "Calibri")
    fonts.set(qn("w:eastAsia"), "Microsoft JhengHei")
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(key), None)


set_font(word.styles["Normal"], 11)
word.styles["Normal"].paragraph_format.space_after = Pt(7)
word.styles["Normal"].paragraph_format.line_spacing = 1.3
set_font(word.styles["Title"], 22, True, NAVY_RGB)
set_font(word.styles["Heading 1"], 15, True, NAVY_RGB)
word.styles["Heading 1"].paragraph_format.space_before = Pt(14)
word.styles["Heading 1"].paragraph_format.space_after = Pt(6)
set_font(word.styles["Heading 2"], 12, True, NAVY_RGB)
word.styles["Heading 2"].paragraph_format.space_before = Pt(8)
word.styles["Heading 2"].paragraph_format.space_after = Pt(4)
set_font(word.styles["Caption"], 9, False, RGBColor(0x44, 0x44, 0x44))
word.styles["Caption"].font.italic = True
set_font(word.styles["List Bullet"], 11)

TAG = re.compile(r"(</?(?:b|i|super|font)\b[^>]*>)")


def add_runs(par, markup, size=None, color=None):
    """Turn the small markup the PDF uses into Word runs."""
    bold = ital = sup = hi = False
    stack = []
    for part in TAG.split(markup):
        if not part:
            continue
        if TAG.fullmatch(part):
            t = part.lower()
            if t == "<b>":
                bold = True
            elif t == "</b>":
                bold = False
            elif t == "<i>":
                ital = True
            elif t == "</i>":
                ital = False
            elif t == "<super>":
                sup = True
            elif t == "</super>":
                sup = False
            elif t.startswith("<font"):
                stack.append((hi, bold))
                hi = hi or "backcolor" in t
                bold = bold or "body-bold" in t
            else:
                hi, bold = stack.pop()
            continue
        r = par.add_run(unescape(part).replace("\xa0", " "))
        r.bold, r.italic = bold or None, ital or None
        r.font.superscript = sup or None
        if hi:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if size:
            r.font.size = Pt(size)
        if color is not None:
            r.font.color.rgb = color


def P(t, s=None):
    s = s or BODY
    story.append(Paragraph(t, s))
    style = {id(TITLE): "Title", id(H1): "Heading 1", id(H2): "Heading 2", id(CAP): "Caption"}.get(id(s), "Normal")
    par = word.add_paragraph(style=style)
    add_runs(par, t)


H = lambda t: P(t, H1)
S = lambda t: P(t, H2)


def bullets(items):
    story.append(ListFlowable([ListItem(Paragraph(i, BODY), leftIndent=11) for i in items], bulletType="bullet", start="•", bulletFontSize=7, leftIndent=11))
    story.append(Spacer(1, 3))
    for i in items:
        add_runs(word.add_paragraph(style="List Bullet"), i)


def shade(cell, fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(el)


def word_table(rows, widths, size):
    t = word.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Cm(w * WORD_WIDTH_CM)
    for i, row in enumerate(rows):
        tr = t.rows[i]
        pr = tr._tr.get_or_add_trPr()
        el = OxmlElement("w:cantSplit")
        pr.append(el)
        if i == 0:
            head = OxmlElement("w:tblHeader")
            pr.append(head)
        for j, value in enumerate(row):
            cell = tr.cells[j]
            cell.width = Cm(widths[j] * WORD_WIDTH_CM)
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(1)
            par.paragraph_format.line_spacing = 1.0
            par.paragraph_format.keep_with_next = i == 0
            if i == 0:
                add_runs(par, f"<b>{value}</b>", size, RGBColor(0xFF, 0xFF, 0xFF))
                shade(cell, "0A3D62")
            else:
                add_runs(par, str(value), size)
                if i % 2 == 0:
                    shade(cell, "EEF3F7")
    word.add_paragraph().paragraph_format.space_after = Pt(4)


def table(rows, widths, style=(CELL, HEAD), repeat=1):
    body, head = style
    data = [[Paragraph(str(c), head if i == 0 else body) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=repeat)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.3, GREY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)
    story.append(Spacer(1, 8))
    word_table(rows, widths, 9 if style[0] is CELL else 7.5 if style[0] is TINY else 6.5)


def figure(name, ratio, caption, width=0.72):
    w = WIDTH * width
    story.append(KeepTogether([Image(str(FIG / name), width=w, height=w / ratio), Paragraph(caption, CAP)]))
    par = word.add_paragraph()
    par.alignment = 1
    par.paragraph_format.keep_with_next = True
    par.add_run().add_picture(str(FIG / name), width=Cm(WORD_WIDTH_CM * width))
    add_runs(word.add_paragraph(style="Caption"), caption)


def page_break():
    story.append(PageBreak())
    word.add_page_break()


def pct(x, d=1):
    return f"{x * 100:.{d}f}%".replace("-", "−")


def n(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def ci(a, b, d=2):
    return f"[{n(a, d)}, {n(b, d)}]"


TODO = lambda t: f'<font backColor="#fff3bf"><b>[{t}]</b></font>'

V = json.loads((REP / "validation_summary.json").read_text(encoding="utf-8"))
H_ = json.loads((REP / "holdout_results.json").read_text(encoding="utf-8"))
D = json.loads((REP / "diagnostics.json").read_text(encoding="utf-8"))
models = pd.read_csv(REP / "model_summary.csv").set_index("model")
pairs = pd.read_csv(REP / "paired_comparison.csv")
folds = pd.read_csv(REP / "ic_by_fold.csv")
proto = pd.read_csv(REP / "fold_protocol.csv")
triage = pd.read_csv(REP / "triage.csv")
sweep_v, sweep_h = pd.read_csv(REP / "cost_sweep.csv"), pd.read_csv(REP / "cost_sweep_holdout.csv")
back = pd.read_csv(REP / "backtest_validation.csv").set_index("strategy")
alloc = pd.read_csv(REP / "allocators.csv").set_index("allocator")
stress = pd.read_csv(REP / "stress.csv")
conf = pd.read_csv(REP / "conformal.csv")
coef = pd.read_csv(REP / "ridge_coefficients.csv")
shap_f = pd.read_csv(REP / "shap_summary.csv")
SM, VM = H_["metrics"]["strategy"], V["validation"]
HIC = H_["ic"][H_["deployed"]]

# Review support: trial-count sensitivity, IC change, drawdown calibration (all read from result files).
import sys
sys.path.insert(0, str(ROOT / "stages"))
from simulator import deflated_sharpe  # noqa: E402
_led = pd.read_csv(ROOT / "run_log.csv")
_dev = _led[_led.trial_family != "holdout"]
_dist = _dev.drop_duplicates(["trial_family", "trial", "run"])
_gbm = _dist.trial_family == "model_lgbm"
TR_NON_GBM = int((~_gbm).sum())
TR_OPTUNA = int(pd.to_numeric(_dist.loc[_gbm, "run"]).sum())
TR_OPTUNA_ALL = int(pd.to_numeric(_dev.loc[_dev.trial_family == "model_lgbm", "run"]).sum())
TR_RIDGE_ROWS = int((_dist.trial_family == "model_ridge").sum())
_gb = _dev[_dev.trial_family == "model_lgbm"].groupby("trial").config
GBM_CHANGED, GBM_FOLDS = int((_gb.first() != _gb.last()).sum()), int(_gb.ngroups)
_rg = _dev[_dev.trial_family == "model_ridge"].groupby("trial").config
RIDGE_CHANGED = int((_rg.first() != _rg.last()).sum())
TR_RIDGE_GRID = 9 * 7 * 2
TR_CONSERVATIVE = TR_NON_GBM - TR_RIDGE_ROWS + TR_OPTUNA_ALL + TR_RIDGE_GRID
TRIAL_FAMILIES = _dist[~_gbm].groupby("trial_family").size()
def _dsr(m, nt):
    return deflated_sharpe(m["sharpe"], m["years"], m["skew"], m["kurtosis"], nt, V["sharpe_variance"])
DSR_ROWS = [(nt, _dsr(VM, nt), _dsr(SM, nt)) for nt in sorted({V["n_trials"], TR_OPTUNA_ALL + TR_NON_GBM, TR_CONSERVATIVE, 500})]
IC_D = models.loc["ridge", "mean"] - HIC["mean"]
IC_SE = (models.loc["ridge", "se"] ** 2 + HIC["se"] ** 2) ** 0.5
_nvs = pd.read_parquet(ROOT / "data/processed/nav_validation.parquet")
_ddv = _nvs["strategy"].dropna() / _nvs["strategy"].dropna().cummax() - 1
_eps, _in, _lo = [], False, 0.0
for _x in _ddv:
    if _x < 0:
        _lo = min(_lo, _x) if _in else _x
        _in = True
    elif _in:
        _eps.append(_lo); _in = False
if _in:
    _eps.append(_lo)
DD_EPISODES = sorted(e for e in _eps if e < -0.20)
DD_Q01 = float(_ddv.quantile(0.01))
DD_WORST = min(_eps)
H_YEARS = SM["years"]

# ---------- Title ----------
P("Testing Momentum, Flow and Revenue Signals in Taiwan Stocks", TITLE)
P("Midterm proposal, Machine Learning for Algorithmic Trading. Instructor: Dr. Ting-Yu Chang.", META)
team = " · ".join(f"{i} <font name='CJK'>{n_}</font>" for i, n_ in A["members"]) or TODO("Team members")
P(f"Team: {team}", META)
P(f"Date: {A['date'] or TODO('Submission date')} &nbsp; Repository: {A['repo'] or TODO('Repository URL')}", META)
story.append(Spacer(1, 5))
P(f"<b>Summary.</b> This is one complete lap of the ML4T workflow on TWSE and TPEx common stocks. Screening rejected our original momentum and institutional-flow hypothesis. "
  f"The deployed model is a Ridge on two features: distance to the 52-week high and revenue acceleration. On the sealed holdout, run once, the pre-registered test <b>did not pass</b>: "
  f"IC {n(HIC['mean'], 3)} {ci(HIC['low'], HIC['high'], 3)}. Net Sharpe was {n(SM['sharpe'])} {ci(SM['sharpe_low'], SM['sharpe_high'])}, above the 12-1 rule "
  f"({n(H_['metrics']['rule']['sharpe'])}) and below 0050 ({n(H_['metrics']['etf_0050']['sharpe'])}). The fall in IC is not statistically distinguishable from zero, so we classify the outcome as inconclusive, closest to hold, and do not consider the strategy ready for capital.")

# ---------- 1 ----------
H("1. Hypothesis and sources of edge")
P("<b>Entry point:</b> mechanism-first. <b>Mechanism, written before any backtest:</b> Taiwanese investors underreact to firm-specific news, so 6–12-month winners keep outperforming for about one month. "
  "The drift is stronger when foreign and investment-trust institutions accumulate the stock and when monthly revenue accelerates.")
P("<b>Sources of edge</b> (categories from Table 6.1, as listed in the course guideline): informational advantage, because monthly revenue diffuses slowly, and flow predictability, because institutional buying persists.")
table([["Pre-registered falsification test", "Result on validation data"],
       ["Walk-forward IC 95% CI includes zero", "Not triggered for the deployed model. Triggered for 12-1 momentum and for both flow signals."],
       ["A 5–21-day timing placebo does not kill the IC", "<b>Triggered.</b> The IC of both surviving features remains when they are lagged 21 days."],
       ["Net Sharpe after Taiwan costs is below the 12-1 rule", f"Not triggered ({n(VM['sharpe'])} against {n(back.loc['rule', 'sharpe'])})."]], [0.48, 0.52])
P("<b>H1 is falsified.</b> Momentum and both institutional-flow signals failed triage, and the placebo condition was triggered for the features that survived. By our own pre-registered rule, H1 stops here.")
P("<b>H2 is exploratory and post hoc.</b> We continued because the assignment asks for one complete lap. The deployed model rests on a new hypothesis written <i>after</i> seeing validation data: "
  "investors anchor on the 52-week high and digest revenue acceleration slowly, so the two features act as slow state variables rather than timely news. "
  "Validation evidence cannot support H2, because the same data produced it. Only the sealed holdout tests it, and that test is reported in Section 7.")
P("<b>What could break this.</b> H2 was chosen after looking at the data. The two features may proxy for quality or size, which we did not control for directly.")

# ---------- 2 ----------
H("2. Trading setup v1 and evaluation protocol")
P("<font name='Body'>setup.yaml</font> was committed as <font name='Body-Bold'>a40d555</font> before any return was computed. Mechanics did not change after that, so the version stayed at v1.")
table([["Setup field", "Specification"],
       ["Universe", "TWSE and TPEx common stocks. Point-in-time: at least 240 of the last 252 days listed, close above NT$10, 63-day median value above NT$50M. No TWSE disposition or TPEx managed stocks. Delisted names kept."],
       ["Decision schedule", "Snapshot at the last trading day's close of each month, data treated as known at 15:00 Taipei. Fill at the next open."],
       ["Admissible information", "Prices and flows through t. Revenue from the first trading day after the 10th of the next month. Disposition status from its publication date."],
       ["Score to trade", "Long-only. Enter the top 10% by score; keep until the stock leaves the top 20%; equal weight. No new positions with fewer than 100 eligible names."],
       ["Constraints", "Fully invested, no leverage, at most 5% per name and 40% per sector."],
       ["Cost class", "Per-leg: 14.25 bp commission, 30 bp sell tax, half-spread 5/10/20 bp by median value. Round trip about 69–99 bp."]], [0.2, 0.8])
table([["Protocol commitment", "Value"],
       ["Training window", "Expanding from 2012-06, the first month with flow data; at least 5 years"],
       ["Step and retraining", "12 months; annual retrain"],
       ["Test period", "7 walk-forward folds, 2018-01 to 2024-10"],
       ["Label horizon; longest lookback", "21 trading days; 252 trading days"],
       ["Buffers", "Purge of 21 + 1 trading days, because the label ends at t+22. No embargo: training never follows validation."],
       ["Tuning", "Nested: last 252 training days, purged, with the fold's hyperparameters fixed before its test block"],
       ["Sealed holdout", "2024-11-01 to 2026-10-31, sealed 2026-11-05, one run"],
       ["Holdout refit", "Hyperparameters frozen at the 2024 fold's values. Fit once on labels ending by 2024-10-31; refit once on the first trading day after 2025-10-31 with labels that had fully ended. Coded in s06_holdout.py at bd6cece, before the run"],
       ["Confirmatory test", "Holdout IC Newey-West 95% lower bound above 0 AND net Sharpe at least the 12-1 rule"]], [0.3, 0.7])
S("Timeline of sealing (git commit times, Taipei, 2026-11-05)")
table([["Time", "Event"],
       ["14:49", "a40d555: setup.yaml committed with the validation start already at 2018-01 and the holdout dates"],
       ["15:58", "First row of run_log.csv (triage)"],
       ["16:01", "First validation backtest"],
       ["16:09", "bd6cece: validation results, pre-holdout audit, holdout code including the refit rule"],
       ["19:20", "Holdout run once (reports/holdout_touched.json); result committed at 19:22 as 24d9c3e"]], [0.15, 0.85])
P("<b>Disclosure.</b> The course plans six weekly checkpoints. We did not follow that pace: setup, validation, audit and holdout were all completed on one day. "
  "The sealing order still holds, since every commit above precedes the step that uses it, but a compressed schedule leaves less room to notice problems between steps. "
  "The move of the validation start from 2015 to 2018 happened in our planning notes of 2026-11-03, before setup.yaml was first committed, because flow data begins in 2012 and we wanted at least five training years.")
fam = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in TRIAL_FAMILIES.items())
P(f"<b>Search accounting.</b> <font name='Body'>run_log.csv</font> has {len(_led)} rows: one per run, with provenance, configuration, window, metric, decision and gate. "
  f"The trial count is not the row count. Our pre-written rule counts each distinct configuration once and each Optuna trial once, and does not count reruns after a bug fix: "
  f"{TR_NON_GBM} distinct runs ({fam}) plus {TR_OPTUNA} Optuna trials gives <b>{V['n_trials']}</b>.")
P(f"That rule is generous to us. The bug-fix rerun changed the selected LightGBM configuration in {GBM_CHANGED} of {GBM_FOLDS} folds and the Ridge alpha in {RIDGE_CHANGED}, so the rerun was a second search. "
  f"Counting it gives {TR_OPTUNA_ALL} Optuna trials. Each Ridge fit also scans a 9-point alpha grid. A conservative count of {TR_CONSERVATIVE} includes both. The Deflated Sharpe ratio stays below 0.95 under every count:")
table([["Trials assumed", "Deflated Sharpe, validation", "Deflated Sharpe, holdout"]] + [[f"{nt} " + {V['n_trials']: "(our rule, reported)", TR_OPTUNA_ALL + TR_NON_GBM: "(with bug-fix rerun)", TR_CONSERVATIVE: "(rerun + Ridge grid)"}.get(nt, "(stress)"), n(a, 3), n(b, 3)] for nt, a, b in DSR_ROWS], [0.34, 0.33, 0.33])
P("<b>What could break this.</b> The trial count depends on a rule we wrote ourselves. "
  "Adjusted-price checks displayed TSMC and 0050 daily moves dated in the holdout window; no strategy statistic was computed. The two-month gap at the end of validation is explained in Section 5.")

# ---------- 3 ----------
H("3. Data")
P("Route B, own data. If we cannot state when a value became knowable, we do not use it.")
table([["Data", "Source and download", "Timestamp and adjustment"],
       ["Daily prices, corporate actions, TAIEX, 0050", "tw.db, built from TWSE and TPEx official files; snapshot 2026-11-05", "Known at 15:00. Raw prices stored; adjustment factors from dividend, capital-reduction and par-change events"],
       ["Foreign and trust net buying", f"{TODO('Vendor / source')}; files dated 2026-11-03", "Covers one trading day, known after its close. Exact match with official files on 13 sampled days"],
       ["Monthly revenue", "MOPS official pages; downloaded 2026-11-05", "First trading day after the 10th. Latest published value: restatements not separated"],
       ["Disposition notices", "TWSE announcements; downloaded 2026-11-05", "Effective from publication date"],
       ["Industry codes", "TWSE and TPEx Open API; downloaded 2026-11-05", "Current values only"]], [0.27, 0.33, 0.4])
table([["Level", "Point-in-time check", "Result", "Direction of any bias"],
       ["Values", "Features recomputed with data cut at T match the panel (4 dates, 7 features)", "PASS", "None found"],
       ["Values", "Revenue updates only after the 10th (checked on TSMC)", "PASS, with limit", "Late filers may look known early; restatements could flatter revenue features"],
       ["Values", "TPEx flows before 2014-12 compared with official files", "NOT VERIFIED", "Unknown; flows were not deployed"],
       ["Universe", "Eligibility uses information at t; 84 eligible names later delisted", "PASS", "None found"],
       ["Universe", "TPEx disposition and all full-delivery status", "FAIL, disclosed", "Restricted names stay eligible; fills may be optimistic"],
       ["Identifiers", "Reused codes restart eligibility through the listed-days rule", "PASS, with limit", "A reused code could inherit old history"],
       ["Identifiers", "Industry codes as of each date", "FAIL, disclosed", "Sector cap may differ from what was known then"]], [0.11, 0.44, 0.14, 0.31])
P("<b>Quality report.</b> 2,183 stocks appear; eligible names per month range from 188 to 725, median 354. OHLC violations 0; duplicates 0 by primary key. Listed stock-days without a trade: 1.26%, left empty and never forward-filled for fills. "
  "195 names left the sample, 7 to 20 a year. Feature missing share is 0–0.2% for price features, 1.5% for flows and 8.3% for revenue_yoy. The dataset is partitioned Parquet rebuilt by <font name='Body'>s01_build_dataset.py</font>. No synthetic data is used.")
figure("corporate_action_check.png", 2.5, "Figure A. Do adjusted prices handle dividends? Yes: TSMC's adjusted series sits below the raw series before each dividend, shown up to the holdout.", 0.62)
figure("coverage.png", 2.5, "Figure B. How many stocks were eligible? The universe grows from under 200 to over 700 names, so early folds have fewer names to rank.", 0.62)
P("<b>What could break this.</b> Excluding disposition stocks only on TWSE is the largest gap. A stock delisted at a loss below its last close would be valued too high, because the simulator sells at the last close.")

# ---------- 4 ----------
H("4. Labels, features and triage")
val_days = int(models.loc["ridge", "n"])
P(f"<b>Label.</b> Fixed-horizon continuous return: adjusted Open(t+22) / Open(t+1) − 1, anchored to the next-open fill and checked on 5,000 random rows. Daily labels overlap: the {val_days:,} validation days have about {val_days // 21} independent 21-day periods, "
  "which is why every IC interval uses Newey-West with 21 lags. Holdout IC covers 465 days, about 22 independent periods. "
  "<b>Model target.</b> The models do not fit the raw return. Each day the label is ranked across stocks and centered: percentile rank − 0.5, from −0.5 to 0.5. This removes the market's common move and outliers. "
  "Backtests use the raw return, and IC is a rank correlation, so neither depends on this choice.")
table([["Feature", "Role", "Driver hypothesis", "Lookback / cadence", "Representation", "Known failure mode"],
       ["mom_12_1", "Signal", "Persistence", "t−252 to t−21; daily", "Return ÷ 63d vol; rank", "Momentum crashes"],
       ["mom_6_1", "Signal", "Persistence", "t−126 to t−21; daily", "Return ÷ 63d vol; rank", "Momentum crashes"],
       ["dist_52w_high", "Signal", "Anchoring", "252d; daily", "Close ÷ 252d high; rank", "All names near highs"],
       ["rev_21d", "Signal", "Liquidity provision", "21d; daily", "−return; rank", "News-driven moves"],
       ["foreign_net_buy_20d", "Signal", "Flow persistence", "20d; daily", "Net shares ÷ 20d volume; rank", "Index rebalances, FX shocks"],
       ["trust_net_buy_20d", "Signal", "Flow persistence", "20d; daily", "Net shares ÷ 20d volume; rank", "Quarter-end window dressing"],
       ["revenue_yoy", "Signal", "Slow diffusion", "3 months; monthly", "Mean YoY; rank", "One-off revenue, M&amp;A"],
       ["revenue_accel", "Signal", "Slow diffusion", "4 months; monthly", "YoY − prior 3m mean; rank", "Base effects, seasonality"],
       ["market_vol_pctile", "State", "Risk environment", "20d vol in 252d", "Percentile; not cross-sectional", "Single shock dominates"],
       ["taiex_trend", "State", "Market trend", "200d; daily", "Level vs moving average", "Sideways markets"]], [0.2, 0.07, 0.15, 0.17, 0.22, 0.19])
rows = [["Signal", "IC", "95% CI", "ICIR", "Folds, same sign", "Placebo 21d", "Quintile corr.", "Q5−Q1 / cost, bp", "BH 5%", "Decision"]]
for _, x in triage.sort_values("ic_mean", ascending=False).iterrows():
    rows.append([x["feature"], n(x["ic_mean"], 3), ci(x["ic_low"], x["ic_high"], 3), n(x["icir"]), f"{x['folds_same_sign'] * 7:.0f}/7", n(x["placebo_21"], 3), n(x["quintile_spearman"], 1),
                 f"{n(x['q5_q1_bp_21d'], 0)} / {n(x['cost_bp_month'], 0)}", "Yes" if x["bh_reject"] else "No", x["decision"]])
table(rows, [0.19, 0.07, 0.15, 0.06, 0.07, 0.08, 0.08, 0.1, 0.05, 0.1], (TINY, TINYH))
P("The table is the five gates in order: correctness, association, shape, feasibility, search. Decision rules were in the code before the table was read. STOP means the interval lies below zero, or both the association and search gates failed. "
  "No feature passes every gate, so none is PROCEED. The highest rank correlation among signals is 0.59 (12-1 and 6-1), so no feature was removed as a duplicate. The two REVISE features went forward unrevised because revising them would add trials.")
story.append(CondPageBreak(5 * cm))
S("Mechanism plausibility")
table([["Feature", "Confounders (control)", "Mediator (do not control)", "Collider (never control)"],
       ["dist_52w_high", "Sector, size, market beta", "Recent return", "Inclusion in the eligible set"],
       ["revenue_accel", "Sector, seasonality", "Analyst revisions", "Inclusion in the eligible set"],
       ["rev_21d", "Sector, market beta", "Short-term flows", "Inclusion in the eligible set"]], [0.2, 0.3, 0.25, 0.25])
P("This is the DAG in table form; it states our assumptions and is not estimated. Of the four cheap checks, we ran the timing placebo, a sector-neutral IC (0.030 and 0.023 for the two survivors), and the volatility-tercile split. "
  "dist_52w_high falls to about zero in high volatility, while revenue_accel stays positive in all terciles. <b>Not done:</b> a market-beta control and event-time alignment around revenue dates. No Double ML estimate.")
P("<b>What could break this.</b> Selecting 2 of 8 features on validation data inflates validation results. Failing the placebo means these features may describe slow characteristics. Size is an uncontrolled confounder.")

# ---------- 5 ----------
H("5. Models, validation, interpretation and uncertainty")
P("One yardstick: mean daily Spearman IC with a Newey-West 95% CI. The rule fixed in the code: climb a rung only if the paired difference's interval lies above zero. Ridge uses the two surviving signals; LightGBM adds the two state variables.")
rows = [["Rung", "Configuration", "Validation IC [95% CI]", "ICIR", "Positive folds"]]
label = dict(rule=("1. 12-1 rule", "Non-ML; top decile"), ridge=("2. Ridge (deployed)", "Alpha 10<super>−3</super>–10<super>5</super>; per-date ranks; state winsor limits fitted on training folds"), lgbm=("3. LightGBM", f"Depth 3–4, MAE, early stopping, {TR_OPTUNA} Optuna trials ({TR_OPTUNA_ALL} with the bug-fix rerun)"), ensemble=("Optional: ensemble", "Mean z-score of rungs 1–3"))
for m, (a, b) in label.items():
    x = models.loc[m]
    rows.append([a, b, f"{n(x['mean'], 3)} {ci(x['low'], x['high'], 3)}", n(x["icir"]), f"{(folds[m] > 0).sum()}/7"])
table(rows, [0.2, 0.4, 0.21, 0.08, 0.11])
rows = [["Paired comparison", "Mean ΔIC", "95% CI", "Decision"]]
for (_, x), d in zip(pairs.iterrows(), ("Climb to Ridge", "Stay with Ridge", "Not adopted")):
    rows.append([x["step"], n(x["mean"], 3), ci(x["low"], x["high"], 3), d])
table(rows, [0.4, 0.15, 0.25, 0.2])
rows = [["Fold test block", "Last training label", "Alpha", "Rule IC", "Ridge IC", "LightGBM IC"]]
for (_, f), (_, p) in zip(folds.iterrows(), proto.iterrows()):
    rows.append([f"{p['test_start']} to {p['test_end']}", p["last_label_end"], f"{p['alpha']:,.0f}", n(f["rule"], 3), n(f["ridge"], 3), n(f["lgbm"], 3)])
table(rows, [0.3, 0.2, 0.11, 0.13, 0.13, 0.13])
P("The last validation decision is 2024-08-30: the September decision's label would end inside the holdout. That position is held to the end of October 2024.")
figure("fig2_rolling_ic.png", 2.529, f"Figure 2. Is the signal stable? Not fully: the rolling 3-month IC of the deployed model is positive {D['rolling_ic_positive_share'] * 100:.0f}% of the time and has deep negative spells. Dashed line: holdout starts.", 0.72)
figure("fig5_shap.png", 2.382, "Figure 5. Does the model read as the mechanism says? Yes: higher ranks of both features raise the score. For a linear model this restates the coefficients.", 0.6)
lo, hi = coef["dist_52w_high"].min(), coef["dist_52w_high"].max()
ratio = coef["dist_52w_high"] / coef["revenue_accel"]
w = shap_f[shap_f["feature"] == "dist_52w_high"].sort_values("direction").iloc[0]
P(f"<b>Interpretation.</b> Ridge signs are positive in every refit. Coefficient size is not stable: dist_52w_high runs from {lo:.4f} to {hi:.4f}, because alpha switches between 10<super>4</super> and 10<super>5</super>, and alpha hit the top of its grid in 5 of 7 folds. Only the ratio of the two coefficients affects the ranking, and it stays between {ratio.min():.2f} and {ratio.max():.2f}. "
  f"<b>A surprising attribution:</b> in LightGBM's {int(w['fold'])} fold, dist_52w_high's SHAP direction is {n(w['direction'])}, the opposite sign of its average. A long-only anchoring story does not predict this; it may reflect interaction with market state. "
  "Also surprising: revenue_accel works while revenue_yoy does not, which suggests surprise matters more than level. SHAP is descriptive, not causal.")
rc, gc = conf[conf["model"] == "ridge"], conf[conf["model"] == "lgbm"]
P(f"<b>Uncertainty.</b> Conformal 80% intervals have a half-width of about 0.40 on a rank label running from −0.5 to 0.5, about what the label's own distribution gives, so they add little. Ridge coverage is {rc['coverage'].min():.3f}–{rc['coverage'].max():.3f}, with one fold slightly under 80%; LightGBM covers {gc['coverage'].min():.3f}–{gc['coverage'].max():.3f}, under 80% in {int(gc['below_80'].sum())} of 7 folds.")
P("<b>What could break this.</b> Ridge was chosen by a rule, not because it had the highest IC, and LightGBM later had a higher holdout IC. Alpha at the grid boundary means the search range may have been too narrow.")

# ---------- 6 ----------
H("6. Backtest protocol, baseline, portfolio, costs and risk")
table([["Component", "Fixed before interpretation"],
       ["Signal timing", "Month-end close signal, next-open fill. Same-bar fills are not possible in the code."],
       ["Rebalance and holding", "Monthly; holding depends on the 20% buffer"],
       ["Position sizing", "Equal weight; 5% name cap, 40% sector cap; excess goes pro rata, then stays in cash"],
       ["Fill assumptions", "Adjusted open. No fill on a missing open or a limit-locked open: buys are skipped for the month, sells retry each open. Stopped stocks sold at last close."],
       ["Constraints", "Checked inside the simulation. Buys are scaled to cash on hand."],
       ["Engine", "Own daily event-style simulator (stages/simulator.py), so that delayed sells and overlays can affect the next action"]], [0.22, 0.78])
rows = [["Strategy, validation", "CAGR", "Vol", "Net Sharpe [95% CI]", "Max DD", "Two-way turnover"]]
names = dict(ridge="Ridge (deployed)", lgbm="LightGBM", ensemble="Ensemble", etf_0050="0050", equal_weight_universe="Equal-weight universe", rule="12-1 rule")
for k, v in names.items():
    x = back.loc[k]
    rows.append([v, pct(x["cagr"]), pct(x["vol"]), f"{n(x['sharpe'])} {ci(x['sharpe_low'], x['sharpe_high'])}", pct(x["max_drawdown"]), "—" if pd.isna(x["turnover_two_way"]) else f"{x['turnover_two_way']:.1f}x"])
table(rows, [0.28, 0.1, 0.1, 0.26, 0.1, 0.16])
P(f"These rows start at the first decision, 2018-01-02. The headline table and the rest of this report start at the first fill, {V['first_trade']}; that is why turnover reads {back.loc['ridge', 'turnover_two_way']:.1f}x here and {VM['turnover_two_way']:.1f}x elsewhere. Sharpe is not net of a risk-free rate.")
figure("fig3_gross_to_net.png", 2.111, f"Figure 3. Do costs kill it? No: net Sharpe stays positive at 4x assumed costs in both periods. Break-even against zero is {V['breakeven_bp_per_leg_vs_zero']:.0f} / {H_['breakeven_bp_per_leg_vs_zero']:.0f} bp per leg; against 0050 it is {V['breakeven_bp_per_leg_vs_0050']:.0f} / {n(H_['breakeven_bp_per_leg_vs_0050'], 0)} bp.", 0.62)
P(f"<b>Cost regime.</b> Two-way turnover is {VM['turnover_two_way']:.1f}x a year with a {VM['holding_months']:.1f}-month holding period. Against an assumed average of about 40 bp per leg, a break-even of {V['breakeven_bp_per_leg_vs_zero']:.0f} bp is the comfortable zone at a monthly cadence when measured against zero. It is a gray zone against 0050, and the holdout's negative figure means no cost level closes that gap.")
ei, hh = alloc.loc["equal"], alloc.loc["inverse_vol"]
table([["Allocator, same signal and covariance window", "Sharpe", "Max DD", "Turnover", "HHI", "Effective names"],
       ["Equal weight (null, deployed)", n(ei["sharpe"]), pct(ei["max_drawdown"]), f"{ei['turnover']:.1f}x", n(ei["hhi"], 3), f"{ei['effective_names']:.0f}"],
       ["Inverse volatility", n(hh["sharpe"]), pct(hh["max_drawdown"]), f"{hh['turnover']:.1f}x", n(hh["hhi"], 3), f"{hh['effective_names']:.0f}"]], [0.42, 0.1, 0.1, 0.11, 0.1, 0.17])
P(f"Best-minus-worst spread is {hh['sharpe'] - ei['sharpe']:.2f} Sharpe in validation and {H_['metrics']['inverse_vol']['sharpe'] - SM['sharpe']:.2f} in the holdout, second order as expected. No mean-variance optimizer was used.")
story.append(CondPageBreak(5 * cm))
S("Risk term sheet")
table([["Risk", "Proxy", "Control", "Frequency"],
       ["Market", "Rolling beta, drawdown", "Volatility overlay and ladder (reported separately)", "Monthly"],
       ["Factor", "Not measured", "None; a disclosed gap", "—"],
       ["Leverage", "Gross exposure", "No leverage, enforced in the simulator", "Each fill"],
       ["Concentration", "Sector weight, HHI", "5% name cap, 40% sector cap", "Monthly"],
       ["Liquidity", "63-day median value", "NT$50M screen, tiered spread, no limit-locked fills", "Monthly"],
       ["Model", "Rolling IC, PSI (design only)", "Retrain and retire rules in Section 8", "Monthly"],
       ["Operational", "Data checks, audit", "Pre-holdout audit, marker file, fresh-clone test", "Each run"]], [0.15, 0.27, 0.45, 0.13])
stress_rows = [["Episode", "Strategy", "Spreads ×3", "0050"]]
for e, g in stress.groupby("episode", sort=False):
    g = g.set_index("case")["ret"]
    stress_rows.append([e, pct(g["base spreads"]), pct(g["spreads x3"]), pct(g["0050"])])
table(stress_rows, [0.4, 0.2, 0.2, 0.2])
P(f"<b>Tail and path, holdout.</b> 95% daily VaR {pct(SM['var_95'])}, CVaR {pct(SM['cvar_95'])}. Worst drawdown {pct(SM['max_drawdown'])}: peak {SM['dd_peak']}, trough {SM['dd_trough']}, recovered {SM['dd_recovered']}. Mean rolling 63-day beta to 0050 is {D['rolling_beta_mean']:.2f}. "
  "Factor loadings were not estimated.")
P("<b>Overlay.</b> Volatility target 15%, 63-day realized, t−1 data, exposure capped at 100%. The ladder is 5% watch, 10% halve, 15% halt new positions, 20% unwind, checked at each month's decision on the base strategy's path. "
  f"With the overlay, validation Sharpe falls from {n(VM['sharpe'])} to {n(V['overlay']['sharpe'])}, and holdout from {n(SM['sharpe'])} to {n(H_['metrics']['overlay']['sharpe'])}, while holdout drawdown improves to {pct(H_['metrics']['overlay']['max_drawdown'])}. The deployed result excludes it. "
  f"<b>Finding: the ladder was not calibrated.</b> Its thresholds came from setup.yaml before any backtest. The base strategy fell more than 20% in {len(DD_EPISODES)} separate validation episodes ({', '.join(pct(e) for e in DD_EPISODES)}), and its 1% daily drawdown quantile is {pct(DD_Q01)}. "
  f"So a 20% unwind rung is reached in ordinary drawdowns, not only in failures, and the overlay gives up much of the return ({pct(V['overlay']['cagr'])} against {pct(VM['cagr'])}). Section 8 recalibrates the thresholds.")
_nv, _nh = pd.read_parquet(ROOT / "data/processed/nav_validation.parquet")["strategy"].dropna(), pd.read_parquet(ROOT / "data/processed/nav_holdout.parquet")["strategy"].dropna()
_path = (1 + pd.concat([_nv.pct_change().dropna(), _nh.pct_change().dropna()])).cumprod()
_dd = _path / _path.cummax() - 1
_years = sorted(set(_dd[_dd <= -0.20].index.year))
figure("fig6_drawdown.png", 2.529, f"Figure 6. When would the ladder have triggered? The 20% rung is crossed in {', '.join(map(str, _years))} on daily data, though the ladder is only checked monthly. Dotted lines are thresholds, the dashed line the holdout start. The path chains both periods.", 0.72)
P("<b>What could break this.</b> Opens are assumed fillable at the printed price; limit locks are inferred from daily bars. The ladder is checked monthly, so real drawdowns could run past a threshold before it acts. Factor risk is unmeasured.")

# ---------- 7 ----------
page_break()
H("7. Results, diagnosis and next iteration")
figure("headline_table.png", 2.423, "Table 1. Headline results. Holdout is the single run on commit bd6cece, from 2024-11-01 to 2026-10-30.", 1.0)
P(f"<b>Classification (Chapter 20 pattern): inconclusive, closest to hold.</b> IC went from {n(models.loc['ridge', 'mean'], 3)} to {n(HIC['mean'], 3)}, with the same sign. The change is {n(IC_D, 3)} {ci(IC_D - 1.96 * IC_SE, IC_D + 1.96 * IC_SE, 3)} (z = {IC_D / IC_SE:.2f}), so there is no evidence of decay. "
  f"Hold does not fit fully either: the holdout interval includes zero, so the signal is not confirmed. Never resolved does not fit, because the validation interval excludes zero. The honest reading is that two years cannot separate hold from mild decay. "
  f"The confirmatory test fails on its first condition (lower bound {n(HIC['low'], 3)}) and passes its second ({n(SM['sharpe'])} ≥ {n(H_['metrics']['rule']['sharpe'])}). "
  f"Deflated Sharpe is {n(H_['deflated_sharpe'])} with {H_['n_trials']} trials. With about two years of data the Sharpe standard error is about 1, and the minimum track record to confirm this Sharpe is {H_['min_track_record_years']:.1f} years.")
figure("fig1_equity_curves.png", 2.15, f"Figure 1. Did it beat its baselines, net of costs? It beat the 12-1 rule in both periods. In the holdout it lost to 0050: {pct(SM['cagr'])} against {pct(H_['metrics']['etf_0050']['cagr'])} a year. Each holdout series is rebased at the boundary.", 0.78)
figure("fig4_regime_slice.png", 2.105, "Figure 4. Where does it fail? In downtrends, and in mid-volatility uptrends. Thresholds are point-in-time, and this is the only regime definition tried. The holdout has only 70 downtrend days, so those bars are noisy.", 0.66)
story.append(CondPageBreak(5 * cm))
S("Diagnosis")
bullets([f"<b>Where value was lost:</b> at the holdout gate. The signal gate passed on validation and costs did not remove the result. Against 0050 the shortfall appears before costs too: gross CAGR {pct(H_['gross']['cagr'])}, 0050 {pct(H_['metrics']['etf_0050']['cagr'])}.",
         f"<b>Drift:</b> no significant prediction-quality drift: the holdout IC is lower, but the change is within noise. Net Sharpe rose rather than fell, so this is not portfolio-translation drift. The regime pattern is the same in both periods, and strong market returns may explain some of the higher Sharpe.",
         f"<b>Binding constraint:</b> construction robustness. Over the {H_YEARS:.1f}-year holdout 0050 returned {D['etf_0050_return'] * 100:.0f}% in total ({pct(H_['metrics']['etf_0050']['cagr'])} a year) and TSMC {D['tsmc_return'] * 100:.0f}%, while the median eligible stock returned {D['median_universe_stock_return'] * 100:.0f}%. TSMC was in the top decile in {D['months_tsmc_in_top_decile']} of {D['months']} months, and the largest single weight was {H_['max_name_weight'] * 100:.1f}%. The strategy beat the equal-weight universe ({pct(H_['metrics']['equal_weight']['cagr'])}), which suggests some selection value. These are descriptive comparisons, not proof.",
         "<b>LightGBM:</b> its holdout IC was higher (0.071), but the choice was fixed beforehand. Switching now would contaminate the holdout, so this is only a lead for new data."])
P("<b>One next step.</b> Re-test both signals in a sector-neutral form on data not yet seen, to separate stock selection from the missing TSMC exposure.")
P("<b>What could break this.</b> The TSMC explanation fits one two-year window. The holdout was unusually strong for large caps, and the Sharpe interval is too wide to rule out either chance or real skill.")

# ---------- 8 ----------
H("8. Monitoring and deployment plan")
P("A design only. The strategy failed its test and is not going to capital, and a shadow period comes before any capital in any case.")
bullets(["<b>Parity:</b> run the same stage code live, and compare raw data, features, predictions, signals and orders with the backtest at each stage.",
         "<b>Monitoring:</b> rolling Sharpe, drawdown, hit rate and IC over 30, 60 and 90 days, plus the live-to-backtest realization ratio.",
         "<b>Drift:</b> PSI on all model features: watch at 0.10–0.25, retrain above 0.25. Ridge coefficient sign check at each refit.",
         f"<b>Lifecycle:</b> at least 3 months in shadow mode. Pause if the 90-day IC interval lies below zero. Drawdown thresholds come from the validation path, not from defaults: "
         f"watch at {-DD_Q01 * 100:.0f}% (the 1% daily drawdown quantile), pause and review at {-DD_WORST * 100:.0f}% (the worst validation drawdown), retire at {-DD_WORST * 125:.0f}% (1.25 times the worst). "
         f"A 20% retire rule would have stopped this strategy {len(DD_EPISODES)} times in validation.",
         "<b>Diagnosis:</b> separate bugs (fix), signal decay (retrain or retire) and cost inflation (lower the cadence, which would be setup v2)."])
P("<b>What could break this.</b> The drawdown thresholds are calibrated on one validation path, which may understate future tails. The PSI and IC thresholds are textbook defaults. Monthly decisions mean drift is slow to detect.")

# ---------- Appendix ----------
page_break()
H("Appendix A. Repository map and how to reproduce")
table([["Path", "Purpose"],
       ["setup.yaml", "Trading setup and sealed holdout dates"],
       ["run_all.py", "Regenerates everything from the root; replays the recorded holdout with the frozen spec"],
       ["data/fetch_official.py, data/verify_flows.py, data/twdb.py", "Download revenue and notices; check flows; price adjustment logic"],
       ["stages/s01–s07, stages/simulator.py", "Dataset, triage, models, backtest, audit, holdout, results; simulator"],
       ["notebooks/01_setup … 07_results", "One thin notebook per stage. Not yet run one by one."],
       ["reports/", "Tables, figures, audits, holdout result and marker"],
       ["run_log.csv", "Research ledger"]], [0.42, 0.58])
P("<b>Reproducibility test.</b> A fresh clone of commit 1a705c0 ran the whole pipeline from stage 1. Headline table, holdout results, triage, models, backtest and validation summary were all identical to the originals; only the SHAP image file differed. "
  f"Limits of this test: same machine, same tw.db, and already-downloaded raw files. tw.db is 1.5 GB and is not in the repository. {TODO('State how the instructor obtains tw.db')}")
H("Appendix B. Bugs fixed before the holdout (not trials)")
table([["Bug", "How it was found", "Effect"],
       ["Purge one day short: the label ends at t+22", "Assertion in the model stage", "Run stopped; fixed before any model result"],
       ["<b>Look-ahead:</b> stocks with a missing future label were dropped before scoring, using information unavailable at the decision", "Reading the output by hand; the audit had passed", "Net Sharpe 1.13 → 1.12. A check was added to the audit"],
       ["Equal-weight benchmark returned NaN when a holding had no open", "Reading the output by hand", "Benchmark only"],
       ["Disposition query stopped at the 28th of each month", "Code review", "Re-downloaded before the dataset was built"]], [0.5, 0.25, 0.25])
P("After the holdout, commits changed only a replay flag, the replay audit, reporting and documentation. The pre-holdout audit is in <font name='Body'>reports/pre_holdout_audit.md</font>, and the holdout marker records commit bd6cece.")
H("Appendix C. Research ledger")
P("Complete <font name='Body'>run_log.csv</font>, with the config column shortened. Repeated rows are reruns after the bug fixes above. Commit and data snapshot are in the file.")
with open(ROOT / "run_log.csv", encoding="utf-8") as f:
    log = list(csv.DictReader(f))
rows = [["#", "Family", "Trial", "Run", "Time", "Commit", "Config", "Window", "Metric", "Value", "Decision"]]
for i, r in enumerate(log, 1):
    cfg = r["config"].replace('"', "")
    rows.append([i, r["trial_family"], r["trial"], r["run"], r["timestamp"][5:16].replace("T", " "), r["git_commit"], cfg[:44] + ("…" if len(cfg) > 44 else ""), r["window"].replace("..", " to ").removesuffix(" to "), r["metric_name"].replace("_", " "), f"{float(r['metric_value']):.4f}" if "." in r["metric_value"] else r["metric_value"], r["decision"]])
table(rows, [0.04, 0.11, 0.1, 0.04, 0.07, 0.08, 0.16, 0.12, 0.09, 0.07, 0.12], (LOG, LOGH))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.1 * cm, "ML4T midterm proposal")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm, title="ML4T Midterm Proposal", author="")
doc.build(story, onFirstPage=footer, onLaterPages=footer)


def add_field(par, code):
    for kind, text in (("begin", None), (None, code), ("end", None)):
        r = par.add_run()
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.text = text
        r._r.append(el)


foot = word.sections[0].footer.paragraphs[0]
foot.add_run("ML4T midterm proposal    Page ")
add_field(foot, "PAGE")
word.core_properties.title, word.core_properties.author = "ML4T Midterm Proposal", ""
word.save(DOCX)
print(OUT, DOCX)
