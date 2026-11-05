# ML4T Midterm: Taiwan Equity 52-Week-High and Revenue-Acceleration Strategy

One complete lap of the ML4T workflow on Taiwanese stocks. The original hypothesis was momentum plus institutional flows and monthly revenue. Screening rejected momentum and flows, so the deployed model uses two features: distance to the 52-week high and revenue acceleration.

**Result:** the pre-registered holdout test did not pass. Holdout IC 0.037 [-0.003, 0.077]; net Sharpe 1.35 [-0.61, 3.31], above the momentum rule (0.86) and below 0050 (1.84). Full table: [reports/headline_table.md](reports/headline_table.md).

## Run it

Python 3.14. From the repository root:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
set TWDB_PATH=path/to/tw.db
.venv/Scripts/python run_all.py
```

The full run takes about 40 minutes. It reproduces the recorded holdout with the frozen spec using `--reproduce`, which changes no decision.

## Data

| Data | Source | Download date | In this repo |
| --- | --- | --- | --- |
| Daily prices, corporate actions, TAIEX, 0050 | `tw.db`, built by `twdb.py` from TWSE and TPEx | 2026-11-05 snapshot | No, 1.5 GB. Provide `TWDB_PATH` |
| Foreign and investment-trust net buying | `法人資料/` **source to be filled in** | Files dated 2026-11-03 | Yes |
| Monthly revenue | MOPS (mopsov.twse.com.tw) | 2026-11-05 | No, `data/fetch_official.py` fetches it |
| TWSE disposition notices | TWSE | 2026-11-05 | No, fetched |
| Industry codes | TWSE and TPEx Open API, current values only | 2026-11-05 | No, fetched |

Prices are raw with adjustment factors computed from corporate actions. Revenue is taken as known on the first trading day after the 10th of the next month, and shows the latest published value, so restatements are not separated. Sessions close at 13:30 Taipei time; daily data is treated as known at 15:00.

## Layout

| Path | Purpose |
| --- | --- |
| `setup.yaml` | Trading setup contract, committed before any backtest |
| `run_log.csv` | Research ledger. Reruns after bug fixes appear as extra rows; the trial count removes duplicates |
| `stages/s01`-`s07` | Dataset, triage, models, backtest, audit, holdout, results |
| `stages/simulator.py` | Next-open fills, constraints and costs inside the loop |
| `notebooks/` | One notebook per stage, 01_setup to 07_results |
| `data/verify_flows.py` | Checks `法人資料/` against official responses |
| `reports/` | Tables, figures, audit, holdout result and marker |

## Why numbers can differ

- A new `tw.db` or a later revenue download can change values: vendors revise, and new days extend rolling windows.
- LightGBM and Optuna use fixed seeds and four threads; other machines may differ slightly. The deployed Ridge is not affected by them unless the ladder decision changes.

## Holdout

Run once on commit `bd6cece`, recorded in `reports/holdout_touched.json`. Later commits changed only the replay flag, reporting and documentation.
