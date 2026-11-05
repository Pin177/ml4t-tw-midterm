# Data quality report

- Dates: 2011-01-03 to 2026-11-04; stocks seen: 2183
- Eligible names per month: min 188, median 354, max 725
- Listed stock-days with no trade (close left empty, excluded, never forward-filled for fills): 1.26%
- OHLC invariant violations: 0
- Duplicates: 0 by construction (primary key code + date in tw.db)
- Names that left the sample: 195; per year: {2011: 11, 2012: 11, 2013: 9, 2014: 10, 2015: 11, 2016: 16, 2017: 15, 2018: 17, 2019: 14, 2020: 20, 2021: 11, 2022: 15, 2023: 9, 2024: 9, 2025: 7, 2026: 10}
- Names with a current industry code: 1990 of 2183
- Session: TWSE/TPEx close 13:30 Asia/Taipei; prices treated as known at 15:00.
- Storage: data/processed/*.parquet rebuilt by stages/s01_build_dataset.py; no synthetic data used.

## Feature missing share on eligible stock-days

|                     |   missing |
|:--------------------|----------:|
| mom_12_1            |    0.0019 |
| mom_6_1             |    0.0004 |
| dist_52w_high       |    0      |
| rev_21d             |    0.0002 |
| foreign_net_buy_20d |    0.0154 |
| trust_net_buy_20d   |    0.0154 |
| revenue_yoy         |    0.0826 |
| revenue_accel       |    0.0835 |
| market_vol_pctile   |    0      |
| taiex_trend         |    0      |

## Point-in-time checklist

| Level | Check | Result | Direction of any bias |
| --- | --- | --- | --- |
| Values | Prices and flows used only after that day's close | PASS | none |
| Values | Revenue dated first trading day after the 10th of the next month | PASS with limit | late filers may look known early; values are latest published, so restatements are not separated |
| Universe | Eligibility computed from data at t; delisted names kept | PASS | none |
| Universe | TWSE disposition notices excluded by publication date | PASS | none |
| Universe | TPEx disposition and all full-delivery status | FAIL, disclosed | some restricted names stay eligible; fills may be optimistic |
| Identifiers | Code reuse: stock must have 240 of last 252 days listed | PASS with limit | a reused code restarts its eligibility |
| Identifiers | Industry codes are today's, not point-in-time | FAIL, disclosed | sector cap may differ slightly from what was known then |
| Values | Flows before 2014-12 on TPEx could not be checked against the official site | NOT VERIFIED | unknown |