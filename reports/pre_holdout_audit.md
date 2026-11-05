# Pre-holdout audit

Overall: PASS

| Q | Check | Result | Detail |
| --- | --- | --- | --- |
| 1 | Features recomputed with data cut at T match the panel (4 dates, 7 features) | PASS | largest difference 0.00e+00 |
| 1 | Revenue changes only after the 10th of the month (TSMC) | PASS | 153 updates, earliest day of month 11 |
| 1 | Every feature has a stated availability rule in setup.yaml | PASS | ['prices_volume', 'institutional_flows', 'monthly_revenue', 'disposition_status'] |
| 2 | Universe includes names that later left the market | PASS | 84 eligible names later delisted |
| 3 | Label equals adjusted Open(t+22)/Open(t+1)-1 on 5,000 random rows | PASS | 4994 of 4994 |
| 1 | Every eligible stock is scored, including those whose label is missing | PASS | 0 unscored of 32641; 8 have no label |
| 9 | No simulated series contains a gap or a non-positive NAV | PASS | {'strategy': 4.342, 'rule': 1.734, 'equal_weight': 2.143} |
| 3 | Simulator fills buys at the open of the day after the decision | PASS | code check |
| 4 | Winsor limits fitted on the training fold; ranks are within one date | PASS | code check |
| 5 | In every fold the last training label ends before the test block | PASS | [{'fold': '2018', 'last_label_end': '2017-12-28', 'test_start': '2018-01-02'}, {'fold': '2019', 'last_label_end': '2018-12-25', 'test_start' |
| 5 | Purge equals the 21-trading-day label horizon | PASS | purge 21, horizon 21 |
| 6 | No validation score or NAV reaches into the holdout | PASS | last score 2024-09-25, last NAV 2024-10-30 |
| 6 | Validation labels all end before the holdout | PASS | 2024-10-30 |
| 6 | Holdout has not been run | PASS | no marker file |
| 6 | Ledger shows no run inside the holdout window | PASS | 71 rows |
| 7 | Research ledger exists with every family logged | PASS | 71 rows, 9 families |
| 7 | Deflated Sharpe uses the ledger's count, with every Optuna trial | PASS | n_trials 116 |
| 8 | Costs include commission, sell tax and spread | PASS | code check |
| 8 | Cost sweep runs 0 to 4x and net Sharpe falls as costs rise | PASS | [1.36, 1.116, 0.87, 0.624, 0.384] |
| 9 | Caps hold inside the simulation | PASS | max name 0.034, max sector 0.400 |
| 9 | Overlay and ladder read only the path up to the decision date | PASS | code check |
| penalty | No unexplained Sharpe above 3 | PASS | net 1.12, gross 1.36 |

Question 10 (someone else can regenerate from the repository root) is tested on a fresh clone after stage 7.