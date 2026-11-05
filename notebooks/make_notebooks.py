"""Write one thin notebook per stage. Each one runs the stage script and shows its report, so the logic has one home."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [("01_setup", "Trading setup and evaluation protocol", None, "../setup.yaml"),
          ("02_data", "Point-in-time dataset and quality report", "s01_build_dataset.py", "../reports/data_quality.md"),
          ("03_triage", "Five-gate feature triage", "s02_triage.py", "../reports/triage.csv"),
          ("04_models", "Walk-forward model ladder", "s03_models.py", "../reports/model_summary.csv"),
          ("05_backtest", "Backtest, costs, allocators, risk; pre-holdout audit", "s04_backtest.py", "../reports/backtest_validation.csv"),
          ("06_holdout", "Holdout (replay of the single recorded run)", "s06_holdout.py --reproduce", "../reports/holdout_results.json"),
          ("07_results", "Headline table and required figures", "s07_results.py", "../reports/headline_table.md")]


def cell(kind, text):
    base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


for name, title, script, report in STAGES:
    cells = [cell("markdown", f"# {title}\n\nRun from the `notebooks` folder. The logic lives in `stages/`.")]
    if script:
        cells.append(cell("code", f"import subprocess, sys\nsubprocess.run([sys.executable, *'{script}'.split()], cwd='../stages', check=True)"))
    cells.append(cell("code", f"from pathlib import Path\nprint(Path('{report}').read_text(encoding='utf-8')[:6000])"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    (HERE / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
