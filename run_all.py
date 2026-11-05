"""Regenerate the dataset and every result from the repository root.

    python run_all.py              # stages 1-5 and 7, then replay the recorded holdout with the frozen spec
    python run_all.py --from 7     # only rebuild figures and tables

Needs tw.db: set TWDB_PATH, or keep it at ../db/tw.db. Raw revenue and disposition files are fetched if missing.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
STEPS = [(0, ROOT / "data", ["fetch_official.py"]), (1, ROOT / "stages", ["s01_build_dataset.py"]), (2, ROOT / "stages", ["s02_triage.py"]),
         (3, ROOT / "stages", ["s03_models.py"]), (4, ROOT / "stages", ["s04_backtest.py"]), (5, ROOT / "stages", ["s05_audit.py"]),
         (6, ROOT / "stages", ["s06_holdout.py", "--reproduce"]), (7, ROOT / "stages", ["s07_results.py"])]

start = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
for n, cwd, cmd in STEPS:
    if n >= start:
        print(f"=== stage {n}: {cmd[0]}", flush=True)
        subprocess.run([sys.executable, *cmd], cwd=cwd, env=ENV, check=True)
