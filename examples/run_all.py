"""Run the four experiments end to end. No dependencies, no dataset, no network.

python examples/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridsearch.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["all", "--show-paths"]))
