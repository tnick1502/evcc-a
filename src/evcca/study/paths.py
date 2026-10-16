"""Locations of the repository inputs and reference results."""

import os
from pathlib import Path


def _find_root():
    configured = os.environ.get("EVCCA_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates += [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "data/processed/specimens.csv").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Run from inside the evcc-a repository or set EVCCA_ROOT.")


ROOT = _find_root()
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
