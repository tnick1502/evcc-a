"""Small helpers for the notebooks."""

import json
import tempfile

import pandas as pd
from IPython.display import Image, display

from . import figures
from .paths import RESULTS

pd.set_option("display.max_columns", 18)
pd.set_option("display.max_rows", 40)
pd.set_option("display.width", 150)


def table(name):
    """A result table from ``results/``."""
    return pd.read_csv(RESULTS / f"{name}.csv", low_memory=False)


def record(name):
    """A JSON result from ``results/``."""
    return json.loads((RESULTS / f"{name}.json").read_text())


def show_figure(name):
    """Draw an article figure from the result files and display it."""
    with tempfile.TemporaryDirectory() as folder:
        display(Image(data=figures.render(name, folder).read_bytes(), format="png"))
