"""Small synthetic contexts for the tests."""

import numpy as np
import pandas as pd

from evcca import Policy


def frame(values, labels=None, boreholes=None):
    """A small context: one borehole per row unless ``boreholes`` is given."""
    x = np.asarray(values, float)
    n = len(x)
    d = pd.DataFrame(x, columns=["a", "b"])
    d["sample_id"] = [str(i) for i in range(n)]
    d["depth_m"] = np.arange(n, dtype=float)
    d["borehole_id"] = boreholes if boreholes is not None else [str(i) for i in range(n)]
    d["initial_ige"] = labels if labels is not None else ["source"] * n
    return d


def small_policy(**kwargs):
    settings = dict(properties=("a", "b"), limits=(0.3, 0.3), min_size=3, exact_n=12, nu_gate=False)
    settings.update(kwargs)
    return Policy(**settings)
