"""Material barriers derived from borehole logs.

Two records of the same soil family in one borehole may not share a group if
a record of a different family was logged between them. Incomplete records
count: a missing mechanical test does not remove the logged material.
"""

from __future__ import annotations

import pandas as pd


def primary_barriers(logs):
    """``barrier_with_ids`` for every row of ``logs`` (sample_id, site_id,
    borehole_id, depth_m, soil_family). Records of family ``OTHER`` neither
    form nor separate pairs."""
    barriers = {key: set() for key in logs.sample_id}
    for _, hole in logs.groupby(["site_id", "borehole_id"], sort=False):
        for family, same in hole.groupby("soil_family"):
            if family == "OTHER":
                continue
            other = hole[hole.soil_family.ne(family) & hole.soil_family.ne("OTHER")]
            rows = list(same.itertuples())
            for a in rows:
                for b in rows:
                    if a.sample_id >= b.sample_id:
                        continue
                    lo, hi = sorted([a.depth_m, b.depth_m])
                    if (other.depth_m.gt(lo) & other.depth_m.lt(hi)).any():
                        barriers[a.sample_id].add(b.sample_id)
                        barriers[b.sample_id].add(a.sample_id)
    return pd.Series({key: ";".join(sorted(value)) for key, value in barriers.items()}, name="barrier_with_ids")
