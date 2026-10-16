"""Compact versus repair grouping of the P13 loam context (Fig. 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..gates import group_audit
from ..plans import revision_operations
from .cohort import article_policy
from .diagnostics import pairs_between

CONTEXT = "PRIMARY::P13 | LOAM | C"


def depth_bands(specimens, assignments, context=CONTEXT):
    """Split the context at its largest depth gap and count each group's members per band.

    The bands describe the result; they are not used by either solve.
    Returns ``(data, groups, modes)``.
    """
    data = specimens[specimens.context_id.eq(context)].sort_values("sample_id").reset_index(drop=True)
    depths = np.sort(data.depth_m.unique())
    gap = int(np.argmax(np.diff(depths)))
    upper_max = float(depths[gap])
    data["depth_band"] = np.where(data.depth_m <= upper_max, "upper", "lower")
    policy = article_policy("PRIMARY")
    group_rows = []
    mode_rows = []
    for mode in ("compact", "repair"):
        chosen = assignments[
            assignments.context_id.eq(context)
            & assignments.method.eq("evcca")
            & assignments["mode"].eq(mode)
            & assignments.profile.eq("four")
        ]
        labels = data.sample_id.map(chosen.set_index("sample_id").group).to_numpy(int)
        data[mode] = labels
        _, changes = revision_operations(data, labels)
        compliant = []
        for label in sorted(set(labels) - {-1}):
            index = np.flatnonzero(labels == label)
            group = data.iloc[index]
            ok = group_audit(data, index, policy)["compliant"]
            compliant.append(ok)
            group_rows.append(
                dict(
                    mode=mode,
                    group=int(label),
                    n=len(group),
                    upper_n=int(group.depth_band.eq("upper").sum()),
                    lower_n=int(group.depth_band.eq("lower").sum()),
                    depth_min=float(group.depth_m.min()),
                    depth_max=float(group.depth_m.max()),
                    source_units=";".join(sorted(group.initial_ige.unique())),
                    compliant=bool(ok),
                )
            )
        mode_rows.append(
            dict(
                mode=mode,
                retained=int((labels >= 0).sum()),
                A=int((labels < 0).sum()),
                K=len(set(labels) - {-1}),
                D=sum(changes.values()),
                cross_band_pairs=int(pairs_between(labels, data.depth_band.to_numpy())),
                all_groups_compliant=all(compliant),
            )
        )
    return data, pd.DataFrame(group_rows), pd.DataFrame(mode_rows)
