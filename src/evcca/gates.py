"""Admissibility of a single group and the pairwise edit cost of a group.

A context is a table with one row per specimen. Required columns are
``sample_id``, ``borehole_id``, ``depth_m`` and the policy properties.
``initial_ige`` (source unit), ``barrier_with_ids`` and
``fixed_geological_unit`` are optional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOLERANCE = 1e-9


def values(data, policy):
    x = data[list(policy.properties)].to_numpy(dtype=float)
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Constrained properties must be observed and positive.")
    return x


def source_labels(data):
    """Source-unit labels as strings; one common unit when none are given."""
    if "initial_ige" not in data:
        return np.full(len(data), "", dtype=object)
    return data.initial_ige.astype(str).to_numpy()


def has_fixed_units(data):
    return "fixed_geological_unit" in data and data.fixed_geological_unit.notna().any()


def dispersion_cost(cv, policy):
    """Weighted mean of (CV / limit)^2; the final tie-breaker."""
    ratios = (np.asarray(cv, dtype=float) / np.asarray(policy.limits, dtype=float)) ** 2
    if policy.weights is None:
        return np.mean(ratios, axis=-1)
    weights = np.asarray(policy.weights, dtype=float)
    if len(weights) != len(policy.properties) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Weights must be finite, positive and one per property.")
    return np.average(ratios, axis=-1, weights=weights)


def ranks_by_hole(data):
    """Row indices of each borehole, ordered by depth."""
    ranks = []
    for _, hole in data.reset_index(drop=True).groupby("borehole_id", sort=True):
        ranks.append(hole.sort_values(["depth_m", "sample_id"]).index.to_numpy())
    return ranks


def barrier_pairs(data):
    """Index pairs that may not share a group (logged material between them)."""
    if "barrier_with_ids" not in data:
        return []
    lookup = {key: i for i, key in enumerate(data.sample_id)}
    pairs = set()
    for i, value in enumerate(data.barrier_with_ids.fillna("")):
        for key in str(value).split(";"):
            if key in lookup:
                pairs.add(tuple(sorted([i, lookup[key]])))
    return sorted(pairs)


def group_audit(data, members, policy, _x=None, _ranks=None, _barriers=None):
    """Statistics and admissibility of one group of row indices."""
    members = np.asarray(members, int)
    x = (values(data, policy) if _x is None else _x)[members]
    n = len(members)
    if n > 1:
        cv = x.std(axis=0, ddof=1) / x.mean(axis=0)
        s = x.std(axis=0, ddof=int(policy.nu_sample_sd))
        nu = np.max(np.divide(np.abs(x - x.mean(axis=0)), s, out=np.zeros_like(x), where=s > 1e-12), axis=0)
    else:
        cv = np.full(x.shape[1], np.inf)
        nu = np.full(x.shape[1], np.inf)

    contiguous = True
    member_set = set(members)
    if policy.borehole_contiguity:
        for order in ranks_by_hole(data) if _ranks is None else _ranks:
            hits = np.array([i in member_set for i in order])
            positions = np.flatnonzero(hits)
            if len(positions) and not hits[positions.min() : positions.max() + 1].all():
                contiguous = False
                break
        for i, j in barrier_pairs(data) if _barriers is None else _barriers:
            if i in member_set and j in member_set:
                contiguous = False
                break

    compliant = (
        n >= policy.min_size
        and np.all(cv <= policy.cv_limits() + TOLERANCE)
        and (not policy.nu_gate or np.all(nu <= policy.nu_limit(n) + TOLERANCE))
        and contiguous
    )
    if has_fixed_units(data):
        compliant &= data.iloc[members].fixed_geological_unit.fillna("__unspecified").nunique() == 1
    cost = n / len(data) * 0.05 * dispersion_cost(cv, policy)
    return dict(
        n=n,
        cv=cv,
        nu=nu,
        nu_critical=policy.nu_limit(n),
        contiguous=contiguous,
        compliant=bool(compliant),
        cost=float(cost),
    )


def mask_members(mask, n):
    return np.flatnonzero([(mask >> j) & 1 for j in range(n)])


def pair_edit_cost(data, members):
    """Change in the number of disagreeing pairs when ``members`` form a group.

    The pair-edit distance of a partition equals ``original_pairs(data)`` plus
    the sum of this term over its groups.
    """
    labels = source_labels(data)
    _, counts = np.unique(labels[np.asarray(members, int)], return_counts=True)
    n = len(members)
    return n * (n - 1) // 2 - int(np.sum(counts * (counts - 1)))


def original_pairs(data):
    counts = pd.Series(source_labels(data)).value_counts().to_numpy()
    return int(np.sum(counts * (counts - 1) // 2))
