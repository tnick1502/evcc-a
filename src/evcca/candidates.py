"""Enumeration of admissible groups.

Groups are encoded as integer bit masks over the rows of a context. Every
function returns ``{mask: dispersion cost}`` for the admissible groups only.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .gates import TOLERANCE, barrier_pairs, dispersion_cost, has_fixed_units, ranks_by_hole, values


def _fixed_unit_ok(data, membership):
    units = data.fixed_geological_unit.fillna("__unspecified").astype(str).to_numpy()
    touched = np.stack([membership[:, units == unit].any(axis=1) for unit in np.unique(units)], axis=1)
    return touched.sum(axis=1) == 1


def _single_run(membership, ranks):
    """True where every borehole contributes at most one contiguous interval."""
    ok = np.ones(len(membership), bool)
    for order in ranks:
        a = membership[:, order]
        runs = a[:, 0].astype(int) + np.sum(a[:, 1:] & ~a[:, :-1], axis=1)
        ok &= runs <= 1
    return ok


def exact_columns(data, policy):
    """Screen every subset of the context (contexts with at most ``exact_n`` rows)."""
    n = len(data)
    if n > policy.exact_n:
        raise ValueError("Context too large for subset enumeration.")
    x = values(data, policy)
    masks = np.arange(1 << n, dtype=np.uint32)
    membership = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    sizes = membership.sum(axis=1)
    eligible = sizes >= policy.min_size
    masks, membership, sizes = masks[eligible], membership[eligible], sizes[eligible]

    scaled = x / x.max(axis=0)
    means = membership @ scaled / sizes[:, None]
    variances = np.maximum((membership @ (scaled**2) - sizes[:, None] * means**2) / (sizes[:, None] - 1), 0)
    cvs = np.sqrt(variances) / means
    good = np.all(cvs <= policy.cv_limits() + TOLERANCE, axis=1)
    if policy.nu_gate:
        lo = np.stack([np.min(np.where(membership, scaled[:, j], np.inf), axis=1) for j in range(x.shape[1])], axis=1)
        hi = np.stack([np.max(np.where(membership, scaled[:, j], -np.inf), axis=1) for j in range(x.shape[1])], axis=1)
        if policy.nu_sample_sd:
            sd = np.sqrt(variances)
        else:
            sd = np.sqrt(variances * (sizes[:, None] - 1) / sizes[:, None])
        deviation = np.maximum(hi - means, means - lo)
        nu = np.divide(deviation, sd, out=np.zeros_like(deviation), where=sd > 1e-12)
        limits = np.array([policy.nu_limit(int(k)) for k in sizes])
        good &= np.all(nu <= limits[:, None] + TOLERANCE, axis=1)
    if policy.borehole_contiguity:
        good &= _single_run(membership, ranks_by_hole(data))
        for i, j in barrier_pairs(data):
            good &= ~(membership[:, i] & membership[:, j])
    if has_fixed_units(data):
        good &= _fixed_unit_ok(data, membership)
    costs = sizes / len(data) * 0.05 * dispersion_cost(cvs, policy)
    return {int(m): float(c) for m, c in zip(masks[good], costs[good])}


def interval_combinations(data):
    """Size of the product of per-borehole interval choices (``N_geom``)."""
    return int(np.prod([1 + len(order) * (len(order) + 1) // 2 for order in ranks_by_hole(data)], dtype=object))


def geometry_combinations(data, policy):
    """Interval product, stopping early once it exceeds the enumeration budget."""
    if not policy.borehole_contiguity or len(data) > 60:
        return np.inf
    count = 1
    for order in ranks_by_hole(data):
        count *= 1 + len(order) * (len(order) + 1) // 2
        if count > policy.max_exact_geometry_combinations:
            return count
    return count


def exact_geometric_columns(data, policy):
    """Screen every combination of one interval (or none) per borehole.

    Under the contiguity rule this product contains every possible group, so
    the result is the complete admissible family.
    """
    if geometry_combinations(data, policy) > policy.max_exact_geometry_combinations:
        raise ValueError("Interval product exceeds the enumeration budget.")
    options = []
    for order in ranks_by_hole(data):
        masks = [0]
        for start in range(len(order)):
            mask = 0
            for end in range(start, len(order)):
                mask |= 1 << int(order[end])
                masks.append(mask)
        options.append(masks)
    output = {}
    batch = []
    for components in product(*options):
        mask = 0
        for value in components:
            mask |= value
        if mask.bit_count() < policy.min_size:
            continue
        batch.append(mask)
        if len(batch) >= 16384:
            output.update(screen_masks(data, policy, np.array(batch, dtype=np.uint64)))
            batch = []
    if batch:
        output.update(screen_masks(data, policy, np.array(batch, dtype=np.uint64)))
    return output


def screen_masks(data, policy, masks, x=None, ranks=None, barriers=None):
    """Vectorised version of ``group_audit`` for a batch of masks."""
    n = len(data)
    x = values(data, policy) if x is None else x
    ranks = ranks_by_hole(data) if ranks is None else ranks
    barriers = barrier_pairs(data) if barriers is None else barriers
    masks = np.unique(masks)
    if policy.borehole_contiguity:
        valid = np.ones(len(masks), bool)
        for i, j in barriers:
            pair = np.uint64((1 << i) | (1 << j))
            valid &= (masks & pair) != pair
        masks = masks[valid]
    if not len(masks):
        return {}
    membership = ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)
    sizes = membership.sum(axis=1)
    scaled = x / x.max(axis=0)
    means = membership @ scaled / sizes[:, None]
    variance = np.maximum((membership @ (scaled**2) - sizes[:, None] * means**2) / (sizes[:, None] - 1), 0)
    cv = np.sqrt(variance) / means
    good = (sizes >= policy.min_size) & np.all(cv <= policy.cv_limits() + TOLERANCE, axis=1)
    if policy.nu_gate:
        lo = np.stack([np.min(np.where(membership, scaled[:, j], np.inf), axis=1) for j in range(x.shape[1])], axis=1)
        hi = np.stack([np.max(np.where(membership, scaled[:, j], -np.inf), axis=1) for j in range(x.shape[1])], axis=1)
        if policy.nu_sample_sd:
            sd = np.sqrt(variance)
        else:
            sd = np.sqrt(variance * (sizes[:, None] - 1) / sizes[:, None])
        diff = np.maximum(hi - means, means - lo)
        nu = np.divide(diff, sd, out=np.zeros_like(diff), where=sd > 1e-12)
        table = np.array([policy.nu_limit(j) for j in range(n + 1)])
        good &= np.all(nu <= table[sizes, None] + TOLERANCE, axis=1)
    if policy.borehole_contiguity:
        good &= _single_run(membership, ranks)
    if has_fixed_units(data):
        good &= _fixed_unit_ok(data, membership)
    costs = sizes / n * 0.05 * dispersion_cost(cv, policy)
    return {int(mask): float(cost) for mask, cost in zip(masks[good], costs[good])}
