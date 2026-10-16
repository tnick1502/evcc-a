"""Set-partitioning master problem and an independent exact-cover oracle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from .gates import mask_members, original_pairs, pair_edit_cost, source_labels


@dataclass
class Partition:
    """A grouping of one context.

    ``labels`` holds the group of every row (-1 = referred for review).
    ``objective`` is (A, K, B) in compact mode and (A, D, K, B) in repair mode.
    """

    labels: np.ndarray
    objective: tuple
    status: str
    scope: str
    optimal: bool
    seconds: float
    candidate_count: int = 0
    order: str = ""
    mip_gap: float = np.nan


def _empty_objective(data, mode):
    n = len(data)
    return (n, original_pairs(data), 0, 0.0) if mode == "repair" else (n, 0, 0.0)


def _two_group_covers(columns, n):
    """Full covers by one or two admissible groups."""
    full = (1 << n) - 1
    covers = []
    if full in columns:
        covers.append((full,))
    for mask in sorted(columns):
        complement = full ^ mask
        if mask < complement and complement in columns:
            covers.append((mask, complement))
    return covers


def _cover_key(data, columns, pack, mode):
    n = len(data)
    b = sum(columns[mask] for mask in pack)
    d = original_pairs(data) + sum(pair_edit_cost(data, mask_members(mask, n)) for mask in pack)
    return (0, d, len(pack), b) if mode == "repair" else (0, len(pack), b)


def solve_master(data, policy, columns, mode="compact", scope="complete_subset_space", witness=None):
    """Lexicographic set partitioning over the admissible groups ``columns``.

    Compact mode minimizes (A, K, B); repair mode minimizes (A, D, K, B).
    Each stage is solved to a zero gap and fixed before the next one. A known
    feasible ``witness`` bounds A from above.
    """
    started = perf_counter()
    n = len(data)
    masks = sorted(columns)
    m = len(masks)
    labels = np.full(n, -1, int)
    if not masks:
        return Partition(
            labels, _empty_objective(data, mode), "no_supported_group", scope, True, perf_counter() - started, 0
        )

    full = (1 << n) - 1
    if mode == "compact" and full in columns:
        # A = 0 and K = 1 are both at their lower bounds.
        return Partition(
            np.zeros(n, int),
            (0, 1, columns[full]),
            "optimal",
            "global_one_group_bound",
            True,
            perf_counter() - started,
            m,
        )
    if mode == "repair":
        initial = pd.factorize(source_labels(data), sort=True)[0]
        initial_masks = [sum(1 << int(i) for i in np.flatnonzero(initial == j)) for j in set(initial)]
        if len(initial_masks) <= policy.max_groups and all(mask in columns for mask in initial_masks):
            # The source partition is admissible: A = 0 and D = 0.
            cost = sum(columns[mask] for mask in initial_masks)
            return Partition(
                initial,
                (0, 0, len(initial_masks), cost),
                "optimal",
                "global_zero_edit_bound",
                True,
                perf_counter() - started,
                m,
            )

    # With fewer than 3 * min_size rows a full cover has at most two groups,
    # so full covers can be listed directly.
    if n < 3 * policy.min_size:
        covers = _two_group_covers(columns, n)
        if covers:
            pack = min(covers, key=lambda pack: _cover_key(data, columns, pack, mode))
            for label, mask in enumerate(pack):
                labels[mask_members(mask, n)] = label
            return Partition(
                labels, _cover_key(data, columns, pack, mode), "optimal", scope, True, perf_counter() - started, m
            )

    rows, cols, nums = [], [], []
    for j, mask in enumerate(masks):
        members = mask_members(mask, n)
        rows.extend(members)
        cols.extend([j] * len(members))
        nums.extend([1.0] * len(members))
    rows.extend(range(n))
    cols.extend(range(m, m + n))
    nums.extend([1.0] * n)
    matrix = coo_matrix((nums, (rows, cols)), shape=(n, m + n)).tocsr()
    lower = np.ones(n)
    upper = np.ones(n)

    group_vector = np.r_[np.ones(m), np.zeros(n)]
    matrix = vstack([matrix, coo_matrix(group_vector.reshape(1, -1))], format="csr")
    lower = np.r_[lower, 1.0]
    upper = np.r_[upper, min(policy.max_groups, n // policy.min_size)]

    review = np.r_[np.zeros(m), np.ones(n)]
    known_a = int(witness.objective[0]) if witness is not None else None
    if known_a is not None:
        matrix = vstack([matrix, coo_matrix(review.reshape(1, -1))], format="csr")
        lower = np.r_[lower, 0.0]
        upper = np.r_[upper, known_a]

    cost = np.r_[[columns[k] for k in masks], np.zeros(n)]
    edits = np.r_[[pair_edit_cost(data, mask_members(k, n)) for k in masks], np.zeros(n)]
    stages = [review, edits, group_vector, cost] if mode == "repair" else [review, group_vector, cost]

    result = None
    all_optimal = True
    objective = []
    for c in stages:
        if c is review and known_a == 0:
            objective.append(0.0)
            continue
        result = milp(
            c,
            integrality=np.ones(m + n),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": policy.seconds, "mip_rel_gap": 0.0},
        )
        if result.x is None:
            return Partition(labels, (n, 0, 0.0), "solver_no_incumbent", scope, False, perf_counter() - started, m)
        optimum = float(c @ np.rint(result.x))
        objective.append(optimum)
        gap = getattr(result, "mip_gap", np.nan)
        if result.status != 0 or not np.isfinite(gap) or gap > 1e-8:
            all_optimal = False
            break
        if c is not cost:
            matrix = vstack([matrix, coo_matrix(c.reshape(1, -1))], format="csr")
            lower = np.r_[lower, optimum]
            upper = np.r_[upper, optimum]

    for label, j in enumerate(np.flatnonzero(result.x[:m] > 0.5)):
        labels[mask_members(masks[j], n)] = label
    if mode == "repair" and len(objective) >= 2:
        objective[1] += original_pairs(data)
    return Partition(
        labels,
        tuple(objective),
        "optimal" if all_optimal else "limit_incumbent",
        scope,
        all_optimal,
        perf_counter() - started,
        m,
        mip_gap=float(getattr(result, "mip_gap", np.nan)),
    )


def exact_oracle(data, policy, columns, mode="compact"):
    """Recursive exact cover over ``columns``; an independent check of ``solve_master``."""
    started = perf_counter()
    n = len(data)
    full = (1 << n) - 1
    if mode == "compact" and full in columns:
        return Partition(
            np.zeros(n, int),
            (0, 1, columns[full]),
            "optimal",
            "independent_global_one_group_bound",
            True,
            perf_counter() - started,
            len(columns),
        )
    if n < 3 * policy.min_size:
        packs = []
        for mask in columns:
            if mask == full:
                packs.append((mask,))
            elif (full - mask) in columns and mask < (full - mask):
                packs.append((mask, full - mask))
        if packs:
            pack = min(packs, key=lambda pack: _cover_key(data, columns, pack, mode))
            labels = np.full(n, -1, int)
            for label, mask in enumerate(pack):
                labels[mask_members(mask, n)] = label
            return Partition(
                labels,
                _cover_key(data, columns, pack, mode),
                "optimal",
                "independent_complete_subset_space",
                True,
                perf_counter() - started,
                len(columns),
            )

    containing = [[] for _ in range(n)]
    for mask, cost in columns.items():
        edit = pair_edit_cost(data, mask_members(mask, n))
        for j in mask_members(mask, n):
            containing[j].append((mask, cost, edit))
    kmax = min(policy.max_groups, n // policy.min_size)

    def key(q):
        return q if mode == "repair" else (q[0], q[2], q[3])

    @lru_cache(None)
    def visit(remaining, k_left):
        # Returns ((A, D-increment, K, B), chosen masks) for the rows in ``remaining``.
        if remaining == 0:
            return (0, 0, 0, 0.0), ()
        if k_left == 0:
            return (remaining.bit_count(), 0, 0, 0.0), ()
        pivot = (remaining & -remaining).bit_length() - 1
        tail, groups = visit(remaining & ~(1 << pivot), k_left)
        best = (tail[0] + 1, tail[1], tail[2], tail[3])
        chosen = groups
        for mask, cost, edit in containing[pivot]:
            if remaining & mask != mask:
                continue
            tail, groups = visit(remaining ^ mask, k_left - 1)
            option = (tail[0], tail[1] + edit, tail[2] + 1, tail[3] + cost)
            if key(option) < key(best):
                best = option
                chosen = (mask, *groups)
        return best, chosen

    objective, groups = visit(full, kmax)
    labels = np.full(n, -1, int)
    for label, mask in enumerate(groups):
        labels[mask_members(mask, n)] = label
    a, d, k, b = objective
    result = (a, d + original_pairs(data), k, b) if mode == "repair" else (a, k, b)
    return Partition(
        labels, result, "optimal", "independent_complete_subset_space", True, perf_counter() - started, len(columns)
    )
