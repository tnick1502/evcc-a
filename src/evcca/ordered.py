"""Ordered comparator: optimal segmentation of a fixed scalar order.

For each tested order, dynamic programming chooses non-interleaving spans
that pass the same gates, with single-specimen review steps. The best order
is reported. Every ordered solution is also feasible for the complete model.
"""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .candidates import screen_masks
from .gates import barrier_pairs, group_audit, mask_members, original_pairs, pair_edit_cost, ranks_by_hole, values
from .partition import Partition


def orderings(data, policy, random_count=None):
    """Named row orders: each property (with tie orders), principal components,
    depth, source unit then depth, and optional random directions."""
    x = values(data, policy)
    z = StandardScaler().fit_transform(np.log(x))
    output = {}
    for j, name in enumerate(policy.properties):
        for k, secondary in enumerate(policy.properties):
            if j == k:
                continue
            output[name + "|tie:" + secondary] = np.lexsort((np.arange(len(data)), x[:, k], x[:, j]))
    components = PCA(n_components=min(z.shape)).fit_transform(z)
    for j in range(components.shape[1]):
        output["PC" + str(j + 1)] = np.argsort(components[:, j], kind="stable")
    output["depth"] = np.lexsort((np.arange(len(data)), data.depth_m.to_numpy(float)))
    if "initial_ige" in data:
        output["initial_ige|depth"] = np.lexsort(
            (data.depth_m.to_numpy(float), data.initial_ige.astype(str).to_numpy())
        )
    count = policy.random_orders if random_count is None else random_count
    rng = np.random.default_rng(policy.seed)
    for j in range(count):
        direction = rng.normal(size=z.shape[1])
        output["direction_" + str(j)] = np.argsort(z @ direction, kind="stable")

    unique = {}
    seen = set()
    for name, order in output.items():
        key = tuple(order)
        if key not in seen and tuple(order[::-1]) not in seen:
            unique[name] = order
            seen.add(key)
    return unique


def ordered_columns(data, policy, order, trim_cap=None):
    """Admissible groups that are spans of ``order`` with up to ``trim_cap`` interior removals."""
    n = len(data)
    trim_cap = policy.budget(n) if trim_cap is None else trim_cap
    output = {}
    x = values(data, policy)
    ranks = ranks_by_hole(data)
    barriers = barrier_pairs(data)
    if n > 18:
        batch = []

        def flush():
            nonlocal batch
            if batch:
                output.update(screen_masks(data, policy, np.array(batch, dtype=np.uint64), x, ranks, barriers))
                batch = []

        pair_bits = [((1 << i) | (1 << j), i, j) for i, j in barriers]
        for start in range(n):
            for end in range(start + policy.min_size, n + 1):
                segment = order[start:end]
                base = sum(1 << int(i) for i in segment)
                max_trim = min(trim_cap, len(segment) - policy.min_size)
                # A matching of barrier pairs inside the span bounds the number
                # of removals needed from below.
                used = set()
                required = 0
                for bits, i, j in pair_bits:
                    if base & bits == bits and i not in used and j not in used:
                        used.update([i, j])
                        required += 1
                if required > max_trim:
                    continue
                for removed_n in range(required, max_trim + 1):
                    for removed in combinations(segment[1:-1], removed_n):
                        mask = base
                        for i in removed:
                            mask ^= 1 << int(i)
                        batch.append(mask)
                        if len(batch) >= 16384:
                            flush()
        flush()
        return output

    for start in range(n):
        for end in range(start + policy.min_size, n + 1):
            segment = order[start:end]
            for removed_n in range(min(trim_cap, len(segment) - policy.min_size) + 1):
                # Removals at the span ends are covered by review steps of the DP.
                for removed in combinations(segment[1:-1], removed_n):
                    members = np.array([i for i in segment if i not in removed])
                    row = group_audit(data, members, policy, _x=x, _ranks=ranks, _barriers=barriers)
                    if row["compliant"]:
                        output[sum(1 << int(i) for i in members)] = row["cost"]
    return output


def ordered_dp(data, policy, columns, order, name, mode="compact"):
    """Best segmentation of ``order`` using groups from ``columns``."""
    n = len(data)
    started = perf_counter()
    positions = np.empty(n, int)
    positions[order] = np.arange(n)
    edges = [[] for _ in range(n)]
    for mask, cost in columns.items():
        members = mask_members(mask, n)
        pos = positions[members]
        start = int(pos.min())
        end = int(pos.max()) + 1
        skipped = end - start - len(members)
        edges[start].append((end, skipped, pair_edit_cost(data, members), cost, mask))

    kmax = min(policy.max_groups, n // policy.min_size)
    states = {(0, 0): (0, 0, 0.0)}
    previous = {}
    for pos in range(n):
        for k in range(kmax + 1):
            if (pos, k) not in states:
                continue
            a, d, c = states[pos, k]
            options = [(pos + 1, k, a + 1, d, c, 0)]
            if k < kmax:
                options.extend(
                    (end, k + 1, a + skip, d + edit, c + cost, mask) for end, skip, edit, cost, mask in edges[pos]
                )
            for end, kk, aa, dd, cc, mask in options:
                new_key = (aa, dd, cc) if mode == "repair" else (aa, cc)
                old = states.get((end, kk))
                if old is None:
                    old_key = None
                elif mode == "repair":
                    old_key = (old[0], old[1], old[2])
                else:
                    old_key = (old[0], old[2])
                if old is None or new_key < old_key:
                    states[end, kk] = (aa, dd, cc)
                    previous[end, kk] = (pos, k, mask)

    candidates = []
    for k in range(1, kmax + 1):
        if (n, k) in states:
            a, d, c = states[n, k]
            objective = (a, d + original_pairs(data), k, c) if mode == "repair" else (a, k, c)
            candidates.append((objective, k))
    labels = np.full(n, -1, int)
    if not candidates:
        objective = (n, original_pairs(data), 0, 0.0) if mode == "repair" else (n, 0, 0.0)
        return Partition(
            labels,
            objective,
            "no_supported_group",
            "ordered_" + name,
            True,
            perf_counter() - started,
            len(columns),
            name,
        )
    objective, k = min(candidates)
    state = (n, k)
    label = 0
    while state[0]:
        pos, old_k, mask = previous[state]
        if mask:
            labels[mask_members(mask, n)] = label
            label += 1
        state = (pos, old_k)
    return Partition(
        labels, objective, "optimal", "ordered_" + name, True, perf_counter() - started, len(columns), name
    )


def best_ordered(data, policy, columns=None, mode="compact", random_count=None):
    """Best result over all tested orders, and the union of their candidate groups."""
    started = perf_counter()
    pool = {}
    choices = []
    orders = orderings(data, policy, random_count)
    if columns is None:
        for name, order in orders.items():
            candidates = ordered_columns(data, policy, order, trim_cap=0)
            pool.update(candidates)
            choices.append(ordered_dp(data, policy, candidates, order, name, mode))
        if any(choice.objective[0] == 0 for choice in choices):
            # A = 0 cannot be improved by allowing interior removals.
            best = min(choices, key=lambda c: c.objective)
            best.seconds = perf_counter() - started
            return best, pool
        choices = []
    for name, order in orders.items():
        candidates = columns if columns is not None else ordered_columns(data, policy, order)
        pool.update(candidates)
        choices.append(ordered_dp(data, policy, candidates, order, name, mode))
    best = min(choices, key=lambda c: c.objective)
    best.seconds = perf_counter() - started
    return best, pool
