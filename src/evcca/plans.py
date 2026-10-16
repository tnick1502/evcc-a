"""Revision plans: how a new grouping changes the source interpretation.

The source units and the new groups form a bipartite incidence graph. Each
connected component is one operation: preservation, split, merge, partial
transfer or rebuild. Referred specimens are reported as a separate review
operation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .gates import TOLERANCE, mask_members, original_pairs, pair_edit_cost, source_labels

OPERATIONS = ("preserve_assigned", "split", "merge", "partial_transfer", "rebuild", "review")


def partition_masks(labels):
    return tuple(sorted(sum(1 << int(i) for i in np.flatnonzero(labels == g)) for g in set(labels) - {-1}))


def enumerate_leading_plans(data, columns, solution, mode="compact", cap=100_000):
    """All unordered partitions that tie ``solution`` on every criterion before B.

    Returns ``(plans, complete)``; ``complete`` is False if ``cap`` was reached.
    """
    n = len(data)
    n_review = int(np.sum(solution.labels < 0))
    n_groups = len(set(solution.labels) - {-1})
    if n_groups == 0:
        return [tuple()], True
    full = (1 << n) - 1
    by_row = [[] for _ in range(n)]
    for mask in sorted(columns):
        for i in mask_members(mask, n):
            by_row[i].append(mask)
    base_pairs = original_pairs(data)
    wanted_d = int(solution.objective[1]) if mode == "repair" else None
    edits = {mask: pair_edit_cost(data, mask_members(mask, n)) for mask in columns}
    output = []

    def descend(uncovered, k, chosen):
        if len(output) >= cap:
            return
        if k == 0:
            if uncovered == 0 and (wanted_d is None or base_pairs + sum(edits[m] for m in chosen) == wanted_d):
                output.append(tuple(sorted(chosen)))
            return
        if uncovered.bit_count() < 6 * k:
            return
        if k == 1:
            if uncovered in columns:
                descend(0, 0, chosen + (uncovered,))
            return
        i = (uncovered & -uncovered).bit_length() - 1
        for mask in by_row[i]:
            if mask & uncovered == mask:
                descend(uncovered ^ mask, k - 1, chosen + (mask,))

    for reviewed in combinations(range(n), n_review):
        review_mask = sum(1 << i for i in reviewed)
        descend(full ^ review_mask, n_groups, tuple())
        if len(output) >= cap:
            return output, False
    return sorted(set(output)), True


def revision_operations(data, labels):
    """Operations of a partition and the decomposition of its changed pairs."""
    old = source_labels(data)
    labels = np.asarray(labels)
    edges = {(str(s), int(g)) for s, g in zip(old, labels) if g >= 0}
    remaining = set(edges)
    operations = []
    while remaining:
        first = next(iter(sorted(remaining)))
        sources = {first[0]}
        targets = {first[1]}
        changed = True
        while changed:
            size = (len(sources), len(targets))
            sources |= {s for s, g in edges if g in targets}
            targets |= {g for s, g in edges if s in sources}
            changed = size != (len(sources), len(targets))
        component = {(s, g) for s, g in edges if s in sources and g in targets}
        remaining -= component
        if len(sources) == 1 and len(targets) == 1:
            operation = "preserve_assigned"
        elif len(sources) == 1:
            operation = "split"
        elif len(targets) == 1:
            operation = "merge"
        elif len(sources) == len(targets) == 2 and len(component) == 3:
            operation = "partial_transfer"
        else:
            operation = "rebuild"
        members = np.flatnonzero(np.isin(old, list(sources)) & (labels >= 0))
        operations.append(
            dict(
                operation=operation,
                source_labels=sorted(sources),
                target_groups=sorted(targets),
                assigned_n=len(members),
                sample_ids=data.iloc[members].sample_id.tolist(),
            )
        )
    reviewed = np.flatnonzero(labels < 0)
    if len(reviewed):
        operations.append(
            dict(
                operation="review",
                source_labels=sorted(set(old[reviewed])),
                target_groups=[],
                assigned_n=0,
                review_n=len(reviewed),
                sample_ids=data.iloc[reviewed].sample_id.tolist(),
            )
        )

    # Every changed pair is exactly one of these.
    counts = dict(review_pair_losses=0, assigned_pair_splits=0, pair_fusions=0)
    for i, k in combinations(range(len(data)), 2):
        before = old[i] == old[k]
        after = labels[i] >= 0 and labels[i] == labels[k]
        if before and not after:
            counts["review_pair_losses" if labels[i] < 0 or labels[k] < 0 else "assigned_pair_splits"] += 1
        elif after and not before:
            counts["pair_fusions"] += 1
    return operations, counts


def build_plan(data, columns, solution, mode):
    """Summary row and full plan (operations and every tied alternative)."""
    packs, complete = enumerate_leading_plans(data, columns, solution, mode)
    costs = np.array([sum(columns[m] for m in pack) for pack in packs])
    best = float(solution.objective[-1])
    full_ties = np.isclose(costs, best, atol=TOLERANCE, rtol=0)
    operations, pairs = revision_operations(data, solution.labels)
    n = len(data)
    expected = original_pairs(data) + sum(
        pair_edit_cost(data, mask_members(m, n)) for m in partition_masks(solution.labels)
    )
    if sum(pairs.values()) != expected:
        raise AssertionError("Pair decomposition does not add up to the edit distance.")

    alternatives = []
    for index, (pack, cost) in enumerate(zip(packs, costs)):
        assigned = np.zeros(n, bool)
        groups = []
        for mask in pack:
            members = mask_members(mask, n)
            assigned[members] = True
            groups.append(data.iloc[members].sample_id.tolist())
        alternatives.append(
            dict(
                plan=index,
                B=float(cost),
                full_objective_optimal=bool(full_ties[index]),
                groups=groups,
                review=data.loc[~assigned, "sample_id"].tolist(),
            )
        )
    summary = dict(
        mode=mode,
        n=n,
        A=int(np.sum(solution.labels < 0)),
        K=len(set(solution.labels) - {-1}),
        D=sum(pairs.values()),
        B=best,
        leading_optimal_plans=len(packs),
        full_objective_optimal_plans=int(full_ties.sum()),
        alternatives_complete=complete,
        objective_optimal=bool(solution.optimal),
        plan_unique=bool(complete and full_ties.sum() == 1),
        **pairs,
    )
    for name in OPERATIONS:
        summary[name + "_operations"] = sum(o["operation"] == name for o in operations)
    return summary, dict(summary=summary, operations=operations, alternatives=alternatives)


def gate_diagnoses(data, policy, labels):
    """Why each referred specimen was referred.

    ``ASSIGNMENT_COMPETITION``: the specimen belongs to some admissible group
    that lost in the global cover. ``SINGLE_GATE_DEFICIT``: relaxing one gate
    admits a group containing it (the witness is reported).
    ``JOINT_GATE_DEFICIT``: no single relaxation is enough. Small contexts only
    (every subset is screened).
    """
    reviewed = np.flatnonzero(np.asarray(labels) < 0)
    n = len(data)
    if not len(reviewed):
        return pd.DataFrame()
    masks = np.arange(1, 1 << n, dtype=np.uint32)
    membership = ((masks[:, None] >> np.arange(n)) & 1).astype(bool)
    sizes = membership.sum(axis=1)
    keep = sizes >= 2
    masks, membership, sizes = masks[keep], membership[keep], sizes[keep]

    x = data[list(policy.properties)].to_numpy(float)
    x = x / x.max(axis=0)
    mean = membership @ x / sizes[:, None]
    var = np.maximum((membership @ (x * x) - sizes[:, None] * mean * mean) / (sizes[:, None] - 1), 0)
    sd = np.sqrt(var)
    cv = sd / mean
    lo = np.stack([np.min(np.where(membership, x[:, j], np.inf), axis=1) for j in range(x.shape[1])], axis=1)
    hi = np.stack([np.max(np.where(membership, x[:, j], -np.inf), axis=1) for j in range(x.shape[1])], axis=1)
    nu = np.divide(np.maximum(hi - mean, mean - lo), sd, out=np.zeros_like(sd), where=sd > 1e-12)

    gates = {"minimum_size": sizes >= policy.min_size}
    observed = {"minimum_size": sizes.astype(float)}
    for j, name in enumerate(policy.properties):
        gates["cv:" + name] = cv[:, j] <= policy.cv_limits()[j] + TOLERANCE
        observed["cv:" + name] = cv[:, j]
        gates["nu:" + name] = nu[:, j] <= np.array([policy.nu_limit(int(m)) for m in sizes]) + TOLERANCE
        observed["nu:" + name] = nu[:, j]
    contiguous = np.ones(len(masks), bool)
    for _, hole in data.reset_index(drop=True).groupby("borehole_id"):
        order = hole.sort_values(["depth_m", "sample_id"]).index.to_numpy()
        a = membership[:, order]
        contiguous &= a[:, 0].astype(int) + np.sum(a[:, 1:] & ~a[:, :-1], axis=1) <= 1
    gates["contiguity"] = contiguous
    barrier = np.ones(len(masks), bool)
    lookup = {str(v): i for i, v in enumerate(data.sample_id)}
    for i, row in enumerate(data.get("barrier_with_ids", pd.Series("", index=data.index)).fillna("")):
        for sample in str(row).split(";"):
            if sample in lookup:
                barrier &= ~(membership[:, i] & membership[:, lookup[sample]])
    gates["material_barrier"] = barrier
    unit = np.ones(len(masks), bool)
    if "fixed_geological_unit" in data and data.fixed_geological_unit.notna().any():
        u = data.fixed_geological_unit.fillna("__unspecified").astype(str).to_numpy()
        touched = np.stack([membership[:, u == v].any(axis=1) for v in np.unique(u)], axis=1)
        unit = touched.sum(axis=1) <= 1
    gates["fixed_unit"] = unit

    names = list(gates)
    matrix = np.stack([gates[k] for k in names], axis=1)
    failed = (~matrix).sum(axis=1)

    def witness(k):
        return int(sizes[k]), ";".join(data.loc[membership[k], "sample_id"])

    output = []
    for i in reviewed:
        containing = membership[:, i]
        valid = containing & (failed == 0)
        sample_id = data.iloc[i].sample_id
        if valid.any():
            k = int(np.flatnonzero(valid)[0])
            size, ids = witness(k)
            output.append(
                dict(
                    sample_id=sample_id,
                    reason_code="ASSIGNMENT_COMPETITION",
                    gate="none",
                    witness_n=size,
                    witness_sample_ids=ids,
                    additional_complete_records_lower_bound=0,
                )
            )
            continue
        found = False
        for j, name in enumerate(names):
            possible = containing & (~matrix[:, j]) & (failed == 1)
            if not possible.any():
                continue
            found = True
            candidates = np.flatnonzero(possible)
            k = int(candidates[np.argmax(sizes[candidates])] if name == "minimum_size" else candidates[0])
            size, ids = witness(k)
            output.append(
                dict(
                    sample_id=sample_id,
                    reason_code="SINGLE_GATE_DEFICIT",
                    gate=name,
                    witness_n=size,
                    witness_sample_ids=ids,
                    observed=float(observed[name][k]) if name in observed else None,
                    additional_complete_records_lower_bound=max(0, policy.min_size - size)
                    if name == "minimum_size"
                    else 0,
                )
            )
        if not found:
            k = int(np.flatnonzero(containing)[np.argmin(failed[containing])])
            size, ids = witness(k)
            output.append(
                dict(
                    sample_id=sample_id,
                    reason_code="JOINT_GATE_DEFICIT",
                    gate=";".join(name for j, name in enumerate(names) if not matrix[k, j]),
                    minimum_failed_gate_count=int(failed[k]),
                    witness_n=size,
                    witness_sample_ids=ids,
                    additional_complete_records_lower_bound=max(0, policy.min_size - size),
                )
            )
    return pd.DataFrame(output)
