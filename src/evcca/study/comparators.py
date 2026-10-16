"""Graph Ward and spectral-COP comparisons under the article's group gates.

Both methods generate candidate groups. The same endpoint trimming,
admissibility screening and lexicographic master select their final groups.
The master certifies its input pool, not ungenerated candidate groups.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from ..candidates import exact_geometric_columns, screen_masks
from ..gates import barrier_pairs, group_audit, ranks_by_hole, values
from ..partition import solve_master
from .cohort import load_specimens, make_contexts
from .paths import DATA, RESULTS

GRAPH_NEIGHBORS = (3, 5, 8, "all")
SPECTRAL_NEIGHBORS = (3, 5, "all")
BANDWIDTH_FACTORS = (0.5, 1.0, 2.0)
MAX_ITERATIONS = 30
SEED_BASE = 20261016
POLICIES = (("nominal", 0.0, 0.0), ("nu_5pct", 0.0, 0.05), ("cv_10pct_nu_5pct", 0.1, 0.05))
KEYS = ["context_id", "policy", "mode"]


def stable_seed(context_id, run):
    return SEED_BASE + run + int(hashlib.sha256(context_id.encode()).hexdigest()[:7], 16)


def mask_of(indices):
    return sum(1 << int(i) for i in indices)


@dataclass
class _SearchContext:
    name: str
    data: pd.DataFrame
    policy: object

    def __post_init__(self):
        self.n = len(self.data)
        self.x = values(self.data, self.policy)
        assert np.isfinite(self.x).all() and (self.x > 0).all()
        self.holes = ranks_by_hole(self.data)
        self.barriers = barrier_pairs(self.data)
        self.pair_bits = [(1 << i) | (1 << j) for i, j in self.barriers]
        self.forbidden = [0] * self.n
        for i, j in self.barriers:
            self.forbidden[i] |= 1 << j
            self.forbidden[j] |= 1 << i
        self.fixed = None
        if "fixed_geological_unit" in self.data and self.data.fixed_geological_unit.notna().any():
            self.fixed = self.data.fixed_geological_unit.fillna("__unspecified").astype(str).to_numpy()
            for i in range(self.n):
                for j in range(i):
                    if self.fixed[i] != self.fixed[j]:
                        self.forbidden[i] |= 1 << j
                        self.forbidden[j] |= 1 << i
        self.source = self.data.initial_ige.astype(str).to_numpy()
        self.full = (1 << self.n) - 1
        self.original = sum(self.source[i] == self.source[j] for i, j in combinations(range(self.n), 2))
        units = [mask_of(np.flatnonzero(self.source == label)) for label in sorted(set(self.source))]
        self.seeds = {self.full, *units, *(a | b for a, b in combinations(units, 2))}
        self._scores = {}
        self._geometry_cache = {}
        self._trim_cache = {}
        self._audit_cache = {}

    def geometry_failures(self, mask):
        if mask in self._geometry_cache:
            return self._geometry_cache[mask]
        contiguity = 0
        for order in self.holes:
            positions = [rank for rank, i in enumerate(order) if mask & (1 << int(i))]
            contiguity += bool(positions and positions[-1] - positions[0] + 1 != len(positions))
        barriers = sum((mask & pair) == pair for pair in self.pair_bits)
        fixed = 0
        if self.fixed is not None:
            fixed = max(0, len({self.fixed[i] for i in range(self.n) if mask & (1 << i)}) - 1)
        result = int(contiguity), int(barriers), int(fixed)
        self._geometry_cache[mask] = result
        return result

    def geometric(self, mask):
        return not any(self.geometry_failures(mask))

    def scores(self, masks):
        masks = [int(mask) for mask in masks]
        missing = list(set(masks) - self._scores.keys())
        if missing:
            bits = np.array(missing, dtype=np.uint64)
            member = ((bits[:, None] >> np.arange(self.n, dtype=np.uint64)) & 1).astype(bool)
            sizes = member.sum(1)
            scaled = self.x / self.x.max(0)
            means = member @ scaled / np.maximum(sizes[:, None], 1)
            variance = np.maximum(
                (member @ (scaled**2) - sizes[:, None] * means**2) / np.maximum(sizes[:, None] - 1, 1), 0
            )
            sd = np.sqrt(variance)
            cv = np.divide(sd, means, out=np.zeros_like(sd), where=means > 0)
            lows = np.stack(
                [np.min(np.where(member, scaled[:, j], np.inf), axis=1) for j in range(self.x.shape[1])], axis=1
            )
            highs = np.stack(
                [np.max(np.where(member, scaled[:, j], -np.inf), axis=1) for j in range(self.x.shape[1])], axis=1
            )
            deviation = np.maximum(highs - means, means - lows)
            nu = np.divide(deviation, sd, out=np.zeros_like(sd), where=sd > 1e-12)
            limit = np.array([self.policy.nu_limit(int(n)) for n in sizes])
            penalty = np.maximum(cv / self.policy.cv_limits() - 1, 0).sum(1)
            penalty += np.maximum(nu / limit[:, None] - 1, 0).sum(1)
            penalty += np.maximum(self.policy.min_size - sizes, 0) * 10
            for i, mask in enumerate(missing):
                if mask == 0:
                    score = 1e6
                else:
                    score = float(penalty[i] + 10 * sum(self.geometry_failures(mask)))
                    # Among passing states, prefer retained support before dispersion.
                    score += -0.001 * int(sizes[i]) + 0.00001 * float((cv[i] / self.policy.cv_limits()).sum())
                self._scores[mask] = score
        return np.array([self._scores[mask] for mask in masks])

    def trim_path(self, mask):
        if mask in self._trim_cache:
            return self._trim_cache[mask]
        sequence = []
        suffix = frozenset()
        while True:
            if mask in self._trim_cache:
                suffix = self._trim_cache[mask]
                break
            sequence.append(mask)
            if mask.bit_count() <= self.policy.min_size:
                break
            endpoints = set()
            for order in self.holes:
                hits = [int(i) for i in order if mask & (1 << int(i))]
                if hits:
                    endpoints.update([hits[0], hits[-1]])
            options = [mask ^ (1 << i) for i in sorted(endpoints)]
            if not options:
                break
            mask = options[int(np.argmin(self.scores(options)))]
        for state in reversed(sequence):
            suffix = suffix | {state}
            self._trim_cache[state] = suffix
        return self._trim_cache[sequence[0]] if sequence else suffix

    def feasible_pool(self, raw):
        proposals = set(raw) | self.seeds
        expanded = set(proposals)
        for mask in sorted(proposals):
            if mask.bit_count() >= self.policy.min_size:
                expanded.update(self.trim_path(mask))
        expanded.discard(0)
        eligible = [mask for mask in sorted(expanded) if mask.bit_count() >= self.policy.min_size]
        columns = screen_masks(self.data, self.policy, np.array(eligible, dtype=np.uint64)) if eligible else {}
        return columns, expanded

    def independent_audit(self, mask):
        if mask in self._audit_cache:
            return self._audit_cache[mask]
        indices = np.array([i for i in range(self.n) if mask & (1 << i)], int)
        n = len(indices)
        assert n > 0
        x = self.x[indices]
        cv = x.std(axis=0, ddof=1) / x.mean(0) if n > 1 else np.full(x.shape[1], np.inf)
        sd = x.std(axis=0, ddof=1) if n > 1 else np.zeros(x.shape[1])
        nu = (
            np.max(np.divide(np.abs(x - x.mean(0)), sd, out=np.zeros_like(x), where=sd > 1e-12), axis=0)
            if n > 1
            else np.full(x.shape[1], np.inf)
        )
        contiguity, barriers, fixed = self.geometry_failures(mask)
        fail = dict(
            minimum_size=n < self.policy.min_size,
            CV=bool(np.any(cv > self.policy.cv_limits() + 1e-9)),
            extremum=bool(n > 1 and np.any(nu > self.policy.nu_limit(n) + 1e-9)),
            contiguity=bool(contiguity),
            material_barrier=bool(barriers),
            fixed_unit=bool(fixed),
        )
        weights = self.policy.weights
        ratios = (cv / np.asarray(self.policy.limits)) ** 2
        b = n / self.n * 0.05 * float(np.average(ratios, weights=weights))
        self._audit_cache[mask] = (fail, b)
        return fail, b

    def solution(self, result):
        labels = result.labels
        groups = [mask_of(np.flatnonzero(labels == g)) for g in sorted(set(labels) - {-1})]
        for mask in groups:
            failures, cost = self.independent_audit(mask)
            assert not any(failures.values()), (self.name, failures)
            audit = group_audit(self.data, np.array([i for i in range(self.n) if mask & (1 << i)]), self.policy)
            assert audit["compliant"] and np.isclose(cost, audit["cost"], atol=1e-10, rtol=1e-10)
        a = int((labels < 0).sum())
        d = sum(
            (self.source[i] == self.source[j]) != (labels[i] >= 0 and labels[i] == labels[j])
            for i, j in combinations(range(self.n), 2)
        )
        b = sum(self.independent_audit(mask)[1] for mask in groups)
        strict = bool(groups) and a <= self.policy.budget(self.n)
        return dict(
            n=self.n,
            A=a,
            K=len(groups),
            D=int(d),
            B=b,
            assigned_n=self.n - a,
            strict_accepted_n=self.n - a if strict else 0,
            strict_feasible=strict,
            selected_group_violations=0,
        )


def _representations(ctx):
    for name, x in [("raw", ctx.x), ("log", np.log(ctx.x))]:
        sd = x.std(0)
        yield name, (x - x.mean(0)) / np.where(sd > 1e-12, sd, 1.0)


def _graph_for(ctx, z, neighbors):
    distance = ((z[:, None] - z[None, :]) ** 2).sum(2)
    adjacency = np.zeros((ctx.n, ctx.n), bool)
    if neighbors == "all":
        adjacency[:] = True
    else:
        for i in range(ctx.n):
            order = np.argsort(distance[i], kind="stable")
            order = order[order != i][: int(neighbors)]
            adjacency[i, order] = True
        adjacency |= adjacency.T
    for order in ctx.holes:
        for i, j in zip(order[:-1], order[1:]):
            adjacency[i, j] = adjacency[j, i] = True
    for i in range(ctx.n):
        for j in range(ctx.n):
            if ctx.forbidden[i] & (1 << j):
                adjacency[i, j] = False
    np.fill_diagonal(adjacency, False)
    return adjacency, distance


def _ward_proposals(ctx):
    output = set()
    merges = 0
    for _, z in _representations(ctx):
        for neighbors in GRAPH_NEIGHBORS:
            adjacency, _ = _graph_for(ctx, z, neighbors)
            active = {1 << i: (1, z[i].copy()) for i in range(ctx.n)}
            output.update(active)
            while True:
                best = None
                for a, b in combinations(sorted(active), 2):
                    joined = a | b
                    if not ctx.geometric(joined):
                        continue
                    ia = [i for i in range(ctx.n) if a & (1 << i)]
                    ib = [i for i in range(ctx.n) if b & (1 << i)]
                    if not adjacency[np.ix_(ia, ib)].any():
                        continue
                    na, ca = active[a]
                    nb, cb = active[b]
                    cost = na * nb / (na + nb) * float(np.sum((ca - cb) ** 2))
                    candidate = (cost, a, b)
                    if best is None or candidate < best:
                        best = candidate
                if best is None:
                    break
                _, a, b = best
                na, ca = active.pop(a)
                nb, cb = active.pop(b)
                active[a | b] = na + nb, (na * ca + nb * cb) / (na + nb)
                output.add(a | b)
                merges += 1
    return output, dict(hierarchies=8, merges=merges)


def _spectral_embeddings(ctx):
    configurations = []
    kmax = min(ctx.policy.max_groups, ctx.n // ctx.policy.min_size)
    for name, z in _representations(ctx):
        for neighbors in SPECTRAL_NEIGHBORS:
            adjacency, distance = _graph_for(ctx, z, neighbors)
            positive = distance[np.triu(adjacency, 1) & (distance > 1e-12)]
            scale = float(np.median(positive)) if len(positive) else 1.0
            for factor in BANDWIDTH_FACTORS:
                affinity = np.exp(-distance / (2 * scale * factor**2)) * adjacency
                degree = affinity.sum(1)
                inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
                normalized = affinity * inv[:, None] * inv[None, :]
                _, vectors = eigh(normalized)
                for k in range(1, kmax + 1):
                    embedding = vectors[:, -k:][:, ::-1]
                    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
                    embedding = np.divide(embedding, norm, out=np.zeros_like(embedding), where=norm > 1e-12)
                    configurations.append((name, neighbors, factor, k, embedding))
    return configurations


def _cop_assign(ctx, embedding, k, rng):
    chosen = [int(rng.integers(ctx.n))]
    while len(chosen) < k:
        squared = ((embedding[:, None] - embedding[chosen][None, :]) ** 2).sum(2).min(1)
        squared[chosen] = 0
        if squared.sum() > 1e-12:
            chosen.append(int(rng.choice(ctx.n, p=squared / squared.sum())))
        else:
            chosen.append(int(rng.choice(sorted(set(range(ctx.n)) - set(chosen)))))
    centers = embedding[chosen].copy()
    hole_order = rng.permutation(len(ctx.holes))
    previous = np.full(ctx.n, -2, int)
    for iteration in range(MAX_ITERATIONS):
        labels = np.full(ctx.n, -1, int)
        group_masks = [0] * k
        for hole_index in hole_order:
            closed = set()
            last = -1
            for point in ctx.holes[hole_index]:
                point = int(point)
                distances = np.sum((centers - embedding[point]) ** 2, axis=1)
                permitted = [
                    int(label)
                    for label in np.argsort(distances, kind="stable")
                    if label not in closed and not (group_masks[label] & ctx.forbidden[point])
                ]
                label = permitted[0] if permitted else -1
                if last >= 0 and label != last:
                    closed.add(last)
                if label >= 0:
                    labels[point] = label
                    group_masks[label] |= 1 << point
                last = label
        if np.array_equal(labels, previous):
            return labels, iteration + 1, True
        previous = labels.copy()
        for label in range(k):
            points = embedding[labels == label]
            centers[label] = points.mean(0) if len(points) else embedding[int(rng.integers(ctx.n))]
    return labels, MAX_ITERATIONS, False


def _spectral_proposals(ctx, configurations, run):
    rng = np.random.default_rng(stable_seed(ctx.name, run))
    output = set()
    converged = iterations = raw_reviews = 0
    for _, _, _, k, embedding in configurations:
        labels, it, done = _cop_assign(ctx, embedding, k, rng)
        for label in set(labels) - {-1}:
            mask = mask_of(np.flatnonzero(labels == label))
            assert ctx.geometric(mask)
            output.add(mask)
        converged += done
        iterations += it
        raw_reviews += int((labels < 0).sum())
    return output, dict(
        partitions=len(configurations),
        converged=converged,
        assignment_iterations=iterations,
        reviews_before_property_gates=raw_reviews,
    )


def graph_ward_columns(data, policy):
    """Feasible columns from eight graph-constrained Ward forests."""
    context = _SearchContext(str(data.context_id.iloc[0]), data.reset_index(drop=True), policy)
    proposals, _ = _ward_proposals(context)
    return context.feasible_pool(proposals)[0]


def spectral_cop_columns(data, policy, run=0):
    """Feasible columns from the spectral grid at one declared random seed."""
    context = _SearchContext(str(data.context_id.iloc[0]), data.reset_index(drop=True), policy)
    proposals, _ = _spectral_proposals(context, _spectral_embeddings(context), run)
    return context.feasible_pool(proposals)[0]


def _diagnostic(context, masks):
    counts = Counter()
    eligible = [mask for mask in masks if mask.bit_count() >= context.policy.min_size]
    for mask in eligible:
        failures, _ = context.independent_audit(mask)
        counts.update({key: int(value) for key, value in failures.items()})
        counts["any_violation"] += int(any(failures.values()))
    return dict(
        raw_proposals=len(masks),
        raw_below_minimum=len(masks) - len(eligible),
        raw_size_eligible=len(eligible),
        **dict(counts),
    )


def _save_json(value, path):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _reference_points():
    reference = pd.read_csv(RESULTS / "reserve_grid.csv", low_memory=False)
    reference = reference[reference.profile.eq("four")].copy()
    names = {(delta, eta): name for name, delta, eta in POLICIES}
    reference["policy"] = [names.get((float(d), float(e))) for d, e in zip(reference.cv_reserve, reference.nu_reserve)]
    reference = reference[
        reference.policy.notna() & (reference["mode"].eq("compact") | reference.dataset.eq("PRIMARY"))
    ]
    reference["run"] = -1
    return reference


def compare_objectives(frame):
    """Match generated solutions to the complete optimum, with B tolerance 1e-9."""
    reference = _reference_points()
    exact = reference[reference.method.eq("evcca")]
    gold = exact[KEYS + ["A", "D", "K", "B"]].rename(columns={k: "exact_" + k for k in ["A", "D", "K", "B"]})
    combined = pd.concat([frame, reference], ignore_index=True)
    combined = combined.merge(gold, on=KEYS, validate="many_to_one")
    leading, full = [], []
    for row in combined.to_dict("records"):
        names = ["A", "D", "K"] if row["mode"] == "repair" else ["A", "K"]
        actual = tuple(int(row[name]) for name in names)
        target = tuple(int(row["exact_" + name]) for name in names)
        if actual < target or (actual == target and row["B"] < row["exact_B"] - 1e-8):
            raise AssertionError("A proposal solution is better than the complete reference objective.")
        leading.append(actual == target)
        full.append(actual == target and abs(row["B"] - row["exact_B"]) <= 1e-9)
    combined["leading_match"] = leading
    combined["full_match"] = full
    return combined


def cohort_summary(compared):
    """Cohort totals per run, followed by minimum/median/maximum over runs."""
    totals = (
        compared.groupby(["policy", "mode", "method", "run"], sort=False)
        .agg(
            contexts=("context_id", "size"),
            assigned=("assigned_n", "sum"),
            accepted=("strict_accepted_n", "sum"),
            leading=("leading_match", "sum"),
            full=("full_match", "sum"),
            strict_contexts=("strict_feasible", "sum"),
        )
        .reset_index()
    )
    for row in totals.to_dict("records"):
        if row["contexts"] != (19 if row["mode"] == "compact" else 11):
            raise AssertionError("Incomplete context coverage in a comparator run.")
    rows = []
    for (point, mode, method), group in totals.groupby(["policy", "mode", "method"], sort=False):
        row = dict(policy=point, mode=mode, method=method, runs=len(group), contexts=int(group.contexts.iloc[0]))
        for metric in ["assigned", "accepted", "leading", "full", "strict_contexts"]:
            row.update(
                {metric + "_" + stat: float(getattr(group[metric], stat)()) for stat in ["min", "median", "max"]}
            )
        rows.append(row)
    return pd.DataFrame(rows), totals


def _record_solution(context, columns, method, run, base, elapsed, modes, rows, assignments, audits):
    for mode in modes:
        result = solve_master(context.data, context.policy, columns, mode, scope=method + "_proposal_pool")
        if not result.optimal:
            raise RuntimeError("The restricted master did not finish its lexicographic optimization.")
        score = context.solution(result)
        rows.append(
            dict(
                **base,
                method=method,
                run=run,
                mode=mode,
                **score,
                feasible_columns=len(columns),
                pool_optimal=bool(result.optimal),
                scope=result.scope,
                generation_seconds=elapsed,
                selection_seconds=result.seconds,
            )
        )
        for sample, label in zip(context.data.sample_id, result.labels):
            assignments.append(dict(**base, method=method, run=run, mode=mode, sample_id=sample, group=int(label)))
        for group in sorted(set(result.labels) - {-1}):
            members = np.flatnonzero(result.labels == group)
            audit = group_audit(context.data, members, context.policy)
            audits.append(
                dict(
                    **base,
                    method=method,
                    run=run,
                    mode=mode,
                    group=int(group),
                    n=len(members),
                    compliant=bool(audit["compliant"]),
                    contiguous=bool(audit["contiguous"]),
                    max_cv_ratio=float(np.max(audit["cv"] / context.policy.cv_limits())),
                    max_nu_ratio=float(np.max(audit["nu"] / audit["nu_critical"])),
                )
            )


def _check_complete_reference():
    reference = _reference_points()
    rows = []
    for point, delta, eta in POLICIES:
        for context_id, data, policy in make_contexts(load_specimens(), delta=delta, eta=eta):
            columns = exact_geometric_columns(data, policy)
            modes = ["compact", "repair"] if data.dataset.iloc[0] == "PRIMARY" else ["compact"]
            for mode in modes:
                result = solve_master(data, policy, columns, mode)
                if not result.optimal:
                    raise AssertionError("Complete reference recomputation is not certified.")
                context = _SearchContext(context_id, data, policy)
                score = context.solution(result)
                old = reference[
                    reference.context_id.eq(context_id)
                    & reference.policy.eq(point)
                    & reference["mode"].eq(mode)
                    & reference.method.eq("evcca")
                ].iloc[0]
                names = ["A", "D", "K", "B"] if mode == "repair" else ["A", "K", "B"]
                if any(not np.isclose(score[name], old[name], atol=1e-8, rtol=0) for name in names):
                    raise AssertionError("Complete objective differs from the reference.")
                rows.append(dict(context_id=context_id, policy=point, mode=mode, **score, matches_reference=True))
    return pd.DataFrame(rows)


def run_comparison(output, seeds=20):
    """Run the two proposal comparators and the shared-seed control on fresh outputs."""
    if seeds != 20:
        raise ValueError("The article protocol uses exactly 20 spectral runs.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, assignments, audits, diagnostics = [], [], [], []
    for context_id, data, nominal in make_contexts(load_specimens()):
        original = _SearchContext(context_id, data, nominal)
        started = perf_counter()
        ward, ward_meta = _ward_proposals(original)
        ward_seconds = perf_counter() - started
        started = perf_counter()
        embeddings = _spectral_embeddings(original)
        embedding_seconds = perf_counter() - started
        spectral = []
        for seed in range(seeds):
            started = perf_counter()
            proposals, meta = _spectral_proposals(original, embeddings, seed)
            spectral.append((proposals, meta, perf_counter() - started + embedding_seconds))
        for point, delta, eta in POLICIES:
            policy = replace(nominal, cv_margin_delta=delta, nu_margin_eta=eta)
            context = _SearchContext(context_id, data, policy)
            modes = ["compact", "repair"] if data.dataset.iloc[0] == "PRIMARY" else ["compact"]
            base = dict(
                context_id=context_id,
                dataset=data.dataset.iloc[0],
                policy=point,
                cv_reserve=delta,
                nu_reserve=eta,
                profile="four",
            )
            methods = [("source_union", -1, set(), {}, 0.0), ("graph_ward", -1, ward, ward_meta, ward_seconds)]
            methods += [
                ("spectral_cop", seed, raw, meta, elapsed) for seed, (raw, meta, elapsed) in enumerate(spectral)
            ]
            for method, seed, proposals, meta, generation in methods:
                started = perf_counter()
                columns, expanded = context.feasible_pool(proposals)
                elapsed = generation + perf_counter() - started
                diagnostics.append(
                    dict(
                        **base,
                        method=method,
                        run=seed,
                        screened_proposals=len(expanded),
                        **_diagnostic(context, proposals),
                        **meta,
                    )
                )
                _record_solution(context, columns, method, seed, base, elapsed, modes, rows, assignments, audits)
        print("Compared:", context_id, flush=True)
    complete_check = _check_complete_reference()
    results = pd.DataFrame(rows)
    compared = compare_objectives(results)
    summary, totals = cohort_summary(compared)
    frames = dict(
        results=results,
        assignments=pd.DataFrame(assignments),
        group_audit=pd.DataFrame(audits),
        proposal_diagnostics=pd.DataFrame(diagnostics),
        compared_results=compared,
        summary=summary,
        cohort_runs=totals,
        complete_reference_check=complete_check,
    )
    for name, frame in frames.items():
        frame.to_csv(output / (name + ".csv"), index=False, mode="x")
    provenance = dict(
        profile="four",
        contexts=19,
        specimens=247,
        spectral_runs=seeds,
        seed_base=SEED_BASE,
        policies=[dict(name=name, cv_reserve=d, nu_reserve=e) for name, d, e in POLICIES],
        graph_neighbors=list(GRAPH_NEIGHBORS),
        spectral_neighbors=list(SPECTRAL_NEIGHBORS),
        bandwidth_factors=list(BANDWIDTH_FACTORS),
        max_assignment_iterations=MAX_ITERATIONS,
        shared_seeds="Full context, source units, pairwise unions of source units",
        shared_selection="Greedy endpoint trimming, raw-property gates, lexicographic restricted master",
        spectral_assignment="Row-normalized eigenvectors, nearest permitted centers, borehole order, review",
        scope="Three reserve settings, four-property cohort; repair on primary contexts only",
        complete_reference_objectives_recomputed=90,
        python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]},
        input_sha256=hashlib.sha256((DATA / "processed/specimens.csv").read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing="Descriptive only: shared preprocessing and memoization; no wall-time ranking",
    )
    _save_json(provenance, output / "provenance.json")
    manifest = dict(
        status="complete",
        rows=len(results),
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()},
    )
    _save_json(manifest, output / "manifest.json")
    return verify_comparison(output)


def verify_comparison(directory=None, deep=True):
    """Verify checksums, objective comparisons, and every selected specimen group."""
    directory = Path(directory) if directory is not None else RESULTS / "comparators"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["rows"] != 1980:
        raise AssertionError("Unexpected comparison coverage.")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise AssertionError("Comparison checksum mismatch: " + name)
    rows = pd.read_csv(directory / "results.csv")
    if rows.duplicated(KEYS + ["method", "run"]).any() or len(rows) != 1980:
        raise AssertionError("Missing or duplicate comparator results.")
    expected = compare_objectives(rows)
    saved = pd.read_csv(directory / "compared_results.csv")
    check_keys = KEYS + ["method", "run"]
    expected = expected.sort_values(check_keys).set_index(check_keys)
    saved = saved.sort_values(check_keys).set_index(check_keys)
    if not expected.index.equals(saved.index) or not expected[["leading_match", "full_match"]].equals(
        saved[["leading_match", "full_match"]]
    ):
        raise AssertionError("Objective comparisons differ from reference results.")
    summary, _ = cohort_summary(expected.reset_index())
    reported = pd.read_csv(directory / "summary.csv")
    summary = summary.set_index(["policy", "mode", "method"]).sort_index()
    reported = reported.set_index(["policy", "mode", "method"]).sort_index()
    if not summary.index.equals(reported.index) or not np.allclose(
        summary.select_dtypes("number"), reported.select_dtypes("number")
    ):
        raise AssertionError("Cohort summaries differ.")
    audited = 0
    if deep:
        provenance = json.loads((directory / "provenance.json").read_text())
        if provenance["input_sha256"] != hashlib.sha256((DATA / "processed/specimens.csv").read_bytes()).hexdigest():
            raise AssertionError("Comparison inputs differ.")
        contexts = {name: (data, policy) for name, data, policy in make_contexts(load_specimens())}
        assignments = pd.read_csv(directory / "assignments.csv")
        for key, group in assignments.groupby(KEYS + ["method", "run"], sort=False):
            context_id, point, mode, method, seed = key
            data, nominal = contexts[context_id]
            _, delta, eta = next(p for p in POLICIES if p[0] == point)
            context = _SearchContext(context_id, data, replace(nominal, cv_margin_delta=delta, nu_margin_eta=eta))
            if group.sample_id.duplicated().any() or set(group.sample_id) != set(data.sample_id):
                raise AssertionError("A comparator assignment omits or repeats a specimen.")
            labels = data.sample_id.map(group.set_index("sample_id").group).to_numpy(int)
            holder = type("Assignment", (), {"labels": labels})()
            actual = context.solution(holder)
            record = rows[
                rows.context_id.eq(context_id)
                & rows.policy.eq(point)
                & rows["mode"].eq(mode)
                & rows.method.eq(method)
                & rows.run.eq(seed)
            ].iloc[0]
            for name in ["A", "D", "K", "B", "assigned_n", "strict_accepted_n"]:
                if not np.isclose(actual[name], record[name], atol=1e-9, rtol=0):
                    raise AssertionError("Assignment statistics differ: " + name)
            audited += actual["K"]
    return dict(
        status="verified",
        outcomes=len(rows),
        selected_groups_audited=audited,
        admissibility_violations=0,
        contexts=19,
        specimens=247,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="new folder for complete comparison results")
    action.add_argument("--verify", type=Path, help="check saved comparison results")
    args = parser.parse_args()
    report = run_comparison(args.output) if args.output else verify_comparison(args.verify)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
