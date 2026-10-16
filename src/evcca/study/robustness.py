"""Robustness of the comparisons and of the complete family.

``python -m evcca.study.robustness --output runs/robustness`` writes

- ``review_budgets.csv``: within-budget specimens (assigned specimens of strictly
  feasible contexts) and strictly feasible contexts of every compact result at
  the three comparison policies when the review budget changes;
- ``deletion_outcomes.csv``: borehole deletions, split by whether the context was
  strictly feasible before the deletion, with the reason for each loss;
- ``wide_search.csv``: the pools of Graph Ward and all 20 spectral runs joined,
  and Graph Ward and Spectral + COP over a wider grid with 100 seeds, separately
  and joined, under the same gates and master;
- ``family_check.csv``: the interval-enumerated family of every context against
  a screen of all its subsets;
- ``gate_failures.csv``: how often each gate rejects an interval candidate of the
  primary contexts.

The first two tables are derived from ``results/``; the others are computed from
the data (about ten minutes on two cores). ``--verify`` checks a saved folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..candidates import exact_geometric_columns, interval_combinations, screen_masks
from ..gates import TOLERANCE, barrier_pairs, ranks_by_hole, values
from ..partition import solve_master
from . import comparators
from .cohort import load_specimens, make_contexts, with_reserves
from .paths import RESULTS

# Review budgets: (name, fraction of n, at least). The study uses "10pct".
BUDGETS = (
    ("none", 0.0, 0),
    ("one", 0.0, 1),
    ("10pct", 0.10, 1),
    ("20pct", 0.20, 0),
    ("25pct", 0.25, 0),
    ("unlimited", 1.0, 0),
)
WIDE_GRID = dict(
    GRAPH_NEIGHBORS=(2, 3, 4, 5, 6, 8, 10, 12, "all"),
    SPECTRAL_NEIGHBORS=(2, 3, 4, 5, 6, 8, 10, "all"),
    BANDWIDTH_FACTORS=(0.25, 0.5, 1.0, 1.5, 2.0, 4.0),
    MAX_ITERATIONS=100,
)
WIDE_SEEDS = 100
FAMILY_POLICIES = ((0.0, 0.0), (0.10, 0.05))
OUTPUT = RESULTS / "robustness"


def compact_results(results=RESULTS):
    """(A, K) of every compact four-property result at the three comparison policies."""
    names = {(delta, eta): name for name, delta, eta in comparators.POLICIES}
    grid = pd.read_csv(Path(results) / "reserve_grid.csv")
    grid = grid[grid["mode"].eq("compact") & grid.profile.eq("four")].copy()
    grid["policy"] = [names.get(key) for key in zip(grid.cv_reserve, grid.nu_reserve)]
    grid = grid[grid.policy.notna()].assign(run=-1)
    proposals = pd.read_csv(Path(results) / "comparators/results.csv")
    proposals = proposals[proposals["mode"].eq("compact")]
    columns = ["policy", "method", "run", "context_id", "n", "A", "K", "strict_accepted_n"]
    return pd.concat([grid[columns], proposals[columns]], ignore_index=True)


def review_budgets(results=RESULTS):
    """Within-budget specimens and strictly feasible contexts per method when the review budget changes."""
    frame = compact_results(results)
    tables = []
    for name, fraction, at_least in BUDGETS:
        limit = np.maximum(at_least, np.floor(fraction * frame.n))
        strict = frame.K.ge(1) & frame.A.le(limit)
        totals = (
            frame.assign(within=np.where(strict, frame.n - frame.A, 0), strict=strict)
            .groupby(["policy", "method", "run"])[["within", "strict"]]
            .sum()
            .groupby(["policy", "method"])
            .agg(
                runs=("within", "size"),
                within_budget_min=("within", "min"),
                within_budget_median=("within", "median"),
                within_budget_max=("within", "max"),
                strict_min=("strict", "min"),
                strict_max=("strict", "max"),
            )
            .reset_index()
        )
        totals.insert(2, "budget", name)
        tables.append(totals)
    return pd.concat(tables, ignore_index=True)


def deletion_outcomes(results=RESULTS):
    """Borehole deletions by strict feasibility before the deletion, with the reason for each loss."""
    grid = pd.read_csv(Path(results) / "reserve_grid.csv")
    grid = grid[grid["mode"].eq("compact")]
    before = grid.set_index(["profile", "context_id", "cv_reserve", "nu_reserve", "method"]).strict_feasible
    deletions = pd.read_csv(Path(results) / "borehole_deletions.csv")
    keys = ["profile", "context_id", "cv_reserve", "nu_reserve", "method"]
    deletions["strict_before"] = before.reindex(pd.MultiIndex.from_frame(deletions[keys])).to_numpy(bool)
    lost = ~deletions.strict_feasible
    too_few = deletions.status.eq("insufficient_training_support")
    no_group = lost & ~too_few & deletions.K.eq(0)
    over = lost & ~too_few & deletions.K.ge(1)
    frame = deletions.assign(
        too_few_specimens=too_few,
        no_admissible_group=no_group,
        over_budget=over,
        over_budget_by_one=over & deletions.A.eq(deletions.review_budget + 1),
    )
    return (
        frame.groupby(["profile", "cv_reserve", "nu_reserve", "method", "strict_before"])
        .agg(
            deletions=("strict_feasible", "size"),
            strict_after=("strict_feasible", "sum"),
            too_few_specimens=("too_few_specimens", "sum"),
            no_admissible_group=("no_admissible_group", "sum"),
            over_budget=("over_budget", "sum"),
            over_budget_by_one=("over_budget_by_one", "sum"),
        )
        .reset_index()
    )


@contextmanager
def _comparator_grid(settings):
    saved = {name: getattr(comparators, name) for name in settings}
    for name, value in settings.items():
        setattr(comparators, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(comparators, name, value)


def _proposals(context_id, data, policy, seeds):
    original = comparators._SearchContext(context_id, data, policy)
    ward, _ = comparators._ward_proposals(original)
    embeddings = comparators._spectral_embeddings(original)
    spectral = set()
    for seed in range(seeds):
        spectral |= comparators._spectral_proposals(original, embeddings, seed)[0]
    return set(ward), spectral


def _wide_rows(item):
    context_id, data, nominal = item
    ward, spectral = _proposals(context_id, data, nominal, 20)
    pools = {"joined": ward | spectral}
    with _comparator_grid(WIDE_GRID):
        ward, spectral = _proposals(context_id, data, nominal, WIDE_SEEDS)
    pools.update(graph_ward_wide=ward, spectral_cop_wide=spectral, joined_wide=ward | spectral)
    rows = []
    for point, delta, eta in comparators.POLICIES:
        policy = replace(nominal, cv_margin_delta=delta, nu_margin_eta=eta)
        context = comparators._SearchContext(context_id, data, policy)
        modes = ["compact", "repair"] if data.dataset.iloc[0] == "PRIMARY" else ["compact"]
        for method, raw in pools.items():
            columns, _ = context.feasible_pool(raw)
            for mode in modes:
                result = solve_master(context.data, policy, columns, mode, scope=method + "_proposal_pool")
                if not result.optimal:
                    raise RuntimeError("The restricted master did not finish its lexicographic optimization.")
                rows.append(
                    dict(
                        context_id=context_id,
                        policy=point,
                        mode=mode,
                        method=method,
                        **context.solution(result),
                        proposals=len(raw),
                        feasible_columns=len(columns),
                    )
                )
    return rows


def wide_search(processes=2):
    """Joined proposal pools of the article protocol and of a wider grid with more seeds."""
    from multiprocessing import Pool

    items = make_contexts(load_specimens())
    with Pool(processes) as pool:
        rows = pool.map(_wide_rows, items, chunksize=1)
    return pd.DataFrame([row for part in rows for row in part])


def subset_family(data, policy, chunk=1 << 20):
    """Admissible groups found by screening every subset of the context."""
    n = len(data)
    x, ranks, barriers = values(data, policy), ranks_by_hole(data), barrier_pairs(data)
    columns = {}
    for start in range(0, 1 << n, chunk):
        masks = np.arange(start, min(start + chunk, 1 << n), dtype=np.uint64)
        sizes = sum(((masks >> np.uint64(i)) & np.uint64(1)).astype(np.int64) for i in range(n))
        masks = masks[sizes >= policy.min_size]
        if len(masks):
            columns.update(screen_masks(data, policy, masks, x, ranks, barriers))
    return columns


def family_check():
    """Interval enumeration against the screen of all subsets, for every context."""
    rows = []
    for delta, eta in FAMILY_POLICIES:
        for context_id, data, base in make_contexts(load_specimens()):
            policy = with_reserves(base, delta, eta)
            intervals = exact_geometric_columns(data, policy)
            subsets = subset_family(data, policy)
            identical = set(intervals) == set(subsets) and all(
                abs(intervals[mask] - subsets[mask]) < 1e-12 for mask in intervals
            )
            rows.append(
                dict(
                    context_id=context_id,
                    cv_reserve=delta,
                    nu_reserve=eta,
                    n=len(data),
                    subsets=2 ** len(data),
                    N_geom=interval_combinations(data),
                    interval_family=len(intervals),
                    subset_family=len(subsets),
                    identical=identical,
                )
            )
    return pd.DataFrame(rows)


def gate_failures():
    """Interval candidates of the primary contexts (at least six specimens, no barrier) rejected by each gate."""
    rows = []
    for context_id, data, policy in make_contexts(load_specimens()):
        if data.dataset.iloc[0] != "PRIMARY":
            continue
        n = len(data)
        x, ranks = values(data, policy), ranks_by_hole(data)
        options = []
        for order in ranks:
            masks = [0]
            for start in range(len(order)):
                mask = 0
                for end in range(start, len(order)):
                    mask |= 1 << int(order[end])
                    masks.append(mask)
            options.append(masks)
        candidates = {sum(parts) for parts in product(*options)}
        masks = np.array(sorted(m for m in candidates if m.bit_count() >= policy.min_size), dtype=np.uint64)
        for i, j in barrier_pairs(data):
            pair = np.uint64((1 << i) | (1 << j))
            masks = masks[(masks & pair) != pair]
        member = ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)
        sizes = member.sum(axis=1)
        means = member @ x / sizes[:, None]
        sd = np.sqrt(np.maximum((member @ x**2 - sizes[:, None] * means**2) / (sizes[:, None] - 1), 0))
        cv_fail = sd / means > policy.cv_limits() + TOLERANCE
        low = np.stack([np.where(member, x[:, j], np.inf).min(axis=1) for j in range(x.shape[1])], axis=1)
        high = np.stack([np.where(member, x[:, j], -np.inf).max(axis=1) for j in range(x.shape[1])], axis=1)
        deviation = np.maximum(high - means, means - low)
        nu = np.divide(deviation, sd, out=np.zeros_like(deviation), where=sd > 1e-12)
        limit = np.array([policy.nu_limit(int(k)) for k in sizes])
        nu_fail = nu > limit[:, None] + TOLERANCE
        for j, name in enumerate(policy.properties):
            rows.append(
                dict(
                    context_id=context_id,
                    property=name,
                    candidates=len(masks),
                    cv_fail=int(cv_fail[:, j].sum()),
                    extremum_fail=int(nu_fail[:, j].sum()),
                )
            )
    return pd.DataFrame(rows)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables = dict(
        review_budgets=review_budgets(),
        deletion_outcomes=deletion_outcomes(),
        wide_search=wide_search(),
        family_check=family_check(),
        gate_failures=gate_failures(),
    )
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, mode="x")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}
    settings = dict(WIDE_GRID, GRAPH_NEIGHBORS=list(map(str, WIDE_GRID["GRAPH_NEIGHBORS"])))
    settings["SPECTRAL_NEIGHBORS"] = list(map(str, WIDE_GRID["SPECTRAL_NEIGHBORS"]))
    manifest = dict(wide_grid=settings, wide_seeds=WIDE_SEEDS, budgets=[list(b) for b in BUDGETS], files=files)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(output)


def verify(directory=OUTPUT):
    """Checksums, the derived tables against ``results/``, and the family check."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise AssertionError("Robustness checksum mismatch: " + name)
    for name, function in (("review_budgets", review_budgets), ("deletion_outcomes", deletion_outcomes)):
        saved = pd.read_csv(directory / f"{name}.csv")
        derived = function()
        if not np.allclose(saved.select_dtypes("number"), derived.select_dtypes("number")):
            raise AssertionError(f"{name} differs from the results it is derived from.")
    budgets = pd.read_csv(directory / "review_budgets.csv").query("budget == '10pct'")
    accepted = compact_results().groupby(["policy", "method", "run"]).strict_accepted_n.sum()
    accepted = accepted.groupby(["policy", "method"]).agg(["min", "max"])
    study = budgets.set_index(["policy", "method"])[["within_budget_min", "within_budget_max"]]
    if not np.array_equal(study.loc[accepted.index].to_numpy(), accepted.to_numpy()):
        raise AssertionError("The study budget does not reproduce the saved within-budget counts.")
    family = pd.read_csv(directory / "family_check.csv")
    if not family.identical.all():
        raise AssertionError("Interval enumeration and subset screening differ.")
    wide = pd.read_csv(directory / "wide_search.csv")
    return dict(
        status="verified",
        family_contexts=int(family.context_id.nunique()),
        family_policies=int(len(family) // family.context_id.nunique()),
        wide_outcomes=len(wide),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="new folder for the robustness tables")
    action.add_argument("--verify", type=Path, help="check a saved folder")
    args = parser.parse_args()
    report = run(args.output) if args.output else verify(args.verify)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
