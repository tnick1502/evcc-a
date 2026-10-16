"""Recompute every result of the article from the files in ``data/``.

Usage: ``evcca-reproduce --output runs/full-study`` (about 40 min on one core).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import date
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import t

from .. import __version__
from ..candidates import interval_combinations
from ..gates import group_audit
from ..ordered import best_ordered
from ..partition import solve_master
from ..plans import build_plan, gate_diagnoses
from ..policy import NU_TABLE_AMENDMENT_1
from ..solve import candidate_family, summarize
from . import diagnostics
from .cohort import EXCLUDED_SITES, load_specimens, make_contexts, with_reserves
from .paths import DATA
from .repair import depth_bands

CV_RESERVES = (0.0, 0.05, 0.10)
NU_RESERVES = (0.0, 0.01, 0.02, 0.05)
DELETION_RESERVES = ((0.0, 0.0), (0.0, 0.02), (0.0, 0.05), (0.10, 0.05))
PLAN_NU_RESERVES = (0.0, 0.05)


def _json(value, path):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))
        stream.write("\n")


def _csv(frame, path):
    frame.to_csv(path, index=False, mode="x")


def _modes(data):
    return ["compact", "repair"] if data.dataset.iloc[0] == "PRIMARY" else ["compact"]


def solve_pair(data, policy, columns, scope, mode):
    """Ordered comparator and EVCC-A over a complete family; both re-audited."""
    ordered, _ = best_ordered(data, policy, columns, mode)
    joint = solve_master(data, policy, columns, mode, scope=scope, witness=ordered)
    if not joint.optimal:
        raise RuntimeError("Master problem not solved to optimality.")
    for partition in (joint, ordered):
        for label in set(partition.labels) - {-1}:
            if not group_audit(data, np.flatnonzero(partition.labels == label), policy)["compliant"]:
                raise RuntimeError("A selected group fails its gates.")
    return {"ordered": ordered, "evcca": joint}


def provenance():
    packages = ["numpy", "pandas", "scipy", "scikit-learn"]
    return dict(
        evcca_version=__version__,
        date=date.today().isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        machine=platform.machine(),
        packages={name: importlib.metadata.version(name) for name in packages},
        input_sha256=hashlib.sha256((DATA / "processed/all_specimens.csv").read_bytes()).hexdigest(),
        cv_reserves=CV_RESERVES,
        nu_reserves=NU_RESERVES,
        timing="Wall time of candidate enumeration plus both solves; input, imports and figures excluded.",
    )


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    specimens = load_specimens()
    all_specimens = load_specimens(active=False)
    if len(specimens) != 247 or specimens.context_id.nunique() != 19:
        raise RuntimeError("Unexpected active cohort.")
    _json(provenance(), out / "provenance.json")

    grid, timings, audits, assignments, plan_rows, reasons, oracle_rows = [], [], [], [], [], [], []
    plans = {}
    contexts = {}
    nominal = {}
    columns_nominal = {}
    for profile in ("four", "nine"):
        contexts[profile] = make_contexts(specimens, profile)
        for delta, eta in product(CV_RESERVES, NU_RESERVES):
            accepted = {"ordered": 0, "evcca": 0}
            for context_id, data, base in contexts[profile]:
                policy = with_reserves(base, delta, eta)
                started = perf_counter()
                columns, scope = candidate_family(data, policy)
                enumeration = perf_counter() - started
                if columns is None:
                    raise RuntimeError(f"{context_id}: complete family exceeds the enumeration budget.")
                for mode in _modes(data):
                    started = perf_counter()
                    results = solve_pair(data, policy, columns, scope, mode)
                    paired = perf_counter() - started
                    for method, solution in results.items():
                        row, groups = summarize(data, solution, policy)
                        row.update(
                            context_id=context_id,
                            dataset=data.dataset.iloc[0],
                            site_id=data.site_id.iloc[0],
                            mode=mode,
                            method=method,
                            profile=profile,
                            cv_reserve=delta,
                            nu_reserve=eta,
                        )
                        grid.append(row)
                        if mode == "compact":
                            accepted[method] += row["strict_accepted_n"]
                        if delta == eta == 0:
                            groups[["context_id", "method", "mode", "profile"]] = [context_id, method, mode, profile]
                            audits.append(groups)
                            labelled = data[["sample_id", "context_id", "borehole_id", "depth_m", "initial_ige"]].copy()
                            labelled[["method", "mode", "profile"]] = [method, mode, profile]
                            labelled["group"] = solution.labels
                            assignments.append(labelled)
                    if delta == eta == 0:
                        nominal[profile, context_id, mode] = results["evcca"]
                        columns_nominal[profile, context_id] = columns
                        timings.append(
                            dict(
                                profile=profile,
                                context_id=context_id,
                                mode=mode,
                                n=len(data),
                                boreholes=data.borehole_id.nunique(),
                                N_geom=interval_combinations(data),
                                admissible_groups=len(columns),
                                enumeration_seconds=enumeration,
                                paired_solve_seconds=paired,
                                paired_total_seconds=enumeration + paired,
                                scope=scope,
                            )
                        )
                        if scope == "complete_subset_space":
                            check = diagnostics.oracle_check(data, policy, columns, mode, results["evcca"])
                            check.update(context_id=context_id, profile=profile)
                            oracle_rows.append(check)
                    primary = data.dataset.iloc[0] == "PRIMARY"
                    if primary and delta == 0 and eta in PLAN_NU_RESERVES:
                        summary, plan = build_plan(data, columns, results["evcca"], mode)
                        summary.update(profile=profile, context_id=context_id, cv_reserve=delta, nu_reserve=eta)
                        plan["summary"] = summary
                        plans[f"{profile}::{context_id}::{mode}::nu{eta:.2f}"] = plan
                        plan_rows.append(summary)
                    if primary and delta == eta == 0 and mode == "compact":
                        reason = gate_diagnoses(data, policy, results["evcca"].labels)
                        if not reason.empty:
                            reason["context_id"] = context_id
                            reason["profile"] = profile
                            reasons.append(reason)
            print(f"grid  {profile}  cv {delta:.2f}  nu {eta:.2f}  accepted {accepted}", flush=True)

    grid = pd.DataFrame(grid)
    _csv(grid, out / "reserve_grid.csv")
    _csv(grid[grid.cv_reserve.eq(0) & grid.nu_reserve.eq(0)], out / "metrics.csv")
    _csv(pd.concat(audits, ignore_index=True), out / "group_audit.csv")
    assignments = pd.concat(assignments, ignore_index=True)
    _csv(assignments, out / "assignments.csv")
    _csv(pd.DataFrame(timings), out / "scale_timing.csv")
    _csv(pd.DataFrame(plan_rows), out / "revision_plan_summary.csv")
    _json(plans, out / "revision_plans.json")
    _csv(pd.concat(reasons, ignore_index=True), out / "review_reasons.csv")
    _csv(pd.DataFrame(oracle_rows), out / "oracle_checks.csv")

    deletions = []
    for profile in ("four", "nine"):
        for delta, eta in DELETION_RESERVES:
            for context_id, data, base in contexts[profile]:
                policy = with_reserves(base, delta, eta)
                for hole in sorted(data.borehole_id.unique()):
                    train = data[data.borehole_id.ne(hole)].reset_index(drop=True)
                    common = dict(
                        profile=profile, context_id=context_id, cv_reserve=delta, nu_reserve=eta, excluded_borehole=hole
                    )
                    if len(train) < policy.min_size:
                        for method in ("ordered", "evcca"):
                            deletions.append(
                                dict(
                                    common,
                                    method=method,
                                    n=len(train),
                                    strict_feasible=False,
                                    A=len(train),
                                    K=0,
                                    strict_accepted_n=0,
                                    status="insufficient_training_support",
                                )
                            )
                        continue
                    columns, scope = candidate_family(train, policy)
                    for method, solution in solve_pair(train, policy, columns, scope, "compact").items():
                        row, _ = summarize(train, solution, policy)
                        deletions.append(dict(row, **common, method=method))
            print(f"deletions  {profile}  cv {delta:.2f}  nu {eta:.2f}", flush=True)
    _csv(pd.DataFrame(deletions), out / "borehole_deletions.csv")

    four = contexts["four"]
    tolsa = next(d for c, d, p in four if c == diagnostics.TOLSA)
    tolsa_policy = next(p for c, d, p in four if c == diagnostics.TOLSA)
    checks, witnesses = diagnostics.hull_certificate(tolsa, columns_nominal["four", diagnostics.TOLSA])
    _csv(checks, out / "tolsa_projection_checks.csv")
    _json(witnesses, out / "tolsa_hull_witnesses.json")
    tolsa_labels = nominal["four", diagnostics.TOLSA, "compact"].labels
    _csv(diagnostics.depth_trend(tolsa, tolsa_labels), out / "tolsa_depth_trend.csv")
    _csv(diagnostics.block_resampling(tolsa, tolsa_labels, tolsa_policy), out / "tolsa_block_resampling.csv")
    _csv(diagnostics.tolsa_ablations(tolsa), out / "tolsa_ablations.csv")
    print("tolsa diagnostics", flush=True)

    _csv(diagnostics.source_units(four), out / "source_unit_support.csv")
    _csv(diagnostics.void_ratio_density(four), out / "void_ratio_density.csv")
    signature, export = diagnostics.strength_signature(all_specimens)
    _csv(signature, out / "strength_signature.csv")
    _csv(diagnostics.row_signature_sensitivity(export, EXCLUDED_SITES), out / "row_signature_sensitivity.csv")
    _json(diagnostics.haarajoki_crust(specimens), out / "haarajoki_crust.json")
    _csv(diagnostics.profile_comparison(specimens), out / "profile_comparison.csv")
    print("source diagnostics", flush=True)

    returned = []
    for profile in ("four", "nine"):
        solutions = {(c, m): s for (p, c, m), s in nominal.items() if p == profile}
        frame = diagnostics.return_experiment(contexts[profile], solutions)
        frame.insert(0, "profile", profile)
        returned.append(frame)
    _csv(pd.concat(returned, ignore_index=True), out / "return_experiment.csv")
    print("return experiment", flush=True)

    _, groups, modes = depth_bands(specimens, assignments)
    _csv(groups, out / "repair_depth_groups.csv")
    _csv(modes, out / "repair_depth_modes.csv")

    grubbs = []
    for n, critical in NU_TABLE_AMENDMENT_1.items():
        q = t.ppf(1 - 0.05 / (2 * n), n - 2)
        g = (n - 1) / np.sqrt(n) * np.sqrt(q * q / (n - 2 + q * q))
        grubbs.append(dict(n=n, reference=critical, two_sided_grubbs_5pct=g, difference=critical - g))
    _csv(pd.DataFrame(grubbs), out / "grubbs_comparison.csv")

    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    _json(
        dict(active_contexts=19, active_specimens=247, nu_table_version="amendment_1_2024", files=files),
        out / "manifest.json",
    )
    print("done:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Recompute all article results into a new directory.")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the results")
    run(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
