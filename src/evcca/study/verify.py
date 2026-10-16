"""Check the reference results, and optionally compare them with a new run.

Usage: ``evcca-verify`` or ``evcca-verify --run runs/full-study``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import rand_score

from .. import certificate
from ..plans import revision_operations
from . import ficlay, robustness
from .barriers import primary_barriers
from .cohort import load_specimens
from .comparators import verify_comparison
from .paths import DATA, RESULTS

# Tables compared with a new run: key columns and value columns.
COMPARED = {
    "reserve_grid": (
        ["profile", "context_id", "mode", "method", "cv_reserve", "nu_reserve"],
        ["n", "A", "K", "strict_feasible", "strict_accepted_n", "D", "B", "optimal"],
    ),
    "borehole_deletions": (
        ["profile", "context_id", "method", "cv_reserve", "nu_reserve", "excluded_borehole"],
        ["n", "A", "K", "strict_feasible", "strict_accepted_n", "B", "optimal"],
    ),
    "revision_plan_summary": (
        ["profile", "context_id", "mode", "cv_reserve", "nu_reserve"],
        ["A", "D", "K", "B", "leading_optimal_plans", "full_objective_optimal_plans", "plan_unique"],
    ),
    "profile_comparison": (
        ["profile", "context_id"],
        ["A", "K", "strict_feasible", "strict_accepted_n", "source_units_passing"],
    ),
    "oracle_checks": (["profile", "context_id", "mode"], ["matches"]),
    "tolsa_ablations": (["variant", "method"], ["A", "K", "optimal_or_at_bound"]),
    "return_experiment": (
        ["profile", "context_id", "mode", "method", "excluded_borehole"],
        ["test_n", "assigned_new_n", "augmented_groups", "augmented_violating_groups"],
    ),
    "tolsa_block_resampling": (["group", "property"], ["cv", "p05", "p95", "exceedance_fraction"]),
}


class VerificationError(AssertionError):
    pass


def check(condition, message):
    if not condition:
        raise VerificationError(message)


def checksums(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        check(actual == expected, f"checksum mismatch: {name}")
    return len(manifest["files"])


def cohort():
    specimens = load_specimens()
    check(len(specimens) == 247 and specimens.context_id.nunique() == 19, "active cohort is not 19 / 247")
    check(not specimens.sample_id.duplicated().any(), "duplicate sample_id")
    released = pd.read_csv(DATA / "processed/specimens.csv", low_memory=False)
    check(released.sample_id.tolist() == specimens.sample_id.tolist(), "specimens.csv differs from the active cohort")
    logs = pd.read_csv(DATA / "processed/primary_logs.csv", low_memory=False)
    derived = primary_barriers(logs)
    check(derived.tolist() == logs.barrier_with_ids.fillna("").tolist(), "barriers differ from the borehole logs")
    primary = specimens[specimens.dataset.eq("PRIMARY")].set_index("sample_id").barrier_with_ids.fillna("")
    check(primary.equals(derived.reindex(primary.index)), "specimen barriers differ from the borehole logs")
    fi_clay = ficlay.select()
    released = load_specimens(active=False)
    released = released[released.dataset.eq("FI-CLAY")].sort_values("sample_id").reset_index(drop=True)
    check(fi_clay.sample_id.tolist() == released.sample_id.tolist(), "FI-CLAY records differ from the workbook")
    numeric = fi_clay.select_dtypes("number").columns
    check(
        np.allclose(fi_clay[numeric], released[numeric], rtol=1e-12, atol=0, equal_nan=True),
        "FI-CLAY values differ from the workbook",
    )
    return dict(contexts=19, specimens=247, primary_barrier_rows=len(logs), fi_clay_records=len(fi_clay))


def plans(directory):
    specimens = load_specimens()
    reported = json.loads((Path(directory) / "revision_plans.json").read_text())
    selected, everything = Counter(), Counter()
    alternatives = rand_checks = 0
    for plan in reported.values():
        summary = plan["summary"]
        data = specimens[specimens.context_id.eq(summary["context_id"])].sort_values("sample_id").reset_index(drop=True)
        position = {sample: i for i, sample in enumerate(data.sample_id)}
        for alternative in plan["alternatives"]:
            members = [s for group in alternative["groups"] for s in group] + alternative["review"]
            check(sorted(members) == sorted(data.sample_id), "plan does not cover its context exactly")
            labels = np.full(len(data), -1)
            for group, samples in enumerate(alternative["groups"]):
                labels[[position[s] for s in samples]] = group
            operations, pairs = revision_operations(data, labels)
            everything.update(o["operation"] for o in operations)
            alternatives += 1
            check(
                int((labels < 0).sum()) == summary["A"] and len(alternative["groups"]) == summary["K"],
                "alternative has a different A or K",
            )
            if summary["mode"] == "repair":
                check(sum(pairs.values()) == summary["D"], "alternative has a different D")
            if alternative["full_objective_optimal"]:
                selected.update(o["operation"] for o in operations)
                singletons = np.where(labels < 0, len(data) + np.arange(len(data)), labels)
                ri = rand_score(data.initial_ige.astype(str), singletons)
                check(
                    np.isclose(ri, 1 - sum(pairs.values()) / comb(len(data), 2), atol=1e-12, rtol=0),
                    "Rand index identity fails",
                )
                rand_checks += 1
    check(rand_checks == len(reported), "a plan set has no unique full-objective optimum")
    return dict(
        plan_sets=len(reported),
        alternatives=alternatives,
        selected_operations=dict(selected),
        all_operations=dict(everything),
    )


def compare(directory):
    """Compare the deterministic columns of a new run with the reference."""
    directory = Path(directory)
    checksums(directory)
    compared = {}
    for name, (keys, columns) in COMPARED.items():
        old = pd.read_csv(RESULTS / f"{name}.csv", low_memory=False).sort_values(keys).set_index(keys)
        new = pd.read_csv(directory / f"{name}.csv", low_memory=False).sort_values(keys).set_index(keys)
        check(old.index.equals(new.index), f"{name}: different rows")
        for column in columns:
            if pd.api.types.is_numeric_dtype(old[column]) and pd.api.types.is_numeric_dtype(new[column]):
                same = np.allclose(old[column], new[column], atol=1e-8, rtol=1e-8, equal_nan=True)
            else:
                same = old[column].astype(str).equals(new[column].astype(str))
            check(same, f"{name}.{column} differs")
        compared[name] = len(new)
    return compared


def main():
    parser = argparse.ArgumentParser(description="Verify the reference results.")
    parser.add_argument("--run", type=Path, help="compare a new evcca-reproduce output with results/")
    args = parser.parse_args()
    report = dict(reference_files=checksums(RESULTS), cohort=cohort(), plans=plans(RESULTS))
    witnesses = json.loads((RESULTS / "tolsa_hull_witnesses.json").read_text())["witnesses"]
    report["hull_certificate"] = certificate.verify(witnesses[0]["exact_decimal_certificate"])
    if (RESULTS / "comparators/manifest.json").is_file():
        report["graph_spectral"] = verify_comparison()
    if (robustness.OUTPUT / "manifest.json").is_file():
        report["robustness"] = robustness.verify()
    if args.run:
        report["compared_rows"] = compare(args.run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
