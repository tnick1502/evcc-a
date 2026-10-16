"""Checks on the article data: Tolsa, the P13 loam context and the barrier lists."""

import numpy as np
import pandas as pd
import pytest

from evcca import (
    best_ordered,
    exact_columns,
    exact_geometric_columns,
    group_audit,
    solve_context,
    solve_master,
    summarize,
)
from evcca.study import diagnostics, ficlay
from evcca.study.barriers import primary_barriers
from evcca.study.cohort import article_policy, load_specimens
from evcca.study.paths import DATA
from evcca.study.repair import CONTEXT, depth_bands

from helpers import frame, small_policy


@pytest.fixture(scope="module")
def specimens():
    return load_specimens()


@pytest.fixture(scope="module")
def tolsa(specimens):
    return specimens[specimens.site_id.eq("Tolsa, Kirkkonummi")].sort_values("sample_id").reset_index(drop=True)


def test_tolsa_joint_search_retains_all_specimens(tolsa):
    policy = article_policy("FI-CLAY")
    columns = exact_geometric_columns(tolsa, policy)
    ordered, _ = best_ordered(tolsa, policy, columns)
    joint = solve_master(tolsa, policy, columns, witness=ordered)
    assert len(columns) == 2001
    assert ordered.objective[:2] == (1, 3) and joint.objective[:2] == (0, 2) and joint.optimal
    summary, groups = summarize(tolsa, joint, policy)
    assert summary["strict_feasible"] and summary["strict_accepted_n"] == 24
    assert groups[groups.n.eq(16)].iloc[0].nu_critical == 2.59


def test_tolsa_cv_reserve_costs_a_group(tolsa):
    policy = article_policy("FI-CLAY", delta=0.10)
    columns = exact_geometric_columns(tolsa, policy)
    ordered, _ = best_ordered(tolsa, policy, columns)
    result = solve_master(tolsa, policy, columns, witness=ordered)
    assert result.optimal and result.objective[:2] == (0, 3)
    for group in set(result.labels) - {-1}:
        audit = group_audit(tolsa, np.flatnonzero(result.labels == group), policy)
        assert np.all(audit["cv"] <= np.array([0.27, 0.135, 0.135, 0.27]) + 1e-9)


def test_p13_repair_separates_upper_and_lower_loam(specimens):
    context = specimens[specimens.context_id.eq(CONTEXT)].sort_values("sample_id").reset_index(drop=True)
    policy = article_policy("PRIMARY")
    columns = exact_columns(context, policy)
    assignments = []
    for mode in ("compact", "repair"):
        ordered, _ = best_ordered(context, policy, columns, mode)
        result = solve_master(context, policy, columns, mode, witness=ordered)
        assert result.optimal
        assignments.append(
            context[["sample_id", "context_id"]].assign(method="evcca", mode=mode, profile="four", group=result.labels)
        )
    _, groups, modes = depth_bands(specimens, pd.concat(assignments))
    modes = modes.set_index("mode")
    assert modes.retained.eq(17).all() and modes.K.eq(2).all()
    assert modes.D.to_dict() == {"compact": 59, "repair": 39}
    assert modes.cross_band_pairs.to_dict() == {"compact": 33, "repair": 0}
    repair = groups[groups["mode"].eq("repair")]
    assert sorted(zip(repair.upper_n, repair.lower_n)) == [(0, 11), (6, 0)]


def test_barriers_follow_from_the_borehole_logs(specimens):
    logs = pd.read_csv(DATA / "processed/primary_logs.csv", low_memory=False)
    derived = primary_barriers(logs)
    assert derived.tolist() == logs.barrier_with_ids.fillna("").tolist()
    primary = specimens[specimens.dataset.eq("PRIMARY")].set_index("sample_id").barrier_with_ids.fillna("")
    assert primary.equals(derived.reindex(primary.index))


def test_returned_specimens_join_their_source_unit():
    # Two source units in three boreholes and no fixed geological units.
    values = [[1.0, 1.0], [1.05, 1.02], [10.0, 10.0], [10.2, 10.1]] * 3
    data = frame(values, labels=["upper", "upper", "lower", "lower"] * 3)
    data["borehole_id"] = [hole for hole in "ABC" for _ in range(4)]
    data["depth_m"] = [1.0, 2.0, 5.0, 6.0] * 3
    data["dataset"] = "PRIMARY"
    policy = small_policy(min_size=4)
    solutions = {("site", mode): solve_context(data, policy, mode)[0]["evcca"] for mode in ("compact", "repair")}
    returned = diagnostics.return_experiment([("site", data, policy)], solutions)
    assert len(returned) == 2 * 2 * 3
    assert returned.assigned_new_n.eq(4).all() and returned.augmented_violating_groups.eq(0).all()


def test_fi_clay_records_follow_from_the_workbook():
    selected = ficlay.select()
    released = load_specimens(active=False)
    released = released[released.dataset.eq("FI-CLAY")].sort_values("sample_id").reset_index(drop=True)
    assert selected.sample_id.tolist() == released.sample_id.tolist()
    assert selected.groupby("site_id").size().to_dict() == {
        "Haapa-Kimola": 24,
        "Haarajoki": 21,
        "Söderkulla-Nikkilä": 18,
        "Tolsa, Kirkkonummi": 24,
        "Östersundom": 8,
    }
    for column in selected.columns:
        if pd.api.types.is_numeric_dtype(selected[column]):
            assert np.allclose(selected[column], released[column], rtol=1e-12, atol=0, equal_nan=True), column
        else:
            assert selected[column].tolist() == released[column].tolist(), column
    sites = ficlay.site_counts()
    assert len(sites) == 33 and set(sites.index[sites.eligible]) == set(selected.site_id)
