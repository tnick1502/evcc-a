"""Real-data regression checks for graph and spectral proposal comparators."""

import numpy as np
import pytest

from evcca import group_audit, solve_master
from evcca.study import comparators
from evcca.study.cohort import article_policy, load_specimens
from evcca.study.repair import CONTEXT


@pytest.fixture(scope="module")
def specimens():
    return load_specimens()


@pytest.mark.parametrize("method,expected", [("graph", (1, 3)), ("spectral", (0, 3))])
def test_tolsa_comparator_outcome_and_all_group_gates(specimens, method, expected):
    data = specimens[specimens.site_id.eq("Tolsa, Kirkkonummi")].sort_values("sample_id").reset_index(drop=True)
    policy = article_policy("FI-CLAY")
    function = comparators.graph_ward_columns if method == "graph" else comparators.spectral_cop_columns
    columns = function(data, policy)
    result = solve_master(data, policy, columns, scope="restricted_proposal_pool")
    assert result.objective[:2] == expected and result.optimal
    for label in set(result.labels) - {-1}:
        assert group_audit(data, np.flatnonzero(result.labels == label), policy)["compliant"]


def test_spectral_seed_is_repeatable(specimens):
    data = specimens[specimens.site_id.eq("Tolsa, Kirkkonummi")].sort_values("sample_id").reset_index(drop=True)
    policy = article_policy("FI-CLAY")
    assert comparators.spectral_cop_columns(data, policy, run=3) == comparators.spectral_cop_columns(
        data, policy, run=3
    )


def test_common_source_union_pool_reproduces_p13_repair(specimens):
    data = specimens[specimens.context_id.eq(CONTEXT)].sort_values("sample_id").reset_index(drop=True)
    policy = article_policy("PRIMARY")
    context = comparators._SearchContext(CONTEXT, data, policy)
    columns, _ = context.feasible_pool(set())
    result = solve_master(data, policy, columns, mode="repair")
    assert result.objective[:3] == (0, 39, 2)
    bands = []
    for label in set(result.labels):
        group = data[result.labels == label]
        bands.append((len(group), float(group.depth_m.min()), float(group.depth_m.max())))
    assert sorted(bands) == [(6, 41.0, 53.0), (11, 67.0, 84.5)]
