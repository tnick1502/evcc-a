import numpy as np

from evcca import exact_geometric_columns
from evcca.study import robustness

from helpers import frame, small_policy


def test_complete_search_keeps_at_least_as_many_specimens_under_every_budget():
    budgets = robustness.review_budgets()
    for (policy, budget), part in budgets.groupby(["policy", "budget"]):
        complete = part.loc[part.method.eq("evcca"), "within_budget_min"].item()
        assert (part.within_budget_max <= complete).all(), (policy, budget)
    study = budgets[budgets.budget.eq("10pct") & budgets.policy.eq("nu_5pct")].set_index("method")
    assert study.within_budget_min.to_dict() == {
        "evcca": 120,
        "graph_ward": 105,
        "ordered": 105,
        "source_union": 83,
        "spectral_cop": 105,
    }


def test_deletions_split_by_feasibility_before_the_deletion():
    outcomes = robustness.deletion_outcomes().set_index(
        ["profile", "cv_reserve", "nu_reserve", "method", "strict_before"]
    )
    kept = outcomes.loc[("four", 0.0, 0.0, "evcca", True)]
    gained = outcomes.loc[("four", 0.0, 0.0, "evcca", False)]
    assert (kept.deletions, kept.strict_after, gained.deletions, gained.strict_after) == (75, 58, 46, 4)
    lost = kept.deletions - kept.strict_after
    assert kept.too_few_specimens + kept.no_admissible_group + kept.over_budget == lost


def test_interval_enumeration_matches_the_screen_of_all_subsets():
    rng = np.random.default_rng(5)
    data = frame(np.exp(rng.normal(size=(11, 2)) * 0.2), boreholes=["A"] * 4 + ["B"] * 4 + ["C"] * 3)
    policy = small_policy()
    intervals = exact_geometric_columns(data, policy)
    subsets = robustness.subset_family(data, policy, chunk=256)
    assert intervals and set(intervals) == set(subsets)
    assert all(np.isclose(intervals[mask], subsets[mask]) for mask in intervals)
