import numpy as np

from evcca import best_ordered, exact_columns, exact_oracle, solve_context, solve_master, summarize

from helpers import frame, small_policy


def test_repair_keeps_an_admissible_source_partition():
    d = frame(np.ones((6, 2)), ["x"] * 3 + ["y"] * 3)
    policy = small_policy()
    columns = exact_columns(d, policy)
    assert solve_master(d, policy, columns).objective[:2] == (0, 1)
    repair = solve_master(d, policy, columns, mode="repair")
    assert repair.objective[:3] == (0, 0, 2)
    assert summarize(d, repair, policy)[0]["D"] == 0


def test_master_oracle_and_orders_agree_on_a_small_case():
    d = frame([[1, 1], [1.1, 1], [0.9, 1], [3, 3], [3.1, 3], [2.9, 3]])
    policy = small_policy()
    columns = exact_columns(d, policy)
    master = solve_master(d, policy, columns)
    oracle = exact_oracle(d, policy, columns)
    ordered, _ = best_ordered(d, policy, columns)
    assert master.objective[:2] == oracle.objective[:2] == ordered.objective[:2] == (0, 2)
    assert np.isclose(master.objective[-1], oracle.objective[-1])


def test_partial_cover_reports_review():
    d = frame([[1, 1]] * 5 + [[9, 9]])
    policy = small_policy()
    row, _ = summarize(d, solve_master(d, policy, exact_columns(d, policy)), policy)
    assert row["A"] == 1 and row["assigned_n"] == 5
    assert row["review_budget"] == 1 and row["strict_feasible"]


def test_joint_search_is_never_worse_than_orders():
    # Two groups that interleave along every single property.
    rng = np.random.default_rng(5)
    a = np.exp(rng.normal([0, 0], 0.05, (6, 2)))
    b = np.exp(rng.normal([0.25, -0.25], 0.05, (6, 2)))
    d = frame(np.r_[a, b], boreholes=["A"] * 6 + ["B"] * 6)
    policy = small_policy(min_size=6, exact_n=12, limits=(0.12, 0.12))
    result, _ = solve_context(d, policy)
    assert result["evcca"].objective[:2] <= result["ordered"].objective[:2]
    assert result["evcca"].objective[:2] == result["oracle"].objective[:2]


def test_compact_mode_needs_no_source_units():
    d = frame([[1, 1], [1.1, 1], [0.9, 1], [3, 3], [3.1, 3], [2.9, 3]]).drop(columns="initial_ige")
    result, _ = solve_context(d, small_policy())
    assert result["evcca"].objective[:2] == (0, 2)
