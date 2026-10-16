from dataclasses import replace

import numpy as np

from evcca import Policy, exact_columns, exact_geometric_columns, group_audit
from evcca.candidates import screen_masks

from helpers import frame, small_policy


def members(mask, n):
    return np.flatnonzero([(int(mask) >> j) & 1 for j in range(n)])


def all_masks(n, min_size):
    return np.array([m for m in range(1, 1 << n) if m.bit_count() >= min_size], dtype=np.uint64)


def test_subset_batch_and_direct_screening_agree(random_context):
    d = random_context
    policy = small_policy()
    masks = all_masks(10, 3)
    direct = {int(m): group_audit(d, members(m, 10), policy) for m in masks}
    expected = {m for m, row in direct.items() if row["compliant"]}
    batch = screen_masks(d, policy, masks)
    assert set(exact_columns(d, policy)) == set(batch) == expected
    for mask in expected:
        assert np.isclose(batch[mask], direct[mask]["cost"])


def test_interval_enumeration_is_complete():
    rng = np.random.default_rng(88)
    d = frame(np.exp(rng.normal(size=(10, 2)) * 0.2), boreholes=["A"] * 5 + ["B"] * 5)
    policy = small_policy()
    subsets = exact_columns(d, policy)
    intervals = exact_geometric_columns(d, policy)
    assert set(subsets) == set(intervals)
    assert all(np.isclose(subsets[m], intervals[m]) for m in subsets)


def test_extremum_gate_with_reserve_agrees_across_paths():
    rng = np.random.default_rng(772)
    d = frame(np.exp(rng.normal(0, 0.18, (10, 2))), boreholes=["A"] * 5 + ["B"] * 5)
    policy = Policy(properties=("a", "b"), limits=(0.3, 0.3), nu_margin_eta=0.05)
    masks = all_masks(10, 6)
    expected = {int(m) for m in masks if group_audit(d, members(m, 10), policy)["compliant"]}
    assert (
        set(exact_columns(d, policy))
        == set(exact_geometric_columns(d, policy))
        == set(screen_masks(d, policy, masks))
        == expected
    )
    assert expected < set(exact_columns(d, replace(policy, nu_margin_eta=0)))


def test_logged_barrier_blocks_a_group():
    d = frame(np.ones((6, 2)))
    d["barrier_with_ids"] = ["5", "", "", "", "", "0"]
    assert not group_audit(d, np.arange(6), small_policy())["compliant"]
    assert (1 << 6) - 1 not in exact_columns(d, small_policy())


def test_empty_fixed_unit_column_changes_nothing():
    d = frame(np.ones((6, 2)))
    before = group_audit(d, np.arange(6), small_policy())
    d["fixed_geological_unit"] = np.nan
    after = group_audit(d, np.arange(6), small_policy())
    assert before["compliant"] and after["compliant"]


def test_extremum_gate_catches_one_outlier_in_a_large_group():
    d = frame(np.c_[np.r_[np.ones(11), 1.8], np.ones(12)])
    row = group_audit(d, np.arange(12), Policy(properties=("a", "b"), limits=(0.3, 0.3)))
    assert row["cv"][0] < 0.3 and row["nu"][0] > row["nu_critical"]
    assert not row["compliant"]


def test_property_weights_enter_the_tie_breaker():
    d = frame([[1, 1], [1.1, 1.25], [0.9, 0.95], [1.05, 1.1]])
    policy = Policy(properties=("a", "b"), limits=(0.3, 0.3), weights=(3.0, 1.0), min_size=3, nu_gate=False)
    mask = (1 << len(d)) - 1
    expected = group_audit(d, np.arange(len(d)), policy)["cost"]
    assert np.isclose(exact_columns(d, policy)[mask], expected)
    assert np.isclose(screen_masks(d, policy, np.array([mask], dtype=np.uint64))[mask], expected)
