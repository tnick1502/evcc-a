import numpy as np
import pandas as pd

from evcca import Partition, Policy, build_plan, gate_diagnoses, revision_operations


def uniform(n):
    return pd.DataFrame(
        {
            "a": np.ones(n),
            "b": np.ones(n),
            "sample_id": [str(i) for i in range(n)],
            "borehole_id": [str(i) for i in range(n)],
            "depth_m": np.arange(n),
            "initial_ige": ["x"] * n,
        }
    )


def test_an_optimum_need_not_be_unique():
    d = uniform(12)
    half = (1 << 6) - 1
    full = (1 << 12) - 1
    even = sum(1 << i for i in range(0, 12, 2))
    columns = {half: 0.0, full ^ half: 0.0, even: 0.0, full ^ even: 0.0}
    solution = Partition(np.r_[np.zeros(6, int), np.ones(6, int)], (0, 2, 0.0), "optimal", "fixture", True, 0)
    summary, plan = build_plan(d, columns, solution, "compact")
    assert summary["objective_optimal"] and not summary["plan_unique"]
    assert summary["leading_optimal_plans"] == summary["full_objective_optimal_plans"] == 2
    assert len(plan["alternatives"]) == 2


def test_review_losses_are_not_splits():
    d = uniform(12)
    d["initial_ige"] = ["x"] * 6 + ["y"] * 6
    operations, pairs = revision_operations(d, np.r_[np.zeros(5, int), -1, np.ones(6, int)])
    assert pairs == dict(review_pair_losses=5, assigned_pair_splits=0, pair_fusions=0)
    assert [o["operation"] for o in operations].count("review") == 1
    assert not any(o["operation"] == "split" for o in operations)


def test_barrier_is_reported_as_the_blocking_gate():
    d = uniform(8)
    d["barrier_with_ids"] = [";".join(str(i) for i in range(1, 8))] + ["0"] * 7
    reasons = gate_diagnoses(d, Policy(properties=("a", "b"), limits=(0.3, 0.3)), np.r_[-1, np.zeros(7, int)])
    assert set(reasons.sample_id) == {"0"}
    assert "material_barrier" in set(reasons.gate)
    assert not reasons.reason_code.eq("ASSIGNMENT_COMPETITION").any()
