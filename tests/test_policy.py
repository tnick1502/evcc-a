from dataclasses import replace

import pytest

from evcca import Policy, nu_critical
from evcca.policy import NU_TABLE_AMENDMENT_1


def test_amended_table_values():
    assert set(NU_TABLE_AMENDMENT_1) == set(range(3, 51))
    assert [nu_critical(n) for n in (3, 6, 8, 16, 24, 50)] == [1.16, 1.89, 2.13, 2.59, 2.80, 3.14]
    with pytest.raises(ValueError, match="only for"):
        nu_critical(51)


def test_amended_table_requires_sample_sd():
    with pytest.raises(ValueError, match="sample SD"):
        Policy(nu_table_version="amendment_1_2024", nu_sample_sd=False)


def test_default_and_2012_policies_differ():
    assert Policy().nu_sample_sd and Policy().nu_limit(16) == 2.59
    old = Policy.historical_2012()
    assert not old.nu_sample_sd and old.nu_limit(16) == 2.67


def test_reserves_scale_the_limits():
    policy = replace(Policy(), cv_margin_delta=0.10, nu_margin_eta=0.05)
    assert policy.nu_limit(16) == pytest.approx(2.59 * 0.95)
    assert policy.cv_limits() == pytest.approx([0.135, 0.135, 0.27, 0.27])


def test_review_budget():
    policy = Policy()
    assert [policy.budget(n) for n in (8, 12, 24)] == [1, 1, 2]
