"""Acceptance policy: CV limits, extremum critical values and the review budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t

# fmt: off
# GOST 20522-2012, Table E.1 as first published. The software convention of
# that period divided by the population standard deviation.
NU_TABLE_2012 = {
    3: 1.41, 4: 1.71, 5: 1.92, 6: 2.07, 7: 2.18, 8: 2.27, 9: 2.35, 10: 2.41,
    11: 2.47, 12: 2.52, 13: 2.56, 14: 2.60, 15: 2.64, 16: 2.67, 17: 2.70,
    18: 2.73, 19: 2.75, 20: 2.78, 21: 2.80, 22: 2.82, 23: 2.84, 24: 2.86,
    25: 2.88, 26: 2.90, 27: 2.91, 28: 2.93, 29: 2.94, 30: 2.96, 31: 2.97,
    32: 2.98, 33: 3.00, 34: 3.01, 35: 3.02, 36: 3.03, 37: 3.04, 38: 3.05,
    39: 3.06, 40: 3.07, 41: 3.08, 42: 3.09, 43: 3.10, 44: 3.11, 45: 3.12,
    46: 3.13, 47: 3.14, 48: 3.14, 49: 3.15, 50: 3.16,
}

# Table E.1 as replaced by Amendment No. 1 (in force from 1 January 2024);
# used with the sample standard deviation.
NU_TABLE_AMENDMENT_1 = dict(zip(range(3, 51), (
    1.16, 1.48, 1.72, 1.89, 2.02, 2.13, 2.22, 2.29, 2.36, 2.41, 2.46, 2.51,
    2.55, 2.59, 2.62, 2.65, 2.68, 2.71, 2.73, 2.76, 2.78, 2.80, 2.82, 2.84,
    2.86, 2.88, 2.89, 2.91, 2.92, 2.94, 2.95, 2.97, 2.98, 2.99, 3.00, 3.01,
    3.02, 3.04, 3.05, 3.06, 3.07, 3.08, 3.09, 3.10, 3.11, 3.12, 3.13, 3.14,
)))
# fmt: on

TABLE_VERSIONS = ("legacy_2012", "amendment_1_2024")


def nu_critical(n, version="amendment_1_2024"):
    """Critical value of the extremum statistic for a group of ``n`` specimens."""
    if version not in TABLE_VERSIONS:
        raise ValueError(f"Unknown table version: {version}")
    table = NU_TABLE_2012 if version == "legacy_2012" else NU_TABLE_AMENDMENT_1
    if n in table:
        return table[n]
    if n < 3:
        return np.inf
    if version == "amendment_1_2024":
        raise ValueError("The amended table is defined only for n = 3..50.")
    # Grubbs-type extension above the 2012 table (population SD convention).
    q = t.ppf(1 - 0.05 / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(q * q / (n - 2 + q * q)) * np.sqrt(n / (n - 1))


@dataclass(frozen=True)
class Policy:
    """Group acceptance rules and solver settings.

    ``limits`` are the CV limits of ``properties``. ``cv_margin_delta`` and
    ``nu_margin_eta`` tighten the CV and extremum limits by a fraction.
    """

    properties: tuple[str, ...] = ("rho", "e0", "cohesion", "modulus")
    limits: tuple[float, ...] = (0.15, 0.15, 0.30, 0.30)
    weights: tuple[float, ...] | None = None
    min_size: int = 6
    max_groups: int = 12
    review_fraction: float = 0.10
    at_least_one_review: bool = True
    nu_gate: bool = True
    nu_sample_sd: bool = True
    nu_table_version: str = "amendment_1_2024"
    cv_margin_delta: float = 0.0
    nu_margin_eta: float = 0.0
    borehole_contiguity: bool = True
    exact_n: int = 18
    max_exact_geometry_combinations: int = 500_000
    seconds: float = 15
    seed: int = 1701
    random_orders: int = 0

    def __post_init__(self):
        if self.nu_table_version not in TABLE_VERSIONS:
            raise ValueError(f"Unknown table version: {self.nu_table_version}")
        if self.nu_table_version == "amendment_1_2024" and not self.nu_sample_sd:
            raise ValueError("The amended table requires the sample SD (nu_sample_sd=True).")
        if not 0 <= self.cv_margin_delta < 1:
            raise ValueError("cv_margin_delta must be in [0, 1).")
        if not 0 <= self.nu_margin_eta < 1:
            raise ValueError("nu_margin_eta must be in [0, 1).")
        if not self.properties or len(self.properties) != len(self.limits):
            raise ValueError("Give one CV limit per property.")
        limits = np.asarray(self.limits, dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("CV limits must be positive and finite.")

    def cv_limits(self):
        return np.asarray(self.limits) * (1 - self.cv_margin_delta)

    def nu_limit(self, n):
        return nu_critical(n, self.nu_table_version) * (1 - self.nu_margin_eta)

    def budget(self, n):
        """Largest number of referred specimens for a strictly feasible context."""
        return max(int(self.at_least_one_review), int(np.floor(self.review_fraction * n)))

    @classmethod
    def historical_2012(cls, **kwargs):
        """The 2012 table with the population SD, as used before the amendment."""
        return cls(nu_sample_sd=False, nu_table_version="legacy_2012", **kwargs)
