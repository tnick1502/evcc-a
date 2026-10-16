"""Expanded property profiles for cohesive soils (nine and seven properties).

Physical properties use a CV limit of 0.15 and mechanical properties 0.30.
Physical and mechanical blocks receive equal total weight in the tie-breaker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .policy import Policy

# Canonical name -> accepted source columns (first found wins; values must agree).
ALIASES = {
    "rs": ("rs", "rho_s"),
    "rho": ("rho", "r"),
    "rd": ("rd", "rho_d"),
    "e": ("e", "e0"),
    "W": ("W", "water_pct"),
    "Ip": ("Ip", "ip", "plasticity"),
    "c": ("c", "cohesion"),
    "phi": ("phi", "fi", "j"),
    "modulus": ("modulus", "E50", "E"),
}
PHYSICAL = ("rs", "rho", "rd", "e", "W", "Ip")
MECHANICAL = ("c", "tan_phi", "modulus")
COHESIVE = frozenset({"CLAY", "LOAM", "SANDY_LOAM"})


@dataclass(frozen=True)
class PropertyRule:
    limit: float
    weight: float


def _numeric(column, name):
    if pd.api.types.is_numeric_dtype(column):
        out = pd.to_numeric(column, errors="coerce").astype(float)
    else:
        text = column.astype("string").str.strip().str.replace(",", ".", regex=False)
        out = pd.to_numeric(text, errors="coerce").astype(float)
        if bool((column.notna() & text.ne("").fillna(False) & out.isna()).any()):
            raise ValueError(f"Non-numeric values in {name!r}.")
    if bool(np.isinf(out.to_numpy()).any()):
        raise ValueError(f"Infinite values in {name!r}.")
    return out


def prepare(data, family, tolerance=1e-3):
    """Copy ``data`` and add the canonical columns of the expanded profile."""
    if family not in COHESIVE:
        raise ValueError(f"Expanded profiles are defined for cohesive soils, not {family!r}.")
    prepared = data.copy(deep=True)
    for canonical, aliases in ALIASES.items():
        merged = pd.Series(np.nan, index=data.index, dtype=float)
        found = [alias for alias in aliases if alias in data]
        for alias in found:
            candidate = _numeric(data[alias], alias)
            both = merged.notna() & candidate.notna()
            if bool((both & ~np.isclose(merged, candidate, rtol=tolerance, atol=1e-8)).any()):
                raise ValueError(f"Conflicting columns for {canonical!r}: {found}.")
            merged = merged.combine_first(candidate)
        if canonical not in ("phi", "c") and bool((merged.notna() & merged.le(0)).any()):
            raise ValueError(f"Non-positive {canonical!r} value.")
        if canonical == "c" and bool((merged.dropna() < 0).any()):
            raise ValueError("Negative cohesion.")
        prepared[canonical] = merged
    angle = prepared["phi"]
    prepared["tan_phi"] = np.tan(np.deg2rad(angle.where(angle.gt(0) & angle.lt(90))))
    return prepared


def nine_property_rules():
    rules = {name: PropertyRule(0.15, 1 / len(PHYSICAL)) for name in PHYSICAL}
    rules.update({name: PropertyRule(0.30, 1 / len(MECHANICAL)) for name in MECHANICAL})
    return rules


def profile_policy(base: Policy, profile="nine"):
    """``base`` with the nine-property rules, or seven without particle and dry density."""
    rules = nine_property_rules()
    if profile == "seven":
        rules = {k: v for k, v in rules.items() if k not in ("rs", "rd")}
    elif profile != "nine":
        raise ValueError(profile)
    return replace(
        base,
        properties=tuple(rules),
        limits=tuple(r.limit for r in rules.values()),
        weights=tuple(r.weight for r in rules.values()),
    )
