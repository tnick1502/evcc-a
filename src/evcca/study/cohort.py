"""The study cohort and the policies applied to it."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .. import profiles
from ..gates import group_audit
from ..policy import Policy
from .paths import DATA

# Export projects whose strength fields are not documented well enough to be
# pooled with the other cohesion values (see the article, section 3).
EXCLUDED_SITES = frozenset({"P11", "P12", "P16"})


def article_policy(dataset, delta=0.0, eta=0.0):
    """Four-property policy used in the article for a data source."""
    settings = dict(nu_table_version="amendment_1_2024", nu_sample_sd=True, cv_margin_delta=delta, nu_margin_eta=eta)
    if dataset == "FI-CLAY":
        settings.update(properties=("alpha", "rho", "water_pct", "su"), limits=(0.30, 0.15, 0.15, 0.30))
    elif dataset not in ("PRIMARY", "EXPORT"):
        raise ValueError(f"Unknown data source: {dataset}")
    return Policy(**settings)


def load_specimens(active=True):
    """All 338 analysed specimens, or the 247 of the active cohort."""
    specimens = pd.read_csv(DATA / "processed/all_specimens.csv", low_memory=False)
    if active:
        specimens = specimens[~specimens.site_id.isin(EXCLUDED_SITES)].copy()
    return specimens


def make_contexts(specimens, profile="four", delta=0.0, eta=0.0):
    """List of ``(context_id, data, policy)``.

    ``profile`` is ``"four"`` (all sources) or ``"seven"``/``"nine"`` (primary
    statements only).
    """
    contexts = []
    for context_id, raw in specimens.groupby("context_id", sort=True):
        data = raw.sort_values("sample_id").reset_index(drop=True)
        dataset = data.dataset.iloc[0]
        if profile == "four":
            policy = article_policy(dataset, delta, eta)
        else:
            if dataset != "PRIMARY":
                continue
            data = profiles.prepare(data, family=data.soil_family.iloc[0])
            policy = profiles.profile_policy(article_policy("PRIMARY", delta, eta), profile)
        contexts.append((context_id, data, policy))
    return contexts


def source_unit_audit(contexts):
    """Admissibility of every source unit (IGE) as recorded in the source."""
    rows = []
    for context_id, data, policy in contexts:
        for label, group in data.groupby("initial_ige"):
            audit = group_audit(data, group.index.to_numpy(), policy)
            with np.errstate(invalid="ignore"):  # single-specimen units have no SD
                nu_ratio = np.max(audit["nu"] / audit["nu_critical"])
            rows.append(
                dict(
                    context_id=context_id,
                    source_group=str(label),
                    n=len(group),
                    below_minimum=len(group) < policy.min_size,
                    compliant=audit["compliant"],
                    max_cv_ratio=float(np.max(audit["cv"] / policy.cv_limits())),
                    max_nu_ratio=float(nu_ratio),
                )
            )
    return pd.DataFrame(rows)


def with_reserves(policy, delta, eta):
    return replace(policy, cv_margin_delta=delta, nu_margin_eta=eta)
