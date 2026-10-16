"""Diagnostics reported in the article and its supplement."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr, theilslopes

from .. import certificate
from ..gates import group_audit, mask_members
from ..ordered import ordered_dp
from ..partition import exact_oracle
from ..solve import solve_context, summarize
from .cohort import article_policy, make_contexts, source_unit_audit
from .paths import DATA

TOLSA = "FI::Tolsa, Kirkkonummi | CLAY | ILOT"
TOLSA_PROPERTIES = ["alpha", "rho", "water_pct", "su"]


def hull_certificate(data, columns):
    """Convex-hull test for every full two-group cover of the Tolsa context.

    For raw and log-transformed properties, a linear program looks for
    strictly positive weights that give both groups the same convex
    combination. If none exists, the separating direction is used to show
    that an order along it recovers the partition.
    """
    n = len(data)
    full = (1 << n) - 1
    pairs = [(m, full ^ m) for m in columns if m < (full ^ m) and (full ^ m) in columns]
    rows = []
    witnesses = []
    raw = data[TOLSA_PROPERTIES].to_numpy(float)
    for number, (left, right) in enumerate(pairs):
        a = mask_members(left, n)
        b = mask_members(right, n)
        for transform in ("raw", "log"):
            x = np.log(raw) if transform == "log" else raw.copy()
            scale = x.std(axis=0)
            offset = x.mean(axis=0)
            x = (x - offset) / scale
            equalities = np.vstack(
                [
                    np.c_[x[a].T, -x[b].T],
                    np.r_[np.ones(len(a)), np.zeros(len(b))],
                    np.r_[np.zeros(len(a)), np.ones(len(b))],
                ]
            )
            target = np.r_[np.zeros(4), 1.0, 1.0]
            # Maximize the smallest weight t subject to w_i >= t.
            result = linprog(
                np.r_[np.zeros(n), -1],
                A_ub=np.c_[-np.eye(n), np.ones(n)],
                b_ub=np.zeros(n),
                A_eq=np.c_[equalities, np.zeros(6)],
                b_eq=target,
                bounds=[(0, None)] * (n + 1),
                method="highs",
            )
            base = dict(
                partition=number,
                transform=transform,
                group_1_n=len(a),
                group_2_n=len(b),
                B=columns[left] + columns[right],
            )
            if not result.success:
                signs = np.r_[np.ones(len(a)), -np.ones(len(b))]
                design = np.c_[x[np.r_[a, b]], -np.ones(n)] * signs[:, None]
                separator = linprog(
                    np.zeros(5), A_ub=design, b_ub=-np.ones(n), bounds=[(None, None)] * 5, method="highs"
                )
                if not separator.success:
                    raise RuntimeError("Neither a common combination nor a separator was found.")
                order = np.argsort(x @ separator.x[:-1], kind="stable")
                recovered = ordered_dp(data, article_policy("FI-CLAY"), columns, order, "separator_" + transform)
                rows.append(
                    dict(
                        base,
                        hulls_intersect=False,
                        positive_all_weights=False,
                        minimum_weight=0,
                        affine_rank=int(np.linalg.matrix_rank(x - x[0])),
                        residual=None,
                        no_nonconstant_weak_linear_separator=False,
                        separator_order_A=int(recovered.objective[0]),
                        separator_order_K=int(recovered.objective[1]),
                    )
                )
                witnesses.append(
                    dict(
                        partition=number,
                        transform=transform,
                        separator=separator.x.tolist(),
                        offset=offset.tolist(),
                        scale=scale.tolist(),
                        recovered_objective=list(recovered.objective),
                    )
                )
                continue
            w = result.x[:-1]
            residual = float(np.max(np.abs(equalities @ w - target)))
            rank = int(np.linalg.matrix_rank(x - x[0]))
            positive = bool(w.min() > 1e-9)
            rows.append(
                dict(
                    base,
                    hulls_intersect=True,
                    positive_all_weights=positive,
                    minimum_weight=float(w.min()),
                    affine_rank=rank,
                    residual=residual,
                    no_nonconstant_weak_linear_separator=positive and rank == 4,
                )
            )
            record = dict(
                partition=number,
                transform=transform,
                group_1_sample_ids=data.iloc[a].sample_id.tolist(),
                group_2_sample_ids=data.iloc[b].sample_id.tolist(),
                weights_1=w[: len(a)].tolist(),
                weights_2=w[len(a) :].tolist(),
                property_order=TOLSA_PROPERTIES,
                offset=offset.tolist(),
                scale=scale.tolist(),
                residual=residual,
            )
            if transform == "raw":
                record["exact_decimal_certificate"] = certificate.construct(raw[a], raw[b], w)
            witnesses.append(record)
    return pd.DataFrame(rows), dict(two_group_full_coverage_partitions=len(pairs), witnesses=witnesses)


def void_ratio_density(contexts):
    """CV(e0) / CV(rho) for source units with at least two records."""
    rows = []
    for context_id, data, _ in contexts:
        if data.dataset.iloc[0] != "PRIMARY":
            continue
        for label, group in data.groupby("initial_ige"):
            if len(group) < 2:
                continue
            cv = group[["e0", "rho"]].std(ddof=1) / group[["e0", "rho"]].mean()
            if cv.rho > 0:
                rows.append(
                    dict(
                        context_id=context_id,
                        source_group=str(label),
                        n=len(group),
                        cv_e=float(cv.e0),
                        cv_rho=float(cv.rho),
                        ratio=float(cv.e0 / cv.rho),
                    )
                )
    return pd.DataFrame(rows)


def strength_signature(all_specimens):
    """Export records with phi = 0 and c [MPa] = q_f [kPa] / 2000."""
    export = all_specimens[all_specimens.dataset.eq("EXPORT")].copy()
    export["zero_angle_half_deviator"] = export.fi.eq(0) & np.isclose(
        2000 * export.cohesion, export.qf, atol=1.1, rtol=0
    )
    table = (
        export.groupby(["site_id", "context_id"])
        .agg(
            n=("sample_id", "size"),
            zero_angle=("fi", lambda s: int(s.eq(0).sum())),
            signature=("zero_angle_half_deviator", "sum"),
        )
        .reset_index()
    )
    return table, export


def row_signature_sensitivity(export, excluded_sites):
    """Excluded export contexts after removing only the signature rows."""
    remaining = export[~export.zero_angle_half_deviator].copy()
    rows = []
    for context_id, data, policy in make_contexts(remaining):
        if data.site_id.iloc[0] not in excluded_sites or len(data) < 8 or data.borehole_id.nunique() < 2:
            continue
        result, _ = solve_context(data, policy)
        for method in ("ordered", "evcca"):
            row, _ = summarize(data, result[method], policy)
            row.update(context_id=context_id, method=method)
            rows.append(row)
    return pd.DataFrame(rows)


def haarajoki_crust(active):
    """Records in the upper 2 m at Haarajoki and their completeness."""
    raw = pd.read_excel(DATA / "raw/fi_clay/Database_FI_CLAY_14_856_version2.xlsx")
    depth = pd.to_numeric(raw["Depth, average (m)"], errors="coerce")
    crust = raw[raw.Site.eq("Haarajoki") & depth.le(2)].copy()
    required = [
        "Void ratio",
        "Compression index",
        "Unit weight (kN/m3)",
        "Water content (%)",
        "Undrained shear stregth (fall cone) (kPa)",
    ]
    measured = crust[required].apply(pd.to_numeric, errors="coerce")
    complete = measured.gt(0).all(axis=1)
    return dict(
        source_database_crust_records=len(crust),
        ordinary_ilot_records=int(crust["Testb "].astype(str).str.strip().eq("ILOT").sum()),
        complete_positive_property_records=int(complete.sum()),
        incomplete_property_records=int((~complete).sum()),
        active_eligible_crust_records=int((active.site_id.eq("Haarajoki") & active.depth_m.le(2)).sum()),
        missing_property_counts=measured.isna().sum().to_dict(),
    )


def depth_trend(data, labels):
    """Water content against depth in each Tolsa group: rank correlation and Theil-Sen slope."""
    rows = []
    for group in sorted(set(labels) - {-1}):
        part = data[labels == group]
        rows.append(
            dict(
                group=group,
                n=len(part),
                boreholes=part.borehole_id.nunique(),
                depth_min=part.depth_m.min(),
                depth_max=part.depth_m.max(),
                water_mean=part.water_pct.mean(),
                water_cv=part.water_pct.std(ddof=1) / part.water_pct.mean(),
                depth_water_spearman=spearmanr(part.depth_m, part.water_pct)[0],
                theil_sen_slope_pct_per_m=theilslopes(part.water_pct, part.depth_m)[0],
            )
        )
    return pd.DataFrame(rows)


def block_resampling(data, labels, policy):
    """Exact borehole-block resampling of the CV in each group.

    Every ordered draw of boreholes with replacement is enumerated, so the
    percentiles describe a finite conditional distribution.
    """
    rows = []
    for label in sorted(set(labels) - {-1}):
        group = data[labels == label]
        holes = sorted(group.borehole_id.unique())
        draws = []
        for selected in product(holes, repeat=len(holes)):
            x = np.concatenate(
                [group.loc[group.borehole_id.eq(h), list(policy.properties)].to_numpy(float) for h in selected]
            )
            draws.append(x.std(axis=0, ddof=1) / x.mean(axis=0))
        draws = np.asarray(draws)
        point = group[list(policy.properties)].std(ddof=1) / group[list(policy.properties)].mean()
        for j, name in enumerate(policy.properties):
            rows.append(
                dict(
                    group=int(label),
                    property=name,
                    n=len(group),
                    boreholes=len(holes),
                    equiprobable_ordered_draws=len(draws),
                    cv=float(point.iloc[j]),
                    p05=float(np.quantile(draws[:, j], 0.05)),
                    p95=float(np.quantile(draws[:, j], 0.95)),
                    exceedance_fraction=float(np.mean(draws[:, j] > policy.cv_limits()[j])),
                    limit=float(policy.cv_limits()[j]),
                )
            )
    return pd.DataFrame(rows)


def profile_comparison(specimens):
    """Four-, seven- and nine-property screens on the primary statements."""
    rows = []
    for profile in ("four", "seven", "nine"):
        for context_id, data, policy in make_contexts(specimens, profile):
            if data.dataset.iloc[0] != "PRIMARY":
                continue
            result, _ = solve_context(data, policy)
            summary, _ = summarize(data, result["evcca"], policy)
            units = source_unit_audit([(context_id, data, policy)])
            summary.update(
                context_id=context_id,
                profile=profile,
                source_units=len(units),
                source_units_passing=int(units.compliant.sum()),
            )
            rows.append(summary)
    return pd.DataFrame(rows)


def tolsa_ablations(data):
    """Tolsa with one gate or one property removed, and with 32 random orders."""
    base = article_policy("FI-CLAY")
    variants = [
        ("all_gates", base),
        ("no_extremum_gate", replace(base, nu_gate=False)),
        ("no_borehole_contiguity", replace(base, borehole_contiguity=False)),
    ]
    for j, name in enumerate(base.properties):
        keep = [k for k in range(len(base.properties)) if k != j]
        variants.append(
            (
                "drop_" + name,
                replace(
                    base, properties=tuple(base.properties[k] for k in keep), limits=tuple(base.limits[k] for k in keep)
                ),
            )
        )
    variants.append(("32_random_directions", replace(base, random_orders=32)))
    rows = []
    for variant, policy in variants:
        result, _ = solve_context(data, policy)
        whole = group_audit(data, np.arange(len(data)), policy)["compliant"]
        for method in ("ordered", "evcca"):
            row, _ = summarize(data, result[method], policy)
            # A = 0 with K = 2 is optimal whenever one group is not admissible.
            at_bound = row["A"] == 0 and row["K"] == (1 if whole else 2)
            row.update(variant=variant, method=method, optimal_or_at_bound=bool(row["optimal"] or at_bound))
            rows.append(row)
    return pd.DataFrame(rows)


def return_experiment(contexts, solutions):
    """Delete a borehole, refit, then return its specimens by source unit and depth.

    A held-out specimen joins the group of the nearest-depth training specimen
    with the same source unit (and fixed unit, if any); its own measurements
    are not used. The augmented groups are then audited.
    """
    rows = []
    for context_id, data, policy in contexts:
        modes = ["compact", "repair"] if data.dataset.iloc[0] == "PRIMARY" else ["compact"]
        units = data.get("fixed_geological_unit", pd.Series(index=data.index, dtype=object)).fillna("").to_numpy()
        for mode in modes:
            reference = solutions[context_id, mode]
            summary, _ = summarize(data, reference, policy)
            if not summary["strict_feasible"] or summary["K"] < 2:
                continue
            for hole in sorted(data.borehole_id.unique()):
                keep = ~data.borehole_id.eq(hole).to_numpy()
                train = data[keep].reset_index(drop=True)
                test = data[~keep]
                if len(train) < policy.min_size:
                    continue
                result, _ = solve_context(train, replace(policy, seconds=8), mode)
                for method in ("ordered", "evcca"):
                    partition = result[method]
                    row, _ = summarize(train, partition, policy)
                    labels = []
                    for i, sample in test.iterrows():
                        candidates = np.flatnonzero(
                            (train.initial_ige.to_numpy() == sample.initial_ige)
                            & (units[keep] == units[i])
                            & (partition.labels >= 0)
                        )
                        if not row["strict_feasible"] or not len(candidates):
                            labels.append(-1)
                            continue
                        nearest = min(
                            candidates,
                            key=lambda k: (abs(train.iloc[k].depth_m - sample.depth_m), train.iloc[k].sample_id),
                        )
                        labels.append(partition.labels[nearest])
                    combined = pd.concat([train, test], ignore_index=True)
                    combined_labels = np.r_[partition.labels, np.array(labels)]
                    violating = sum(
                        not group_audit(combined, np.flatnonzero(combined_labels == label), policy)["compliant"]
                        for label in set(labels) - {-1}
                    )
                    rows.append(
                        dict(
                            context_id=context_id,
                            mode=mode,
                            method=method,
                            excluded_borehole=str(hole),
                            train_strict=row["strict_feasible"],
                            test_n=len(test),
                            assigned_new_n=int(np.sum(np.array(labels) >= 0)),
                            augmented_groups=len(set(labels) - {-1}),
                            augmented_violating_groups=violating,
                        )
                    )
    return pd.DataFrame(rows)


def oracle_check(data, policy, columns, mode, solution):
    """Compare a master-problem optimum with the exact-cover oracle."""
    oracle = exact_oracle(data, policy, columns, mode)
    leading_equal = tuple(oracle.objective[:-1]) == tuple(int(v) for v in solution.objective[:-1])
    b_equal = bool(np.isclose(oracle.objective[-1], solution.objective[-1], atol=1e-9, rtol=0))
    return dict(
        mode=mode,
        n=len(data),
        oracle_objective=str(tuple(oracle.objective)),
        evcca_objective=str(tuple(solution.objective)),
        matches=bool(leading_equal and b_equal),
    )


def source_units(contexts):
    primary = [(c, d, p) for c, d, p in contexts if d.dataset.iloc[0] == "PRIMARY"]
    return source_unit_audit(primary)


def pairs_between(labels, bands):
    """Number of co-assigned pairs whose members lie in different bands."""
    return sum(
        bands[i] != bands[j] and labels[i] >= 0 and labels[i] == labels[j]
        for i, j in combinations(range(len(labels)), 2)
    )
