# The model

## Admissible groups

A context is a set of specimens of one material from one site. A group `G` of `m` specimens is admissible when

1. `m >= 6`;
2. for every property `j`, `CV_j = s_j / mean_j <= tau_j`, with the sample standard deviation `s_j`;
3. for every property, the extremum statistic `nu_j = max |x_ij - mean_j| / s_j <= nu_crit(m)`;
4. in every borehole the group occupies one contiguous depth interval of the context's specimens, or none;
5. no two members are separated by a logged layer of different material (`barrier_with_ids`);
6. all members belong to the same `fixed_geological_unit`, if that column is given.

The article uses `tau = 0.15` for physical and `0.30` for mechanical properties, and `nu_crit` from GOST 20522-2012, Table E.1 as amended in 2024 (`evcca.policy.NU_TABLE_AMENDMENT_1`). Both are inputs of `Policy` and can be replaced.

Reserves tighten the limits: `CV_j <= tau_j (1 - delta)` and `nu_j <= nu_crit(m) (1 - eta)`.

## Candidate family

Rule 4 means a group is fixed by choosing one interval (or none) in each borehole. A borehole with `m_h` specimens offers `1 + m_h (m_h + 1) / 2` choices, so the family is enumerated completely by screening the product of these choices (`exact_geometric_columns`). Contexts with at most 18 specimens are screened subset by subset instead (`exact_columns`). Above a budget of 500,000 interval combinations, the solver falls back to the groups found by the ordered comparator and does not mark the result optimal.

## Objectives

Every specimen is either in exactly one selected group or referred for review. With `A` referred specimens, `K` groups, `D` changed specimen pairs relative to the source interpretation and `B` a dispersion tie-breaker:

- compact mode minimizes `(A, K, B)` lexicographically;
- repair mode minimizes `(A, D, K, B)`.

`D` counts the pairs that are grouped together in one interpretation but not in the other; a referred specimen is grouped with nobody. It is linear in the selected groups, so each stage is a mixed-integer program solved by HiGHS to a zero gap and fixed before the next stage. For contexts with at most 18 specimens an independent exact-cover search (`exact_oracle`) gives the same optimum.

A context is *strictly feasible* when at least one group is selected and `A <= max(1, floor(0.1 n))`.

## Ordered comparator

The comparator applies the same gates and objectives, but a group must be a span of a fixed order of the specimens (each property with every tie order, principal components of the log properties, depth, and source unit then depth). Dynamic programming finds the best segmentation of each order. Every ordered solution is also feasible for the complete model, so EVCC-A is never worse; the difference measures the cost of the order restriction.

## Revision plans

The source units and the selected groups form a bipartite graph. Each connected component is classified as preservation, split, merge, partial transfer (two units, two groups, three links) or rebuild. Referred specimens form a separate review operation. All plans that tie the optimum on the criteria before `B` are enumerated.

For a referred specimen, `gate_diagnoses` reports whether it belongs to an admissible group that lost in the global cover, or which single gate (or combination of gates) prevents any admissible group from containing it.
