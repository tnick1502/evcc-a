# Results

Output of `evcca-reproduce` for the published version. `manifest.json` holds the SHA-256 of every file and `provenance.json` the software versions and the input checksum. `evcca-verify` checks both and recomputes the plan and certificate identities.

Throughout, `profile` is `four` (the article policy) or `nine` (primary statements with nine properties), `mode` is `compact` or `repair`, and `method` is `evcca` or `ordered` (the best tested order). `A`, `K`, `D` and `B` are the number of referred specimens, the number of groups, the changed specimen pairs and the dispersion tie-breaker. A context is strictly feasible when it has a group and `A` is within the review budget; `strict_accepted_n` is then the number of assigned specimens, otherwise 0.

| File | Content | Article |
|---|---|---|
| `metrics.csv` | One row per context, mode, method and profile at the nominal policy | Table 2, S2 |
| `reserve_grid.csv` | The same for every CV reserve (0, 5, 10 %) and extremum reserve (0, 1, 2, 5 %) | Fig. 3, S3 |
| `assignments.csv` | Group of every specimen at the nominal policy (`-1`: referred for review) | |
| `group_audit.csv` | CV, extremum statistic and depth trend of every selected group | |
| `scale_timing.csv` | Specimens, boreholes, `N_geom`, admissible groups and wall time per context | Table 6, S2 |
| `oracle_checks.csv` | Exact-cover oracle against the MILP optimum, contexts with at most 18 specimens | S2 |
| `borehole_deletions.csv` | Every borehole removed in turn and the context solved again | S3 |
| `return_experiment.csv` | Held-out specimens returned by source unit and nearest depth | S3 |
| `tolsa_projection_checks.csv` | Convex-hull test of the Tolsa two-group partition, raw and log properties | Fig. 1, S4 |
| `tolsa_hull_witnesses.json` | Hull weights, separator, and the exact rational certificate | Fig. 1, S4 |
| `tolsa_ablations.csv` | Tolsa with one gate or one property removed, and with 32 random directions | Section 4.2 |
| `tolsa_block_resampling.csv` | CV over all borehole-block resamples of each Tolsa group | S3 |
| `tolsa_depth_trend.csv` | Water content against depth in each Tolsa group: Spearman coefficient and Theil–Sen slope | Section 4.2, S6 |
| `revision_plan_summary.csv` | Revision plan of every primary context | Table 5, S5 |
| `revision_plans.json` | Operations of each plan and every alternative tied on the leading criteria | S5 |
| `review_reasons.csv` | Why each referred specimen was referred | S5 |
| `repair_depth_groups.csv`, `repair_depth_modes.csv` | P13 loam: upper and lower observations in the compact and repair groups | Fig. 2, S5 |
| `profile_comparison.csv` | Four-, seven- and nine-property screens of the primary contexts | Table 3 |
| `source_unit_support.csv` | Admissibility of every source unit as recorded | Table 3, S6 |
| `void_ratio_density.csv` | CV(e0) / CV(rho) in the source units | S6 |
| `haarajoki_crust.json` | Database records in the upper 2 m at Haarajoki | S6 |
| `strength_signature.csv`, `row_signature_sensitivity.csv` | Strength fields of the excluded export projects | S6 |
| `grubbs_comparison.csv` | Reference critical values against the two-sided 5 % Grubbs values | S1 |

Run times depend on the machine; `provenance.json` records the one used here.

`comparators/` holds the graph and spectral comparison (Fig. 4, S7) with its own checksum manifest; [docs/comparators.md](../docs/comparators.md) defines its files. It covers three four-property reserve settings.

`robustness/` is written by `python -m evcca.study.robustness` and has its own manifest:

| File | Content | Article |
|---|---|---|
| `review_budgets.csv` | Within-budget specimens and strictly feasible contexts of every compact result at the three comparison policies, for review budgets from none to unlimited | Table 4, S8 |
| `deletion_outcomes.csv` | Borehole deletions split by strict feasibility before the deletion, with the reason for each loss | Section 6, S3 |
| `wide_search.csv` | Joined Graph Ward and spectral pools, and both methods over a wider grid with 100 seeds | Section 4.6, S8 |
| `family_check.csv` | Interval-enumerated family against the screen of all subsets, every context, two policies | Section 2.4, S2 |
| `gate_failures.csv` | Interval candidates of the primary contexts rejected by each property gate | Section 3, S8 |

The first two tables are derived from the files above; `evcca-verify` recomputes them and checks the family comparison.
