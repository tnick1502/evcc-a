# Graph and spectral comparisons

Graph Ward and Spectral + COP generate candidate groups, then use the same gates and lexicographic master as EVCC-A. The comparison covers all 19 four-property contexts at nominal thresholds, a 5% extremum reserve, and a joint 10% CV / 5% extremum reserve. Repair is also evaluated for the 11 primary contexts.

## Candidate construction

Graph Ward uses the Ward sum-of-squares merger cost on standardized raw and log properties. Symmetric feature-neighbour graphs use 3, 5, 8 or all neighbours, augmented by consecutive depth neighbours within each borehole. A merger requires graph contact, one interval per borehole, and no material-barrier or fixed-unit conflict. The method retains every node of each forest. Small or numerically inadmissible intermediate nodes can grow; final groups must pass all gates.

Spectral + COP uses 3, 5 or all feature neighbours and the same borehole edges. Known cannot-link edges are removed. With squared edge distance d² and q the median nonzero squared edge distance, affinity is exp(-d²/(2 f² q)), for f = 0.5, 1 or 2. The K largest eigenvectors of the symmetric normalized affinity are row-normalized. K ranges from 1 to floor(n/6). Nearest permitted centroid assignment traverses each borehole in depth order, closes a group after leaving its interval, and enforces all cannot-links. Unassignable specimens go to review. Centroids are updated for at most 30 iterations.

Each of 20 runs covers the full representation/graph/bandwidth grid. Its seed is 20261016 + run + the integer represented by the first seven hexadecimal digits of SHA-256(context_id). No optimal labels or complete candidate family are supplied.

These constructions adapt [Ward (1963)](https://doi.org/10.1080/01621459.1963.10500845), [Ng, Jordan and Weiss (2001)](https://proceedings.neurips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html) and [Wagstaff et al. (2001)](https://www.wkiri.com/research/papers/wagstaff-kmeans-01.pdf). COP's review state and borehole rule are additions to the published algorithm. The feature graph expresses similarity, not measured lateral geological adjacency.

## Common selection

Both methods receive the whole context, each source unit and every pairwise union of source units. A source-units-only control records the contribution of these seeds. Greedy endpoint deletion adds candidates until the six-specimen minimum is reached. Its score sums positive CV/limit and extremum/limit excesses, adds 10 per geological conflict or missing specimen below the minimum, and uses small support/dispersion tie-breakers. Candidates are then screened independently against the original physical properties. Log transforms affect proposals only.

Compact selection minimizes (A,K,B); repair minimizes (A,D,K,B). The same master optimizes each pool. Its optimum is certified for that pool only, unless it reaches an absolute lower bound (one group and no referrals in compact mode, no changed pairs in repair mode). Failure to generate a group does not prove that the complete family is empty. All selected assignments are re-audited, and the complete reference objectives are independently recomputed after candidate generation.

## Reproduce and verify

```bash
python -m evcca.study.comparators --output runs/graph-spectral
python -m evcca.study.comparators --verify runs/graph-spectral
```

The output path must be new. The saved evidence is in `results/comparators/`, with a separate checksum manifest. `evcca-verify` checks both result sets when the comparison folder is present.

| File | Contents |
|---|---|
| `results.csv` | 1,980 graph, spectral and source-units-only outcomes |
| `assignments.csv` | Every specimen assignment, including review |
| `group_audit.csv` | Selected-group admissibility and normalized gate statistics |
| `proposal_diagnostics.csv` | Intermediate proposals and failed-gate counts before final selection |
| `compared_results.csv` | Outcomes joined to EVCC-A and the ordered reference, with objective matches |
| `complete_reference_check.csv` | Fresh complete-family solves for all 90 context/policy/mode cases |
| `cohort_runs.csv`, `summary.csv` | Per-run totals and observed ranges |
| `provenance.json`, `manifest.json` | Protocol, seeds, dependencies and checksums |

`python -m evcca.study.robustness` adds three checks of this comparison (saved in `results/robustness/`): the totals under review budgets from none to unlimited, the Graph Ward pool joined with all 20 spectral pools, and both methods over a wider grid (Graph Ward with 2, 3, 4, 5, 6, 8, 10, 12 or all neighbours, Spectral + COP with 2, 3, 4, 5, 6, 8, 10 or all neighbours, bandwidth factors 0.25, 0.5, 1, 1.5, 2 and 4, up to 100 assignment iterations, 100 seeds).

Full-objective matching uses absolute B tolerance 1e-9. Compact leading matches compare (A,K), and repair leading matches compare (A,D,K). Seed ranges describe search variability on fixed data. They are not confidence intervals or independent site replications. Timing fields are descriptive because preprocessing and audit caches are shared.
