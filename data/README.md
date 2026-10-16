# Data

| File | Content |
|---|---|
| `processed/specimens.csv` | Active cohort: 247 specimens in 19 contexts |
| `processed/all_specimens.csv` | All 338 analysed specimens, including the excluded export projects P11, P12 and P16 |
| `processed/cohort_registry.csv` | Contexts with their size, boreholes and status |
| `processed/primary_logs.csv` | Borehole logs of the four primary-statement sites (1,014 records, including incomplete ones) |
| `raw/fi_clay/Database_FI_CLAY_14_856_version2.xlsx` | FI-CLAY database, version 2, unchanged |

## Sources

- `FI-CLAY`: five Finnish clay sites from the FI-CLAY database (Löfman and Korkiala-Tanttu, Zenodo release 3.0.0, [10.5281/zenodo.5040083](https://doi.org/10.5281/zenodo.5040083); method paper [10.1080/17499518.2020.1864410](https://doi.org/10.1080/17499518.2020.1864410)). A record is used when the test is an ordinary vertical incrementally loaded oedometer test (`ILOT`) with measured compression index, clay is the principal soil type (code ending in `Sa`), the sample depth interval is consistent and void ratio, compression index, unit weight, water content and fall-cone strength are positive. A site is used when at least eight such records come from at least two boreholes, the eligibility rule of every context; five of the 33 sites qualify (`evcca.study.ficlay.site_counts`). This leaves 95 of the 106 complete ILOT records at the five sites; the other eleven are clayey silt or gyttja and mixed codes (`saSi`, `saLj`, `Sa/Si`) and Tolsa record 503, whose upper depth lies below its lower depth. `evcca.study.ficlay.select` derives these rows from the workbook, and `evcca-verify` checks them.
- `PRIMARY`: laboratory statements of four projects of JSC MOSTDORGEOTREST.
- `EXPORT`: a triaxial-test export of the same company (six projects; P11, P12 and P16 are excluded from the main comparison because their strength fields are not comparable).

Project identifiers are replaced by codes: sites `Pxx`, boreholes `Bxxx`, source units `Uxxx`, specimens `Pxx-Sxxxx`. Codes keep the original sort order within a context. FI-CLAY sites and records keep their database names.

## Columns

| Column | Meaning |
|---|---|
| `sample_id`, `context_id`, `dataset`, `site_id` | Identifiers; a context is one site, material family and origin (and confining pressure for the export) |
| `borehole_id`, `depth_m` | Borehole and sampling depth, m |
| `soil_family`, `genesis` | Material family and origin code from the source |
| `initial_ige`, `initial_ige_known` | Source interpretation unit (FI-CLAY: soil-type code); for the export, whether the IGE label itself was recorded |
| `barrier_with_ids` | Specimens that may not share a group with this one |
| `rho_s`, `rho`, `rho_d` | Particle, bulk and dry density, g/cm³ |
| `e0` | Void ratio |
| `water_pct`, `porosity`, `plasticity`, `organic_pct`, `liquid_limit`, `clay_pct` | Percent |
| `saturation`, `liquidity` | Degree of saturation, liquidity index |
| `cohesion` | Reported cohesion, MPa |
| `phi`, `fi` | Reported friction angle, degrees (`fi` in the export) |
| `modulus` | Reported deformation modulus E (statements) or E50 (export), MPa |
| `qf`, `pressure_kpa` | Export deviator stress at failure and confining pressure, kPa |
| `alpha` | FI-CLAY `Cc / (1 + e0)` |
| `cc` | FI-CLAY compression index (measured) |
| `unit_weight` | FI-CLAY unit weight, kN/m³; `rho = unit_weight / 9.80665` |
| `su` | FI-CLAY fall-cone undrained shear strength, kPa |
| `preconsolidation_kpa`, `effective_stress_kpa` | FI-CLAY, kPa |
| `fixed_geological_unit` | FI-CLAY unit that a group may not cross (the dry crust at Haarajoki) |
| `test_method` | FI-CLAY test type |

Empty fields are missing values; nothing is imputed.

## Barriers

For the primary statements, `barrier_with_ids` follows from `primary_logs.csv`: two records of the same family in one borehole are separated when a record of another family (other than `OTHER`) lies between them. `evcca.study.barriers.primary_barriers` reproduces every list, and `evcca-verify` checks this. Export barriers were derived the same way from the full export logs, which also take the origin code into account; those logs are not included.

## License

FI-CLAY: CC BY 4.0, Löfman and Korkiala-Tanttu. Anonymized project data and result tables: CC BY 4.0, see [LICENSE](LICENSE).
