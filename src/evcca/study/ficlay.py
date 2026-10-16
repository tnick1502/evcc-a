"""The FI-CLAY records used in the study, selected from the released workbook.

A record of FI-CLAY/14/856 version 2 is usable when

- the test is an ordinary vertical incrementally loaded oedometer test
  (``ILOT``) and the compression index was measured, not estimated;
- clay is the principal soil type: the code ends in ``Sa`` (``Sa``, ``laSa``,
  ``liSa``, ``ljSa``, ``kuSa``), so clayey silt or gyttja (``saSi``, ``saLj``)
  and mixed or uncertain codes (``Sa/Si``, ``Sa?``) are left out;
- the sample interval is consistent (upper depth not below lower depth); and
- void ratio, compression index, unit weight, water content and fall-cone
  undrained strength are all positive.

A site enters the study when its usable records number at least eight and come
from at least two boreholes, the eligibility rule of every context. Five of the
33 sites in the workbook qualify.

At Haarajoki the specimens in the upper 2 m form the dry crust, a fixed
geological unit separate from the clay below.
"""

from __future__ import annotations

import pandas as pd

from .paths import DATA

WORKBOOK = DATA / "raw/fi_clay/Database_FI_CLAY_14_856_version2.xlsx"
MIN_RECORDS, MIN_BOREHOLES = 8, 2
CRUST_SITE, CRUST_DEPTH_M = "Haarajoki", 2.0
GRAVITY = 9.80665

FIELDS = {
    "e0": "Void ratio",
    "cc": "Compression index",
    "unit_weight": "Unit weight (kN/m3)",
    "water_pct": "Water content (%)",
    "su": "Undrained shear stregth (fall cone) (kPa)",  # spelling as in the workbook
    "clay_pct": "Clay fraction (%)",
    "liquid_limit": "Liquid limit (%)",
    "effective_stress_kpa": "Effective in-situ stress (kPa)",
    "preconsolidation_kpa": "Pre-consolidation pressure (kPa)",
}
REQUIRED = ("e0", "cc", "unit_weight", "water_pct", "su")


def _usable(raw):
    values = pd.DataFrame({name: pd.to_numeric(raw[column], errors="coerce") for name, column in FIELDS.items()})
    upper = pd.to_numeric(raw["Depth, upper (m)"], errors="coerce")
    lower = pd.to_numeric(raw["Depth, lower (m)"], errors="coerce")
    soil = raw["Soil typea"].astype(str).str.strip()
    keep = (
        raw["Testb "].astype(str).str.strip().eq("ILOT")
        & ~raw["Compression index estimated?"].astype(str).str.strip().eq("Yes")
        & soil.str.endswith("Sa")
        & upper.le(lower)
        & values[list(REQUIRED)].gt(0).all(axis=1)
    )
    return keep, values, soil


def site_counts(workbook=WORKBOOK):
    """Usable records and boreholes of every site in the workbook, and whether the site is eligible."""
    raw = pd.read_excel(workbook)
    keep, _, _ = _usable(raw)
    usable = raw[keep]
    counts = pd.DataFrame(
        {
            "records": usable.groupby("Site").size(),
            "boreholes": usable.groupby("Site").Point.nunique(),
        }
    ).reindex(sorted(raw.Site.dropna().unique()), fill_value=0)
    counts["eligible"] = counts.records.ge(MIN_RECORDS) & counts.boreholes.ge(MIN_BOREHOLES)
    return counts.rename_axis("site").sort_values(["records", "boreholes"], ascending=False)


def select(workbook=WORKBOOK):
    """FI-CLAY specimens of the study, in the columns of ``all_specimens.csv``."""
    raw = pd.read_excel(workbook)
    keep, values, soil = _usable(raw)
    sites = site_counts(workbook)
    keep &= raw.Site.isin(sites.index[sites.eligible])
    raw, values, soil = raw[keep], values[keep], soil[keep]
    depth = pd.to_numeric(raw["Depth, average (m)"])
    site = raw.Site
    crust = site.eq(CRUST_SITE) & depth.le(CRUST_DEPTH_M)
    frame = pd.DataFrame(
        {
            "sample_id": "FI::" + raw["No."].astype(float).astype(str),
            "context_id": "FI::" + site + " | CLAY | ILOT",
            "dataset": "FI-CLAY",
            "site_id": site,
            "borehole_id": site + "::" + raw.Point.astype(str),
            "depth_m": depth,
            "soil_family": "CLAY",
            "initial_ige": soil,
            "rho": values.unit_weight / GRAVITY,
            "alpha": values.cc / (1 + values.e0),
            "fixed_geological_unit": "CLAY",
            "test_method": "ILOT",
        }
    )
    frame.loc[site.eq(CRUST_SITE), "fixed_geological_unit"] = "underlying_clay"
    frame.loc[crust, "fixed_geological_unit"] = "dry_crust"
    frame = pd.concat([frame, values], axis=1)
    return frame.sort_values("sample_id").reset_index(drop=True)
