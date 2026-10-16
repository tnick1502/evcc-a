"""Figures of the article, drawn from the files in ``results/``.

Usage: ``evcca-figures --output runs/figures``
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from .paths import DATA, RESULTS  # noqa: E402
from .repair import CONTEXT as P13_LOAM  # noqa: E402
from .repair import depth_bands  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 220,
    }
)
COLORS = ["#25628d", "#c36b25", "#507c51", "#8763a2"]
MARKERS = ["o", "s", "^"]


def _specimens():
    return pd.read_csv(DATA / "processed/specimens.csv", low_memory=False)


def _assignments():
    return pd.read_csv(RESULTS / "assignments.csv")


def _labels(data, assignments, method="evcca", mode="compact", profile="four"):
    chosen = assignments[
        assignments.context_id.eq(data.context_id.iloc[0])
        & assignments.profile.eq(profile)
        & assignments.method.eq(method)
        & assignments["mode"].eq(mode)
    ]
    return data.sample_id.map(chosen.set_index("sample_id").group).to_numpy(int)


def _save(fig, output, name, pdf=True):
    if pdf:
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def tolsa(output):
    """Fig. 1: Tolsa groupings, hull witness and water-content trend."""
    specimens = _specimens()
    assignments = _assignments()
    d = specimens[specimens.site_id.eq("Tolsa, Kirkkonummi")].sort_values("sample_id").reset_index(drop=True)
    joint = _labels(d, assignments)
    ordered = _labels(d, assignments, "ordered")
    holes = sorted(d.borehole_id.unique())
    fig = plt.figure(figsize=(8.6, 10.0), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.22])
    axes = np.array(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    legend_axes = [fig.add_subplot(grid[2, k]) for k in range(2)]
    for ax in legend_axes:
        ax.axis("off")

    panels = [(axes[0, 0], joint, "Joint: 24 / 2 groups", "a"), (axes[0, 1], ordered, "Ordered: 23 / 3 groups", "b")]
    for ax, labels, title, panel in panels:
        for i, hole in enumerate(holes):
            part_hole = d[d.borehole_id.eq(hole)]
            ax.plot([i, i], [part_hole.depth_m.min(), part_hole.depth_m.max()], color="#b8b8b8", lw=1, zorder=1)
            for group in sorted(set(labels[part_hole.index])):
                part = part_hole[labels[part_hole.index] == group]
                if group >= 0:
                    ax.plot(
                        [i, i],
                        [part.depth_m.min(), part.depth_m.max()],
                        color=COLORS[group],
                        lw=5,
                        alpha=0.32,
                        zorder=2,
                    )
                ax.scatter(
                    np.full(len(part), i),
                    part.depth_m,
                    s=40,
                    marker="x" if group < 0 else MARKERS[group],
                    color="#222" if group < 0 else COLORS[group],
                    zorder=3,
                )
        handles = [
            Line2D(
                [],
                [],
                marker="x" if g < 0 else MARKERS[g],
                linestyle="",
                color="#222" if g < 0 else COLORS[g],
                label="Review" if g < 0 else f"Group {g + 1}",
            )
            for g in sorted(set(labels))
        ]
        ax.legend(handles=handles, frameon=False, fontsize=12, loc="upper right", ncol=1)
        ax.set_xticks(range(3), [h.split("::")[-1] for h in holes])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(6.3, 1.4)
        ax.set_xlabel("Borehole")
        ax.set_ylabel("Sampling depth (m)")
        ax.grid(axis="y", alpha=0.17)
        ax.set_title(f"({panel}) {title}", loc="left", pad=12)

    ax = axes[1, 0]
    for group in (0, 1):
        points = d.loc[joint == group, ["alpha", "water_pct"]].to_numpy()
        hull = ConvexHull(points)
        ax.add_patch(
            Polygon(
                points[hull.vertices],
                closed=True,
                facecolor=COLORS[group],
                alpha=0.13,
                edgecolor=COLORS[group],
                lw=1.3,
            )
        )
        ax.scatter(
            points[:, 0], points[:, 1], s=32, color=COLORS[group], marker=MARKERS[group], label=f"Group {group + 1}"
        )
    witness = json.loads((RESULTS / "tolsa_hull_witnesses.json").read_text())["witnesses"][0]
    exact = witness["exact_decimal_certificate"]
    points = np.array(exact["left_decimal_points"], float)
    weights = np.array([float(Fraction(v)) for v in exact["left_rational_weights"]])
    common = weights @ points
    ax.scatter(common[0], common[2], marker="*", s=160, color="#202020", label="Common convex combination", zorder=5)
    ax.set_xlabel(r"Compressibility $\alpha$ (-)")
    ax.set_ylabel("Water content (%)")
    ax.set_title("(c) Raw-property hull witness", loc="left", pad=12)
    handles, names = ax.get_legend_handles_labels()
    legend_axes[0].legend(handles, names, frameon=False, fontsize=12, loc="center")
    ax.grid(alpha=0.14)

    ax = axes[1, 1]
    for k, hole in enumerate(holes):
        for group in (0, 1):
            part = d[d.borehole_id.eq(hole) & (joint == group)]
            ax.scatter(part.water_pct, part.depth_m, s=34, color=COLORS[group], marker=MARKERS[k])
    ax.set_xlabel("Water content (%)")
    ax.set_ylabel("Sampling depth (m)")
    ax.set_ylim(6.3, 1.4)
    ax.set_title("(d) Retained water-content trend", loc="left", pad=12)
    ax.grid(alpha=0.14)
    trend = pd.read_csv(RESULTS / "tolsa_depth_trend.csv").iloc[0]
    ax.text(
        0.04,
        0.05,
        rf"Group 1: $r_s={trend.depth_water_spearman:.3f}$" + f"\nCV = {trend.water_cv:.3f} < 0.15",
        transform=ax.transAxes,
        fontsize=12,
    )
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[g], label=f"Group {g + 1}") for g in (0, 1)]
    handles += [
        Line2D([], [], marker=MARKERS[k], linestyle="", color="#666", label=h.split("::")[-1])
        for k, h in enumerate(holes)
    ]
    legend_axes[1].legend(
        handles=handles,
        title="Colour: group; shape: borehole",
        frameon=False,
        fontsize=12,
        title_fontsize=11.5,
        loc="center",
        ncol=2,
    )
    _save(fig, output, "fig1_tolsa_geometry")


def revision_plan(output):
    """Fig. 2: logged clay in borehole B013 and the compact/repair groupings of P13 loam C."""
    specimens = _specimens()
    assignments = _assignments()
    logs = pd.read_csv(DATA / "processed/primary_logs.csv", low_memory=False)
    log = logs[logs.site_id.eq("P13") & logs.borehole_id.eq("B013") & logs.depth_m.between(40, 85)]
    log = log.sort_values("depth_m")
    d = specimens[specimens.context_id.eq(P13_LOAM)].sort_values("sample_id").reset_index(drop=True)
    repair = _labels(d, assignments, mode="repair")
    table = pd.crosstab(d.initial_ige, repair).sort_index()

    loam = log[log.soil_family.eq("LOAM")]
    clay = log[log.soil_family.eq("CLAY")]
    top, bottom = clay.depth_m.min(), clay.depth_m.max()
    above, below = loam.depth_m[loam.depth_m < top].min(), loam.depth_m[loam.depth_m > bottom].min()

    fig = plt.figure(figsize=(8.6, 9.3), layout="constrained")
    layout = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    ax = fig.add_subplot(layout[0, 0])
    ax.axhspan(top, bottom, color="#e0e0e0", alpha=0.65, zorder=0)
    ax.scatter(np.full(len(clay), 0.5), clay.depth_m, color=COLORS[2], marker="s", s=45, zorder=3)
    ax.scatter(np.full(len(loam), 0.5), loam.depth_m, color=COLORS[1], marker="o", s=45, zorder=3)
    for unit, part in loam.groupby("initial_ige"):
        ax.annotate("IGE " + unit, (0.5, part.depth_m.mean()), xytext=(8, -3), textcoords="offset points", fontsize=13)
    ax.add_patch(
        FancyArrowPatch((2.05, above), (2.05, below), arrowstyle="<->", mutation_scale=12, lw=1.7, color="#8d4639")
    )
    ax.scatter([2.05], [(top + bottom) / 2], marker="x", s=110, color="#8d4639", lw=2.2, zorder=5)
    ax.text(2.18, (bottom + below) / 2, "Union\nblocked", fontsize=12, va="center", color="#8d4639")
    ax.text(0.8, top + 3.2, f"Observed clay\n{top:.1f}–{bottom:.1f} m", fontsize=12, va="center", color="#444")
    ax.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="", color=COLORS[2], label="Clay"),
            Line2D([], [], marker="o", linestyle="", color=COLORS[1], label="Loam"),
        ],
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    ax.set_xlim(-0.05, 2.95)
    ax.set_ylim(84, 38)
    ax.set_xticks([])
    ax.set_xlabel("Borehole B013")
    ax.set_ylabel("Sampling depth (m)")
    ax.grid(axis="y", alpha=0.17)
    ax.set_title("(a) Project P13, BH B013", loc="left", pad=14)

    ax = fig.add_subplot(layout[0, 1])
    ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=6, aspect="auto")
    for i in range(len(table)):
        for j in range(len(table.columns)):
            value = int(table.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=18, color="white" if value >= 4 else "#222")
    ax.set_yticks(range(len(table)), ["IGE " + v for v in table.index])
    ax.set_xticks(range(len(table.columns)), [f"Revised {int(g) + 1}" for g in table.columns])
    ax.set_xlabel("Repair: A = 0, D = 39, K = 2")
    ax.set_title("(b) Two merges; 17 retained", loc="left", pad=14)

    bands, groups, _ = depth_bands(specimens, assignments)
    ax = fig.add_subplot(layout[1, :])
    positions = {("compact", 0): 0.5, ("compact", 1): 1.5, ("repair", 0): 3.5, ("repair", 1): 4.5}
    for (mode, group), position in positions.items():
        local = bands[bands[mode].eq(group)]
        for band, color, marker in (("upper", COLORS[0], "o"), ("lower", COLORS[1], "s")):
            part = local[local.depth_band.eq(band)].sort_values(["depth_m", "sample_id"])
            jitter = np.linspace(-0.12, 0.12, len(part)) if len(part) > 1 else np.zeros(len(part))
            ax.scatter(position + jitter, part.depth_m, color=color, marker=marker, s=38, zorder=3)
    upper = bands[bands.depth_band.eq("upper")].depth_m
    lower = bands[bands.depth_band.eq("lower")].depth_m
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=COLORS[0],
            label=f"Upper observations: {upper.min():g}–{upper.max():g} m",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            color=COLORS[1],
            label=f"Lower observations: {lower.min():g}–{lower.max():g} m",
        ),
    ]
    ax.legend(handles=handles, loc="center", ncol=2, frameon=True, facecolor="white", edgecolor="none", fontsize=12)
    ticks = []
    for mode, group in positions:
        count = int(groups[groups["mode"].eq(mode) & groups.group.eq(group)].n.iloc[0])
        ticks.append(f"{mode.capitalize()} {group + 1}\nn = {count}")
    ax.set_xticks(list(positions.values()), ticks)
    ax.axvline(2.5, color="#aaa", linestyle=":", lw=1)
    ax.set_xlim(0, 5)
    ax.set_ylim(88, 38)
    ax.set_ylabel("Sampling depth (m)")
    ax.grid(axis="y", alpha=0.17)
    ax.set_title("(c) Compact D = 59; repair D = 39 (17 retained, K = 2)", loc="left", pad=12)
    _save(fig, output, "fig2_revision_plan")


def reserve_map(output):
    """Fig. 3: within-budget specimens under CV and extremum reserves."""
    grid = pd.read_csv(RESULTS / "reserve_grid.csv").query("profile == 'four' and mode == 'compact'")
    totals = grid.groupby(["method", "cv_reserve", "nu_reserve"]).agg(
        accepted=("strict_accepted_n", "sum"), strict=("strict_feasible", "sum")
    )
    deltas, etas = [0, 0.05, 0.10], [0, 0.01, 0.02, 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), layout="constrained")
    for ax, method, title in zip(axes, ["evcca", "ordered"], ["(a) EVCC-A", "(b) Best tested orders"]):
        values = totals.loc[method]
        image = ax.imshow(
            values.accepted.unstack().loc[deltas, etas].to_numpy(), cmap="Blues", vmin=94, vmax=120, aspect="auto"
        )
        for i, delta in enumerate(deltas):
            for j, eta in enumerate(etas):
                row = values.loc[delta, eta]
                ax.text(
                    j,
                    i,
                    f"{int(row.accepted)}\n({int(row.strict)})",
                    ha="center",
                    va="center",
                    fontsize=13.5,
                    color="white" if row.accepted >= 110 else "#202020",
                )
        ax.set_xticks(np.arange(5) - 0.5, minor=True)
        ax.set_yticks(np.arange(4) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=0.8, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.add_patch(Rectangle((2.5, -0.5), 1, 1, fill=False, color="#bf6a22", lw=2.5))
        ax.set_xticks(range(4), [0, 1, 2, 5])
        ax.set_yticks(range(3), [0, 5, 10])
        ax.set_xlabel(r"Extremum reserve $\eta$ (%)")
        ax.set_ylabel(r"CV reserve $\delta$ (%)")
        ax.set_title(title, loc="left", pad=12)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.025)
    colorbar.set_label("Within-budget specimens")
    _save(fig, output, "fig3_reserve_map")


def graphical_abstract(output):
    fig, ax = plt.subplots(figsize=(11.8, 4.72))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 4.72)
    ax.axis("off")
    ax.text(
        0.2, 4.35, "From logged specimens to an inspectable revision plan", fontsize=18, weight="bold", color="#153d56"
    )
    for x, label in ((0.2, "1  Borehole evidence"), (4.1, "2  Admissible groups"), (8.3, "3  Revision plan")):
        ax.text(x, 3.9, label, fontsize=12, weight="bold")
    for j in range(3):
        x = 0.55 + j * 0.82
        ax.add_patch(Rectangle((x, 1.3), 0.33, 2.2, facecolor="#efe9d9", edgecolor="#777", lw=0.8))
        ax.add_patch(Rectangle((x, 1.3), 0.33, 0.95, facecolor="#d9e7df", edgecolor="none"))
        ax.text(x + 0.16, 1.06, f"BH {j + 1}", ha="center", fontsize=9)
        for k, y in enumerate([3.25, 2.9, 2.55, 2.12, 1.78, 1.45]):
            if (j, k) in [(0, 0), (2, 5)]:
                ax.scatter(x + 0.16, y, marker="x", s=65, lw=1.7, color="#222", zorder=3)
            else:
                ax.scatter(
                    x + 0.16, y, s=38, color=COLORS[0] if y > 2.3 else COLORS[1], edgecolors="white", lw=0.6, zorder=3
                )
    ax.plot([0.25, 3.0], [2.3, 2.3], ls="--", color="#8d4639", lw=1.7)
    ax.text(0.27, 2.42, "Logged material break", fontsize=9, color="#8d4639")
    for x0, x1 in [(3.05, 3.75), (7.3, 8.0)]:
        ax.add_patch(
            FancyArrowPatch((x0, 2.45), (x1, 2.45), arrowstyle="-|>", mutation_scale=20, color="#25628d", lw=1.6)
        )
    offsets = [(-0.27, 0.22), (0, 0.32), (0.27, 0.22), (0.32, 0), (0.27, -0.22), (0, -0.32), (-0.27, -0.22), (-0.32, 0)]
    for center, color in (((4.85, 2.95), COLORS[0]), ((6.3, 1.95), COLORS[1])):
        ax.add_patch(Circle(center, 0.58, fill=False, edgecolor=color, lw=1.5))
        for dx, dy in offsets:
            ax.scatter(center[0] + dx, center[1] + dy, color=color, s=38)
    ax.scatter([4.4, 4.75], [1.32, 1.32], marker="x", s=60, color="#222")
    ax.text(5.0, 1.23, "Review referrals", fontsize=9)
    ax.text(4.05, 3.66, "CV + extremum + geology", fontsize=10)
    steps = [
        ("Preserve admissible units", "✓"),
        ("Merge or rebuild groups", "✓"),
        ("List alternative plans", "✓"),
        ("Review named records", "×"),
    ]
    for k, (label, icon) in enumerate(steps):
        y = 3.28 - k * 0.46
        ax.text(8.25, y, icon, fontsize=16, color=COLORS[2] if k < 3 else "#333", va="center")
        ax.text(8.65, y, label, fontsize=10.5, va="center")
    ax.add_patch(
        FancyBboxPatch(
            (8.25, 1.0), 3.18, 0.46, boxstyle="round,pad=0.06", facecolor="#e3eee6", edgecolor=COLORS[2], lw=1.2
        )
    )
    ax.text(9.84, 1.22, "Optimality certified", ha="center", va="center", fontsize=11, weight="bold", color="#315a3a")
    ax.text(
        0.2,
        0.5,
        "19 contexts · 247 specimens     |     5% tighter extremum limits: "
        "all specimens grouped in 6 contexts, 3 for tested heuristics",
        fontsize=11,
        weight="bold",
        color="#153d56",
    )
    ax.text(
        0.2,
        0.13,
        "Schematic workflow. Review is separate from deleting a measurement or inferring a layer boundary.",
        fontsize=8.5,
        color="#555",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(output / "graphical_abstract.pdf")
    fig.savefig(output / "graphical_abstract.png", dpi=150)
    plt.close(fig)


def graph_spectral(output):
    """Fig. 4: assigned and within-budget specimens for complete search and three proposal methods."""
    summary = pd.read_csv(RESULTS / "comparators/summary.csv")
    summary = summary[summary["mode"].eq("compact")]
    points = ["nominal", "nu_5pct", "cv_10pct_nu_5pct"]
    methods = ["evcca", "ordered", "graph_ward", "spectral_cop"]
    labels = ["EVCC-A", "Best tested order", "Graph Ward", "Spectral + COP"]
    colors = ["#26384f", "#92979d", "#95623b", "#377b99"]
    width = 0.19
    with plt.rc_context({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12}):
        fig, axes = plt.subplots(2, 1, figsize=(6.1, 6.6), sharey=True)
        titles = ["(a) Assigned specimens", "(b) Within-budget specimens"]
        for ax, measure, title in zip(axes, ["assigned", "accepted"], titles):
            for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
                rows = [summary[summary.method.eq(method) & summary.policy.eq(point)].iloc[0] for point in points]
                median = np.array([row[measure + "_median"] for row in rows])
                low = np.array([row[measure + "_min"] for row in rows])
                high = np.array([row[measure + "_max"] for row in rows])
                x = np.arange(3) + (j - 1.5) * width
                ax.bar(x, median, width=width, color=color, label=label, zorder=3)
                varies = high > low
                if varies.any():
                    ax.errorbar(
                        x[varies],
                        median[varies],
                        yerr=[median[varies] - low[varies], high[varies] - median[varies]],
                        fmt="none",
                        color="#222",
                        capsize=2,
                        zorder=4,
                    )
                for xx, value, upper in zip(x, median, high):
                    ax.text(xx, upper + 3, str(int(value)), ha="center", va="bottom", fontsize=10.5)
            ax.set_xticks(range(3), ["Nominal", "Extremum 5%", "CV 10% +\nextremum 5%"])
            ax.set_ylim(0, 210)
            ax.set_yticks([0, 50, 100, 150, 200])
            ax.set_title(title, loc="left")
            ax.set_ylabel("Specimens")
            ax.grid(axis="y", alpha=0.18, zorder=0)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=11)
        fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.17, hspace=0.49)
        _save(fig, output, "fig4_graph_spectral")


FIGURES = {
    "fig1_tolsa_geometry": tolsa,
    "fig2_revision_plan": revision_plan,
    "fig3_reserve_map": reserve_map,
    "fig4_graph_spectral": graph_spectral,
    "graphical_abstract": graphical_abstract,
}


def render(name, output):
    """Draw one figure into ``output`` and return the PNG path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    FIGURES[name](output)
    return output / f"{name}.png"


def main():
    parser = argparse.ArgumentParser(description="Draw the article figures.")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the figures")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    for draw in FIGURES.values():
        draw(output)
    print("figures written to", output)


if __name__ == "__main__":
    main()
