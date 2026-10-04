#!/usr/bin/env python3
"""Plot the updated BAO+SN posterior compilation comparison.

Top: dark-energy chi-square improvements from the refined run-summary minima.
Bottom: weighted epsilon medians and 16th/84th posterior percentiles in
flat LambdaCDM+epsilon. This plotting script performs no new profiling or optimization.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sn_bao_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from src.sampler import compute_getdist_constraints
from src.posterior_io import load_run


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path(os.environ.get(
    "SN_BAO_POSTERIOR_ROOT",
    str(ROOT / "results/runs/baseline"),
))
OUTPUT = Path(os.environ.get(
    "SN_BAO_FIGURE_DIR", str(ROOT / "figures"),
)) / "figure1.pdf"
MODELS = ("lcdm", "w0wa", "lcdm-epsilon", "w0wa-epsilon")
SAMPLE_KEYS = ("pantheon", "dovekie", "union3", "union31")

SECTOSA = "#9671bd"
VIRGINIA = "#77b5b6"


@dataclass(frozen=True)
class Compilation:
    """One displayed supernova compilation."""

    label: str


COMPILATIONS = (
    Compilation("Pantheon+"),
    Compilation("DES-Dovekie"),
    Compilation("Union3"),
    Compilation("Union3.1"),
)



def collect_posterior_results(run_root: Path | None = None) -> tuple[tuple, dict]:
    """Calculate plotted values and provenance from refined summaries and weighted posterior draws."""
    run_root = RUN_ROOT if run_root is None else Path(run_root)
    intervals, fixed_epsilon, free_epsilon = [], [], []
    records = []
    for key, compilation in zip(SAMPLE_KEYS, COMPILATIONS, strict=True):
        runs = {model: load_run(run_root, key, model) for model in MODELS}
        epsilon = runs["lcdm-epsilon"]["epsilon"]
        fixed = runs["lcdm"]["minimum_chi2"] - runs["w0wa"]["minimum_chi2"]
        free = runs["lcdm-epsilon"]["minimum_chi2"] - runs["w0wa-epsilon"]["minimum_chi2"]
        intervals.append([epsilon["median"], epsilon["q16"], epsilon["q84"]])
        fixed_epsilon.append(fixed)
        free_epsilon.append(free)
        records.append({"sample": key, "label": compilation.label,
                        "epsilon": epsilon, "delta_chi2_fixed_epsilon": fixed,
                        "delta_chi2_free_epsilon": free, "runs": runs})
    positions = 0.75 * np.arange(len(COMPILATIONS), dtype=float)
    values = (np.asarray(intervals), np.asarray(fixed_epsilon), np.asarray(free_epsilon), positions)
    provenance = {
        "data_combination": "BAO+SN", "run_root": str(run_root.resolve()),
        "minimum_method": "Refined minimum_chi2 from summary.json; no new minimization",
        "epsilon_model": "lcdm-epsilon", "epsilon_quantile_probabilities": [0.16, 0.5, 0.84],
        "epsilon_quantile_method": "GetDist weighted equal-tail percentiles; ignore_rows=0",
        "compilations": records,
    }
    return values, provenance


def collect_results() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted epsilon intervals and refined chi-square differences."""
    return collect_posterior_results()[0]


def make_figure(
    intervals: np.ndarray,
    fixed_epsilon: np.ndarray,
    free_epsilon: np.ndarray,
    positions: np.ndarray,
) -> mpl.figure.Figure:
    """Build a compact single-column two-panel vector figure."""

    # Preserve the existing single-column dimensions.
    width_inch = 3.375
    height_inch = 4.1

    figure = plt.figure(figsize=(width_inch, height_inch))

    # ------------------------------------------------------------------
    # Stacked single-column composition.
    #
    # [left, bottom, width, height]
    #
    # A relatively large left margin is needed for the compilation names.
    # ------------------------------------------------------------------
    epsilon_ax = figure.add_axes([0.25, 0.10, 0.65, 0.36])
    delta_ax = figure.add_axes([0.25, 0.63, 0.65, 0.36])

    labels = [item.label for item in COMPILATIONS]

    # ==============================================================
    # Bottom panel: epsilon posterior medians and percentile intervals
    # ==============================================================

    medians, lower, upper = intervals.T

    epsilon_ax.errorbar(
        medians,
        positions,
        xerr=np.vstack((medians - lower, upper - medians)),
        fmt="o",
        color=SECTOSA,
        markersize=5,
        markeredgewidth=0.8,
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.0,
        zorder=3,
    )

    # epsilon = 0 reference line
    epsilon_ax.axvline(
        0.0,
        color="grey",
        linestyle=(0, (4, 2)),
        linewidth=0.75,
        zorder=1,
    )

    epsilon_ax.set_xlim(min(-0.083, float(np.min(lower)) - 0.005),
                        max(0.007, float(np.max(upper)) + 0.005))

    # Sparse ticks work better at one-column width.
    epsilon_ax.set_xticks([-0.08, -0.06, -0.04, -0.02, 0.00])

    epsilon_ax.set_yticks(
        positions,
        labels,
    )

    epsilon_ax.set_xlabel(
        r"$\epsilon_{\Lambda\mathrm{CDM}}$",
        fontsize=9,
    )

    # ==============================================================
    # Top panel: change in Delta chi^2_DE
    # ==============================================================

    # Arrows show the reduction when epsilon is freed:
    #
    # epsilon = 0  ----->  free epsilon
    #
    # Since the free-epsilon result is generally smaller, the arrows
    # point from right to left in this figure.
    for y, fixed, free in zip(
        positions,
        fixed_epsilon,
        free_epsilon,
        strict=True,
    ):
        arrow = FancyArrowPatch(
            (fixed, y),
            (free, y),
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=1.2,
            color="0.65",
            shrinkA=0,
            shrinkB=2,
            zorder=1,
        )
        delta_ax.add_patch(arrow)

    # epsilon = 0: right-hand points
    delta_ax.scatter(
        fixed_epsilon,
        positions,
        marker="o",
        s=20,
        color=VIRGINIA,
        label=r"$\epsilon=0$",
        zorder=3,
    )

    # Free epsilon: left-hand points
    delta_ax.scatter(
        free_epsilon,
        positions,
        marker="o",
        s=20,
        facecolors=SECTOSA,
        edgecolors=SECTOSA,
        linewidths=1.0,
        label=r"Free $\epsilon$",
        zorder=3,
    )

    # --------------------------------------------------------------
    # Numerical annotations
    # --------------------------------------------------------------

    # Free-epsilon values to the left of their markers
    for y, value in zip(
        positions,
        free_epsilon,
        strict=True,
    ):
        delta_ax.text(
            value - 0.25,
            y,
            f"{value:.2f}",
            va="center",
            ha="right",
            fontsize=8,
        )

    # epsilon=0 values to the right of their markers
    for y, value in zip(
        positions,
        fixed_epsilon,
        strict=True,
    ):
        delta_ax.text(
            value + 0.25,
            y,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    delta_ax.set_xlim(min(0.0, float(min(np.min(fixed_epsilon), np.min(free_epsilon))) - 1.0),
                      max(11.3, float(max(np.max(fixed_epsilon), np.max(free_epsilon))) + 1.0))

    # Sparse ticks complement the exact values printed by the points.
    delta_ax.set_xticks([0, 2.5, 5, 7.5, 10])

    delta_ax.set_yticks(
        positions,
        labels,
    )

    delta_ax.set_xlabel(
        r"$\Delta\chi_{\mathrm{DE}}^2$",
        fontsize=9,
    )

    # Compact legend in the whitespace between the panels.
    delta_ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.8, 0.7),
        frameon=True,
        fancybox=False,
        edgecolor='k',
        linewidth=0.5,
        fontsize=9,
        #ncol=2,
        handlelength=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
        columnspacing=1.2,
    )

    # ==============================================================
    # Shared formatting
    # ==============================================================

    # Slightly asymmetric padding so that the panels remain compact.
    top_pad = 0.27
    bottom_pad = 0.30

    for axis in (epsilon_ax, delta_ax):
        axis.set_ylim(
            positions[0] - top_pad,
            positions[-1] + bottom_pad,
        )

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.spines["left"].set_linewidth(0.7)
        axis.spines["bottom"].set_linewidth(0.7)

        axis.tick_params(
            direction="out",
            length=4.0,
            width=0.7,
            labelsize=8,
        )

        # Pantheon+ at the top, Union3.1 at the bottom.
        axis.invert_yaxis()

    epsilon_ax.tick_params(
        axis="y",
        labelsize=9,
    )

    delta_ax.tick_params(
        axis="y",
        labelsize=9,
    )

    return figure


def main() -> None:
    """Create the figure and print its plotted numerical values."""

    values, provenance = collect_posterior_results()
    intervals, fixed_epsilon, free_epsilon, positions = values

    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": 8,
        "axes.labelsize": 9,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with mpl.rc_context(style):
        figure = make_figure(
            intervals,
            fixed_epsilon,
            free_epsilon,
            positions,
        )

        figure.savefig(
            OUTPUT,
            format="pdf",
            transparent=False,
        )

        plt.close(figure)

    provenance["outputs"] = {
        "pdf": str(OUTPUT.resolve()),
        "width_inches": 3.375, "height_inches": 4.1,
    }
    OUTPUT.with_name(OUTPUT.stem + "_values.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    print(
        "Compilation                  "
        "epsilon median [q16, q84]         fixed    free"
    )

    for compilation, values, fixed, free in zip(
        COMPILATIONS,
        intervals,
        fixed_epsilon,
        free_epsilon,
        strict=True,
    ):
        median, lower, upper = values

        print(
            f"{compilation.label:28} "
            f"{median:+.5f} "
            f"[{lower:+.5f}, {upper:+.5f}]  "
            f"{fixed:7.3f} "
            f"{free:7.3f}"
        )

    print(f"Wrote {OUTPUT}")
    print(f"Wrote {OUTPUT.with_name(OUTPUT.stem + '_values.json')}")


if __name__ == "__main__":
    main()
