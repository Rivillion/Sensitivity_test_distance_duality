#!/usr/bin/env python3
"""Figure 3: dark-energy AIC improvement with no extension, free epsilon or free curvature.

Each AIC difference is the corresponding chi-square difference minus four.
Baseline and free-epsilon minima come from the run summaries (Table III);
free-curvature minima, with epsilon fixed to zero, from the Table V controls
in results/minima/controls.json. No profiling or optimization is performed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Direct execution puts figures/ on the import path; include the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.common import (DARK, FIGURE_DIR, GOLD, PURPLE, ROOT, SAMPLE_KEYS, SAMPLE_LABELS,
                            STYLE, compilation_positions, format_compilation_axis, write_values)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np

from src.posterior_io import RUN_ROOT, load_run
from src.release_io import repo_relative, sha256

OUTPUT = FIGURE_DIR / "figure3.pdf"
PREVIEW = OUTPUT.with_suffix(".png")  # raster copy shown in the repository README
PREVIEW_DPI = 1000
MINIMA_DIR = Path(os.environ.get("SN_BAO_MINIMA_DIR", str(ROOT / "results/minima")))
MODELS = ("lcdm", "w0wa", "lcdm-epsilon", "w0wa-epsilon")
AIC_PENALTY = 4.0  # two extra CPL parameters
CURVATURE_OFFSET = 0.3
ARROW_STYLE = {"linewidth": 1.2, "alpha": 0.50}
# Jeffreys-scale shading (ColorBrewer GnBu, colourblind safe), drawn with REGION_ALPHA.
REGION_ALPHA = 0.25
AIC_REGIONS = (
    ("Moderate", -10.0, -5.0, "#43a2ca"),
    ("Weak", -5.0, -2.0, "#a8ddb5"),
    ("Compatible", -2.0, 2.0, "#e0f3db"),
    ("Weak", 2.0, 5.0, "#a8ddb5"),
    ("Moderate", 5.0, 10.0, "#43a2ca"),
)


def collect_results(run_root: Path = RUN_ROOT, minima_dir: Path = MINIMA_DIR) -> tuple[tuple, dict]:
    """Return the three chi-square improvements per compilation and their provenance."""
    controls_path = minima_dir / "controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    baseline, free_epsilon, free_curvature, records = [], [], [], []
    for key, label in zip(SAMPLE_KEYS, SAMPLE_LABELS, strict=True):
        runs = {model: load_run(run_root, key, model) for model in MODELS}
        fixed = runs["lcdm"]["minimum_chi2"] - runs["w0wa"]["minimum_chi2"]
        free = runs["lcdm-epsilon"]["minimum_chi2"] - runs["w0wa-epsilon"]["minimum_chi2"]
        curvature = controls[key]["curvature"]
        curved = float(curvature["lcdm"]["chi2"]) - float(curvature["w0wa"]["chi2"])
        if not np.isfinite(curved) or not np.isclose(curved, curvature["delta_de"], rtol=0, atol=1e-9):
            raise ValueError(f"Inconsistent curvature minimum comparison for {key}")
        baseline.append(fixed)
        free_epsilon.append(free)
        free_curvature.append(curved)
        records.append({"sample": key, "label": label,
                        "delta_chi2_baseline": fixed, "delta_chi2_free_epsilon": free,
                        "delta_chi2_free_curvature": curved,
                        "delta_aic_baseline": fixed - AIC_PENALTY,
                        "delta_aic_free_epsilon": free - AIC_PENALTY,
                        "delta_aic_free_curvature": curved - AIC_PENALTY,
                        "curvature_fits": curvature, "runs": runs})
    values = (np.asarray(baseline), np.asarray(free_epsilon), np.asarray(free_curvature),
              compilation_positions())
    provenance = {
        "data_combination": "BAO+SN", "run_root": repo_relative(run_root),
        "minimum_method": "Refined run-summary minima and saved curvature control minima; no new minimization",
        "curvature_provenance": {"controls_path": repo_relative(controls_path),
                                 "controls_sha256": sha256(controls_path), "epsilon_fixed": 0.0},
        "delta_aic_definition": "AIC(LambdaCDM) - AIC(w0waCDM), with the same extension in both models",
        "delta_aic_penalty": AIC_PENALTY,
        "compilations": records,
    }
    return values, provenance


def make_figure(baseline, free_epsilon, free_curvature, positions) -> mpl.figure.Figure:
    """Build the single panel comparing dark-energy AIC improvements."""
    baseline, free_epsilon, free_curvature = (
        np.asarray(v) - AIC_PENALTY for v in (baseline, free_epsilon, free_curvature))
    figure = plt.figure(figsize=(3.375, 2.5))
    # Leave a narrow margin above the panel for the shading legend.
    axis = figure.add_axes([0.24, 0.18, 0.75, 0.73])

    # Both extension arrows branch from the baseline point.
    for y, fixed, free, curved in zip(positions, baseline, free_epsilon, free_curvature, strict=True):
        for endpoint, color in (((free, y), PURPLE), ((curved, y + CURVATURE_OFFSET), GOLD)):
            axis.add_patch(FancyArrowPatch((fixed, y), endpoint, arrowstyle="-|>",
                                           mutation_scale=7.0, color=color, shrinkA=2,
                                           shrinkB=2, zorder=1, **ARROW_STYLE))
    marker_style = {"s": 20, "edgecolors": "black", "linewidths": 0.5, "zorder": 3}
    axis.scatter(baseline, positions, marker="o", color=DARK, label="Baseline", **marker_style)
    axis.scatter(free_epsilon, positions, marker="s", facecolors=PURPLE,
                 label=r"Free $\epsilon$", **marker_style)
    axis.scatter(free_curvature, positions + CURVATURE_OFFSET, marker="D", color=GOLD,
                 label=r"Free $\Omega_k$", **marker_style)

    values = np.concatenate((baseline, free_epsilon, free_curvature))
    extent = max(7.0, float(np.ceil(np.max(np.abs(values)) + 0.5)))
    major_extent = 2.0 * np.floor(extent / 2.0)
    axis.set_xlim(-extent, extent)
    axis.set_xticks(np.arange(-major_extent, major_extent + 1.0, 2.0))
    axis.set_xlabel(r"$\mathrm{AIC}_{\Lambda{\rm CDM}(+x)} - \mathrm{AIC}_{w_0w_a{\rm CDM}(+x)}$",
                    fontsize=9)
    for _, lower, upper, color in AIC_REGIONS:
        axis.axvspan(lower, upper, facecolor=color, edgecolor="none", alpha=REGION_ALPHA, zorder=0)
    # Spans are clipped to the displayed scale without changing its limits.
    axis.set_xlim(-extent, extent)
    axis.axvline(0.0, color="grey", linestyle="--", linewidth=0.8, zorder=0.5)

    legend = axis.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="k",
                         bbox_to_anchor=(0.015, 0.02), fontsize=9, framealpha=1,
                         handlelength=1.0, handletextpad=0.4, borderaxespad=0.0,
                         columnspacing=1.2)
    legend.get_frame().set_linewidth(0.5)
    region_colors = {label: color for label, _, _, color in AIC_REGIONS}
    figure.legend(
        handles=[Patch(facecolor=mpl.colors.to_rgba(region_colors[label], REGION_ALPHA),
                       edgecolor="0.7", linewidth=0.5, label=label)
                 for label in ("Compatible", "Weak", "Moderate")],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), bbox_transform=axis.transAxes,
        ncol=3, frameon=False, fontsize=9, handlelength=1.75, handletextpad=0.4,
        columnspacing=1.3, borderpad=0.25, borderaxespad=0.0,
    )
    format_compilation_axis(axis, positions)
    integer_ticks = np.arange(-extent, extent + 1.0)
    axis.set_xticks(integer_ticks[np.mod(integer_ticks, 2) != 0], minor=True)
    axis.tick_params(axis="x", which="minor", direction="out", length=2.0, width=0.7,
                     labelbottom=False)
    axis.xaxis.set_label_coords(0.5, -0.20)
    axis.xaxis.label.set_verticalalignment("center")
    axis.text(0.05, 0.925, r"$\leftarrow$ favors $\Lambda$CDM", transform=axis.transAxes,
              ha="left", va="center", fontsize=8, clip_on=False)
    axis.text(0.95, 0.925, r"favors $w_0w_a{\rm CDM}$ $\rightarrow$", transform=axis.transAxes,
              ha="right", va="center", fontsize=8, clip_on=False)
    return figure


def main() -> None:
    """Create the figure and record its numerical values and provenance."""
    values, provenance = collect_results()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure = make_figure(*values)
        width_inches, height_inches = map(float, figure.get_size_inches())
        figure.savefig(OUTPUT, format="pdf", transparent=False)
        figure.savefig(PREVIEW, format="png", dpi=PREVIEW_DPI, transparent=False)
        plt.close(figure)
    provenance["outputs"] = {
        "pdf": repo_relative(OUTPUT), "png": repo_relative(PREVIEW), "png_dpi": PREVIEW_DPI,
        "width_inches": width_inches, "height_inches": height_inches,
        "shaded_regions": [{"label": label, "lower": lower, "upper": upper, "color": color,
                            "alpha": REGION_ALPHA}
                           for label, lower, upper, color in AIC_REGIONS],
        "markers": {"baseline": "o", "free_epsilon": "s", "free_curvature": "D"},
        "marker_colors": {"baseline": DARK, "free_epsilon": PURPLE, "free_curvature": GOLD},
        "row_offsets": {"baseline": 0.0, "free_epsilon": 0.0, "free_curvature": CURVATURE_OFFSET},
        "arrows": {"direction": "baseline to extension", **ARROW_STYLE},
    }
    value_output = write_values(OUTPUT, provenance)
    print("Compilation   chi2 baseline/epsilon/curvature   AIC baseline/epsilon/curvature")
    for label, *chi2 in zip(SAMPLE_LABELS, *values[:3], strict=True):
        aic = [value - AIC_PENALTY for value in chi2]
        print(f"{label:13} " + " ".join(f"{v:7.3f}" for v in chi2) + "   "
              + " ".join(f"{v:7.3f}" for v in aic))
    print(f"Wrote {OUTPUT}\nWrote {PREVIEW}\nWrote {value_output}")


if __name__ == "__main__":
    main()
