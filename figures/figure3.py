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

from figures.common import (DARK, FIGURE_DIR, GOLD, PURPLE, TEAL, ROOT, SAMPLE_KEYS, SAMPLE_LABELS,
                            STYLE, compilation_positions, format_compilation_axis, write_values)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from src.posterior_io import RUN_ROOT, load_run
from src.release_io import repo_relative, sha256

OUTPUT = FIGURE_DIR / "figure3.pdf"
PREVIEW = OUTPUT.with_suffix(".png")  # raster copy shown in the repository README
PREVIEW_DPI = 1000
MINIMA_DIR = Path(os.environ.get("SN_BAO_MINIMA_DIR", str(ROOT / "results/minima")))
MODELS = ("lcdm", "w0wa", "lcdm-epsilon", "w0wa-epsilon")
AIC_PENALTY = 4.0  # two extra CPL parameters
CURVATURE_OFFSET = 0.15
EPSILON_OFFSET = -CURVATURE_OFFSET  # free epsilon sits on the other side of the baseline
# Row padding above the first and below the last compilation; the top leaves room for the
# model names in the panel corners.
ROW_PADDING = {"top": 0.70, "bottom": 0.38}
ARROW_STYLE = {"linewidth": 1.2, "alpha": 0.5}
# Delta AIC labels: right of the baseline points, left of the extension points.
LABEL_STYLE = {"fontsize": 7.5, "format": "{:.2f}", "offset_points": 4.0}
LABEL_ROOM = 1.5  # axis units kept free beyond the outermost point for its label
# Sides of zero favoured by each model: (fill, text). ColorBrewer RdBu (colourblind safe);
# the fills are its lightest classes blended halfway to white.
ZERO_LINE_STYLE = {"color": "grey", "linestyle": "--", "linewidth": 0.8, "zorder": 0.5}
ZERO_LINE_GAP_POINTS = 1.5  # clearance between the zero line and a label crossing it
# Optical correction for the legend: vertical shift of each entry's text in points
# (positive is up). The subscript in Omega_k makes that entry look raised.
LEGEND_TEXT_SHIFT_POINTS = {"Baseline": 0.0, "Free epsilon": 0.0, "Free Omega_k": -1.0}
FAVORED_REGIONS = {"lcdm": ("#feede3", "#b2182b"), "w0wa": ("#e8f2f7", "#2166ac")}


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


def draw_zero_line(axis, labels) -> None:
    """Draw the dashed zero line, interrupted wherever a value label crosses it.

    The gaps let the background shading show through behind the text. Call this once the
    axis limits are final, since the label extents are measured in data coordinates.
    """
    renderer = axis.figure.canvas.get_renderer()
    to_data = axis.transData.inverted()
    pad = ZERO_LINE_GAP_POINTS * axis.figure.dpi / 72.0
    gaps = []
    for label in labels:
        # Only labels whose text crosses zero open a gap; the clearance is vertical.
        box = label.get_window_extent(renderer)
        box = box.from_extents(box.x0, box.y0 - pad, box.x1, box.y1 + pad).transformed(to_data)
        if min(box.x0, box.x1) < 0.0 < max(box.x0, box.x1):
            gaps.append(sorted((box.y0, box.y1)))
    start, stop = sorted(axis.get_ylim())
    for low, high in sorted(gaps):
        if low > start:
            axis.plot([0.0, 0.0], [start, low], **ZERO_LINE_STYLE)
        start = max(start, high)
    if start < stop:
        axis.plot([0.0, 0.0], [start, stop], **ZERO_LINE_STYLE)


def make_figure(baseline, free_epsilon, free_curvature, positions) -> mpl.figure.Figure:
    """Build the single panel comparing dark-energy AIC improvements."""
    baseline, free_epsilon, free_curvature = (
        np.asarray(v) - AIC_PENALTY for v in (baseline, free_epsilon, free_curvature))
    figure = plt.figure(figsize=(3.375, 2.5))
    # Leave a narrow margin above the panel for the legend.
    axis = figure.add_axes([0.24, 0.18, 0.75, 0.73])

    # Both extension arrows branch from the baseline point.
    for y, fixed, free, curved in zip(positions, baseline, free_epsilon, free_curvature, strict=True):
        for endpoint, color in (((free, y + EPSILON_OFFSET), PURPLE), ((curved, y + CURVATURE_OFFSET), GOLD)):
            axis.add_patch(FancyArrowPatch((fixed, y), endpoint, arrowstyle="-|>",
                                           mutation_scale=7.0, color=color, shrinkA=2,
                                           shrinkB=2, zorder=1, **ARROW_STYLE))
    marker_style = {"s": 20, "edgecolors": "black", "linewidths": 0.5, "zorder": 3}
    axis.scatter(baseline, positions, marker="D", color=TEAL, label="Baseline", **marker_style)
    axis.scatter(free_epsilon, positions + EPSILON_OFFSET, marker="o", facecolors=PURPLE,
                 label=r"Free $\epsilon$", **marker_style)
    axis.scatter(free_curvature, positions + CURVATURE_OFFSET, marker="s", color=GOLD,
                 label=r"Free $\Omega_k$", **marker_style)
    offset = LABEL_STYLE["offset_points"]
    labels = []
    for xs, ys, side in ((baseline, positions, 1.0), (free_epsilon, positions + EPSILON_OFFSET, -1.0),
                         (free_curvature, positions + CURVATURE_OFFSET, -1.0)):
        for x, y in zip(xs, ys, strict=True):
            labels.append(axis.annotate(
                LABEL_STYLE["format"].format(x).replace("-", "\u2212"), (x, y),
                xytext=(side * offset, 0.0), textcoords="offset points",
                ha="left" if side > 0 else "right", va="center",
                fontsize=LABEL_STYLE["fontsize"], zorder=4))

    values = np.concatenate((baseline, free_epsilon, free_curvature))
    # Each side is fitted to its outermost point plus room for that point's label.
    lower = float(np.floor(np.min(values) - LABEL_ROOM))
    upper = float(np.ceil(np.max(values) + LABEL_ROOM))
    axis.set_xlim(lower, upper)
    axis.set_xticks(np.arange(2.0 * np.ceil(lower / 2.0), upper + 1.0, 2.0))
    axis.set_xlabel(r"$\Delta\mathrm{AIC}_{x} = \mathrm{AIC}_{\Lambda{\rm CDM}(+x)} - \mathrm{AIC}_{w_0w_a{\rm CDM}(+x)}$",
                    fontsize=9)
    axis.axvspan(lower, 0.0, facecolor=FAVORED_REGIONS["lcdm"][0], edgecolor="none", zorder=0)
    axis.axvspan(0.0, upper, facecolor=FAVORED_REGIONS["w0wa"][0], edgecolor="none", zorder=0)

    legend = axis.legend(loc="lower center", frameon=False, bbox_to_anchor=(0.5, 0.99), ncol=3,
                         fontsize=9, handlelength=1.0, handletextpad=0.4, borderaxespad=0.0,
                         columnspacing=1.2)
    for text, shift in zip(legend.get_texts(), LEGEND_TEXT_SHIFT_POINTS.values(), strict=True):
        text.set_transform(text.get_transform() + mpl.transforms.ScaledTranslation(
            0.0, shift / 72.0, figure.dpi_scale_trans))
    format_compilation_axis(axis, positions)
    axis.set_ylim(positions[-1] + ROW_PADDING["bottom"], positions[0] - ROW_PADDING["top"])
    draw_zero_line(axis, labels)
    integer_ticks = np.arange(lower, upper + 1.0)
    axis.set_xticks(integer_ticks[np.mod(integer_ticks, 2) != 0], minor=True)
    axis.tick_params(axis="x", which="minor", direction="out", length=2.0, width=0.7,
                     labelbottom=False)
    axis.xaxis.set_label_coords(0.5, -0.20)
    axis.xaxis.label.set_verticalalignment("center")
    axis.text(0.02, 0.96, r"$\Lambda$CDM", transform=axis.transAxes, ha="left", va="top",
              fontsize=8, color=FAVORED_REGIONS["lcdm"][1])
    axis.text(0.98, 0.96, r"$w_0w_a{\rm CDM}$", transform=axis.transAxes, ha="right", va="top",
              fontsize=8, color=FAVORED_REGIONS["w0wa"][1])
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
        "point_labels": {"quantity": "delta_aic", **LABEL_STYLE,
                         "side": {"baseline": "right", "free_epsilon": "left",
                                  "free_curvature": "left"}},
        "favored_regions": {model: {"fill": fill, "text": text}
                            for model, (fill, text) in FAVORED_REGIONS.items()},
        "markers": {"baseline": "D", "free_epsilon": "o", "free_curvature": "s"},
        "marker_colors": {"baseline": TEAL, "free_epsilon": PURPLE, "free_curvature": GOLD},
        "row_padding": ROW_PADDING,
        "legend_text_shift_points": LEGEND_TEXT_SHIFT_POINTS,
        "row_offsets": {"baseline": 0.0, "free_epsilon": EPSILON_OFFSET, "free_curvature": CURVATURE_OFFSET},
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
