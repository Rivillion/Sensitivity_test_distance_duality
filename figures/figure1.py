#!/usr/bin/env python3
"""Figure 1: epsilon constraints for LambdaCDM + epsilon and w0waCDM + epsilon.

Medians and 16th/84th percentiles of the weighted BAO+SN posteriors for each
supernova compilation. No sampling or minimization is performed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct execution puts figures/ on the import path; include the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.common import (FIGURE_DIR, GREEN, PURPLE, SAMPLE_KEYS, SAMPLE_LABELS, STYLE,
                            compilation_positions, format_compilation_axis, write_values)

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar
import numpy as np

from src.posterior_io import RUN_ROOT, load_run
from src.release_io import repo_relative

OUTPUT = FIGURE_DIR / "figure1.pdf"
MODELS = ("lcdm-epsilon", "w0wa-epsilon")
MODEL_LABELS = (r"$\Lambda\mathrm{CDM}+\epsilon$", r"$w_0w_a\mathrm{CDM}+\epsilon$")
MODEL_COLORS = (PURPLE, GREEN)
MODEL_MARKERS = ("o", "s")
ROW_OFFSETS = (-0.09, 0.09)
ROW_PADDING = 0.51


def collect_results(run_root: Path = RUN_ROOT) -> tuple[tuple, dict]:
    """Collect both models with the same weighted equal-tail estimator."""
    intervals = {model: [] for model in MODELS}
    records = []
    for key, label in zip(SAMPLE_KEYS, SAMPLE_LABELS, strict=True):
        runs = {model: load_run(run_root, key, model) for model in MODELS}
        for model, run in runs.items():
            epsilon = run["epsilon"]
            intervals[model].append([epsilon["median"], epsilon["q16"], epsilon["q84"]])
        records.append({"sample": key, "label": label, "runs": runs})
    values = tuple(np.asarray(intervals[model]) for model in MODELS) + (compilation_positions(),)
    provenance = {
        "data_combination": "BAO+SN", "run_root": repo_relative(run_root),
        "epsilon_models": list(MODELS), "epsilon_quantile_probabilities": [0.16, 0.5, 0.84],
        "epsilon_quantile_method": "GetDist weighted equal-tail percentiles; ignore_rows=0",
        "compilations": records,
    }
    return values, provenance


def make_figure(lcdm_intervals, w0wa_intervals, positions) -> mpl.figure.Figure:
    """Build the comparison with distinct marker shapes and colours per model."""
    figure = plt.figure(figsize=(3.375, 2.5))
    axis = figure.add_axes([0.25, 0.17, 0.72, 0.75])
    for intervals, marker, offset, label, color in zip(
        (lcdm_intervals, w0wa_intervals), MODEL_MARKERS, ROW_OFFSETS,
        MODEL_LABELS, MODEL_COLORS, strict=True,
    ):
        medians, lower, upper = intervals.T
        axis.errorbar(
            medians, positions + offset,
            xerr=np.vstack((medians - lower, upper - medians)),
            fmt=marker, color=color, markersize=4.0, markeredgewidth=0.8,
            elinewidth=1.2, capsize=3.0, capthick=1.0, zorder=3, label=label,
        )
    axis.axvline(0.0, color="grey", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    all_intervals = np.concatenate((lcdm_intervals, w0wa_intervals))
    # Center zero in the frame while keeping every interval inside the limits.
    extent = max(0.083, float(np.max(np.abs(all_intervals[:, 1:]))) + 0.005)
    axis.set_xlim(-extent, extent)
    axis.set_xticks(np.arange(np.ceil(-extent / 0.04), np.floor(extent / 0.04) + 1) * 0.04)
    axis.set_xlabel(r"$\epsilon$", fontsize=9)
    format_compilation_axis(axis, positions)
    # Equal margins above the first and below the last row (inverted axis).
    axis.set_ylim(positions[-1] + ROW_PADDING, positions[0] - ROW_PADDING)
    axis.legend(
        *axis.get_legend_handles_labels(), loc="upper center",
        bbox_to_anchor=(0.5, 1.1), frameon=False,
        ncol=2, fontsize=9, handlelength=2.2, handletextpad=0.45,
        handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=0.9)},
        columnspacing=1.5, borderaxespad=0.0, borderpad=0.15,
    )
    return figure


def main() -> None:
    """Save the plot and full-precision posterior intervals with provenance."""
    values, provenance = collect_results()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure = make_figure(*values)
        figure.savefig(OUTPUT, format="pdf", transparent=False)
        plt.close(figure)
    provenance["outputs"] = {
        "pdf": repo_relative(OUTPUT), "width_inches": 3.375, "height_inches": 2.5,
        "markers": dict(zip(MODELS, MODEL_MARKERS)),
        "row_offsets": dict(zip(MODELS, ROW_OFFSETS)), "row_padding": ROW_PADDING,
    }
    value_output = write_values(OUTPUT, provenance)
    print("Compilation   model             epsilon median [q16, q84]")
    for label, lcdm, w0wa in zip(SAMPLE_LABELS, *values[:2], strict=True):
        for model, (median, lower, upper) in zip(MODELS, (lcdm, w0wa), strict=True):
            print(f"{label:13} {model:17} {median:+.5f} [{lower:+.5f}, {upper:+.5f}]")
    print(f"Wrote {OUTPUT}\nWrote {value_output}")


if __name__ == "__main__":
    main()
