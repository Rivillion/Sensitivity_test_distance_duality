#!/usr/bin/env python3
"""Figure 2: DESI DR2 + Pantheon+ (w0, wa) posteriors with epsilon = 0 and free epsilon."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Direct execution puts figures/ on the import path; include the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.common import FIGURE_DIR, PURPLE, STYLE, TEAL, write_values

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, chains, plots
from matplotlib.lines import Line2D

from src.posterior_io import RUN_ROOT, load_run
from src.release_io import repo_relative, resolve

OUTPUT = FIGURE_DIR / "figure2.pdf"
PARAMETER_LABELS = {"Omega_m": r"\Omega_{\rm m}", "w0": r"w_0", "wa": r"w_a", "epsilon": r"\epsilon"}


def load_posterior(path: Path, name_tag: str) -> MCSamples:
    """Load one validated portable posterior for GetDist."""
    with np.load(path, allow_pickle=False) as posterior:
        samples = np.asarray(posterior["samples"], dtype=float)
        weights = np.asarray(posterior["weights"], dtype=float)
        names = [str(name) for name in posterior["parameter_names"]]
    if not {"w0", "wa"}.issubset(names):
        raise ValueError(f"Posterior does not contain both w0 and wa: {path}")
    previous = chains.print_load_details
    chains.print_load_details = False
    try:
        return MCSamples(samples=samples, weights=weights, names=names,
                         labels=[PARAMETER_LABELS.get(name, name) for name in names],
                         name_tag=name_tag, settings={"ignore_rows": 0})
    finally:
        chains.print_load_details = previous


def make_figure(fixed: MCSamples, free: MCSamples) -> mpl.figure.Figure:
    """Create the single-column two-dimensional marginalized contour plot."""
    width_inch = 0.9 * 3.375
    height_inch = 0.9 * width_inch
    plotter = plots.get_single_plotter(width_inch=width_inch, ratio=height_inch / width_inch,
                                       scaling=False)
    plotter.settings.axes_fontsize = 8
    plotter.settings.axes_labelsize = 9
    plotter.settings.legend_fontsize = 9
    plotter.plot_2d([free, fixed], "w0", "wa", filled=True, colors=[PURPLE, TEAL],
                    lws=[1.2, 1.2], lims=[-1.15, -0.35, -1.75, 1.60], alpha=[0.2, 0.2])

    ax = plotter.fig.axes[0]
    ax.plot(-1.0, 0.0, marker="d", markersize=7.5, markerfacecolor="black",
            markeredgecolor="black", linestyle="none", zorder=20)
    ax.set_title("DESI DR2 + Pantheon+", fontsize=9, pad=4.5)
    ax.set_xlabel(r"$w_0$", fontsize=9)
    ax.set_ylabel(r"$w_a$", fontsize=9)
    ax.set_xticks([-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4])
    ax.set_yticks(np.arange(-1.5, 1.51, 0.5))
    ax.tick_params(direction="out", length=4, width=0.7, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.hlines(y=0, xmin=-1.5, xmax=0, linestyle="dashed", linewidth=0.75, color="grey")
    ax.vlines(x=-1, ymin=-2.0, ymax=2.0, linestyle="dashed", linewidth=0.75, color="grey")

    handles = [
        Line2D([], [], color="black", marker="d", markersize=5, linestyle="none",
               label=r"$\Lambda$CDM"),
        Line2D([], [], color=TEAL, linewidth=1.2, label=r"$w_0w_a{\rm CDM}$"),
        Line2D([], [], color=PURPLE, linewidth=1.2, label=r"$w_0w_a{\rm CDM} + \varepsilon$"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
              handlelength=1.5, borderaxespad=0.3, labelspacing=0.45)
    plotter.fig.set_size_inches(width_inch, height_inch, forward=True)
    ax.set_position([0.16, 0.16, 0.83, 0.78])
    return plotter.fig


def main() -> None:
    """Load the selected saved runs and write the vector PDF."""
    records, posteriors = {}, {}
    for model, tag in (("w0wa", "CPL, epsilon=0"), ("w0wa-epsilon", "CPL, free epsilon")):
        record = load_run(RUN_ROOT, "pantheon", model)
        summary = json.loads(resolve(record["provenance"]["summary_path"]).read_text())
        if summary.get("nlive") != 3000 or summary.get("discard_exploration") is not True:
            raise ValueError(f"Expected nlive=3000 and discarded exploration: {model}")
        records[model] = {"nlive": summary["nlive"], "discard_exploration": True,
                          **record["provenance"]}
        posteriors[model] = load_posterior(resolve(record["provenance"]["posterior_path"]), tag)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure = make_figure(posteriors["w0wa"], posteriors["w0wa-epsilon"])
        figure.savefig(OUTPUT, format="pdf", transparent=False)
        plt.close(figure)
    write_values(OUTPUT, {
        "run_root": repo_relative(RUN_ROOT), "runs": records,
        "plotted_parameters": ["w0", "wa"], "contour_probabilities": [0.68, 0.95],
        "method": "GetDist weighted marginalized 2D density with default smoothing; ignore_rows=0",
        "calibration_treatment": "explicit_sampling; calibration coordinates marginalized",
        "pdf": repo_relative(OUTPUT),
    })
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
