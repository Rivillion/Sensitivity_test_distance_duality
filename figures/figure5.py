#!/usr/bin/env python3
"""Figure 5: w0waCDM and w0waCDM + epsilon triangle plots for each SN compilation.

Each 3.5-inch panel is a native GetDist triangle plot overlaying the CPL
posterior with epsilon = 0 (no epsilon subplots) and with free epsilon. All
panels share the same limits and ticks so they can be compared directly. The
four vector panels are assembled, without rasterization, into one 7 x 7 inch PDF.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Direct execution puts figures/ on the import path; include the project root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.common import FIGURE_DIR, PURPLE, SAMPLE_KEYS, STYLE, TEAL, write_values

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from getdist import MCSamples, chains, plots
from matplotlib.lines import Line2D

from src.posterior_io import RUN_ROOT, load_run
from src.release_io import repo_relative

OUTPUT = FIGURE_DIR / "figure5.pdf"
# (model, colour, legend label); free epsilon is drawn first, as in Fig. 2.
MODELS = (
    ("w0wa-epsilon", PURPLE, r"$w_0w_a{\rm CDM} + \varepsilon$"),
    ("w0wa", TEAL, r"$w_0w_a{\rm CDM}$"),
)
PARAMETER_LABELS = {"Omega_m": r"\Omega_{\rm m}", "w0": r"w_0", "wa": r"w_a", "epsilon": r"\epsilon"}
PARAMETERS = tuple(PARAMETER_LABELS)
# Shared limits, tick positions and tick labels for every panel.
AXES = {
    "Omega_m": ((0.10, 0.45), [0.2, 0.35], [r"$0.2$", r"$0.35$"]),
    "w0": ((-1.20, 0.00), [-1.1, -0.7, -0.3], [r"$-1.1$", r"$-0.7$", r"$-0.3$"]),
    "wa": ((-3.5, 2.0), [-2.5, -1.0, 0.5], [r"$-2.5$", r"$-1$", r"$0.5$"]),
    "epsilon": ((-0.15, 0.2), [-0.1, 0.0, 0.1], [r"$-0.1$", r"$0$", r"$0.1$"]),
}
PANEL_TITLES = (r"(a) BAO$+$Pantheon$+$", r"(b) BAO$+$DES-Dovekie",
                r"(c) BAO$+$Union3", r"(d) BAO$+$Union3.1")
FIGURE_SIZE_IN = 7.0
PANEL_SIZE_IN = 3.5
CONTOUR_PROBABILITIES = [0.68, 0.95, 0.99]


def load_posterior(sample: str, model: str) -> MCSamples:
    """Marginalize a validated nlive=3000 posterior to the plotted coordinates."""
    record = load_run(RUN_ROOT, sample, model)
    run = RUN_ROOT / f"{sample}+desi" / model
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("nlive") != 3000 or summary.get("discard_exploration") is not True:
        raise ValueError(f"Expected nlive=3000 and discarded exploration in {run}")
    names = record["provenance"]["parameter_names"]
    parameters = [p for p in PARAMETERS if p in names]
    with np.load(run / "posterior.npz", allow_pickle=False) as posterior:
        # Selecting columns marginalizes the sampled calibration coordinates.
        samples = posterior["samples"][:, [names.index(p) for p in parameters]]
        weights = posterior["weights"]
        loglikes = -posterior["log_likelihoods"]
    previous = chains.print_load_details
    chains.print_load_details = False
    try:
        return MCSamples(samples=samples, weights=weights, loglikes=loglikes, names=parameters,
                         labels=[PARAMETER_LABELS[p] for p in parameters],
                         name_tag=f"{sample} {model}", settings={"ignore_rows": 0})
    finally:
        chains.print_load_details = previous


def render_panel(title: str, samples: list[MCSamples], output_path: Path, *, legend: bool) -> None:
    """Render one native GetDist triangle plot as a vector PDF."""
    g = plots.get_subplot_plotter(width_inch=PANEL_SIZE_IN, scaling=False)
    g.settings.num_plot_contours = len(CONTOUR_PROBABILITIES)
    g.settings.axes_fontsize = 8
    g.settings.axes_labelsize = 9
    g.settings.fontsize = 9
    g.settings.linewidth = 1.0
    g.settings.linewidth_contour = 1.0
    g.settings.alpha_filled_add = 0.30
    g.settings.solid_contour_palefactor = 0.65
    g.settings.figure_legend_frame = False
    g.triangle_plot(
        samples, params=list(PARAMETERS), filled=True,
        contour_colors=[color for _, color, _ in MODELS],
        line_args=[{"color": color, "lw": 1.0} for _, color, _ in MODELS],
        contour_args=[{"alpha": 0.8} for _ in MODELS],
        legend_labels=[],  # panel (a) receives a custom legend below
        param_limits={p: AXES[p][0] for p in PARAMETERS},
    )
    # g.subplots[row, column]: column is the x parameter, row the y parameter.
    for row, y_parameter in enumerate(PARAMETERS):
        for column, x_parameter in enumerate(PARAMETERS[:row + 1]):
            axis = g.subplots[row, column]
            if axis is None:
                continue
            limits, ticks, labels = AXES[x_parameter]
            axis.set_xlim(limits)
            axis.set_xticks(ticks)
            axis.set_xticklabels(labels)
            if row > column:
                limits, ticks, labels = AXES[y_parameter]
                axis.set_ylim(limits)
                axis.set_yticks(ticks)
                axis.set_yticklabels(labels)
    g.fig.set_size_inches(PANEL_SIZE_IN, PANEL_SIZE_IN, forward=True)
    g.fig.subplots_adjust(left=0.18, right=0.985, bottom=0.15, top=0.91, wspace=0.0, hspace=0.0)
    g.fig.text(0.5, 0.965, title, ha="left", va="top", fontsize=9)
    if legend:
        handles = [Line2D([], [], color=color, linewidth=1.2, label=label)
                   for _, color, label in MODELS]
        g.fig.legend(handles=handles[::-1], loc="upper right", bbox_to_anchor=(0.985, 0.91),
                     frameon=False, fontsize=9, handlelength=1.5, labelspacing=0.45)
    # No bbox_inches="tight": every panel must keep the same physical size.
    g.fig.savefig(output_path, format="pdf", transparent=False)
    plt.close(g.fig)


def assemble_panels(panel_paths: list[Path], output_path: Path) -> None:
    """Place the four vector panels on one page: (a) (b) on top, (c) (d) below."""
    size, step = 72.0 * PANEL_SIZE_IN, 72.0 * PANEL_SIZE_IN
    margin = 72.0 * (FIGURE_SIZE_IN - 2.0 * PANEL_SIZE_IN) / 2.0
    document = pymupdf.open()
    page = document.new_page(width=72.0 * FIGURE_SIZE_IN, height=72.0 * FIGURE_SIZE_IN)
    for index, panel_path in enumerate(panel_paths):
        x0 = margin + step * (index % 2)
        y0 = margin + step * (index // 2)  # PyMuPDF uses an upper-left origin.
        with pymupdf.open(panel_path) as source:
            page.show_pdf_page(pymupdf.Rect(x0, y0, x0 + size, y0 + size), source, 0,
                               keep_proportion=True, overlay=True)
    document.save(output_path, garbage=4, deflate=True)
    document.close()


def main() -> None:
    """Generate four GetDist triangles and assemble the final figure."""
    style = {**STYLE, "xtick.labelsize": 8, "ytick.labelsize": 8}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(style), tempfile.TemporaryDirectory() as temporary:
        panel_paths = []
        for index, (sample, title) in enumerate(zip(SAMPLE_KEYS, PANEL_TITLES, strict=True)):
            samples = [load_posterior(sample, model) for model, _, _ in MODELS]
            panel_paths.append(Path(temporary) / f"panel_{index}.pdf")
            render_panel(title, samples, panel_paths[-1], legend=index == 0)
        assemble_panels(panel_paths, OUTPUT)

    records = {f"{sample}/{model}": {"nlive": 3000, "discard_exploration": True,
                                     **load_run(RUN_ROOT, sample, model)["provenance"]}
               for sample in SAMPLE_KEYS for model, _, _ in MODELS}
    write_values(OUTPUT, {
        "run_root": repo_relative(RUN_ROOT), "runs": records,
        "plotted_parameters": list(PARAMETERS), "contour_probabilities": CONTOUR_PROBABILITIES,
        "axes": {p: {"limits": list(limits), "ticks": ticks} for p, (limits, ticks, _) in AXES.items()},
        "method": "GetDist weighted marginalized densities with default smoothing; ignore_rows=0",
        "calibration_treatment": "explicit_sampling; H0rd and nu marginalized",
        "pdf": repo_relative(OUTPUT),
    })
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
