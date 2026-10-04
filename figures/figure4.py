#!/usr/bin/env python3
"""Render four native GetDist triangle plots in a 2x2 APS-width figure.

Each individual panel is generated directly with g.triangle_plot(), so
the 1D and 2D posterior subplots retain GetDist's native contiguous
triangle layout.

All four panels use identical, manually specified plotting limits and
tick positions for each parameter, so posterior widths and locations can
be compared directly between supernova compilations.

The four vector PDF panels are then assembled into one 7 x 7 inch PDF,
without rasterization.

Panels:
(a) Pantheon+
(b) DES-Dovekie
(c) Union3
(d) Union3.1
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sn_bao_matplotlib")

import pymupdf as fitz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, chains, plots
from src.posterior_io import RUN_ROOT, load_run


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = Path(os.environ.get("SN_BAO_FIGURE_DIR", str(ROOT / "figures")))
OUTPUT = FIGURE_DIR / "figure4.pdf"

SECTOSA = "#9671bd"

PARAMETERS = (
    "Omega_m",
    "w0",
    "wa",
    "epsilon",
)

PARAMETER_LABELS = {
    "Omega_m": r"\Omega_{\rm m}",
    "w0": r"w_0",
    "wa": r"w_a",
    "epsilon": r"\epsilon",
}


# ======================================================================
# ======================================================================
#
#                       TEST AXES HERE
#
# This is the ONLY section you normally need to modify while tuning
# the parameter axes for the final figure.
#
# For every parameter:
#
#   "limits"
#       Sets the minimum and maximum displayed value.
#
#   "ticks"
#       Sets the exact numerical positions of the ticks.
#
#   "ticklabels"
#       Sets exactly what is printed at those tick positions.
#
# All four posterior panels use these same settings.
#
# Examples:
#
#     "limits": (-1.5, 0.0)
#
# means that EVERY w0 axis in EVERY triangle runs from -1.5 to 0.
#
#     "ticks": [-1.5, -0.75, 0.0]
#
# puts exactly three ticks at those positions.
#
#     "ticklabels": [r"$-1.5$", r"$-0.75$", r"$0$"]
#
# determines exactly how those three values are printed.
#
# The values below are simply sensible starting choices. Adjust them
# after looking at the resulting PDF.
#
# ======================================================================
# ======================================================================

AXIS_CONFIG = {
    "Omega_m": {
        "limits": (0.10, 0.45),
        "ticks": [0.2, 0.35],
        "ticklabels": [
            r"$0.2$",
            r"$0.35$",
        ],
    },

    "w0": {
        "limits": (-1.20, 0.00),
        "ticks": [-1.1, -0.7, -0.3],
        "ticklabels": [
            r"$-1.1$",
            r"$-0.7$",
            r"$-0.3$",
        ],
    },

    "wa": {
        "limits": (-3.5, 2),
        "ticks": [-2.5, -1.0, 0.5],
        "ticklabels": [
            r"$-2.5$",
            r"$-1$",
            r"$0.5$",
        ],
    },

    "epsilon": {
        "limits": (-0.15, 0.2),
        "ticks": [-0.1, 0.00, 0.1],
        "ticklabels": [
            r"$-0.1$",
            r"$0$",
            r"$0.1$",
        ],
    },
}


# ----------------------------------------------------------------------
# Physical layout
# ----------------------------------------------------------------------

# Full APS two-column width.
FIGURE_WIDTH_IN = 7.0
FIGURE_HEIGHT_IN = 7.0

# Complete panel size, INCLUDING tick labels, axis labels and panel title.
PANEL_SIZE_IN = 3.5

# Horizontal and vertical separation between complete panels.
PANEL_GAP_IN = 0.0  # e.g. use 0.25 if desired

OUTER_MARGIN_IN = (
    FIGURE_WIDTH_IN
    - 2.0 * PANEL_SIZE_IN
    - PANEL_GAP_IN
) / 2.0


@dataclass(frozen=True)
class PosteriorPanel:
    """One posterior triangle and its metadata."""

    panel_label: str
    title_plain: str
    title_display: str
    run_root: Path
    color: str


PANELS = (
    PosteriorPanel(
        panel_label="(a)",
        title_plain="Pantheon+",
        title_display=r"Pantheon$+$",
        run_root=(
            RUN_ROOT
            / "pantheon+desi"
        ),
        color=SECTOSA,
    ),
    PosteriorPanel(
        panel_label="(b)",
        title_plain="DES-Dovekie",
        title_display="DES-Dovekie",
        run_root=(
            RUN_ROOT
            / "dovekie+desi"
        ),
        color=SECTOSA,
    ),
    PosteriorPanel(
        panel_label="(c)",
        title_plain="Union3",
        title_display="Union3",
        run_root=(
            RUN_ROOT
            / "union3+desi"
        ),
        color=SECTOSA,
    ),
    PosteriorPanel(
        panel_label="(d)",
        title_plain="Union3.1",
        title_display="Union3.1",
        run_root=(
            RUN_ROOT
            / "union31+desi"
        ),
        color=SECTOSA,
    ),
)


def validate_axis_config() -> None:
    """Check the manually specified plotting limits and ticks."""

    for parameter in PARAMETERS:

        if parameter not in AXIS_CONFIG:
            raise ValueError(
                f"No manual axis configuration supplied for {parameter}"
            )

        config = AXIS_CONFIG[parameter]

        lower, upper = config["limits"]
        ticks = config["ticks"]
        ticklabels = config["ticklabels"]

        if lower >= upper:
            raise ValueError(
                f"Invalid limits for {parameter}: "
                f"{config['limits']}"
            )

        if len(ticks) != len(ticklabels):
            raise ValueError(
                f"{parameter}: number of ticks "
                f"({len(ticks)}) does not match number of labels "
                f"({len(ticklabels)})"
            )

        if any(
            tick < lower or tick > upper
            for tick in ticks
        ):
            raise ValueError(
                f"{parameter}: all ticks must lie inside "
                f"the limits {config['limits']}. "
                f"Received {ticks}"
            )

        if any(
            ticks[index] >= ticks[index + 1]
            for index in range(len(ticks) - 1)
        ):
            raise ValueError(
                f"{parameter}: ticks must be strictly increasing."
            )


def get_parameter_limits() -> dict[str, tuple[float, float]]:
    """Return GetDist-compatible manually selected parameter limits."""

    return {
        parameter: tuple(AXIS_CONFIG[parameter]["limits"])
        for parameter in PARAMETERS
    }


def load_posterior(
    panel: PosteriorPanel,
) -> MCSamples:
    """Marginalize a validated explicit nlive=3000 posterior to four coordinates."""

    run = panel.run_root / "w0wa-epsilon"

    summary_path = run / "summary.json"
    posterior_path = run / "posterior.npz"

    if not summary_path.is_file() or not posterior_path.is_file():
        raise FileNotFoundError(
            f"Missing w0wa-epsilon run products under {run}"
        )

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    sample = panel.run_root.name.removesuffix("+desi")
    load_run(panel.run_root.parent, sample, "w0wa-epsilon")

    if (
        summary.get("status") != "completed"
        or summary.get("successful") is not True
    ):
        raise ValueError(
            f"Run did not finish successfully: {summary_path}"
        )

    if summary.get("nlive") != 3000 or summary.get("discard_exploration") is not True:
        raise ValueError(
            f"Expected nlive=3000 and discarded exploration in {summary_path}"
        )

    with np.load(
        posterior_path,
        allow_pickle=False,
    ) as posterior:

        required = {
            "samples",
            "weights",
            "log_likelihoods",
            "parameter_names",
        }

        missing = required.difference(
            posterior.files
        )

        if missing:
            raise ValueError(
                f"{posterior_path} is missing arrays: "
                f"{sorted(missing)}"
            )

        samples = np.asarray(
            posterior["samples"],
            dtype=float,
        )

        weights = np.asarray(
            posterior["weights"],
            dtype=float,
        )

        loglikes = -np.asarray(
            posterior["log_likelihoods"],
            dtype=float,
        )

        names = [
            str(name)
            for name in posterior["parameter_names"]
        ]

    if not set(PARAMETERS).issubset(names):
        raise ValueError(
            f"Unexpected parameters in {posterior_path}: {names}"
        )

    if samples.shape != (
        weights.size,
        len(names),
    ):
        raise ValueError(
            f"Inconsistent posterior shapes in {posterior_path}"
        )

    if not (
        np.isfinite(samples).all()
        and np.isfinite(weights).all()
        and np.isfinite(loglikes).all()
    ):
        raise ValueError(
            f"Non-finite posterior values in {posterior_path}"
        )

    if (
        np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            f"Invalid posterior weights in {posterior_path}"
        )

    previous_load_details = chains.print_load_details
    chains.print_load_details = False
    # Selecting columns marginalizes the sampled calibration coordinates
    # without conditioning them to fitted values or imposing new cuts.
    samples = samples[:, [names.index(parameter) for parameter in PARAMETERS]]
    names = list(PARAMETERS)

    try:
        getdist_samples = MCSamples(
            samples=samples,
            weights=weights,
            loglikes=loglikes,
            names=names,
            labels=[
                PARAMETER_LABELS[name]
                for name in names
            ],
            name_tag=panel.title_plain,
            settings={
                "ignore_rows": 0,
            },
        )

    finally:
        chains.print_load_details = previous_load_details

    return getdist_samples


# ----------------------------------------------------------------------
# Individual triangle panels
# ----------------------------------------------------------------------


def render_panel(
    panel: PosteriorPanel,
    samples: MCSamples,
    parameter_limits: dict[str, tuple[float, float]],
    output_path: Path,
) -> None:
    """Render one native GetDist triangle plot as a vector PDF."""

    # --------------------------------------------------------------
    # Native GetDist subplot plotter
    # --------------------------------------------------------------

    g = plots.get_subplot_plotter(
        width_inch=PANEL_SIZE_IN,
        scaling=False,
    )

    # GetDist styling.
    g.settings.num_plot_contours = 3

    g.settings.axes_fontsize = 8
    g.settings.axes_labelsize = 9
    g.settings.fontsize = 9

    g.settings.linewidth = 1.0
    g.settings.linewidth_contour = 1.0

    # Filled 2D contours only.
    # The diagonal 1D posterior remains an unfilled line.
    g.settings.alpha_filled_add = 0.30
    g.settings.solid_contour_palefactor = 0.65

    g.settings.figure_legend_frame = False

    # Do not let GetDist scale fonts down.
    g.settings.scaling = False

    # --------------------------------------------------------------
    # Native GetDist triangle
    # --------------------------------------------------------------

    g.triangle_plot(
        [samples],
        params=list(PARAMETERS),
        filled=True,
        contour_colors=[panel.color],
        line_args=[
            {
                "color": panel.color,
                "lw": 1.0,
            }
        ],
        param_limits=parameter_limits,
    )

    # --------------------------------------------------------------
    # Apply the EXACT manually selected limits, ticks and labels.
    #
    # g.subplots[row, column]:
    #
    #   column -> x parameter
    #   row    -> y parameter
    #
    # The same AXIS_CONFIG is used for all four triangle plots.
    # --------------------------------------------------------------

    for row in range(len(PARAMETERS)):

        for column in range(row + 1):

            axis = g.subplots[row, column]

            if axis is None:
                continue

            # ------------------------------------------------------
            # X axis
            # ------------------------------------------------------

            x_parameter = PARAMETERS[column]
            x_config = AXIS_CONFIG[x_parameter]

            axis.set_xlim(
                x_config["limits"]
            )

            axis.set_xticks(
                x_config["ticks"]
            )

            axis.set_xticklabels(
                x_config["ticklabels"]
            )

            # ------------------------------------------------------
            # Y axis
            # ------------------------------------------------------

            if row > column:

                y_parameter = PARAMETERS[row]
                y_config = AXIS_CONFIG[y_parameter]

                axis.set_ylim(
                    y_config["limits"]
                )

                axis.set_yticks(
                    y_config["ticks"]
                )

                axis.set_yticklabels(
                    y_config["ticklabels"]
                )

    # --------------------------------------------------------------
    # Force complete panel to remain square.
    # --------------------------------------------------------------

    g.fig.set_size_inches(
        PANEL_SIZE_IN,
        PANEL_SIZE_IN,
        forward=True,
    )

    # Keep native GetDist cells directly adjacent to each other.
    g.fig.subplots_adjust(
        left=0.18,
        right=0.985,
        bottom=0.15,
        top=0.91,
        wspace=0.0,
        hspace=0.0,
    )

    # --------------------------------------------------------------
    # Panel label + dataset name
    # --------------------------------------------------------------

    g.fig.text(
        0.5,
        0.965,
        f"{panel.panel_label} {panel.title_display}",
        ha="left",
        va="top",
        fontsize=9,
    )

    # No bbox_inches="tight": all source PDFs must have exactly the
    # same physical dimensions before assembly.
    g.fig.savefig(
        output_path,
        format="pdf",
        transparent=False,
    )

    plt.close(g.fig)


# ----------------------------------------------------------------------
# Assemble the four vector panels
# ----------------------------------------------------------------------


def assemble_panels(
    panel_paths: list[Path],
    output_path: Path,
) -> None:
    """Assemble four vector PDF panels onto one 7 x 7 inch PDF."""

    if len(panel_paths) != 4:
        raise ValueError(
            "Exactly four panel PDFs are required."
        )

    pt_per_in = 72.0

    page_width = (
        FIGURE_WIDTH_IN * pt_per_in
    )

    page_height = (
        FIGURE_HEIGHT_IN * pt_per_in
    )

    margin = (
        OUTER_MARGIN_IN * pt_per_in
    )

    panel_size = (
        PANEL_SIZE_IN * pt_per_in
    )

    gap = (
        PANEL_GAP_IN * pt_per_in
    )

    # PyMuPDF uses an upper-left coordinate origin.
    x_left = margin
    x_right = margin + panel_size + gap

    y_top = margin
    y_bottom = margin + panel_size + gap

    rectangles = (

        # (a) top left
        fitz.Rect(
            x_left,
            y_top,
            x_left + panel_size,
            y_top + panel_size,
        ),

        # (b) top right
        fitz.Rect(
            x_right,
            y_top,
            x_right + panel_size,
            y_top + panel_size,
        ),

        # (c) bottom left
        fitz.Rect(
            x_left,
            y_bottom,
            x_left + panel_size,
            y_bottom + panel_size,
        ),

        # (d) bottom right
        fitz.Rect(
            x_right,
            y_bottom,
            x_right + panel_size,
            y_bottom + panel_size,
        ),
    )

    output_document = fitz.open()

    page = output_document.new_page(
        width=page_width,
        height=page_height,
    )

    for panel_path, rectangle in zip(
        panel_paths,
        rectangles,
        strict=True,
    ):

        source_document = fitz.open(
            panel_path
        )

        try:

            # Embed vector PDF content directly.
            page.show_pdf_page(
                rectangle,
                source_document,
                0,
                keep_proportion=True,
                overlay=True,
            )

        finally:
            source_document.close()

    output_document.save(
        output_path,
        garbage=4,
        deflate=True,
    )

    output_document.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    """Generate four GetDist triangles and assemble the final figure."""

    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "font.size": 8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Validate your manually selected axes before doing any plotting.
    validate_axis_config()

    parameter_limits = get_parameter_limits()

    # Print the current manual configuration each time the script runs.
    # This is useful while iterating on the figure.
    print("Manual parameter axes:")

    for parameter in PARAMETERS:

        config = AXIS_CONFIG[parameter]

        print(
            f"  {parameter:10s}: "
            f"limits={config['limits']}, "
            f"ticks={config['ticks']}, "
            f"labels={config['ticklabels']}"
        )

    with mpl.rc_context(style):

        # ----------------------------------------------------------
        # Load all four posteriors.
        # ----------------------------------------------------------

        samples_list = [
            load_posterior(panel)
            for panel in PANELS
        ]

        # ----------------------------------------------------------
        # Render temporary vector panels.
        # ----------------------------------------------------------

        with tempfile.TemporaryDirectory() as temporary_directory:

            temporary_directory = Path(
                temporary_directory
            )

            panel_paths: list[Path] = []

            for index, (panel, samples) in enumerate(
                zip(
                    PANELS,
                    samples_list,
                    strict=True,
                )
            ):

                panel_path = (
                    temporary_directory
                    / f"panel_{index}.pdf"
                )

                render_panel(
                    panel,
                    samples,
                    parameter_limits,
                    panel_path,
                )

                panel_paths.append(
                    panel_path
                )

            # ------------------------------------------------------
            # Assemble final 2x2 vector figure.
            # ------------------------------------------------------

            assemble_panels(
                panel_paths,
                OUTPUT,
            )

    records = {}
    for panel in PANELS:
        sample = panel.run_root.name.removesuffix("+desi")
        record = load_run(panel.run_root.parent, sample, "w0wa-epsilon")
        records[sample] = {"nlive": 3000, "discard_exploration": True,
                          **record["provenance"]}
    OUTPUT.with_name(OUTPUT.stem + "_values.json").write_text(json.dumps({
        "run_root": str(RUN_ROOT.resolve()), "runs": records,
        "plotted_parameters": list(PARAMETERS), "contour_probabilities": [0.68, 0.95, 0.99],
        "method": "GetDist weighted marginalized densities with default smoothing; ignore_rows=0",
        "calibration_treatment": "explicit_sampling; H0rd and nu marginalized",
        "pdf": str(OUTPUT.resolve()),
    }, indent=2) + "\n")
    print(f"Wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
