#!/usr/bin/env python3
"""Plot Pantheon+ + DESI DR2 CPL posteriors with GetDist defaults."""

from __future__ import annotations

import os
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sn_bao_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, chains, plots
from matplotlib.lines import Line2D
from src.posterior_io import RUN_ROOT as POSTERIOR_ROOT, load_run


ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = POSTERIOR_ROOT / "pantheon+desi"
FIXED_EPSILON = RUN_ROOT / "w0wa" / "posterior.npz"
FREE_EPSILON = RUN_ROOT / "w0wa-epsilon" / "posterior.npz"
OUTPUT = Path(os.environ.get("SN_BAO_FIGURE_DIR", str(ROOT / "figures"))) / "figure2.pdf"

SECTOSA = "#9671bd"
VIRGINIA = "#77b5b6"
PARAMETER_LABELS = {
    "Omega_m": r"\Omega_{\rm m}",
    "w0": r"w_0",
    "wa": r"w_a",
    "epsilon": r"\epsilon",
}


def load_posterior(path: Path, name_tag: str) -> MCSamples:
    """Load and validate one portable posterior for GetDist."""
    if not path.is_file():
        raise FileNotFoundError(f"Posterior not found: {path}")

    with np.load(path, allow_pickle=False) as posterior:
        required = {"samples", "weights", "parameter_names"}
        missing = required.difference(posterior.files)
        if missing:
            raise ValueError(
                f"{path} is missing required arrays: {sorted(missing)}"
            )
        samples = np.asarray(posterior["samples"], dtype=float)
        weights = np.asarray(posterior["weights"], dtype=float)
        names = [str(name) for name in posterior["parameter_names"]]

    if samples.ndim != 2 or weights.ndim != 1:
        raise ValueError(f"Invalid sample or weight dimensions in {path}")
    if samples.shape != (weights.size, len(names)):
        raise ValueError(f"Inconsistent posterior array shapes in {path}")
    if not {"w0", "wa"}.issubset(names):
        raise ValueError(f"Posterior does not contain both w0 and wa: {path}")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(weights)):
        raise ValueError(f"Posterior contains non-finite values: {path}")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError(f"Posterior weights are invalid: {path}")

    previous_load_details = chains.print_load_details
    chains.print_load_details = False
    try:
        return MCSamples(
            samples=samples,
            weights=weights,
            names=names,
            labels=[PARAMETER_LABELS.get(name, name) for name in names],
            name_tag=name_tag,
            settings={"ignore_rows": 0},
        )
    finally:
        chains.print_load_details = previous_load_details


def make_figure(fixed: MCSamples, free: MCSamples) -> mpl.figure.Figure:
    """Create the PRD-sized two-dimensional marginalized contour plot."""
    width_inch = 0.9 *3.375#0.95 * 248.16 / 72.0
    height_inch = 0.9 * 0.9 * 3.375#0.95 * 219.36 / 72.0

    plotter = plots.get_single_plotter(
        width_inch=width_inch,
        ratio=height_inch / width_inch,
        scaling=False,
    )
    plotter.settings.axes_fontsize = 8
    plotter.settings.axes_labelsize = 9
    plotter.settings.legend_fontsize = 9

    plotter.plot_2d(
        [free, fixed],
        "w0",
        "wa",
        filled=True,
        colors=[SECTOSA, VIRGINIA],
        lws=[1.2, 1.2],
        lims=[-1.15, -0.45, -1.75, 1.60],
        alpha = [0.2, 0.2]
    )

    ax = plotter.fig.axes[0]
    ax.plot(
        -1.0,
        0.0,
        marker="d",
        markersize=7.5,
        markerfacecolor="black",
        markeredgecolor="black",
        linestyle="none",
        zorder=20,
    )
    ax.set_title("DESI DR2 + Pantheon+", fontsize=9, pad=4.5)
    ax.set_xlabel(r"$w_0$", fontsize=9)
    ax.set_ylabel(r"$w_a$", fontsize=9)
    ax.set_xticks([-1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5])
    ax.set_yticks(np.arange(-1.5, 1.51, 0.5))
    ax.tick_params(direction="out", length=4, width=0.7, labelsize=8)
    ax.spines["top"].set_visible(0.7)
    ax.spines["right"].set_visible(0.7)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.hlines(y=0, xmin=-1.5, xmax=0, linestyle="dashed", linewidth=0.75, color="grey")
    ax.vlines(x=-1, ymin=-2.0, ymax=2.0, linestyle="dashed", linewidth=0.75, color="grey")

    handles = [
        Line2D(
            [],
            [],
            color="black",
            marker="d",
            markersize=5,
            linestyle="none",
            label=r"$\Lambda$CDM",
        ),
        Line2D(
            [],
            [],
            color=SECTOSA,
            linewidth=1.2,
            label=r"CPL, free $\epsilon$",
        ),
        Line2D(
            [],
            [],
            color=VIRGINIA,
            linewidth=1.2,
            label=r"CPL, $\epsilon=0$"),
    ]
    ax.legend(
        handles=[handles[0], handles[2], handles[1]],
        loc="upper right",
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        borderaxespad=0.3,
        labelspacing=0.45,
    )

    plotter.fig.set_size_inches(width_inch, height_inch, forward=True)
    ax.set_position([0.16, 0.16, 0.83, 0.78])
    return plotter.fig


def main() -> None:
    """Load the selected saved runs and write the vector PDF."""
    records = {}
    for model in ("w0wa", "w0wa-epsilon"):
        record = load_run(POSTERIOR_ROOT, "pantheon", model)
        summary = json.loads(Path(record["provenance"]["summary_path"]).read_text())
        if summary.get("nlive") != 3000 or summary.get("discard_exploration") is not True:
            raise ValueError(f"Expected nlive=3000 and discarded exploration: {model}")
        records[model] = {"nlive": summary["nlive"], "discard_exploration": True,
                          **record["provenance"]}
    fixed = load_posterior(FIXED_EPSILON, "CPL, epsilon=0")
    free = load_posterior(FREE_EPSILON, "CPL, free epsilon")

    style = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.unicode_minus": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(style):
        figure = make_figure(fixed, free)
        figure.savefig(OUTPUT, format="pdf", transparent=False)
        plt.close(figure)

    OUTPUT.with_name(OUTPUT.stem + "_values.json").write_text(json.dumps({
        "run_root": str(POSTERIOR_ROOT.resolve()), "runs": records,
        "plotted_parameters": ["w0", "wa"], "contour_probabilities": [0.68, 0.95],
        "method": "GetDist weighted marginalized 2D density with default smoothing; ignore_rows=0",
        "calibration_treatment": "explicit_sampling; calibration coordinates marginalized",
        "pdf": str(OUTPUT.resolve()),
    }, indent=2) + "\n")
    print(f"Wrote {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
