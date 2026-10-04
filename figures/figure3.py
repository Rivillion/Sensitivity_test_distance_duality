#!/usr/bin/env python3
"""Reproduce only the alpha/epsilon panel using the full Sec. IV.B posterior.

From the release root: python -m figures.figure3
Uses the same GetDist contour and STIX typography approach as figures.figure2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sn_bao_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import MCSamples, chains, plots

from figures.figure2 import SECTOSA
from src.sample_section4b import derived_coordinates
from src.sampler import compute_getdist_constraints

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSTERIOR = ROOT / "results" / "runs" / "calibration" / "joint" / "lcdm-epsilon" / "posterior.npz"
DEFAULT_OUTPUT = Path(os.environ.get("SN_BAO_FIGURE_DIR", str(ROOT / "figures"))) / "figure3.pdf"
WIDTH = .9 * 3.375
HEIGHT = .9 * WIDTH
LIMITS = [.90, 1.008, -.084, .026]
LEVELS = [.68, .95]


def load_posterior(path):
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    if (summary.get("status") != "completed" or summary.get("model") != "lcdm-epsilon"
            or summary.get("likelihoods") != ["pantheon", "desi"]
            or summary.get("calibration_treatment") != "full_explicit_sampling_with_Gaussian_calibrations"):
        raise ValueError("Figure requires the completed joint full-calibration posterior")
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["samples"], dtype=float)
        weights = np.asarray(data["weights"], dtype=float)
        names = data["parameter_names"].tolist()
        loglikes = np.asarray(data["log_likelihoods"], dtype=float)
    if (raw.shape != (len(weights), len(names)) or loglikes.shape != weights.shape
            or np.any(~np.isfinite(raw)) or np.any(~np.isfinite(weights))
            or np.any(~np.isfinite(loglikes)) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid posterior arrays")
    # Check that the saved alpha column agrees with all sampled calibrations.
    derived = derived_coordinates(raw, names, "joint")
    np.testing.assert_allclose(raw[:, names.index("alpha")], derived["alpha"], rtol=1e-12)
    samples = raw[:, [names.index("alpha"), names.index("epsilon")]]
    constraints = compute_getdist_constraints(samples, weights, loglikes, ("alpha", "epsilon"))
    for name, c in constraints.items():
        for key in ("median", "lower_68", "upper_68"):
            if not np.isclose(c[key], summary["constraints_68_equal_tail"][name][key], rtol=0, atol=1e-12):
                raise ValueError(f"Saved posterior and summary disagree: {name}/{key}")
    previous = chains.print_load_details
    chains.print_load_details = False
    try:
        posterior = MCSamples(samples=samples, weights=weights, loglikes=-loglikes,
            names=["alpha", "epsilon"], labels=[r"\alpha", r"\varepsilon"],
            name_tag=r"$\Lambda$CDM + $\varepsilon$",
            settings={"ignore_rows": 0, "contours": LEVELS})
    finally:
        chains.print_load_details = previous
    return posterior, summary, constraints


def make_figure(posterior, constraints):
    plotter = plots.get_single_plotter(width_inch=WIDTH, ratio=HEIGHT/WIDTH, scaling=False)
    plotter.settings.axes_fontsize = 8
    plotter.settings.axes_labelsize = 9
    plotter.settings.legend_fontsize = 9
    plotter.settings.num_plot_contours = 2
    plotter.plot_2d([posterior], "alpha", "epsilon", filled=True,
                   colors=[SECTOSA], lws=[1.2], lims=LIMITS, alphas=[1.])
    plotter.add_2d_contours(posterior, "alpha", "epsilon", filled=False,
                            color=SECTOSA, lw=1.2, add_legend_proxy=False)
    ax = plotter.fig.axes[0]
    ax.axhline(0, linestyle="dashed", linewidth=.75, color="grey", zorder=5)
    ax.axvline(1, linestyle="dashed", linewidth=.75, color="grey", zorder=5)
    ax.set_title(r"DESI DR2$+$Pantheon$+$ ($\Lambda$CDM + $\varepsilon$)", fontsize=9, pad=4.5)
    ax.plot(
            1.0,
            0.0,
            marker="d",
            markersize=7.5,
            markerfacecolor="black",
            markeredgecolor="black",
            linestyle="none",
            zorder=10,
        )
    ax.legend(
        handles=[Line2D(
            [], [], color="black", marker="d", markersize=5,
            linestyle="none", label="Exact CDDR",
        )],
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=0.5,
        borderaxespad=0.3,
    )
    ax.set_xlabel(r"$\alpha$", fontsize=9)
    ax.set_ylabel(r"$\varepsilon$", fontsize=9)
    ax.set_xticks(np.arange(.90, 1.001, .02))
    ax.set_yticks(np.arange(-.08, .021, .02))
    ax.tick_params(direction="out", length=4, width=.7, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.7)
    plotter.fig.set_size_inches(WIDTH, HEIGHT, forward=True)
    ax.set_position([.18, .16, .81, .78])
    return plotter.fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf suffix")
    posterior, summary, constraints = load_posterior(args.posterior)
    style = {"font.family": "serif", "font.serif": ["STIXGeneral"],
             "mathtext.fontset": "stix", "pdf.fonttype": 42,
             "axes.unicode_minus": True, "figure.facecolor": "white",
             "axes.facecolor": "white", "savefig.facecolor": "white"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(style):
        figure = make_figure(posterior, constraints)
        figure.savefig(args.output, format="pdf", transparent=False)
        plt.close(figure)
    density = posterior.get2DDensity("alpha", "epsilon", num_plot_contours=2)
    thresholds = density.getContourLevels(LEVELS)
    np.savez_compressed(args.output.with_name(args.output.stem + "_density.npz"),
                       alpha=density.x, epsilon=density.y, density=density.P,
                       contour_density_thresholds=thresholds, contour_probabilities=LEVELS)
    manifest = {
        "posterior_path": str(args.posterior.resolve()),
        "posterior_sha256": hashlib.sha256(args.posterior.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(args.posterior.with_name("summary.json").read_bytes()).hexdigest(),
        "plotting_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "posterior_draws": summary["posterior_samples"],
        "effective_sample_size": summary["effective_sample_size"],
        "nlive": summary["nlive"], "discard_exploration": summary["discard_exploration"],
        "constraints_68_equal_tail": constraints, "contour_probabilities": LEVELS,
        "density_estimator": "GetDist weighted 2D KDE, default bandwidth and bias correction",
        "density_thresholds": thresholds.tolist(),
        "axis_limits": LIMITS, "width_inch": WIDTH, "height_inch": HEIGHT,
        "style_reference": "figures.figure2", "color": SECTOSA,
        "point": "Black diamond: exact CDDR at (alpha, epsilon)=(1,0)",
        "legend": "Top-left black diamond labeled Exact DDR; styling from figures.figure2",
        "reference_lines": "alpha=1 and epsilon=0 (exact CDDR)",
        "caption": "Joint calibrated DESI DR2 + Pantheon+ flat LambdaCDM + epsilon posterior. Filled contours enclose 68% and 95% of the two-dimensional marginalized posterior. The black diamond, labeled Exact DDR in the legend, marks (alpha,epsilon)=(1,0); grey dashed lines mark alpha=1 and epsilon=0. Independent Gaussian r_d and M_B calibration priors are marginalized.",
        "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__,
                     "getdist": __import__("getdist").__version__},
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
