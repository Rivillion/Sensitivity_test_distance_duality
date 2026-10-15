#!/usr/bin/env python3
"""Figure 4: calibrated (alpha, epsilon) posterior and distance-modulus residuals.

Left: the full explicit-calibration Sec. IV.B DESI DR2 + Pantheon+ LambdaCDM + epsilon
posterior. Right: calibrated Pantheon+ and transverse DESI BAO residuals, prepared
by figures.calibrated_residuals. From the release root: python -m figures.figure4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Direct file execution puts figures/, rather than the repository root, on
# the import path. Resolve the root from this file so the working directory
# chosen by the editor does not affect package imports or input locations.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures.common import DARK, FIGURE_DIR, GREEN, ORANGE, PURPLE, ROOT, STYLE, write_values

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from getdist import MCSamples, chains, plots

from figures.calibrated_residuals import prepare_residuals
from src.release_io import repo_relative, sha256
from src.sample_section4b import derived_coordinates
from src.sampler import compute_getdist_constraints

DEFAULT_POSTERIOR = ROOT / "results" / "runs" / "calibration" / "joint" / "lcdm-epsilon" / "posterior.npz"
DEFAULT_OUTPUT = FIGURE_DIR / "figure4.pdf"
RESIDUAL_INPUTS = ("Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov",
                   "desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt")
# Panel layout in inches: left margin, gap between panels, right and bottom
# margins, and room above for the title. Fixed in inches so the text keeps
# its space at the fixed font sizes. The figure spans the two-column PRD width;
# the panels take 90% of the width a 0.82 in gap would leave them, and the
# remainder widens the gap between them.
WIDTH = 7.0
MARGINS = {"left": .6, "right": .04, "bottom": .47, "top": .24}
PANEL_WIDTH = .9 * (WIDTH - MARGINS["left"] - .82 - MARGINS["right"]) / 2
PANEL_HEIGHT = .9 * 2.6
MARGINS["gap"] = WIDTH - MARGINS["left"] - 2 * PANEL_WIDTH - MARGINS["right"]
HEIGHT = MARGINS["bottom"] + PANEL_HEIGHT + MARGINS["top"]
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 9
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


def draw_residual_panel(ax, residuals):
    """Display correlated distance estimates and their binned best-fit model."""
    sn, bao = residuals["supernova"], residuals["bao"]
    sn_errors = np.sqrt(np.diag(sn["covariance_mag2"]))
    bao_errors = np.sqrt(np.diag(bao["covariance_mag2"]))
    ax.axhline(0, color="grey", linestyle="dashed", linewidth=.8)
    ax.errorbar(sn["redshift"], sn["residual_mag"], yerr=sn_errors,
                fmt="o", markersize=3, capsize=2, linewidth=.9,
                color=ORANGE, label="Binned Pantheon+")
    ax.plot(sn["redshift"], sn["model_residual_mag"], color=PURPLE,
            linewidth=1.5, label=r"$\Lambda$CDM + $\varepsilon$")
    ax.errorbar(bao["redshift"], bao["residual_mag"], yerr=bao_errors,
                fmt="s", markersize=3.3, capsize=2, linewidth=.9,
                color=GREEN, label=r"DESI $D_\mathrm{M}$")
    ax.set(xscale="log", xlim=(.009, 3), ylim=(-.23, .12))
    ax.set_xticks([.01, .03, .1, .3, 1, 2])
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "2"])
    ax.set_xlabel(r"Redshift $z$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\mu-\mu_{\Lambda \rm CDM}$ (mag)", fontsize=LABEL_SIZE)
    # Short handles keep the legend clear of the DESI point at z = 0.51.
    ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE,
              handlelength=1.0, handletextpad=.4, borderaxespad=.3)


def make_figure(posterior, constraints, residuals):
    plotter = plots.get_single_plotter(width_inch=WIDTH, ratio=HEIGHT/WIDTH, scaling=False)
    plotter.settings.axes_fontsize = TICK_SIZE
    plotter.settings.axes_labelsize = LABEL_SIZE
    plotter.settings.legend_fontsize = LEGEND_SIZE
    plotter.settings.num_plot_contours = 2
    plotter.plot_2d([posterior], "alpha", "epsilon", filled=True,
                   colors=[PURPLE], lws=[1.2], lims=LIMITS, alphas=[1.])
    plotter.add_2d_contours(posterior, "alpha", "epsilon", filled=False,
                            color=PURPLE, lw=1.2, add_legend_proxy=False)
    ax = plotter.fig.axes[0]
    ax.axhline(0, linestyle="dashed", linewidth=.8, color="grey", zorder=5)
    ax.axvline(1, linestyle="dashed", linewidth=.8, color="grey", zorder=5)
    ax.set_title(r"DESI DR2$+$Pantheon$+$ ($\Lambda$CDM + $\varepsilon$)", fontsize=LABEL_SIZE, pad=4.5)
    ax.plot(1.0, 0.0, marker="d", markersize=7.5, markerfacecolor=DARK,
            markeredgecolor=DARK, linestyle="none", zorder=10)
    ax.legend(handles=[Line2D([], [], color=DARK, marker="d", markersize=5,
                              linestyle="none", label="Exact CDDR")],
              loc="upper left", frameon=False, fontsize=LEGEND_SIZE,
              handlelength=0.5, borderaxespad=0.3)
    ax.set_xlabel(r"$\alpha$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\varepsilon$", fontsize=LABEL_SIZE)
    ax.set_xticks(np.arange(.90, 1.001, .02))
    ax.set_yticks(np.arange(-.08, .021, .02))
    plotter.fig.set_size_inches(WIDTH, HEIGHT, forward=True)
    # Equal plotting widths, with room between panels for the right y-label.
    m = MARGINS
    panel_width, panel_height = PANEL_WIDTH / WIDTH, PANEL_HEIGHT / HEIGHT
    bottom = m["bottom"] / HEIGHT
    ax.set_position([m["left"] / WIDTH, bottom, panel_width, panel_height])
    residual_ax = plotter.fig.add_axes([(m["left"] + m["gap"]) / WIDTH + panel_width, bottom,
                                        panel_width, panel_height])
    draw_residual_panel(residual_ax, residuals)
    for panel in (residual_ax, ax):
        panel.tick_params(direction="out", length=4, width=.7, labelsize=TICK_SIZE)
        for spine in panel.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.7)
    return plotter.fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must have a .pdf suffix")
    posterior, summary, constraints = load_posterior(args.posterior)
    residuals = prepare_residuals()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        figure = make_figure(posterior, constraints, residuals)
        figure.savefig(args.output, format="pdf", transparent=False)
        plt.close(figure)
    density = posterior.get2DDensity("alpha", "epsilon", num_plot_contours=2)
    thresholds = density.getContourLevels(LEVELS)
    np.savez_compressed(args.output.with_name(args.output.stem + "_density.npz"),
                       alpha=density.x, epsilon=density.y, density=density.P,
                       contour_density_thresholds=thresholds, contour_probabilities=LEVELS)
    manifest = {
        "posterior_path": repo_relative(args.posterior),
        "posterior_sha256": sha256(args.posterior),
        "summary_sha256": sha256(args.posterior.with_name("summary.json")),
        "plotting_script_sha256": sha256(__file__),
        "residual_script_sha256": sha256(Path(__file__).with_name("calibrated_residuals.py")),
        "residual_input_sha256": {name: sha256(ROOT / "data" / name) for name in RESIDUAL_INPUTS},
        "calibrated_residuals": residuals,
        "posterior_draws": summary["posterior_samples"],
        "effective_sample_size": summary["effective_sample_size"],
        "nlive": summary["nlive"], "discard_exploration": summary["discard_exploration"],
        "constraints_68_equal_tail": constraints, "contour_probabilities": LEVELS,
        "density_estimator": "GetDist weighted 2D KDE, default bandwidth and bias correction",
        "density_thresholds": thresholds.tolist(),
        "axis_limits": LIMITS, "width_inch": WIDTH, "height_inch": HEIGHT,
        "layout_margins_inch": MARGINS,
        "panel_size_inch": [PANEL_WIDTH, PANEL_HEIGHT],
        "color": PURPLE, "point": "Dark diamond: exact CDDR at (alpha, epsilon)=(1,0)",
        "reference_lines": "alpha=1 and epsilon=0 (exact CDDR)",
        "caption": "Left: marginalized (alpha, epsilon) posterior for DESI DR2 BAO + Pantheon+ in flat LambdaCDM + epsilon with a Planck r_d prior and a SH0ES M_B prior; contours enclose 68% and 95%; the diamond marks the exact CDDR point (1, 0). Right: distance-modulus residuals relative to a LambdaCDM fit to the full DESI DR2 likelihood. Transverse BAO distances use the Planck r_d prior and that LambdaCDM fit; Pantheon+ uses the SH0ES M_B prior and 18 approximately equal-population bins (simultaneous generalized-least-squares estimates, correlated, for illustration only). The curve is the best-fitting uncalibrated BAO+SN LambdaCDM + epsilon distance modulus, normalized with the SH0ES prior and binned with the same operator. Bars show diagonal standard deviations including the common M_B or r_d uncertainty. Radial BAO and the D_V point enter the fits but are not displayed.",
        "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__,
                     "getdist": __import__("getdist").__version__},
    }
    print(f"Wrote {args.output}\nWrote {write_values(args.output, manifest)}")


if __name__ == "__main__":
    main()
