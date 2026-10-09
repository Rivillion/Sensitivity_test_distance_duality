"""Shared paths, colours, typography and provenance output for the figures.

Import this module before matplotlib so its cache directory is configured.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sn_bao_matplotlib")

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = Path(os.environ.get("SN_BAO_FIGURE_DIR", str(ROOT / "figures")))

SAMPLE_KEYS = ("pantheon", "dovekie", "union3", "union31")
SAMPLE_LABELS = ("Pantheon+", "DES-Dovekie", "Union3", "Union3.1")

PURPLE = "#7570b3"   # LambdaCDM + epsilon and w0waCDM + epsilon
TEAL = "#77b5b6"     # w0waCDM with epsilon = 0
GREEN = "#1b9e77"    # w0waCDM + epsilon in Fig. 1; DESI BAO in Fig. 4
ORANGE = "#d95f02"   # binned Pantheon+ in Fig. 4
DARK = "#333333"     # LambdaCDM / exact CDDR markers and AIC baseline
GOLD = "#c9a66b"     # free spatial curvature in Fig. 3

STYLE = {
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


def compilation_positions():
    """Vertical row positions shared by the single-column comparison figures."""
    import numpy as np

    return 0.75 * np.arange(len(SAMPLE_KEYS), dtype=float)


def format_compilation_axis(axis, positions) -> None:
    """Apply the row order, margins, spines and typography of Figs. 1 and 3."""
    axis.set_yticks(positions, SAMPLE_LABELS)
    axis.set_ylim(positions[0] - 0.58, positions[-1] + 0.57)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(direction="out", length=4.0, width=0.7, labelsize=8)
    axis.tick_params(axis="y", labelsize=9)
    axis.invert_yaxis()


def write_values(pdf: Path, record: dict) -> Path:
    """Save a figure's plotted values and provenance next to its PDF."""
    path = pdf.with_name(pdf.stem + "_values.json")
    path.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path
