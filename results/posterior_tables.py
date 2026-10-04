#!/usr/bin/env python3
"""Export updated BAO+SN or BAO+SN+CC intervals and refined minima to LaTeX.

Run from the release root: python -m results.posterior_tables
Add --include-cc for the full-2022 CC dataset and H0/r_d columns.
No sampling or minimization is performed. Output uses REVTeX ruledtabular.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np

from src.posterior_io import MODELS, RUN_ROOT, SAMPLE_KEYS, load_run
from src.sampler import compute_getdist_constraints

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ("Omega_m", "w0", "wa", "epsilon")
SAMPLE_LABELS = (r"Pantheon$+$", "DES-Dovekie", "Union3", "Union3.1")
MODEL_LABELS = {
    "lcdm": r"$\Lambda{\rm CDM}$",
    "w0wa": r"$w_0w_a{\rm CDM}$",
    "lcdm-epsilon": r"$\Lambda{\rm CDM} + \varepsilon$",
    "w0wa-epsilon": r"$w_0w_a{\rm CDM} + \varepsilon$",
}


def collect_rows(run_root: Path, *, include_cc: bool = False) -> list[dict]:
    """Validate each run, then compute cosmological intervals from its weights."""
    rows = []
    columns = COLUMNS + (("H0", "r_d") if include_cc else ())
    for sample, label in zip(SAMPLE_KEYS, SAMPLE_LABELS, strict=True):
        for model in MODELS:
            record = load_run(run_root, sample, model, include_cc=include_cc)
            if include_cc:
                for key, filename in (("data", "Hz_full_2022_mod.txt"),
                                      ("covariance", "covmat_Hz_full_2022.txt")):
                    digest = hashlib.sha256((ROOT / "data" / filename).read_bytes()).hexdigest()
                    if record["cc"].get(f"{key}_sha256") != digest:
                        raise ValueError(f"CC {key} differs from current full-2022 dataset: {sample}/{model}")
            summary = json.loads(Path(record["provenance"]["summary_path"]).read_text())
            if summary.get("fixed_parameters", {}).get("Omega_k") != 0:
                raise ValueError("This table requires spatially flat runs")
            with np.load(record["provenance"]["posterior_path"], allow_pickle=False) as posterior:
                names = posterior["parameter_names"].tolist()
                if include_cc and not {"H0", "r_d", "nu"}.issubset(names):
                    raise ValueError(f"Missing directly sampled CC calibration coordinates: {sample}/{model}")
                free = [name for name in columns if name in names]
                samples = posterior["samples"][:, [names.index(name) for name in free]]
                weights, loglikes = posterior["weights"], posterior["log_likelihoods"]
                varying = [i for i in range(len(free)) if np.ptp(samples[weights > 0, i]) > 0]
                constraints = compute_getdist_constraints(
                    samples[:, varying], weights, loglikes, [free[i] for i in varying],
                ) if varying else {}
                coordinates = {}
                for name in columns:
                    if name in constraints:
                        c = constraints[name]
                        coordinates[name] = {"median": c["median"], "q16": c["lower_68"],
                                             "q84": c["upper_68"], "sampled": True}
                    elif name in free:
                        value = float(samples[weights > 0, free.index(name)][0])
                        coordinates[name] = {"median": value, "q16": value, "q84": value, "sampled": True}
                    else:
                        try:
                            value = float(summary["fixed_parameters"][name])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(f"Missing cosmological coordinate {name} in {sample}/{model}") from exc
                        coordinates[name] = {"fixed": value, "sampled": False}
                    c = coordinates[name]
                    if c["sampled"]:
                        if not (np.isfinite([c["q16"], c["median"], c["q84"]]).all()
                                and c["q16"] <= c["median"] <= c["q84"]):
                            raise ValueError(f"Invalid posterior interval for {sample}/{model}/{name}")
                    elif not np.isfinite(c["fixed"]):
                        raise ValueError(f"Invalid fixed coordinate {name}")
            rows.append({"sample": sample, "sample_label": label, "model": model,
                         "minimum_chi2": record["minimum_chi2"], "coordinates": coordinates,
                         "optimizer_status": record["minimum_optimizer_status"],
                         "provenance": record["provenance"]})
            if include_cc:
                rows[-1]["cc"] = record["cc"]
    return rows


def format_uncertainty(value: float) -> tuple[str, int]:
    """Round to two significant digits, retaining trailing zeros.

    Return the formatted uncertainty and its final decimal exponent so the
    median can be printed at a compatible precision. Handle decade crossings
    after rounding (0.00996 becomes 0.010, rather than 0.0100).
    """
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("Uncertainties must be finite and nonnegative")
    if number == 0:
        return "0", 0
    exponent = number.adjusted() - 1
    rounded = number.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_UP)
    exponent = rounded.adjusted() - 1
    rounded = rounded.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_UP)
    return format(rounded, "f"), exponent


def format_coordinate(name: str, coordinate: dict) -> str:
    if not coordinate["sampled"]:
        return f'${coordinate["fixed"]:g}$'
    median = coordinate["median"]
    plus, minus = coordinate["q84"] - median, median - coordinate["q16"]
    plus_text, plus_exponent = format_uncertainty(plus)
    minus_text, minus_exponent = format_uncertainty(minus)
    exponent = min(plus_exponent, minus_exponent)
    rounded_median = Decimal(str(median)).quantize(
        Decimal(1).scaleb(exponent), rounding=ROUND_HALF_UP,
    )
    if rounded_median == 0:
        rounded_median = abs(rounded_median)
    return f"${rounded_median:f}^{{+{plus_text}}}_{{-{minus_text}}}$"


def render_latex(rows: list[dict], *, include_cc: bool = False) -> str:
    columns = COLUMNS + (("H0", "r_d") if include_cc else ())
    caption = (r"One-dimensional marginalized posterior medians and 16th/84th-percentile errors for DESI DR2 BAO plus each SN compilation and CC.\@ Here $H_0$ is in units of ${\rm km}\, {\rm s}^{-1}\, {\rm Mpc}^{-1}$ and $r_{\rm d}$ in units of Mpc. "
               if include_cc else r"Marginalized posterior medians and 16th/84th-percentile errors for DESI DR2 BAO plus each SN compilation. ")
    caption += r"The $\chi_{\min}^2$ column contains separately optimized likelihood minima."
    label = "tab:posteriorCC" if include_cc else "tab:posterior"
    header = r"Sample & Model & $\chi^2_{\min}$ & $\Omega_{\rm m}$ & $w_0$ & $w_a$ & $\varepsilon$"
    if include_cc:
        header += r" & $H_0$ & $r_{\rm d}$"
    lines = [r"\begin{table*}[t]",
             r"\caption{" + caption + r"}\label{" + label + "}",
             r"\begin{ruledtabular}", r"\begin{tabular}{llr" + "c" * len(columns) + "}",
             header + r" \\"]
    for index, row in enumerate(rows):
        first = index % len(MODELS) == 0
        if first:
            lines.append(r"\hline\rule{0pt}{1.4em}%")
        cells = [row["sample_label"] if first else "", MODEL_LABELS[row["model"]],
                 f'${row["minimum_chi2"]:.2f}$']
        cells.extend(format_coordinate(name, row["coordinates"][name]) for name in columns)
        lines.append(" & ".join(cells) + r" \\[0.4em]")
    lines.extend([r"\end{tabular}", r"\end{ruledtabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--include-cc", action="store_true",
                        help="Export BAO+SN+CC using the full-2022 CC runs")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(os.environ.get("SN_BAO_TABLE_DIR", str(ROOT / "results/tables"))))
    args = parser.parse_args()
    rows = collect_rows(args.runs_dir, include_cc=args.include_cc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / ("posterior_bao_sn_cc.tex" if args.include_cc else "posterior_bao_sn.tex")
    tex_path.write_text(render_latex(rows, include_cc=args.include_cc))
    metadata = {"data_combination": "BAO+SN+CC" if args.include_cc else "BAO+SN", "spatial_geometry": "flat",
                "run_root": str(args.runs_dir.resolve()),
                "quantile_probabilities": [0.16, 0.5, 0.84],
                "quantile_method": "GetDist weighted equal-tail percentiles; ignore_rows=0",
                "uncertainty_format": "Two significant digits; median at finer uncertainty decimal place",
                "minimum_chi2_decimal_places": 2,
                "minimum_method": "Refined minimum_chi2 from summary.json; no new minimization",
                "rows": rows}
    tex_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} rows to {tex_path.resolve()}")


if __name__ == "__main__":
    main()
