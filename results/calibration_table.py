"""Recompute the Sec. IV.B calibrated quantities (and Eq. 21) from the weighted draws."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.calibration import weighted_correlation
from src.release_io import sha256, validate_completed
from src.sample_section4b import CASES, OUTPUT_ROOT, derived_coordinates
from src.sampler import compute_getdist_constraints

ROOT = Path(__file__).resolve().parent.parent


def main(run_root=OUTPUT_ROOT, output_dir=ROOT / "results/tables"):
    constraints, moments, records = {}, {}, {}
    rho = None
    for case, (model, _, _) in CASES.items():
        directory = Path(run_root) / case / model
        summary = validate_completed(directory)
        if (summary.get("case") != case or summary.get("model") != model
                or summary.get("calibration_treatment") != "full_explicit_sampling_with_Gaussian_calibrations"):
            raise ValueError(f"Wrong calibrated case: {directory}")
        with np.load(directory / "posterior.npz", allow_pickle=False) as p:
            names, samples = p["parameter_names"].tolist(), p["samples"]
            weights, loglikes = p["weights"], p["log_likelihoods"]
            weights = weights / weights.sum()
            derived = derived_coordinates(samples, names, case)
            for name, values in derived.items():
                np.testing.assert_allclose(samples[:, names.index(name)], values, atol=1e-12, rtol=1e-12)
            constraints[case] = compute_getdist_constraints(samples, weights, loglikes, names)
            for name in names:
                for key in ("median", "lower_68", "upper_68"):
                    if abs(constraints[case][name][key] - summary["constraints_68_equal_tail"][name][key]) > 1e-10:
                        raise ValueError(f"Saved and recomputed intervals differ: {case}/{name}")
            h0 = derived["H0"]
            mean = float(weights @ h0)
            moments[case] = {"mean": mean, "std": float(np.sqrt(weights @ (h0 - mean)**2))}
            if case == "joint":
                rho = weighted_correlation(derived["a"], samples[:, names.index("epsilon")], weights)
        records[case] = {"posterior_sha256": sha256(directory / "posterior.npz"),
                         "summary_sha256": sha256(directory / "summary.json"),
                         "minimum_chi2": summary["minimum_chi2"], "nlive": summary["nlive"]}
        if "calibrated_best_fit" in summary:
            records[case]["calibrated_best_fit"] = summary["calibrated_best_fit"]
    ratio = constraints["bao"]["H0"]["median"] / constraints["pantheon"]["H0"]["median"]
    alpha = constraints["joint"]["alpha"]
    # alpha < 1: measure the distance to exact CDDR with the upper half-width.
    alpha_sigma = (1 - alpha["median"]) / alpha["plus"]
    # Exact CDDR normalization: the calibration offset is carried by epsilon alone.
    epsilon_unit_alpha = constraints["joint-alpha1"]["epsilon"]
    unit_alpha_fit = records["joint-alpha1"]["calibrated_best_fit"]
    delta_chi2_unit_alpha = (unit_alpha_fit["data_chi2"] + unit_alpha_fit["calibration_chi2"]
                             - records["joint"]["minimum_chi2"])
    separation = abs(moments["pantheon"]["mean"] - moments["bao"]["mean"]) / math.hypot(moments["pantheon"]["std"], moments["bao"]["std"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = {"constraints": constraints, "H0_moments": moments, "runs": records,
              "H0_difference_sigma_gaussian": separation, "separate_fit_median_ratio": ratio,
              "separate_fit_magnitude_offset": 5 * math.log10(ratio), "rho_a_epsilon": rho,
              "alpha_deviation_from_unity_sigma": alpha_sigma,
              "epsilon_unit_alpha": epsilon_unit_alpha, "delta_chi2_unit_alpha": delta_chi2_unit_alpha}
    (output / "section4b.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tex = [r"\begin{tabular}{lc}", r"Quantity & Posterior median and 68\% interval \\", r"\hline"]
    for label, case, name, digits in ((r"$H_0^{\rm BAO}$", "bao", "H0", 2),
            (r"$H_0^{\rm SN}$", "pantheon", "H0", 2), (r"$\alpha$", "joint", "alpha", 3),
            (r"$\varepsilon$", "joint", "epsilon", 3)):
        q = constraints[case][name]
        tex.append(f"{label} & ${q['median']:.{digits}f}^{{+{q['plus']:.{digits}f}}}_{{-{q['minus']:.{digits}f}}}$ " + r"\\")
    tex.append(r"\end{tabular}")
    (output / "section4b.tex").write_text("\n".join(tex) + "\n")
    (output / "section4b.md").write_text(
        "# Calibrated Section IV.B quantities\n\n"
        f"Gaussian H0 separation from posterior moments: {separation:.6f} sigma.\n\n"
        f"Joint alpha = {alpha['median']:.3f} +{alpha['plus']:.3f} -{alpha['minus']:.3f}, "
        f"{alpha_sigma:.2f} sigma below exact CDDR.\n\n"
        f"Ratio of separate H0 medians: {ratio:.8f}; magnitude offset: {5*math.log10(ratio):.8f} mag.\n\n"
        f"Joint weighted correlation of 5 log10(alpha) and epsilon: {rho:.6f}.\n\n"
        f"Joint epsilon at fixed alpha = 1: {epsilon_unit_alpha['median']:.3f} "
        f"+{epsilon_unit_alpha['plus']:.3f} -{epsilon_unit_alpha['minus']:.3f} "
        f"(16th-84th percentiles {epsilon_unit_alpha['lower_68']:.3f} to {epsilon_unit_alpha['upper_68']:.3f}); "
        f"best-fit chi-square increase over free alpha, including the calibration terms: {delta_chi2_unit_alpha:.2f}.\n\n"
        "The ratio of separate medians is a descriptive normalization. The joint alpha interval is computed separately from the joint draws. H0 is in km/s/Mpc. Gaussian calibrations are priors and are excluded from the data chi-square.\n")
    print(f"Wrote calibrated quantities to {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    main(args.run_root, args.output_dir)
