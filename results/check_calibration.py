"""Independent quadrature check of separate calibrated H0 posteriors.

This is validation, not the production sampler. SN nu and M_B can be
integrated exactly at fixed geometry; BAO Gaussian b integrals include the
1/b**2 Jacobian required by the release's uniform H0rd prior.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm
from scipy.stats import norm

from src.background import Background, c_kms
from src.calibration import (
    pantheon_and_bao, sn_intercept, RD_CALIBRATION, MB_CALIBRATION,
)
from src.sampler import PROJECT_ROOT, write_json_atomic
from src.sample_section4b import ALPHA_FIXED_CASES, CASES, OUTPUT_ROOT, derived_coordinates


def compare_runs(reference, run_root):
    checks = {}
    for case, (model, _, calibrations) in CASES.items():
        directory = run_root / case / model
        summary = json.loads((directory / "summary.json").read_text())
        if (summary["status"] != "completed" or summary["nlive"] != 3000
                or summary["discard_exploration"] is not True
                or summary["effective_sample_size"] < 10000):
            raise RuntimeError(f"Incomplete or incorrectly configured production run: {case}")
        with np.load(directory / "posterior.npz", allow_pickle=False) as posterior:
            samples = posterior["samples"]
            names = posterior["parameter_names"].tolist()
            w = posterior["weights"]
            if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(w)) or np.any(w < 0):
                raise RuntimeError(f"Invalid posterior arrays: {case}")
            if not np.isclose(w.sum(), 1):
                raise RuntimeError(f"Posterior weights are not normalized: {case}")
            for name, derived in derived_coordinates(samples, names, case).items():
                np.testing.assert_allclose(samples[:, names.index(name)], derived, atol=1e-12, rtol=1e-12)
            record = {"posterior_draws": len(w), "effective_sample_size": summary["effective_sample_size"],
                      "derived_coordinate_check": "passed", "calibration_prior_checks": {}}
            ess = summary["effective_sample_size"]
            # At fixed alpha the data constrain the calibrations through nu.
            for name in () if case in ALPHA_FIXED_CASES else calibrations:
                x = samples[:, names.index(name)]
                mean, std = float(w@x), float(np.sqrt(w@(x-w@x)**2))
                specification = summary["priors"][name]
                shift = mean-specification["mean"]
                if abs(shift) > 5*specification["sigma"]/math.sqrt(ess) or abs(std/specification["sigma"]-1) > .05:
                    raise RuntimeError(f"Gaussian calibration posterior disagrees with factorized prior: {case}/{name}")
                record["calibration_prior_checks"][name] = {"mean": mean, "std": std, "mean_shift": shift}
            if case in ("bao", "pantheon"):
                delta = {key: summary["H0_moments"][key]-reference[case][key] for key in ("mean", "std")}
                if abs(delta["mean"]) > 5*reference[case]["std"]/math.sqrt(ess) or abs(delta["std"]) > .05*reference[case]["std"]:
                    raise RuntimeError(f"Production H0 posterior disagrees with independent quadrature: {case}")
                record["H0_sampling_minus_quadrature"] = delta
                if case == "pantheon":
                    c = summary["constraints_68_equal_tail"]["H0"]
                    record["H0_quantile_differences"] = {
                        "q16": c["lower_68"]-reference[case]["q16"],
                        "median": c["median"]-reference[case]["median"],
                        "q84": c["upper_68"]-reference[case]["q84"]}
            checks[case] = record
    return checks


def integrate_separate(n_grid):
    sn, bao = pantheon_and_bao()
    from scipy.linalg import cho_solve
    precision_ones = cho_solve((sn.L, True), np.ones(len(sn.mask)))
    omega = np.linspace(.1, .5, n_grid)
    sn_chi2, nu_hat, bao_chi2, b_hat, b_sigma = np.empty((5, n_grid))
    for i, om in enumerate(omega):
        bg = Background(Omega_m=om)
        sn_chi2[i] = sn.profiled_linear_nuisances(bg)[0]
        nu_hat[i] = sn_intercept(sn, bg, precision_ones)
        bao_chi2[i] = bao.profile_chi2(bg)
        b_hat[i], b_sigma[i] = bao.conditional_b(bg)

    # Flat nu prior is so broad that truncation is negligible. Its conditional
    # variance is geometry-independent, so the integrated geometry density
    # is exactly exp(-profile_chi2/2), up to that negligible truncation.
    sn_density = np.exp(-.5 * (sn_chi2 - sn_chi2.min()))
    sn_density /= simpson(sn_density, x=omega)
    k = math.log(10)/5
    log_mean = math.log(c_kms) + k*(MB_CALIBRATION[0] + 25 - nu_hat)
    log_std = k * math.hypot(MB_CALIBRATION[1], 1/math.sqrt(sn.offset_norm))

    def cdf(h0):
        return simpson(sn_density * norm.cdf((np.log(h0) - log_mean)/log_std), x=omega)

    qs = {str(q): brentq(lambda h: cdf(h)-q, 50., 100.) for q in (.16, .5, .84)}
    sn_mean = simpson(sn_density * np.exp(log_mean + .5*log_std**2), x=omega)
    sn_second = simpson(sn_density * np.exp(2*log_mean + 2*log_std**2), x=omega)

    # p(b|Omega,data) db proportional to exp(-chi2/2) b^-2 db.
    # Thus the integrated geometry density also contains sigma_b <b^-2>.
    nodes, gh_weights = roots_hermitenorm(80)
    gh_weights /= math.sqrt(2*math.pi)
    b = b_hat[:, None] + b_sigma[:, None] * nodes
    jacobian = np.where((b >= 10) & (b <= 100), b**-2, 0)
    integrated = jacobian @ gh_weights
    density = np.exp(-.5*(bao_chi2-bao_chi2.min())) * b_sigma * integrated
    density /= simpson(density, x=omega)
    # rd tails are far from zero; Gauss-Hermite integration is well conditioned.
    rd = RD_CALIBRATION[0] + RD_CALIBRATION[1]*nodes
    rd_inverse = gh_weights @ (1/rd)
    rd_inverse_second = gh_weights @ (1/rd**2)
    product_mean = c_kms * ((jacobian/b) @ gh_weights)/integrated
    product_second = c_kms**2 * ((jacobian/b**2) @ gh_weights)/integrated
    bao_mean = simpson(density * product_mean, x=omega) * rd_inverse
    bao_second = simpson(density * product_second, x=omega) * rd_inverse_second
    return {
        "omega_grid_points": n_grid,
        "pantheon": {"mean": float(sn_mean), "std": float(np.sqrt(sn_second-sn_mean**2)),
                     "q16": qs["0.16"], "median": qs["0.5"], "q84": qs["0.84"]},
        "bao": {"mean": float(bao_mean), "std": float(np.sqrt(bao_second-bao_mean**2))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "tables" / "quadrature_audit.json")
    parser.add_argument("--compare-runs", action="store_true", help="Also validate completed 3000-live-point production runs")
    parser.add_argument("--run-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    coarse, fine = integrate_separate(401), integrate_separate(801)
    convergence = {case: {key: fine[case][key]-coarse[case][key] for key in fine[case]}
                   for case in ("bao", "pantheon")}
    if max(abs(v) for case in convergence.values() for v in case.values()) > 1e-5:
        raise RuntimeError("Independent integration failed grid convergence")
    result = {"method": "independent_quadrature_including_H0rd_prior_Jacobian",
              "fine": fine, "coarse": coarse, "grid_differences": convergence}
    if args.compare_runs:
        result["production_checks"] = compare_runs(fine, args.run_root)
    write_json_atomic(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
