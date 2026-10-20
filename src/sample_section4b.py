"""Full calibrated Section IV.B sampling in the explicit release coordinates.

Run from the release root: python -m results.sample_calibration --cores 4
The four independent runs are resumable and use a separate output root.
Gaussian calibrations are sampled, not fixed or added as likelihood penalties.
The joint-alpha1 run repeats the joint fit at alpha = 1, where the SN intercept
nu is no longer free but follows from the sampled calibrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from src.background import c_kms
from src.calibration import RD_CALIBRATION, MB_CALIBRATION, weighted_correlation
from src.sampler import (
    PROJECT_ROOT, DATA_DIR, PRIOR_BOUNDS, N_EFF, NUMERICS_VERSION,
    DEFAULT_N_GRID, initialize_likelihoods, sampled_parameters,
    compute_getdist_constraints, maximum_likelihood_result, minimize_explicit,
    write_json_atomic, write_posterior_atomic, print_results, utc_now,
    positive_integer, nonnegative_integer,
)

CASES = {
    "bao": ("lcdm", ("desi",), ("r_d",)),
    "pantheon": ("lcdm", ("pantheon",), ("M_B",)),
    "joint": ("lcdm-epsilon", ("pantheon", "desi"), ("r_d", "M_B")),
    "joint-alpha1": ("lcdm-epsilon", ("pantheon", "desi"), ("r_d", "M_B")),
}
# Cases with exact CDDR normalization: alpha = 1 ties nu to H0rd, r_d and M_B.
ALPHA_FIXED_CASES = ("joint-alpha1",)
CALIBRATIONS = {"r_d": RD_CALIBRATION, "M_B": MB_CALIBRATION}
OUTPUT_ROOT = PROJECT_ROOT / "results" / "runs" / "calibration"
VERSION = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_parameters(case):
    """Uniformly sampled cosmological and data-calibration coordinates."""
    model, datasets, _ = CASES[case]
    names = sampled_parameters(model, datasets)
    if case in ALPHA_FIXED_CASES:
        names = tuple(name for name in names if name != "nu")
    return names


def case_prior(case):
    from nautilus import Prior

    prior = Prior()
    calibrations = CASES[case][2]
    names = active_parameters(case) + calibrations
    specs = {}
    for name in active_parameters(case):
        prior.add_parameter(name, dist=PRIOR_BOUNDS[name])
        specs[name] = {"distribution": "uniform", "bounds": list(PRIOR_BOUNDS[name])}
    for name in calibrations:
        mean, sigma = CALIBRATIONS[name]
        prior.add_parameter(name, dist=norm(loc=mean, scale=sigma))
        specs[name] = {"distribution": "Gaussian", "mean": mean, "sigma": sigma}
    return prior, names, specs


def unit_alpha_intercept(H0rd, r_d, M_B):
    """SN intercept of Eq. (13) at alpha = 1, with H0 = H0rd / r_d."""
    return np.asarray(M_B) + 25 + 5 * np.log10(c_kms * np.asarray(r_d) / np.asarray(H0rd))


class UnitAlphaLikelihood:
    """Explicit likelihood with nu set by the calibrations (alpha = 1)."""

    def __init__(self, likelihood):
        self.likelihood = likelihood

    def __call__(self, parameters):
        nu = unit_alpha_intercept(parameters["H0rd"], parameters["r_d"], parameters["M_B"])
        return self.likelihood({**parameters, "nu": nu})


def calibration_chi2(parameters):
    return sum(((np.asarray(parameters[name]) - mean) / sigma) ** 2
               for name, (mean, sigma) in CALIBRATIONS.items())


def minimize_unit_alpha(likelihood, names, initial_point):
    """Refine the alpha = 1 best fit of the data plus calibration chi-square.

    With nu tied to the calibrations, the data alone would move r_d and M_B
    without limit, so the Gaussian calibration terms enter this optimization.
    They remain priors, never likelihood terms, in the sampling itself.
    Returns the point, its data chi-square, calibration chi-square and status.
    """
    offset, scale, bounds = [], [], []
    for name in names:
        if name in CALIBRATIONS:
            offset.append(CALIBRATIONS[name][0])
            scale.append(CALIBRATIONS[name][1])
            bounds.append((None, None))
        else:
            lower, upper = PRIOR_BOUNDS[name]
            offset.append(lower)
            scale.append(upper - lower)
            bounds.append((0, 1))
    offset, scale = np.asarray(offset), np.asarray(scale)

    def chi2_terms(unit):
        parameters = dict(zip(names, offset + np.asarray(unit) * scale, strict=True))
        return -2.0 * likelihood(parameters), float(calibration_chi2(parameters))

    def objective(unit):
        return sum(chi2_terms(unit))

    initial = (np.asarray(initial_point) - offset) / scale
    result = minimize(
        objective, initial, method="L-BFGS-B", jac="3-point", bounds=bounds,
        options={"ftol": 1e-13, "gtol": 1e-6, "maxiter": 1000, "maxls": 40},
    )
    candidates = [initial]
    if np.all(np.isfinite(result.x)):
        candidates.append(result.x)
    best = min(candidates, key=objective)
    data_chi2, cal_chi2 = chi2_terms(best)
    if not np.isfinite(data_chi2):
        raise RuntimeError("Fixed-alpha best-fit optimization found no finite likelihood.")
    status = {"success": bool(result.success), "message": str(result.message),
              "evaluations": int(result.nfev)}
    return offset + best * scale, float(data_chi2), cal_chi2, status


def derived_coordinates(samples, names, case):
    """Physical transformations of Eq. (13), preserving posterior weights.

    nu = M_B + 25 + 5 log10(c/H0) + 5 log10(alpha).
    In the joint fit H0 is the BAO calibrated scale, so alpha = H0/H0_SN.
    At fixed alpha = 1, nu is derived from H0rd, r_d and M_B instead.
    """
    p = dict(zip(names, np.asarray(samples).T, strict=True))
    values = {}
    if "H0rd" in p:
        values["H0"] = p["H0rd"] / p["r_d"]
    if case in ALPHA_FIXED_CASES:
        values["nu"] = unit_alpha_intercept(p["H0rd"], p["r_d"], p["M_B"])
    elif "nu" in p:
        sn_scale = c_kms * 10 ** ((p["M_B"] + 25 - p["nu"]) / 5)
        if case == "pantheon":
            values["H0"] = sn_scale
        elif case == "joint":
            values["alpha"] = values["H0"] / sn_scale
            values["a"] = 5 * np.log10(values["alpha"])
    if any(np.any(~np.isfinite(x)) for x in values.values()):
        raise ValueError("Non-finite physical calibration transform")
    return values


def signature(case, args, names, specs):
    datasets = CASES[case][1]
    filenames = []
    if "desi" in datasets:
        filenames += ["desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt"]
    if "pantheon" in datasets:
        filenames += ["Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"]
    source_files = [Path(__file__), PROJECT_ROOT / "src" / "sampler.py",
                    PROJECT_ROOT / "src" / "background.py", PROJECT_ROOT / "src" / "likelihoods.py"]
    return {
        "version": VERSION, "case": case, "parameters": list(names), "priors": specs,
        "nlive": args.nlive, "n_eff": args.n_eff, "discard_exploration": True,
        "seed": args.seed + list(CASES).index(case), "zmin": args.zmin,
        "numerics_version": NUMERICS_VERSION, "n_grid": DEFAULT_N_GRID,
        "input_sha256": {f: sha256(DATA_DIR / f) for f in filenames},
        "source_sha256": {p.name: sha256(p) for p in source_files},
    }


def run_case(case, args):
    from nautilus import Sampler

    start = perf_counter()
    model, datasets, calibrations = CASES[case]
    prior, names, specs = case_prior(case)
    sig = signature(case, args, names, specs)
    directory = args.output_root / case / model
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "summary.json"
    checkpoint = directory / "sampler.hdf5"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        from src.release_io import validate_completed, signatures_match
        if previous.get("status") == "completed":
            validate_completed(directory, sig)
        elif not signatures_match(previous.get("resume_signature"), sig):
            raise RuntimeError(f"Incompatible existing run at {directory}; choose a new --output-root")
        if previous.get("status") == "completed":
            if not (directory / "posterior.npz").is_file():
                raise RuntimeError(f"Completed run lacks posterior: {directory}")
            print(f"Already completed: {directory}", flush=True)
            return previous
    elif checkpoint.exists():
        raise RuntimeError(f"Checkpoint without metadata: {directory}")
    likelihood = initialize_likelihoods(datasets, zmin=args.zmin)
    alpha_fixed = case in ALPHA_FIXED_CASES
    if alpha_fixed:
        likelihood = UnitAlphaLikelihood(likelihood)
    summary = {
        "status": "running", "created_at": utc_now(), "model": model,
        "case": case, "likelihoods": list(datasets), "parameters": list(names),
        "priors": specs, "calibration_treatment": "full_explicit_sampling_with_Gaussian_calibrations",
        "fixed_parameters": {"Omega_k": 0.0, "w0": -1.0, "wa": 0.0,
                             **({"epsilon": 0.0} if model == "lcdm" else {}),
                             **({} if case == "joint" else {"alpha": 1.0})},
        "nlive": args.nlive, "discard_exploration": True, "n_eff_target": args.n_eff,
        "seed": sig["seed"], "cores": args.cores, "zmin": args.zmin,
        "resume_signature": sig,
        "coordinate_note": "Uniform H0rd and nu as in the baseline runs; H0 and alpha derived draw by draw. No additional uniform H0 or alpha prior.",
        "likelihood_note": "Full data chi-square with no profiling or analytic marginalization; Gaussian calibrations belong to the prior. ln Z omits fixed data-normalization constants, as in src.sampler.",
    }
    if alpha_fixed:
        summary["alpha_note"] = ("Exact CDDR normalization: nu = M_B + 25 + 5 log10(c r_d / H0rd) "
                                 "is derived draw by draw, so the calibrations enter the SN likelihood.")
    write_json_atomic(summary_path, summary)
    sampler = Sampler(prior, likelihood, n_live=args.nlive, n_batch=256,
                      vectorized=True, pool=None if args.cores == 1 else args.cores,
                      seed=sig["seed"], filepath=checkpoint, resume=True)
    print(f"\n{case}: sampling {', '.join(names)}, n_live={args.nlive}, discard_exploration=True", flush=True)
    try:
        success = sampler.run(n_eff=args.n_eff, discard_exploration=True, verbose=True)
        if not success:
            raise RuntimeError("Nautilus stopped before reaching requested convergence")
        points, log_weights, loglikes = sampler.posterior(return_as_dict=False)
        weights = np.exp(log_weights)
        derived = derived_coordinates(points, names, case)
        all_names = list(names) + list(derived)
        all_points = np.column_stack((points, *derived.values()))
        constraints = compute_getdist_constraints(all_points, weights, loglikes, all_names)
        # Optimize only the data likelihood's active coordinates. The Gaussian
        # calibration variables are likelihood-independent, not penalty terms.
        active = sampled_parameters(model, datasets)
        if alpha_fixed:
            # Unconstrained calibrations can absorb the tied nu, so the data-only
            # minimum is that of the free-alpha fit. Start from the best draw.
            best_draw = dict(zip(all_names, all_points[np.argmax(loglikes)], strict=True))
            best = np.array([best_draw[name] for name in active])
            data_likelihood = likelihood.likelihood
        else:
            best, _ = maximum_likelihood_result(sampler, points, loglikes, len(names))
            data_likelihood = likelihood
        best_active, chi2, status = minimize_explicit(data_likelihood, active, best[:len(active)])
        ml = dict(zip(active, map(float, best_active), strict=True))
        write_posterior_atomic(directory / "posterior.npz", samples=all_points,
            weights=weights, log_weights=log_weights, log_likelihoods=loglikes,
            parameter_names=all_names, constraints=constraints, minimum_chi2=chi2)
        summary.update(status="completed", successful=True, updated_at=utc_now(),
            total_seconds=perf_counter() - start, likelihood_evaluations=int(sampler.n_like),
            effective_sample_size=float(sampler.n_eff), posterior_samples=len(points),
            minimum_chi2=chi2, log_z=float(sampler.log_z),
            maximum_likelihood_parameters=ml, maximum_likelihood_optimizer=status,
            derived_parameters=list(derived), posterior_parameters=all_names,
            constraints_68_equal_tail=constraints)
        # Weighted moments support the explicitly Gaussian approximation used
        # to compare the two independent H0 distributions.
        h0 = derived["H0"]
        w = weights / weights.sum()
        mean = float(w @ h0)
        summary["H0_moments"] = {"mean": mean, "std": float(np.sqrt(w @ (h0 - mean)**2))}
        if alpha_fixed:
            # The cost of alpha = 1 appears once the calibration terms are included.
            initial = points[np.argmax(loglikes - .5 * calibration_chi2(dict(zip(names, points.T))))]
            point, data_chi2, cal_chi2, fit_status = minimize_unit_alpha(likelihood, names, initial)
            fit = dict(zip(names, map(float, point), strict=True))
            fit["nu"] = float(unit_alpha_intercept(fit["H0rd"], fit["r_d"], fit["M_B"]))
            summary["calibrated_best_fit"] = {"parameters": fit, "data_chi2": data_chi2,
                                              "calibration_chi2": cal_chi2, "optimizer": fit_status}
        if case == "joint":
            summary["rho_a_epsilon"] = weighted_correlation(derived["a"], points[:, names.index("epsilon")], weights)
        write_json_atomic(summary_path, summary)
        print_results(constraints, ml, chi2, float(sampler.log_z))
        print(f"Saved: {directory}", flush=True)
        return summary
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       updated_at=utc_now(), error=str(exc), total_seconds=perf_counter() - start)
        write_json_atomic(summary_path, summary)
        raise
    finally:
        pool = getattr(sampler, "pool_l", None)
        if hasattr(pool, "close"):
            pool.close()
            pool.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=positive_integer, default=4)
    parser.add_argument("--nlive", type=positive_integer, default=3000)
    parser.add_argument("--n-eff", type=positive_integer, default=N_EFF)
    parser.add_argument("--seed", type=nonnegative_integer, default=20261004)
    parser.add_argument("--zmin", type=float, default=.01)
    parser.add_argument("--only", choices=tuple(CASES))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "results" / "tables")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not np.isfinite(args.zmin) or args.zmin < 0:
        parser.error("--zmin must be finite and non-negative")
    if not args.report_only:
        for case in ([args.only] if args.only else CASES):
            run_case(case, args)
    if args.report_only:
        from results.calibration_table import main as report
        report(args.output_root, args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
