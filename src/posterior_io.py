"""Validate saved weighted posteriors shared by plots and tables."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from src.release_io import repo_relative
from src.sampler import compute_getdist_constraints
ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = Path(os.environ.get("SN_BAO_POSTERIOR_ROOT", str(ROOT / "results/runs/baseline")))
MODELS = ("lcdm", "w0wa", "lcdm-epsilon", "w0wa-epsilon")
SAMPLE_KEYS = ("pantheon", "dovekie", "union3", "union31")

def load_run(run_root: Path, sample: str, model: str, *, include_cc: bool = False) -> dict:
    """Read and validate a completed explicit BAO+SN (optionally CC) run."""
    suffix = "+cc" if include_cc else ""
    directory = run_root / f"{sample}+desi{suffix}" / model
    summary_path, posterior_path = directory / "summary.json", directory / "posterior.npz"
    if not summary_path.is_file() or not posterior_path.is_file():
        raise FileNotFoundError(f"Missing completed posterior files in {directory}")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if (summary.get("status") != "completed" or not summary.get("successful")
            or summary.get("model") != model
            or summary.get("likelihoods") != [sample, "desi"] + (["cc"] if include_cc else [])
            or summary.get("calibration_treatment") != "explicit_sampling"):
        raise ValueError(f"Incompatible or incomplete BAO+SN run: {summary_path}")
    if include_cc:
        cc = summary.get("cc") or {}
        if cc.get("dataset") != "full-2022" or cc.get("h0_mode") != "explicit":
            raise ValueError(f"Expected full-2022 explicit CC likelihood: {summary_path}")
    with np.load(posterior_path, allow_pickle=False) as posterior:
        required = {"samples", "weights", "log_likelihoods", "parameter_names"}
        if not required.issubset(posterior.files):
            raise ValueError(f"Missing posterior arrays: {posterior_path}")
        samples = np.asarray(posterior["samples"], dtype=float)
        weights = np.asarray(posterior["weights"], dtype=float)
        loglikes = np.asarray(posterior["log_likelihoods"], dtype=float)
        names = posterior["parameter_names"].tolist()
    if (samples.ndim != 2 or samples.shape[0] == 0
            or weights.shape != (samples.shape[0],)
            or loglikes.shape != weights.shape
            or not isinstance(names, list)
            or len(names) != samples.shape[1] or len(set(names)) != len(names)
            or names != summary.get("parameters")):
        raise ValueError(f"Inconsistent posterior dimensions or columns: {posterior_path}")
    if (np.any(~np.isfinite(samples)) or np.any(~np.isfinite(loglikes))
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError(f"Invalid posterior samples, likelihoods, or weights: {posterior_path}")
    sampled_minimum = float(-2.0 * np.max(loglikes))
    try:
        refined_minimum = float(summary["minimum_chi2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid refined minimum chi-square: {summary_path}") from exc
    if (not np.isfinite(refined_minimum) or refined_minimum < 0
            or refined_minimum > sampled_minimum + 1e-6):
        raise ValueError(f"Invalid refined minimum chi-square: {summary_path}")
    result = {
        "minimum_chi2": refined_minimum,
        "minimum_optimizer_status": summary.get("maximum_likelihood_optimizer"),
        "sampled_minimum_chi2": sampled_minimum,
        "provenance": {
            "posterior_path": repo_relative(posterior_path),
            "posterior_sha256": hashlib.sha256(posterior_path.read_bytes()).hexdigest(),
            "summary_path": repo_relative(summary_path),
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "posterior_draw_count": len(samples),
            "parameter_names": names,
        },
    }
    if include_cc:
        result["cc"] = summary["cc"]
    if model in ("lcdm-epsilon", "w0wa-epsilon"):
        if "epsilon" not in names:
            raise ValueError(f"Missing epsilon coordinate: {posterior_path}")
        epsilon = samples[:, names.index("epsilon")]
        if np.ptp(epsilon[weights > 0]) == 0:
            value = float(epsilon[weights > 0][0])
            result["epsilon"] = {"q16": value, "median": value, "q84": value}
        else:
            # Use the project's existing weighted GetDist percentile estimator.
            constraints = compute_getdist_constraints(
                epsilon[:, None], weights, loglikes, ("epsilon",),
            )["epsilon"]
            result["epsilon"] = {
                "q16": constraints["lower_68"], "median": constraints["median"],
                "q84": constraints["upper_68"],
            }
        e = result["epsilon"]
        if not e["q16"] <= e["median"] <= e["q84"]:
            raise ValueError(f"Invalid weighted epsilon quantiles: {posterior_path}")
    return result


