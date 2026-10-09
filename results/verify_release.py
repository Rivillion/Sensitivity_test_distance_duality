"""Verify inputs, 54 completed posteriors, and saved likelihood evaluations."""
import argparse
import json
from pathlib import Path

import numpy as np

from src import robustness, sampler
from src.release_io import sha256, validate_completed, verify_inputs
from src.sample_section4b import CASES

ROOT = Path(__file__).resolve().parent.parent


def check_likelihood(directory, summary, likelihood):
    with np.load(directory / "posterior.npz", allow_pickle=False) as p:
        names, samples, loglikes = p["parameter_names"].tolist(), p["samples"], p["log_likelihoods"]
        indices = np.unique([0, len(samples) // 2, len(samples) - 1, int(loglikes.argmax())])
        parameters = dict(zip(names, samples[indices].T, strict=True))
        error = float(np.max(np.abs(likelihood(parameters) - loglikes[indices])))
    minimum_error = abs(-2 * likelihood(summary["maximum_likelihood_parameters"]) - summary["minimum_chi2"])
    if error > 1e-7 or minimum_error > 1e-7:
        raise ValueError(f"Saved likelihood evaluations do not reproduce: {directory}: {error}, {minimum_error}")
    return {"checked_draw_loglike_error": error, "minimum_chi2_error": float(minimum_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums-only", action="store_true")
    args = parser.parse_args()
    verify_inputs()
    manifest = json.loads((ROOT / "results/ARCHIVED_RUNS.json").read_text())
    for name, digest in manifest.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Published run checksum mismatch: {name}")
    count, max_error = 0, 0.0
    for kind, expected_count in (("baseline", 38), ("robustness", 13)):
        jobs = json.loads((ROOT / f"results/jobs_{kind}.json").read_text())
        if len(jobs) != expected_count or [j["task_id"] for j in jobs] != list(range(expected_count)):
            raise ValueError(f"Incomplete {kind} inventory")
        for job in jobs:
            directory = ROOT / job["output"]
            summary = validate_completed(directory)
            if summary["nlive"] != 3000 or summary["discard_exploration"] is not True:
                raise ValueError(f"Incorrect sampling configuration: {directory}")
            if not args.checksums_only:
                likelihood = (robustness.likelihood_for_job(job) if kind == "robustness"
                              else sampler.initialize_likelihoods(job["likelihoods"]))
                check = check_likelihood(directory, summary, likelihood)
                max_error = max(max_error, *check.values())
            count += 1
    for case, (model, datasets, _) in CASES.items():
        directory = ROOT / "results/runs/calibration" / case / model
        summary = validate_completed(directory)
        if summary["nlive"] != 3000 or summary["discard_exploration"] is not True:
            raise ValueError(f"Incorrect calibration configuration: {directory}")
        if not args.checksums_only:
            check = check_likelihood(directory, summary, sampler.initialize_likelihoods(datasets, zmin=.01))
            max_error = max(max_error, *check.values())
        count += 1
    print(f"Verified input checksums and {count} published posteriors; maximum likelihood error {max_error:.3g}.")


if __name__ == "__main__":
    main()
