"""Check data bytes and distinguish published posteriors from checkpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_relative(path):
    """Portable provenance path: relative to the repository root when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    """Inverse of repo_relative for paths read back from provenance records."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def verify_inputs():
    for line in (ROOT / "data/SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Observational input checksum mismatch: {name}")


def portable_signature(value, *, archived=False):
    """Input hashes identify relocated data; archived source hashes stay recorded."""
    if isinstance(value, dict):
        return {k: portable_signature(v, archived=archived) for k, v in value.items()
                if not k.endswith("_path") and not (archived and k == "source_sha256")}
    if isinstance(value, list):
        return [portable_signature(v, archived=archived) for v in value]
    return value


def signatures_match(saved, expected, *, archived=False):
    return portable_signature(saved, archived=archived) == portable_signature(expected, archived=archived)


def validate_completed(directory, expected=None):
    """Validate a completed run without requiring a resumable HDF5 checkpoint.

    A source hash change is accepted only for byte-identical release archives.
    New interrupted runs still require their original compatible checkpoint.
    """
    directory = Path(directory)
    if expected is not None:
        verify_inputs()
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("successful") is not True:
        raise ValueError(f"Run is not successfully completed: {directory}")
    manifest = json.loads((ROOT / "results/ARCHIVED_RUNS.json").read_text())
    archived = False
    try:
        names = [str((directory / n).resolve().relative_to(ROOT))
                 for n in ("summary.json", "posterior.npz")]
        archived = all(name in manifest and sha256(ROOT / name) == manifest[name] for name in names)
        # An explicitly listed archive must never silently become a different run.
        if all(name in manifest for name in names) and not archived:
            raise ValueError(f"Published posterior checksum mismatch: {directory}")
    except ValueError as exc:
        if directory.resolve().is_relative_to(ROOT):
            raise exc
    if expected is not None and not signatures_match(summary.get("resume_signature"), expected, archived=archived):
        raise ValueError(f"Completed run does not match requested configuration: {directory}")
    with np.load(directory / "posterior.npz", allow_pickle=False) as p:
        samples, weights, loglikes = p["samples"], p["weights"], p["log_likelihoods"]
        names = p["parameter_names"].tolist()
        wanted = summary.get("posterior_parameters", summary["parameters"])
        if (names != wanted or samples.ndim != 2 or samples.shape[1] != len(names)
                or len(samples) == 0 or weights.shape != (len(samples),) or loglikes.shape != weights.shape
                or not np.isfinite(samples).all() or not np.isfinite(weights).all()
                or not np.isfinite(loglikes).all() or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError(f"Invalid posterior arrays: {directory}")
        if summary["minimum_chi2"] > -2 * loglikes.max() + 1e-6:
            raise ValueError(f"Optimized minimum exceeds the best sampled draw: {directory}")
    return summary
