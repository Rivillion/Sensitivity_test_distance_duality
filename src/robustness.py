"""Select one robustness dataset and delegate to the shared sampler.


This driver supplies the dataset selection, output directory, and expanded
checkpoint signature. Every likelihood call uses the original full covariance
residual and all four parameters are sampled explicitly.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from src import sampler

ROOT = Path(__file__).resolve().parent.parent


def input_hashes(job):
    files = (["Pantheon+SH0ES.dat", "Pantheon+SH0ES_STAT+SYS.cov"]
             if job["sample"] == "pantheon" else ["DES-Dovekie_HD.csv", "STAT+SYS.npz"])
    files += ["desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt"]
    return {name: hashlib.sha256((ROOT/"data"/name).read_bytes()).hexdigest() for name in files}


def signature(job):
    result = sampler.resume_signature(job["model"], job["likelihoods"], 3000, True,
                                      zmin=job["zmin"])
    result["robustness_selection"] = dict(version=1, case_id=job["case_id"],
         bao_drop_indices=job["bao_drop_indices"], n_sn=job["n_sn"], n_bao=job["n_bao"],
         input_sha256=input_hashes(job))
    return result


def likelihood_for_job(job):
    likelihood = sampler.initialize_likelihoods(job["likelihoods"], zmin=job["zmin"])
    sn, bao = likelihood.likelihoods
    if job["bao_drop_indices"]:
        # Marginalize the deleted observations by retaining the principal
        # submatrix of the COVARIANCE, then factor it again. Never slice a
        # precision matrix or the existing Cholesky factor.
        covariance = np.loadtxt(ROOT/"data/desi_gaussian_bao_ALL_GCcomb_cov.txt")
        keep = np.ones(len(bao.z_obs), dtype=bool)
        keep[job["bao_drop_indices"]] = False
        retained = np.flatnonzero(keep)
        bao.z_obs = bao.z_obs[retained]
        bao.data_vector = bao.data_vector[retained]
        bao.quantities = bao.quantities[retained]
        bao.L = np.asfortranarray(cholesky(covariance[np.ix_(retained, retained)], lower=True))
        bao.whitened_data = solve_triangular(bao.L, bao.data_vector, lower=True, check_finite=False)
        bao.whitened_data_norm = float(bao.whitened_data@bao.whitened_data)
        bao.transverse = bao.quantities == "DM_over_rs"
        bao.radial = bao.quantities == "DH_over_rs"
        bao.volume_averaged = bao.quantities == "DV_over_rs"
    if len(sn.z_cmb) != job["n_sn"] or len(bao.z_obs) != job["n_bao"]:
        raise ValueError(f"Sample counts do not match manuscript for {job['case_id']}")
    return likelihood


def arguments(job, cores):
    return sampler.parse_arguments(["--model", "lcdm-epsilon", "--likelihoods", *job["likelihoods"],
        "--zmin", str(job["zmin"]), "--cores", str(cores), "--nlive", "3000",
        "--seed", str(job["seed"]), "--discard-exploration"])[1]


def run(job, cores, output_root=None, fresh=False):
    sampler.require_runtime_dependencies(save=True)
    args = arguments(job, cores)
    args.fresh = fresh
    directory = (Path(output_root) / job["case_id"]) if output_root else ROOT/job["output"]
    likelihood = likelihood_for_job(job)
    expected_signature = signature(job)
    # The upstream function uses module-level dataset and persistence hooks.
    # Override these hooks only inside this process and restore them on exit;
    # the full likelihood and prior objects remain the upstream implementations.
    with patch.object(sampler, "initialize_likelihoods", return_value=likelihood), \
         patch.object(sampler, "run_directory", return_value=directory), \
         patch.object(sampler, "resume_signature", return_value=expected_signature):
        return sampler.run_sampler(args)


