"""Prepare the calibrated distance residuals displayed in Figure 4.

All fits use the repository likelihoods. The 18-bin compression is only a
display operation: its full covariance, including common calibrations, is
retained in the returned record. The BAO reference geometry is held fixed.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar

from src.background import Background, c_kms
from src.calibration import MB_CALIBRATION, RD_CALIBRATION, pantheon_and_bao
from src.profile_fits import fit
from src.sampler import PRIOR_BOUNDS


def binning_operator(redshifts, covariance_cholesky, number_of_bins=18):
    """Return simultaneous GLS binning, covariance, and redshift positions.

    Whiten the bin indicator matrix with the existing likelihood Cholesky
    factor; this avoids forming the large supernova precision matrix.
    """
    edges = np.quantile(redshifts, np.linspace(0, 1, number_of_bins + 1))
    membership = np.searchsorted(edges[1:-1], redshifts, side="right")
    counts = np.bincount(membership, minlength=number_of_bins)
    if np.any(counts == 0):
        raise ValueError("Equal-population redshift binning produced an empty bin")
    indicators = np.eye(number_of_bins)[membership]
    whitened = solve_triangular(covariance_cholesky, indicators, lower=True)
    normal_matrix = whitened.T @ whitened
    factor = cho_factor(normal_matrix)
    covariance = cho_solve(factor, np.eye(number_of_bins))
    # W B = L_SN^{-T} L_SN^{-1} B, where L_SN factors the SN covariance.
    precision_indicators = solve_triangular(
        covariance_cholesky.T, whitened, lower=False,
    )
    operator = cho_solve(factor, precision_indicators.T)
    positions = np.array([
        np.exp(np.mean(np.log(redshifts[membership == index])))
        for index in range(number_of_bins)
    ])
    return operator, covariance, positions, edges, counts


def prepare_residuals():
    """Fit the BAO reference and joint tilted model, then prepare both series."""
    sn, bao = pantheon_and_bao()
    rd, sigma_rd = RD_CALIBRATION
    mb, sigma_mb = MB_CALIBRATION

    reference_fit = minimize_scalar(
        lambda omega: float(bao.profile_chi2(Background(Omega_m=omega))),
        bounds=PRIOR_BOUNDS["Omega_m"], method="bounded",
        options={"xatol": 1e-10},
    )
    if not reference_fit.success:
        raise RuntimeError("BAO reference fit failed")
    reference_omega = float(reference_fit.x)
    amplitude, _ = bao.conditional_b(Background(Omega_m=reference_omega))
    reference_h0 = float(c_kms / (amplitude * rd))
    reference = Background(Omega_m=reference_omega, H0=reference_h0)
    reference_moduli = 25 + 5 * np.log10(reference.d_L(sn.z_cmb, sn.z_hel))
    sn_residuals = sn.mb_obs - mb - reference_moduli

    operator, bin_covariance, positions, edges, counts = binning_operator(sn.z_cmb, sn.L)
    sn_display_covariance = bin_covariance + sigma_mb**2

    joint_fit = fit(sn, bao, cpl=False, free_epsilon=True)
    joint = Background(Omega_m=float(joint_fit.geometry[0]), epsilon=joint_fit.epsilon)
    model_shape = joint.sn_log_shape(sn.z_cmb, sn.z_hel)
    whitened_residual = solve_triangular(sn.L, sn.mb_obs - model_shape, lower=True)
    intercept = float(sn.offset_direction @ whitened_residual / sn.offset_norm)
    model_residuals = intercept + model_shape - mb - reference_moduli

    transverse = bao.transverse
    bao_redshifts = bao.z_obs[transverse]
    bao_distances = bao.data_vector[transverse]
    bao_residuals = 5 * np.log10(rd * bao_distances / reference.D_M(bao_redshifts))
    bao_covariance = bao.L @ bao.L.T
    transverse_covariance = bao_covariance[np.ix_(transverse, transverse)]
    log_derivatives = 5 / (np.log(10) * bao_distances)
    bao_display_covariance = (
        transverse_covariance * np.outer(log_derivatives, log_derivatives)
        + (5 / np.log(10) * sigma_rd / rd)**2
    )

    return {
        "reference": {"Omega_m": reference_omega, "H0": reference_h0,
                      "b": float(amplitude), "chi2_bao": float(reference_fit.fun)},
        "calibrations": {"r_d_Mpc": rd, "sigma_r_d_Mpc": sigma_rd,
                         "M_B_mag": mb, "sigma_M_B_mag": sigma_mb},
        "joint_fit": {"Omega_m": float(joint_fit.geometry[0]),
                      "epsilon": joint_fit.epsilon, "nu": intercept,
                      "chi2": joint_fit.chi2},
        "supernova": {
            "entries": len(sn.z_cmb), "redshift_selection": "zHD >= 0.01, IS_CALIBRATOR == 0",
            "redshift": positions.tolist(), "bin_edges": edges.tolist(),
            "bin_counts": counts.tolist(), "residual_mag": (operator @ sn_residuals).tolist(),
            "model_residual_mag": (operator @ model_residuals).tolist(),
            "covariance_mag2": sn_display_covariance.tolist(),
        },
        "bao": {"quantity": "DM_over_rs", "redshift": bao_redshifts.tolist(),
                "residual_mag": bao_residuals.tolist(),
                "covariance_mag2": bao_display_covariance.tolist()},
        "uncertainty": "Diagonal standard deviations of correlated estimates; common M_B/r_d uncertainties included; reference geometry and H0 fixed",
        "binning": "18 equal-population bins, simultaneous generalized least squares; geometric-mean redshifts; same operator applied to model",
    }
