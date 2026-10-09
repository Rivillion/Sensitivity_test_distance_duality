"""Independent profile fits for the DESI DR2 BAO and supernova checks.

This module implements the publication likelihood profiles.
It profiles the BAO amplitude and Pantheon+ offset analytically, profiles the
SN epsilon coefficient where applicable, and minimizes only the remaining
cosmological parameters.  Each BAO deletion is made from the covariance
matrix, after which the retained principal submatrix is Cholesky factored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from src.background import Background
from src.likelihoods import DESI_BAO, PantheonPlus
from src.sampler import PRIOR_BOUNDS


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BLOCKS: dict[str, tuple[int, ...]] = {
    "None (all 13)": (),
    "BGS": (0,),
    "LRG1": (1, 2),
    "LRG2": (3, 4),
    "LRG3+ELG1": (5, 6),
    "ELG2": (7, 8),
    "QSO": (9, 10),
    "Lyalpha": (11, 12),
}

@dataclass(frozen=True)
class Fit:
    chi2: float
    geometry: np.ndarray
    epsilon: float


def configure_tilt(sn: object, kind: str) -> None:
    """Set the linear magnitude perturbation used by the SN profile.

    ``log`` is 5 epsilon log10(1+z), ``linear`` is kappa*z, and
    ``saturating`` is lambda*z/(1+z). The same intercept is reprofiled.
    """
    z = np.asarray(sn.z_cmb)
    directions = {
        "log": 5.0 * np.log10(1.0 + z),
        "linear": z,
        "saturating": z / (1.0 + z),
    }
    if kind not in directions:
        raise ValueError(f"Unknown tilt kind: {kind}")
    sn.tilt_redshift_term = directions[kind]
    if hasattr(sn, "L"):
        sn.tilt_direction = solve_triangular(
            sn.L, sn.tilt_redshift_term, lower=True, check_finite=False
        )
    else:
        sn.tilt_direction = sn.precision_cholesky.T @ sn.tilt_redshift_term


def subset_bao(dropped: tuple[int, ...]) -> DESI_BAO:
    """Construct a DESI likelihood from a retained covariance submatrix."""
    mean_path = DATA / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    covariance = np.loadtxt(DATA / "desi_gaussian_bao_ALL_GCcomb_cov.txt")
    frame = pd.read_csv(
        mean_path,
        sep=r"\s+",
        comment="#",
        names=["z", "value", "quantity"],
    )
    keep = np.ones(len(frame), dtype=bool)
    keep[list(dropped)] = False
    retained = np.flatnonzero(keep)
    selected_covariance = covariance[np.ix_(retained, retained)]

    likelihood = DESI_BAO.__new__(DESI_BAO)
    likelihood.amplitude_mode = "profile"
    likelihood.z_obs = frame["z"].to_numpy()[retained]
    likelihood.data_vector = frame["value"].to_numpy()[retained]
    likelihood.quantities = frame["quantity"].to_numpy()[retained]
    likelihood.L = cholesky(selected_covariance, lower=True)
    likelihood.whitened_data = solve_triangular(
        likelihood.L,
        likelihood.data_vector,
        lower=True,
        check_finite=False,
    )
    likelihood.whitened_data_norm = float(
        np.dot(likelihood.whitened_data, likelihood.whitened_data)
    )
    likelihood.transverse = likelihood.quantities == "DM_over_rs"
    likelihood.radial = likelihood.quantities == "DH_over_rs"
    likelihood.volume_averaged = likelihood.quantities == "DV_over_rs"
    return likelihood


def total_chi2(
    geometry: np.ndarray,
    sn: PantheonPlus,
    bao: DESI_BAO,
    *,
    cpl: bool,
    free_epsilon: bool,
    fixed_epsilon: float = 0.0,
    return_epsilon: bool = False,
    curvature_on: bool = False,
    tilt_bounds: tuple[float, float] | None = None,
    cc: object | None = None,
) -> float | tuple[float, float]:
    omega_m = float(geometry[0])
    index = 1
    omega_k = float(geometry[index]) if curvature_on else 0.0
    index += int(curvature_on)
    w0, wa = (
        (float(geometry[index]), float(geometry[index + 1]))
        if cpl else (-1.0, 0.0)
    )
    try:
        background = Background(
            Omega_m=omega_m,
            w0=w0,
            wa=wa,
            epsilon=fixed_epsilon,
            alpha=1.0,
            Omega_k=omega_k,
        )
        sn_chi2, epsilon = sn.profiled_linear_nuisances(
            background,
            profile_epsilon=free_epsilon,
            epsilon_bounds=(
                tilt_bounds if tilt_bounds is not None else PRIOR_BOUNDS["epsilon"]
            ) if free_epsilon else None,
        )
        value = float(sn_chi2 + bao.profile_chi2(background))
        if cc is not None:
            value += float(cc.profile_chi2(background))
    except (FloatingPointError, OverflowError, RuntimeError, ValueError):
        value, epsilon = np.inf, np.nan
    return (value, float(epsilon)) if return_epsilon else value


def fit(
    sn: PantheonPlus,
    bao: DESI_BAO,
    *,
    cpl: bool,
    free_epsilon: bool,
    curvature_on: bool = False,
    tilt_bounds: tuple[float, float] | None = None,
    cc: object | None = None,
) -> Fit:
    bounds = [PRIOR_BOUNDS["Omega_m"]]
    starts = [np.array([0.30])]
    if curvature_on:
        bounds += [PRIOR_BOUNDS["Omega_k"]]
        starts = [np.array([0.30, 0.0])]
    if cpl:
        bounds += [PRIOR_BOUNDS["w0"], PRIOR_BOUNDS["wa"]]
        cpl_starts = [
            np.array([0.30, -1.0, 0.0]),
            np.array([0.22, -0.65, -1.2]),
            np.array([0.42, -1.25, 0.8]),
            np.array([0.10001, -0.8, -1.0]),
            np.array([0.49999, -1.0, 0.0]),
        ]
        starts = [
            np.insert(point, 1, 0.0) if curvature_on else point
            for point in cpl_starts
        ]

    objective = lambda x: total_chi2(
        x,
        sn,
        bao,
        cpl=cpl,
        free_epsilon=free_epsilon,
        curvature_on=curvature_on,
        tilt_bounds=tilt_bounds,
        cc=cc,
    )
    candidates: list[np.ndarray] = starts.copy()
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            bounds=bounds,
            options={"xtol": 2e-9, "ftol": 2e-12, "maxiter": 1000},
        )
        if np.all(np.isfinite(result.x)):
            candidates.append(np.asarray(result.x))

    # Explicitly test the Omega_m faces.  This is important for the Lyalpha
    # deletion, whose CPL likelihood has an almost flat direction ending on
    # the lower prior boundary; a generic bounded Powell pass can stop at an
    # indistinguishable interior point.
    if cpl and not curvature_on:
        for omega_edge in (bounds[0][0], bounds[0][1]):
            for start in starts[:4]:
                result = minimize(
                    lambda dark_energy: objective(
                        np.array([omega_edge, *dark_energy])
                    ),
                    start[1:],
                    method="Powell",
                    bounds=bounds[1:],
                    options={"xtol": 2e-9, "ftol": 2e-12, "maxiter": 1000},
                )
                if np.all(np.isfinite(result.x)):
                    candidates.append(
                        np.array([omega_edge, *np.asarray(result.x)])
                    )

    best = min(candidates, key=objective)
    # A final local pass makes comparisons insensitive to the winning start.
    refined = minimize(
        objective,
        best,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 2e-9, "fatol": 2e-10, "maxiter": 2500},
    )
    if np.isfinite(refined.fun) and refined.fun < objective(best):
        best = np.asarray(refined.x)
    chi2, epsilon = total_chi2(
        best,
        sn,
        bao,
        cpl=cpl,
        free_epsilon=free_epsilon,
        return_epsilon=True,
        curvature_on=curvature_on,
        tilt_bounds=tilt_bounds,
        cc=cc,
    )
    if not np.isfinite(chi2) or any(
        coordinate < low - 1e-8 or coordinate > high + 1e-8
        for coordinate, (low, high) in zip(best, bounds, strict=True)
    ):
        raise RuntimeError("Profile search did not return a valid bounded minimum")
    return Fit(float(chi2), best, epsilon)
