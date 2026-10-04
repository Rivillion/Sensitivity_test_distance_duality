import os
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.linalg import cholesky, cho_solve, solve_triangular  # type: ignore
from scipy.linalg.blas import dsymm, dsymv  # type: ignore
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _sn_base_shape(background, z_cmb, standard_redshift_term):
    """Evaluate the epsilon-independent SN distance shape."""
    transverse_distance = np.asarray(
        background.D_M_dimensionless(z_cmb),
        dtype=np.float64,
    )
    if np.any(transverse_distance <= 0.0):
        raise ValueError("SN redshifts must give a positive D_M(z).")

    distance_term = 5.0 * np.log10(transverse_distance)
    if transverse_distance.ndim == 1:
        return distance_term + standard_redshift_term
    return distance_term + standard_redshift_term[None, :]


def _sn_log_shape(
    background,
    z_cmb,
    standard_redshift_term,
    tilt_redshift_term,
):
    """Evaluate the SN shape with distinct HEL and HD redshift terms."""
    base_shape = _sn_base_shape(background, z_cmb, standard_redshift_term)

    epsilon = np.asarray(background.epsilon, dtype=np.float64)
    if base_shape.ndim == 1:
        return base_shape + epsilon * tilt_redshift_term
    return base_shape + epsilon[:, None] * tilt_redshift_term[None, :]


def _profile_whitened_sn(
    whitened_residual,
    offset_direction,
    offset_norm,
    tilt_direction,
    *,
    profile_epsilon,
    epsilon,
    epsilon_bounds,
):
    """Profile a scalar SN residual over its offset and optionally epsilon."""
    projected_residual = whitened_residual - (
        np.dot(whitened_residual, offset_direction)
        / offset_norm
        * offset_direction
    )
    projected_tilt = tilt_direction - (
        np.dot(tilt_direction, offset_direction)
        / offset_norm
        * offset_direction
    )

    if profile_epsilon:
        tilt_norm = np.dot(projected_tilt, projected_tilt)
        if tilt_norm <= 0.0:
            raise ValueError("SN epsilon direction must be non-degenerate.")
        epsilon = np.dot(projected_tilt, projected_residual) / tilt_norm
        if epsilon_bounds is not None:
            epsilon = float(np.clip(epsilon, *epsilon_bounds))

    profiled_residual = projected_residual - epsilon * projected_tilt
    chi2 = max(float(np.dot(profiled_residual, profiled_residual)), 0.0)
    return chi2, float(epsilon)


def _explicit_sn_log_likelihood(self, background):
    """Full manuscript intercept likelihood; no offset projection."""
    nu = getattr(background, "nu", None)
    if nu is None:
        raise ValueError("Explicit SN likelihood requires nu.")
    shape = _sn_log_shape(background, self.z_cmb,
                          self.standard_redshift_term, self.tilt_redshift_term)
    intercept = np.asarray(nu, dtype=np.float64)
    if np.any(~np.isfinite(intercept)):
        raise ValueError("nu must be finite.")
    if shape.ndim == 1:
        if intercept.ndim != 0:
            raise ValueError("Scalar SN evaluation requires scalar nu.")
        prediction = shape + intercept
    else:
        if intercept.shape != (shape.shape[0],):
            raise ValueError("Vectorized nu must match the batch size.")
        prediction = shape + intercept[:, None]
    observed = self.mb_obs if hasattr(self, "mb_obs") else self.mu_obs
    residual = observed - prediction
    if hasattr(self, "L"):
        whitened = solve_triangular(self.L, residual if residual.ndim == 1 else residual.T,
                                   lower=True, check_finite=False)
    else:
        whitened = self.precision_cholesky.T @ (residual if residual.ndim == 1 else residual.T)
    return -0.5 * np.sum(whitened**2, axis=0)


def _validated_zmin(zmin):
    """Return a finite, non-negative supernova redshift cutoff."""
    value = float(zmin)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("zmin must be finite and non-negative.")
    return value


class PantheonPlus:
    """Likelihood for the Pantheon+ Type Ia supernova sample.

    Cepheid calibrators are excluded. The remaining constant magnitude
    calibration is marginalized analytically.
    """

    M_FIDUCIAL = -19.253
    supports_profiled_best_fit = True
    profiles_epsilon = True

    explicit_log_likelihood = _explicit_sn_log_likelihood

    def __init__(
        self,
        data_path,
        cov_path,
        M_fiducial=M_FIDUCIAL,
        zmin=0.01,
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        if not os.path.exists(cov_path):
            raise FileNotFoundError(f"File not found: {cov_path}")

        self.zmin = _validated_zmin(zmin)
        data = pd.read_csv(data_path, sep=r"\s+")
        hubble_flow = (
            (data["zHD"].to_numpy() >= self.zmin)
            & (data["IS_CALIBRATOR"].to_numpy() == 0)
        )

        self.mask = np.flatnonzero(hubble_flow)
        if self.mask.size == 0:
            raise ValueError(
                f"Pantheon+ has no supernovae at zHD >= {self.zmin}."
            )
        self.z_cmb = data["zHD"].to_numpy()[self.mask]
        self.z_hel = data["zHEL"].to_numpy()[self.mask]
        self.mb_obs = data["m_b_corr"].to_numpy()[self.mask]
        self.M_fiducial = float(M_fiducial)

        raw_covariance = np.loadtxt(cov_path)
        n_points = int(raw_covariance[0])
        if n_points != len(data):
            raise ValueError(
                "Pantheon+ covariance dimension does not match the data: "
                f"{n_points} != {len(data)}."
            )
        if raw_covariance.size != 1 + n_points**2:
            raise ValueError("Pantheon+ covariance file has an invalid size.")
        full_covariance = raw_covariance[1:].reshape(
            (n_points, n_points)
        )
        covariance = full_covariance[np.ix_(self.mask, self.mask)]

        self.L = np.asfortranarray(cholesky(covariance, lower=True))

        ones = np.ones_like(self.mb_obs)
        self.offset_direction = solve_triangular(
            self.L,
            ones,
            lower=True,
            check_finite=False,
        )
        self.offset_norm = np.dot(
            self.offset_direction,
            self.offset_direction,
        )
        self.standard_redshift_term = 5.0 * np.log10(1.0 + self.z_hel)
        self.tilt_redshift_term = 5.0 * np.log10(1.0 + self.z_cmb)
        self.tilt_direction = solve_triangular(
            self.L,
            self.tilt_redshift_term,
            lower=True,
            check_finite=False,
        )

        print(
            "Pantheon+ Likelihood initialized. "
            f"Total SN: {len(self.mask)}"
        )

    def profiled_linear_nuisances(
        self,
        background,
        *,
        profile_epsilon=False,
        epsilon_bounds=None,
    ):
        """Return chi-square, profiling nu and optionally the linear tilt."""
        model_magnitude = (
            _sn_base_shape(
                background,
                self.z_cmb,
                self.standard_redshift_term,
            )
            + self.M_fiducial
        )
        whitened_residual = solve_triangular(
            self.L,
            self.mb_obs - model_magnitude,
            lower=True,
            check_finite=False,
        )
        return _profile_whitened_sn(
            whitened_residual,
            self.offset_direction,
            self.offset_norm,
            self.tilt_direction,
            profile_epsilon=profile_epsilon,
            epsilon=background.epsilon,
            epsilon_bounds=epsilon_bounds,
        )

    def log_likelihood(self, background):
        if getattr(self, "calibration_mode", "profile") == "explicit":
            return self.explicit_log_likelihood(background)
        model_magnitude = (
            # background.sn_log_shape(self.z_cmb, self.z_hel)
            _sn_log_shape(
                background,
                self.z_cmb,
                self.standard_redshift_term,
                self.tilt_redshift_term,
            )
            + self.M_fiducial
        )
        residual = self.mb_obs - model_magnitude

        if residual.ndim == 1:
            whitened_residual = solve_triangular(
                self.L,
                residual,
                lower=True,
                check_finite=False,
            )
            offset_projection = np.dot(
                whitened_residual,
                self.offset_direction,
            )
            residual_norm = np.dot(
                whitened_residual,
                whitened_residual,
            )
        else:
            whitened_residual = solve_triangular(
                self.L,
                residual.T,
                lower=True,
                check_finite=False,
            ).T
            offset_projection = (
                whitened_residual @ self.offset_direction
            )
            residual_norm = np.einsum(
                "ij,ij->i",
                whitened_residual,
                whitened_residual,
            )

        chi2 = residual_norm - offset_projection**2 / self.offset_norm
        chi2 = np.maximum(chi2, 0.0)

        return -0.5 * chi2


class DES_Dovekie:
    """Likelihood for the DES-SN5YR Dovekie sample.

    The supplied file contains the inverse covariance. The remaining
    constant magnitude calibration is marginalized analytically.
    """

    M_FIDUCIAL = -19.253
    supports_profiled_best_fit = True
    profiles_epsilon = True

    explicit_log_likelihood = _explicit_sn_log_likelihood

    def __init__(
        self,
        data_path,
        cov_path,
        M_fiducial=M_FIDUCIAL,
        zmin=0.0,
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        if not os.path.exists(cov_path):
            raise FileNotFoundError(f"File not found: {cov_path}")

        self.zmin = _validated_zmin(zmin)
        data = pd.read_csv(data_path, comment="#", sep=r"\s+")
        self.mask = np.flatnonzero(
            data["zHD"].to_numpy() >= self.zmin
        )
        if self.mask.size == 0:
            raise ValueError(
                "DES-Dovekie has no supernovae at "
                f"zHD >= {self.zmin}."
            )
        self.z_cmb = data["zHD"].to_numpy()[self.mask]
        self.z_hel = data["zHEL"].to_numpy()[self.mask]
        self.mu_obs = data["MU"].to_numpy()[self.mask]
        self.M_fiducial = float(M_fiducial)

        with np.load(cov_path) as covariance_data:
            if "nsn" not in covariance_data or "cov" not in covariance_data:
                raise ValueError(
                    "DES-Dovekie covariance archive must contain nsn and cov."
                )
            n_points = int(covariance_data["nsn"][0])
            upper_triangle = np.asarray(
                covariance_data["cov"], dtype=np.float64
            )

        if n_points != len(data):
            raise ValueError(
                "DES-Dovekie covariance dimension does not match the data: "
                f"{n_points} != {len(data)}."
            )
        expected_triangle_size = n_points * (n_points + 1) // 2
        if upper_triangle.size != expected_triangle_size:
            raise ValueError(
                "DES-Dovekie inverse-covariance triangle has an invalid size."
            )

        inverse_covariance = np.zeros((n_points, n_points))
        upper_indices = np.triu_indices(n_points)
        inverse_covariance[upper_indices] = upper_triangle

        lower_indices = np.tril_indices(n_points, -1)
        inverse_covariance[lower_indices] = inverse_covariance.T[
            lower_indices
        ]

        precision_factor = cholesky(
            inverse_covariance,
            lower=True,
            check_finite=True,
        )
        full_covariance = cho_solve(
            (precision_factor, True),
            np.eye(n_points),
            check_finite=False,
        )
        full_covariance = 0.5 * (
            full_covariance + full_covariance.T
        )
        covariance = full_covariance[np.ix_(self.mask, self.mask)]
        self.L = np.asfortranarray(cholesky(covariance, lower=True))

        ones = np.ones_like(self.mu_obs)
        self.offset_direction = solve_triangular(
            self.L,
            ones,
            lower=True,
            check_finite=False,
        )
        self.offset_norm = np.dot(
            self.offset_direction,
            self.offset_direction,
        )
        self.standard_redshift_term = 5.0 * np.log10(1.0 + self.z_hel)
        self.tilt_redshift_term = 5.0 * np.log10(1.0 + self.z_cmb)
        self.tilt_direction = solve_triangular(
            self.L,
            self.tilt_redshift_term,
            lower=True,
            check_finite=False,
        )

        print(
            "DES-SN5YR Dovekie Likelihood initialized. "
            f"Total SN: {len(self.z_cmb)}"
        )

    def profiled_linear_nuisances(
        self,
        background,
        *,
        profile_epsilon=False,
        epsilon_bounds=None,
    ):
        """Return chi-square, profiling nu and optionally the linear tilt."""
        model_modulus = (
            _sn_base_shape(
                background,
                self.z_cmb,
                self.standard_redshift_term,
            )
            + self.M_fiducial
        )
        whitened_residual = solve_triangular(
            self.L,
            self.mu_obs - model_modulus,
            lower=True,
            check_finite=False,
        )
        return _profile_whitened_sn(
            whitened_residual,
            self.offset_direction,
            self.offset_norm,
            self.tilt_direction,
            profile_epsilon=profile_epsilon,
            epsilon=background.epsilon,
            epsilon_bounds=epsilon_bounds,
        )

    def log_likelihood(self, background):
        if getattr(self, "calibration_mode", "profile") == "explicit":
            return self.explicit_log_likelihood(background)
        model_modulus = (
            # background.sn_log_shape(self.z_cmb, self.z_hel)
            _sn_log_shape(
                background,
                self.z_cmb,
                self.standard_redshift_term,
                self.tilt_redshift_term,
            )
            + self.M_fiducial
        )
        residual = self.mu_obs - model_modulus

        if residual.ndim == 1:
            whitened_residual = solve_triangular(
                self.L,
                residual,
                lower=True,
                check_finite=False,
            )
            offset_projection = np.dot(
                whitened_residual,
                self.offset_direction,
            )
            residual_norm = np.dot(
                whitened_residual,
                whitened_residual,
            )
        else:
            whitened_residual = solve_triangular(
                self.L,
                residual.T,
                lower=True,
                check_finite=False,
            ).T
            offset_projection = (
                whitened_residual @ self.offset_direction
            )
            residual_norm = np.einsum(
                "ij,ij->i",
                whitened_residual,
                whitened_residual,
            )

        chi2 = residual_norm - offset_projection**2 / self.offset_norm
        chi2 = np.maximum(chi2, 0.0)

        return -0.5 * chi2


class UnionBinned:
    """Calibration-profiled Union3/Union3.1 compressed-distance likelihood.

    The official release stores a simple 23x23 big-endian float64 primary
    FITS image: redshifts in the first row, distance moduli in the first
    column, and the 22x22 inverse covariance in the remaining block.  Reading
    this fixed image directly avoids introducing an Astropy runtime
    dependency for a seven-card FITS header.
    """

    MATRIX_SIZE = 23
    FITS_BLOCK_BYTES = 2880
    supports_profiled_best_fit = True
    profiles_epsilon = True

    explicit_log_likelihood = _explicit_sn_log_likelihood

    def __init__(self, data_path):
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        header = path.read_bytes()[: self.FITS_BLOCK_BYTES]
        required_cards = (
            b"SIMPLE  =                    T",
            b"BITPIX  =                  -64",
            b"NAXIS   =                    2",
            b"NAXIS1  =                   23",
            b"NAXIS2  =                   23",
        )
        if any(card not in header for card in required_cards):
            raise ValueError(
                f"Unsupported Union compressed FITS layout in {path}."
            )
        matrix = np.fromfile(
            path,
            dtype=">f8",
            count=self.MATRIX_SIZE**2,
            offset=self.FITS_BLOCK_BYTES,
        ).astype(np.float64).reshape(
            (self.MATRIX_SIZE, self.MATRIX_SIZE)
        )
        if matrix.shape != (23, 23) or np.any(~np.isfinite(matrix)):
            raise ValueError("Union compressed matrix is invalid.")

        self.z_cmb = matrix[0, 1:].copy()
        self.z_hel = self.z_cmb.copy()
        self.mu_obs = matrix[1:, 0].copy()
        inverse_covariance = matrix[1:, 1:].copy()
        if np.any(self.z_cmb <= 0.0) or np.any(np.diff(self.z_cmb) <= 0.0):
            raise ValueError("Union redshifts must be positive and increasing.")
        if not np.allclose(
            inverse_covariance,
            inverse_covariance.T,
            rtol=1e-12,
            atol=1e-10,
        ):
            raise ValueError("Union inverse covariance is not symmetric.")
        self.inverse_covariance = np.asfortranarray(
            0.5 * (inverse_covariance + inverse_covariance.T)
        )
        self.precision_cholesky = cholesky(
            self.inverse_covariance,
            lower=True,
            check_finite=True,
        )

        self.ones = np.ones_like(self.mu_obs)
        self.precision_ones = dsymv(
            1.0,
            self.inverse_covariance,
            self.ones,
            lower=0,
        )
        self.offset_norm = np.dot(self.ones, self.precision_ones)
        self.standard_redshift_term = 5.0 * np.log10(1.0 + self.z_cmb)
        self.tilt_redshift_term = self.standard_redshift_term.copy()

        print(
            "Union binned likelihood initialized. "
            f"Total bins: {len(self.z_cmb)}"
        )

    def profiled_linear_nuisances(
        self,
        background,
        *,
        profile_epsilon=False,
        epsilon_bounds=None,
    ):
        """Return chi-square, profiling nu and optionally the linear tilt."""
        model_modulus = _sn_base_shape(
            background,
            self.z_cmb,
            self.standard_redshift_term,
        )
        precision_whitener = self.precision_cholesky.T
        return _profile_whitened_sn(
            precision_whitener @ (self.mu_obs - model_modulus),
            precision_whitener @ self.ones,
            self.offset_norm,
            precision_whitener @ self.tilt_redshift_term,
            profile_epsilon=profile_epsilon,
            epsilon=background.epsilon,
            epsilon_bounds=epsilon_bounds,
        )

    def log_likelihood(self, background):
        if getattr(self, "calibration_mode", "profile") == "explicit":
            return self.explicit_log_likelihood(background)
        model_modulus = _sn_log_shape(
            background,
            self.z_cmb,
            self.standard_redshift_term,
            self.tilt_redshift_term,
        )
        residual = self.mu_obs - model_modulus

        if residual.ndim == 1:
            precision_residual = dsymv(
                1.0,
                self.inverse_covariance,
                residual,
                lower=0,
            )
            offset_projection = np.dot(
                self.ones,
                precision_residual,
            )
            residual_norm = np.dot(residual, precision_residual)
        else:
            precision_residual = dsymm(
                1.0,
                self.inverse_covariance,
                np.asfortranarray(residual.T),
                side=0,
                lower=0,
            ).T
            offset_projection = precision_residual @ self.ones
            residual_norm = np.einsum(
                "ij,ij->i",
                residual,
                precision_residual,
            )

        chi2 = residual_norm - offset_projection**2 / self.offset_norm
        return -0.5 * np.maximum(chi2, 0.0)


class DESI_BAO:
    """DESI DR2 BAO likelihood with analytic or shared explicit calibration.

    If the background supplies both ``H0`` and ``r_d``, the physical BAO
    amplitude is ``c/(H0*r_d)``. Otherwise ``amplitude_mode`` selects either
    the profile likelihood or its analytic marginalization, including the
    geometry-dependent conditional-amplitude width. The explicit branch is
    retained for BAO+CC so that both probes share the same Hubble constant.
    """

    C_KMS = 299792.458
    supports_profiled_best_fit = True
    profiles_epsilon = False

    def __init__(self, mean_path, cov_path, amplitude_mode="profile"):
        if amplitude_mode not in ("profile", "marginalized"):
            raise ValueError(
                "amplitude_mode must be 'profile' or 'marginalized'."
            )
        self.amplitude_mode = amplitude_mode
        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"File not found: {mean_path}")

        if not os.path.exists(cov_path):
            raise FileNotFoundError(f"File not found: {cov_path}")

        data = pd.read_csv(
            mean_path,
            sep=r"\s+",
            comment="#",
            names=["z", "value", "quantity"],
        )
        self.z_obs = data["z"].to_numpy()
        self.data_vector = data["value"].to_numpy()
        self.quantities = data["quantity"].to_numpy()

        supported_quantities = {
            "DM_over_rs",
            "DH_over_rs",
            "DV_over_rs",
        }
        unknown_quantities = set(self.quantities) - supported_quantities
        if unknown_quantities:
            raise ValueError(
                f"Unsupported BAO quantities: {sorted(unknown_quantities)}"
            )

        covariance = np.loadtxt(cov_path)
        self.L = cholesky(covariance, lower=True)

        self.whitened_data = solve_triangular(
            self.L,
            self.data_vector,
            lower=True,
            check_finite=False,
        )
        self.whitened_data_norm = np.dot(
            self.whitened_data,
            self.whitened_data,
        )
        self.transverse = self.quantities == "DM_over_rs"
        self.radial = self.quantities == "DH_over_rs"
        self.volume_averaged = self.quantities == "DV_over_rs"

        print(
            "DESI BAO Likelihood initialized. "
            f"Total data points: {len(self.z_obs)}"
        )

    def _theory_vector(self, background):
        transverse_distance = np.asarray(
            background.D_M_dimensionless(self.z_obs)
        )
        expansion_rate = np.asarray(background.E(self.z_obs))

        theory_vector = np.empty_like(
            transverse_distance,
            dtype=np.float64,
        )

        theory_vector[..., self.transverse] = (
            transverse_distance[..., self.transverse]
        )
        theory_vector[..., self.radial] = (
            1.0 / expansion_rate[..., self.radial]
        )

        volume_distance_cubed = (
            self.z_obs[self.volume_averaged]
            * transverse_distance[..., self.volume_averaged] ** 2
            / expansion_rate[..., self.volume_averaged]
        )
        theory_vector[..., self.volume_averaged] = np.cbrt(
            volume_distance_cubed
        )

        return theory_vector

    def quadratic_terms(self, background):
        """Return A=t^T W t, B=t^T W d, and D=d^T W d."""
        theory_vector = self._theory_vector(background)
        if theory_vector.ndim == 1:
            whitened_theory = solve_triangular(
                self.L,
                theory_vector,
                lower=True,
                check_finite=False,
            )
            theory_norm = np.dot(whitened_theory, whitened_theory)
            data_projection = np.dot(
                whitened_theory,
                self.whitened_data,
            )
        else:
            whitened_theory = solve_triangular(
                self.L,
                theory_vector.T,
                lower=True,
                check_finite=False,
            ).T
            theory_norm = np.einsum(
                "ij,ij->i",
                whitened_theory,
                whitened_theory,
            )
            data_projection = whitened_theory @ self.whitened_data

        if np.any(np.asarray(theory_norm) <= 0.0):
            raise ValueError("BAO quadratic coefficient must be positive.")
        return theory_norm, data_projection, self.whitened_data_norm

    def conditional_b(self, background):
        """Return the conditional BAO amplitude and its standard deviation."""
        theory_norm, data_projection, _ = self.quadratic_terms(background)
        return data_projection / theory_norm, 1.0 / np.sqrt(theory_norm)

    def profile_chi2(self, background):
        """Return the BAO chi-square profiled over its linear amplitude."""
        theory_norm, data_projection, data_norm = self.quadratic_terms(background)
        return np.maximum(
            data_norm - data_projection**2 / theory_norm,
            0.0,
        )

    def explicit_log_likelihood(self, background):
        """Evaluate BAO at the sampled product H0rd or shared H0, r_d."""
        product = getattr(background, "H0rd", None)
        if product is None:
            h0, rd = getattr(background, "H0", None), getattr(background, "r_d", None)
            if h0 is None or rd is None:
                raise ValueError("Explicit BAO requires H0rd or both H0 and r_d.")
            product = np.asarray(h0) * np.asarray(rd)
        product = np.asarray(product, dtype=np.float64)
        if np.any(~np.isfinite(product)) or np.any(product <= 0):
            raise ValueError("BAO scale must be finite and positive.")
        theory = self._theory_vector(background)
        if theory.ndim == 1:
            if product.ndim != 0:
                raise ValueError("Scalar BAO requires a scalar scale.")
            residual = self.data_vector - self.C_KMS / product * theory
        else:
            if product.shape != (theory.shape[0],):
                raise ValueError("Vectorized BAO scale must match the batch size.")
            residual = self.data_vector - self.C_KMS / product[:, None] * theory
        whitened = solve_triangular(self.L, residual if residual.ndim == 1 else residual.T,
                                   lower=True, check_finite=False)
        return -0.5 * np.sum(whitened**2, axis=0)

    def log_likelihood(self, background):
        if getattr(self, "calibration_mode", "profile") == "explicit":
            return self.explicit_log_likelihood(background)
        H0 = getattr(background, "H0", None)
        r_d = getattr(background, "r_d", None)
        explicit_scale = H0 is not None or r_d is not None
        if explicit_scale:
            if H0 is None or r_d is None:
                raise ValueError(
                    "Explicit BAO calibration requires both H0 and r_d."
                )
            h0_array = np.asarray(H0, dtype=np.float64)
            rd_array = np.asarray(r_d, dtype=np.float64)
            if (
                np.any(~np.isfinite(h0_array))
                or np.any(~np.isfinite(rd_array))
                or np.any(h0_array <= 0.0)
                or np.any(rd_array <= 0.0)
            ):
                raise ValueError("BAO requires finite positive H0 and r_d.")

        theory_norm, data_projection, data_norm = self.quadratic_terms(
            background
        )

        if explicit_scale:
            scale = self.C_KMS / (h0_array * rd_array)
            if np.ndim(theory_norm) == 0:
                if np.ndim(scale) != 0:
                    raise ValueError(
                        "Scalar BAO evaluation requires scalar H0 and r_d."
                    )
                chi2 = (
                    data_norm
                    - 2.0 * float(scale) * data_projection
                    + float(scale) ** 2 * theory_norm
                )
            else:
                if np.shape(scale) != np.shape(theory_norm):
                    raise ValueError(
                        "Vectorized BAO H0 and r_d must match the batch size."
                    )
                chi2 = (
                    data_norm
                    - 2.0 * scale * data_projection
                    + scale**2 * theory_norm
                )
        else:
            chi2 = data_norm - data_projection**2 / theory_norm
        chi2 = np.maximum(chi2, 0.0)

        log_likelihood = -0.5 * chi2
        if not explicit_scale and self.amplitude_mode == "marginalized":
            log_likelihood = log_likelihood - 0.5 * np.log(theory_norm)

        return log_likelihood


class CosmicChronometer:
    """CC H(z) likelihood using a supplied data table and full covariance.

    The three table columns are z, H(z), and the quoted uncertainty. The
    covariance file is used unchanged, in table row order: no diagonal
    errors, population systematics, jitter, or normalization are added.
    Sampling uses explicit H0. Profile and marginalized modes are retained
    only for independent reference calculations.
    """

    def __init__(self, data_path, cov_path, *, h0_mode="explicit"):
        if h0_mode not in ("explicit", "profile", "marginalized"):
            raise ValueError("h0_mode must be 'explicit', 'profile', or 'marginalized'.")
        self.data_path = Path(data_path)
        self.covariance_path = Path(cov_path)
        table = np.loadtxt(self.data_path, dtype=np.float64, ndmin=2)
        if table.ndim != 2 or table.shape[1] != 3 or len(table) == 0:
            raise ValueError("CC table must contain three columns: z, H(z), uncertainty.")
        if np.any(~np.isfinite(table)):
            raise ValueError("CC data must be finite.")
        self.z_obs, self.H_obs, self.err_hz = table.T.copy()
        if np.any(self.z_obs < 0) or np.any(np.diff(self.z_obs) <= 0):
            raise ValueError("CC redshifts must be non-negative and strictly increasing.")
        if np.any(self.H_obs <= 0) or np.any(self.err_hz <= 0):
            raise ValueError("CC H(z) and quoted uncertainties must be positive.")
        self.covariance = np.loadtxt(self.covariance_path, dtype=np.float64, ndmin=2)
        expected = (len(table), len(table))
        if self.covariance.shape != expected:
            raise ValueError(f"CC covariance has shape {self.covariance.shape}; expected {expected}.")
        if np.any(~np.isfinite(self.covariance)):
            raise ValueError("CC covariance must be finite.")
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-12, atol=1e-10):
            raise ValueError("CC covariance must be symmetric.")
        self.L = np.asfortranarray(cholesky(self.covariance, lower=True, check_finite=True))
        self.h0_mode = h0_mode
        self.whitened_H = solve_triangular(self.L, self.H_obs, lower=True, check_finite=False)
        self.D = float(np.dot(self.whitened_H, self.whitened_H))
    def _expansion_rate(self, background: Any) -> FloatArray:
        expansion_rate = np.asarray(
            background.E(self.z_obs),
            dtype=np.float64,
        )
        if expansion_rate.ndim not in (1, 2):
            raise ValueError("CC E(z) must be a vector or a batch of vectors.")
        expected_shape = (
            (len(self.z_obs),)
            if expansion_rate.ndim == 1
            else (expansion_rate.shape[0], len(self.z_obs))
        )
        if expansion_rate.shape != expected_shape:
            raise ValueError(
                f"CC E(z) has shape {expansion_rate.shape}; expected "
                f"{expected_shape}."
            )
        if np.any(~np.isfinite(expansion_rate)) or np.any(
            expansion_rate <= 0.0
        ):
            raise ValueError("CC requires finite, strictly positive E(z).")
        return expansion_rate

    def quadratic_terms(
        self,
        background: Any,
    ) -> tuple[FloatArray | float, FloatArray | float, float]:
        """Return ``A, B, D`` without forming or storing ``C^{-1}``."""
        expansion_rate = self._expansion_rate(background)
        if expansion_rate.ndim == 1:
            whitened_E = solve_triangular(
                self.L,
                expansion_rate,
                lower=True,
                check_finite=False,
            )
            A: FloatArray | float = float(np.dot(whitened_E, whitened_E))
            B: FloatArray | float = float(
                np.dot(whitened_E, self.whitened_H)
            )
        else:
            whitened_E = solve_triangular(
                self.L,
                expansion_rate.T,
                lower=True,
                check_finite=False,
            ).T
            A = np.einsum("ij,ij->i", whitened_E, whitened_E)
            B = whitened_E @ self.whitened_H
        if np.any(np.asarray(A) <= 0.0):
            raise ValueError("CC quadratic coefficient A must be positive.")
        return A, B, self.D

    def conditional_h0(
        self,
        background: Any,
    ) -> tuple[FloatArray | float, FloatArray | float]:
        """Return analytic ``H0_hat=B/A`` and ``sigma(H0)=1/sqrt(A)``."""
        A, B, _ = self.quadratic_terms(background)
        return B / A, 1.0 / np.sqrt(A)

    def profile_chi2(self, background: Any) -> FloatArray | float:
        """Return ``D-B**2/A``, clipped only against roundoff below zero."""
        A, B, D = self.quadratic_terms(background)
        return np.maximum(D - B**2 / A, 0.0)

    def log_likelihood(self, background: Any) -> FloatArray | float:
        """Evaluate scalar or vectorized CC log likelihood."""
        if self.h0_mode in ("profile", "marginalized"):
            A, B, D = self.quadratic_terms(background)
            log_likelihood = -0.5 * np.maximum(D - B**2 / A, 0.0)
            if self.h0_mode == "marginalized":
                log_likelihood = log_likelihood - 0.5 * np.log(A)
            return log_likelihood

        expansion_rate = self._expansion_rate(background)
        H0 = getattr(background, "H0", None)
        if H0 is None:
            raise ValueError("Explicit CC mode requires a shared positive H0.")
        h0_array = np.asarray(H0, dtype=np.float64)
        if np.any(~np.isfinite(h0_array)) or np.any(h0_array <= 0.0):
            raise ValueError("Explicit CC mode requires finite positive H0.")

        if expansion_rate.ndim == 1:
            if h0_array.ndim != 0:
                raise ValueError("Scalar CC evaluation requires scalar H0.")
            residual = self.H_obs - float(h0_array) * expansion_rate
            whitened = solve_triangular(
                self.L,
                residual,
                lower=True,
                check_finite=False,
            )
            return -0.5 * float(np.dot(whitened, whitened))

        if h0_array.shape != (expansion_rate.shape[0],):
            raise ValueError(
                "Vectorized CC H0 must match the cosmology batch size."
            )
        residual = self.H_obs[None, :] - h0_array[:, None] * expansion_rate
        whitened = solve_triangular(
            self.L,
            residual.T,
            lower=True,
            check_finite=False,
        ).T
        return -0.5 * np.einsum("ij,ij->i", whitened, whitened)
