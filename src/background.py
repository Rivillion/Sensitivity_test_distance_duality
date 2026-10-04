from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np  # type: ignore
from numpy.typing import ArrayLike, NDArray  # type: ignore
from scipy.integrate import cumulative_simpson  # type: ignore
from scipy.interpolate import CubicHermiteSpline  # type: ignore


c_kms = 299792.458
DEFAULT_N_GRID = 385

FloatArray = NDArray[np.float64]
FloatOrArray = float | FloatArray


def _transverse_distance_from_integral(
    comoving_integral: ArrayLike,
    Omega_k: ArrayLike,
) -> FloatArray:
    """Map the radial comoving integral to transverse comoving distance."""
    integral, curvature = np.broadcast_arrays(
        np.asarray(comoving_integral, dtype=np.float64),
        np.asarray(Omega_k, dtype=np.float64),
    )
    curvature_argument = curvature * integral**2
    result = np.empty_like(integral)

    # The common Taylor series is accurate around the flat limit and avoids
    # dividing two quantities that simultaneously approach zero.
    near_flat = np.abs(curvature_argument) < 1.0e-8
    result[near_flat] = integral[near_flat] * (
        1.0
        + curvature_argument[near_flat] / 6.0
        + curvature_argument[near_flat] ** 2 / 120.0
    )

    open_geometry = (~near_flat) & (curvature > 0.0)
    open_argument = np.sqrt(curvature[open_geometry]) * integral[open_geometry]
    result[open_geometry] = (
        integral[open_geometry]
        * np.sinh(open_argument)
        / open_argument
    )

    closed_geometry = (~near_flat) & (curvature < 0.0)
    closed_argument = (
        np.sqrt(-curvature[closed_geometry]) * integral[closed_geometry]
    )
    result[closed_geometry] = (
        integral[closed_geometry]
        * np.sin(closed_argument)
        / closed_argument
    )

    if np.any(~np.isfinite(result)):
        raise ValueError("The selected curvature gives an invalid D_M(z).")
    return result


class Background:
    """CPL background with optional curvature and distance-duality violation.

    The expansion history and dimensionless distances depend only on
    ``Omega_m``, ``Omega_k``, ``w0`` and ``wa``.  ``H0`` and ``alpha`` are
    optional absolute-normalization parameters; likelihoods that profile
    their calibration should use :meth:`I`, the dimensionless BAO distances,
    and :meth:`sn_log_shape`.

    Each instance represents one parameter set.  Construct a new instance
    for new expansion parameters rather than mutating them, because the
    comoving integral is cached during initialization.

    Parameters
    ----------
    Omega_m
        Present-day matter density fraction.
    Omega_k
        Present-day curvature density fraction. Positive values correspond
        to open spatial sections. The default is zero (spatially flat).
    w0, wa
        CPL dark-energy parameters, ``w(z) = w0 + wa*z/(1+z)``.
    epsilon, alpha
        Distance-duality parameters,
        ``eta(z) = alpha*(1+z)**epsilon``.
    H0
        Hubble constant in km/s/Mpc.  It is only required for physical
        distances and :meth:`H`; profiled likelihoods can leave it unset.
    z_max
        Largest redshift covered by the cached distance integral.
    n_grid
        Number of points in the cached integration grid.
    """

    def __init__(
        self,
        Omega_m: float,
        w0: float = -1.0,
        wa: float = 0.0,
        epsilon: float = 0.0,
        alpha: float = 1.0,
        H0: Optional[float] = None,
        z_max: float = 2.6,
        n_grid: int = DEFAULT_N_GRID,
        Omega_k: float = 0.0,
        eos: str = "cpl",
    ) -> None:
        self.Omega_m = self._validate_finite_parameter("Omega_m", Omega_m)
        self.Omega_k = self._validate_finite_parameter("Omega_k", Omega_k)
        self.w0 = self._validate_finite_parameter("w0", w0)
        self.wa = self._validate_finite_parameter("wa", wa)
        if eos not in ("cpl", "linear"):
            raise ValueError("eos must be 'cpl' or 'linear'.")
        self.eos = eos
        self.epsilon = self._validate_finite_parameter("epsilon", epsilon)
        self.alpha = self._validate_finite_parameter("alpha", alpha)
        self.z_max = self._validate_finite_parameter("z_max", z_max)
        self.n_grid = int(n_grid)

        self.H0: Optional[float] = None
        if H0 is not None:
            self.H0 = self._validate_finite_parameter("H0", H0)

        if not 0.0 < self.Omega_m < 1.0:
            raise ValueError("Omega_m must lie strictly between 0 and 1.")

        if self.alpha <= 0.0:
            raise ValueError("alpha must be strictly positive.")

        if self.H0 is not None and self.H0 <= 0.0:
            raise ValueError("H0 must be strictly positive when supplied.")

        if self.z_max <= 0.0:
            raise ValueError("z_max must be strictly positive.")

        if self.n_grid < 2:
            raise ValueError("n_grid must be at least 2.")

        self.Omega_de = 1.0 - self.Omega_m - self.Omega_k

        self._distance_scale: Optional[float] = None
        if self.H0 is not None:
            self._distance_scale = c_kms / self.H0

        self._build_comoving_integral_cache()

    @staticmethod
    def _validate_finite_parameter(name: str, value: float) -> float:
        """Convert a scalar parameter to ``float`` and require finiteness."""
        validated_value = float(value)

        if not np.isfinite(validated_value):
            raise ValueError(f"{name} must be finite.")

        return validated_value

    def _build_comoving_integral_cache(self) -> None:
        """Build the comoving-distance integral once for this parameter set."""
        (
            self._z_grid,
            log_one_plus_z_grid,
            redshift_fraction_grid,
            matter_evolution_grid,
            curvature_evolution_grid,
        ) = self._integration_grid_basis(
            self.z_max,
            self.n_grid,
        )

        expansion_rate_grid = self._expansion_rate_from_basis(
            log_one_plus_z_grid,
            redshift_fraction_grid,
            matter_evolution_grid,
            curvature_evolution_grid,
        )
        inverse_expansion_rate_grid = 1.0 / expansion_rate_grid

        self._I_grid = cumulative_simpson(
            inverse_expansion_rate_grid,
            x=self._z_grid,
            initial=0.0,
        )

        self._I_spline = CubicHermiteSpline(
            self._z_grid,
            self._I_grid,
            inverse_expansion_rate_grid,
            extrapolate=False,
        )

    @staticmethod
    @lru_cache(maxsize=16)
    def _integration_grid_basis(
        z_max: float,
        n_grid: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Return immutable redshift-only arrays shared across backgrounds."""
        z_grid = np.linspace(
            0.0,
            z_max,
            n_grid,
            dtype=np.float64,
        )
        log_one_plus_z_grid = np.log1p(z_grid)
        redshift_fraction_grid = z_grid / (1.0 + z_grid)
        matter_evolution_grid = np.exp(3.0 * log_one_plus_z_grid)
        curvature_evolution_grid = np.exp(2.0 * log_one_plus_z_grid)

        arrays = (
            z_grid,
            log_one_plus_z_grid,
            redshift_fraction_grid,
            matter_evolution_grid,
            curvature_evolution_grid,
        )
        for array in arrays:
            array.flags.writeable = False

        return arrays

    @staticmethod
    def _match_input_shape(
        result: ArrayLike,
        *inputs: ArrayLike,
    ) -> FloatOrArray:
        """Return a float for scalar inputs and an array otherwise."""
        result_array = np.asarray(result, dtype=np.float64)

        all_inputs_are_scalars = all(np.ndim(value) == 0 for value in inputs)
        if all_inputs_are_scalars:
            return float(result_array)

        return result_array

    def _validate_redshift(
        self,
        z: ArrayLike,
        *,
        require_integral_range: bool = False,
    ) -> FloatArray:
        z_array = np.asarray(z, dtype=np.float64)

        if np.any(~np.isfinite(z_array)):
            raise ValueError("Redshifts must be finite.")
        if np.any(z_array < 0.0):
            raise ValueError("Redshifts must be non-negative.")
        if require_integral_range and np.any(z_array > self.z_max):
            raise ValueError(
                f"Distance requested above z_max={self.z_max}. "
                "Construct Background with a larger z_max."
            )

        return z_array

    def _require_distance_scale(self) -> float:
        if self._distance_scale is None:
            raise ValueError(
                "H0 is required for physical distances. "
                "Use the dimensionless methods for profiled likelihoods."
            )
        return self._distance_scale

    def _expansion_rate_from_basis(
        self,
        log_one_plus_z: FloatArray,
        redshift_fraction: FloatArray,
        matter_evolution: FloatArray,
        curvature_evolution: FloatArray,
    ) -> FloatArray:
        """Evaluate ``E`` from reusable redshift-only basis arrays."""
        with np.errstate(over="ignore", invalid="ignore"):
            matter_density = self.Omega_m * matter_evolution
            curvature_density = self.Omega_k * curvature_evolution

            if self.eos == "linear":
                redshift = np.expm1(log_one_plus_z)
                log_dark_energy_evolution = (
                    3.0 * (1.0 + self.w0 - self.wa) * log_one_plus_z
                    + 3.0 * self.wa * redshift
                )
            else:
                log_dark_energy_evolution = (
                    3.0 * (1.0 + self.w0 + self.wa) * log_one_plus_z
                    - 3.0 * self.wa * redshift_fraction
                )

            dark_energy_density = self.Omega_de * np.exp(
                log_dark_energy_evolution
            )

            expansion_rate_squared = (
                matter_density + curvature_density + dark_energy_density
            )
            expansion_rate = np.sqrt(expansion_rate_squared)

        expansion_rate_is_invalid = np.any(~np.isfinite(expansion_rate))
        expansion_rate_is_nonpositive = np.any(expansion_rate <= 0.0)

        if expansion_rate_is_invalid or expansion_rate_is_nonpositive:
            raise ValueError("The selected parameters give an invalid E(z).")

        return expansion_rate

    def E(self, z: ArrayLike) -> FloatOrArray:
        """Return the dimensionless expansion rate ``H(z)/H0``."""
        redshift = self._validate_redshift(z)
        log_one_plus_z = np.log1p(redshift)
        redshift_fraction = redshift / (1.0 + redshift)
        matter_evolution = np.exp(3.0 * log_one_plus_z)
        curvature_evolution = np.exp(2.0 * log_one_plus_z)
        expansion_rate = self._expansion_rate_from_basis(
            log_one_plus_z,
            redshift_fraction,
            matter_evolution,
            curvature_evolution,
        )

        return self._match_input_shape(expansion_rate, z)

    def H(self, z: ArrayLike) -> FloatOrArray:
        """Return ``H(z)`` in km/s/Mpc; requires ``H0``."""
        if self.H0 is None:
            raise ValueError(
                "H0 is required for H(z). Use E(z) when H0 is profiled."
            )

        dimensionless_expansion_rate = np.asarray(
            self.E(z),
            dtype=np.float64,
        )
        hubble_rate = self.H0 * dimensionless_expansion_rate

        return self._match_input_shape(hubble_rate, z)

    def I(self, z: ArrayLike) -> FloatOrArray:
        """Return the dimensionless comoving integral ``int_0^z dz/E(z)``."""
        redshift = self._validate_redshift(
            z,
            require_integral_range=True,
        )
        comoving_integral = self._I_spline(redshift)

        if np.any(~np.isfinite(comoving_integral)):
            raise ValueError("Failed to evaluate the comoving integral.")

        return self._match_input_shape(comoving_integral, z)

    def D_M_dimensionless(self, z: ArrayLike) -> FloatOrArray:
        """Return the dimensionless transverse comoving distance."""
        comoving_integral = np.asarray(self.I(z), dtype=np.float64)
        transverse_distance = _transverse_distance_from_integral(
            comoving_integral,
            self.Omega_k,
        )
        return self._match_input_shape(transverse_distance, z)

    def D_H_dimensionless(self, z: ArrayLike) -> FloatOrArray:
        """Return the BAO radial-distance template ``1/E(z)``."""
        expansion_rate = np.asarray(self.E(z), dtype=np.float64)
        radial_distance = 1.0 / expansion_rate

        return self._match_input_shape(radial_distance, z)

    def D_V_dimensionless(self, z: ArrayLike) -> FloatOrArray:
        """Return ``[z D_M(z)^2/E(z)]^(1/3)`` dimensionlessly."""
        redshift = self._validate_redshift(
            z,
            require_integral_range=True,
        )
        transverse_distance = np.asarray(
            self.D_M_dimensionless(redshift),
            dtype=np.float64,
        )
        expansion_rate = np.asarray(
            self.E(redshift),
            dtype=np.float64,
        )

        volume_distance_cubed = (
            redshift * transverse_distance**2 / expansion_rate
        )
        volume_distance = np.cbrt(volume_distance_cubed)

        return self._match_input_shape(volume_distance, z)

    def D_M(self, z: ArrayLike) -> FloatOrArray:
        """Return the transverse comoving distance in Mpc."""
        distance_scale = self._require_distance_scale()
        dimensionless_distance = np.asarray(
            self.D_M_dimensionless(z),
            dtype=np.float64,
        )
        distance_mpc = distance_scale * dimensionless_distance

        return self._match_input_shape(distance_mpc, z)

    def D_H(self, z: ArrayLike) -> FloatOrArray:
        """Return the Hubble distance ``c/H(z)`` in Mpc."""
        distance_scale = self._require_distance_scale()
        dimensionless_distance = np.asarray(
            self.D_H_dimensionless(z),
            dtype=np.float64,
        )
        distance_mpc = distance_scale * dimensionless_distance

        return self._match_input_shape(distance_mpc, z)

    def D_V(self, z: ArrayLike) -> FloatOrArray:
        """Return the volume-averaged BAO distance in Mpc."""
        distance_scale = self._require_distance_scale()
        dimensionless_distance = np.asarray(
            self.D_V_dimensionless(z),
            dtype=np.float64,
        )
        distance_mpc = distance_scale * dimensionless_distance

        return self._match_input_shape(distance_mpc, z)

    def eta(self, z: ArrayLike) -> FloatOrArray:
        """Return ``alpha*(1+z)**epsilon``."""
        redshift = self._validate_redshift(z)

        log_redshift_evolution = self.epsilon * np.log1p(redshift)
        distance_duality_factor = self.alpha * np.exp(
            log_redshift_evolution
        )

        return self._match_input_shape(distance_duality_factor, z)

    def d_L(
        self,
        z_cmb: ArrayLike,
        z_hel: Optional[ArrayLike] = None,
    ) -> FloatOrArray:
        """Return luminosity distance in Mpc.

        ``D_M(z)`` and ``eta(z)`` are evaluated at the CMB-frame/Hubble-
        diagram redshift, while the ordinary ``(1+z)`` factor uses the
        heliocentric redshift.
        """
        if z_hel is None:
            z_hel = z_cmb

        cmb_redshift = self._validate_redshift(
            z_cmb,
            require_integral_range=True,
        )
        heliocentric_redshift = self._validate_redshift(z_hel)

        comoving_distance = np.asarray(
            self.D_M(cmb_redshift),
            dtype=np.float64,
        )
        distance_duality_factor = np.asarray(
            self.eta(cmb_redshift),
            dtype=np.float64,
        )

        redshift_factor = 1.0 + heliocentric_redshift
        luminosity_distance = (
            redshift_factor
            * distance_duality_factor
            * comoving_distance
        )

        return self._match_input_shape(
            luminosity_distance,
            z_cmb,
            z_hel,
        )

    def sn_log_shape(
        self,
        z_cmb: ArrayLike,
        z_hel: Optional[ArrayLike] = None,
    ) -> FloatOrArray:
        """Return the calibration-free SN distance shape in magnitudes.

        This is

        ``5 log10[(1+z_hel) D_M(z_cmb)] + 5 epsilon log10(1+z_cmb)``.

        The omitted additive constant contains ``H0``, the absolute
        magnitude, ``alpha`` and the Mpc conversion, and is profiled by
        the supernova likelihood.
        """
        if z_hel is None:
            z_hel = z_cmb

        cmb_redshift = self._validate_redshift(
            z_cmb,
            require_integral_range=True,
        )
        heliocentric_redshift = self._validate_redshift(z_hel)

        transverse_distance = np.asarray(
            self.D_M_dimensionless(cmb_redshift),
            dtype=np.float64,
        )

        if np.any(transverse_distance <= 0.0):
            raise ValueError("SN redshifts must give a positive D_M(z).")

        comoving_distance_term = 5.0 * np.log10(transverse_distance)
        standard_redshift_term = 5.0 * np.log10(
            1.0 + heliocentric_redshift
        )
        duality_violation_term = (
            self.epsilon * 5.0 * np.log10(1.0 + cmb_redshift)
        )

        supernova_shape = (
            comoving_distance_term
            + standard_redshift_term
            + duality_violation_term
        )

        return self._match_input_shape(
            supernova_shape,
            z_cmb,
            z_hel,
        )

    def comoving_distance(self, z: ArrayLike) -> FloatOrArray:
        """Alias for :meth:`D_M`, in Mpc."""
        return self.D_M(z)

    def luminosity_distance(
        self,
        z_cmb: ArrayLike,
        z_hel: Optional[ArrayLike] = None,
    ) -> FloatOrArray:
        """Alias for :meth:`d_L`, in Mpc."""
        return self.d_L(z_cmb, z_hel)


class _BatchedBackground:
    """Vectorized background used internally by the sampler likelihood."""

    def __init__(
        self,
        Omega_m: ArrayLike,
        w0: ArrayLike,
        wa: ArrayLike,
        epsilon: ArrayLike,
        alpha: ArrayLike,
        Omega_k: ArrayLike = 0.0,
        *,
        z_max: float = 2.6,
        n_grid: int = DEFAULT_N_GRID,
    ) -> None:
        parameters = np.broadcast_arrays(
            np.asarray(Omega_m, dtype=np.float64),
            np.asarray(w0, dtype=np.float64),
            np.asarray(wa, dtype=np.float64),
            np.asarray(epsilon, dtype=np.float64),
            np.asarray(alpha, dtype=np.float64),
            np.asarray(Omega_k, dtype=np.float64),
        )
        if parameters[0].ndim != 1:
            raise ValueError("Batched parameters must be one-dimensional.")

        (
            omega_m,
            w0_array,
            wa_array,
            epsilon_array,
            alpha_array,
            omega_k,
        ) = parameters
        self.batch_size = omega_m.size
        self.z_max = float(z_max)
        self.n_grid = int(n_grid)

        parameter_valid = (
            np.isfinite(omega_m)
            & np.isfinite(w0_array)
            & np.isfinite(wa_array)
            & np.isfinite(epsilon_array)
            & np.isfinite(alpha_array)
            & np.isfinite(omega_k)
            & (omega_m > 0.0)
            & (omega_m < 1.0)
            & (alpha_array > 0.0)
        )

        self.Omega_m = np.where(parameter_valid, omega_m, 0.3)
        self.w0 = np.where(parameter_valid, w0_array, -1.0)
        self.wa = np.where(parameter_valid, wa_array, 0.0)
        self.epsilon = np.where(parameter_valid, epsilon_array, 0.0)
        self.alpha = np.where(parameter_valid, alpha_array, 1.0)
        self.Omega_k = np.where(parameter_valid, omega_k, 0.0)
        self.Omega_de = 1.0 - self.Omega_m - self.Omega_k

        (
            self._z_grid,
            log_one_plus_z_grid,
            redshift_fraction_grid,
            matter_evolution_grid,
            curvature_evolution_grid,
        ) = Background._integration_grid_basis(
            self.z_max,
            self.n_grid,
        )

        expansion_rate_grid = self._expansion_rate_from_basis(
            log_one_plus_z_grid,
            redshift_fraction_grid,
            matter_evolution_grid,
            curvature_evolution_grid,
        )
        expansion_valid = np.all(
            np.isfinite(expansion_rate_grid)
            & (expansion_rate_grid > 0.0),
            axis=1,
        )
        self.valid = parameter_valid & expansion_valid

        safe_expansion_rate = np.where(
            self.valid[:, None],
            expansion_rate_grid,
            1.0,
        )
        inverse_expansion_rate = 1.0 / safe_expansion_rate
        self._I_grid = cumulative_simpson(
            inverse_expansion_rate,
            x=self._z_grid,
            axis=1,
            initial=0.0,
        )
        self._I_spline = CubicHermiteSpline(
            self._z_grid,
            self._I_grid,
            inverse_expansion_rate,
            axis=1,
            extrapolate=False,
        )

    def _expansion_rate_from_basis(
        self,
        log_one_plus_z: FloatArray,
        redshift_fraction: FloatArray,
        matter_evolution: FloatArray,
        curvature_evolution: FloatArray,
    ) -> FloatArray:
        with np.errstate(over="ignore", invalid="ignore"):
            omega_m = self.Omega_m[:, None]
            w0 = self.w0[:, None]
            wa = self.wa[:, None]
            omega_k = self.Omega_k[:, None]
            matter_density = omega_m * matter_evolution
            curvature_density = omega_k * curvature_evolution
            dark_energy_density = (1.0 - omega_m - omega_k) * np.exp(
                3.0 * (1.0 + w0 + wa) * log_one_plus_z
                - 3.0 * wa * redshift_fraction
            )
            return np.sqrt(
                matter_density + curvature_density + dark_energy_density
            )

    def _validate_redshift(
        self,
        z: ArrayLike,
        *,
        require_integral_range: bool = False,
    ) -> FloatArray:
        redshift = np.asarray(z, dtype=np.float64)
        if np.any(~np.isfinite(redshift)):
            raise ValueError("Redshifts must be finite.")
        if np.any(redshift < 0.0):
            raise ValueError("Redshifts must be non-negative.")
        if require_integral_range and np.any(redshift > self.z_max):
            raise ValueError(
                f"Distance requested above z_max={self.z_max}."
            )
        return redshift

    def E(self, z: ArrayLike) -> FloatArray:
        """Return ``E(z)`` with the batch dimension first."""
        redshift = self._validate_redshift(z)
        flattened_redshift = redshift.reshape(-1)
        log_one_plus_z = np.log1p(flattened_redshift)
        expansion_rate = self._expansion_rate_from_basis(
            log_one_plus_z,
            flattened_redshift / (1.0 + flattened_redshift),
            np.exp(3.0 * log_one_plus_z),
            np.exp(2.0 * log_one_plus_z),
        )
        output_shape = (self.batch_size,) + redshift.shape
        return expansion_rate.reshape(output_shape)

    def I(self, z: ArrayLike) -> FloatArray:
        """Return ``I(z)`` with the batch dimension first."""
        redshift = self._validate_redshift(
            z,
            require_integral_range=True,
        )
        comoving_integral = np.asarray(
            self._I_spline(redshift),
            dtype=np.float64,
        )
        if np.any(~np.isfinite(comoving_integral)):
            raise ValueError("Failed to evaluate the comoving integral.")
        return comoving_integral

    def D_M_dimensionless(self, z: ArrayLike) -> FloatArray:
        """Return transverse comoving distance with batch dimension first."""
        comoving_integral = self.I(z)
        curvature = self.Omega_k.reshape(
            (self.batch_size,) + (1,) * (comoving_integral.ndim - 1)
        )
        return _transverse_distance_from_integral(
            comoving_integral,
            curvature,
        )
