"""Regression tests for the numerical likelihood optimizations."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky, solve_triangular

from src.background import Background, _BatchedBackground
from src.likelihoods import DESI_BAO, DES_Dovekie, PantheonPlus
from src.sampler import CombinedLikelihood, initialize_likelihoods


class FakeBackground:
    """Small deterministic background for likelihood algebra tests."""

    epsilon = 0.12

    @staticmethod
    def I(z):
        redshift = np.asarray(z, dtype=np.float64)
        return redshift / (1.0 + 0.25 * redshift)

    @classmethod
    def D_M_dimensionless(cls, z):
        return cls.I(z)

    @staticmethod
    def E(z):
        redshift = np.asarray(z, dtype=np.float64)
        return np.sqrt(1.0 + 0.7 * redshift)


class ZeroLikelihood:
    """Likelihood used to isolate batched parameter validation."""

    @staticmethod
    def log_likelihood(background):
        return np.zeros_like(
            np.asarray(background.epsilon, dtype=np.float64)
        )


class NumericalOptimizationTests(unittest.TestCase):
    def test_background_reuses_immutable_grid_basis(self) -> None:
        first = Background(0.3)
        second = Background(0.32)

        self.assertIs(first._z_grid, second._z_grid)
        self.assertFalse(first._z_grid.flags.writeable)

        redshifts = np.array([0.01, 0.2, 0.8, 1.5, 2.6])
        reference = np.array(
            [
                quad(
                    lambda value: 1.0 / first.E(value),
                    0.0,
                    redshift,
                    epsabs=1e-13,
                    epsrel=1e-13,
                )[0]
                for redshift in redshifts
            ]
        )
        np.testing.assert_allclose(
            first.I(redshifts),
            reference,
            rtol=0.0,
            atol=5e-8,
        )
        np.testing.assert_array_equal(
            first.D_M_dimensionless(redshifts),
            first.I(redshifts),
        )

    def test_sn_shape_uses_hel_redshift_but_hd_tilt(self) -> None:
        background = Background(0.3, epsilon=0.17)
        z_hd = np.array([0.05, 0.3, 0.8])
        z_hel = np.array([0.049, 0.304, 0.795])
        expected = (
            5.0 * np.log10(background.I(z_hd))
            + 5.0 * np.log10(1.0 + z_hel)
            + background.epsilon * 5.0 * np.log10(1.0 + z_hd)
        )
        np.testing.assert_allclose(
            background.sn_log_shape(z_hd, z_hel),
            expected,
            rtol=0.0,
            atol=2e-14,
        )

    def test_curved_background_matches_reference_equations(self) -> None:
        redshifts = np.array([0.05, 0.3, 0.8, 1.5, 2.5])
        omega_m = 0.3

        for omega_k in (-0.25, 0.25):
            with self.subTest(Omega_k=omega_k):
                background = Background(omega_m, Omega_k=omega_k)
                omega_de = 1.0 - omega_m - omega_k
                expected_expansion = np.sqrt(
                    omega_m * (1.0 + redshifts) ** 3
                    + omega_k * (1.0 + redshifts) ** 2
                    + omega_de
                )
                np.testing.assert_allclose(
                    background.E(redshifts),
                    expected_expansion,
                    rtol=2e-15,
                    atol=0.0,
                )

                reference_integral = np.array(
                    [
                        quad(
                            lambda value: 1.0 / background.E(value),
                            0.0,
                            redshift,
                            epsabs=1e-13,
                            epsrel=1e-13,
                        )[0]
                        for redshift in redshifts
                    ]
                )
                root_curvature = np.sqrt(abs(omega_k))
                if omega_k > 0.0:
                    expected_distance = (
                        np.sinh(root_curvature * reference_integral)
                        / root_curvature
                    )
                else:
                    expected_distance = (
                        np.sin(root_curvature * reference_integral)
                        / root_curvature
                    )
                np.testing.assert_allclose(
                    background.D_M_dimensionless(redshifts),
                    expected_distance,
                    rtol=0.0,
                    atol=6e-8,
                )

    def test_transverse_distance_is_continuous_at_flat_limit(self) -> None:
        redshifts = np.array([0.1, 0.7, 1.8, 2.6])
        flat = Background(0.3).D_M_dimensionless(redshifts)
        for omega_k in (-1.0e-12, 1.0e-12):
            curved = Background(
                0.3,
                Omega_k=omega_k,
            ).D_M_dimensionless(redshifts)
            np.testing.assert_allclose(curved, flat, rtol=2e-12, atol=1e-13)

    def test_batched_background_matches_scalar_backgrounds(self) -> None:
        parameters = np.array(
            [
                [0.30, 0.0, -1.0, 0.0, 0.0],
                [0.22, 0.2, -0.8, 0.5, 0.1],
                [0.42, -0.3, -1.4, -0.7, -0.2],
                [0.15, 0.1, 0.2, -2.0, 0.3],
            ]
        )
        batched = _BatchedBackground(
            Omega_m=parameters[:, 0],
            Omega_k=parameters[:, 1],
            w0=parameters[:, 2],
            wa=parameters[:, 3],
            epsilon=parameters[:, 4],
            alpha=np.ones(len(parameters)),
        )
        redshifts = np.array([0.05, 0.3, 0.8, 1.5, 2.5])

        scalar_integrals = []
        scalar_transverse_distances = []
        scalar_expansion_rates = []
        for omega_m, omega_k, w0, wa, epsilon in parameters:
            background = Background(
                omega_m,
                Omega_k=omega_k,
                w0=w0,
                wa=wa,
                epsilon=epsilon,
            )
            scalar_integrals.append(background.I(redshifts))
            scalar_transverse_distances.append(
                background.D_M_dimensionless(redshifts)
            )
            scalar_expansion_rates.append(background.E(redshifts))

        np.testing.assert_allclose(
            batched.I(redshifts),
            scalar_integrals,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            batched.D_M_dimensionless(redshifts),
            scalar_transverse_distances,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            batched.E(redshifts),
            scalar_expansion_rates,
            rtol=2e-14,
            atol=2e-14,
        )

    def test_all_sampler_likelihood_combinations_vectorize(self) -> None:
        initialized = initialize_likelihoods(
            ("pantheon", "dovekie", "desi")
        ).likelihoods
        by_name = dict(zip(("pantheon", "dovekie", "desi"), initialized))
        self.assertEqual(by_name["desi"].amplitude_mode, "marginalized")
        combinations = (
            ("pantheon",),
            ("dovekie",),
            ("desi",),
            ("pantheon", "desi"),
            ("dovekie", "desi"),
        )
        parameters = {
            "Omega_m": np.array([0.2, 0.3, 0.4, 0.25]),
            "Omega_k": np.array([-0.2, 0.0, 0.15, 0.3]),
            "w0": np.array([-1.2, -1.0, -0.8, -1.5]),
            "wa": np.array([0.4, 0.0, -0.5, 1.0]),
            "epsilon": np.array([-0.1, 0.0, 0.1, 0.2]),
            "nu": np.array([24.0, 24.1, 24.2, 24.3]),
            "H0rd": np.array([9800.0, 9900.0, 10000.0, 10100.0]),
        }

        for names in combinations:
            with self.subTest(likelihoods=names):
                likelihood = CombinedLikelihood(
                    [by_name[name] for name in names]
                )
                batched = likelihood(parameters)
                scalar = np.array(
                    [
                        likelihood(
                            {
                                name: values[index]
                                for name, values in parameters.items()
                            }
                        )
                        for index in range(len(parameters["Omega_m"]))
                    ]
                )
                np.testing.assert_allclose(
                    batched,
                    scalar,
                    rtol=0.0,
                    atol=5e-7,
                )
                np.testing.assert_array_equal(
                    np.round(batched, 2),
                    np.round(scalar, 2),
                )

    def test_invalid_batch_rows_do_not_poison_valid_rows(self) -> None:
        likelihood = CombinedLikelihood([ZeroLikelihood()])
        result = likelihood(
            {
                "Omega_m": np.array([0.3, np.nan, 1.2, 0.25]),
                "w0": np.array([-1.0, -1.0, -1.0, np.inf]),
            }
        )

        self.assertEqual(result[0], 0.0)
        self.assertTrue(np.all(np.isneginf(result[1:])))

    def test_pantheon_profile_identity(self) -> None:
        likelihood = PantheonPlus.__new__(PantheonPlus)
        likelihood.z_cmb = np.array([0.05, 0.2, 0.5, 0.9])
        likelihood.z_hel = np.array([0.049, 0.203, 0.497, 0.905])
        likelihood.mb_obs = np.array([16.1, 18.2, 20.4, 21.8])
        likelihood.M_fiducial = PantheonPlus.M_FIDUCIAL
        likelihood.standard_redshift_term = 5.0 * np.log10(
            1.0 + likelihood.z_hel
        )
        likelihood.tilt_redshift_term = 5.0 * np.log10(
            1.0 + likelihood.z_cmb
        )

        covariance = np.array(
            [
                [0.10, 0.01, 0.00, 0.00],
                [0.01, 0.12, 0.02, 0.00],
                [0.00, 0.02, 0.15, 0.01],
                [0.00, 0.00, 0.01, 0.11],
            ]
        )
        likelihood.L = np.asfortranarray(
            cholesky(covariance, lower=True)
        )
        ones = np.ones_like(likelihood.mb_obs)
        likelihood.offset_direction = solve_triangular(
            likelihood.L,
            ones,
            lower=True,
            check_finite=False,
        )
        likelihood.offset_norm = np.dot(
            likelihood.offset_direction,
            likelihood.offset_direction,
        )
        likelihood.tilt_direction = solve_triangular(
            likelihood.L,
            likelihood.tilt_redshift_term,
            lower=True,
            check_finite=False,
        )

        background = FakeBackground()
        model = (
            5.0 * np.log10(
                background.D_M_dimensionless(likelihood.z_cmb)
            )
            + likelihood.standard_redshift_term
            + background.epsilon * likelihood.tilt_redshift_term
            + likelihood.M_fiducial
        )
        whitened = solve_triangular(
            likelihood.L,
            likelihood.mb_obs - model,
            lower=True,
            check_finite=False,
        )
        best_offset = (
            np.dot(whitened, likelihood.offset_direction)
            / likelihood.offset_norm
        )
        profiled = (
            whitened - best_offset * likelihood.offset_direction
        )
        expected = -0.5 * np.dot(profiled, profiled)

        self.assertAlmostEqual(
            likelihood.log_likelihood(background),
            expected,
            places=10,
        )

        fixed_chi2, fixed_epsilon = likelihood.profiled_linear_nuisances(
            background
        )
        self.assertAlmostEqual(fixed_chi2, -2.0 * expected, places=10)
        self.assertEqual(fixed_epsilon, background.epsilon)

        base_model = (
            5.0
            * np.log10(background.D_M_dimensionless(likelihood.z_cmb))
            + likelihood.standard_redshift_term
            + likelihood.M_fiducial
        )
        whitened_base_residual = solve_triangular(
            likelihood.L,
            likelihood.mb_obs - base_model,
            lower=True,
            check_finite=False,
        )
        design = np.column_stack(
            (likelihood.offset_direction, likelihood.tilt_direction)
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            whitened_base_residual,
            rcond=None,
        )
        direct_residual = whitened_base_residual - design @ coefficients
        free_chi2, free_epsilon = likelihood.profiled_linear_nuisances(
            background,
            profile_epsilon=True,
        )
        self.assertAlmostEqual(
            free_chi2,
            np.dot(direct_residual, direct_residual),
            places=10,
        )
        self.assertAlmostEqual(free_epsilon, coefficients[1], places=10)

    def test_dovekie_symmetric_profile_identity(self) -> None:
        likelihood = DES_Dovekie.__new__(DES_Dovekie)
        likelihood.z_cmb = np.array([0.08, 0.25, 0.6, 1.0])
        likelihood.z_hel = np.array([0.079, 0.252, 0.597, 1.004])
        likelihood.mu_obs = np.array([36.0, 40.1, 42.7, 44.2])
        likelihood.M_fiducial = DES_Dovekie.M_FIDUCIAL
        likelihood.standard_redshift_term = 5.0 * np.log10(
            1.0 + likelihood.z_hel
        )
        likelihood.tilt_redshift_term = 5.0 * np.log10(
            1.0 + likelihood.z_cmb
        )

        factor = np.array(
            [
                [3.0, 0.0, 0.0, 0.0],
                [0.2, 2.5, 0.0, 0.0],
                [0.0, 0.3, 2.8, 0.0],
                [0.1, 0.0, 0.2, 2.4],
            ]
        )
        covariance = factor @ factor.T
        likelihood.L = np.asfortranarray(
            cholesky(covariance, lower=True)
        )
        ones = np.ones_like(likelihood.mu_obs)
        likelihood.offset_direction = solve_triangular(
            likelihood.L,
            ones,
            lower=True,
            check_finite=False,
        )
        likelihood.offset_norm = np.dot(
            likelihood.offset_direction,
            likelihood.offset_direction,
        )
        likelihood.tilt_direction = solve_triangular(
            likelihood.L,
            likelihood.tilt_redshift_term,
            lower=True,
            check_finite=False,
        )

        background = FakeBackground()
        model = (
            5.0 * np.log10(
                background.D_M_dimensionless(likelihood.z_cmb)
            )
            + likelihood.standard_redshift_term
            + background.epsilon * likelihood.tilt_redshift_term
            + likelihood.M_fiducial
        )
        residual = likelihood.mu_obs - model
        whitened = solve_triangular(
            likelihood.L,
            residual,
            lower=True,
            check_finite=False,
        )
        best_offset = (
            np.dot(whitened, likelihood.offset_direction)
            / likelihood.offset_norm
        )
        profiled = whitened - best_offset * likelihood.offset_direction
        expected = -0.5 * np.dot(profiled, profiled)

        self.assertAlmostEqual(
            likelihood.log_likelihood(background),
            expected,
            places=10,
        )

    def test_desi_profile_identity(self) -> None:
        likelihood = DESI_BAO.__new__(DESI_BAO)
        likelihood.amplitude_mode = "profile"
        likelihood.z_obs = np.array([0.3, 0.6, 1.0])
        likelihood.data_vector = np.array([8.0, 12.0, 15.0])
        likelihood.quantities = np.array(
            ["DM_over_rs", "DH_over_rs", "DV_over_rs"]
        )
        likelihood.transverse = np.array([True, False, False])
        likelihood.radial = np.array([False, True, False])
        likelihood.volume_averaged = np.array([False, False, True])
        likelihood.L = np.asfortranarray(
            cholesky(
                np.array(
                    [
                        [0.20, 0.01, 0.00],
                        [0.01, 0.25, 0.02],
                        [0.00, 0.02, 0.30],
                    ]
                ),
                lower=True,
            )
        )
        likelihood.whitened_data = solve_triangular(
            likelihood.L,
            likelihood.data_vector,
            lower=True,
            check_finite=False,
        )
        likelihood.whitened_data_norm = np.dot(
            likelihood.whitened_data,
            likelihood.whitened_data,
        )

        background = FakeBackground()
        theory = likelihood._theory_vector(background)
        whitened_theory = solve_triangular(
            likelihood.L,
            theory,
            lower=True,
            check_finite=False,
        )
        amplitude = (
            np.dot(whitened_theory, likelihood.whitened_data)
            / np.dot(whitened_theory, whitened_theory)
        )
        profiled = (
            likelihood.whitened_data - amplitude * whitened_theory
        )
        expected = -0.5 * np.dot(profiled, profiled)

        self.assertAlmostEqual(
            likelihood.log_likelihood(background),
            expected,
            places=11,
        )

        likelihood.amplitude_mode = "marginalized"
        theory_norm = np.dot(whitened_theory, whitened_theory)
        self.assertAlmostEqual(
            likelihood.log_likelihood(background),
            expected - 0.5 * np.log(theory_norm),
            places=11,
        )


if __name__ == "__main__":
    unittest.main()
