"""Algebraic, synthetic, vectorized, and inverse-ladder CC tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares, minimize_scalar

from src.background import Background, _BatchedBackground
from src.likelihoods import CosmicChronometer
from src.likelihoods import DESI_BAO
from src.sampler import CombinedLikelihood


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CC_DATA = PROJECT_ROOT / "data"
BAO_MEAN = PROJECT_ROOT / "data" / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
BAO_COV = PROJECT_ROOT / "data" / "desi_gaussian_bao_ALL_GCcomb_cov.txt"


def cc(mode: str = "explicit") -> CosmicChronometer:
    return CosmicChronometer(CC_DATA / "Hz_full_2022_mod.txt",
                            CC_DATA / "covmat_Hz_full_2022.txt", h0_mode=mode)


class CCLikelihoodTests(unittest.TestCase):
    def test_analytic_h0_matches_direct_optimization(self) -> None:
        likelihood = cc(mode="profile")
        background = Background(0.33758094)
        analytic_h0, analytic_sigma = likelihood.conditional_h0(background)

        explicit = cc(mode="explicit")

        def chi2(h0: float) -> float:
            explicit_background = Background(
                0.33758094,
                H0=h0,
            )
            return float(-2.0 * explicit.log_likelihood(explicit_background))

        result = minimize_scalar(
            chi2,
            bounds=(50.0, 85.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.x, analytic_h0, places=6)
        e = background.E(likelihood.z_obs)
        expected_sigma = 1 / np.sqrt(e @ np.linalg.solve(likelihood.covariance, e))
        self.assertAlmostEqual(analytic_sigma, expected_sigma, places=12)

    def test_profile_and_marginalized_modes_are_distinct_and_consistent(
        self,
    ) -> None:
        profile = cc(mode="profile")
        marginalized = cc(mode="marginalized")
        background = Background(0.31)
        A, _, _ = profile.quadratic_terms(background)
        expected_difference = -0.5 * np.log(A)
        actual_difference = (
            marginalized.log_likelihood(background)
            - profile.log_likelihood(background)
        )
        self.assertAlmostEqual(actual_difference, expected_difference, places=12)

    def test_explicit_at_h0_hat_equals_profile(self) -> None:
        profile = cc(mode="profile")
        shape_background = Background(0.29)
        h0_hat, _ = profile.conditional_h0(shape_background)
        explicit = cc(mode="explicit")
        explicit_background = Background(0.29, H0=float(h0_hat))
        self.assertAlmostEqual(
            explicit.log_likelihood(explicit_background),
            profile.log_likelihood(shape_background),
            places=12,
        )

    def test_synthetic_data_recovery(self) -> None:
        redshift = np.linspace(0.1, 1.5, 15)
        true_h0 = 71.25
        expansion_rate = Background(0.3).E(redshift)
        hubble_rate = true_h0 * expansion_rate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "synthetic.dat"
            covariance_path = root / "covariance.dat"
            np.savetxt(data_path, np.column_stack((redshift, hubble_rate, np.full(15, 2.0))))
            np.savetxt(covariance_path, np.eye(15) * 4)
            likelihood = CosmicChronometer(data_path, covariance_path, h0_mode="profile")
            recovered, _ = likelihood.conditional_h0(Background(0.3))
            self.assertAlmostEqual(recovered, true_h0, places=9)
            self.assertAlmostEqual(
                likelihood.profile_chi2(Background(0.3)),
                0.0,
                places=10,
            )

    def test_vectorized_and_repeated_evaluation_is_stable(self) -> None:
        likelihood = cc(mode="explicit")
        parameters = {
            "Omega_m": np.array([0.2, 0.3, 0.4, 0.25]),
            "H0": np.array([65.0, 68.0, 71.0, 74.0]),
        }
        combined = CombinedLikelihood([likelihood])
        first = combined(parameters)
        second = combined(parameters)
        scalar = np.array(
            [
                combined(
                    {
                        "Omega_m": parameters["Omega_m"][index],
                        "H0": parameters["H0"][index],
                    }
                )
                for index in range(4)
            ]
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, scalar, rtol=0.0, atol=2e-13)

    def test_invalid_explicit_h0_is_rejected_cleanly(self) -> None:
        likelihood = CombinedLikelihood([cc(mode="explicit")])
        self.assertTrue(
            np.isneginf(likelihood({"Omega_m": 0.3, "H0": -1.0}))
        )
        batch = likelihood(
            {
                "Omega_m": np.array([0.3, 0.3, 0.3]),
                "H0": np.array([68.0, -1.0, np.nan]),
            }
        )
        self.assertTrue(np.isfinite(batch[0]))
        self.assertTrue(np.all(np.isneginf(batch[1:])))

    def test_cc_and_bao_cc_are_exactly_blind_to_epsilon(self) -> None:
        chronometers = cc(mode="explicit")
        bao = DESI_BAO(BAO_MEAN, BAO_COV)
        cc_only = CombinedLikelihood([chronometers])
        joint = CombinedLikelihood([bao, chronometers])
        shared = {
            "Omega_m": 0.3,
            "H0": 68.5,
            "r_d": 148.0,
        }
        for likelihood in (cc_only, joint):
            with self.subTest(number_of_terms=len(likelihood.likelihoods)):
                zero = likelihood({**shared, "epsilon": 0.0})
                nonzero = likelihood({**shared, "epsilon": 0.31})
                self.assertEqual(zero, nonzero)

    def test_full_2022_flat_lcdm_fit(self) -> None:
        likelihood = cc(mode="explicit")

        def residual(parameters: np.ndarray) -> np.ndarray:
            omega_m, h0 = parameters
            expansion_rate = Background(omega_m).E(likelihood.z_obs)
            return solve_triangular(
                likelihood.L,
                likelihood.H_obs - h0 * expansion_rate,
                lower=True,
                check_finite=False,
            )

        fit = least_squares(residual, np.array([0.34, 67.0]))
        self.assertTrue(fit.success)
        np.testing.assert_allclose(
            fit.x,
            np.array([0.32279528, 67.81801669]),
            rtol=0.0,
            atol=2e-6,
        )
        self.assertAlmostEqual(2.0 * fit.cost, 14.62932475, places=7)

    def test_bao_cc_inverse_ladder_uses_shared_h0(self) -> None:
        chronometers = cc(mode="explicit")
        bao = DESI_BAO(BAO_MEAN, BAO_COV)

        def joint_residual(parameters: np.ndarray) -> np.ndarray:
            omega_m, h0, sound_horizon = parameters
            background = Background(omega_m)

            cc_theory = h0 * background.E(chronometers.z_obs)
            cc_residual = solve_triangular(
                chronometers.L,
                chronometers.H_obs - cc_theory,
                lower=True,
                check_finite=False,
            )

            bao_template = bao._theory_vector(background)
            bao_theory = (
                bao.C_KMS / (h0 * sound_horizon) * bao_template
            )
            bao_residual = solve_triangular(
                bao.L,
                bao.data_vector - bao_theory,
                lower=True,
                check_finite=False,
            )
            return np.concatenate((cc_residual, bao_residual))

        fit = least_squares(
            joint_residual,
            np.array([0.30, 69.0, 148.0]),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        self.assertTrue(fit.success)
        # Both calibrated amplitudes at the joint optimum must reproduce
        # their independent conditional Gaussian solutions at this geometry.
        bg = Background(fit.x[0])
        h0_hat, _ = chronometers.conditional_h0(bg)
        b_hat, _ = bao.conditional_b(bg)
        np.testing.assert_allclose(fit.x[1:], [h0_hat, bao.C_KMS / (h0_hat * b_hat)],
                                   rtol=0, atol=2e-5)
        self.assertAlmostEqual(2 * fit.cost,
            chronometers.profile_chi2(bg) + bao.profile_chi2(bg), places=8)

        # Confirm that the public scalar and vectorized paths use the same
        # explicit H0 and r_d instead of profiling either term independently.
        combined = CombinedLikelihood([chronometers, bao])
        scalar = combined(
            {
                "Omega_m": fit.x[0],
                "H0": fit.x[1],
                "r_d": fit.x[2],
            }
        )
        batch = combined(
            {
                "Omega_m": np.array([fit.x[0], fit.x[0] + 0.001]),
                "H0": np.array([fit.x[1], fit.x[1]]),
                "r_d": np.array([fit.x[2], fit.x[2]]),
            }
        )
        self.assertAlmostEqual(batch[0], scalar, places=10)

    def test_batched_profile_mode_matches_scalar(self) -> None:
        likelihood = cc(mode="marginalized")
        omega_m = np.array([0.2, 0.3, 0.4])
        background = _BatchedBackground(
            Omega_m=omega_m,
            w0=np.full(3, -1.0),
            wa=np.zeros(3),
            epsilon=np.zeros(3),
            alpha=np.ones(3),
        )
        batched = likelihood.log_likelihood(background)
        scalar = np.array(
            [
                likelihood.log_likelihood(Background(value))
                for value in omega_m
            ]
        )
        np.testing.assert_allclose(batched, scalar, rtol=0.0, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
