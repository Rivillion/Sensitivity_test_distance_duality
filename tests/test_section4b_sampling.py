"""Physical calibration and full-likelihood checks for Sec. IV.B."""
import unittest

import numpy as np

from src.background import Background, c_kms
from src.sample_section4b import (
    UnitAlphaLikelihood, case_prior, derived_coordinates, unit_alpha_intercept,
)
from src.sampler import initialize_likelihoods


class Section4BTests(unittest.TestCase):
    def test_gaussian_calibrations_are_sampled(self):
        for case in ("bao", "pantheon", "joint", "joint-alpha1"):
            prior, names, specs = case_prior(case)
            physical = prior.unit_to_physical(np.full((1, len(names)), .5))
            for name in ("r_d", "M_B"):
                if name in names:
                    self.assertEqual(specs[name]["distribution"], "Gaussian")
                    self.assertAlmostEqual(physical[0, names.index(name)], specs[name]["mean"])

    def test_h0_and_alpha_orientation_from_physical_intercept(self):
        h0 = np.array([69., 71.])
        alpha = np.array([.94, 1.06])
        rd = np.array([147., 148.])
        mb = np.array([-19.253, -19.22])
        nu = mb + 25 + 5*np.log10(c_kms/h0) + 5*np.log10(alpha)
        names = ("H0rd", "nu", "r_d", "M_B")
        samples = np.column_stack((h0*rd, nu, rd, mb))
        derived = derived_coordinates(samples, names, "joint")
        np.testing.assert_allclose(derived["H0"], h0, rtol=1e-13)
        np.testing.assert_allclose(derived["alpha"], alpha, rtol=1e-13)
        np.testing.assert_allclose(derived["a"], 5*np.log10(alpha), atol=1e-13)
        separate = derived_coordinates(samples, names, "pantheon")
        np.testing.assert_allclose(separate["H0"], h0/alpha, rtol=1e-13)

    def test_full_physical_likelihood_matches_sampled_coordinates(self):
        # Independently build the physical m_B prediction (no profiled nu).
        likelihood = initialize_likelihoods(("pantheon", "desi"), zmin=.01)
        h0, rd, mb, alpha, epsilon = 69., 147.09, -19.253, .942, -.029
        nu = mb + 25 + 5*np.log10(c_kms/h0) + 5*np.log10(alpha)
        params = dict(Omega_m=.30, epsilon=epsilon, H0rd=h0*rd, nu=nu, r_d=rd, M_B=mb)
        bg = Background(Omega_m=.30, epsilon=epsilon, alpha=alpha, H0=h0)
        bg.r_d = rd
        sn, bao = likelihood.likelihoods
        from scipy.linalg import solve_triangular
        prediction = mb + 25 + 5*np.log10(c_kms/h0 * bg.I(sn.z_cmb) * (1+sn.z_hel) * alpha)
        prediction += 5*epsilon*np.log10(1+sn.z_cmb)
        residual = solve_triangular(sn.L, sn.mb_obs-prediction, lower=True)
        physical_loglike = -.5*np.dot(residual, residual) + bao.explicit_log_likelihood(bg)
        self.assertAlmostEqual(likelihood(params), physical_loglike, places=9)
        batch = {k: np.repeat(v, 2) for k, v in params.items()}
        np.testing.assert_allclose(likelihood(batch), physical_loglike, atol=1e-9, rtol=0)
        # Independent calibrations are priors; they cannot be double-counted.
        changed = dict(params, r_d=149., M_B=-19.3)
        self.assertEqual(likelihood(params), likelihood(changed))

    def test_unit_alpha_ties_nu_to_calibrations(self):
        _, names, _ = case_prior("joint-alpha1")
        self.assertEqual(names, ("Omega_m", "epsilon", "H0rd", "r_d", "M_B"))
        h0, rd, mb = np.array([67., 69., 71.]), np.array([146.8, 147.09, 147.4]), np.array([-19.28, -19.253, -19.22])
        nu = unit_alpha_intercept(h0*rd, rd, mb)
        derived = derived_coordinates(np.column_stack((h0*rd, nu, rd, mb)), ("H0rd", "nu", "r_d", "M_B"), "joint")
        np.testing.assert_allclose(derived["alpha"], 1, rtol=1e-13)
        fixed = derived_coordinates(np.column_stack((h0*rd, rd, mb)), ("H0rd", "r_d", "M_B"), "joint-alpha1")
        np.testing.assert_allclose(fixed["nu"], nu, rtol=1e-13)
        self.assertNotIn("alpha", fixed)

        likelihood = initialize_likelihoods(("pantheon", "desi"), zmin=.01)
        params = dict(Omega_m=.30, epsilon=-.05, H0rd=69.*147.09, r_d=147.09, M_B=-19.253)
        explicit = likelihood(dict(params, nu=unit_alpha_intercept(params["H0rd"], params["r_d"], params["M_B"])))
        self.assertEqual(UnitAlphaLikelihood(likelihood)(params), explicit)
        batch = {k: np.repeat(v, 2) for k, v in params.items()}
        np.testing.assert_allclose(UnitAlphaLikelihood(likelihood)(batch), explicit, atol=1e-9, rtol=0)
        # At alpha = 1 the calibrations change the SN prediction.
        self.assertNotEqual(UnitAlphaLikelihood(likelihood)(dict(params, M_B=-19.3)), explicit)


if __name__ == "__main__":
    unittest.main()
