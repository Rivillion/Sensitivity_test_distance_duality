"""Independent check of the Section IV linear-in-redshift equation of state."""

import unittest

import numpy as np
from scipy.integrate import quad

from src.background import Background


class LinearEquationOfStateTests(unittest.TestCase):
    def test_density_factor_and_distance(self):
        omega_m, w0, wa = 0.29, -0.8, 0.25
        z = 1.15
        background = Background(Omega_m=omega_m, w0=w0, wa=wa, eos="linear")

        def analytic_e(redshift):
            factor = (1 + redshift) ** (3 * (1 + w0 - wa)) * np.exp(3 * wa * redshift)
            return np.sqrt(omega_m * (1 + redshift) ** 3 + (1 - omega_m) * factor)

        self.assertAlmostEqual(background.E(z), analytic_e(z), places=12)
        integral, _ = quad(lambda redshift: 1 / analytic_e(redshift), 0, z)
        self.assertAlmostEqual(background.I(z), integral, delta=1e-8)


if __name__ == "__main__":
    unittest.main()
