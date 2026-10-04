"""Validate the supplied CC covariance without reconstructing or inflating it."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import LinAlgError
from src.likelihoods import CosmicChronometer

DATA = Path(__file__).resolve().parents[1] / "data"


class CCCovarianceTests(unittest.TestCase):
    def test_supplied_covariance_is_used_exactly(self):
        cc = CosmicChronometer(DATA / "Hz_full_2022_mod.txt", DATA / "covmat_Hz_full_2022.txt")
        supplied = np.loadtxt(DATA / "covmat_Hz_full_2022.txt")
        self.assertEqual(supplied.shape, (33, 33))
        np.testing.assert_array_equal(cc.covariance, supplied)
        np.testing.assert_allclose(cc.L @ cc.L.T, supplied, rtol=1e-13, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(supplied).min(), 0)
        self.assertFalse(np.allclose(np.diag(supplied), cc.err_hz**2))
        self.assertFalse(np.allclose(cc.covariance, supplied + np.diag(cc.err_hz**2)))
        self.assertGreater(np.count_nonzero(supplied - np.diag(np.diag(supplied))), 0)

    def test_invalid_covariance_is_rejected(self):
        table = np.array([[0.1, 70, 2], [0.2, 80, 3]])
        invalid = [np.eye(3), np.array([[1, np.nan], [np.nan, 1]]),
                   np.array([[1, 0.5], [0, 1]]), np.array([[1, 2], [2, 1]])]
        with tempfile.TemporaryDirectory() as tmp:
            d, c = Path(tmp) / "data.txt", Path(tmp) / "cov.txt"
            np.savetxt(d, table)
            for covariance in invalid:
                with self.subTest(covariance=covariance):
                    np.savetxt(c, covariance)
                    with self.assertRaises((ValueError, LinAlgError)):
                        CosmicChronometer(d, c)

    def test_covariance_off_diagonal_changes_likelihood(self):
        from src.background import Background
        from scipy.linalg import solve_triangular
        cc = CosmicChronometer(DATA / "Hz_full_2022_mod.txt", DATA / "covmat_Hz_full_2022.txt")
        bg = Background(0.3, H0=70)
        residual = cc.H_obs - bg.H0 * bg.E(cc.z_obs)
        expected = residual @ np.linalg.solve(cc.covariance, residual)
        diagonal = np.sum(residual**2 / np.diag(cc.covariance))
        self.assertAlmostEqual(-2 * cc.log_likelihood(bg), expected, places=11)
        self.assertGreater(abs(expected - diagonal), 1)


if __name__ == "__main__":
    unittest.main()
