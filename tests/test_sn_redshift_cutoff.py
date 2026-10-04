"""Tests for Pantheon+ and DES-Dovekie redshift filtering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.likelihoods import DES_Dovekie, PantheonPlus


class SupernovaRedshiftCutoffTests(unittest.TestCase):
    def test_pantheon_retains_boundary_and_excludes_calibrators(self) -> None:
        covariance = np.array(
            [
                [0.10, 0.01, 0.00, 0.00],
                [0.01, 0.12, 0.02, 0.00],
                [0.00, 0.02, 0.15, 0.01],
                [0.00, 0.00, 0.01, 0.11],
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "pantheon.dat"
            covariance_path = root / "pantheon.cov"
            data_path.write_text(
                "zHD zHEL m_b_corr IS_CALIBRATOR\n"
                "0.049 0.049 16.0 0\n"
                "0.050 0.050 17.0 0\n"
                "0.070 0.070 18.0 1\n"
                "0.100 0.100 19.0 0\n",
                encoding="utf-8",
            )
            np.savetxt(
                covariance_path,
                np.concatenate(([len(covariance)], covariance.ravel())),
            )

            likelihood = PantheonPlus(
                data_path,
                covariance_path,
                zmin=0.05,
            )

        np.testing.assert_array_equal(likelihood.mask, [1, 3])
        np.testing.assert_allclose(likelihood.z_cmb, [0.05, 0.10])
        np.testing.assert_allclose(
            likelihood.L @ likelihood.L.T,
            covariance[np.ix_([1, 3], [1, 3])],
        )

    def test_dovekie_slices_covariance_not_precision(self) -> None:
        factor = np.array(
            [
                [1.2, 0.0, 0.0, 0.0],
                [0.3, 1.1, 0.0, 0.0],
                [0.2, 0.4, 1.3, 0.0],
                [0.1, 0.2, 0.3, 1.4],
            ]
        )
        covariance = factor @ factor.T
        precision = np.linalg.inv(covariance)
        retained = np.array([1, 2, 3])

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "dovekie.csv"
            covariance_path = root / "dovekie.npz"
            data_path.write_text(
                "zHD zHEL MU\n"
                "0.050 0.050 36.0\n"
                "0.100 0.100 37.0\n"
                "0.200 0.200 39.0\n"
                "0.300 0.300 41.0\n",
                encoding="utf-8",
            )
            np.savez(
                covariance_path,
                nsn=np.array([len(covariance)]),
                cov=precision[np.triu_indices(len(covariance))],
            )

            likelihood = DES_Dovekie(
                data_path,
                covariance_path,
                zmin=0.1,
            )

        selected_covariance = likelihood.L @ likelihood.L.T
        expected_covariance = covariance[np.ix_(retained, retained)]
        incorrectly_sliced = np.linalg.inv(
            precision[np.ix_(retained, retained)]
        )
        np.testing.assert_array_equal(likelihood.mask, retained)
        np.testing.assert_allclose(likelihood.z_cmb, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            selected_covariance,
            expected_covariance,
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertFalse(
            np.allclose(selected_covariance, incorrectly_sliced)
        )

    def test_cutoff_rejects_empty_samples(self) -> None:
        covariance = np.eye(1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pantheon_data = root / "pantheon.dat"
            pantheon_covariance = root / "pantheon.cov"
            pantheon_data.write_text(
                "zHD zHEL m_b_corr IS_CALIBRATOR\n"
                "0.010 0.010 16.0 0\n",
                encoding="utf-8",
            )
            np.savetxt(
                pantheon_covariance,
                np.concatenate(([1], covariance.ravel())),
            )
            with self.assertRaisesRegex(ValueError, "no supernovae"):
                PantheonPlus(
                    pantheon_data,
                    pantheon_covariance,
                    zmin=0.02,
                )


if __name__ == "__main__":
    unittest.main()
