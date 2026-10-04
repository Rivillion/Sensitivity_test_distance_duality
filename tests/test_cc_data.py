"""CC table validation and row alignment with the supplied covariance."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.likelihoods import CosmicChronometer

DATA = Path(__file__).resolve().parents[1] / "data"


class CCDataTests(unittest.TestCase):
    def test_default_table_preserves_all_rows(self):
        from src.sampler import initialize_likelihoods
        cc = initialize_likelihoods(("cc",)).likelihoods[0]
        table = np.loadtxt(DATA / "Hz_full_2022_mod.txt")
        self.assertEqual(len(cc.z_obs), 33)
        np.testing.assert_array_equal(cc.z_obs, table[:, 0])
        np.testing.assert_array_equal(cc.H_obs, table[:, 1])
        np.testing.assert_array_equal(cc.err_hz, table[:, 2])
        self.assertEqual(cc.z_obs[0], 0.07)
        self.assertEqual(cc.z_obs[-1], 1.965)

    def test_bad_tables_fail_before_use(self):
        good = np.array([[0.1, 70, 2], [0.2, 80, 3]])
        invalid = [good[:, :2], good[::-1], np.array([[0.1, 70, 2], [0.1, 80, 3]])]
        for column, value in ((0, -0.1), (1, 0), (2, -1), (1, np.inf)):
            changed = good.copy(); changed[0, column] = value; invalid.append(changed)
        with tempfile.TemporaryDirectory() as tmp:
            d, c = Path(tmp) / "data.txt", Path(tmp) / "cov.txt"
            np.savetxt(c, np.eye(2))
            for table in invalid:
                with self.subTest(table=table):
                    np.savetxt(d, table)
                    with self.assertRaises(ValueError):
                        CosmicChronometer(d, c)


if __name__ == "__main__":
    unittest.main()
