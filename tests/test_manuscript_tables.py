"""Regenerate the tables into a temporary directory and compare with the manuscript."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from results import make_tables


class ManuscriptTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        out = Path(cls.directory.name)
        with patch.object(sys, "argv", ["make_tables", "--output-dir", str(out)]), \
                contextlib.redirect_stdout(io.StringIO()):
            make_tables.main()
        cls.tables = {path.stem: json.loads(path.read_text()) for path in out.glob("*.json")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def rows(self, number):
        return self.tables[f"table_{number:02d}"]["rows"]

    def test_outputs_follow_manuscript_numbering(self):
        expected = {f"table_{n:02d}" for n in (1, 3, 4, 5, 6, 7, 8, 9, 10)} | {"section4a", "section4b"}
        self.assertTrue(expected.issubset(self.tables))
        self.assertNotIn("table_02", self.tables)

    def test_table_i_samples(self):
        self.assertEqual([row[1] for row in self.rows(1)], [1580, 1820, 22, 22])
        self.assertEqual(self.rows(1)[0][2], "0.010--2.261")

    def test_tables_iii_iv_v(self):
        self.assertEqual(self.rows(3)[0][1], "4.65 / 1.66")
        self.assertEqual(self.rows(3)[1][1:], ["5.06 / 1.75", "2.18 / 0.96", "57"])
        self.assertEqual(self.rows(4)[1][1:], ["3.36", "1.36"])
        self.assertEqual(self.rows(5)[2][1:], ["10.03", "5.75", "6.47", "5.20", "8.40"])

    def test_table_vi_tracers(self):
        rows = {row[0]: row[1:] for row in self.rows(6)}
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows["LRG1"][6:9], ["9.97", "4.64", "54"])
        self.assertEqual(rows["LRG3+ELG1"][9:], ["6.21", "5.82", "6"])

    def test_table_vii_cutoffs(self):
        epsilon = [row[3] for row in self.rows(7)]
        self.assertEqual(epsilon, [r"$-0.029 \pm 0.017$", r"$-0.025 \pm 0.017$", r"$-0.029 \pm 0.019$",
                                   r"$-0.029 \pm 0.022$", r"$-0.029 \pm 0.015$", r"$-0.021 \pm 0.017$",
                                   r"$-0.008 \pm 0.020$"])

    def test_calibrated_and_cc_quantities(self):
        section4a = self.tables["section4a"]
        self.assertEqual(f"{section4a['non_sn_delta_de']['cc']:.2f}", "0.22")
        sigmas = [(round(v["n_sigma"], 1), round(v["n_sigma_given_tilt"], 1))
                  for v in section4a["bao_sn_cc_significance"].values()]
        self.assertEqual(sigmas, [(1.7, 0.9), (2.1, 1.4), (2.6, 1.8), (2.0, 1.7)])
        self.assertEqual(f"{self.tables['section4b']['alpha_deviation_from_unity_sigma']:.1f}", "4.1")


if __name__ == "__main__":
    unittest.main()
