"""Checks for relocation, archived runs, and manuscript table corrections."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src import sampler
from src.release_io import signatures_match, validate_completed, verify_inputs

ROOT = Path(__file__).resolve().parent.parent


class ReleaseTests(unittest.TestCase):
    def test_complete_job_inventory_and_canonical_output_paths(self):
        jobs = json.loads((ROOT / "results/jobs_baseline.json").read_text())
        self.assertEqual(len(jobs), 38)
        self.assertEqual(len({j["output"] for j in jobs}), 38)
        with patch.object(sampler, "RUNS_DIR", ROOT / "results/runs/baseline"):
            for job in jobs:
                self.assertEqual(sampler.run_directory(job["model"], job["likelihoods"]), ROOT / job["output"])

    def test_relocation_requires_same_input_hashes(self):
        saved = {"cc": {"data_path": "/original/data.txt", "data_sha256": "abc"}, "nlive": 3000}
        moved = {"cc": {"data_path": "/release/data.txt", "data_sha256": "abc"}, "nlive": 3000}
        self.assertTrue(signatures_match(saved, moved))
        moved["cc"]["data_sha256"] = "different"
        self.assertFalse(signatures_match(saved, moved))

    def test_completed_posterior_does_not_require_checkpoint_or_invoke_nautilus(self):
        _, args = sampler.parse_arguments(["--model", "lcdm", "--likelihoods", "desi", "--cores", "1"])
        directory = ROOT / "results/runs/baseline/desi/lcdm"
        self.assertFalse((directory / "sampler.hdf5").exists())
        with patch.object(sampler, "RUNS_DIR", ROOT / "results/runs/baseline"), \
             patch("nautilus.Sampler", side_effect=AssertionError("Must skip completed sampling")):
            self.assertEqual(sampler.run_sampler(args), 0)
        incompatible = sampler.resume_signature("lcdm", ("desi",), 1000, True)
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_completed(directory, incompatible)

    def test_data_checksums(self):
        verify_inputs()

    def test_audited_boundary_and_rounding_corrections(self):
        cuts = json.loads((ROOT / "results/minima/cuts.json").read_text())
        case = cuts["dovekie"]["0.100"]
        self.assertEqual(f"{case['fits']['lcdm']['chi2']-case['fits']['w0wa']['chi2']:.2f}", "0.95")
        self.assertEqual(f"{case['fits']['lcdm-epsilon']['chi2']-case['fits']['w0wa-epsilon']['chi2']:.2f}", "0.82")
        self.assertAlmostEqual(case["fits"]["w0wa"]["parameters"]["Omega_m"], .1, places=7)
        tracers = json.loads((ROOT / "results/minima/tracers.json").read_text())
        case = tracers["union3"]["LRG1"]
        self.assertEqual(f"{100*(1-case['delta_de_given_tilt']/case['delta_de']):.0f}", "54")


if __name__ == "__main__":
    unittest.main()
