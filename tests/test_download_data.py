"""Integrity and failure checks for public data preparation; no network needed."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parent.parent / "data/download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.payload = b"verified upstream input\n"
        self.remote = self.record("raw/sn/union3/input.fits", self.payload, "https://example.org/pinned/input")
        self.cc = self.record("cc.txt", b"author-prepared CC\n", None)
        (self.data / "cc.txt").write_bytes(b"author-prepared CC\n")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"files": [self.remote, self.cc]}))
        self.manifest_patch = patch.object(downloader, "MANIFEST", manifest)
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.destination = downloader.input_path(self.data, self.remote)

    @staticmethod
    def record(name, payload, url):
        return {"local_path": "data/" + name, "bytes": len(payload),
                "local_sha256": hashlib.sha256(payload).hexdigest(), "download_url": url}

    def test_fresh_install_and_offline_reuse(self):
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)) as fetch:
            downloader.prepare(self.data)
            fetch.assert_called_once()
        self.assertEqual(self.destination.read_bytes(), self.payload)
        with patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("Offline")):
            downloader.prepare(self.data)
            downloader.prepare(self.data, check=True)
        self.assertFalse(list(self.data.rglob(".download-*.tmp")))

    def test_corrupt_download_cannot_replace_existing_input(self):
        self.destination.parent.mkdir(parents=True)
        original = b"existing input\n"
        self.destination.write_bytes(original)
        bad = b"X" * len(self.payload)
        with patch.object(downloader.urllib.request, "urlopen", return_value=io.BytesIO(bad)):
            with self.assertRaisesRegex(ValueError, "checksum"):
                downloader.prepare(self.data, force=True)
        self.assertEqual(self.destination.read_bytes(), original)
        self.assertFalse(list(self.data.rglob(".download-*.tmp")))

    def test_interrupted_download_does_not_publish_partial_input(self):
        class Interrupted(io.BytesIO):
            def read(self, size):
                if self.tell():
                    raise OSError("Connection interrupted")
                return super().read(5)

        with patch.object(downloader.urllib.request, "urlopen", return_value=Interrupted(self.payload)):
            with self.assertRaisesRegex(OSError, "interrupted"):
                downloader.prepare(self.data)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.data.rglob(".download-*.tmp")))

    def test_existing_mismatch_requires_explicit_force(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"changed")
        with patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("Must not fetch")):
            with self.assertRaisesRegex(ValueError, "--force"):
                downloader.prepare(self.data)
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                downloader.prepare(self.data, check=True)

    def test_cc_is_never_overwritten_even_with_force(self):
        path = self.data / "cc.txt"
        path.write_bytes(b"changed CC")
        with patch.object(downloader.urllib.request, "urlopen", side_effect=AssertionError("Must not fetch")):
            with self.assertRaisesRegex(ValueError, "author-prepared CC"):
                downloader.prepare(self.data, force=True)
        self.assertEqual(path.read_bytes(), b"changed CC")


if __name__ == "__main__":
    unittest.main()
