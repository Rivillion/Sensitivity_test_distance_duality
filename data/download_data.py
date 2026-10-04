"""Fetch the eight pinned upstream inputs and verify all ten manuscript inputs.

Uses only the Python standard library. Author-prepared CC files are never
downloaded or overwritten. URLs and checksums live in UPSTREAM_PROVENANCE.json.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import urllib.error
import urllib.request


DATA_DIR = Path(__file__).resolve().parent
MANIFEST = DATA_DIR / "UPSTREAM_PROVENANCE.json"


def input_path(data_dir, record):
    relative = PurePosixPath(record["local_path"])
    if relative.parts[0] != "data" or ".." in relative.parts or len(relative.parts) < 2:
        raise ValueError(f"Invalid input path: {relative}")
    return Path(data_dir).joinpath(*relative.parts[1:])


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == record["local_sha256"]


def download(path, record):
    """Publish a download only after validating its complete size and hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(
            record["download_url"], headers={"User-Agent": "Sensitivity-test-distance-duality/1"}
        )
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                prefix=".download-", suffix=".tmp", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > record["bytes"]:
                        raise ValueError(f"Unexpected download size for {path.name}")
                    digest.update(block)
                    stream.write(block)
        if size != record["bytes"] or digest.hexdigest() != record["local_sha256"]:
            raise ValueError(f"Downloaded checksum or size does not match for {path.name}")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(data_dir=DATA_DIR, *, check=False, force=False):
    records = json.loads(MANIFEST.read_text())["files"]
    # Validate the bundled author-prepared inputs before making any requests.
    for record in records:
        if record["download_url"] is None:
            path = input_path(data_dir, record)
            if not matches(path, record):
                raise ValueError(
                    f"Bundled author-prepared CC input is missing or changed: {path}. "
                    "Restore it from this release; it cannot be downloaded."
                )
    for record in records:
        path = input_path(data_dir, record)
        if matches(path, record):
            print(f"Verified {record['local_path']}")
            continue
        if check:
            raise ValueError(f"Input is missing or changed: {path}. Run data/download_data.py.")
        if path.exists() and not force:
            raise ValueError(
                f"Existing input has a different checksum: {path}. "
                "Use --force to replace upstream inputs with the pinned versions."
            )
        print(f"Downloading {record['local_path']}", flush=True)
        download(path, record)
    print(f"Verified all {len(records)} manuscript inputs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all inputs without network access.")
    parser.add_argument("--force", action="store_true", help="Replace changed upstream inputs; never replace CC files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Destination data directory (defaults to this script's directory).")
    args = parser.parse_args()
    if args.check and args.force:
        parser.error("--check and --force cannot be combined")
    try:
        prepare(args.data_dir, check=args.check, force=args.force)
    except (OSError, ValueError, urllib.error.URLError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
