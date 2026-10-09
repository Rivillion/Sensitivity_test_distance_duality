"""Regenerate all five manuscript figures from saved weighted posteriors."""
import subprocess
import sys
from src.release_io import ROOT, verify_inputs


def main():
    verify_inputs()
    for number in range(1, 6):
        subprocess.run([sys.executable, '-m', f'figures.figure{number}'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
