"""Regenerate optimized minima used by manuscript comparison tables.

Commands: ladder, controls, tracers, cuts, information, cc-ladder, non-sn.
These calculations minimize likelihoods and never provide posterior intervals.
"""
import argparse
import json
from pathlib import Path
from src import minima
from src.release_io import verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=minima.COMMANDS)
    parser.add_argument('--output-dir', type=Path, default=minima.OUT)
    args = parser.parse_args()
    verify_inputs()
    minima.OUT = args.output_dir
    result = minima.COMMANDS[args.command]()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.command}.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
