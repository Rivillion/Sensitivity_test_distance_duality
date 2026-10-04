"""Run the 38 baseline fits, or select one task for a cluster array."""
import argparse
import json
from pathlib import Path

from src import sampler
from src.release_io import verify_inputs

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(38))
    parser.add_argument("--cores", type=sampler.positive_integer, default=4)
    parser.add_argument("--nlive", type=sampler.positive_integer, default=3000)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/runs/baseline")
    parser.add_argument("--fresh", action="store_true", help="Replace the selected runs in the output root")
    parser.add_argument("--list", action="store_true", help="Show jobs without sampling")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "results/jobs_baseline.json").read_text())
    selected = jobs if args.task_id is None else [jobs[args.task_id]]
    verify_inputs()
    sampler.RUNS_DIR = args.output_root.resolve()
    for job in selected:
        command = ["--model", job["model"], "--likelihoods", *job["likelihoods"],
                   "--nlive", str(args.nlive), "--cores", str(args.cores),
                   "--seed", str(job["seed"]), "--discard-exploration"]
        if args.fresh:
            command.append("--fresh")
        print(f"[{job['task_id']:02d}] {'+'.join(job['likelihoods'])}: {job['model']}", flush=True)
        if not args.list:
            status = sampler.main(command)
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
