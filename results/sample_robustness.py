"""Sample the 13 tracer-deletion and redshift-cut LambdaCDM+epsilon fits."""
import argparse
import json
from pathlib import Path

from src import robustness, sampler
from src.release_io import verify_inputs

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=range(13))
    parser.add_argument("--cores", type=sampler.positive_integer, default=4)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/runs/robustness")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    jobs = json.loads((ROOT / "results/jobs_robustness.json").read_text())
    selected = jobs if args.task_id is None else [jobs[args.task_id]]
    verify_inputs()
    for job in selected:
        print(f"[{job['task_id']:02d}] {job['case_id']}", flush=True)
        if not args.list:
            status = robustness.run(job, args.cores, args.output_root.resolve(), args.fresh)
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
