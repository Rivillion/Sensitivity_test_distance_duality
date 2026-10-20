#!/usr/bin/env bash
#SBATCH --job-name=distance-duality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/run_logs/sampling-%A_%a.out
#SBATCH --error=results/run_logs/sampling-%A_%a.err
set -euo pipefail
# results/submit.sh selects the array and starts this script in the release root.
PYTHON_EXECUTABLE="${SAMPLING_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
    echo "Create .venv or set SAMPLING_PYTHON to an executable Python." >&2
    exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
FAMILY="${1:?Missing inventory: baseline, calibration or robustness}"
OUTPUT_ROOT="${2:?Missing persistent output root}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?Missing array task ID}"
if [[ "$FAMILY" == calibration ]]; then
    CASES=(bao pantheon joint joint-alpha1)
    exec srun --ntasks=1 "$PYTHON_EXECUTABLE" -u -m results.sample_calibration \
        --only "${CASES[$TASK_ID]}" --cores "${SLURM_CPUS_PER_TASK:-4}" --output-root "$OUTPUT_ROOT"
else
    exec srun --ntasks=1 "$PYTHON_EXECUTABLE" -u -m "results.sample_$FAMILY" \
        --task-id "$TASK_ID" --cores "${SLURM_CPUS_PER_TASK:-4}" --output-root "$OUTPUT_ROOT"
fi
