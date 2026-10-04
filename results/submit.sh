#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FAMILY="${1:?Usage: bash results/submit.sh baseline|calibration|robustness [sbatch options]}"
shift
case "$FAMILY" in
    baseline) ARRAY=0-37 ;;
    calibration) ARRAY=0-2 ;;
    robustness) ARRAY=0-15 ;;
    *) echo "Unknown inventory: $FAMILY" >&2; exit 2 ;;
esac
cd "$RELEASE_ROOT"
mkdir -p results/run_logs
# Persistent shared storage; new inference never changes the released archives.
OUTPUT_ROOT="$RELEASE_ROOT/results/reruns/$FAMILY"
exec sbatch --chdir="$RELEASE_ROOT" --array="$ARRAY" "$@" \
    "$RELEASE_ROOT/results/sampling.slurm" "$FAMILY" "$OUTPUT_ROOT"
