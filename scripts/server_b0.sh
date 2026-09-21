#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
archive="${1:-b0_jobs.zip}"
run="runs/b0_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p runs
# The Python entrypoint creates the run directory after validating the private input package.
# The separate log survives validation failures as well as model failures.
python -u -m egc.b0 run --archive "$archive" --run-dir "$run" 2>&1 | tee "${run}.log"
cp "${run}.log" "$run/run.log"
printf '\nReturn: %s/b0_results.zip\nLog: %s.log\n' "$run" "$run"
