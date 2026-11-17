#!/usr/bin/env bash
set -euo pipefail
NAME="$1"
BASE="${2:-}"
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
"$PY" -c "import kloppy" || { echo "kloppy missing in .venv"; exit 1; }
rm -f submission.csv
"$PY" main.py
test -s submission.csv
"$PY" evaluation/validate_submission.py submission.csv
cp submission.csv "evaluation/results/${NAME}.csv"
if [ -n "$BASE" ]; then
  "$PY" experiments/changed_labels.py "evaluation/results/${BASE}.csv" "evaluation/results/${NAME}.csv"
  "$PY" experiments/paired_compare.py "evaluation/results/${BASE}.csv" "evaluation/results/${NAME}.csv"
fi
"$PY" evaluation/run_evaluation.py --no-detail
