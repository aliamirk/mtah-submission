#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
OUT="experiments/cache/sens"
mkdir -p "$OUT"
run() {
  NAME="$1"; KEY="$2"; VAL="$3"
  sed -e "s/^${KEY}:.*/${KEY}: ${VAL}/" config.yaml > "/tmp/cfg_${NAME}.yaml"
  grep -q "^${KEY}: ${VAL}" "/tmp/cfg_${NAME}.yaml" || echo "${KEY}: ${VAL}" >> "/tmp/cfg_${NAME}.yaml"
  rm -f submission.csv
  "$PY" main.py --config "/tmp/cfg_${NAME}.yaml" > "/tmp/run_${NAME}.log" 2>&1
  cp submission.csv "$OUT/${NAME}.csv"
  echo "done ${NAME}"
}
run tagcap1 dead_ball_max_duration_frames 1
run pen25 switch_penalty 25
run pen38 switch_penalty 38
run thr12 quiet_speed_threshold 1.2
run thr16 quiet_speed_threshold 1.6
run min8 quiet_min_seconds 8
run min20 quiet_min_seconds 20
rm -f submission.csv
"$PY" main.py > /tmp/run_final.log 2>&1
cp submission.csv "$OUT/final.csv"
echo "done final"
