#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
cd "$REPO"
rm -f submission.csv
"$PY" main.py > /tmp/main_orig.log 2>&1
ORIG=$(shasum submission.csv | cut -d' ' -f1)
rm -rf /tmp/clean_clone
git clone -q "$REPO" /tmp/clean_clone
rm -rf /tmp/clean_clone/data
ln -s "$REPO/data" /tmp/clean_clone/data
rm -f /tmp/clean_clone/submission.csv
( cd /tmp/clean_clone && "$PY" main.py > /tmp/main_clone.log 2>&1 )
CLONE=$(shasum /tmp/clean_clone/submission.csv | cut -d' ' -f1)
echo "original: $ORIG"
echo "clone:    $CLONE"
if [ "$ORIG" = "$CLONE" ]; then echo "PASS: identical file"; else echo "FAIL: files differ"; fi
