#!/usr/bin/env bash
# Wrapper that forces the correct stdlib lib-dynload before running main.py.
# Needed because the venv was created from Anaconda Python, which leaks
# /opt/anaconda3/lib/python3.13/lib-dynload into sys.path, causing an
# incompatible _ctypes to shadow the system one.
SYSTEM_DYNLOAD="/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/lib-dynload"
PYTHONPATH="${SYSTEM_DYNLOAD}${PYTHONPATH:+:$PYTHONPATH}" \
  "$(dirname "$0")/.venv/bin/python" main.py "$@"
