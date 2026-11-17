import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
tagging = importlib.import_module("load_smart_tagging")

groups = {}
for e in tagging.load():
    code = (e["code"] or "").strip().lower()
    typ = (e["tags"].get("Type") or "").strip().lower()
    key = code + (" / " + typ if typ else "")
    groups.setdefault(key, []).append((e["end_frame"] - e["start_frame"]) / 25.0)

print("code / type                                   n   min  median   max  (s)")
for key in sorted(groups):
    d = np.array(groups[key])
    print(f"{key:<42} {len(d):>3}  {d.min():5.1f}  {np.median(d):6.1f}  {d.max():5.1f}")
