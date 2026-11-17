import os, sys, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
tagging = importlib.import_module("load_smart_tagging")

PAIRS = [("build up", "defending in attacking third"),
         ("progression", "defending in middle third"),
         ("creating chances", "defending in defensive third"),
         ("attacking transition", "defensive transition")]

events = tagging.load()
for att, dfn in PAIRS:
    a = [e for e in events if (e["code"] or "").strip().lower() == att]
    d = [e for e in events if (e["code"] or "").strip().lower() == dfn]
    hit = 0
    for e in a:
        for f in d:
            same = (e["start_frame"], e["end_frame"]) == (f["start_frame"], f["end_frame"])
            if same and e["tags"].get("Team") != f["tags"].get("Team"):
                hit += 1
                break
    print(f"{att:<22} {len(a):>3} tags, {hit:>3} have an exact mirror in '{dfn}' (same frames, other team)")
