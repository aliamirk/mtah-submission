import os, sys, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
tagging = importlib.import_module("load_smart_tagging")
pm = importlib.import_module("possession_metrics")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
dead = [(s, e) for s, e, c in pm.segments(gt, runs) if c == pm.D]

for e in tagging.load():
    if (e["code"] or "").strip().lower() != "set pieces":
        continue
    if e["end_frame"] < 11185 or e["start_frame"] > 18685:
        continue
    print("%s | team %s | tag %d to %d" % (e["tags"].get("Type"), e["tags"].get("Team"),
                                         e["start_frame"], e["end_frame"]))
    for s, t in dead:
        if t > e["start_frame"] and s < e["end_frame"]:
            print("    dead stretch from %+.1f s to %+.1f s relative to tag start"
                  % ((s - e["start_frame"]) / 25.0, (t - e["start_frame"]) / 25.0))
