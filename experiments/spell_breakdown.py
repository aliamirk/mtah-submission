import os
import sys
import importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
pm = importlib.import_module("possession_metrics")
baseline = importlib.import_module("baseline")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
codes = np.load(os.path.join(ROOT, "experiments", "cache", "codes.npy"))
pred = pm.label_codes(baseline.smooth(codes, 50))

runs = pm.valid_runs(gt)
g = pm.possession_spells(gt, runs)
p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)

print("start   secs team | right  other  dead | noBall | pieces")
for gs, ge, gc in g:
    seg = pred[gs:ge]
    right = np.mean(seg == gc)
    dead = np.mean(seg == pm.D)
    other = 1.0 - right - dead
    noball = np.mean(codes[gs:ge] == -1)
    pieces = sum(1 for ps, pe, _ in p if pe > gs and ps < ge)
    print(f"{gs:>6} {(ge - gs) / 25:5.1f}  {pm.LABELS[gc]}   | "
          f"{right:5.2f}  {other:5.2f}  {dead:5.2f} | {noball:5.2f}  | {pieces:>3}")

print("\ndead stretches: start  secs | predicted D | no ball tracked")
for s, e, c in pm.segments(gt, runs):
    if c == pm.D:
        print(f"{s:>6} {(e - s) / 25:5.1f}   | {np.mean(pred[s:e] == pm.D):5.2f}       "
              f"| {np.mean(codes[s:e] == -1):5.2f}")
