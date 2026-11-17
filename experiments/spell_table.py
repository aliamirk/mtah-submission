import os
import sys
import importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
pm = importlib.import_module("possession_metrics")
baseline = importlib.import_module("baseline")

gt_df = pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt",
                                       "BAR-ATH_possession_gt_public.csv"))
gt = pm.label_codes(gt_df)
codes = np.load(os.path.join(ROOT, "experiments", "cache", "codes.npy"))
pred = pm.label_codes(baseline.smooth(codes, 50))

runs = pm.valid_runs(gt)
filled = pm.fill_abstentions(pred, runs)
g = pm.possession_spells(gt, runs)
p = pm.possession_spells(filled, runs)

print("true possession (start end team length) -> best same-team IoU")
for gs, ge, gc in g:
    best = 0.0
    for ps, pe, pc in p:
        if pc != gc or pe <= gs or ge <= ps:
            continue
        iou = (min(ge, pe) - max(gs, ps)) / (max(ge, pe) - min(gs, ps))
        best = max(best, iou)
    print(
        f"{gs:>6} {ge:>6} {pm.LABELS[gc]}  {(ge - gs) / 25:5.1f}s   best IoU {best:.2f}")
print(len(g), "true,", len(p), "predicted")
