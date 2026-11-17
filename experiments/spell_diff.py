import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
g = pm.possession_spells(gt, runs)


def best_iou(path):
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(path)), runs)
    p = pm.possession_spells(pred, runs)
    out = []
    for gs, ge, gc in g:
        best = 0.0
        for ps, pe, pc in p:
            if pc != gc or pe <= gs or ge <= ps:
                continue
            best = max(best, (min(ge, pe) - max(gs, ps)) / float(max(ge, pe) - min(gs, ps)))
        out.append(best)
    return np.array(out), len(p)


a, na = best_iou(sys.argv[1])
b, nb = best_iou(sys.argv[2])
lens = np.array([e - s for s, e, _ in g])
short = lens <= 90
for name, m in (("short (<= 90 frames)", short), ("long (> 90 frames)", ~short)):
    print("%-22s n=%2d  best IoU >= 0.5: A %2d  B %2d   mean best IoU: A %.2f  B %.2f"
          % (name, m.sum(), (a[m] >= 0.5).sum(), (b[m] >= 0.5).sum(), a[m].mean(), b[m].mean()))
print("predicted spells: A %d  B %d" % (na, nb))
