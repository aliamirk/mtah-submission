import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(sys.argv[1])), runs)
g = pm.possession_spells(gt, runs)
p = pm.possession_spells(pred, runs)
name = lambda c: pm.LABELS[int(c)]

print("gt_start frames team | best same-team prediction: start_err end_err IoU | prediction at the GT start / end")
rows = []
for gs, ge, gc in g:
    best, bi = 0.0, None
    for i, (ps, pe, pc) in enumerate(p):
        if pc != gc or pe <= gs or ge <= ps:
            continue
        iou = (min(ge, pe) - max(gs, ps)) / float(max(ge, pe) - min(gs, ps))
        if iou > best:
            best, bi = iou, i
    here = "%s / %s" % (name(pred[gs]), name(pred[ge - 1]))
    if bi is None:
        print("%8d %6d    %s | none | %s" % (gs, ge - gs, name(gc), here))
        rows.append((ge - gs, None, None))
    else:
        ps, pe, _ = p[bi]
        print("%8d %6d    %s | start %+5d  end %+5d  %.2f | %s"
              % (gs, ge - gs, name(gc), ps - gs, pe - ge, best, here))
        rows.append((ge - gs, ps - gs, pe - ge))

longs = [r for r in rows if r[0] > 90 and r[1] is not None]
if longs:
    se = np.array([r[1] for r in longs])
    ee = np.array([r[2] for r in longs])
    print("\nlong spells with a match: %d" % len(longs))
    print("start error, frames: median %+d, median absolute %d" % (np.median(se), np.median(np.abs(se))))
    print("end error, frames:   median %+d, median absolute %d" % (np.median(ee), np.median(np.abs(ee))))
