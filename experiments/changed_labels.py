import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
a = pm.label_codes(pm.read_label_csv(sys.argv[1]))
b = pm.label_codes(pm.read_label_csv(sys.argv[2]))

graded = gt != pm.X
diff = graded & (a != b)
b_right = diff & (b == gt)
a_right = diff & (a == gt)
print("graded frames changed: %d" % diff.sum())
print("  B right, A wrong : %d" % (b_right.sum()))
print("  A right, B wrong : %d" % (a_right.sum()))
print("  both wrong       : %d" % (diff & ~b_right & ~a_right).sum())

segs = pm.segments(gt, pm.valid_runs(gt))
net = np.array([int(b_right[s:e].sum()) - int(a_right[s:e].sum()) for s, e, _ in segs])
moved = np.array([int(diff[s:e].sum()) for s, e, _ in segs])
rng = np.random.RandomState(0)
draws = np.array([net[p].sum() for p in (rng.randint(0, len(segs), len(segs)) for _ in range(5000))])
print("net frames gained by B: %+d   90%% interval over GT segments: %+d to %+d"
      % (net.sum(), np.percentile(draws, 5), np.percentile(draws, 95)))
print("segments where anything changed: %d of %d" % (int((moved > 0).sum()), len(segs)))
