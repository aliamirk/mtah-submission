import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
for path in sys.argv[1:]:
    p = pm.label_codes(pm.read_label_csv(path))
    both = ((gt == pm.H) | (gt == pm.A)) & ((p == pm.H) | (p == pm.A))
    print("%-30s team accuracy where both say alive: %.3f (%d frames)"
          % (os.path.basename(path), np.mean(p[both] == gt[both]), both.sum()))
