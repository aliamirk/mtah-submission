import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

cache = os.path.join(ROOT, "experiments", "cache")
codes = np.load(os.path.join(cache, "codes.npy"))
near = np.load(os.path.join(cache, "inputs.npz"))["near_team"]
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))

plain = np.where(codes == 0, 1, np.where(codes == 1, -1, 0))
truth = np.where(gt == pm.H, 1, -1)
ok = ((gt == pm.H) | (gt == pm.A)) & (plain != 0) & (near != 0)
changed = ok & (plain != near)
print("alive frames with both calls: %d" % ok.sum())
print("per-frame team accuracy: nearest %.3f   reach-gated %.3f" % (
    np.mean(plain[ok] == truth[ok]), np.mean(near[ok] == truth[ok])))
print("frames whose call changed: %d   reach right on %d   nearest right on %d" % (
    changed.sum(), np.sum(changed & (near == truth)), np.sum(changed & (plain == truth))))
