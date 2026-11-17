import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
baseline = importlib.import_module("baseline")
pm = importlib.import_module("possession_metrics")

cache = os.path.join(ROOT, "experiments", "cache", "codes.npy")
os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.isfile(cache):
    codes = np.load(cache)
else:
    codes = baseline.build()      # 0 home, 1 away, -1 no ball; takes a minute or two
    np.save(cache, codes)

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
spells = pm.possession_spells(gt, runs)       # (start, stop, code), H=0, A=1

rows = []
for s, e, c in spells:
    seen = codes[s:e] != -1
    agree = float(np.mean(codes[s:e][seen] == c)) if seen.any() else float("nan")
    rows.append((e - s, agree, float(np.mean(~seen))))

def summ(name, group):
    ag = np.nanmean([g[1] for g in group])
    nb = np.mean([g[2] for g in group])
    print(f"{name:<22} n={len(group):>2}  raw nearest-player agrees on {100*ag:4.0f}% of tracked frames, ball missing {100*nb:4.0f}%")

summ("short (<= 90 frames)", [r for r in rows if r[0] <= 90])
summ("long (> 90 frames)", [r for r in rows if r[0] > 90])
