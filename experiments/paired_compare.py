import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
g = pm.possession_spells(gt, runs)
n_gt = len(g)


def prep(path):
    pred = pm.fill_abstentions(pm.label_codes(pm.read_label_csv(path)), runs)
    p = pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(g, p)
    flags = np.vstack([pm.matched_gt_flags(ov, n_gt, t) for t in pm.SPELL_IOUS])
    return flags, len(p)


def f1(flags, n_pred, picks):
    out = []
    for row in flags:
        tp = int(row[picks].sum())
        out.append(2.0 * tp / (2 * tp + max(0, n_pred - tp) + (n_gt - tp)) if tp else 0.0)
    return float(np.mean(out))


fa, na = prep(sys.argv[1])
fb, nb = prep(sys.argv[2])
rng = np.random.RandomState(0)
d = np.array([f1(fb, nb, p) - f1(fa, na, p)
              for p in (rng.randint(0, n_gt, n_gt) for _ in range(5000))])
full = np.arange(n_gt)
print("spellF1  A %.3f   B %.3f   B minus A %+.3f" % (f1(fa, na, full), f1(fb, nb, full),
                                                      f1(fb, nb, full) - f1(fa, na, full)))
print("paired bootstrap 90%% interval for B minus A: %+.3f to %+.3f" % tuple(np.percentile(d, [5, 95])))
print("share of resamples where B beats A: %.0f%%" % (100.0 * np.mean(d > 0)))
