import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")

SENS = os.path.join(ROOT, "experiments", "cache", "sens")
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
graded = np.zeros(len(gt), dtype=bool)
for a, b in runs:
    graded[a:b] = True


def load(name):
    raw = pm.label_codes(pm.read_label_csv(os.path.join(SENS, name + ".csv")))
    return raw, pm.fill_abstentions(raw, runs)


_, final_fill = load("final")
print("variant  | whole match: possessions  mean s  dead | frames differing from final | public: spellF1  macroF1  D recall  D precision")
for name in ["final", "tagcap1", "pen25", "pen38", "thr12", "thr16", "min8", "min20"]:
    raw, fill = load(name)
    sp = pm.possession_spells(fill, runs)
    mean_s = np.mean([e - s for s, e, _ in sp]) / 25.0
    dead = 100.0 * np.mean(fill[graded] == pm.D)
    diff = 100.0 * np.mean(fill[graded] != final_fill[graded])
    r = pm.evaluate(gt, raw)
    d = r["per_class"]["D"]
    print("%-8s | %20d %7.1f %5.1f%% | %27.1f%% | %15.3f %8.3f %9.2f %12.2f"
          % (name, len(sp), mean_s, dead, diff, r["spell_f1_mean"], r["macro_f1"], d["recall"], d["precision"]))
