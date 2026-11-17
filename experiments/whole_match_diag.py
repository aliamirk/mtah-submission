import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")

FPS = 25
raw = pm.label_codes(pm.read_label_csv(sys.argv[1]))
runs = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
filled = pm.fill_abstentions(raw, runs)
graded = np.zeros(len(raw), dtype=bool)
for a, b in runs:
    graded[a:b] = True
n = int(graded.sum())


def clock(f):
    s = int(f / FPS)
    return "%d:%02d" % (s // 60, s % 60)


print("graded frames: %d" % n)
for name, arr in (("as written", raw), ("after X is filled", filled)):
    print("  %-18s" % name + "  ".join("%s %4.1f%%" % (pm.LABELS[k], 100.0 * np.sum(arr[graded] == k) / n) for k in range(4)))

spells = pm.possession_spells(filled, runs)
dur = np.array([(e - s) / float(FPS) for s, e, _ in spells])
print("\npossessions: %d   mean %.1f s   median %.1f s" % (len(spells), dur.mean(), np.median(dur)))
for t in (30, 60, 120):
    m = dur > t
    print("  longer than %3d s: %3d spells, %4.1f%% of all possession time" % (t, m.sum(), 100.0 * dur[m].sum() / dur.sum()))

print("\nlongest predicted possessions (video clock)")
for i in np.argsort(-dur)[:15]:
    s, e, c = spells[i]
    print("  %-7s to %-7s  %6.1f s  %s" % (clock(s), clock(e), dur[i], pm.LABELS[c]))
