import os, sys, importlib, csv
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
spells = pm.possession_spells(filled, runs)


def x_runs(s, e):
    seg = (raw[s:e] == pm.X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], seg, [0])))
    return [(s + int(a), s + int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


rows = []
for s, e, c in spells:
    xr = x_runs(s, e)
    xs = sum(b - a for a, b in xr)
    rows.append(((e - s) / float(FPS), xs / float(e - s),
                 sum(1 for a, b in xr if b - a >= 5 * FPS),
                 max([b - a for a, b in xr], default=0) / float(FPS)))
arr = np.array(rows)
print("spell length | spells | median share of frames that were X | median X-runs of 5 s or more | median longest X-run (s)")
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 1e9)):
    m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
    if m.any():
        print("%3d to %-4s s | %5d | %4.0f%% | %4.1f | %6.1f" % (
            lo, "inf" if hi > 1e8 else int(hi), m.sum(), 100 * np.median(arr[m, 1]),
            np.median(arr[m, 2]), np.median(arr[m, 3])))

gx = np.zeros(len(raw), dtype=bool)
for a, b in runs:
    gx[a:b] = raw[a:b] == pm.X
edges = np.diff(np.concatenate(([0], gx.astype(np.int8), [0])))
allx = [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
big = [(a, b) for a, b in allx if b - a >= 5 * FPS]
n_graded = sum(b - a for a, b in runs)
print("\nX runs on graded frames: %d, of which 5 s or longer: %d, holding %.1f%% of all graded frames"
      % (len(allx), len(big), 100.0 * sum(b - a for a, b in big) / n_graded))

order = np.argsort([-(e - s) for s, e, _ in spells])[:10]
watch = []
for i in order:
    s, e, c = spells[i]
    for a, b in x_runs(s, e):
        if b - a >= 5 * FPS:
            watch.append((a, b, s, e))
watch = sorted(sorted(watch, key=lambda w: -(w[1] - w[0]))[:40])
out_dir = os.path.join(ROOT, "experiments", "validation")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "long_spell_gaps.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["video_start", "video_end", "secs", "inside_spell", "verdict", "note"])
    for a, b, s, e in watch:
        w.writerow([clock(a - 5 * FPS), clock(b + 5 * FPS), round((b - a) / float(FPS), 1),
                    "%s-%s" % (clock(s), clock(e)), "", ""])
print("wrote %s (%d gaps inside the 10 longest spells)" % (path, len(watch)))
