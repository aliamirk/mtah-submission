import os, sys, importlib, csv
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")

FPS = 25
N = games.source_frames("tactical", games.GAME)
CACHE = os.path.join(ROOT, "experiments", "cache", "spd.npy")

if os.path.isfile(CACHE):
    spd = np.load(CACHE)
else:
    print("reading player speeds, about a minute ...")
    spd = np.full(N, np.nan)
    with open(games.require(["atd_raw"])[0]) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            vals = []
            for g in parts[1].split(";"):
                if g[0] == "N":
                    continue
                f = g.split(",")
                if len(f) > 2:
                    v = float(f[2])
                    if v == v:
                        vals.append(v)
            if vals:
                spd[int(parts[0]) - 1] = sum(vals) / len(vals)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.save(CACHE, spd)

smooth = pd.Series(spd).rolling(50, center=True, min_periods=10).mean().values

raw = pm.label_codes(pm.read_label_csv(sys.argv[1]))
runs = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
filled = pm.fill_abstentions(raw, runs)
spells = pm.possession_spells(filled, runs)
graded = np.zeros(N, dtype=bool)
for a, b in runs:
    graded[a:b] = True
n_graded = int(graded.sum())


def runs_of(mask, min_len, bridge):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    merged = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = int(a), int(b)
        if merged and a - merged[-1][1] <= bridge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_len]


def find(thr, min_s):
    quiet = np.isfinite(smooth) & (smooth < thr) & graded
    return runs_of(quiet, int(min_s * FPS), 3 * FPS)


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


longest = sorted(spells, key=lambda s: -(s[1] - s[0]))[:10]
print("speed below (m/s), min length (s) | quiet runs | share of graded | already mostly D | inside one unsplit spell | of the 10 longest spells, containing one")
for thr, min_s in [(1.2, 45), (1.4, 30), (1.4, 45), (1.4, 60), (1.6, 60)]:
    qr = find(thr, min_s)
    total = sum(b - a for a, b in qr)
    mostly_d = unsplit = 0
    for a, b in qr:
        if np.mean(raw[a:b] == pm.D) >= 0.5:
            mostly_d += 1
        inside = [s for s in spells if s[0] < b and s[1] > a]
        if len(inside) == 1 and inside[0][0] <= a and inside[0][1] >= b:
            unsplit += 1
    contain = sum(1 for s, e, _ in longest if any(s <= a and b <= e for a, b in qr))
    print("%12.1f, %8d | %10d | %14.1f%% | %16d | %24d | %d of 10"
          % (thr, min_s, len(qr), 100.0 * total / n_graded, mostly_d, unsplit, contain))

qr = find(1.4, 45)
out_dir = os.path.join(ROOT, "experiments", "validation")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "quiet_runs_to_watch.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["video_start", "video_end", "secs", "verdict", "note"])
    for a, b in qr:
        w.writerow([clock(a - 10 * FPS), clock(b + 10 * FPS), round((b - a) / float(FPS), 1), "", ""])
print("\nwrote %s (%d runs to watch)" % (path, len(qr)))
