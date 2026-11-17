import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
baseline = importlib.import_module("baseline")
pm = importlib.import_module("possession_metrics")

L, W = games.GAMES[games.GAME]["pitch"]
paths = games.asset_paths()
team_of_slot = baseline.team_of_each_track(paths["atd_meta"])
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
trans = [(f, a, b) for f, a, b in pm.transitions(gt, pm.valid_runs(gt))
         if a in (pm.H, pm.A) and b in (pm.H, pm.A)]
LAG, VMAX = 5, 6.0
need = set()
for f, a, b in trans:
    need.update(range(f - 25 - LAG, f + 26))

rows = {}
with open(paths["atd_raw"]) as fh:
    for line in fh:
        i = int(line[:line.index(":")]) - 1
        if i in need:
            rows[i] = line.rstrip("\n").split(":")
            if len(rows) == len(need):
                break


def parse(parts):
    ball = None
    if parts[2][0] != "N":
        p = parts[2].split(",")
        ball = (float(p[0]) * L, float(p[1]) * W)
    px, py, tm = [], [], []
    for slot, g in enumerate(parts[1].split(";")):
        if g[0] == "N":
            continue
        q = g.split(",")
        px.append(float(q[0]) * L)
        py.append(float(q[1]) * W)
        tm.append(int(team_of_slot[slot]))
    return ball, np.array(px), np.array(py), np.array(tm)


parsed = {i: parse(p) for i, p in rows.items()}
METHODS = ("nearest", "reach 0.5 s", "reach 1.0 s")
ZONES = {"before (25 to 1 ahead)": (-25, -1), "around (+/-10)": (-10, 10), "after (1 to 25)": (1, 25)}


def calls(t):
    if t not in parsed or (t - LAG) not in parsed:
        return None
    ball, px, py, tm = parsed[t]
    prev = parsed[t - LAG][0]
    if ball is None or prev is None or len(px) == 0 or not ((tm == 0).any() and (tm == 1).any()):
        return None
    vx, vy = (ball[0] - prev[0]) * 25.0 / LAG, (ball[1] - prev[1]) * 25.0 / LAG
    out = [int(tm[np.argmin(np.hypot(px - ball[0], py - ball[1]))])]
    for h in (0.5, 1.0):
        reach = np.hypot(px - (ball[0] + vx * h), py - (ball[1] + vy * h)) / VMAX
        out.append(int(tm[np.argmin(reach)]))
    return out


stat = [{z: {m: [0, 0] for m in METHODS} for z in ZONES} for _ in trans]
for hi, (f, a, b) in enumerate(trans):
    for t in range(f - 25, f + 26):
        if gt[t] not in (pm.H, pm.A):
            continue
        c = calls(t)
        if c is None:
            continue
        for z, (lo, hi_) in ZONES.items():
            if lo <= t - f <= hi_:
                for m, call in zip(METHODS, c):
                    stat[hi][z][m][0] += int(call == gt[t])
                    stat[hi][z][m][1] += 1


def pooled(idx, z, m):
    c = sum(stat[i][z][m][0] for i in idx)
    n = sum(stat[i][z][m][1] for i in idx)
    return c / float(max(1, n))


allidx = list(range(len(trans)))
print("handovers: %d" % len(trans))
print("zone                     frames | nearest   reach 0.5 s   reach 1.0 s   (agreement with the label)")
for z in ZONES:
    n = sum(stat[i][z]["nearest"][1] for i in allidx)
    print("%-24s %6d | %6.0f%%   %9.0f%%   %9.0f%%" % (z, n, *(100 * pooled(allidx, z, m) for m in METHODS)))

rng = np.random.RandomState(0)
print("\naround the flip, reach minus nearest (90% interval over handovers):")
for m in METHODS[1:]:
    d = [pooled(i, "around (+/-10)", m) - pooled(i, "around (+/-10)", "nearest")
         for i in (rng.randint(0, len(trans), len(trans)) for _ in range(2000))]
    lo, hi_ = np.percentile(d, [5, 95])
    print("  %-12s %+5.1f points   (%+.1f to %+.1f)"
          % (m, 100 * (pooled(allidx, "around (+/-10)", m) - pooled(allidx, "around (+/-10)", "nearest")), 100 * lo, 100 * hi_))
