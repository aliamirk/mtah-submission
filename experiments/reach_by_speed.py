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
spells = pm.possession_spells(gt, pm.valid_runs(gt))
LAG, VMAX = 5, 6.0
first = min(s for s, _, _ in spells) - LAG
last = max(e for _, e, _ in spells)

rows = {}
with open(paths["atd_raw"]) as fh:
    for line in fh:
        i = int(line[:line.index(":")]) - 1
        if i < first:
            continue
        if i >= last:
            break
        rows[i] = line.rstrip("\n").split(":")


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
EDGES = [0, 1, 3, 6, 10, 1e9]
NAMES = ["0 to 1", "1 to 3", "3 to 6", "6 to 10", "10 and up"]
METHODS = ("nearest", "reach 0.5 s", "reach 1.0 s")

n_sp = len(spells)
ok = np.zeros((n_sp, len(NAMES), 3))
cnt = np.zeros((n_sp, len(NAMES)))
for si, (s, e, c) in enumerate(spells):
    for t in range(s, e):
        if t not in parsed or (t - LAG) not in parsed:
            continue
        ball, px, py, tm = parsed[t]
        prev = parsed[t - LAG][0]
        if ball is None or prev is None or len(px) == 0 or not ((tm == 0).any() and (tm == 1).any()):
            continue
        vx, vy = (ball[0] - prev[0]) * 25.0 / LAG, (ball[1] - prev[1]) * 25.0 / LAG
        speed = float(np.hypot(vx, vy))
        b = int(np.searchsorted(EDGES, speed, side="right")) - 1
        calls = [int(tm[np.argmin(np.hypot(px - ball[0], py - ball[1]))])]
        for h in (0.5, 1.0):
            reach = np.hypot(px - (ball[0] + vx * h), py - (ball[1] + vy * h)) / VMAX
            calls.append(int(tm[np.argmin(reach)]))
        cnt[si, b] += 1
        for k, call in enumerate(calls):
            ok[si, b, k] += int(call == c)

rng = np.random.RandomState(0)
draws = [rng.randint(0, n_sp, n_sp) for _ in range(2000)]


def rate(idx, b, k):
    n = cnt[idx, b].sum()
    return ok[idx, b, k].sum() / n if n else np.nan


allidx = np.arange(n_sp)
print("ball speed m/s | frames | nearest  reach 0.5 s  reach 1.0 s | reach 0.5 s minus nearest, points (90% interval)")
for b, name in enumerate(NAMES):
    d = [100 * (rate(p, b, 1) - rate(p, b, 0)) for p in draws]
    lo, hi = np.nanpercentile(d, [5, 95])
    print("%14s | %6d | %6.0f%% %10.0f%% %10.0f%% | %+5.1f   (%+.1f to %+.1f)"
          % (name, cnt[:, b].sum(), *(100 * rate(allidx, b, k) for k in range(3)),
             100 * (rate(allidx, b, 1) - rate(allidx, b, 0)), lo, hi))
tot = cnt.sum(axis=1)
print("%14s | %6d | %6.0f%% %10.0f%% %10.0f%%" % ("all", tot.sum(),
      *(100 * ok[:, :, k].sum() / tot.sum() for k in range(3))))

print("\nreach 1.0 s minus nearest, points (90% interval)")
for b, name in enumerate(NAMES):
    d = [100 * (rate(p, b, 2) - rate(p, b, 0)) for p in draws]
    lo, hi = np.nanpercentile(d, [5, 95])
    print("%14s | %+5.1f   (%+.1f to %+.1f)" % (name, 100 * (rate(allidx, b, 2) - rate(allidx, b, 0)), lo, hi))
