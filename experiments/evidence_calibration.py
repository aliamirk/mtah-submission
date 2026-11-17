import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
baseline = importlib.import_module("baseline")
pm = importlib.import_module("possession_metrics")

N = games.source_frames("tactical", games.GAME)
L, W = games.GAMES[games.GAME]["pitch"]
paths = games.asset_paths()
team_of_slot = baseline.team_of_each_track(paths["atd_meta"])      # 0 home, 1 away
cache = os.path.join(ROOT, "experiments", "cache", "dist.npz")

if os.path.isfile(cache):
    d = np.load(cache)
    d_home, d_away = d["d_home"], d["d_away"]
else:
    print("reading tracking, about a minute ...")
    d_home = np.full(N, np.nan)
    d_away = np.full(N, np.nan)
    with open(paths["atd_raw"]) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            f = parts[2].split(",")
            bx, by = float(f[0]) * L, float(f[1]) * W
            best = [np.inf, np.inf]
            for slot, g in enumerate(parts[1].split(";")):
                if g[0] == "N":
                    continue
                p = g.split(",")
                dist = np.hypot(float(p[0]) * L - bx, float(p[1]) * W - by)
                t = team_of_slot[slot]
                if dist < best[t]:
                    best[t] = dist
            i = int(parts[0]) - 1
            d_home[i] = best[0] if np.isfinite(best[0]) else np.nan
            d_away[i] = best[1] if np.isfinite(best[1]) else np.nan
    np.savez(cache, d_home=d_home, d_away=d_away)

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
alive = (gt == pm.H) | (gt == pm.A)
ok = alive & np.isfinite(d_home) & np.isfinite(d_away)
margin = d_away - d_home                       # > 0 means the nearest home player is closer
right = (margin > 0) == (gt == pm.H)
dmin = np.minimum(d_home, d_away)
print("alive frames %d, usable %d" % (alive.sum(), ok.sum()))


def table(title, values, edges):
    print("\n" + title)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (values >= lo) & (values < hi)
        if m.sum():
            print("  %4.1f to %-4s m : %5d frames   nearest-player team call right %3.0f%%"
                  % (lo, "inf" if hi > 1e8 else "%.1f" % hi, m.sum(), 100 * right[m].mean()))


table("by margin (gap between the nearest home and nearest away player)",
      np.abs(margin), [0, 0.5, 1, 2, 3, 5, 8, 12, 1e9])
table("by distance from the ball to the nearest player",
      dmin, [0, 1, 2, 3, 5, 8, 1e9])
