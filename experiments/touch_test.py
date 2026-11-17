import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")

N = games.source_frames("tactical", games.GAME)
LENGTH, WIDTH = games.GAMES[games.GAME]["pitch"]
CACHE = os.path.join(ROOT, "experiments", "cache")
codes = np.load(os.path.join(CACHE, "codes.npy"))      # nearest player's team per frame, -1 = no ball

ball_file = os.path.join(CACHE, "ball.npz")
if not os.path.isfile(ball_file):
    print("reading ball positions, about a minute...")
    bx = np.full(N, np.nan)
    by = np.full(N, np.nan)
    with open(games.require(["atd_raw"])[0]) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":
                continue
            f = parts[2].split(",")
            i = int(parts[0]) - 1
            bx[i] = float(f[0]) * LENGTH
            by[i] = float(f[1]) * WIDTH
    np.savez(ball_file, bx=bx, by=by)
d = np.load(ball_file)
bx, by = d["bx"], d["by"]

LAG = 5                                  # frames, velocity over 0.2 s
vx = np.full(N, np.nan)
vy = np.full(N, np.nan)
vx[LAG:] = (bx[LAG:] - bx[:-LAG]) * 25.0 / LAG
vy[LAG:] = (by[LAG:] - by[:-LAG]) * 25.0 / LAG
kink = np.full(N, np.nan)                # change of ball velocity across frame t, m/s
kink[:-LAG] = np.hypot(vx[LAG:] - vx[:-LAG], vy[LAG:] - vy[:-LAG])

HALF = 12                                # +/- 0.5 s


def window_peak(t0):
    seg = kink[t0 - HALF:t0 + HALF + 1]
    if np.all(np.isnan(seg)):
        return None, None
    k = int(np.nanargmax(seg))
    return float(seg[k]), t0 - HALF + k


gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)

handovers = [(f, b) for f, a, b in pm.transitions(gt, runs)
             if a in (pm.H, pm.A) and b in (pm.H, pm.A)]
control = []
for s, e, c in pm.possession_spells(gt, runs):
    if e - s > 100:
        control.extend((t, c) for t in range(s + 50, e - 50, 7))


def peaks(items):
    out = []
    for t, team in items:
        k, kf = window_peak(t)
        out.append((k, kf, team))
    return out


ho, ct = peaks(handovers), peaks(control)
print("handovers (no dead ball between): %d   control windows inside long possessions: %d"
      % (len(ho), len(ct)))
print("handovers with no ball at all in the +/-0.5 s window: %d" % sum(1 for k, _, _ in ho if k is None))
print()
print("threshold | windows with a kink: handovers vs control | nearest player at the kink is on the right team: handovers vs control")


def stats(group, thr):
    have = [(k, kf, team) for k, kf, team in group if k is not None]
    hit = [(kf, team) for k, kf, team in have if k >= thr]
    known = [(kf, team) for kf, team in hit if codes[kf] != -1]
    right = [1 for kf, team in known if codes[kf] == team]
    return len(hit), len(have), len(right), len(known)


for thr in (3, 5, 8, 12):
    h_hit, h_n, h_r, h_k = stats(ho, thr)
    c_hit, c_n, c_r, c_k = stats(ct, thr)
    print("%5d m/s | %2d of %2d (%3.0f%%) vs %3d of %3d (%3.0f%%) | %2d of %2d vs %3d of %3d"
          % (thr, h_hit, h_n, 100.0 * h_hit / max(1, h_n), c_hit, c_n,
             100.0 * c_hit / max(1, c_n), h_r, h_k, c_r, c_k))
