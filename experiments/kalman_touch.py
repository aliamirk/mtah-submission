import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
pm = importlib.import_module("possession_metrics")

CACHE = os.path.join(ROOT, "experiments", "cache")
ball = np.load(os.path.join(CACHE, "ball.npz"))
bx, by = ball["bx"], ball["by"]
codes = np.load(os.path.join(CACHE, "codes.npy"))
N = len(bx)
DT = 1.0 / 25.0
HALF = 12
MAX_COAST = 5


def kalman_nis(sigma_a, sigma_r):
    nis = np.full(N, np.nan)
    R = sigma_r ** 2
    x = y = P = None
    last = -10 ** 9
    for t in range(N):
        if not np.isfinite(bx[t]):
            continue
        gap = t - last
        if P is None or gap > MAX_COAST + 1:
            x = np.array([bx[t], 0.0])
            y = np.array([by[t], 0.0])
            P = np.array([[R, 0.0], [0.0, 25.0]])
            last = t
            continue
        h = gap * DT
        F = np.array([[1.0, h], [0.0, 1.0]])
        Q = sigma_a ** 2 * np.array([[h ** 4 / 4, h ** 3 / 2], [h ** 3 / 2, h ** 2]])
        xp, yp = F @ x, F @ y
        Pp = F @ P @ F.T + Q
        S = Pp[0, 0] + R
        iy = np.array([bx[t] - xp[0], by[t] - yp[0]])
        nis[t] = float(iy @ iy) / S
        K = Pp[:, 0] / S
        x = xp + K * iy[0]
        y = yp + K * iy[1]
        P = Pp - np.outer(K, Pp[0, :])
        last = t
    return nis


LAG = 5
vx = np.full(N, np.nan)
vy = np.full(N, np.nan)
vx[LAG:] = (bx[LAG:] - bx[:-LAG]) * 25.0 / LAG
vy[LAG:] = (by[LAG:] - by[:-LAG]) * 25.0 / LAG
kink = np.full(N, np.nan)
kink[:-LAG] = np.hypot(vx[LAG:] - vx[:-LAG], vy[LAG:] - vy[:-LAG])

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
handovers = [(f, b) for f, a, b in pm.transitions(gt, runs)
             if a in (pm.H, pm.A) and b in (pm.H, pm.A)]
control = []
for s, e, c in pm.possession_spells(gt, runs):
    if e - s > 100:
        control.extend((t, c) for t in range(s + 50, e - 50, 7))


def peaks(stat, items):
    out = []
    for t0, team in items:
        seg = stat[t0 - HALF:t0 + HALF + 1]
        if np.all(np.isnan(seg)):
            out.append((np.nan, -1, team))
        else:
            k = int(np.nanargmax(seg))
            out.append((float(seg[k]), t0 - HALF + k, team))
    return out


def auc(pos, neg):
    pos = np.array([p for p in pos if np.isfinite(p)])
    neg = np.array([p for p in neg if np.isfinite(p)])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


rng = np.random.RandomState(0)


def auc_interval(pos, neg):
    pos = np.array([p for p in pos if np.isfinite(p)])
    draws = [auc(pos[rng.randint(0, len(pos), len(pos))], neg) for _ in range(1000)]
    return np.percentile(draws, [5, 95])


def right_team(pk, thr):
    hit = [(loc, team) for v, loc, team in pk if np.isfinite(v) and v >= thr and codes[loc] != -1]
    return sum(1 for loc, team in hit if codes[loc] == team), len(hit)


print("handovers %d, control windows %d" % (len(handovers), len(control)))
print("statistic                     AUC (90%% interval)    right team at the spike: handovers | control")
ho, ct = peaks(kink, handovers), peaks(kink, control)
lo, hi = auc_interval([p[0] for p in ho], [p[0] for p in ct])
rh, nh = right_team(ho, 3.0)
rc, nc = right_team(ct, 3.0)
print("velocity change >= 3 m/s       %.2f (%.2f to %.2f)      %d of %d | %d of %d"
      % (auc([p[0] for p in ho], [p[0] for p in ct]), lo, hi, rh, nh, rc, nc))
for sa, sr in [(3, 0.4), (6, 0.4), (10, 0.4), (6, 0.8)]:
    nis = kalman_nis(sa, sr)
    ho, ct = peaks(nis, handovers), peaks(nis, control)
    lo, hi = auc_interval([p[0] for p in ho], [p[0] for p in ct])
    rh, nh = right_team(ho, 16.0)
    rc, nc = right_team(ct, 16.0)
    print("Kalman accel %2d, noise %.1f m   %.2f (%.2f to %.2f)      %d of %d | %d of %d"
          % (sa, sr, auc([p[0] for p in ho], [p[0] for p in ct]), lo, hi, rh, nh, rc, nc))
