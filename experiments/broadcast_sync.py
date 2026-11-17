import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
games = importlib.import_module("games")

L, W = games.GAMES[games.GAME]["pitch"]
CACHE_DIR = os.path.join(ROOT, "experiments", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE = os.path.join(CACHE_DIR, "ball_sync.npz")


def read_ball(path):
    ids, xs, ys = [], [], []
    last = 0
    with open(path) as fh:
        for line in fh:
            last = int(line[:line.index(":")])
            ball = line[line.rindex(":") + 1:]
            if ball[0] != "N":
                p = ball.split(",")
                ids.append(last - 1)
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    x = np.full(last, np.nan)
    y = np.full(last, np.nan)
    x[ids] = np.array(xs) * L
    y[ids] = np.array(ys) * W
    return x, y


if os.path.isfile(CACHE):
    d = np.load(CACHE)
    tx, ty, bx, by = d["tx"], d["ty"], d["bx"], d["by"]
else:
    print("reading tactical ball ...")
    tx, ty = read_ball(games.require(["atd_raw"], source="tactical")[0])
    print("reading broadcast ball ...")
    bx, by = read_ball(games.require(["atd_raw"], source="broadcast")[0])
    np.savez(CACHE, tx=tx, ty=ty, bx=bx, by=by)
Nt, Na = len(tx), len(bx)
print("tactical frames %d, broadcast frames %d" % (Nt, Na))


def correlate(a, b, n):
    # c[o] = sum_t a[t] * b[t + o]
    return np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)


def offset_error(ax, ay, min_n):
    n = 1 << int(np.ceil(np.log2(Na + Nt)))
    ma = np.isfinite(ax).astype(float)
    mb = np.isfinite(tx).astype(float)
    axz, ayz = np.where(ma > 0, ax, 0.0), np.where(ma > 0, ay, 0.0)
    txz, tyz = np.where(mb > 0, tx, 0.0), np.where(mb > 0, ty, 0.0)
    cnt = correlate(ma, mb, n)[:Nt]
    p1 = correlate(axz ** 2 + ayz ** 2, mb, n)[:Nt]
    p2 = correlate(ma, txz ** 2 + tyz ** 2, n)[:Nt]
    p3 = (correlate(axz, txz, n) + correlate(ayz, tyz, n))[:Nt]
    s = p1 + p2 - 2 * p3
    rms = np.full(Nt, np.inf)
    ok = cnt >= min_n
    rms[ok] = np.sqrt(np.maximum(s[ok], 0) / cnt[ok])
    return rms, cnt


def peaks(rms, k=5, gap=1500):
    r = rms.copy()
    out = []
    for _ in range(k):
        o = int(np.argmin(r))
        if not np.isfinite(r[o]):
            break
        out.append((o, float(r[o])))
        r[max(0, o - gap):o + gap] = np.inf
    return out


cands = []
for name, flip in (("same orientation", False), ("rotated 180 degrees", True)):
    ax = (L - bx) if flip else bx
    ay = (W - by) if flip else by
    rms, cnt = offset_error(ax, ay, 3000)
    floor = float(np.median(rms[np.isfinite(rms)]))
    print("\n%s: typical error at a random offset %.1f m" % (name, floor))
    for o, e in peaks(rms):
        print("   offset %7d   rms %6.2f m   frames compared %6d" % (o, e, int(cnt[o])))
        if e < 0.75 * floor and cnt[o] >= 20000:
            cands.append((o, flip, e))
print("\ncandidate offsets kept: %d" % len(cands))
cands = [(o + d, f, e) for (o, f, e) in cands for d in range(-4, 5)]

CH = 250
accepted = np.full((Na + CH - 1) // CH, -1, dtype=int)
for c in range(accepted.size):
    s, e = c * CH, min(Na, c * CH + CH)
    idx = np.arange(s, e)
    best, best_good = -1, 0
    for ci, (o, flip, _) in enumerate(cands):
        t = idx + o
        ok = t < Nt
        tt = np.where(ok, t, 0)
        px = (L - bx[s:e]) if flip else bx[s:e]
        py = (W - by[s:e]) if flip else by[s:e]
        both = ok & np.isfinite(px) & np.isfinite(tx[tt])
        nb = int(both.sum())
        if nb < 40:
            continue
        dist = np.hypot(px[both] - tx[tt][both], py[both] - ty[tt][both])
        good = int((dist < 3.0).sum())
        if good >= 40 and good >= 0.6 * nb and good > best_good:
            best, best_good = ci, good
    accepted[c] = best

print("\nbroadcast timeline cut into %d chunks of 10 s" % accepted.size)
for ci, (o, flip, _) in enumerate(cands):
    n = int(np.sum(accepted == ci))
    if n:
        print("   offset %7d (%s): %d chunks" % (o, "rotated" if flip else "same", n))
print("   no verified offset: %d chunks" % int(np.sum(accepted < 0)))

graded = np.zeros(Nt, dtype=bool)
for _, a, b in games.half_video_blocks(games.GAME):
    graded[a:b + 1] = True
fill_x = np.full(Nt, np.nan)
fill_y = np.full(Nt, np.nan)
errs = []
for c, ci in enumerate(accepted):
    if ci < 0:
        continue
    o, flip, _ = cands[ci]
    s, e = c * CH, min(Na, c * CH + CH)
    t = np.arange(s, e) + o
    ok = (t < Nt)
    px = (L - bx[s:e]) if flip else bx[s:e]
    py = (W - by[s:e]) if flip else by[s:e]
    ok &= np.isfinite(px)
    tt, qx, qy = t[ok], px[ok], py[ok]
    both = np.isfinite(tx[tt])
    errs.append(np.hypot(qx[both] - tx[tt][both], qy[both] - ty[tt][both]))
    need = ~np.isfinite(tx[tt]) & graded[tt]
    fill_x[tt[need]] = qx[need]
    fill_y[tt[need]] = qy[need]

missing = graded & ~np.isfinite(tx)
got = np.isfinite(fill_x)
print("\ngraded frames %d, tactical ball missing on %d (%.1f%%)"
      % (graded.sum(), missing.sum(), 100.0 * missing.sum() / graded.sum()))
print("verified broadcast ball fills %d of those (%.1f%% of the missing frames)"
      % (got.sum(), 100.0 * got.sum() / max(1, missing.sum())))
if errs:
    allerr = np.concatenate(errs)
    print("where both cameras see the ball in verified chunks: median gap %.2f m, 90th percentile %.2f m"
          % (np.median(allerr), np.percentile(allerr, 90)))
np.savez(os.path.join(CACHE_DIR, "ball_fill.npz"), fx=fill_x, fy=fill_y)
print("saved experiments/cache/ball_fill.npz")
