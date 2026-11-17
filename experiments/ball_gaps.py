import os, sys, importlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
tagging = importlib.import_module("load_smart_tagging")
pm = importlib.import_module("possession_metrics")

FPS = 25
N = games.source_frames("tactical", games.GAME)
d = np.load(os.path.join(ROOT, "experiments", "cache", "feats.npz"))
bx, by, spd = d["bx"], d["by"], d["spd"]      # ball x, y already in metres

blocks = games.half_video_blocks(games.GAME)
kick2 = tagging.KICKOFF_VIDEO_FRAMES["2nd Half"]
graded = np.zeros(N, dtype=bool)
graded[blocks[0][1]:blocks[0][2] + 1] = True
graded[kick2:blocks[1][2] + 1] = True         # skips the 2,934 pre-kickoff frames

seen = np.isfinite(bx)
missing = ~seen & graded
edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
gaps = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

tempo = pd.Series(spd).rolling(2250, center=True, min_periods=500).median().values
tags = [(e["start_frame"], e["end_frame"]) for e in tagging.load()
        if (e["code"] or "").strip().lower() == "set pieces"]


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


rows = []
for a, b in gaps:
    a, b = int(a), int(b)
    jump = speed = np.nan
    if a > 0 and b < N and seen[a - 1] and seen[b]:
        jump = float(np.hypot(bx[b] - bx[a - 1], by[b] - by[a - 1]))
        speed = jump / ((b - a + 1) / FPS)
    mid = (a + b) // 2
    seg = spd[a:b]
    sp = float(np.nanmedian(seg)) if np.isfinite(seg).any() else np.nan
    ratio = sp / tempo[mid] if np.isfinite(tempo[mid]) and tempo[mid] > 0 else np.nan
    near = any(s - 3 * FPS <= b and a <= e + 3 * FPS for s, e in tags)
    rows.append(dict(start=a, end=b, secs=(b - a) / FPS, jump_m=jump, implied_ms=speed,
                     tempo_ratio=ratio, near_set_piece=near))
df = pd.DataFrame(rows)

print("graded frames with no ball: %.1f%%   gaps: %d" % (100 * missing.sum() / graded.sum(), len(df)))
print("\ngap length | gaps | share of missing frames | median tempo ratio | near a set piece")
bins = [0, 1, 2, 5, 10, 20, 1e9]
for lo, hi in zip(bins[:-1], bins[1:]):
    s = df[(df.secs >= lo) & (df.secs < hi)]
    if not len(s):
        continue
    share = (s.secs.sum() * FPS) / missing.sum()
    print("%5g-%-5g | %4d | %8.1f%%              | %5.2f              | %4.0f%%" % (
        lo, min(hi, 999), len(s), 100 * share, s.tempo_ratio.median(), 100 * s.near_set_piece.mean()))

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
cov = gt != pm.X
dead = gt == pm.D
for name, m in (("ball missing", cov & ~seen), ("ball tracked", cov & seen)):
    if m.any():
        print("public window, %s: %.0f%% of those frames are truly dead" % (name, 100 * dead[m].mean()))

picks = []
for lo, hi in [(1, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]:
    pool = df[(df.secs >= lo) & (df.secs < hi) & (df.start > 18684)]
    if len(pool):
        picks.append(pool.sample(min(4, len(pool)), random_state=0))
sample = pd.concat(picks).sort_values("start")
sample["video_start"] = [clock(a - 5 * FPS) for a in sample.start]
sample["video_end"] = [clock(b + 5 * FPS) for b in sample.end]
sample["verdict"] = ""
sample["note"] = ""
os.makedirs(os.path.join(ROOT, "experiments", "validation"), exist_ok=True)
out = os.path.join(ROOT, "experiments", "validation", "gaps_to_watch.csv")
sample.to_csv(out, index=False)
print("\nwrote %s (%d gaps to watch)" % (out, len(sample)))
