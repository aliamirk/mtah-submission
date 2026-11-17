import os, sys, importlib, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")

CARRY = 25
FPS = 25
HA = (pm.H, pm.A)
raw = pm.label_codes(pm.read_label_csv(sys.argv[1]))
runs = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
filled = pm.fill_abstentions(raw, runs)
n_all = sum(1 for f, a, b in pm.transitions(filled, runs) if a in HA and b in HA)


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


gaps, rows = [], []
for lo, hi in runs:
    isx = (raw[lo:hi] == pm.X).astype(np.int8)
    e = np.diff(np.concatenate(([0], isx, [0])))
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        rs, re = lo + int(s), lo + int(t)
        gs = rs - (CARRY - 1)
        if gs - 1 < lo or re >= hi:
            continue
        pre, fill, post = raw[gs - 1], raw[rs - 1], raw[re]
        if pre not in HA or fill not in HA or post not in HA:
            continue
        if pre == post:
            kind = "same team on both sides"
        elif fill == post:
            kind = "switch placed at the gap start"
        else:
            kind = "switch placed at the gap end"
        gaps.append(((re - gs) / float(FPS), kind))
        if pre != post:
            rows.append([clock(gs - 5 * FPS), clock(re + 5 * FPS), round((re - gs) / float(FPS), 1),
                         pm.LABELS[pre], pm.LABELS[post], "", ""])

print("X runs with a team label on both sides: %d" % len(gaps))
for kind in ("same team on both sides", "switch placed at the gap start", "switch placed at the gap end"):
    ls = np.array([L for L, k in gaps if k == kind])
    if len(ls):
        print("  %-32s %4d   gap length: median %.1f s, mean %.1f s" % (kind, len(ls), np.median(ls), ls.mean()))
sw = [L for L, k in gaps if k != "same team on both sides"]
print("H/A switches in the final labels: %d, of which inside a ball gap: %d (%.0f%%)"
      % (n_all, len(sw), 100.0 * len(sw) / max(1, n_all)))
if sw:
    print("if a handover is equally likely anywhere in its gap: expected error %.1f s at either end, %.1f s at the midpoint"
          % (np.mean(sw) / 2, np.mean(sw) / 4))
path = os.path.join(ROOT, "experiments", "validation", "gap_switches.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["video_start", "video_end", "secs", "team_before", "team_after", "handover_time", "note"])
    w.writerows(rows)
print("wrote %s (%d gaps containing a switch)" % (path, len(rows)))
