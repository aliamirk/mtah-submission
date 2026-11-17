import os, sys, importlib, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
games = importlib.import_module("games")
quiet = importlib.import_module("src.quiet")

FPS = 25
N = games.source_frames("tactical", games.GAME)
spd = quiet.player_speed_series(games.require(["atd_raw"])[0], N,
                                os.path.join(ROOT, "cache", "player_speed.npy"))
allowed = np.zeros(N, dtype=bool)
for _, a, b in games.half_video_blocks(games.GAME):
    allowed[a:b + 1] = True
kick2 = importlib.import_module("starter.load_smart_tagging").KICKOFF_VIDEO_FRAMES["2nd Half"]
allowed[:11185] = False
allowed[90652:kick2] = False

for lo, hi in ((15, 20), (20, 30), (30, 45), (45, 1e9)):
    runs = quiet.quiet_runs(spd, 50, 1.4, int(lo * FPS), 3 * FPS, allowed)
    runs = [r for r in runs if (r[1] - r[0]) < hi * FPS]
    print("runs of %2d s to %s s: %d" % (lo, "inf" if hi > 1e8 else int(hi), len(runs)))

batch = [r for r in quiet.quiet_runs(spd, 50, 1.4, 20 * FPS, 3 * FPS, allowed) if (r[1] - r[0]) < 45 * FPS]


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


path = os.path.join(ROOT, "experiments", "validation", "quiet_runs_20_45.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["video_start", "video_end", "secs", "verdict", "note"])
    for a, b in batch:
        w.writerow([clock(a - 10 * FPS), clock(b + 10 * FPS), round((b - a) / float(FPS), 1), "", ""])
print("wrote %s (%d runs to watch)" % (path, len(batch)))
