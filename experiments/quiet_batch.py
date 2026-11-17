import os, sys, importlib, csv
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
games = importlib.import_module("games")
quiet = importlib.import_module("src.quiet")
config_utils = importlib.import_module("src.config_utils")

FPS = 25
lo, hi = float(sys.argv[1]), float(sys.argv[2])
cfg = config_utils.load_config(os.path.join(ROOT, "config.yaml"))
N = games.source_frames("tactical", games.GAME)
spd = quiet.player_speed_series(games.require(["atd_raw"])[0], N,
                                os.path.join(ROOT, "cache", "player_speed.npy"))
allowed = np.ones(N, dtype=bool)
for a, b in cfg["nonplay_regions"]:
    allowed[a:b + 1] = False
allowed[190198:] = False
runs = quiet.quiet_runs(spd, int(cfg["quiet_smooth_frames"]), float(cfg["quiet_speed_threshold"]),
                        int(lo * FPS), int(cfg["quiet_bridge_seconds"] * FPS), allowed)
runs = [r for r in runs if (r[1] - r[0]) < hi * FPS]
print("quiet runs of %g to %g s: %d" % (lo, hi, len(runs)))


def clock(f):
    s = max(0, int(f / FPS))
    return "%d:%02d" % (s // 60, s % 60)


out = os.path.join(ROOT, "experiments", "validation", "quiet_runs_%d_%d.csv" % (lo, hi))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["video_start", "video_end", "secs", "verdict", "note", "ball_out", "restart"])
    for a, b in runs:
        w.writerow([clock(a - 10 * FPS), clock(b + 10 * FPS), round((b - a) / float(FPS), 1), "", "", "", ""])
print("wrote", out)
