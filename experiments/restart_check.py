import os, sys, importlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
tagging = importlib.import_module("starter.load_smart_tagging")
pm = importlib.import_module("possession_metrics")

FPS = 25
FOULED = not (len(sys.argv) > 3 and sys.argv[3] == "flip")
rows = pd.read_csv(sys.argv[1], dtype=str).fillna("")
pred = pm.label_codes(pm.read_label_csv(sys.argv[2]))


def to_s(t):
    p = str(t).strip().split(":")
    try:
        if len(p) == 3 and p[2] != "00":
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return int(p[0]) * 60 + int(p[1])
    except (ValueError, IndexError):
        return None


def team_in(text):
    t = text.lower()
    if "athletic" in t or "atletic" in t:
        return "A"
    if "barca" in t or "barcelona" in t:
        return "H"
    return None


def kind(text):
    t = text.lower()
    if "corner" in t:
        return "corner kick"
    if "goal kick" in t:
        return "goal kick"
    if "throw" in t:
        return "throw-in"
    if "foul" in t or "offside" in t:
        return "free kick"
    return None


def restart_team(text):
    t = text.lower()
    team = team_in(t)
    if team is None:
        return None
    other = "H" if team == "A" else "A"
    if "offside" in t:
        return other
    if "foul" in t:
        return team if FOULED else other
    return team


def tag_team(e):
    t = (e["tags"].get("Team") or "").lower()
    if "barcelona" in t:
        return "H"
    if "atletic" in t or "athletic" in t:
        return "A"
    return None


KINDS = ("corner kick", "goal kick", "free kick", "throw-in")
tags = [e for e in tagging.load()
        if (e["code"] or "").strip().lower() == "set pieces"
        and (e["tags"].get("Type") or "").strip().lower() in KINDS]


def majority(arr):
    arr = arr[(arr == pm.H) | (arr == pm.A)]
    if len(arr) == 0:
        return None
    return "H" if np.sum(arr == pm.H) >= np.sum(arr == pm.A) else "A"


n = 0
by_kind = {}
tag_any = tag_kind = tag_named = tag_right = 0
dec_n = dec_ok = same_n = same_yes = 0
for _, r in rows.iterrows():
    a, b = to_s(r["video_start"]), to_s(r["video_end"])
    ty, rt = kind(r["note"]), restart_team(r["note"])
    if a is None or b is None or ty is None or rt is None:
        continue
    fs, fe = (a + 10) * FPS, (b - 10) * FPS
    n += 1
    by_kind[ty] = by_kind.get(ty, 0) + 1
    near = [e for e in tags if e["start_frame"] <= fe + 5 * FPS and e["end_frame"] >= fs - 5 * FPS]
    tag_any += bool(near)
    same = [e for e in near if (e["tags"].get("Type") or "").strip().lower() == ty]
    if same:
        tag_kind += 1
        named = [tag_team(e) for e in same if tag_team(e)]
        if named:
            tag_named += 1
            tag_right += int(named[0] == rt)
    after = majority(pred[fe:fe + 3 * FPS])
    before = majority(pred[max(0, fs - 3 * FPS):fs])
    if after:
        dec_n += 1
        dec_ok += int(after == rt)
    if after and before:
        same_n += 1
        same_yes += int(before == rt)

print("stoppages with a usable note: %d (%s)" % (n, ", ".join("%s %d" % kv for kv in sorted(by_kind.items()))))
print("fouls read as: " + ("the named team was fouled and restarts" if FOULED else "the named team committed the foul"))
print("set-piece tag within 5 s of the quiet run: %d of %d" % (tag_any, n))
print("  tag of the same kind: %d; with a named team: %d; its team is the restarting team: %d of %d"
      % (tag_kind, tag_named, tag_right, tag_named))
print("pipeline gives the restarting team in the 3 s after the run: %d of %d" % (dec_ok, dec_n))
print("the team in possession before the run is the restarting team: %d of %d" % (same_yes, same_n))
