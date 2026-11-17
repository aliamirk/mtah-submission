import numpy as np

KINDS = ("corner kick", "goal kick", "free kick")


def restart_prior_from_runs(tags, runs, n_frames, cfg):
    fps = int(cfg.get("fps", 25))
    slack = int(float(cfg.get("restart_tag_slack_seconds", 5)) * fps)
    length = int(cfg.get("restart_prior_frames", 50))
    named = []
    for tag in tags:
        kind = tag["tags"].get("Type", "").strip().lower()
        team = tag["tags"].get("Team", "").strip().lower()
        if kind not in KINDS:
            continue
        if team == "barcelona":
            named.append((tag["start_frame"], tag["end_frame"], 1))
        elif team in ("atletic club", "athletic club"):
            named.append((tag["start_frame"], tag["end_frame"], -1))
    prior = np.zeros(n_frames, dtype=np.int8)
    for a, b in runs:
        cands = [(abs(start - b), sign) for start, end, sign in named
                 if start <= b + slack and end >= a - slack]
        if cands:
            prior[b:min(n_frames, b + length)] = min(cands)[1]
    return prior
