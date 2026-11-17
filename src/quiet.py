import os

import numpy as np
import pandas as pd


def player_speed_series(raw_path, n_frames, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        if len(cached) == n_frames:
            return cached
    spd = np.full(n_frames, np.nan)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            total, count = 0.0, 0
            for g in parts[1].split(";"):
                if g[0] == "N":
                    continue
                f = g.split(",")
                if len(f) > 2:
                    v = float(f[2])
                    if v == v:
                        total += v
                        count += 1
            if count:
                spd[int(parts[0]) - 1] = total / count
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, spd)
    return spd


def quiet_runs(spd, smooth_frames, threshold, min_frames, bridge_frames, allowed):
    smooth = pd.Series(spd).rolling(smooth_frames, center=True,
                                    min_periods=max(1, smooth_frames // 5)).mean().values
    mask = np.isfinite(smooth) & (smooth < threshold) & allowed
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    merged = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        a, b = int(a), int(b)
        if merged and a - merged[-1][1] <= bridge_frames:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_frames]
