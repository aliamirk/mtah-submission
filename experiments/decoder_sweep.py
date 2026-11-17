import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
tagging = importlib.import_module("load_smart_tagging")
pm = importlib.import_module("possession_metrics")

N = games.source_frames("tactical", games.GAME)
codes = np.load(os.path.join(ROOT, "experiments", "cache", "codes.npy"))  # 0 home, 1 away, -1 no ball
evidence = np.where(codes == 0, 1, np.where(codes == 1, -1, 0)).astype(np.int8)

blocks = games.half_video_blocks(games.GAME)            # graded halves
live = [(blocks[0][1], blocks[0][2] + 1),
        (tagging.KICKOFF_VIDEO_FRAMES["2nd Half"], blocks[1][2] + 1)]
pre_kickoff = (blocks[1][1], tagging.KICKOFF_VIDEO_FRAMES["2nd Half"])  # graded, all dead
DEAD_GAP = 101   # same rule as baseline_smooth50: this many frames with no ball in a row = D


def decode(e, penalty):
    """Two-state Viterbi. A frame whose evidence disagrees with the state costs 1,
    a switch costs `penalty`, and frames with no ball cost nothing either way."""
    n = len(e)
    c_home = (e == -1).astype(np.float64)
    c_away = (e == 1).astype(np.float64)
    back = np.zeros((n, 2), dtype=np.int8)
    ch, ca = c_home[0], c_away[0]
    for t in range(1, n):
        if ch <= ca + penalty:
            nh, bh = ch, 0
        else:
            nh, bh = ca + penalty, 1
        if ca <= ch + penalty:
            na, ba = ca, 1
        else:
            na, ba = ch + penalty, 0
        back[t, 0], back[t, 1] = bh, ba
        ch, ca = nh + c_home[t], na + c_away[t]
    state = 0 if ch <= ca else 1
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        state = back[t, state]
    return out


def long_gaps(missing, min_len):
    mask = np.zeros(len(missing), dtype=bool)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a >= min_len:
            mask[a:b] = True
    return mask


def build(penalty):
    labels = np.full(N, "X", dtype="<U1")
    labels[pre_kickoff[0]:pre_kickoff[1]] = "D"
    for a, b in live:
        state = decode(evidence[a:b], penalty)
        seg = np.where(state == 0, "H", "A")
        dead = long_gaps(codes[a:b] == -1, DEAD_GAP)
        seg[dead] = "D"
        labels[a:b] = seg
    return labels


def block_counts(spells):
    counts = []
    for a, b in live:
        starts = np.array([s for s, e, c in spells if a <= s < b])
        for k in range((b - a) // 7500):
            lo = a + k * 7500
            counts.append(int(np.sum((starts >= lo) & (starts < lo + 7500))))
    return counts


gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs_full = [(blocks[0][1], blocks[0][2] + 1), (blocks[1][1], blocks[1][2] + 1)]

print("penalty | public window: macroF1 spellF1 poss transF1@25 | whole match: poss  mean_s  per-5-min min/median/max")
for penalty in (5, 10, 15, 25, 40, 60, 90):
    labels = build(penalty)
    pred = pm.label_codes(labels)
    r = pm.evaluate(gt, pred)
    spells = pm.possession_spells(pred, runs_full)
    mean_s = np.mean([e - s for s, e, c in spells]) / 25.0
    bc = block_counts(spells)
    print(f"{penalty:>7} | {r['macro_f1']:.3f}   {r['spell_f1_mean']:.3f}   {r['n_pred_spells']:>3}   "
          f"{r['transitions'][25]['f1']:.3f}  | {len(spells):>5}  {mean_s:5.1f}   "
          f"{min(bc)}/{int(np.median(bc))}/{max(bc)}")
