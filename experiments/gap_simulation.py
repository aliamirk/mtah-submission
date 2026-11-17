import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")
config_utils = importlib.import_module("src.config_utils")
assigners = importlib.import_module("src.decoder_assigner")

W0, W1, CARRY = 11185, 18685, 25
cfg = config_utils.load_config(os.path.join(ROOT, "config.yaml"))
d = np.load(os.path.join(ROOT, "experiments", "cache", "inputs.npz"))
near, dead, gap0 = d["near_team"], d["dead_mask"], d["ball_gap"]
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
gt_trans = pm.transitions(gt, pm.valid_runs(gt))
assigner = assigners.DecoderAssigner()

graded = np.zeros(len(near), dtype=bool)
for _, a, b in games.half_video_blocks(games.GAME):
    graded[a:b + 1] = True
e = np.diff(np.concatenate(([0], (gap0 > 0).astype(np.int8), [0])))
lengths = np.array([b - a for a, b in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1))
                    if graded[a] and graded[b - 1] and 75 <= b - a <= 2500])


def synth(seed, coverage=0.30):
    rng = np.random.RandomState(seed)
    n, g = near.copy(), gap0.copy()
    taken = np.zeros(len(near), dtype=bool)
    covered = 0
    for _ in range(400):
        if covered >= coverage * (W1 - W0):
            break
        L = int(rng.choice(lengths))
        if W1 - 30 - L <= W0 + 30:
            continue
        s = int(rng.randint(W0 + 30, W1 - 30 - L))
        if dead[s - 1:s + L + 1].any() or taken[s - 1:s + L + 1].any():
            continue
        taken[s:s + L] = True
        n[s:s + L] = 0
        g[s:s + L] = np.arange(1, L + 1)
        covered += L
    return assigner.assign_labels(n, d["restart_team"], dead, g, cfg,
                                  restart_prior=d["restart_prior"], margin=d["margin"])


def apply_policy(lab, policy):
    out, changed = lab.copy(), 0
    isx = (lab[W0:W1] == "X").astype(np.int8)
    ed = np.diff(np.concatenate(([0], isx, [0])))
    for s, t in zip(np.flatnonzero(ed == 1), np.flatnonzero(ed == -1)):
        rs, re = W0 + int(s), W0 + int(t)
        gs = rs - (CARRY - 1)
        if gs - 1 < W0 or re >= W1:
            continue
        pre, post = lab[gs - 1], lab[re]
        if pre not in ("H", "A") or post not in ("H", "A") or pre == post:
            continue
        L = re - gs
        if policy == "end":
            cut = re
        elif policy == "mid":
            cut = gs + L // 2
        else:
            inside = [f for f, a, b in gt_trans if gs <= f < re and a in (pm.H, pm.A) and b in (pm.H, pm.A)]
            cut = inside[0] if inside else gs + L // 2
        out[gs:cut] = pre
        out[cut:re] = post
        changed += 1
    return out, changed


policies = ["current", "end", "mid", "oracle"]
res = {p: [] for p in policies}
touched = []
for seed in range(20):
    lab = synth(seed)
    for p in policies:
        out, ch = (lab, 0) if p == "current" else apply_policy(lab, p)
        if p == "mid":
            touched.append(ch)
        r = pm.evaluate(gt, pm.label_codes(out))
        res[p].append((r["spell_f1_mean"], r["transitions"][25]["f1"], r["n_pred_spells"]))

ref = pm.evaluate(gt, pm.label_codes(assigner.assign_labels(
    near, d["restart_team"], dead, gap0, cfg, restart_prior=d["restart_prior"], margin=d["margin"])))
print("reference, real window with no synthetic gaps: spellF1 %.3f  transF1@25 %.3f  possessions %d"
      % (ref["spell_f1_mean"], ref["transitions"][25]["f1"], ref["n_pred_spells"]))
print("gaps with a switch that the midpoint rule changed, per seed: %.1f" % np.mean(touched))
print("\npolicy   | spellF1 mean (sd) | transF1@25 | possessions | paired difference vs current: mean   seeds better")
base = np.array([r[0] for r in res["current"]])
for p in policies:
    a = np.array(res[p])
    diff = a[:, 0] - base
    print("%-8s | %.3f (%.3f)      | %.3f      | %5.1f       | %+.3f   %d of 20"
          % (p, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean(), diff.mean(), int((diff > 0).sum())))
