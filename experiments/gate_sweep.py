import os, sys, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
games = importlib.import_module("games")
pm = importlib.import_module("possession_metrics")
config_utils = importlib.import_module("src.config_utils")
assigners = importlib.import_module("src.decoder_assigner")

cfg = config_utils.load_config(os.path.join(ROOT, "config.yaml"))
inp = np.load(os.path.join(ROOT, "experiments", "cache", "inputs.npz"))
dist = np.load(os.path.join(ROOT, "experiments", "cache", "dist.npz"))
dmin = np.fmin(dist["d_home"], dist["d_away"])
assigner = assigners.DecoderAssigner()

gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs = pm.valid_runs(gt)
runs_full = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
long_gt = [s for s in pm.possession_spells(gt, runs) if s[1] - s[0] > 90]


def run(radius, penalty):
    near = inp["near_team"].copy()
    if radius:
        near[~(dmin <= radius)] = 0
    c = dict(cfg)
    c["switch_penalty"] = penalty
    c["soft_evidence"] = False
    return assigner.assign_labels(near, inp["restart_team"], inp["dead_mask"], inp["ball_gap"],
                                  c, restart_prior=inp["restart_prior"], margin=inp["margin"])


def long_stats(pred):
    p = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    ious, se, ee = [], [], []
    for gs, ge, gc in long_gt:
        best, bi = 0.0, None
        for i, (ps, pe, pc) in enumerate(p):
            if pc != gc or pe <= gs or ge <= ps:
                continue
            iou = (min(ge, pe) - max(gs, ps)) / float(max(ge, pe) - min(gs, ps))
            if iou > best:
                best, bi = iou, i
        ious.append(best)
        if bi is not None and best >= 0.3:
            se.append(abs(p[bi][0] - gs))
            ee.append(abs(p[bi][1] - ge))
    return np.mean(ious), (np.median(se) if se else float("nan")), (np.median(ee) if ee else float("nan"))


print("gate m  penalty | public: poss spellF1 transF1 | 11 long spells: mean best IoU, median start/end error (frames) | whole match: possessions, mean s")
for radius in (None, 3.0, 2.0, 1.5, 1.0):
    for penalty in (10, 20, 30):
        pred = pm.label_codes(run(radius, penalty))
        r = pm.evaluate(gt, pred)
        mi, se, ee = long_stats(pred)
        sp = pm.possession_spells(pm.fill_abstentions(pred, runs_full), runs_full)
        mean_s = np.mean([e - s for s, e, _ in sp]) / 25.0
        print("%6s %8d | %11d %7.3f %7.3f | %13.2f %9.0f %6.0f | %17d %6.1f"
              % ("none" if radius is None else radius, penalty, r["n_pred_spells"], r["spell_f1_mean"],
                 r["transitions"][25]["f1"], mi, se, ee, len(sp), mean_s))
