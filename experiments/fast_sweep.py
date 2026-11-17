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
d = np.load(os.path.join(ROOT, "experiments", "cache", "inputs.npz"))
gt = pm.label_codes(pm.read_label_csv(os.path.join(
    ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
runs_full = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
assigner = assigners.DecoderAssigner()


def run(penalty, soft):
    c = dict(cfg)
    c["switch_penalty"] = penalty
    c["soft_evidence"] = soft
    return assigner.assign_labels(d["near_team"], d["restart_team"], d["dead_mask"],
                                  d["ball_gap"], c, restart_prior=d["restart_prior"],
                                  margin=d["margin"])


print("mode penalty | public: poss spellF1 transF1@25 macroF1 | whole match: possessions mean_s")
jobs = [("hard", p) for p in (20, 25, 30, 32, 35, 37, 40, 43, 46, 50, 55, 60, 70)]
jobs += [("soft", p) for p in (35, 50, 70)]
for mode, p in jobs:
    pred = pm.label_codes(run(p, mode == "soft"))
    r = pm.evaluate(gt, pred)
    sp = pm.possession_spells(pm.fill_abstentions(pred, runs_full), runs_full)
    mean_s = np.mean([e - s for s, e, _ in sp]) / 25.0
    print("%4s %7d | %12d %7.3f %10.3f %7.3f | %20d %6.1f"
          % (mode, p, r["n_pred_spells"], r["spell_f1_mean"], r["transitions"][25]["f1"],
             r["macro_f1"], len(sp), mean_s))
