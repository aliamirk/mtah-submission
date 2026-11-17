import numpy as np

from src.decoder import decode_segments


def _midpoint_gap_switches(labels, carry):
    out = labels.copy()
    isx = (labels == "X").astype(np.int8)
    e = np.diff(np.concatenate(([0], isx, [0])))
    for rs, re in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        gs, re = int(rs) - (carry - 1), int(re)
        if gs < 1 or re >= len(labels):
            continue
        pre, post = labels[gs - 1], labels[re]
        if pre in ("H", "A") and post in ("H", "A") and pre != post:
            cut = gs + (re - gs) // 2
            out[gs:cut] = pre
            out[cut:re] = post
    return out


class DecoderAssigner:
    """Viterbi decode of the team evidence, hard (+1/-1) or weighted by reliability."""

    def assign_labels(self, near_team, restart_team, dead_mask, ball_gap, cfg,
                      restart_prior=None, margin=None):
        long_gap = ball_gap >= int(cfg["max_carry_gap"])
        soft = bool(cfg.get("soft_evidence", False)) and margin is not None
        evidence = near_team.astype(np.float64)
        if soft:
            gap_m = np.abs(np.nan_to_num(margin, nan=0.0, posinf=1e6, neginf=1e6))
            weight = np.minimum(float(cfg["evidence_a"]) + float(cfg["evidence_b"]) * gap_m,
                                float(cfg["evidence_cap"]))
            evidence = near_team * weight
        w_tag = float(cfg.get("tag_vote_weight", 0.0))
        if w_tag:
            evidence = evidence + w_tag * restart_team
        w_prior = float(cfg.get("restart_prior_weight", 0.0))
        if restart_prior is not None and w_prior:
            evidence = evidence + w_prior * restart_prior
        evidence[dead_mask | long_gap] = 0.0
        if not soft:
            evidence = np.sign(evidence)

        state = decode_segments(evidence, ~dead_mask, float(cfg["switch_penalty"]))
        labels = np.where(state == 0, "H", "A").astype("<U1")
        labels[state < 0] = "D"
        labels[dead_mask] = "D"
        labels[long_gap & ~dead_mask] = "X"
        if str(cfg.get("gap_switch_policy", "current")) == "mid":
            labels = _midpoint_gap_switches(labels, int(cfg["max_carry_gap"]))
        return labels
