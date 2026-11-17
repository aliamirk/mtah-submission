"""main.py — Possession-pipeline entry point.

Orchestrates the full nine-step pipeline:

1. Load configuration from ``config.yaml``.
2. Load the tactical ATD frame table and the public ground-truth labels.
3. Compute ball kinematics and player velocities.
4. Build the dead-ball mask and team-prior arrays from Smart Tags.
5. Detect ball-motion events.
6. Score team control (p_home) at each event frame.
7. Assign H/A/D/X labels to every frame via the state-machine assigner.
8. Validate and write ``submission.csv``.
9. Evaluate against the public ground-truth window and print a report.

Usage
-----
    python main.py                     # uses config.yaml by default
    python main.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Ensure repository root is on sys.path so that ``games`` and ``src.*`` are
# importable regardless of the working directory from which main.py is called.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config_utils import load_config
from src.evaluate import LocalEvaluator
from src.load_data import TrackingLoader
from src.output import write_submission, validate_submission_columns
from src.decoder_assigner import DecoderAssigner
from src.preprocess import KinematicsBuilder, nearest_player_team, nearest_team_margin
from src.smart_tags import TagProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full possession-prediction pipeline and write submission.csv."""
    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Ball-possession prediction pipeline for Metrica More Than a Hack 2027."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ==================================================================
    # Step 1 — Load configuration
    # ==================================================================
    logger.info("Step 1/9 — Loading configuration from '%s'", args.config)
    cfg: dict = load_config(args.config)
    logger.info("Configuration loaded successfully (%d keys).", len(cfg))

    # ==================================================================
    # Step 2 — Load ATD frame table and public ground-truth labels
    # ==================================================================
    logger.info("Step 2/9 — Loading ATD frame table and ground-truth labels")
    loader = TrackingLoader()
    frame_table = loader.load_frame_table(cfg)
    gt = loader.load_ground_truth(cfg)
    logger.info(
        "Frame table: %d rows × %d columns.  GT: %d rows.",
        len(frame_table),
        len(frame_table.columns),
        len(gt),
    )

    # ==================================================================
    # Step 3 — Build ball kinematics and player velocities
    # ==================================================================
    logger.info("Step 3/9 — Computing ball kinematics and player velocities")
    builder = KinematicsBuilder()
    frame_table = builder.build_ball_kinematics(frame_table, float(cfg["fps"]))
    frame_table = builder.build_player_velocities(frame_table, float(cfg["fps"]))
    logger.info(
        "Kinematics added. Frame table now has %d columns.", len(frame_table.columns)
    )

    # ==================================================================
    # Step 4 — Build dead-ball mask and team-prior arrays from Smart Tags
    # ==================================================================
    logger.info("Step 4/9 — Processing Smart Tags")
    tagger = TagProcessor()
    tags = tagger.load_tags(cfg)
    dead_mask = tagger.build_dead_ball_mask(tags, len(frame_table), cfg)
    # The kickoff tag is a 5 s window around the restart, not dead ball. Play starts
    # on the published kickoff frame (DATA.md), so only the part before it stays dead.
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    for tag in tags:
        if tag["tags"].get("Type", "").strip().lower() == "kick off (start)":
            for k in KICKOFF_VIDEO_FRAMES.values():
                if tag["start_frame"] <= k <= tag["end_frame"]:
                    dead_mask[k: tag["end_frame"] + 11] = False
    from games import require as _require
    from src.quiet import player_speed_series, quiet_runs
    if cfg.get("quiet_enabled", True):
        fps_q = float(cfg["fps"])
        allowed = np.ones(len(frame_table), dtype=bool)
        for lo, hi in cfg["nonplay_regions"]:
            allowed[lo:hi + 1] = False
        spd = player_speed_series(_require(["atd_raw"])[0], len(frame_table),
                                  os.path.join(_ROOT, "cache", "player_speed.npy"))
        qruns = quiet_runs(spd, int(cfg["quiet_smooth_frames"]), float(cfg["quiet_speed_threshold"]),
                           int(cfg["quiet_min_seconds"] * fps_q), int(cfg["quiet_bridge_seconds"] * fps_q),
                           allowed)
        before = int(dead_mask.sum())
        for lo, hi in qruns:
            dead_mask[lo:hi] = True
        logger.info("Quiet runs: %d runs, %d frames; dead mask %d -> %d.",
                    len(qruns), sum(b - a for a, b in qruns), before, int(dead_mask.sum()))
    home_prior, away_prior = tagger.build_team_priors(tags, len(frame_table), cfg)
    logger.info(
        "Dead-ball mask: %d frames masked (%.1f%%).  Tags loaded: %d.",
        int(dead_mask.sum()),
        100.0 * dead_mask.mean(),
        len(tags),
    )

    # ==================================================================
    # Step 5 — Compute nearest-player-to-ball team for every frame
    # ==================================================================
    logger.info("Step 5/9 — Computing nearest-player team signal")
    near_team: np.ndarray = nearest_player_team(frame_table)
    margin: np.ndarray = nearest_team_margin(frame_table)
    both = (near_team != 0) & np.isfinite(margin) & (margin != 0)
    logger.info('Margin sign agrees with nearest-player on %.4f of %d frames.',
                float(np.mean(np.sign(margin[both]) == near_team[both])), int(both.sum()))
    n_home_raw = int((near_team == 1).sum())
    n_away_raw = int((near_team == -1).sum())
    logger.info(
        "Nearest-player signal: home=%d frames, away=%d frames, no-ball=%d frames.",
        n_home_raw, n_away_raw, int((near_team == 0).sum()),
    )

    # ==================================================================
    # Step 6 — Build restart-team array and restart prior from Smart Tags
    # ==================================================================
    if cfg.get("reach_enabled", False):
        from src.preprocess import reach_gated_team
        near_before = near_team.copy()
        near_team = reach_gated_team(frame_table, near_team, gate=float(cfg["reach_gate_mps"]),
                                     horizon=float(cfg["reach_horizon_s"]), fps=float(cfg["fps"]))
        logger.info("Reach gate: %d frames changed their team call (%d frames above the gate speed).",
                    int((near_team != near_before).sum()), int(((near_team != 0) & (near_before != 0)).sum()))
    logger.info("Step 6/9 — Building restart-team array and restart prior from Smart Tags")
    restart_team: np.ndarray = tagger.build_restart_team_array(
        tags, len(frame_table), cfg
    )
    restart_prior: np.ndarray = tagger.build_restart_prior(
        tags, dead_mask, len(frame_table), cfg
    )
    if cfg.get('restart_from_runs', False) and cfg.get('quiet_enabled', True):
        from src.restart import restart_prior_from_runs
        restart_prior = restart_prior_from_runs(tags, qruns, len(frame_table), cfg)
    logger.info(
        "Restart-team array: home=%d frames, away=%d frames, neutral=%d frames.",
        int((restart_team == 1).sum()),
        int((restart_team == -1).sum()),
        int((restart_team == 0).sum()),
    )
    logger.info(
        "Restart prior: home=%d frames, away=%d frames, neutral=%d frames.",
        int((restart_prior == 1).sum()),
        int((restart_prior == -1).sum()),
        int((restart_prior == 0).sum()),
    )

    # ==================================================================
    # Step 7 — Assign H/A/D/X labels via AnchoredVoteAssigner
    # ==================================================================
    logger.info("Step 7/9 — Assigning possession labels (DecoderAssigner)")
    avassigner = DecoderAssigner()
    ball_gap: np.ndarray = frame_table["ball_gap"].to_numpy(dtype=np.int64)
    os.makedirs(os.path.join(_ROOT, "experiments", "cache"), exist_ok=True)
    np.savez(os.path.join(_ROOT, "experiments", "cache", "inputs.npz"),
             near_team=near_team, restart_team=restart_team, dead_mask=dead_mask,
             ball_gap=ball_gap, margin=margin, restart_prior=restart_prior)
    labels: np.ndarray = avassigner.assign_labels(
        near_team, restart_team, dead_mask, ball_gap, cfg,
        restart_prior=restart_prior, margin=margin,
    )
    unique, counts = np.unique(labels, return_counts=True)
    dist = {k: int(v) for k, v in zip(unique, counts)}
    logger.info("Label distribution: %s", dist)

    # ==================================================================
    # Step 8 — Validate and write submission.csv
    # ==================================================================
    logger.info("Step 8/9 — Writing submission.csv")
    submission_path = os.path.join(_ROOT, "submission.csv")
    write_submission(labels, submission_path)
    validate_submission_columns(submission_path)
    logger.info("submission.csv validated: format OK.")
    logger.info("submission.csv written to '%s'.", submission_path)

    # ==================================================================
    # Step 9 — Evaluate against the public ground-truth window
    # ==================================================================
    logger.info("Step 9/9 — Running local evaluation")
    evaluator = LocalEvaluator()
    metrics: dict = evaluator.run_evaluation(labels, gt, cfg)
    evaluator.print_report(metrics)

    diagnostic_path = os.path.join(_ROOT, "evaluation", "diagnostics", "diagnostic.csv")
    os.makedirs(os.path.dirname(diagnostic_path), exist_ok=True)
    evaluator.write_diagnostic_csv(labels, gt, diagnostic_path)
    logger.info("Diagnostic CSV written to '%s'.", diagnostic_path)

    # ==================================================================
    # Step 9.5 — Transition coverage diagnostic (disabled for new assigner)
    # ==================================================================
    # The event-triggered transition diagnostic is not applicable to the
    # AnchoredVoteAssigner; skip it silently.

    # ==================================================================
    # Final summary
    # ==================================================================
    macro_f1: float = float(metrics.get("macro_f1", 0.0))
    spell_f1: float = float(metrics.get("spell_f1_mean", 0.0))
    t_elapsed = time.perf_counter() - t_start

    print(
        f"Done. macro_f1={macro_f1:.4f}  spell_f1={spell_f1:.4f}"
        f"  submission rows=199319  elapsed={t_elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def _write_transition_coverage(
    labels: np.ndarray,
    gt: "pd.Series",
    event_frames: np.ndarray,
    cfg: dict,
    out_path: str,
) -> None:
    """Write a per-GT-transition diagnostic CSV to *out_path*.

    For each ground-truth possession transition, records whether the event
    detector fired nearby and whether the state machine matched the correct
    team label.

    Columns
    -------
    transition_frame, gt_from, gt_to, nearest_event_frame, event_distance,
    has_event_within_12, nearest_pred_trans_frame, pred_trans_distance,
    pred_trans_matches_gt_to, label_at_nearest_event

    Parameters
    ----------
    labels:
        Full prediction array of shape ``(199319,)``; values ``H/A/D/X``.
    gt:
        Ground-truth ``pd.Series`` indexed ``0..199318``.
    event_frames:
        Boolean array of shape ``(199319,)``; True where an event was detected.
    cfg:
        Pipeline configuration dict (unused; kept for interface consistency).
    out_path:
        Destination CSV path.  Parent directories are created if missing.
    """
    import sys as _sys

    eval_dir = str(_ROOT + os.sep + "evaluation")
    if eval_dir not in _sys.path:
        _sys.path.insert(0, eval_dir)
    import possession_metrics as pm  # noqa: PLC0415

    gt_array = np.array(gt.values, dtype=object)
    gt_codes: np.ndarray = pm.label_codes(gt_array)
    pred_array = np.array([str(v).strip().upper() for v in labels], dtype=object)
    pred_codes: np.ndarray = pm.label_codes(pred_array)

    runs = pm.valid_runs(gt_codes)
    filled_pred = pm.fill_abstentions(pred_codes, runs)

    gt_trans = pm.transitions(gt_codes, runs)      # [(frame, from_code, to_code)]
    pred_trans = pm.transitions(filled_pred, runs)  # [(frame, from_code, to_code)]

    # Pre-compute event frame indices for fast nearest-event lookup.
    event_indices = np.flatnonzero(event_frames)

    tol = 12  # ±12 frames

    rows = []
    for t_frame, from_code, to_code in gt_trans:
        # --- nearest event frame ---
        if len(event_indices) == 0:
            nearest_event_frame = None
            event_distance = None
            has_event = False
        else:
            dists = np.abs(event_indices - t_frame)
            nearest_idx = int(dists.argmin())
            event_distance = int(dists[nearest_idx])
            nearest_event_frame = int(event_indices[nearest_idx])
            has_event = event_distance <= tol

        # --- nearest predicted transition ---
        if pred_trans:
            pt_frames = np.array([pt[0] for pt in pred_trans], dtype=np.int64)
            pt_dists = np.abs(pt_frames - t_frame)
            nearest_pt_idx = int(pt_dists.argmin())
            nearest_pt_dist = int(pt_dists[nearest_pt_idx])
            nearest_pt_frame = int(pred_trans[nearest_pt_idx][0])
            nearest_pt_to = int(pred_trans[nearest_pt_idx][2])
            pred_trans_matches = (nearest_pt_dist <= tol) and (nearest_pt_to == to_code)
        else:
            nearest_pt_frame = None
            nearest_pt_dist = None
            pred_trans_matches = False

        # --- label the state machine assigned at the nearest event ---
        if nearest_event_frame is not None and has_event:
            label_at_event = str(labels[nearest_event_frame])
        else:
            label_at_event = None

        rows.append({
            "transition_frame": t_frame,
            "gt_from": pm.LABELS[from_code],
            "gt_to": pm.LABELS[to_code],
            "nearest_event_frame": nearest_event_frame,
            "event_distance": event_distance,
            "has_event_within_12": has_event,
            "nearest_pred_trans_frame": nearest_pt_frame,
            "pred_trans_distance": nearest_pt_dist,
            "pred_trans_matches_gt_to": pred_trans_matches,
            "label_at_nearest_event": label_at_event,
        })

    import pandas as _pd  # noqa: PLC0415
    df = _pd.DataFrame(rows, columns=[
        "transition_frame", "gt_from", "gt_to",
        "nearest_event_frame", "event_distance", "has_event_within_12",
        "nearest_pred_trans_frame", "pred_trans_distance",
        "pred_trans_matches_gt_to", "label_at_nearest_event",
    ])

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary statistics.
    n_total = len(rows)
    n_covered = sum(1 for r in rows if r["has_event_within_12"])
    pct = 100.0 * n_covered / n_total if n_total else 0.0
    covered_dists = [r["event_distance"] for r in rows if r["has_event_within_12"]]
    mean_dist = (sum(covered_dists) / len(covered_dists)) if covered_dists else float("nan")
    n_match = sum(1 for r in rows if r["has_event_within_12"] and r["label_at_nearest_event"] ==
                  r["gt_to"])

    logger.info(
        "Transition coverage: %d/%d transitions have event within ±12 frames (%.1f%%)",
        n_covered, n_total, pct,
    )
    logger.info("Mean event-to-transition distance: %.1f frames", mean_dist)
    logger.info("Label matches at covered transitions: %d/%d", n_match, n_covered)
    print(
        f"Transition coverage: {n_covered}/{n_total} transitions have "
        f"event within ±12 frames ({pct:.1f}%)"
    )
    print(f"Mean event-to-transition distance: {mean_dist:.1f} frames")
    print(f"Label matches at covered transitions: {n_match}/{n_covered}")


def _write_debug_events(
    labels: np.ndarray,
    event_frames: np.ndarray,
    p_home_values: np.ndarray,
    frame_table: "pd.DataFrame",
    cfg: dict,
    out_path: str,
) -> None:
    """Write a per-event debug CSV to *out_path*.

    For every detected ball-motion event, records kinematics, the p_home
    score, the event classification, and the label the state machine assigned.

    Columns
    -------
    submission_row, atd_frame_id, ball_speed_raw, ball_speed_smoothed,
    p_home, event_type, chosen_candidate, current_label_after

    Parameters
    ----------
    labels:
        Full prediction label array, shape ``(n_frames,)``.
    event_frames:
        Boolean event mask, shape ``(n_frames,)``.
    p_home_values:
        Control-score array, shape ``(n_frames,)``.
    frame_table:
        Kinematics-enriched frame table (must contain ``ball_speed_mps_raw``,
        ``ball_speed_mps``, ``ball_direction_rad``, ``ball_gap``).
    cfg:
        Pipeline configuration dict; reads thresholds for event classification.
    out_path:
        Destination CSV path.  Parent directories are created if missing.
    """
    import math as _math
    import pandas as _pd  # noqa: PLC0415

    speed_jump_thresh: float = float(cfg["speed_jump_threshold"])
    dir_change_thresh: float = float(cfg["direction_change_threshold"])
    stop_thresh: float = float(cfg["stop_speed_threshold"])
    reappear_gap: int = int(cfg["reappear_gap_threshold"])
    home_threshold: float = float(cfg["home_threshold"])
    away_threshold: float = float(cfg["away_threshold"])

    event_indices = np.flatnonzero(event_frames)

    # Pre-extract columns as numpy arrays for fast index access.
    has_speed_raw = "ball_speed_mps_raw" in frame_table.columns
    has_speed = "ball_speed_mps" in frame_table.columns
    has_dir = "ball_direction_rad" in frame_table.columns
    has_gap = "ball_gap" in frame_table.columns

    speed_raw_arr = frame_table["ball_speed_mps_raw"].to_numpy() if has_speed_raw else None
    speed_arr = frame_table["ball_speed_mps"].to_numpy() if has_speed else None
    dir_arr = frame_table["ball_direction_rad"].to_numpy() if has_dir else None
    gap_arr = frame_table["ball_gap"].to_numpy(dtype=np.int64) if has_gap else None

    rows = []
    for idx in event_indices:
        submission_row = int(idx)
        atd_frame_id = submission_row + 1

        ball_speed_raw = float(speed_raw_arr[idx]) if speed_raw_arr is not None else float("nan")
        ball_speed_smoothed = float(speed_arr[idx]) if speed_arr is not None else float("nan")

        # Classify event type by checking which triggers fired.
        triggers: list[str] = []

        # Speed jump: compare to previous valid raw speed within ±50 frames.
        prev_speed_raw = float("nan")
        if speed_raw_arr is not None:
            for j in range(int(idx) - 1, max(int(idx) - 51, -1), -1):
                v = float(speed_raw_arr[j])
                if not _math.isnan(v):
                    prev_speed_raw = v
                    break
        if (not _math.isnan(ball_speed_raw) and not _math.isnan(prev_speed_raw)
                and abs(ball_speed_raw - prev_speed_raw) > speed_jump_thresh):
            triggers.append("speed_jump")

        # Direction change: compare to previous valid direction.
        if dir_arr is not None:
            curr_dir = float(dir_arr[idx])
            prev_dir = float("nan")
            for j in range(int(idx) - 1, max(int(idx) - 51, -1), -1):
                v = float(dir_arr[j])
                if not _math.isnan(v):
                    prev_dir = v
                    break
            if (not _math.isnan(curr_dir) and not _math.isnan(prev_dir)
                    and not _math.isnan(ball_speed_smoothed)
                    and ball_speed_smoothed > float(cfg["min_speed_for_direction"])):
                angle_diff = abs(_math.atan2(
                    _math.sin(curr_dir - prev_dir), _math.cos(curr_dir - prev_dir)
                ))
                if angle_diff > dir_change_thresh:
                    triggers.append("direction_change")

        # Stop: ball transitions from moving to stopped.
        if (not _math.isnan(ball_speed_smoothed) and not _math.isnan(prev_speed_raw)
                and prev_speed_raw > stop_thresh
                and ball_speed_smoothed <= stop_thresh):
            triggers.append("stop")

        # Reappear: ball was absent for >= reappear_gap_threshold frames.
        if gap_arr is not None and idx > 0:
            prev_gap = int(gap_arr[idx - 1])
            if prev_gap >= reappear_gap:
                triggers.append("reappear")

        if len(triggers) == 0:
            event_type = "unknown"
        elif len(triggers) == 1:
            event_type = triggers[0]
        else:
            event_type = "multiple"

        # Candidate label from p_home.
        p = float(p_home_values[idx]) if not _math.isnan(float(p_home_values[idx])) else float("nan")
        if _math.isnan(p):
            chosen_candidate = "X"
        elif p >= home_threshold:
            chosen_candidate = "H"
        elif p <= away_threshold:
            chosen_candidate = "A"
        else:
            chosen_candidate = "X"

        current_label_after = str(labels[idx])

        rows.append({
            "submission_row": submission_row,
            "atd_frame_id": atd_frame_id,
            "ball_speed_raw": ball_speed_raw,
            "ball_speed_smoothed": ball_speed_smoothed,
            "p_home": p,
            "event_type": event_type,
            "chosen_candidate": chosen_candidate,
            "current_label_after": current_label_after,
        })

    df = _pd.DataFrame(rows, columns=[
        "submission_row", "atd_frame_id", "ball_speed_raw", "ball_speed_smoothed",
        "p_home", "event_type", "chosen_candidate", "current_label_after",
    ])

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
