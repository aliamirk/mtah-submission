"""main.py: possession-pipeline entry point.

The pipeline, step by step (the reasoning is in NOTES.md and the technical document):

1. Load the configuration (config.yaml).
2. Load the tactical tracking (ATD) into a frame table, and the public ground truth,
   which is only used by step 9 to print a local report.
3. Compute ball kinematics and player velocities.
4. Build the dead-ball mask: fixed non-play ranges and capped Smart Tag windows
   (kickoff tags are unmasked from the published kickoff frame), plus sustained
   low player speed ("quiet runs").
5. Compute the nearest-player team signal and its margin.
6. Build the restart-team array and the restart prior from Smart Tags.
7. Decode H/A with a two-state Viterbi decoder; D comes from the dead mask and X
   from long ball gaps.
8. Validate and write the submission file.
9. Evaluate against the public ground-truth window and print a report.

Usage
-----
    python main.py                           # writes evaluation/results/los-sonambulos.csv
    python main.py --config config.yaml
    python main.py --output path/to/file.csv # relative paths start at the repository root
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time

import numpy as np

# Make the repository root importable, whatever directory main.py is run from.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from games import require as _require  # noqa: E402
from src.config_utils import load_config  # noqa: E402
from src.decoder_assigner import DecoderAssigner  # noqa: E402
from src.evaluate import LocalEvaluator  # noqa: E402
from src.load_data import TrackingLoader  # noqa: E402
from src.output import validate_submission_columns, write_submission  # noqa: E402
from src.preprocess import (  # noqa: E402
    KinematicsBuilder,
    nearest_player_team,
    nearest_team_margin,
    reach_gated_team,
)
from src.quiet import player_speed_series, quiet_runs  # noqa: E402
from src.restart import restart_prior_from_runs  # noqa: E402
from src.smart_tags import TagProcessor  # noqa: E402
from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The submission file is named after the team, as SUBMISSION.md asks.
DEFAULT_OUTPUT = os.path.join("evaluation", "results", "los-sonambulos.csv")


def _resolve_output(cli_value: str | None, cfg: dict) -> str:
    """Output path: command-line flag, then the 'output_path' config key, then the default.

    Relative paths are taken from the repository root, not from the working directory.
    """
    chosen = cli_value or cfg.get("output_path") or DEFAULT_OUTPUT
    return chosen if os.path.isabs(chosen) else os.path.join(_ROOT, chosen)


def _sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run the full possession-prediction pipeline and write the submission file."""
    parser = argparse.ArgumentParser(
        description="Ball-possession prediction pipeline for Metrica More Than a Hack 2027."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the submission CSV (default: %s)." % DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ==================================================================
    # Step 1: configuration
    # ==================================================================
    logger.info("Step 1/9 - Loading configuration from '%s'", args.config)
    cfg: dict = load_config(args.config)
    logger.info("Configuration loaded successfully (%d keys).", len(cfg))

    # ==================================================================
    # Step 2: ATD frame table and public ground-truth labels
    # ==================================================================
    logger.info("Step 2/9 - Loading ATD frame table and ground-truth labels")
    loader = TrackingLoader()
    frame_table = loader.load_frame_table(cfg)
    # only read again by the local evaluation in step 9
    gt = loader.load_ground_truth(cfg)
    logger.info(
        "Frame table: %d rows x %d columns.  GT: %d rows.",
        len(frame_table),
        len(frame_table.columns),
        len(gt),
    )

    # ==================================================================
    # Step 3: ball kinematics and player velocities
    # ==================================================================
    logger.info("Step 3/9 - Computing ball kinematics and player velocities")
    builder = KinematicsBuilder()
    frame_table = builder.build_ball_kinematics(frame_table, float(cfg["fps"]))
    frame_table = builder.build_player_velocities(
        frame_table, float(cfg["fps"]))
    logger.info(
        "Kinematics added. Frame table now has %d columns.", len(
            frame_table.columns)
    )

    # ==================================================================
    # Step 4: dead-ball mask
    # ==================================================================
    logger.info(
        "Step 4/9 - Processing Smart Tags and building the dead-ball mask")
    tagger = TagProcessor()
    tags = tagger.load_tags(cfg)
    dead_mask = tagger.build_dead_ball_mask(tags, len(frame_table), cfg)

    # A kickoff tag is a 5 s window around the restart, not dead ball. Play starts on the
    # published kickoff frame (DATA.md), so only the part before it stays dead.
    for tag in tags:
        if tag["tags"].get("Type", "").strip().lower() == "kick off (start)":
            for k in KICKOFF_VIDEO_FRAMES.values():
                if tag["start_frame"] <= k <= tag["end_frame"]:
                    dead_mask[k: tag["end_frame"] + 11] = False

    # Quiet runs: sustained low player speed inside live play is a stoppage.
    qruns: list = []
    if cfg.get("quiet_enabled", True):
        fps_q = float(cfg["fps"])
        allowed = np.ones(len(frame_table), dtype=bool)
        for lo, hi in cfg["nonplay_regions"]:
            allowed[lo:hi + 1] = False
        spd = player_speed_series(
            _require(["atd_raw"])[0],
            len(frame_table),
            os.path.join(_ROOT, "cache", "player_speed.npy"),
        )
        qruns = quiet_runs(
            spd,
            int(cfg["quiet_smooth_frames"]),
            float(cfg["quiet_speed_threshold"]),
            int(cfg["quiet_min_seconds"] * fps_q),
            int(cfg["quiet_bridge_seconds"] * fps_q),
            allowed,
        )
        before = int(dead_mask.sum())
        for lo, hi in qruns:
            dead_mask[lo:hi] = True
        logger.info(
            "Quiet runs: %d runs, %d frames; dead mask %d -> %d.",
            len(qruns), sum(b - a for a, b in qruns), before, int(dead_mask.sum()),
        )

    # Computed for completeness; the decoder does not use these arrays.
    home_prior, away_prior = tagger.build_team_priors(
        tags, len(frame_table), cfg)
    logger.info(
        "Dead-ball mask: %d frames masked (%.1f%%).  Tags loaded: %d.",
        int(dead_mask.sum()),
        100.0 * dead_mask.mean(),
        len(tags),
    )

    # ==================================================================
    # Step 5: nearest-player team signal
    # ==================================================================
    logger.info("Step 5/9 - Computing nearest-player team signal")
    near_team: np.ndarray = nearest_player_team(frame_table)
    margin: np.ndarray = nearest_team_margin(frame_table)
    both = (near_team != 0) & np.isfinite(margin) & (margin != 0)
    logger.info(
        "Margin sign agrees with nearest-player on %.4f of %d frames.",
        float(np.mean(np.sign(margin[both]) ==
              near_team[both])), int(both.sum()),
    )
    logger.info(
        "Nearest-player signal: home=%d frames, away=%d frames, no-ball=%d frames.",
        int((near_team == 1).sum()), int(
            (near_team == -1).sum()), int((near_team == 0).sum()),
    )

    # Speed-gated reachability: tested and rejected (see NOTES.md), off by default.
    if cfg.get("reach_enabled", False):
        near_before = near_team.copy()
        near_team = reach_gated_team(
            frame_table, near_team,
            gate=float(cfg["reach_gate_mps"]),
            horizon=float(cfg["reach_horizon_s"]),
            fps=float(cfg["fps"]),
        )
        logger.info("Reach gate: %d frames changed their team call.",
                    int((near_team != near_before).sum()))

    # ==================================================================
    # Step 6: restart-team array and restart prior
    # ==================================================================
    logger.info(
        "Step 6/9 - Building restart-team array and restart prior from Smart Tags")
    restart_team: np.ndarray = tagger.build_restart_team_array(
        tags, len(frame_table), cfg)
    restart_prior: np.ndarray = tagger.build_restart_prior(
        tags, dead_mask, len(frame_table), cfg
    )
    if cfg.get("restart_from_runs", False) and cfg.get("quiet_enabled", True):
        restart_prior = restart_prior_from_runs(
            tags, qruns, len(frame_table), cfg)
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
    # Step 7: decode the labels
    # ==================================================================
    logger.info("Step 7/9 - Assigning possession labels (DecoderAssigner)")
    assigner = DecoderAssigner()
    ball_gap: np.ndarray = frame_table["ball_gap"].to_numpy(dtype=np.int64)

    # The decoder inputs are saved so the scripts in experiments/ can rerun the decoder
    # in seconds. The folder is not tracked by git.
    os.makedirs(os.path.join(_ROOT, "experiments", "cache"), exist_ok=True)
    np.savez(
        os.path.join(_ROOT, "experiments", "cache", "inputs.npz"),
        near_team=near_team, restart_team=restart_team, dead_mask=dead_mask,
        ball_gap=ball_gap, margin=margin, restart_prior=restart_prior,
    )

    labels: np.ndarray = assigner.assign_labels(
        near_team, restart_team, dead_mask, ball_gap, cfg,
        restart_prior=restart_prior, margin=margin,
    )
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Label distribution: %s", {
                str(k): int(v) for k, v in zip(unique, counts)})

    # ==================================================================
    # Step 8: validate and write the submission file
    # ==================================================================
    logger.info("Step 8/9 - Writing the submission file")
    submission_path = _resolve_output(args.output, cfg)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    write_submission(labels, submission_path)
    validate_submission_columns(submission_path)
    logger.info("Submission validated: format OK.")
    logger.info("Submission written to '%s'.", submission_path)

    # ==================================================================
    # Step 9: local evaluation on the public ground-truth window
    # ==================================================================
    logger.info("Step 9/9 - Running local evaluation")
    evaluator = LocalEvaluator()
    metrics: dict = evaluator.run_evaluation(labels, gt, cfg)
    evaluator.print_report(metrics)

    diagnostic_path = os.path.join(
        _ROOT, "evaluation", "diagnostics", "diagnostic.csv")
    os.makedirs(os.path.dirname(diagnostic_path), exist_ok=True)
    evaluator.write_diagnostic_csv(labels, gt, diagnostic_path)
    logger.info("Diagnostic CSV written to '%s'.", diagnostic_path)

    macro_f1: float = float(metrics.get("macro_f1", 0.0))
    spell_f1: float = float(metrics.get("spell_f1_mean", 0.0))
    t_elapsed = time.perf_counter() - t_start

    print(
        f"Done. macro_f1={macro_f1:.4f}  spell_f1={spell_f1:.4f}"
        f"  rows={len(labels)}  elapsed={t_elapsed:.1f}s"
    )
    print(f"Submission file: {submission_path}")
    print(f"SHA-1: {_sha1(submission_path)}")


if __name__ == "__main__":
    main()
