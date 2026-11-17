"""config_utils.py — Configuration loading and validation.

Loads ``config.yaml`` (or any YAML file at the supplied path), checks that
every required key is present, and raises a descriptive ``KeyError`` naming
the *first* missing key if any are absent.
"""

from __future__ import annotations

import yaml


# ---------------------------------------------------------------------------
# All keys that every downstream module expects to find in the config dict.
# Keep this list in sync with ``config.yaml``.
# ---------------------------------------------------------------------------
_REQUIRED_KEYS: tuple[str, ...] = (
    # Pitch / video
    "pitch_length",
    "pitch_width",
    "fps",
    "nonplay_regions",
    # Event detection
    "speed_jump_threshold",
    "direction_change_threshold",
    "min_speed_for_direction",
    "stop_speed_threshold",
    "reappear_gap_threshold",
    # Control-score weights
    "w_distance",
    "w_approach",
    "w_ball_side",
    "w_space",
    # Label decision
    "home_threshold",
    "away_threshold",
    "epsilon",
    # Possession stability
    "max_carry_gap",
    "min_lock_frames",
    "consecutive_events_required",
    "confirm_threshold_low",
    "confirm_threshold_high",
    # Smart Tag priors
    "tag_prior_weight",
    "dead_ball_buffer_frames",
)


def load_config(path: str) -> dict:
    """Load and validate a YAML configuration file.

    Parameters
    ----------
    path:
        File-system path to the YAML config file (e.g. ``"config.yaml"``).

    Returns
    -------
    dict
        The parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If a required key is absent from the file.  The message names the
        missing key so the caller can fix the config without digging through
        source code.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    if cfg is None:
        cfg = {}

    for key in _REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(
                f"Required key '{key}' is missing from config file '{path}'. "
                f"Add it to {path} and retry."
            )

    return cfg
