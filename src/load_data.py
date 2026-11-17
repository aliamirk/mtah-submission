"""load_data.py — ATD loading, frame-index mapping, and coordinate conversion.

This module exposes two pure utility functions used by all downstream pipeline
stages, plus the ``TrackingLoader`` class that handles the full ATD and
ground-truth loading.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository root — needed to locate games.py and the GT CSV
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import games  # noqa: E402 — must follow sys.path setup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------


def frame_id_to_row(frame_id: int) -> int:
    """Convert an ATD frame_id to a zero-based submission row index.

    The ATD numbers records starting at 1; the submission CSV is indexed
    starting at 0.  The mapping is therefore ``row = frame_id - 1``.

    Parameters
    ----------
    frame_id:
        The 1-based frame identifier as it appears in the ATD file.
        Must be >= 1.

    Returns
    -------
    int
        The corresponding zero-based submission row (0..199318).

    Raises
    ------
    ValueError
        If *frame_id* is less than 1 (ATD frame ids are strictly positive).
    """
    if int(frame_id) < 1:
        raise ValueError(
            f"frame_id must be >= 1, got {frame_id!r}. "
            "ATD frame ids start at 1; submission rows start at 0."
        )
    return int(frame_id) - 1


def norm_to_metres(
    x_norm: float,
    y_norm: float,
    pitch_length: float,
    pitch_width: float,
) -> tuple[float, float]:
    """Convert Metrica normalised coordinates [0, 1] to pitch metres.

    Metrica ATD coordinates are normalised so that (0, 0) is the top-left
    corner of the pitch and (1, 1) is the bottom-right corner.  Multiply by
    the physical pitch dimensions to obtain metres.

    Parameters
    ----------
    x_norm:
        Normalised x-coordinate in [0, 1] (left goal line → right goal line).
    y_norm:
        Normalised y-coordinate in [0, 1] (bottom touchline → top touchline).
    pitch_length:
        Physical pitch length in metres (x-axis).  Typically 105.3 m.
    pitch_width:
        Physical pitch width in metres (y-axis).  Typically 68.0 m.

    Returns
    -------
    tuple[float, float]
        ``(x_metres, y_metres)`` in physical pitch coordinates.
    """
    return x_norm * pitch_length, y_norm * pitch_width


# ---------------------------------------------------------------------------
# Total number of frames in the submission
# ---------------------------------------------------------------------------
_N_FRAMES: int = 199319


class TrackingLoader:
    """Load the tactical ATD and the public ground-truth labels.

    All file paths are resolved through :func:`games.asset_paths` or derived
    from the repository root — no filenames are ever hardcoded.
    """

    # ------------------------------------------------------------------
    # Frame table
    # ------------------------------------------------------------------

    def load_frame_table(self, cfg: dict) -> pd.DataFrame:
        """Load the tactical ATD and return a tidy per-frame DataFrame.

        The DataFrame has exactly 199,319 rows (one per submission row) indexed
        0..199318.  Each row carries:

        * ``ball_x_m``, ``ball_y_m``  — ball coordinates in metres (NaN when
          the ball is not tracked in that frame).
        * ``ball_visible``             — ``True`` iff both ball coordinates are
          non-NaN.
        * ``{slot}_x_m``, ``{slot}_y_m`` — player x/y in metres for each
          track slot (named by the player's ``player_id`` in the ATD, e.g.
          ``track_1``).  NaN when the player is not tracked in that frame.
        * ``{slot}_team``              — ``"home"``, ``"away"``, or
          ``"unknown"``; derived from the ATD metadata once at load time.

        Parameters
        ----------
        cfg:
            Pipeline configuration dict (must contain ``pitch_length`` and
            ``pitch_width``).

        Returns
        -------
        pd.DataFrame
            Frame table with integer index 0..199318.

        Raises
        ------
        IOError
            If the ATD files are not found (propagated from
            :func:`games.require`).
        ImportError
            If *kloppy* is not installed.
        """
        try:
            from kloppy import metrica
            from kloppy.domain import Ground
        except ImportError as exc:
            raise ImportError(
                "kloppy is required by TrackingLoader.  "
                "Install it with:  pip install kloppy==3.18"
            ) from exc

        pitch_length: float = cfg["pitch_length"]
        pitch_width: float = cfg["pitch_width"]

        # Resolve file paths via games — never hardcode filenames.
        paths = games.asset_paths()  # default: game=GAME, source="tactical"
        meta_path: str = paths["atd_meta"]
        raw_path: str = paths["atd_raw"]

        logger.info("Loading ATD from %s", raw_path)
        dataset = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path)

        # ------------------------------------------------------------------
        # Build a list of row-dicts; one dict per ATD record.
        # ------------------------------------------------------------------
        rows: list[dict] = []

        for frame in dataset:
            row_idx: int = frame_id_to_row(frame.frame_id)
            row: dict = {"_row": row_idx}

            # Ball coordinates (Point3D; may be None or have NaN x/y).
            bc = frame.ball_coordinates
            if (
                bc is not None
                and bc.x is not None
                and bc.y is not None
                and not (isinstance(bc.x, float) and math.isnan(bc.x))
                and not (isinstance(bc.y, float) and math.isnan(bc.y))
            ):
                bx_m, by_m = norm_to_metres(bc.x, bc.y, pitch_length, pitch_width)
                row["ball_x_m"] = bx_m
                row["ball_y_m"] = by_m
            else:
                row["ball_x_m"] = float("nan")
                row["ball_y_m"] = float("nan")

            # Player coordinates — keyed by Player object.
            for player, player_data in frame.players_data.items():
                # player.player_id is the raw ATD id (e.g. "1", "2" …).
                # Prefix with "track_" to produce the kloppy-convention slot
                # name used everywhere in the pipeline (e.g. "track_1").
                slot: str = f"track_{player.player_id}"

                # Determine team string from the player's team ground.
                ground = player.team.ground if player.team is not None else None
                if ground == Ground.HOME:
                    team_str = "home"
                elif ground == Ground.AWAY:
                    team_str = "away"
                else:
                    team_str = "unknown"

                row[f"{slot}_team"] = team_str

                coords = player_data.coordinates
                if (
                    coords is not None
                    and coords.x is not None
                    and coords.y is not None
                    and not (isinstance(coords.x, float) and math.isnan(coords.x))
                    and not (isinstance(coords.y, float) and math.isnan(coords.y))
                ):
                    px_m, py_m = norm_to_metres(
                        coords.x, coords.y, pitch_length, pitch_width
                    )
                    row[f"{slot}_x_m"] = px_m
                    row[f"{slot}_y_m"] = py_m
                else:
                    row[f"{slot}_x_m"] = float("nan")
                    row[f"{slot}_y_m"] = float("nan")

            rows.append(row)

        # ------------------------------------------------------------------
        # Build DataFrame indexed by submission row.
        # ------------------------------------------------------------------
        df = pd.DataFrame(rows)
        df = df.set_index("_row")
        df.index.name = None  # clean integer index

        # Reindex to the full 0..199318 range; any missing ATD frames become
        # NaN rows so downstream code never sees index gaps.
        full_index = pd.RangeIndex(0, _N_FRAMES)
        n_before = len(df)
        df = df.reindex(full_index, fill_value=float("nan"))
        n_filled = _N_FRAMES - n_before
        if n_filled > 0:
            logger.warning(
                "%d ATD frame(s) were missing and filled with NaN "
                "(expected %d records, got %d).",
                n_filled,
                _N_FRAMES,
                n_before,
            )

        # ------------------------------------------------------------------
        # ball_visible: True only where both ball coordinates are finite.
        # ------------------------------------------------------------------
        df["ball_visible"] = (
            df["ball_x_m"].notna() & df["ball_y_m"].notna()
        )

        # ------------------------------------------------------------------
        # Ensure *_team columns are string "unknown" for any NaN-filled rows
        # (reindex fills them with NaN float, but downstream expects strings).
        # ------------------------------------------------------------------
        team_cols = [c for c in df.columns if c.endswith("_team")]
        for col in team_cols:
            df[col] = df[col].fillna("unknown")

        logger.info(
            "Frame table built: %d rows, %d columns, %.1f%% ball visible.",
            len(df),
            len(df.columns),
            100.0 * df["ball_visible"].mean(),
        )
        return df

    # ------------------------------------------------------------------
    # Ground-truth loader
    # ------------------------------------------------------------------

    def load_ground_truth(self, cfg: dict) -> pd.Series:
        """Load the public ground-truth labels and return a full-length Series.

        The GT CSV lives at ``evaluation/gt/BAR-ATH_possession_gt_public.csv``
        relative to the repository root.  It is **not** a match asset returned
        by :func:`games.asset_paths`; its path is constructed from ``_ROOT``
        directly.

        The CSV has columns ``frame`` (int, 0-based submission row) and
        ``label`` (one of ``H``, ``A``, ``D``, ``X``).  Frames not present in
        the file receive the value ``"X"`` — they are ungraded.  The returned
        Series always has exactly 199,319 elements indexed ``0..199318``.

        Parameters
        ----------
        cfg:
            Pipeline configuration dict.  Not currently used for path
            resolution but accepted for interface consistency with
            ``load_frame_table``.

        Returns
        -------
        pd.Series
            Integer-indexed Series (0..199318) with string labels in
            ``{"H", "A", "D", "X"}``.  dtype is ``object``.

        Raises
        ------
        FileNotFoundError
            If the GT CSV does not exist at the expected path.
        """
        gt_path = os.path.join(
            _ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"
        )
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"Public ground-truth CSV not found at: {gt_path}\n"
                "Expected location: evaluation/gt/BAR-ATH_possession_gt_public.csv "
                "relative to the repository root."
            )

        gt_df = pd.read_csv(gt_path, dtype={"frame": int, "label": str})

        # Build the full-length Series, defaulting every frame to "X".
        full_index = pd.RangeIndex(0, _N_FRAMES)
        gt_series = pd.Series("X", index=full_index, dtype=object)

        # Overwrite with whatever the CSV provides (may already be all rows).
        gt_df = gt_df.set_index("frame")
        gt_series.update(gt_df["label"])

        logger.info(
            "Ground-truth loaded: %d total frames, %d graded (non-X).",
            len(gt_series),
            int((gt_series != "X").sum()),
        )
        return gt_series
