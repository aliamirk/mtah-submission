"""preprocess.py — Ball kinematics and player velocity computation.

This module enriches the raw frame table produced by :mod:`src.load_data`
with derived kinematic columns required by the possession engine.

Two methods are exposed through :class:`KinematicsBuilder`:

* :meth:`KinematicsBuilder.build_ball_kinematics` — adds ``ball_gap``,
  ``ball_speed_mps``, ``ball_direction_rad``, and ``ball_accel_mpss``
  (implemented in Task 5).
* :meth:`KinematicsBuilder.build_player_velocities` — adds
  ``{slot}_vx_mps`` and ``{slot}_vy_mps`` for every track slot present in
  the frame table (implemented here, Task 6).

All computation is fully vectorised; no Python-level loops iterate over the
199,319 rows of the frame table.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regex that identifies player-position x columns in the frame table.
# Column names follow the pattern  track_<N>_x_m  (e.g. track_1_x_m).
# ---------------------------------------------------------------------------
_SLOT_X_RE = re.compile(r"^(track_\d+)_x_m$")

# Look-back window for player velocity (in frames).
# The actual ffill limit applied to the *shifted* series is one less than
# this value because the shift(1) already consumes one step of the window.
_VELOCITY_LOOKBACK: int = 5
_VELOCITY_FFILL_LIMIT: int = _VELOCITY_LOOKBACK - 1  # = 4


class KinematicsBuilder:
    """Compute ball and player kinematics from the raw frame table.

    Every method takes a :class:`pandas.DataFrame` and returns a **copy**
    with the new columns appended.  The original DataFrame is never mutated.
    """

    # ------------------------------------------------------------------
    # build_ball_kinematics is implemented in Task 5 and will be merged
    # into this class.  A placeholder is provided here so the class is
    # importable from either partial implementation during parallel
    # development.
    # ------------------------------------------------------------------

    def build_ball_kinematics(self, df: pd.DataFrame, fps: float) -> pd.DataFrame:
        """Append ball-kinematics columns to a copy of *df*.

        The four columns added are:

        * **ball_gap** (int64) — length of the consecutive invisible-ball run
          ending at each row; 0 wherever ``ball_visible`` is ``True``.
        * **ball_speed_mps** (float64) — 3-frame centred rolling mean of the
          per-frame Euclidean speed (m/s).  ``NaN`` when ``ball_visible`` is
          ``False`` *or* when no consecutive pair exists; smoothed to reduce
          noise-triggered events (``min_periods=1``).
        * **ball_speed_mps_raw** (float64) — unsmoothed per-frame speed used
          for diagnostics.  Same NaN pattern as the pre-smoothing speed.
        * **ball_direction_rad** (float64) — ``atan2(dy, dx)`` between the
          same consecutive pair.  ``NaN`` under the same conditions as the
          raw speed.  Not smoothed.
        * **ball_accel_mpss** (float64) — change in raw per-frame
          ``ball_speed_mps_raw`` per second between consecutive speed-valid
          frames.  ``NaN`` when either neighbour is ``NaN``.

        All computation is fully vectorised — no Python loop over rows.

        Parameters
        ----------
        df:
            Frame table as produced by
            :meth:`~src.load_data.TrackingLoader.load_frame_table`.
            Must contain columns ``ball_x_m``, ``ball_y_m``, and
            ``ball_visible``.
        fps:
            Frames per second of the tactical video feed (typically 25.0).

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the four kinematics columns appended.
            All other columns are unchanged.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")

        out = df.copy()
        dt: float = 1.0 / fps

        visible: pd.Series = out["ball_visible"].astype(bool)

        # ------------------------------------------------------------------
        # ball_gap
        # ------------------------------------------------------------------
        # Vectorised run-length encoding of invisible streaks.
        #
        # Strategy:
        #   Every time ball_visible transitions to True, visible.cumsum()
        #   increments — giving each contiguous invisible run a unique group
        #   id.  Within each group, cumsum of the invisible indicator counts
        #   how many consecutive invisible frames have accumulated.
        invisible: pd.Series = (~visible).astype(np.int8)
        group_key: pd.Series = visible.cumsum()
        ball_gap: pd.Series = (
            invisible.groupby(group_key).cumsum().astype(np.int64)
        )
        out["ball_gap"] = ball_gap

        # ------------------------------------------------------------------
        # Consecutive-pair mask
        # ------------------------------------------------------------------
        # Speed and direction are only valid when both this frame AND the
        # immediately preceding frame are visible (strict consecutive pair).
        consec: pd.Series = visible & visible.shift(1, fill_value=False)

        # ------------------------------------------------------------------
        # ball_speed_mps  and  ball_direction_rad
        # ------------------------------------------------------------------
        bx: pd.Series = out["ball_x_m"]
        by: pd.Series = out["ball_y_m"]

        dx: pd.Series = bx - bx.shift(1)
        dy: pd.Series = by - by.shift(1)

        # Speed: √(dx²+dy²) / dt — valid only on consecutive pairs.
        speed_raw: pd.Series = np.sqrt(dx**2 + dy**2) / dt
        ball_speed = pd.Series(np.nan, index=out.index, dtype=np.float64)
        ball_speed[consec] = speed_raw[consec]
        out["ball_speed_mps"] = ball_speed

        # Preserve raw speed for diagnostics, then apply a 3-frame centred rolling mean
        # to reduce noise-triggered events. NaN-aware: min_periods=1 keeps valid frames valid.
        out["ball_speed_mps_raw"] = out["ball_speed_mps"].copy()
        out["ball_speed_mps"] = (
            out["ball_speed_mps"]
            .rolling(window=3, center=True, min_periods=1)
            .mean()
        )

        # Direction: atan2(dy, dx) — same mask.
        dir_raw: pd.Series = np.arctan2(dy, dx)
        ball_dir = pd.Series(np.nan, index=out.index, dtype=np.float64)
        ball_dir[consec] = dir_raw[consec]
        out["ball_direction_rad"] = ball_dir

        # ------------------------------------------------------------------
        # ball_accel_mpss
        # ------------------------------------------------------------------
        # Acceleration between consecutive speed-valid frames only.
        # Use the raw (unsmoothed) speed so the NaN pattern is preserved:
        # accel is only valid where two consecutive raw-speed frames exist.
        speed_valid: pd.Series = out["ball_speed_mps_raw"].notna()
        consec_speed: pd.Series = speed_valid & speed_valid.shift(1, fill_value=False)

        d_speed: pd.Series = out["ball_speed_mps_raw"] - out["ball_speed_mps_raw"].shift(1)
        accel = pd.Series(np.nan, index=out.index, dtype=np.float64)
        accel[consec_speed] = d_speed[consec_speed] / dt
        out["ball_accel_mpss"] = accel

        return out

    # ------------------------------------------------------------------
    # Player velocity  (Task 6)
    # ------------------------------------------------------------------

    def build_player_velocities(
        self, df: pd.DataFrame, fps: float
    ) -> pd.DataFrame:
        """Add per-player velocity columns to the frame table.

        For every track slot ``s`` found in *df* (detected by scanning for
        columns matching ``{s}_x_m``), two new columns are appended:

        * ``{s}_vx_mps``  — x-component of velocity in m/s.
        * ``{s}_vy_mps``  — y-component of velocity in m/s.

        Velocity at frame *i* is computed as::

            vx[i] = (x[i] - x[j]) / ((i - j) * dt)

        where *j* is the most recent prior row within the last
        :data:`_VELOCITY_LOOKBACK` (= 5) frames where ``{s}_x_m`` is
        non-NaN, and ``dt = 1 / fps``.

        Frames where either the current slot position is NaN *or* no valid
        prior position exists within 5 frames receive velocity ``(0.0, 0.0)``.
        Row 0 always receives ``(0.0, 0.0)`` for every slot.

        The implementation is fully vectorised: ``ffill(limit=5)`` propagates
        the last valid value (and its index) forward by at most 5 steps,
        producing the correct "most recent prior non-NaN" look-back without
        a Python loop over rows.

        Parameters
        ----------
        df:
            Frame table as returned by :class:`src.load_data.TrackingLoader`.
            Must be indexed 0..N-1 with integer index.
        fps:
            Tracking frame rate in frames per second (typically 25).

        Returns
        -------
        pd.DataFrame
            A copy of *df* with ``{slot}_vx_mps`` and ``{slot}_vy_mps``
            columns appended for every track slot present in *df*.  All
            existing columns are preserved unchanged.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")

        dt: float = 1.0 / fps

        # Work on a copy so the caller's DataFrame is never mutated.
        out = df.copy()

        # Integer frame indices for vectorised gap computation.
        frame_idx = pd.Series(out.index, index=out.index, dtype=np.float64)

        # Identify every track slot by scanning for *_x_m columns.
        slots: list[str] = [
            m.group(1)
            for col in out.columns
            if (m := _SLOT_X_RE.match(col)) is not None
        ]

        for slot in slots:
            x_col = out[f"{slot}_x_m"]
            y_col = out[f"{slot}_y_m"]

            valid_mask: pd.Series = x_col.notna()  # bool: current frame is valid

            # ----------------------------------------------------------
            # For each row i, find the most recent prior row j (within 5
            # frames) where the slot position is non-NaN.
            #
            # Strategy:
            #   1. Keep only valid values (mask out NaN → NaN).
            #   2. Forward-fill with limit=5.  After the fill:
            #      - filled_x[i] is x[j], the last valid x value seen
            #        within the last 5 rows.
            #      - filled_idx[i] is j, the integer row index of that
            #        last valid observation.
            #   3. Velocity = (x[i] - filled_x[i]) / ((i - j) * dt)
            #      This correctly handles variable gaps (j may be 1..5
            #      steps before i).
            #
            # We use shift(1) before ffill so that the "prior" value is
            # strictly before the current frame (row i−1 or earlier), not
            # row i itself.  This prevents trivially zero velocity when the
            # slot was also valid one frame prior.
            # ----------------------------------------------------------

            # Prior valid x/y: shift by 1 so row i can only see rows < i.
            prior_x_shifted: pd.Series = x_col.where(valid_mask).shift(1)
            prior_y_shifted: pd.Series = y_col.where(valid_mask).shift(1)
            prior_idx_shifted: pd.Series = frame_idx.where(valid_mask).shift(1)

            # Forward-fill to reach back to the most recent valid prior frame
            # (original gap of 1..5 frames).
            #
            # Because prior_*_shifted was already shifted by 1 (consuming one
            # step of the look-back window), we fill with limit=4 so that the
            # total reach from frame i is at most 1 + 4 = 5 original frames.
            prior_x_ffill: pd.Series = prior_x_shifted.ffill(limit=_VELOCITY_FFILL_LIMIT)
            prior_y_ffill: pd.Series = prior_y_shifted.ffill(limit=_VELOCITY_FFILL_LIMIT)
            prior_idx_ffill: pd.Series = prior_idx_shifted.ffill(limit=_VELOCITY_FFILL_LIMIT)

            # Frame gap from j to i (integer number of frames).
            gap_frames: pd.Series = (frame_idx - prior_idx_ffill).clip(lower=1)

            # Raw velocity (NaN where either current or prior is NaN).
            dx: pd.Series = x_col - prior_x_ffill
            dy: pd.Series = y_col - prior_y_ffill

            vx_raw: pd.Series = dx / (gap_frames * dt)
            vy_raw: pd.Series = dy / (gap_frames * dt)

            # Zero out:
            #   - frames where the current position is NaN (slot absent).
            #   - frames where no valid prior exists within 5 frames
            #     (prior_idx_ffill is NaN, meaning ffill found nothing).
            #   - row 0 always (no prior possible).
            has_prior: pd.Series = prior_idx_ffill.notna() & (frame_idx > 0)
            current_valid: pd.Series = valid_mask

            keep: pd.Series = has_prior & current_valid

            out[f"{slot}_vx_mps"] = vx_raw.where(keep, other=0.0)
            out[f"{slot}_vy_mps"] = vy_raw.where(keep, other=0.0)

        return out


# ---------------------------------------------------------------------------
# Nearest-player-to-ball team assignment  (vectorised, used by AnchoredVoteAssigner)
# ---------------------------------------------------------------------------


def nearest_player_team(df: pd.DataFrame) -> np.ndarray:
    """Return the team of the player nearest the ball for every frame.

    This is a fully vectorised implementation of the baseline's inner loop.
    For every frame where the ball is visible and at least one player is
    tracked, it finds the track slot with the smallest Euclidean distance to
    the ball in metres and returns that slot's team label.

    Parameters
    ----------
    df:
        Frame table that already contains ``ball_x_m``, ``ball_y_m``,
        ``ball_visible``, and the player position/team columns produced by
        :meth:`TrackingLoader.load_frame_table`.  Index must be 0-based
        integer (0..N-1).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(N,)`` with dtype ``int8``:

        * ``1``  — home team player is nearest
        * ``-1`` — away team player is nearest
        * ``0``  — ball not visible or no players tracked in this frame

    Notes
    -----
    Slots whose ``{slot}_team`` is ``"unknown"`` are excluded from
    contention so they never influence the result.  If only unknown-team
    slots are present the frame receives ``0``.
    """
    n = len(df)
    out = np.zeros(n, dtype=np.int8)

    ball_visible = df["ball_visible"].to_numpy(dtype=bool)
    ball_x = df["ball_x_m"].to_numpy(dtype=np.float64)
    ball_y = df["ball_y_m"].to_numpy(dtype=np.float64)

    # Collect all player x/y/team column triplets.
    slot_re = re.compile(r"^(track_\d+)_x_m$")
    slots: list[str] = [
        m.group(1)
        for col in df.columns
        if (m := slot_re.match(col)) is not None
    ]

    if not slots:
        return out

    # Stack player positions: shape (N, n_slots)
    xs = np.column_stack([df[f"{s}_x_m"].to_numpy(dtype=np.float64) for s in slots])
    ys = np.column_stack([df[f"{s}_y_m"].to_numpy(dtype=np.float64) for s in slots])

    # Team codes: +1 home, -1 away, 0 unknown — shape (n_slots,)
    team_codes = np.array(
        [
            1 if str(df[f"{s}_team"].iloc[0]) == "home"
            else (-1 if str(df[f"{s}_team"].iloc[0]) == "away" else 0)
            for s in slots
        ],
        dtype=np.int8,
    )

    # Team is stored per-frame in the frame table (slot can change identity),
    # so we need to read team per row, not just from row 0.
    # Build a full (N, n_slots) team matrix.
    team_mat = np.zeros((n, len(slots)), dtype=np.int8)
    for j, s in enumerate(slots):
        col = df[f"{s}_team"].to_numpy(dtype=object)
        team_mat[:, j] = np.where(col == "home", np.int8(1),
                         np.where(col == "away", np.int8(-1), np.int8(0)))

    # Squared distances — NaN where player is invisible.
    dx = xs - ball_x[:, np.newaxis]
    dy = ys - ball_y[:, np.newaxis]
    dist2 = dx * dx + dy * dy  # (N, n_slots), NaN where player absent

    # Mask out unknown-team slots and frames where the ball is invisible.
    # Also mask rows where team_mat is 0 (unknown).
    unknown_mask = (team_mat == 0)
    dist2[unknown_mask] = np.nan

    # For invisible-ball frames, all distances are irrelevant — will be zeroed at the end.
    dist2[~ball_visible] = np.nan

    # argmin ignoring NaN: use nanargmin row-wise; where all are NaN keep 0.
    with np.errstate(all="ignore"):
        # Replace NaN with inf for argmin, then check if any finite value exists.
        dist2_inf = np.where(np.isnan(dist2), np.inf, dist2)
        has_visible_player = np.any(np.isfinite(dist2_inf), axis=1)  # (N,)

        nearest_j = np.argmin(dist2_inf, axis=1)  # (N,) index into slots

    # Gather the team code for the nearest slot at each row.
    row_idx = np.arange(n)
    nearest_team = team_mat[row_idx, nearest_j]  # (N,)

    # Zero out frames with no visible player or no visible ball.
    out = np.where(has_visible_player & ball_visible, nearest_team, np.int8(0))
    return out.astype(np.int8)


def nearest_team_margin(df: pd.DataFrame) -> np.ndarray:
    """Metres between the nearest away and the nearest home player to the ball.

    Positive: the nearest home player is closer. Negative: the nearest away player is.
    NaN where the ball is invisible or no player of either team is visible.
    """
    n = len(df)
    ball_visible = df["ball_visible"].to_numpy(dtype=bool)
    ball_x = df["ball_x_m"].to_numpy(dtype=np.float64)
    ball_y = df["ball_y_m"].to_numpy(dtype=np.float64)
    slots = [m.group(1) for col in df.columns if (m := _SLOT_X_RE.match(col)) is not None]
    xs = np.column_stack([df[f"{s}_x_m"].to_numpy(dtype=np.float64) for s in slots])
    ys = np.column_stack([df[f"{s}_y_m"].to_numpy(dtype=np.float64) for s in slots])
    team_mat = np.zeros((n, len(slots)), dtype=np.int8)
    for j, s in enumerate(slots):
        col = df[f"{s}_team"].to_numpy(dtype=object)
        team_mat[:, j] = np.where(col == "home", 1, np.where(col == "away", -1, 0))
    with np.errstate(invalid="ignore"):
        dist = np.hypot(xs - ball_x[:, None], ys - ball_y[:, None])
    dist[~ball_visible] = np.nan
    usable = np.isfinite(dist)
    home = np.where(usable & (team_mat == 1), dist, np.inf).min(axis=1)
    away = np.where(usable & (team_mat == -1), dist, np.inf).min(axis=1)
    with np.errstate(invalid="ignore"):
        margin = away - home
    margin[np.isinf(home) & np.isinf(away)] = np.nan
    margin[~ball_visible] = np.nan
    return margin


def reach_gated_team(df, near_team, lag=5, gate=10.0, horizon=1.0, fps=25.0):
    """Team of the player nearest the ball's extrapolated position, for fast-moving balls only.

    Below `gate` m/s, or where the ball or its position `lag` frames earlier is missing,
    the nearest-player call is returned unchanged.
    """
    n = len(df)
    visible = df["ball_visible"].to_numpy(dtype=bool)
    bx = df["ball_x_m"].to_numpy(dtype=np.float64)
    by = df["ball_y_m"].to_numpy(dtype=np.float64)
    slots = [m.group(1) for col in df.columns if (m := _SLOT_X_RE.match(col)) is not None]
    xs = np.column_stack([df[f"{s}_x_m"].to_numpy(dtype=np.float64) for s in slots])
    ys = np.column_stack([df[f"{s}_y_m"].to_numpy(dtype=np.float64) for s in slots])
    team_mat = np.zeros((n, len(slots)), dtype=np.int8)
    for j, s in enumerate(slots):
        col = df[f"{s}_team"].to_numpy(dtype=object)
        team_mat[:, j] = np.where(col == "home", 1, np.where(col == "away", -1, 0))

    prev_ok = np.zeros(n, dtype=bool)
    prev_ok[lag:] = visible[lag:] & visible[:-lag]
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[lag:] = (bx[lag:] - bx[:-lag]) * fps / lag
    vy[lag:] = (by[lag:] - by[:-lag]) * fps / lag
    with np.errstate(invalid="ignore"):
        use = prev_ok & (np.hypot(vx, vy) >= gate)
    out = near_team.copy()
    idx = np.flatnonzero(use)
    if len(idx) == 0:
        return out
    px = bx[idx] + vx[idx] * horizon
    py = by[idx] + vy[idx] * horizon
    with np.errstate(invalid="ignore"):
        d = np.hypot(xs[idx] - px[:, None], ys[idx] - py[:, None])
    tm = team_mat[idx]
    d[~np.isfinite(d)] = np.inf
    home = np.where(tm == 1, d, np.inf).min(axis=1)
    away = np.where(tm == -1, d, np.inf).min(axis=1)
    both = np.isfinite(home) & np.isfinite(away)
    call = np.where(home <= away, 1, -1).astype(np.int8)
    out[idx[both]] = call[both]
    return out
