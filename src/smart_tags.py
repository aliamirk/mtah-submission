"""smart_tags.py — Smart Tagging XML loader and frame-index converter.

Loads the Metrica Smart Tagging XML file for the BAR-ATH match and converts
its interval timestamps (seconds from the first tactical-video frame) into
zero-based submission frame indices.

The :class:`TagProcessor` class produced here is consumed by
``src/possession.py`` to build a dead-ball mask and team-prior arrays.
Nothing in this module treats Smart Tags as ground truth; they are weak
priors only.

Notes
-----
- Uses ``xml.etree.ElementTree`` from the standard library only.
- All text comparisons are case-insensitive (``str.strip().lower()``).
- File paths are resolved via :func:`games.asset_paths`; no filenames are
  hardcoded.
"""

from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so `games` can be imported even
# when this module is imported directly (e.g. from tests).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import games  # noqa: E402 — must follow sys.path setup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant: total submission frames
# ---------------------------------------------------------------------------
_N_FRAMES: int = 199_319

# ---------------------------------------------------------------------------
# Dead-ball type strings used by build_dead_ball_mask.
# Comparisons are always lower-cased and stripped.
# ---------------------------------------------------------------------------
_DEAD_BALL_TYPES: frozenset[str] = frozenset(
    {
        "throw-in",
        "corner kick",
        "goal kick",
        "free kick",
        "kick off (after goal)",
        "kick off (start)",
    }
)


def _seconds_to_frame(seconds: float, fps: float, n_frames: int) -> int:
    """Convert a Smart Tag timestamp to a clamped frame index.

    Parameters
    ----------
    seconds:
        Seconds from the first frame of the tactical video (as stored in the
        XML ``<start>`` / ``<end>`` tags).
    fps:
        Frames per second (typically 25).
    n_frames:
        Total number of frames in the submission (199,319 for BAR-ATH).

    Returns
    -------
    int
        Frame index in ``[0, n_frames - 1]``.
    """
    raw: int = int(round(seconds * fps))
    return max(0, min(raw, n_frames - 1))


class TagProcessor:
    """Parse the Smart Tagging XML and build possession-pipeline arrays.

    All three public methods are stateless; no instance attributes are set
    after construction.  Instances can be reused across calls.

    Methods
    -------
    load_tags(cfg)
        Parse the XML and return a list of tag dicts.
    build_dead_ball_mask(tags, n_frames, cfg)
        Boolean array marking dead-ball and non-play frames.
    build_team_priors(tags, n_frames, cfg)
        Float64 arrays of home/away prior bonuses from Smart Tag intervals.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_tags(self, cfg: dict) -> list[dict[str, Any]]:
        """Parse the Smart Tagging XML and return all instances as tag dicts.

        Each tag dict has the following keys:

        ``code`` : str
            The ``<code>`` text of the instance (e.g. ``"SET PIECES"``).
        ``start_s`` : float
            Start time in seconds from the first tactical-video frame.
        ``end_s`` : float
            End time in seconds.
        ``start_frame`` : int
            ``round(start_s * fps)`` clamped to ``[0, n_frames - 1]``.
        ``end_frame`` : int
            ``round(end_s * fps)`` clamped to ``[0, n_frames - 1]``.
        ``tags`` : dict[str, str]
            Label group → text map, e.g.
            ``{"Team": "Barcelona", "Type": "Corner Kick"}``.
            If an instance has no ``<label>`` children the dict is empty.
            When a group appears more than once, the last occurrence wins.

        Parameters
        ----------
        cfg:
            Pipeline configuration dict.  Must contain ``"fps"``.

        Returns
        -------
        list[dict]
            Tag dicts in file order.  No filtering or sorting is applied.

        Raises
        ------
        IOError
            If the Smart Tagging XML file does not exist (propagated from
            :func:`games.require`).
        xml.etree.ElementTree.ParseError
            If the XML is malformed.  The file path is prepended to the
            message for easier diagnosis.
        """
        fps: float = float(cfg["fps"])
        n_frames: int = _N_FRAMES

        xml_path: str = games.asset_paths()["smart_tagging"]
        logger.info("Loading Smart Tags from %s", xml_path)

        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            raise ET.ParseError(
                f"Malformed Smart Tagging XML at {xml_path!r}: {exc}"
            ) from exc

        result: list[dict[str, Any]] = []

        for instance in tree.iter("instance"):
            # ------------------------------------------------------------------
            # Timestamps — findtext returns None if the element is absent;
            # we skip the instance in that case rather than crashing.
            # ------------------------------------------------------------------
            start_text = instance.findtext("start")
            end_text = instance.findtext("end")
            if start_text is None or end_text is None:
                logger.warning(
                    "Skipping Smart Tag instance with missing <start> or <end>: %s",
                    ET.tostring(instance, encoding="unicode")[:120],
                )
                continue

            start_s: float = float(start_text)
            end_s: float = float(end_text)

            # ------------------------------------------------------------------
            # Code — default to empty string when absent.
            # ------------------------------------------------------------------
            code: str = instance.findtext("code") or ""

            # ------------------------------------------------------------------
            # Label group → text pairs.
            # If a group appears more than once, the last value wins (matches
            # the starter's behaviour and the single-valued groups in this file).
            # ------------------------------------------------------------------
            tag_dict: dict[str, str] = {}
            for label_el in instance.findall("label"):
                group = label_el.findtext("group")
                text = label_el.findtext("text")
                if group is not None and text is not None:
                    tag_dict[group] = text

            # ------------------------------------------------------------------
            # Frame conversion with clamping.
            # ------------------------------------------------------------------
            start_frame: int = _seconds_to_frame(start_s, fps, n_frames)
            end_frame: int = _seconds_to_frame(end_s, fps, n_frames)

            result.append(
                {
                    "code": code,
                    "start_s": start_s,
                    "end_s": end_s,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "tags": tag_dict,
                }
            )

        logger.info("Loaded %d Smart Tag instances.", len(result))
        return result

    def build_dead_ball_mask(
        self,
        tags: list[dict[str, Any]],
        n_frames: int,
        cfg: dict,
    ) -> np.ndarray:
        """Build a boolean dead-ball mask for all submission frames.

        A frame is masked ``True`` when it falls inside a non-play region
        (pre-match, half-time, post-match) **or** within a buffered Smart Tag
        interval whose ``Type`` is a recognised dead-ball type.

        Parameters
        ----------
        tags:
            Tag dicts as returned by :meth:`load_tags`.
        n_frames:
            Total number of frames (199,319 for BAR-ATH).
        cfg:
            Pipeline configuration dict.  Must contain:

            * ``"nonplay_regions"`` — list of ``[first_frame, last_frame]``
              pairs (inclusive) for unconditional dead-ball regions.
            * ``"dead_ball_buffer_frames"`` — integer; frames padded either
              side of a Smart Tag dead-ball interval.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_frames,)``.  ``True`` where the frame
            is dead ball or non-play.
        """
        mask = np.zeros(n_frames, dtype=bool)

        # ------------------------------------------------------------------
        # Step 1: unconditional non-play regions from config.
        # ------------------------------------------------------------------
        for region in cfg["nonplay_regions"]:
            first, last = int(region[0]), int(region[1])
            mask[first : last + 1] = True

        # ------------------------------------------------------------------
        # Step 2: Smart Tag dead-ball intervals with buffer padding.
        # Each interval is capped at dead_ball_max_duration_frames frames
        # from the tag's own start_frame (before the buffer is applied).
        # This prevents 875-frame annotation-artefact windows from masking
        # live play that belongs to the next possession.
        # ------------------------------------------------------------------
        buffer: int = int(cfg["dead_ball_buffer_frames"])
        max_dur: int = int(cfg.get("dead_ball_max_duration_frames", 999_999))

        n_tag_intervals = 0
        for tag in tags:
            type_str: str = tag["tags"].get("Type", "").strip().lower()
            if type_str not in _DEAD_BALL_TYPES:
                continue
            n_tag_intervals += 1
            # Cap the raw end frame so the dead window cannot exceed max_dur frames.
            capped_end: int = min(
                tag["end_frame"],
                tag["start_frame"] + max_dur - 1,
            )
            lo = max(0, tag["start_frame"] - buffer)
            hi = min(n_frames - 1, capped_end + buffer)
            mask[lo : hi + 1] = True

        logger.info(
            "Dead-ball mask: nonplay_region frames=%d, tag_intervals=%d, "
            "total_masked=%d (%.1f%% of match).",
            int(np.sum(mask[:n_frames])),
            n_tag_intervals,
            int(mask.sum()),
            100.0 * mask.mean(),
        )
        logger.debug(
            "Dead-ball mask: %d / %d frames marked (%.1f%%).",
            int(mask.sum()),
            n_frames,
            100.0 * mask.mean(),
        )
        return mask

    def build_team_priors(
        self,
        tags: list[dict[str, Any]],
        n_frames: int,
        cfg: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build home and away team-prior bonus arrays from Smart Tag intervals.

        For each tag whose ``Team`` label matches a known team name, the
        configured weight is added to the corresponding prior array over the
        tag's entire interval.  Priors are additive; overlapping intervals of
        the same team simply accumulate.

        The ``ControlScorer`` normalises via
        ``p_home = home_sum / (home_sum + away_sum + epsilon)``, so unbounded
        prior values cannot produce out-of-range ``p_home`` results.

        Parameters
        ----------
        tags:
            Tag dicts as returned by :meth:`load_tags`.
        n_frames:
            Total number of frames (199,319 for BAR-ATH).
        cfg:
            Pipeline configuration dict.  Must contain ``"tag_prior_weight"``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(home_prior, away_prior)`` — each a float64 array of shape
            ``(n_frames,)``.  Values are ``0.0`` where no Smart Tag covers
            the frame.
        """
        home_prior = np.zeros(n_frames, dtype=np.float64)
        away_prior = np.zeros(n_frames, dtype=np.float64)
        weight: float = float(cfg["tag_prior_weight"])

        for tag in tags:
            team: str = tag["tags"].get("Team", "").strip().lower()
            lo: int = tag["start_frame"]
            hi: int = min(tag["end_frame"], n_frames - 1)

            if team == "barcelona":
                home_prior[lo : hi + 1] += weight
            elif team in ("atletic club", "athletic club"):
                away_prior[lo : hi + 1] += weight
            # "n/a" and unrecognised values are silently ignored.

        logger.debug(
            "Team priors built: home max=%.3f  away max=%.3f.",
            float(home_prior.max()) if home_prior.max() > 0 else 0.0,
            float(away_prior.max()) if away_prior.max() > 0 else 0.0,
        )
        return home_prior, away_prior

    def build_restart_team_array(
        self,
        tags: list[dict[str, Any]],
        n_frames: int,
        cfg: dict,
    ) -> np.ndarray:
        """Build a per-frame restart-team array from all team-labelled Smart Tags.

        Unlike :meth:`build_team_priors`, this method returns a *hard* team
        signal derived from every tag whose ``Team`` label resolves to a known
        side (Barcelona = home, Atletic/Athletic Club = away).  It is used by
        :class:`~src.possession.AnchoredVoteAssigner` to override the rolling
        nearest-player vote at restart frames.

        The signal is built from **all** tag types that carry a named team,
        including:

        * ``SET PIECES`` — Corner Kick, Goal Kick, Free Kick (team = restarter)
        * ``ATTACKING TRANSITION`` — team currently in an attacking transition
        * ``BALL IN FINAL THIRD`` — team whose ball is in the final third
        * ``PROGRESSION`` — team progressing the ball
        * Any other tag whose ``Team`` is ``"Barcelona"`` or
          ``"Atletic Club"`` / ``"Athletic Club"``

        Kick-off tags (Team = N/A) and throw-ins (also N/A in this dataset)
        are explicitly excluded because the restarting team is not encoded.

        Where multiple overlapping tags conflict, the **last tag in file
        order** wins for each frame (later tags are assumed to be more recent
        annotations of the same moment).

        Parameters
        ----------
        tags:
            Tag dicts as returned by :meth:`load_tags`.
        n_frames:
            Total number of submission frames (199,319).
        cfg:
            Pipeline configuration dict.  Not currently used but kept for
            interface consistency.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_frames,)`` with dtype ``int8``:

            * ``1``  — home team (Barcelona) signal at this frame
            * ``-1`` — away team (Atletic Club) signal at this frame
            * ``0``  — no named-team tag covers this frame
        """
        restart = np.zeros(n_frames, dtype=np.int8)

        for tag in tags:
            team: str = tag["tags"].get("Team", "").strip().lower()
            lo: int = tag["start_frame"]
            hi: int = min(tag["end_frame"], n_frames - 1)
            if lo > hi:
                continue

            if team == "barcelona":
                restart[lo : hi + 1] = np.int8(1)
            elif team in ("atletic club", "athletic club"):
                restart[lo : hi + 1] = np.int8(-1)
            # "n/a", kick-offs, throw-ins, and unrecognised teams → skip

        n_home = int((restart == 1).sum())
        n_away = int((restart == -1).sum())
        logger.debug(
            "Restart team array: home=%d frames, away=%d frames, neutral=%d frames.",
            n_home, n_away, n_frames - n_home - n_away,
        )
        return restart

    def build_restart_prior(
        self,
        tags: list[dict[str, Any]],
        dead_mask: np.ndarray,
        n_frames: int,
        cfg: dict,
    ) -> np.ndarray:
        """Build a per-frame restart-team prior from named-team dead-ball tags.

        For each dead-ball Smart Tag whose ``Team`` is Barcelona or
        Atletic/Athletic Club, the prior signal (+1 home / -1 away) is
        applied for ``restart_prior_frames`` frames starting at the first
        live frame after the capped dead-ball interval ends.

        The capped D interval is computed identically to
        :meth:`build_dead_ball_mask`::

            capped_end    = min(tag.end_frame, tag.start_frame + max_dur - 1)
            dead_end      = capped_end + buffer
            restart_frame = dead_end + 1

        **Overlap resolution (deterministic):**
        When two tags produce overlapping restart-prior windows, the tag
        whose capped D interval ends *latest* wins.  If two tags share the
        same ``dead_end``, the tag with the *earlier* ``start_frame`` wins.
        This ensures the most recent (and most specific) restart signal
        dominates at the point of overlap.

        The prior is zeroed at every frame covered by ``dead_mask`` so it
        never applies during a dead-ball interval.

        Parameters
        ----------
        tags:
            Tag dicts as returned by :meth:`load_tags`.
        dead_mask:
            Boolean array of shape ``(n_frames,)`` — ``True`` where the frame
            is dead ball or non-play.
        n_frames:
            Total number of submission frames (199,319).
        cfg:
            Pipeline configuration dict.  Required keys:
            ``dead_ball_buffer_frames``, ``dead_ball_max_duration_frames``,
            ``restart_prior_frames``.

        Returns
        -------
        np.ndarray
            Int8 array of shape ``(n_frames,)``:
            ``+1`` for home (Barcelona) prior,
            ``-1`` for away (Atletic/Athletic Club) prior,
            ``0`` elsewhere.
        """
        prior = np.zeros(n_frames, dtype=np.int8)

        buffer: int = int(cfg["dead_ball_buffer_frames"])
        max_dur: int = int(cfg.get("dead_ball_max_duration_frames", 999_999))
        prior_frames: int = int(cfg.get("restart_prior_frames", 0))

        if prior_frames <= 0:
            return prior

        # ------------------------------------------------------------------
        # Step 1: Collect qualifying tags — dead-ball type AND named team.
        # ------------------------------------------------------------------
        candidates: list[dict[str, Any]] = []
        for tag in tags:
            type_str: str = tag["tags"].get("Type", "").strip().lower()
            if type_str not in _DEAD_BALL_TYPES:
                continue

            team: str = tag["tags"].get("Team", "").strip().lower()
            if team == "barcelona":
                signal = np.int8(1)
            elif team in ("atletic club", "athletic club"):
                signal = np.int8(-1)
            else:
                # N/A, throw-ins, kick-offs, unrecognised → no prior.
                continue

            # Compute capped D interval (identical to build_dead_ball_mask).
            capped_end: int = min(
                tag["end_frame"],
                tag["start_frame"] + max_dur - 1,
            )
            dead_end: int = capped_end + buffer
            restart_frame: int = dead_end + 1

            candidates.append(
                {
                    "start_frame": tag["start_frame"],
                    "dead_end": dead_end,
                    "restart_frame": restart_frame,
                    "signal": signal,
                }
            )

        if not candidates:
            return prior

        # ------------------------------------------------------------------
        # Step 2: Sort by deterministic tie-break.
        #   1. dead_end descending  — latest-ending D interval wins.
        #   2. start_frame ascending — earlier start wins on tie.
        # Tags processed first take priority; later tags only fill unassigned
        # frames.
        # ------------------------------------------------------------------
        candidates.sort(key=lambda c: (-c["dead_end"], c["start_frame"]))

        # ------------------------------------------------------------------
        # Step 3: Apply priors in priority order.
        # ------------------------------------------------------------------
        for c in candidates:
            lo: int = max(0, c["restart_frame"])
            hi: int = min(n_frames, c["restart_frame"] + prior_frames)
            if lo >= hi:
                continue
            # Only set where prior is still 0 (not yet claimed by a
            # higher-priority tag).  prior[lo:hi] is a view for basic slicing,
            # so boolean-index assignment propagates back to `prior`.
            view = prior[lo:hi]
            unassigned = view == 0
            view[unassigned] = c["signal"]

        # ------------------------------------------------------------------
        # Step 4: Zero out any prior frames inside the dead-ball mask so the
        # prior never applies during a D interval.
        # ------------------------------------------------------------------
        prior[dead_mask] = 0

        n_home = int((prior == 1).sum())
        n_away = int((prior == -1).sum())
        logger.info(
            "Restart prior: %d candidates, home=%d frames, away=%d frames, "
            "neutral=%d frames.",
            len(candidates),
            n_home,
            n_away,
            n_frames - n_home - n_away,
        )
        return prior
