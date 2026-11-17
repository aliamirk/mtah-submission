"""evaluate.py — Local evaluation against the public ground-truth window.

This module wraps ``evaluation/possession_metrics.py`` to provide a
convenient :class:`LocalEvaluator` that scores a submission against the
public GT window (frames 11185..18684), prints a human-readable report,
and writes a per-frame diagnostic CSV.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository root — needed to locate evaluation/possession_metrics.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent       # .../src/
_ROOT = _HERE.parent                          # repository root

# Public GT window (inclusive on both ends).
_GT_START: int = 11185
_GT_END: int = 18684
_N_FRAMES: int = 199319

# Label ordering used by possession_metrics (H=0, A=1, D=2, X=3).
_LABELS = ("H", "A", "D", "X")
_SCORED = ("H", "A", "D")


def _import_possession_metrics() -> Any:
    """Import ``possession_metrics`` from ``evaluation/`` at runtime.

    We add the ``evaluation/`` directory to ``sys.path`` rather than using a
    static import so this module can be imported anywhere in the repository
    without requiring the evaluation package to be installed.

    Returns
    -------
    module
        The ``possession_metrics`` module object.
    """
    eval_dir = str(_ROOT / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import possession_metrics  # noqa: PLC0415
    return possession_metrics


class LocalEvaluator:
    """Evaluate a label array against the public ground-truth window.

    All three methods are stateless; the class is provided as a namespace
    consistent with the rest of the pipeline.

    Public GT window: frames 11185..18684 inclusive.  Labels outside this
    range are kept in the arrays passed to ``possession_metrics.evaluate``
    so that its internal ``valid_runs()`` logic can derive the correct
    scored windows from the GT itself.
    """

    # ------------------------------------------------------------------
    # run_evaluation
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        labels: np.ndarray,
        gt: pd.Series,
        cfg: dict,
    ) -> dict:
        """Compute all metrics for a submission against the public GT.

        The full 199,319-element arrays are passed to
        ``possession_metrics.evaluate()``; that function's ``valid_runs()``
        call extracts the graded windows from the GT internally, so no
        frame-range slicing is needed here.

        The returned dict is the standard output of
        ``possession_metrics.evaluate()`` augmented with a ``"stats"`` key
        that holds ``possession_metrics.possession_stats()`` computed over
        the scored window.

        Parameters
        ----------
        labels:
            Prediction array of shape ``(199319,)`` with dtype ``"<U1"`` or
            equivalent; values in ``{"H", "A", "D", "X"}``.
        gt:
            Ground-truth ``pd.Series`` indexed ``0..199318``, values in
            ``{"H", "A", "D", "X"}``.  Frames outside the public window are
            ``"X"``.
        cfg:
            Pipeline configuration dict (not used for metric computation but
            accepted for interface consistency).

        Returns
        -------
        dict
            ``possession_metrics.evaluate()`` output augmented with::

                {
                    "stats": {
                        "n_spells": int,
                        "mean_duration_s": float,
                        "home_share": float,
                    }
                }

        Raises
        ------
        ValueError
            If ``labels`` does not have exactly 199,319 elements or contains
            values outside ``{"H", "A", "D", "X"}``.
        """
        pm = _import_possession_metrics()

        if len(labels) != _N_FRAMES:
            raise ValueError(
                f"labels must have {_N_FRAMES} elements, got {len(labels)}."
            )
        if len(gt) != _N_FRAMES:
            raise ValueError(
                f"gt must have {_N_FRAMES} elements, got {len(gt)}."
            )

        # Convert string arrays to integer codes expected by possession_metrics.
        gt_array = np.array(gt.values, dtype=object)
        pred_array = np.array([str(v).strip().upper() for v in labels], dtype=object)

        gt_codes: np.ndarray = pm.label_codes(gt_array)
        pred_codes: np.ndarray = pm.label_codes(pred_array)

        # Full evaluation — possession_metrics.evaluate() handles windowing.
        metrics: dict = pm.evaluate(gt_codes, pred_codes)

        # Augment with possession_stats over the scored window.
        runs = pm.valid_runs(gt_codes)
        filled_pred = pm.fill_abstentions(pred_codes, runs)
        stats = pm.possession_stats(filled_pred, runs)
        metrics["stats"] = stats

        return metrics

    # ------------------------------------------------------------------
    # print_report
    # ------------------------------------------------------------------

    def print_report(self, metrics: dict) -> None:
        """Print a human-readable evaluation report to stdout.

        Printed sections
        ----------------
        1. 4×4 confusion matrix — rows = GT label, columns = predicted label.
        2. Per-class precision, recall and F1 for H, A and D.
        3. Macro-F1 and spell_f1_mean.
        4. Predicted possession spell count and mean duration (seconds).

        Parameters
        ----------
        metrics:
            Dict returned by :meth:`run_evaluation`.
        """
        # ------------------------------------------------------------------
        # 1. Confusion matrix
        # ------------------------------------------------------------------
        cm_list = metrics.get("confusion", [])
        cm = np.array(cm_list, dtype=np.int64)  # shape (4, 4)

        col_w = 8  # column width for numbers
        lbl_w = 6  # row-label width

        print()
        print("=" * 50)
        print("EVALUATION REPORT")
        print("=" * 50)
        print()
        print("Confusion matrix (rows=GT, cols=predicted):")
        header = " " * lbl_w + "".join(f"{lbl:>{col_w}}" for lbl in _LABELS)
        print(header)
        print(" " * lbl_w + "-" * (col_w * len(_LABELS)))
        for i, row_lbl in enumerate(_LABELS):
            row_str = f"{row_lbl:<{lbl_w}}" + "".join(
                f"{int(cm[i, j]):>{col_w}}" for j in range(len(_LABELS))
            )
            print(row_str)
        print()

        # Graded-frame summary (GT-X rows are excluded from scoring).
        n_graded = metrics.get("n_graded", 0)
        n_abstain = metrics.get("n_abstain_on_graded", 0)
        abstain_pct = 100.0 * n_abstain / n_graded if n_graded else 0.0
        print(
            f"Graded frames: {n_graded}  "
            f"Abstentions on graded: {n_abstain}  "
            f"({abstain_pct:.1f}%)"
        )
        print()

        # ------------------------------------------------------------------
        # 2. Per-class precision, recall, F1
        # ------------------------------------------------------------------
        per_class = metrics.get("per_class", {})
        print(f"{'Class':<8}{'Precision':>12}{'Recall':>12}{'F1':>10}{'Support':>10}")
        print("-" * 52)
        for cls in _SCORED:
            info = per_class.get(cls, {})
            p = info.get("precision", 0.0)
            r = info.get("recall", 0.0)
            f = info.get("f1", 0.0)
            sup = info.get("support", 0)
            print(f"{cls:<8}{p:>12.4f}{r:>12.4f}{f:>10.4f}{sup:>10d}")
        print()

        # ------------------------------------------------------------------
        # 3. Summary metrics
        # ------------------------------------------------------------------
        macro = metrics.get("macro_f1", 0.0)
        spell_f1 = metrics.get("spell_f1_mean", 0.0)
        alive_dead = metrics.get("alive_dead_f1", 0.0)
        accuracy = metrics.get("accuracy", 0.0)
        kappa = metrics.get("kappa", 0.0)

        print(f"macro_f1          : {macro:.4f}")
        print(f"spell_f1_mean     : {spell_f1:.4f}")
        print(f"alive_dead_f1     : {alive_dead:.4f}")
        print(f"accuracy          : {accuracy:.4f}")
        print(f"cohen_kappa       : {kappa:.4f}")
        print()

        # ------------------------------------------------------------------
        # 4. Possession spell statistics
        # ------------------------------------------------------------------
        stats = metrics.get("stats", {})
        n_spells = stats.get("n_spells", 0)
        mean_dur = stats.get("mean_duration_s", 0.0)
        home_share = stats.get("home_share", 0.0)

        # Ground-truth counterparts are available directly from the metrics dict.
        n_gt_spells = metrics.get("n_gt_spells", 0)
        gt_mean_dur = metrics.get("gt_mean_spell_duration_s", 0.0)

        print("Possession spell statistics:")
        print(f"  predicted  : {n_spells:>6d} spells,  mean duration {mean_dur:.2f} s")
        print(f"  ground truth: {n_gt_spells:>6d} spells, mean duration {gt_mean_dur:.2f} s")
        print(f"  home share (predicted): {home_share:.3f}")
        print()

        # Predicted label distribution (sums over all GT classes).
        n_h_pred = int(cm[:, 0].sum())
        n_a_pred = int(cm[:, 1].sum())
        n_d_pred = int(cm[:, 2].sum())
        n_x_pred = int(cm[:, 3].sum())
        print(
            f"  predicted H frames: {n_h_pred}   "
            f"predicted A frames: {n_a_pred}   "
            f"predicted D frames: {n_d_pred}   "
            f"predicted X frames: {n_x_pred}"
        )
        print()
        print("=" * 50)
        print()

    # ------------------------------------------------------------------
    # write_diagnostic_csv
    # ------------------------------------------------------------------

    def write_diagnostic_csv(
        self,
        labels: np.ndarray,
        gt: pd.Series,
        out_path: str,
    ) -> None:
        """Write a per-frame diagnostic CSV for the public GT window.

        Columns: ``frame``, ``predicted``, ``ground_truth``.  The file covers
        frames 11185..18684 inclusive (7,500 rows).

        Parameters
        ----------
        labels:
            Prediction array of shape ``(199319,)``; values in
            ``{"H", "A", "D", "X"}``.
        gt:
            Ground-truth ``pd.Series`` indexed ``0..199318``.
        out_path:
            Destination file path.  Parent directories are created if they
            do not exist.

        Raises
        ------
        ValueError
            If ``labels`` does not have at least ``_GT_END + 1`` elements.
        """
        if len(labels) <= _GT_END:
            raise ValueError(
                f"labels must have at least {_GT_END + 1} elements "
                f"to cover the GT window 0..{_GT_END}, got {len(labels)}."
            )

        frames = list(range(_GT_START, _GT_END + 1))
        predicted = [str(labels[i]).strip().upper() for i in frames]
        ground_truth = [str(gt.iloc[i]) for i in frames]

        diag_df = pd.DataFrame(
            {
                "frame": frames,
                "predicted": predicted,
                "ground_truth": ground_truth,
            }
        )

        # Create parent directory if it does not exist.
        parent = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(parent, exist_ok=True)

        diag_df.to_csv(out_path, index=False)
