"""src/output.py — Submission writer and validator.

Writes the final submission.csv (exactly 199,319 rows with a ``frame`` and
``label`` column) and validates both the label alphabet and the row count
before and after writing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

EXPECTED_ROWS: int = 199_319
VALID_LABELS: frozenset[str] = frozenset({"H", "A", "D", "X"})
_EXPECTED_COLS: frozenset[str] = frozenset({"frame", "label"})


def write_submission(labels: np.ndarray, out_path: str) -> None:
    """Validate *labels* and write the submission CSV to *out_path*.

    Parameters
    ----------
    labels:
        1-D array of length 199,319; every element must be one of
        ``{"H", "A", "D", "X"}``.
    out_path:
        Destination file path.  Parent directories are created automatically
        if they do not yet exist.

    Raises
    ------
    ValueError
        * If ``len(labels) != 199319`` — reports actual vs expected count.
        * If any label value is not in ``{"H", "A", "D", "X"}`` — reports
          the offending values.
        * If the file written back to disk does not contain exactly 199,319
          data rows (guards against filesystem truncation).
    """
    # ------------------------------------------------------------------
    # Pre-write validation 1: row count
    # ------------------------------------------------------------------
    if len(labels) != EXPECTED_ROWS:
        raise ValueError(
            f"labels must have exactly {EXPECTED_ROWS} rows, "
            f"but got {len(labels)}."
        )

    # ------------------------------------------------------------------
    # Pre-write validation 2: label alphabet
    # ------------------------------------------------------------------
    unique = set(labels)
    bad = unique - VALID_LABELS
    if bad:
        raise ValueError(
            f"labels contains invalid values: {sorted(bad)}. "
            f"Only {sorted(VALID_LABELS)} are permitted."
        )

    # ------------------------------------------------------------------
    # Ensure parent directory exists
    # ------------------------------------------------------------------
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        {
            "frame": np.arange(EXPECTED_ROWS, dtype=np.int64),
            "label": labels,
        }
    )
    df.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Post-write validation: re-read and verify row count
    # ------------------------------------------------------------------
    written = pd.read_csv(out_path)
    actual_rows = len(written)
    if actual_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Post-write row count mismatch: expected {EXPECTED_ROWS} rows "
            f"in '{out_path}', but found {actual_rows}. "
            "The file may have been truncated during writing."
        )


def validate_submission_columns(path: str) -> None:
    """Read *path* and verify that the submission CSV is fully valid.

    Checks performed
    ----------------
    1. Exactly 2 columns named ``frame`` and ``label``.
    2. The ``frame`` column contains integers 0..199318 with no duplicates.
    3. Row count == 199,319.
    4. All ``label`` values are in ``{"H", "A", "D", "X"}``.

    Parameters
    ----------
    path:
        Path to the submission CSV file.

    Raises
    ------
    ValueError
        With a descriptive message if any check fails.
    """
    df = pd.read_csv(path)

    # Check 1: columns
    actual_cols = set(df.columns.str.strip().str.lower())
    if actual_cols != {"frame", "label"}:
        raise ValueError(
            f"Expected exactly columns 'frame' and 'label', "
            f"but found: {sorted(df.columns.tolist())}."
        )
    # Normalise column names to lowercase for the rest of the checks.
    df.columns = df.columns.str.strip().str.lower()

    # Check 3: row count
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, but found {len(df)}."
        )

    # Check 2: frame values are integers 0..199318 with no duplicates
    frames = df["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError(
            f"'frame' column must be integer dtype, got {frames.dtype}."
        )
    expected_frames = np.arange(EXPECTED_ROWS, dtype=np.int64)
    if not np.array_equal(frames, expected_frames):
        if len(frames) != len(np.unique(frames)):
            raise ValueError("'frame' column contains duplicate values.")
        raise ValueError(
            f"'frame' column must be exactly 0..{EXPECTED_ROWS - 1} in order."
        )

    # Check 4: label alphabet
    cleaned = set(str(v).strip().upper() for v in df["label"].values)
    bad = cleaned - VALID_LABELS
    if bad:
        raise ValueError(
            f"'label' column contains invalid values: {sorted(bad)}. "
            f"Only {sorted(VALID_LABELS)} are permitted."
        )
