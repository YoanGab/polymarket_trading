"""Canonical split definitions for train/val/test/holdout.

All split logic MUST use ``get_split`` from this module.
Do NOT define cutoffs elsewhere.
"""

from __future__ import annotations

TRAIN_CUTOFF = "2025-10-01"
VAL_CUTOFF = "2026-01-01"
HOLDOUT_CUTOFF = "2026-02-01"  # locked holdout, never evaluate


def get_split(resolution_ts: str | None) -> str:
    """Canonical split assignment by resolution timestamp.

    Args:
        resolution_ts: ISO-8601 resolution timestamp, or None if unresolved.

    Returns:
        One of "train", "val", "test", "holdout", or "unknown".
    """
    if resolution_ts is None:
        return "unknown"
    if resolution_ts < TRAIN_CUTOFF:
        return "train"
    if resolution_ts < VAL_CUTOFF:
        return "val"
    if resolution_ts < HOLDOUT_CUTOFF:
        return "test"
    return "holdout"  # NEVER evaluate on this
