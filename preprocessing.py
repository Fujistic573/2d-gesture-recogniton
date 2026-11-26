"""
preprocessing.py
~~~~~~~~~~~~~~~~
Normalizes raw 2-D point sequences (single or multi-stroke) into a fixed-length,
centred, aspect-ratio-preserving vector ready for classification.

Pipeline
--------
1. Resample  – linear interpolation to exactly N equally-spaced points
               (across all strokes, without interpolating pen-up jumps).
2. Centre    – subtract the centroid so the mean is (0, 0).
3. Scale     – uniform (aspect-ratio-preserving) scale into [-1, 1].
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Default number of resampled points
N_POINTS: int = 32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(
    points: Sequence[tuple[float, float]],
    n: int = N_POINTS,
) -> np.ndarray:
    """Return a normalised (2*n,) flat vector from a single raw stroke."""
    return normalize_strokes([points], n)


def normalize_to_array(
    points: Sequence[tuple[float, float]],
    n: int = N_POINTS,
) -> np.ndarray:
    """Return a normalised (n, 2) array from a single raw stroke."""
    return normalize(points, n).reshape(n, 2)


def normalize_strokes(
    strokes: Sequence[Sequence[tuple[float, float]]],
    n: int = N_POINTS,
) -> np.ndarray:
    """Return a normalised (2*n,) flat vector from one or more strokes.

    Each stroke is resampled independently in proportion to its arc length
    so pen-up jumps between strokes are never interpolated.
    """
    return normalize_strokes_to_array(strokes, n).flatten()


def normalize_strokes_to_array(
    strokes: Sequence[Sequence[tuple[float, float]]],
    n: int = N_POINTS,
) -> np.ndarray:
    """Return a normalised (n, 2) array from one or more strokes."""
    valid: list[np.ndarray] = []
    for stroke in strokes:
        pts = np.array(stroke, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and len(pts) >= 2:
            valid.append(pts)

    if not valid:
        raise ValueError("At least one stroke with 2+ (x, y) points is required.")

    lengths = np.array([_path_lengths(s)[-1] for s in valid], dtype=float)
    total_len = float(lengths.sum())

    if total_len < 1e-9:
        pts = np.tile(valid[0][0], (n, 1))
    else:
        counts = _allocate_counts(lengths, n)
        resampled = [_resample(s, int(c)) for s, c in zip(valid, counts)]
        pts = np.vstack(resampled)

    pts = _centre(pts)
    pts = _scale(pts)
    return pts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _allocate_counts(lengths: np.ndarray, total_n: int) -> np.ndarray:
    """Distribute *total_n* points proportionally across strokes (each >= 1)."""
    k = len(lengths)
    if k == 1:
        return np.array([total_n], dtype=int)

    raw = (lengths / lengths.sum()) * total_n
    counts = np.maximum(1, np.floor(raw).astype(int))

    # Adjust rounding remainder so sum(counts) == total_n.
    diff = total_n - int(counts.sum())
    if diff > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:diff]:
            counts[idx] += 1
    elif diff < 0:
        order = np.argsort(raw - counts)
        for idx in order:
            if diff == 0:
                break
            if counts[idx] > 1:
                take = min(int(counts[idx] - 1), -diff)
                counts[idx] -= take
                diff += take
    return counts


def _path_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-lengths along a polyline (shape (m,), starts at 0)."""
    deltas = np.diff(pts, axis=0)
    seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate *pts* to exactly *n* equally-spaced points."""
    if n <= 1:
        return pts[:1].copy()
    cum = _path_lengths(pts)
    total = cum[-1]

    if total < 1e-9:
        return np.tile(pts[0], (n, 1))

    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return np.column_stack([xs, ys])


def _centre(pts: np.ndarray) -> np.ndarray:
    """Subtract the centroid so the mean position is (0, 0)."""
    return pts - pts.mean(axis=0)


def _scale(pts: np.ndarray) -> np.ndarray:
    """Scale uniformly (aspect-ratio preserving) so all coords ∈ [-1, 1]."""
    max_abs = np.abs(pts).max()
    return pts / (max_abs + 1e-9)
