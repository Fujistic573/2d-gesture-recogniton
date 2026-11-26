"""
knn_classifier.py
~~~~~~~~~~~~~~~~~
k-Nearest Neighbours classifier for 2-D gesture recognition.

Written from scratch using only *math* and *numpy* (no scikit-learn).
Distance metric: Euclidean distance in the flattened normalised feature space.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Sample:
    """One labelled gesture sample."""

    __slots__ = ("label", "vector")

    def __init__(self, label: str, vector: np.ndarray) -> None:
        self.label = label
        self.vector: np.ndarray = np.asarray(vector, dtype=float)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class KNNClassifier:
    """k-Nearest Neighbours gesture classifier.

    Parameters
    ----------
    k:
        Number of nearest neighbours to consider (odd values avoid ties).
    dataset_path:
        Path to the JSON file where samples are persisted.
    """

    def __init__(
        self,
        k: int = 3,
        dataset_path: str | Path = Path("data/dataset.json"),
    ) -> None:
        self.k = k
        self.dataset_path = Path(dataset_path)
        self._samples: list[Sample] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Total number of stored samples."""
        return len(self._samples)

    @property
    def labels(self) -> list[str]:
        """Sorted list of unique class labels."""
        return sorted({s.label for s in self._samples})

    def class_counts(self) -> dict[str, int]:
        """Return a mapping of ``{label: sample_count}`` sorted by label."""
        counts = Counter(s.label for s in self._samples)
        return dict(sorted(counts.items()))

    def get_prototype(self, label: str) -> Optional[np.ndarray]:
        """Return the first stored ``(N, 2)`` shape array for *label*."""
        for s in self._samples:
            if s.label == label:
                return s.vector.reshape(-1, 2)
        return None

    def add_sample(self, label: str, vector: np.ndarray) -> None:
        """Append one labelled sample and persist to disk.

        Parameters
        ----------
        label:
            Human-readable gesture name (e.g. ``"circle"``).
        vector:
            Flattened normalised feature vector (shape ``(2*N,)``).
        """
        self._samples.append(Sample(label, vector))
        self._save()

    def add_samples(self, samples: list[tuple[str, np.ndarray]]) -> None:
        """Append several labelled samples and persist them in one write."""
        if not samples:
            return
        self._samples.extend(Sample(label, vector) for label, vector in samples)
        self._save()

    def remove_label(self, label: str) -> int:
        """Remove all samples belonging to *label* and persist.

        Returns the number of removed samples.
        """
        before = len(self._samples)
        self._samples = [s for s in self._samples if s.label != label]
        removed = before - len(self._samples)
        if removed:
            self._save()
        return removed

    def predict(
        self, vector: np.ndarray
    ) -> Optional[tuple[str, float]]:
        """Predict the gesture class for *vector*.

        Parameters
        ----------
        vector:
            Flattened normalised feature vector (same dimensionality as
            the stored samples).

        Returns
        -------
        (label, confidence) or ``None`` when the dataset is empty.
            *confidence* ∈ (0, 1] is the fraction of the *k* nearest
            neighbours that voted for the winning class.
        """
        top = self.predict_top_k(vector, top_n=1)
        return top[0] if top else None

    def predict_top_k(
        self, vector: np.ndarray, top_n: int = 3
    ) -> list[tuple[str, float]]:
        """Return up to *top_n* ``(label, confidence)`` pairs ranked by k-NN votes.

        Uses symmetric Chamfer Euclidean distance between the 2-D point clouds
        so recognition is invariant to drawing direction, starting point, and
        multi-stroke order.
        """
        if not self._samples:
            return []

        q = np.asarray(vector, dtype=float).reshape(-1, 2)
        batch = np.stack([s.vector.reshape(-1, 2) for s in self._samples])
        dists = _chamfer_batch(q, batch)
        distances = sorted(
            ((float(d), s.label) for d, s in zip(dists, self._samples)),
            key=lambda t: t[0],
        )

        k_eff = min(self.k, len(distances))
        weights: dict[str, float] = {}
        total_weight = 0.0
        for dist, label in distances[:k_eff]:
            w = 1.0 / (dist + 1e-4)
            weights[label] = weights.get(label, 0.0) + w
            total_weight += w

        best_dist: dict[str, float] = {}
        for dist, label in distances:
            if label not in best_dist:
                best_dist[label] = dist

        ranked = sorted(
            best_dist.items(),
            key=lambda item: (-weights.get(item[0], 0.0), item[1]),
        )[:top_n]
        return [
            (label, weights.get(label, 0.0) / total_weight if total_weight > 0 else 0.0)
            for label, _ in ranked
        ]

    def clear(self) -> None:
        """Remove all samples and wipe the JSON file."""
        self._samples.clear()
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Populate *self._samples* from the JSON file (if it exists)."""
        if not self.dataset_path.exists():
            return
        try:
            with self.dataset_path.open("r", encoding="utf-8") as fh:
                raw: list[dict] = json.load(fh)
            self._samples = [
                Sample(entry["label"], np.array(entry["vector"], dtype=float))
                for entry in raw
            ]
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupt file – start fresh.
            self._samples = []

    def _save(self) -> None:
        """Serialise all samples to the JSON file."""
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"label": s.label, "vector": s.vector.tolist()}
            for s in self._samples
        ]
        temp_path = self.dataset_path.with_suffix(self.dataset_path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        temp_path.replace(self.dataset_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chamfer_batch(q: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Vectorized symmetric Chamfer Euclidean distance from *q* (N, 2) to *batch* (M, N, 2)."""
    # Bound temporary memory use when a large dataset is evaluated repeatedly.
    result = np.empty(len(batch), dtype=float)
    chunk_size = 256
    for start in range(0, len(batch), chunk_size):
        end = min(start + chunk_size, len(batch))
        diff = q[None, :, None, :] - batch[start:end, None, :, :]
        dmat = np.sqrt(np.sum(diff * diff, axis=-1))
        result[start:end] = 0.5 * (
            dmat.min(axis=2).mean(axis=1) + dmat.min(axis=1).mean(axis=1)
        )
    return result
