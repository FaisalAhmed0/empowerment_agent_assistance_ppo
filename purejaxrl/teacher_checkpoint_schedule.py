from typing import Sequence

import numpy as np


def split_teacher_checkpoint_budget(
    total: int, bounds: Sequence[float], weights: Sequence[float]
) -> np.ndarray:
    """Per-segment checkpoint counts (largest-remainder rounding, sums to ``total``)."""
    bounds = np.asarray(bounds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    assert len(weights) == len(bounds) - 1, (
        "TEACHER_CHECKPOINT_WEIGHTS needs one entry per segment "
        f"(got {len(weights)} weights for {len(bounds)} bounds)"
    )
    assert bounds[0] == 0.0 and bounds[-1] == 1.0, (
        f"TEACHER_CHECKPOINT_BOUNDS must start at 0.0 and end at 1.0, got {bounds.tolist()}"
    )
    assert np.all(np.diff(bounds) > 0), (
        f"TEACHER_CHECKPOINT_BOUNDS must be strictly increasing, got {bounds.tolist()}"
    )
    assert np.all(weights >= 0) and weights.sum() > 0, (
        f"TEACHER_CHECKPOINT_WEIGHTS must be non-negative with a positive sum, got {weights.tolist()}"
    )
    quotas = weights / weights.sum() * total
    counts = np.floor(quotas).astype(np.int64)
    leftover = int(total - counts.sum())
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:leftover]] += 1
    return counts


def compute_teacher_checkpoint_indices(
    num_updates: int,
    num_ckpts: int,
    bounds: Sequence[float],
    weights: Sequence[float],
) -> np.ndarray:
    """Strictly increasing update indices, evenly spaced within each weighted segment."""
    if num_ckpts <= 0 or num_updates <= 0:
        return np.array([], dtype=np.int64)
    count = min(num_ckpts, num_updates)
    counts = split_teacher_checkpoint_budget(count, bounds, weights)
    if count == 1:
        return np.array([num_updates - 1], dtype=np.int64)
    num_segments = len(counts)
    u = np.concatenate(
        [
            np.linspace(bounds[i], bounds[i + 1], counts[i], endpoint=i == num_segments - 1)
            for i in range(num_segments)
        ]
    )
    idx = np.round((num_updates - 1) * u).astype(np.int64)
    for i in range(1, count):
        idx[i] = max(idx[i], idx[i - 1] + 1)
    for i in range(count - 1, -1, -1):
        idx[i] = min(idx[i], num_updates - 1 - (count - 1 - i))
    return idx
