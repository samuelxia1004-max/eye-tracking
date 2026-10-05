"""ISC estimators and event-boundary contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CHANNEL_MAP = {
    "x": (0,),
    "y": (1,),
    "pupil": (2,),
    "gaze": (0, 1),
    "composite": (0, 1, 2),
}


@dataclass(frozen=True)
class ContrastResult:
    effect: float
    group_changes: dict[str, float]
    target_group: str
    control_mean: float


def _as_3d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError("data must have shape (participants, time[, channels])")
    return array


def _row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    am = a - np.nanmean(a, axis=1, keepdims=True)
    bm = b - np.nanmean(b, axis=1, keepdims=True)
    numerator = np.nansum(am * bm, axis=1)
    denominator = np.sqrt(np.nansum(am * am, axis=1) * np.nansum(bm * bm, axis=1))
    out = np.full(a.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return np.clip(out, -1.0, 1.0)


def pairwise_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Return a Pearson matrix across participant rows, with constant rows as NaN."""

    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("pairwise_correlation_matrix expects (participants, time)")
    centered = x - np.nanmean(x, axis=1, keepdims=True)
    denom = np.sqrt(np.nansum(centered * centered, axis=1))
    product = centered @ centered.T
    scale = np.outer(denom, denom)
    corr = np.full_like(product, np.nan, dtype=float)
    np.divide(product, scale, out=corr, where=scale > 0)
    np.fill_diagonal(corr, np.where(denom > 0, 1.0, np.nan))
    return np.clip(corr, -1.0, 1.0)


def _transform(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return values
    if transform == "fisher":
        return np.arctanh(np.clip(values, -0.999999, 0.999999))
    if transform == "legacy":
        return (1.0 + values) / 2.0
    raise ValueError(f"unknown transform: {transform}")


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    """Row means that remain NaN for empty rows without emitting warnings."""

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    counts = valid.sum(axis=1)
    totals = np.nansum(array, axis=1)
    out = np.full(array.shape[0], np.nan, dtype=float)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def compute_isc(
    data: np.ndarray,
    groups: Iterable[str] | None = None,
    *,
    estimator: str = "pairwise",
    reference: str = "within",
    channel: str = "composite",
    transform: str = "raw",
) -> np.ndarray:
    """Compute participant-level ISC for one time window.

    Parameters
    ----------
    data
        Array with shape ``(participants, time, channels)``. Channel order is
        X, Y, pupil when a named composite is used.
    groups
        Group labels. Required for ``reference='within'``.
    estimator
        ``'pairwise'`` averages correlations with eligible partners;
        ``'loo'`` correlates each participant with the eligible leave-one-out
        mean time series.
    reference
        ``'within'`` uses only participants in the same group; ``'all'`` uses
        all other participants.
    channel
        One of ``x``, ``y``, ``pupil``, ``gaze``, or ``composite``.
    transform
        ``raw`` for Pearson r, ``fisher`` for Fisher-z aggregation, or
        ``legacy`` for the original ``mean(corrcoef)=(1+r)/2`` scale.
    """

    array = _as_3d(data)
    n = array.shape[0]
    if channel not in CHANNEL_MAP:
        raise ValueError(f"unknown channel definition: {channel}")
    channel_indices = tuple(i for i in CHANNEL_MAP[channel] if i < array.shape[2])
    if not channel_indices:
        raise ValueError("requested channel is absent")
    if reference not in {"within", "all"}:
        raise ValueError("reference must be 'within' or 'all'")
    if estimator not in {"pairwise", "loo"}:
        raise ValueError("estimator must be 'pairwise' or 'loo'")
    if reference == "within":
        if groups is None:
            raise ValueError("groups are required for a within-group reference")
        labels = np.asarray(list(groups), dtype=object)
        if labels.shape != (n,):
            raise ValueError("groups must contain one label per participant")
    else:
        labels = np.repeat("all", n)

    channel_scores: list[np.ndarray] = []
    for ci in channel_indices:
        x = array[:, :, ci]
        scores = np.full(n, np.nan, dtype=float)
        if estimator == "pairwise":
            corr = pairwise_correlation_matrix(x)
            corr = _transform(corr, transform)
            for label in np.unique(labels):
                idx = np.flatnonzero(labels == label)
                if idx.size < 2:
                    continue
                block = corr[np.ix_(idx, idx)].copy()
                np.fill_diagonal(block, np.nan)
                scores[idx] = _nanmean_rows(block)
        else:
            for label in np.unique(labels):
                idx = np.flatnonzero(labels == label)
                if idx.size < 2:
                    continue
                block = x[idx]
                total = np.nansum(block, axis=0, keepdims=True)
                reference_mean = (total - block) / (idx.size - 1)
                scores[idx] = _transform(_row_corr(block, reference_mean), transform)
        channel_scores.append(scores)
    return _nanmean_rows(np.stack(channel_scores, axis=1))


def boundary_contrast(
    boundary_scores: np.ndarray,
    pre_scores: np.ndarray,
    groups: Iterable[str],
    *,
    target_group: str = "Short-R",
    control_groups: Iterable[str] | None = None,
) -> ContrastResult:
    """Compute the target-versus-controls difference in boundary response."""

    boundary = np.asarray(boundary_scores, dtype=float)
    pre = np.asarray(pre_scores, dtype=float)
    if boundary.shape != pre.shape:
        raise ValueError("boundary and pre score arrays must have the same shape")
    if boundary.ndim == 2:
        change = np.nanmean(boundary - pre, axis=1)
    elif boundary.ndim == 1:
        change = boundary - pre
    else:
        raise ValueError("scores must be participant or participant-by-boundary arrays")
    labels = np.asarray(list(groups), dtype=object)
    if labels.shape != (change.shape[0],):
        raise ValueError("groups must match participant count")
    unique = list(dict.fromkeys(labels.tolist()))
    controls = list(control_groups) if control_groups is not None else [g for g in unique if g != target_group]
    if target_group not in unique or not controls:
        raise ValueError("target and at least one control group are required")
    group_changes = {g: float(np.nanmean(change[labels == g])) for g in unique}
    control_mean = float(np.mean([group_changes[g] for g in controls]))
    effect = group_changes[target_group] - control_mean
    return ContrastResult(effect, group_changes, target_group, control_mean)


def window_slice(boundary: int, width_seconds: float, rate_hz: int, phase: str) -> slice:
    """Return a half-open sample slice for a pre, boundary, or post window."""

    width = int(round(width_seconds * rate_hz))
    if width < 2:
        raise ValueError("window must contain at least two samples")
    if phase == "pre":
        start = boundary - width
    elif phase == "boundary":
        start = boundary - width // 2
    elif phase == "post":
        start = boundary
    else:
        raise ValueError("phase must be pre, boundary, or post")
    return slice(start, start + width)
