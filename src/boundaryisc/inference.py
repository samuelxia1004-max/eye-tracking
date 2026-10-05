"""Permutation, bootstrap, and pseudo-boundary inference."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .core import CHANNEL_MAP, pairwise_correlation_matrix, window_slice


def _group_edge_mean(matrix: np.ndarray, indices: np.ndarray) -> float:
    if indices.size < 2:
        return float("nan")
    block = matrix[np.ix_(indices, indices)]
    upper = block[np.triu_indices(indices.size, 1)]
    return float(np.nanmean(upper))


def pairwise_boundary_matrices(
    data: np.ndarray,
    boundaries: list[int],
    *,
    width_seconds: float = 10,
    rate_hz: int = 10,
    channel: str = "composite",
    transform: str = "raw",
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary and pre pairwise matrices for every event boundary."""

    indices = CHANNEL_MAP[channel]
    n = data.shape[0]
    boundary_mats = np.empty((len(boundaries), n, n), dtype=np.float32)
    pre_mats = np.empty_like(boundary_mats)
    for bi, boundary in enumerate(boundaries):
        channel_boundary = []
        channel_pre = []
        for ci in indices:
            for phase, target in (("boundary", channel_boundary), ("pre", channel_pre)):
                sl = window_slice(boundary, width_seconds, rate_hz, phase)
                corr = pairwise_correlation_matrix(data[:, sl, ci])
                if transform == "fisher":
                    corr = np.arctanh(np.clip(corr, -0.999999, 0.999999))
                elif transform == "legacy":
                    corr = (1.0 + corr) / 2.0
                elif transform != "raw":
                    raise ValueError(f"unknown transform {transform}")
                target.append(corr)
        boundary_mats[bi] = np.nanmean(np.stack(channel_boundary), axis=0)
        pre_mats[bi] = np.nanmean(np.stack(channel_pre), axis=0)
    return boundary_mats, pre_mats


def contrast_from_pairwise(
    difference: np.ndarray,
    labels: np.ndarray,
    *,
    target: str = "Short-R",
    group_order: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    matrix = np.nanmean(difference, axis=0) if difference.ndim == 3 else difference
    order = group_order or list(OrderedDict.fromkeys(labels.tolist()))
    means = {group: _group_edge_mean(matrix, np.flatnonzero(labels == group)) for group in order}
    controls = [means[group] for group in order if group != target]
    return float(means[target] - np.mean(controls)), means


def label_permutation_test(
    difference: np.ndarray,
    labels: np.ndarray,
    *,
    target: str,
    group_order: list[str],
    n_permutations: int,
    seed: int,
) -> dict:
    observed, observed_groups = contrast_from_pairwise(
        difference, labels, target=target, group_order=group_order
    )
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    sizes = [int(np.sum(labels == group)) for group in group_order]
    mean_difference = np.nanmean(difference, axis=0)
    for iteration in range(n_permutations):
        permuted = rng.permutation(labels.size)
        cursor = 0
        means = {}
        for group, size in zip(group_order, sizes, strict=True):
            idx = permuted[cursor : cursor + size]
            cursor += size
            means[group] = _group_edge_mean(mean_difference, idx)
        null[iteration] = means[target] - np.mean([means[g] for g in group_order if g != target])
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1)
    return {
        "observed": observed,
        "group_changes": observed_groups,
        "p_value": p_value,
        "null": null,
        "n_permutations": n_permutations,
    }


def hierarchical_bayesian_bootstrap(
    difference: np.ndarray,
    labels: np.ndarray,
    *,
    target: str,
    group_order: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Two-level weighted bootstrap over participants and event boundaries."""

    if difference.ndim != 3:
        raise ValueError("difference must have shape (boundaries, participants, participants)")
    rng = np.random.default_rng(seed)
    b_count = difference.shape[0]
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        boundary_weights = rng.dirichlet(np.ones(b_count))
        group_means: dict[str, float] = {}
        for group in group_order:
            idx = np.flatnonzero(labels == group)
            weights = rng.dirichlet(np.ones(idx.size))
            denominator = 1.0 - float(np.sum(weights * weights))
            if denominator <= 0:
                group_means[group] = np.nan
                continue
            block = difference[:, idx][:, :, idx]
            per_boundary = np.einsum("i,bij,j->b", weights, block, weights, optimize=True) / denominator
            group_means[group] = float(boundary_weights @ per_boundary)
        draws[draw] = group_means[target] - np.mean(
            [group_means[g] for g in group_order if g != target]
        )
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return {"draws": draws, "ci_low": float(low), "ci_high": float(high), "n_bootstrap": n_bootstrap}


def _rolling_pairwise_mean(group_data: np.ndarray, width: int) -> np.ndarray:
    """Mean pairwise rolling Pearson r for every circular window start."""

    n, time = group_data.shape
    if n < 2:
        return np.full(time, np.nan)
    extended = np.concatenate([group_data, group_data[:, : width - 1]], axis=1).astype(np.float64)
    total = np.zeros(time, dtype=np.float64)
    pairs = 0
    for i in range(n - 1):
        x = extended[i]
        sx = np.concatenate([[0.0], np.cumsum(x)])
        sx2 = np.concatenate([[0.0], np.cumsum(x * x)])
        xsum = sx[width:] - sx[:-width]
        x2sum = sx2[width:] - sx2[:-width]
        for j in range(i + 1, n):
            y = extended[j]
            sy = np.concatenate([[0.0], np.cumsum(y)])
            sy2 = np.concatenate([[0.0], np.cumsum(y * y)])
            sxy = np.concatenate([[0.0], np.cumsum(x * y)])
            ysum = sy[width:] - sy[:-width]
            y2sum = sy2[width:] - sy2[:-width]
            xysum = sxy[width:] - sxy[:-width]
            numerator = xysum - (xsum * ysum / width)
            denom = np.sqrt(
                np.maximum(x2sum - xsum * xsum / width, 0)
                * np.maximum(y2sum - ysum * ysum / width, 0)
            )
            corr = np.zeros(time, dtype=np.float64)
            np.divide(numerator, denom, out=corr, where=denom > 0)
            total += np.clip(corr, -1.0, 1.0)
            pairs += 1
    return total / pairs


def pseudo_boundary_test(
    data: np.ndarray,
    groups: np.ndarray,
    boundaries: list[int],
    *,
    target_group: str = "Short-R",
    group_order: list[str] | None = None,
    width_seconds: float = 10,
    rate_hz: int = 10,
    channel: str = "composite",
    n_shifts: int = 10000,
    seed: int = 20260816,
) -> dict:
    """Test boundary specificity with common circular shifts of all boundaries."""

    order = group_order or list(OrderedDict.fromkeys(groups.tolist()))
    width = int(round(width_seconds * rate_hz))
    time = data.shape[1]
    channel_indices = CHANNEL_MAP[channel]
    curves: dict[str, np.ndarray] = {}
    for group in order:
        idx = np.flatnonzero(groups == group)
        per_channel = [_rolling_pairwise_mean(data[idx, :, ci], width) for ci in channel_indices]
        curves[group] = np.mean(np.stack(per_channel), axis=0)

    boundary_starts = (np.asarray(boundaries, dtype=int) - width // 2) % time
    pre_starts = (np.asarray(boundaries, dtype=int) - width) % time

    def contrast_for_shift(shift: int) -> float:
        changes = {
            group: float(
                np.mean(
                    curves[group][(boundary_starts + shift) % time]
                    - curves[group][(pre_starts + shift) % time]
                )
            )
            for group in order
        }
        return changes[target_group] - np.mean([changes[g] for g in order if g != target_group])

    observed = contrast_for_shift(0)
    candidates = np.arange(1, time, dtype=int)
    rng = np.random.default_rng(seed)
    if n_shifts > candidates.size:
        shifts = rng.choice(candidates, size=n_shifts, replace=True)
    else:
        shifts = rng.choice(candidates, size=n_shifts, replace=False)
    null = np.asarray([contrast_for_shift(int(shift)) for shift in shifts])
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_shifts + 1)
    return {
        "observed": float(observed),
        "p_value": float(p_value),
        "null": null,
        "shifts": shifts,
        "n_shifts": n_shifts,
    }
