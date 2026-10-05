"""Inference and frozen-score transport for the ocular attention benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .attention import FEATURE_COLUMNS


@dataclass(frozen=True)
class FrozenOcularModel:
    feature_columns: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    c_value: float

    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        standardized = (values - self.scaler_mean) / self.scaler_scale
        linear = standardized @ self.coefficient + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -35, 35)))


def participant_signflip_test(
    predictions: pd.DataFrame,
    *,
    n_permutations: int = 10000,
    seed: int = 20260820,
) -> dict[str, object]:
    """Test held-out attentive-minus-distracted probabilities by participant sign flips."""

    paired = (
        predictions.pivot_table(
            index=["participant", "video"],
            columns="condition",
            values="probability_attentive",
            aggfunc="mean",
        )
        .dropna(subset=["attentive", "distracted"])
        .assign(difference=lambda frame: frame["attentive"] - frame["distracted"])
        .reset_index()
    )
    participant_difference = paired.groupby("participant")["difference"].mean()
    observed = float(participant_difference.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(n_permutations, participant_difference.size),
        replace=True,
    )
    null = signs @ participant_difference.to_numpy(float) / participant_difference.size
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1)
    return {
        "observed_mean_paired_probability_difference": observed,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "n_participants": int(participant_difference.size),
        "null": null,
        "paired_video_records": paired,
        "participant_differences": participant_difference.reset_index(),
    }


def two_way_auc_bootstrap(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = 20260820,
) -> dict[str, object]:
    """Bootstrap participants and videos within experiment using product weights."""

    frame = predictions.copy()
    frame["experiment"] = frame["video"].str.extract(r"^exp(\d+)_", expand=False)
    if frame["experiment"].isna().any():
        raise ValueError("video IDs must start with exp<number>_ for two-way bootstrap")
    truth = frame["truth_attentive"].to_numpy(int)
    probability = frame["probability_attentive"].to_numpy(float)
    observed = float(roc_auc_score(truth, probability))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    experiments = frame["experiment"].unique()
    for draw in range(n_bootstrap):
        weight = np.zeros(len(frame), dtype=float)
        for experiment in experiments:
            mask = frame["experiment"] == experiment
            subset = frame.loc[mask]
            participants = subset["participant"].unique()
            videos = subset["video"].unique()
            participant_sample = rng.choice(participants, size=len(participants), replace=True)
            video_sample = rng.choice(videos, size=len(videos), replace=True)
            participant_counts = pd.Series(participant_sample).value_counts()
            video_counts = pd.Series(video_sample).value_counts()
            local = (
                subset["participant"].map(participant_counts).fillna(0).to_numpy(float)
                * subset["video"].map(video_counts).fillna(0).to_numpy(float)
            )
            weight[np.flatnonzero(mask)] = local
        if weight[truth == 1].sum() <= 0 or weight[truth == 0].sum() <= 0:
            draws[draw] = np.nan
        else:
            draws[draw] = roc_auc_score(truth, probability, sample_weight=weight)
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return {
        "observed_auc": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_bootstrap": int(n_bootstrap),
        "draws": draws,
    }


def fit_frozen_ocular_model(
    features: pd.DataFrame,
    *,
    c_value: float,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    positive_label: str = "attentive",
    seed: int = 20260820,
) -> FrozenOcularModel:
    """Fit the final model after outer validation and return serializable parameters."""

    x = features.loc[:, feature_columns].to_numpy(float)
    y = (features["condition"].to_numpy(object) == positive_label).astype(int)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        random_state=seed,
        max_iter=2000,
    ).fit(scaler.transform(x), y)
    return FrozenOcularModel(
        feature_columns=tuple(feature_columns),
        scaler_mean=scaler.mean_.astype(float),
        scaler_scale=scaler.scale_.astype(float),
        coefficient=model.coef_[0].astype(float),
        intercept=float(model.intercept_[0]),
        c_value=float(c_value),
    )


def _circular_rolling_corr(x: np.ndarray, y: np.ndarray, width: int) -> np.ndarray:
    """Rolling Pearson correlations for matched participant rows."""

    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be matched participant-by-time arrays")
    extended_x = np.concatenate([x, x[:, : width - 1]], axis=1).astype(float)
    extended_y = np.concatenate([y, y[:, : width - 1]], axis=1).astype(float)

    def rolling_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate(
            [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1
        )
        return cumulative[:, width:] - cumulative[:, :-width]

    sx = rolling_sum(extended_x)
    sy = rolling_sum(extended_y)
    sx2 = rolling_sum(extended_x * extended_x)
    sy2 = rolling_sum(extended_y * extended_y)
    sxy = rolling_sum(extended_x * extended_y)
    numerator = sxy - sx * sy / width
    denominator = np.sqrt(
        np.maximum(sx2 - sx * sx / width, 0.0)
        * np.maximum(sy2 - sy * sy / width, 0.0)
    )
    correlation = np.full_like(numerator, np.nan, dtype=float)
    np.divide(numerator, denominator, out=correlation, where=denominator > 0)
    return np.clip(correlation, -1.0, 1.0)


def rolling_loo_features(data: np.ndarray, width: int) -> np.ndarray:
    """Return Fisher-z LOO-reference features at every circular window start."""

    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("data must have shape (participants, time, 3)")
    if not np.isfinite(values).all():
        raise ValueError("rolling Sherlock features require finite processed signals")
    participant_n = values.shape[0]
    if participant_n < 3:
        raise ValueError("at least three participants are required")
    features = np.empty_like(values, dtype=float)
    for channel in range(3):
        channel_data = values[:, :, channel]
        reference = (channel_data.sum(axis=0, keepdims=True) - channel_data) / (
            participant_n - 1
        )
        correlation = _circular_rolling_corr(channel_data, reference, width)
        features[:, :, channel] = np.arctanh(np.clip(correlation, -0.999999, 0.999999))
    return features


def transported_probability_curve(
    data: np.ndarray,
    model: FrozenOcularModel,
    *,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute window features and frozen-model probability at every start time."""

    features = rolling_loo_features(data, width)
    probability = model.predict_probability(features)
    return features, probability


def group_contrast(values: np.ndarray, groups: np.ndarray, target: str) -> tuple[float, dict[str, float]]:
    """Target mean minus the equal-weight mean of all other group means."""

    unique = list(dict.fromkeys(groups.tolist()))
    means = {group: float(np.mean(values[groups == group])) for group in unique}
    controls = [means[group] for group in unique if group != target]
    return float(means[target] - np.mean(controls)), means


def boundary_probability_contrast(
    probability_curve: np.ndarray,
    groups: np.ndarray,
    boundaries: Sequence[int],
    *,
    width: int,
    target: str,
    shift: int = 0,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Compute participant boundary-minus-pre score and the target contrast."""

    timepoints = probability_curve.shape[1]
    boundaries_array = np.asarray(boundaries, dtype=int)
    centered_start = (boundaries_array - width // 2 + shift) % timepoints
    pre_start = (boundaries_array - width + shift) % timepoints
    change = np.mean(
        probability_curve[:, centered_start] - probability_curve[:, pre_start], axis=1
    )
    effect, means = group_contrast(change, groups, target)
    return effect, means, change


def fixed_size_group_permutation(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    target: str,
    n_permutations: int,
    seed: int,
) -> dict[str, object]:
    """Shuffle participant labels while preserving all group sizes."""

    observed, means = group_contrast(values, groups, target)
    order = list(dict.fromkeys(groups.tolist()))
    sizes = [int(np.sum(groups == group)) for group in order]
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        shuffled = rng.permutation(len(groups))
        cursor = 0
        permuted = np.empty(groups.shape, dtype=object)
        for group, size in zip(order, sizes, strict=True):
            permuted[shuffled[cursor : cursor + size]] = group
            cursor += size
        null[iteration], _ = group_contrast(values, permuted, target)
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1)
    return {
        "observed": observed,
        "group_means": means,
        "p_value": float(p_value),
        "null": null,
        "n_permutations": int(n_permutations),
    }


def boundary_location_test(
    probability_curve: np.ndarray,
    groups: np.ndarray,
    boundaries: Sequence[int],
    *,
    width: int,
    target: str,
    n_shifts: int,
    seed: int,
) -> dict[str, object]:
    """Common circular shifts preserving the complete boundary configuration."""

    observed, means, _ = boundary_probability_contrast(
        probability_curve, groups, boundaries, width=width, target=target, shift=0
    )
    candidates = np.arange(1, probability_curve.shape[1], dtype=int)
    rng = np.random.default_rng(seed)
    shifts = rng.choice(candidates, size=n_shifts, replace=n_shifts > candidates.size)
    null = np.empty(n_shifts, dtype=float)
    for index, shift in enumerate(shifts):
        null[index], _, _ = boundary_probability_contrast(
            probability_curve,
            groups,
            boundaries,
            width=width,
            target=target,
            shift=int(shift),
        )
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_shifts + 1)
    return {
        "observed": observed,
        "group_means": means,
        "p_value": float(p_value),
        "null": null,
        "shifts": shifts,
        "n_shifts": int(n_shifts),
    }


def group_weighted_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    target: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Bayesian participant bootstrap for an equal-control-weight group contrast."""

    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1 or observations.shape != groups.shape:
        raise ValueError("values and groups must be matched participant vectors")
    order = list(dict.fromkeys(groups.tolist()))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        means = {}
        for group in order:
            group_values = observations[groups == group]
            weights = rng.dirichlet(np.ones(group_values.size))
            means[group] = float(weights @ group_values)
        draws[draw] = means[target] - np.mean([means[group] for group in order if group != target])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"draws": draws, "ci_low": float(low), "ci_high": float(high)}


def participant_boundary_bootstrap(
    probability_curve: np.ndarray,
    groups: np.ndarray,
    boundaries: Sequence[int],
    *,
    width: int,
    target: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Bayesian bootstrap over participants within group and annotated boundaries."""

    boundaries_array = np.asarray(boundaries, dtype=int)
    centered_start = (boundaries_array - width // 2) % probability_curve.shape[1]
    pre_start = (boundaries_array - width) % probability_curve.shape[1]
    change = probability_curve[:, centered_start] - probability_curve[:, pre_start]
    order = list(dict.fromkeys(groups.tolist()))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        boundary_weights = rng.dirichlet(np.ones(boundaries_array.size))
        means = {}
        for group in order:
            block = change[groups == group]
            participant_weights = rng.dirichlet(np.ones(block.shape[0]))
            means[group] = float(participant_weights @ block @ boundary_weights)
        draws[draw] = means[target] - np.mean([means[group] for group in order if group != target])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"draws": draws, "ci_low": float(low), "ci_high": float(high)}
