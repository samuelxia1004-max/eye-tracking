"""Attention-condition features and leakage-resistant benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "isc_x_mean",
    "isc_y_mean",
    "isc_pupil_mean",
)


@dataclass(frozen=True)
class AttentionBenchmarkResult:
    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: dict[str, float]
    selected_c: pd.DataFrame


def _safe_corr(a: np.ndarray, b: np.ndarray, min_valid_fraction: float) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.mean() < min_valid_fraction or valid.sum() < 3:
        return float("nan")
    av = a[valid] - np.mean(a[valid])
    bv = b[valid] - np.mean(b[valid])
    denom = np.sqrt(float(av @ av) * float(bv @ bv))
    if denom <= 0:
        return float("nan")
    return float(np.clip((av @ bv) / denom, -1.0, 1.0))


def _window_starts(length: int, width: int) -> np.ndarray:
    if width < 3:
        raise ValueError("window width must contain at least three samples")
    return np.arange(0, length - width + 1, width, dtype=int)


def build_attention_features(
    records: np.ndarray,
    participant: Sequence[str],
    video: Sequence[str],
    condition: Sequence[str],
    *,
    rate_hz: int = 10,
    window_seconds: float = 10.0,
    min_valid_fraction: float = 0.5,
    fisher: bool = True,
    target_indices: Sequence[int] | None = None,
    reference_mask: Sequence[bool] | None = None,
    validity_records: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize windowed ISC against a pooled, condition-label-free reference.

    Parameters
    ----------
    records
        Array shaped ``(recordings, time, 3)`` in X, Y, pupil order. Records
        may contain NaNs and may be padded to a common duration.
    participant, video, condition
        One identifier per recording. The reference for a record is the mean
        of every *other participant* watching the same video, pooled across
        conditions. Condition labels are never used to construct references.

    Notes
    -----
    This is intentionally a cohort-relative, transductive feature definition:
    held-out-video records can contribute to one another's unlabeled reference
    but never to supervised model fitting. The manuscript must state this
    boundary explicitly.
    """

    data = np.asarray(records, dtype=float)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("records must have shape (recordings, time, 3)")
    validity = None if validity_records is None else np.asarray(validity_records, dtype=float)
    if validity is not None and validity.shape != data.shape:
        raise ValueError("validity_records must match records")
    n = data.shape[0]
    ids = np.asarray(participant, dtype=object)
    videos = np.asarray(video, dtype=object)
    conditions = np.asarray(condition, dtype=object)
    if any(values.shape != (n,) for values in (ids, videos, conditions)):
        raise ValueError("metadata must contain one value per recording")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    targets = np.arange(n, dtype=int) if target_indices is None else np.asarray(target_indices, dtype=int)
    if targets.ndim != 1 or np.any((targets < 0) | (targets >= n)):
        raise ValueError("target_indices contain an invalid record index")
    allowed_reference = (
        np.ones(n, dtype=bool)
        if reference_mask is None
        else np.asarray(reference_mask, dtype=bool)
    )
    if allowed_reference.shape != (n,):
        raise ValueError("reference_mask must contain one value per recording")
    width = int(round(float(window_seconds) * int(rate_hz)))
    starts = _window_starts(data.shape[1], width)
    reference_cache: dict[tuple[object, object], np.ndarray] = {}
    for target_video in np.unique(videos[targets]):
        base_mask = allowed_reference & (videos == target_video)
        base_data = data[base_mask]
        total_sum = np.nansum(base_data, axis=0)
        total_count = np.isfinite(base_data).sum(axis=0)
        for target_participant in np.unique(ids[targets][videos[targets] == target_video]):
            own_mask = base_mask & (ids == target_participant)
            own_data = data[own_mask]
            own_sum = np.nansum(own_data, axis=0) if own_data.size else 0.0
            own_count = np.isfinite(own_data).sum(axis=0) if own_data.size else 0
            reference_sum = total_sum - own_sum
            reference_count = total_count - own_count
            reference = np.full(data.shape[1:], np.nan, dtype=float)
            np.divide(
                reference_sum,
                reference_count,
                out=reference,
                where=reference_count > 0,
            )
            reference_cache[(target_video, target_participant)] = reference
    rows: list[dict[str, object]] = []
    for index in targets:
        reference = reference_cache[(videos[index], ids[index])]
        channel_values: list[list[float]] = [[], [], []]
        for start in starts:
            stop = start + width
            for channel_index in range(3):
                if validity is not None:
                    local_quality = validity[index, start:stop, channel_index]
                    quality = (
                        float(np.nanmean(local_quality))
                        if np.isfinite(local_quality).any()
                        else np.nan
                    )
                    if not np.isfinite(quality) or quality < min_valid_fraction:
                        channel_values[channel_index].append(float("nan"))
                        continue
                value = _safe_corr(
                    data[index, start:stop, channel_index],
                    reference[start:stop, channel_index],
                    0.0 if validity is not None else min_valid_fraction,
                )
                if fisher and np.isfinite(value):
                    value = float(np.arctanh(np.clip(value, -0.999999, 0.999999)))
                channel_values[channel_index].append(value)
        summaries: dict[str, float] = {}
        labels = ("x", "y", "pupil")
        for label, values in zip(labels, channel_values, strict=True):
            vector = np.asarray(values, dtype=float)
            finite = vector[np.isfinite(vector)]
            summaries[f"isc_{label}_mean"] = float(np.mean(finite)) if finite.size else np.nan
            summaries[f"isc_{label}_sd"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
            )
        matrix = np.asarray(channel_values, dtype=float)
        composite_counts = np.isfinite(matrix).sum(axis=0)
        composite = np.full(matrix.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(matrix, axis=0),
            composite_counts,
            out=composite,
            where=composite_counts > 0,
        )
        summaries["isc_composite_mean"] = float(np.nanmean(composite))
        summaries["isc_composite_sd"] = float(np.nanstd(composite, ddof=1))
        rows.append(
            {
                "record_index": index,
                "participant": str(ids[index]),
                "video": str(videos[index]),
                "condition": str(conditions[index]),
                "n_windows": int(starts.size),
                **summaries,
            }
        )
    return pd.DataFrame(rows)


def nested_attention_benchmark_from_records(
    records: np.ndarray,
    participant: Sequence[str],
    video: Sequence[str],
    condition: Sequence[str],
    *,
    rate_hz: int = 10,
    window_seconds: float = 10.0,
    min_valid_fraction: float = 0.5,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    positive_label: str = "attentive",
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    participant_folds: int = 5,
    seed: int = 20260820,
    validity_records: np.ndarray | None = None,
) -> AttentionBenchmarkResult:
    """Run outer CV while rebuilding references without held participants.

    The held-out video's unlabeled records from non-held participants may form
    its stimulus-locked reference, but no held participant contributes to any
    outer-fold feature or normalization estimate.
    """

    ids = np.asarray(participant, dtype=object)
    videos = np.asarray(video, dtype=object)
    conditions = np.asarray(condition, dtype=object)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for train_index, test_index, participant_fold, held_video in double_block_splits(
        ids, videos, participant_folds=participant_folds, seed=seed
    ):
        held_ids = np.unique(ids[test_index])
        reference_mask = ~np.isin(ids, held_ids)
        fold_features = build_attention_features(
            records,
            ids,
            videos,
            conditions,
            rate_hz=rate_hz,
            window_seconds=window_seconds,
            min_valid_fraction=min_valid_fraction,
            target_indices=np.concatenate([train_index, test_index]),
            reference_mask=reference_mask,
            validity_records=validity_records,
        )
        train_features = fold_features[fold_features["record_index"].isin(train_index)].copy()
        test_features = fold_features[fold_features["record_index"].isin(test_index)].copy()
        if not np.isfinite(train_features.loc[:, feature_columns].to_numpy(float)).all():
            raise ValueError("non-finite outer-training features")
        if not np.isfinite(test_features.loc[:, feature_columns].to_numpy(float)).all():
            raise ValueError("non-finite outer-test features")
        c_value = _select_c(
            train_features,
            feature_columns,
            c_grid,
            positive_label=positive_label,
            participant_folds=max(2, participant_folds - 1),
            seed=seed + participant_fold,
        )
        probability = _fit_predict(
            train_features,
            test_features,
            feature_columns,
            c_value=c_value,
            positive_label=positive_label,
            seed=seed + participant_fold,
        )
        y_test = (test_features["condition"] == positive_label).astype(int).to_numpy()
        predicted = (probability >= 0.5).astype(int)
        fold = test_features[["record_index", "participant", "video", "condition"]].copy()
        fold["participant_fold"] = participant_fold
        fold["held_video"] = held_video
        fold["probability_attentive"] = probability
        fold["truth_attentive"] = y_test
        predictions.append(fold)
        if np.unique(y_test).size == 2:
            metrics.append(
                {
                    "participant_fold": participant_fold,
                    "held_video": held_video,
                    "n_test": len(test_features),
                    "auc": float(roc_auc_score(y_test, probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
                    "brier": float(brier_score_loss(y_test, probability)),
                }
            )
        selections.append(
            {
                "participant_fold": participant_fold,
                "held_video": held_video,
                "selected_c": c_value,
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if prediction_frame["record_index"].duplicated().any():
        raise RuntimeError("outer predictions are not one-to-one with records")
    truth = prediction_frame["truth_attentive"].to_numpy(int)
    probability = prediction_frame["probability_attentive"].to_numpy(float)
    aggregate = {
        "n_records": float(len(prediction_frame)),
        "auc": float(roc_auc_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, probability >= 0.5)),
        "brier": float(brier_score_loss(truth, probability)),
    }
    return AttentionBenchmarkResult(
        predictions=prediction_frame,
        fold_metrics=pd.DataFrame(metrics),
        aggregate_metrics=aggregate,
        selected_c=pd.DataFrame(selections),
    )


def double_block_splits(
    participant: Iterable[str],
    video: Iterable[str],
    *,
    participant_folds: int = 5,
    seed: int = 20260820,
) -> Iterator[tuple[np.ndarray, np.ndarray, int, str]]:
    """Yield outer splits whose train rows share neither participant nor video with test rows."""

    ids = np.asarray(list(participant), dtype=object)
    videos = np.asarray(list(video), dtype=object)
    if ids.shape != videos.shape:
        raise ValueError("participant and video arrays must match")
    unique_ids = np.unique(ids)
    unique_videos = np.unique(videos)
    if participant_folds < 2 or participant_folds > unique_ids.size:
        raise ValueError("invalid participant_folds")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    id_folds = np.array_split(shuffled, participant_folds)
    for fold_index, held_ids in enumerate(id_folds):
        for held_video in unique_videos:
            test = np.flatnonzero(np.isin(ids, held_ids) & (videos == held_video))
            train = np.flatnonzero(~np.isin(ids, held_ids) & (videos != held_video))
            if test.size and train.size:
                yield train, test, fold_index, str(held_video)


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    c_value: float,
    positive_label: str,
    seed: int,
) -> np.ndarray:
    x_train = train.loc[:, feature_columns].to_numpy(float)
    x_test = test.loc[:, feature_columns].to_numpy(float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("benchmark features must be finite before model fitting")
    y_train = (train["condition"].to_numpy(object) == positive_label).astype(int)
    if np.unique(y_train).size != 2:
        raise ValueError("training fold must contain both conditions")
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        random_state=seed,
        max_iter=2000,
    )
    model.fit(scaler.transform(x_train), y_train)
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def _select_c(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    c_grid: Sequence[float],
    *,
    positive_label: str,
    participant_folds: int,
    seed: int,
) -> float:
    scores = {float(c): [] for c in c_grid}
    for inner_train, inner_test, _, _ in double_block_splits(
        train["participant"],
        train["video"],
        participant_folds=min(participant_folds, train["participant"].nunique()),
        seed=seed + 1,
    ):
        inner_train_frame = train.iloc[inner_train]
        inner_test_frame = train.iloc[inner_test]
        y_test = (inner_test_frame["condition"] == positive_label).astype(int).to_numpy()
        if np.unique(y_test).size != 2:
            continue
        for c_value in scores:
            probability = _fit_predict(
                inner_train_frame,
                inner_test_frame,
                feature_columns,
                c_value=c_value,
                positive_label=positive_label,
                seed=seed,
            )
            scores[c_value].append(roc_auc_score(y_test, probability))
    mean_scores = {c: float(np.mean(values)) if values else -np.inf for c, values in scores.items()}
    best = max(mean_scores.values())
    # Prefer stronger regularization (smaller C) when inner-CV scores tie.
    return min(c for c, value in mean_scores.items() if np.isclose(value, best))


def nested_attention_benchmark(
    features: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    positive_label: str = "attentive",
    c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    participant_folds: int = 5,
    seed: int = 20260820,
) -> AttentionBenchmarkResult:
    """Evaluate an L2 logistic benchmark with participant-and-video blocking."""

    required = {"participant", "video", "condition", *feature_columns}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    frame = features.reset_index(drop=True).copy()
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for train_index, test_index, participant_fold, held_video in double_block_splits(
        frame["participant"], frame["video"], participant_folds=participant_folds, seed=seed
    ):
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        c_value = _select_c(
            train,
            feature_columns,
            c_grid,
            positive_label=positive_label,
            participant_folds=max(2, participant_folds - 1),
            seed=seed + participant_fold,
        )
        probability = _fit_predict(
            train,
            test,
            feature_columns,
            c_value=c_value,
            positive_label=positive_label,
            seed=seed + participant_fold,
        )
        y_test = (test["condition"] == positive_label).astype(int).to_numpy()
        predicted = (probability >= 0.5).astype(int)
        fold = test[["record_index", "participant", "video", "condition"]].copy()
        fold["participant_fold"] = participant_fold
        fold["held_video"] = held_video
        fold["probability_attentive"] = probability
        fold["truth_attentive"] = y_test
        predictions.append(fold)
        if np.unique(y_test).size == 2:
            metrics.append(
                {
                    "participant_fold": participant_fold,
                    "held_video": held_video,
                    "n_test": len(test),
                    "auc": float(roc_auc_score(y_test, probability)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
                    "brier": float(brier_score_loss(y_test, probability)),
                }
            )
        selections.append(
            {
                "participant_fold": participant_fold,
                "held_video": held_video,
                "selected_c": c_value,
            }
        )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    if prediction_frame["record_index"].duplicated().any():
        raise RuntimeError("outer predictions are not one-to-one with records")
    truth = prediction_frame["truth_attentive"].to_numpy(int)
    probability = prediction_frame["probability_attentive"].to_numpy(float)
    aggregate = {
        "n_records": float(len(prediction_frame)),
        "auc": float(roc_auc_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, probability >= 0.5)),
        "brier": float(brier_score_loss(truth, probability)),
    }
    return AttentionBenchmarkResult(
        predictions=prediction_frame,
        fold_metrics=pd.DataFrame(metrics),
        aggregate_metrics=aggregate,
        selected_c=pd.DataFrame(selections),
    )
