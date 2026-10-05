"""Minimal CorrCA implementation for held-participant EEG convergence checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class CorrCAFit:
    filters: np.ndarray
    eigenvalues: np.ndarray
    channel_mean: np.ndarray


def fit_corrca(data: np.ndarray, *, components: int = 3, ridge: float = 1e-6) -> CorrCAFit:
    """Fit reliable components to subjects viewing the same stimulus.

    ``data`` has shape participants × time × channels. Spatial filters are
    estimated only from the supplied participants.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("CorrCA requires at least three participant time series")
    if not np.isfinite(values).all():
        raise ValueError("CorrCA input must be finite")
    participant_n, timepoints, channels = values.shape
    if components < 1 or components > channels:
        raise ValueError("invalid CorrCA component count")
    channel_mean = values.mean(axis=(0, 1))
    centered = values - channel_mean
    within_sum = np.einsum("ntc,ntd->cd", centered, centered, optimize=True)
    summed = centered.sum(axis=0)
    between_sum = summed.T @ summed - within_sum
    within = within_sum / (participant_n * timepoints)
    between = between_sum / (participant_n * (participant_n - 1) * timepoints)
    regularizer = ridge * float(np.trace(within) / channels)
    within_regularized = within + np.eye(channels) * max(regularizer, np.finfo(float).eps)
    eigenvalues, eigenvectors = eigh(between, within_regularized, check_finite=True)
    order = np.argsort(eigenvalues)[::-1][:components]
    return CorrCAFit(
        filters=eigenvectors[:, order],
        eigenvalues=eigenvalues[order],
        channel_mean=channel_mean,
    )


def project_corrca(data: np.ndarray, fit: CorrCAFit) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.shape[-1] != fit.filters.shape[0]:
        raise ValueError("channel count does not match CorrCA filters")
    return (values - fit.channel_mean) @ fit.filters


def held_participant_corrca_isc(
    training: np.ndarray,
    held: np.ndarray,
    *,
    components: int = 3,
    ridge: float = 1e-6,
) -> tuple[float, np.ndarray, CorrCAFit]:
    """Correlate a held participant with the training-participant component mean."""

    fit = fit_corrca(training, components=components, ridge=ridge)
    projected_training = project_corrca(training, fit)
    projected_held = project_corrca(held, fit)
    reference = projected_training.mean(axis=0)
    correlations = np.empty(components, dtype=float)
    for component in range(components):
        correlations[component] = np.corrcoef(
            projected_held[:, component], reference[:, component]
        )[0, 1]
    fisher = np.arctanh(np.clip(correlations, -0.999999, 0.999999))
    return float(np.mean(fisher)), correlations, fit
