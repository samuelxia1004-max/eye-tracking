"""Intersection-union inference for effect and temporal-specificity nulls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossedNullDecision:
    """Decision record for two necessary evidential conditions.

    Rejection requires both the between-group label test and the temporal
    specificity test to reject at the same prespecified alpha.  The resulting
    intersection-union p-value is the larger component p-value.
    """

    p_label: float
    p_temporal: float
    p_iut: float
    alpha: float
    reject_both: bool


def intersection_union_p(p_label: float, p_temporal: float) -> float:
    """Return the valid p-value for the union null H_label OR H_temporal."""

    for value in (p_label, p_temporal):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("component p-values must lie in [0, 1]")
    return max(float(p_label), float(p_temporal))


def crossed_null_decision(
    p_label: float,
    p_temporal: float,
    *,
    alpha: float = 0.05,
) -> CrossedNullDecision:
    """Combine component tests without treating either as sufficient alone."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    p_iut = intersection_union_p(p_label, p_temporal)
    return CrossedNullDecision(
        p_label=float(p_label),
        p_temporal=float(p_temporal),
        p_iut=p_iut,
        alpha=float(alpha),
        reject_both=bool(p_iut <= alpha),
    )
