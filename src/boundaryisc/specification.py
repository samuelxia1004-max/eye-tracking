"""Prespecified multiverse for event-boundary ISC."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .core import boundary_contrast, compute_isc, window_slice


def run_specification_grid(
    data: np.ndarray,
    groups: np.ndarray,
    boundaries: list[int],
    *,
    grid: dict,
    rate_hz: int = 10,
    target_group: str = "Short-R",
) -> pd.DataFrame:
    """Run and return every prespecified specification without filtering."""

    rows: list[dict] = []
    iterator = product(
        grid["channels"],
        grid["estimators"],
        grid["transforms"],
        grid["window_seconds"],
        grid["references"],
    )
    for spec_id, (channel, estimator, transform, width, reference) in enumerate(iterator, start=1):
        boundary_scores = []
        pre_scores = []
        for boundary in boundaries:
            bsl = window_slice(boundary, width, rate_hz, "boundary")
            psl = window_slice(boundary, width, rate_hz, "pre")
            boundary_scores.append(
                compute_isc(
                    data[:, bsl, :],
                    groups,
                    estimator=estimator,
                    reference=reference,
                    channel=channel,
                    transform=transform,
                )
            )
            pre_scores.append(
                compute_isc(
                    data[:, psl, :],
                    groups,
                    estimator=estimator,
                    reference=reference,
                    channel=channel,
                    transform=transform,
                )
            )
        result = boundary_contrast(
            np.stack(boundary_scores, axis=1),
            np.stack(pre_scores, axis=1),
            groups,
            target_group=target_group,
        )
        row = {
            "spec_id": spec_id,
            "channel": channel,
            "estimator": estimator,
            "transform": transform,
            "window_seconds": width,
            "reference": reference,
            "effect": result.effect,
            "direction_matches_primary_hypothesis": result.effect < 0,
        }
        row.update({f"change_{group}": value for group, value in result.group_changes.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["effect_rank"] = frame["effect"].rank(method="first")
    return frame
