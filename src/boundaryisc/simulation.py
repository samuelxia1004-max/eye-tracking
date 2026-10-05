"""AR(1) recovery simulation for event-boundary ISC estimators."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .core import compute_isc


def _ar1_noise(rng: np.random.Generator, n: int, time: int, channels: int, rho: float) -> np.ndarray:
    innovations = rng.normal(size=(n, time, channels))
    out = np.empty_like(innovations)
    out[:, 0] = innovations[:, 0] / np.sqrt(max(1 - rho * rho, 1e-8))
    scale = np.sqrt(1 - rho * rho)
    for t in range(1, time):
        out[:, t] = rho * out[:, t - 1] + scale * innovations[:, t]
    return out


def _one_simulation(
    rng: np.random.Generator,
    *,
    rho: float,
    group_size: int,
    signal: float,
    affected: str,
    time: int,
    boundary: int,
    width: int,
) -> tuple[float, float, float, float]:
    n = group_size * 2
    groups = np.asarray(["target"] * group_size + ["control"] * group_size)
    data = _ar1_noise(rng, n, time, 3, rho)
    shared = _ar1_noise(rng, 1, time, 3, rho)[0]
    b_slice = slice(boundary - width // 2, boundary + width // 2)
    channel_indices = (0,) if affected == "x" else (0, 1, 2)
    for ci in channel_indices:
        data[group_size:, b_slice, ci] += signal * shared[b_slice, ci]
    p_slice = slice(boundary - width, boundary)
    composite_b = compute_isc(data[:, b_slice], groups, channel="composite")
    composite_p = compute_isc(data[:, p_slice], groups, channel="composite")
    x_b = compute_isc(data[:, b_slice], groups, channel="x")
    x_p = compute_isc(data[:, p_slice], groups, channel="x")
    comp_change = composite_b - composite_p
    x_change = x_b - x_p
    comp_effect = float(np.mean(comp_change[:group_size]) - np.mean(comp_change[group_size:]))
    x_effect = float(np.mean(x_change[:group_size]) - np.mean(x_change[group_size:]))
    se = np.sqrt(
        np.var(comp_change[:group_size], ddof=1) / group_size
        + np.var(comp_change[group_size:], ddof=1) / group_size
    )
    z = comp_effect / se if se > 0 else 0.0
    return comp_effect, x_effect, z, float(se)


def simulate_recovery(config: dict) -> pd.DataFrame:
    """Run the frozen factorial AR(1) recovery simulation."""

    rng = np.random.default_rng(int(config["seed"]))
    rows: list[dict] = []
    scenarios = product(
        config["ar1"], config["group_size"], config["shared_signal"], config["affected_channel"]
    )
    for scenario_id, (rho, size, signal, affected) in enumerate(scenarios, start=1):
        for replicate in range(int(config["replicates"])):
            comp, x_only, z, se = _one_simulation(
                rng,
                rho=float(rho),
                group_size=int(size),
                signal=float(signal),
                affected=str(affected),
                time=int(config["series_length"]),
                boundary=int(config["boundary_index"]),
                width=int(config["window_samples"]),
            )
            population_r = float(signal) ** 2 / (1.0 + float(signal) ** 2)
            # The centered boundary window and immediately preceding window
            # overlap for half their duration. The induced signal occupies that
            # overlap in both estimates, so the population difference is half
            # the boundary-window correlation under this frozen design.
            contrast_fraction = 0.5
            truth_composite = -population_r * contrast_fraction * (1.0 / 3.0 if affected == "x" else 1.0)
            truth_x = -population_r * contrast_fraction
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "replicate": replicate,
                    "ar1": rho,
                    "group_size": size,
                    "shared_signal": signal,
                    "affected_channel": affected,
                    "composite_effect": comp,
                    "x_effect": x_only,
                    "truth_composite": truth_composite,
                    "truth_x": truth_x,
                    "composite_bias": comp - truth_composite,
                    "ci_low": comp - 1.959963984540054 * se,
                    "ci_high": comp + 1.959963984540054 * se,
                    "ci_covers_truth": (comp - 1.959963984540054 * se)
                    <= truth_composite
                    <= (comp + 1.959963984540054 * se),
                    "reject_two_sided_0_05": abs(z) > 1.959963984540054,
                }
            )
    return pd.DataFrame(rows)
