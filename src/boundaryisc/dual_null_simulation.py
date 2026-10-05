"""Recovery simulation for global and event-boundary-specific group effects."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .crossed_null import intersection_union_p


@dataclass(frozen=True)
class DualNullStatistics:
    global_effect: float
    boundary_effect: float
    p_global_label: float
    p_boundary_label: float
    p_boundary_temporal: float
    p_boundary_iut: float


def _ar1_panel(
    rng: np.random.Generator,
    participants: int,
    timepoints: int,
    rho: float,
) -> np.ndarray:
    burn_in = 100
    innovation = rng.normal(size=(participants, timepoints + burn_in))
    scale = np.sqrt(max(1.0 - rho * rho, 1e-10))
    panel = lfilter([scale], [1.0, -rho], innovation, axis=1)
    return panel[:, burn_in:]


def _boundary_change(panel: np.ndarray, boundaries: np.ndarray, width: int) -> np.ndarray:
    participant_changes = []
    for boundary in boundaries:
        pre = panel[:, boundary - width : boundary].mean(axis=1)
        centered = panel[:, boundary - width // 2 : boundary + width // 2].mean(axis=1)
        participant_changes.append(centered - pre)
    return np.mean(np.stack(participant_changes, axis=1), axis=1)


def _difference(values: np.ndarray, labels: np.ndarray) -> float:
    return float(values[labels].mean() - values[~labels].mean())


def _label_p(
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float, float]:
    observed = _difference(values, labels)
    target_n = int(labels.sum())
    control_n = values.size - target_n
    keys = rng.random((permutations, values.size))
    selected = np.argpartition(keys, target_n - 1, axis=1)[:, :target_n]
    selected_sum = values[selected].sum(axis=1)
    null = selected_sum / target_n - (values.sum() - selected_sum) / control_n
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (permutations + 1)
    return observed, float(p_value)


def one_dual_null_simulation(
    rng: np.random.Generator,
    *,
    target_n: int,
    control_n: int,
    timepoints: int,
    boundaries: np.ndarray,
    width: int,
    rho: float,
    global_signal: float,
    common_boundary_signal: float,
    interaction_signal: float,
    smooth_nonstationarity: float,
    label_permutations: int,
    temporal_shifts: int,
) -> DualNullStatistics:
    """Generate one panel and apply global, label, and temporal tests."""

    participant_n = target_n + control_n
    target = np.zeros(participant_n, dtype=bool)
    target[:target_n] = True
    panel = _ar1_panel(rng, participant_n, timepoints, rho)
    panel[target] += float(global_signal)
    if smooth_nonstationarity:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        period = rng.uniform(timepoints / 2.5, timepoints * 1.5)
        time = np.arange(timepoints)
        nuisance = np.sin(2.0 * np.pi * time / period + phase)
        nuisance -= nuisance.mean()
        panel[target] += float(smooth_nonstationarity) * nuisance
    for boundary in boundaries:
        start = boundary - width // 2
        stop = boundary + width // 2
        panel[:, start:stop] += float(common_boundary_signal)
        panel[target, start:stop] += float(interaction_signal)

    global_effect, p_global = _label_p(panel.mean(axis=1), target, rng, label_permutations)
    participant_change = _boundary_change(panel, boundaries, width)
    boundary_effect, p_boundary_label = _label_p(
        participant_change, target, rng, label_permutations
    )

    possible = np.arange(1, timepoints, dtype=int)
    shifts = rng.choice(possible, size=temporal_shifts, replace=temporal_shifts > possible.size)
    group_curve = panel[target].mean(axis=0) - panel[~target].mean(axis=0)
    extended = np.concatenate([group_curve, group_curve[: width - 1]])
    cumulative = np.concatenate([[0.0], np.cumsum(extended)])
    rolling = (cumulative[width:] - cumulative[:-width]) / width
    centered_start = boundaries - width // 2
    pre_start = boundaries - width
    temporal_null = np.mean(
        rolling[(centered_start[None, :] + shifts[:, None]) % timepoints]
        - rolling[(pre_start[None, :] + shifts[:, None]) % timepoints],
        axis=1,
    )
    p_temporal = (1 + int(np.sum(np.abs(temporal_null) >= abs(boundary_effect)))) / (
        temporal_shifts + 1
    )
    p_iut = intersection_union_p(p_boundary_label, p_temporal)
    return DualNullStatistics(
        global_effect=global_effect,
        boundary_effect=boundary_effect,
        p_global_label=p_global,
        p_boundary_label=p_boundary_label,
        p_boundary_temporal=float(p_temporal),
        p_boundary_iut=p_iut,
    )


def simulate_dual_null_recovery(config: dict) -> pd.DataFrame:
    """Run the frozen four-state recovery design across prespecified scenarios."""

    seed = int(config.get("seed", 20260820))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    baseline = config["baseline"]
    scenarios = config.get("scenarios", [{"name": "baseline"}])
    for scenario_override in scenarios:
        scenario = {**baseline, **scenario_override}
        scenario_name = str(scenario.pop("name"))
        boundaries = np.asarray(scenario["boundaries"], dtype=int)
        states = {
            "neither": (0.0, 0.0, 0.0),
            "global_group_only": (float(scenario["global_signal"]), 0.0, 0.0),
            "common_boundary_only": (0.0, float(scenario["common_boundary_signal"]), 0.0),
            "group_boundary_interaction": (0.0, float(scenario["common_boundary_signal"]), float(scenario["interaction_signal"])),
        }
        for state, (global_signal, common_boundary_signal, interaction_signal) in states.items():
            for replicate in range(int(config["replicates"])):
                result = one_dual_null_simulation(
                    rng,
                    target_n=int(scenario["target_n"]),
                    control_n=int(scenario["control_n"]),
                    timepoints=int(scenario["timepoints"]),
                    boundaries=boundaries,
                    width=int(scenario["window_samples"]),
                    rho=float(scenario["ar1"]),
                    global_signal=global_signal,
                    common_boundary_signal=common_boundary_signal,
                    interaction_signal=interaction_signal,
                    smooth_nonstationarity=float(scenario.get("smooth_nonstationarity", 0.0)),
                    label_permutations=int(config["label_permutations"]),
                    temporal_shifts=int(config["temporal_shifts"]),
                )
                rows.append(
                    {
                        "scenario": scenario_name,
                        "state": state,
                        "replicate": replicate,
                        "truth_global_group": bool(global_signal),
                        "truth_common_boundary": bool(common_boundary_signal),
                        "truth_group_boundary_interaction": bool(interaction_signal),
                        "target_n": int(scenario["target_n"]),
                        "control_n": int(scenario["control_n"]),
                        "boundary_n": int(boundaries.size),
                        "ar1": float(scenario["ar1"]),
                        "smooth_nonstationarity": float(scenario.get("smooth_nonstationarity", 0.0)),
                        "global_effect": result.global_effect,
                        "boundary_effect": result.boundary_effect,
                        "p_global_label": result.p_global_label,
                        "p_boundary_label": result.p_boundary_label,
                        "p_boundary_temporal": result.p_boundary_temporal,
                        "p_boundary_iut": result.p_boundary_iut,
                        "detect_global": result.p_global_label <= float(config["alpha"]),
                        "detect_boundary": result.p_boundary_iut <= float(config["alpha"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_dual_null_recovery(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize state recovery without hiding either component test."""

    summary = (
        results.groupby(
            [
                "scenario",
                "state",
                "truth_global_group",
                "truth_common_boundary",
                "truth_group_boundary_interaction",
            ],
            observed=True,
        )
        .agg(
            replicates=("replicate", "size"),
            global_detection_rate=("detect_global", "mean"),
            boundary_iut_detection_rate=("detect_boundary", "mean"),
            boundary_label_detection_rate=("p_boundary_label", lambda x: np.mean(x <= 0.05)),
            boundary_temporal_detection_rate=("p_boundary_temporal", lambda x: np.mean(x <= 0.05)),
            median_global_effect=("global_effect", "median"),
            median_boundary_effect=("boundary_effect", "median"),
        )
        .reset_index()
    )
    z = 1.959963984540054
    for prefix in (
        "global_detection_rate",
        "boundary_iut_detection_rate",
        "boundary_label_detection_rate",
        "boundary_temporal_detection_rate",
    ):
        probability = summary[prefix].to_numpy(float)
        n = summary["replicates"].to_numpy(float)
        denominator = 1.0 + z * z / n
        center = (probability + z * z / (2.0 * n)) / denominator
        margin = z * np.sqrt(
            probability * (1.0 - probability) / n + z * z / (4.0 * n * n)
        ) / denominator
        summary[f"{prefix}_ci_low"] = center - margin
        summary[f"{prefix}_ci_high"] = center + margin
    return summary
