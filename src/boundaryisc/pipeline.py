"""End-to-end analysis and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from .core import CHANNEL_MAP, compute_isc, window_slice
from .inference import (
    contrast_from_pairwise,
    hierarchical_bayesian_bootstrap,
    label_permutation_test,
    pairwise_boundary_matrices,
    pseudo_boundary_test,
)
from .io import load_commercial_cache
from .simulation import simulate_recovery
from .specification import run_specification_grid


def _paths(config: dict) -> tuple[Path, Path, Path]:
    root = Path(config["project_root"])
    tables = root / "results" / "tables"
    figures = root / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return root, tables, figures


def _json_dump(path: Path, payload: dict) -> None:
    def converter(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(payload, indent=2, default=converter) + "\n", encoding="utf-8")


def _participant_scores_from_matrices(matrices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.full((labels.size, matrices.shape[0]), np.nan, dtype=float)
    for bi, matrix in enumerate(matrices):
        for group in np.unique(labels):
            idx = np.flatnonzero(labels == group)
            block = matrix[np.ix_(idx, idx)].astype(float)
            np.fill_diagonal(block, np.nan)
            scores[idx, bi] = np.nanmean(block, axis=1)
    return scores


def _anova_and_paired(scores: pd.DataFrame, scale: str) -> list[dict]:
    subset = scores[(scores["scale"] == scale) & (scores["channel"] == "composite")]
    wide = subset.pivot(index=["participant_id", "group"], columns="phase", values="isc").reset_index()
    groups = [part["boundary"].to_numpy() for _, part in wide.groupby("group", sort=False)]
    group_f, group_p = stats.f_oneway(*groups)
    changes = [
        (part["boundary"] - part["pre"]).to_numpy()
        for _, part in wide.groupby("group", sort=False)
    ]
    interaction_f, interaction_p = stats.f_oneway(*changes)
    results = [
        {"scale": scale, "test": "boundary_group_anova", "group": "all", "statistic": group_f, "p_value": group_p},
        {"scale": scale, "test": "time_by_group_via_change_anova", "group": "all", "statistic": interaction_f, "p_value": interaction_p},
    ]
    for group, part in wide.groupby("group", sort=False):
        test = stats.ttest_rel(part["boundary"], part["pre"])
        difference = (part["boundary"] - part["pre"]).to_numpy()
        dz = float(np.mean(difference) / np.std(difference, ddof=1))
        results.append(
            {
                "scale": scale,
                "test": "paired_boundary_vs_pre",
                "group": group,
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohen_dz": dz,
            }
        )
    return results


def run_reproduction(config: dict) -> None:
    _, tables, _ = _paths(config)
    cache = load_commercial_cache(config)
    data = cache["data"].astype(float)
    labels = cache["group"].astype(str)
    participant_ids = cache["participant_id"].astype(int)
    boundaries = config["boundaries_10hz"][: config["usable_boundary_count"]]
    rows: list[dict] = []
    channel_defs = ["x", "y", "pupil", "gaze", "composite"]
    for scale in ("raw", "legacy"):
        for channel in channel_defs:
            phase_accumulator = {phase: [] for phase in ("pre", "boundary", "post")}
            for boundary in boundaries:
                for phase in phase_accumulator:
                    sl = window_slice(boundary, 10, config["analysis_rate_hz"], phase)
                    phase_accumulator[phase].append(
                        compute_isc(
                            data[:, sl, :],
                            labels,
                            estimator="pairwise",
                            reference="within",
                            channel=channel,
                            transform=scale,
                        )
                    )
            for phase, values in phase_accumulator.items():
                averaged = np.nanmean(np.stack(values, axis=1), axis=1)
                for pid, group, value in zip(participant_ids, labels, averaged, strict=True):
                    rows.append(
                        {
                            "participant_id": int(pid),
                            "group": str(group),
                            "phase": phase,
                            "channel": channel,
                            "scale": scale,
                            "isc": float(value),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "faithful_reproduction_participant.csv", index=False)
    summary = (
        frame.groupby(["scale", "channel", "group", "phase"], sort=False)["isc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(tables / "faithful_reproduction_summary.csv", index=False)
    tests = pd.DataFrame(_anova_and_paired(frame, "raw") + _anova_and_paired(frame, "legacy"))
    reported = {
        ("boundary_group_anova", "all"): (4.145, 0.008),
        ("time_by_group_via_change_anova", "all"): (88.06, 0.001),
        ("paired_boundary_vs_pre", "Long"): (8.29, 0.001),
        ("paired_boundary_vs_pre", "Schema"): (5.26, 0.001),
        ("paired_boundary_vs_pre", "Short-P"): (13.52, 0.001),
        ("paired_boundary_vs_pre", "Short-R"): (-8.13, 0.001),
    }
    tests["reported_statistic"] = [reported.get((row.test, row.group), (np.nan, np.nan))[0] for row in tests.itertuples()]
    tests["reported_p_or_upper_bound"] = [reported.get((row.test, row.group), (np.nan, np.nan))[1] for row in tests.itertuples()]
    tests["absolute_statistic_difference"] = (tests["statistic"] - tests["reported_statistic"]).abs()
    tests.to_csv(tables / "published_statistic_comparison.csv", index=False)

    identity = summary.pivot_table(
        index=["channel", "group", "phase"], columns="scale", values="mean"
    ).reset_index()
    identity["legacy_expected_from_raw"] = (1 + identity["raw"]) / 2
    identity["identity_error"] = identity["legacy"] - identity["legacy_expected_from_raw"]
    identity.to_csv(tables / "legacy_scale_identity.csv", index=False)


def _pairwise_group_tests(
    difference: np.ndarray,
    labels: np.ndarray,
    target: str,
    controls: list[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    matrix = np.nanmean(difference, axis=0)
    rows = []
    for control in controls:
        idx_target = np.flatnonzero(labels == target)
        idx_control = np.flatnonzero(labels == control)
        observed = float(
            np.nanmean(matrix[np.ix_(idx_target, idx_target)][np.triu_indices(idx_target.size, 1)])
            - np.nanmean(matrix[np.ix_(idx_control, idx_control)][np.triu_indices(idx_control.size, 1)])
        )
        pooled = np.concatenate([idx_target, idx_control])
        null = np.empty(n_permutations)
        for draw in range(n_permutations):
            perm = rng.permutation(pooled)
            a = perm[: idx_target.size]
            b = perm[idx_target.size :]
            ma = matrix[np.ix_(a, a)][np.triu_indices(a.size, 1)]
            mb = matrix[np.ix_(b, b)][np.triu_indices(b.size, 1)]
            null[draw] = np.nanmean(ma) - np.nanmean(mb)
        p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_permutations + 1)
        rows.append({"comparison": f"{target} - {control}", "effect": observed, "p_value": p})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = multipletests(frame["p_value"], method="holm")[1]
    return frame


def _memory_associations(
    response: np.ndarray,
    cache: dict[str, np.ndarray],
    tables: Path,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "participant_id": cache["participant_id"].astype(int),
            "group": cache["group"].astype(str),
            "boundary_response": response,
            "remember": cache["remember"].astype(float),
            "detail": cache["detail"].astype(float),
        }
    )
    frame.to_csv(tables / "participant_boundary_response.csv", index=False)
    rows = []
    for outcome in ("remember", "detail"):
        analysis_frame = frame.dropna(subset=[outcome, "boundary_response"]).copy()
        fitted = ols(f"{outcome} ~ boundary_response + C(group)", data=analysis_frame).fit(cov_type="HC3")
        ci = fitted.conf_int().loc["boundary_response"]
        rows.append(
            {
                "outcome": outcome,
                "coefficient": fitted.params["boundary_response"],
                "se_hc3": fitted.bse["boundary_response"],
                "ci_low": ci.iloc[0],
                "ci_high": ci.iloc[1],
                "p_value": fitted.pvalues["boundary_response"],
                "n": int(fitted.nobs),
                "model": f"{outcome} ~ boundary_response + group; HC3",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(tables / "memory_associations.csv", index=False)
    return result


def run_analysis(config: dict) -> None:
    _, tables, _ = _paths(config)
    cache = load_commercial_cache(config)
    data = cache["data"].astype(float)
    labels = cache["group"].astype(str)
    boundaries = config["boundaries_10hz"][: config["usable_boundary_count"]]
    group_order = config["group_order"]
    target = config["primary"]["target_group"]
    boundary_mats, pre_mats = pairwise_boundary_matrices(
        data,
        boundaries,
        width_seconds=config["primary"]["window_seconds"],
        rate_hz=config["analysis_rate_hz"],
        channel=config["primary"]["channel"],
        transform=config["primary"]["transform"],
    )
    difference = boundary_mats - pre_mats
    permutation = label_permutation_test(
        difference,
        labels,
        target=target,
        group_order=group_order,
        n_permutations=config["n_permutations"],
        seed=config["seed"],
    )
    bootstrap = hierarchical_bayesian_bootstrap(
        difference,
        labels,
        target=target,
        group_order=group_order,
        n_bootstrap=config["n_bootstrap"],
        seed=config["seed"] + 1,
    )
    pd.DataFrame({"permuted_effect": permutation.pop("null")}).to_csv(
        tables / "label_permutation_null.csv", index=False
    )
    pd.DataFrame({"bootstrap_effect": bootstrap.pop("draws")}).to_csv(
        tables / "hierarchical_bootstrap_draws.csv", index=False
    )
    primary = {**permutation, **bootstrap}
    _json_dump(tables / "primary_inference.json", primary)

    specs = run_specification_grid(
        data,
        labels,
        boundaries,
        grid=config["specification_grid"],
        rate_hz=config["analysis_rate_hz"],
        target_group=target,
    )
    specs.to_csv(tables / "specification_grid.csv", index=False)
    spec_summary = {
        "n_specifications": int(len(specs)),
        "negative_direction_count": int((specs["effect"] < 0).sum()),
        "negative_direction_proportion": float((specs["effect"] < 0).mean()),
        "effect_min": float(specs["effect"].min()),
        "effect_median": float(specs["effect"].median()),
        "effect_max": float(specs["effect"].max()),
    }
    _json_dump(tables / "specification_summary.json", spec_summary)

    pseudo = pseudo_boundary_test(
        data,
        labels,
        boundaries,
        target_group=target,
        group_order=group_order,
        width_seconds=config["primary"]["window_seconds"],
        rate_hz=config["analysis_rate_hz"],
        channel=config["primary"]["channel"],
        n_shifts=config["n_pseudo"],
        seed=config["seed"] + 2,
    )
    pd.DataFrame({"shift": pseudo.pop("shifts"), "pseudo_effect": pseudo.pop("null")}).to_csv(
        tables / "pseudo_boundary_null.csv", index=False
    )
    _json_dump(tables / "pseudo_boundary_test.json", pseudo)

    controls = [g for g in group_order if g != target]
    group_tests = _pairwise_group_tests(
        difference, labels, target, controls, config["n_permutations"], config["seed"] + 3
    )
    group_tests.to_csv(tables / "secondary_group_contrasts.csv", index=False)

    participant_boundary = _participant_scores_from_matrices(boundary_mats, labels)
    participant_pre = _participant_scores_from_matrices(pre_mats, labels)
    response = np.nanmean(participant_boundary - participant_pre, axis=1)
    _memory_associations(response, cache, tables)

    source_rows = []
    for bi, boundary in enumerate(boundaries, start=1):
        for group in group_order:
            idx = np.flatnonzero(labels == group)
            block = difference[bi - 1][np.ix_(idx, idx)]
            source_rows.append(
                {
                    "boundary_number": bi,
                    "boundary_sample_10hz": boundary,
                    "group": group,
                    "pairwise_change_r": float(np.nanmean(block[np.triu_indices(idx.size, 1)])),
                    "n_participants": idx.size,
                }
            )
    pd.DataFrame(source_rows).to_csv(tables / "primary_boundary_source_data.csv", index=False)


def run_simulation(config: dict) -> None:
    root, tables, _ = _paths(config)
    simulation_config = json.loads((root / "config" / "simulation.json").read_text(encoding="utf-8"))
    frame = simulate_recovery(simulation_config)
    frame.to_csv(tables / "simulation_replicates.csv", index=False)
    summary = (
        frame.groupby(["ar1", "group_size", "shared_signal", "affected_channel"], sort=False)
        .agg(
            mean_effect=("composite_effect", "mean"),
            truth=("truth_composite", "first"),
            bias=("composite_bias", "mean"),
            rmse=("composite_bias", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            rejection_rate=("reject_two_sided_0_05", "mean"),
            interval_coverage=("ci_covers_truth", "mean"),
            mean_x_effect=("x_effect", "mean"),
            replicates=("replicate", "count"),
        )
        .reset_index()
    )
    summary["operating_characteristic"] = np.where(summary["shared_signal"] == 0, "type_I_error", "power")
    summary["composite_to_x_ratio"] = summary["mean_effect"] / summary["mean_x_effect"].replace(0, np.nan)
    summary.to_csv(tables / "simulation_summary.csv", index=False)


def build_static_report(config: dict) -> None:
    root, tables, figures = _paths(config)
    from .figures import generate_all_figures

    generate_all_figures(root)
    primary = json.loads((tables / "primary_inference.json").read_text(encoding="utf-8"))
    pseudo = json.loads((tables / "pseudo_boundary_test.json").read_text(encoding="utf-8"))
    specs = json.loads((tables / "specification_summary.json").read_text(encoding="utf-8"))
    memory = pd.read_csv(tables / "memory_associations.csv")
    lines = [
        "# Static analysis report",
        "",
        "This report was generated from the frozen local configuration. It is not a preregistration record.",
        "",
        "## Primary contrast",
        "",
        f"Observed difference in boundary response: {primary['observed']:.6f} Pearson-r units.",
        f"Two-sided fixed-size label-permutation p = {primary['p_value']:.6g} ({primary['n_permutations']:,} permutations).",
        f"Two-level Bayesian bootstrap 95% interval: [{primary['ci_low']:.6f}, {primary['ci_high']:.6f}] ({primary['n_bootstrap']:,} draws).",
        "",
        "## Specification grid",
        "",
        f"{specs['negative_direction_count']} of {specs['n_specifications']} prespecified estimates were negative ({specs['negative_direction_proportion']:.1%}).",
        f"Effects ranged from {specs['effect_min']:.6f} to {specs['effect_max']:.6f}; median {specs['effect_median']:.6f}.",
        "",
        "## Pseudo-boundaries",
        "",
        f"The common-circular-shift test gave p = {pseudo['p_value']:.6g} from {pseudo['n_shifts']:,} pseudo-boundary sets.",
        "",
        "## Memory associations",
        "",
    ]
    for row in memory.itertuples():
        lines.append(
            f"{row.outcome}: b = {row.coefficient:.6f}, HC3 95% CI [{row.ci_low:.6f}, {row.ci_high:.6f}], p = {row.p_value:.6g}, N = {row.n}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The estimates quantify synchronization of deposited eye-position and pupil traces. They do not, by themselves, identify event segmentation, reward, pleasure, or a neural mechanism.",
        ]
    )
    (root / "results" / "static_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
