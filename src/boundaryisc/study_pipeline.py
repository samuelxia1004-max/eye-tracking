"""Frozen two-dataset pipeline for the BBBD-to-Sherlock methods study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from .attention import (
    FEATURE_COLUMNS,
    build_attention_features,
    nested_attention_benchmark_from_records,
)
from .attention_inference import (
    FrozenOcularModel,
    boundary_location_test,
    boundary_probability_contrast,
    fit_frozen_ocular_model,
    fixed_size_group_permutation,
    group_weighted_bootstrap,
    participant_boundary_bootstrap,
    participant_signflip_test,
    transported_probability_curve,
    two_way_auc_bootstrap,
)
from .bbbd import load_bbbd_ocular_archives, save_bbbd_ocular_cache
from .bbbd_eeg import cache_bbbd_eeg, cross_validated_corrca_scores
from .crossed_null import crossed_null_decision
from .dual_null_simulation import simulate_dual_null_recovery, summarize_dual_null_recovery
from .specification import run_specification_grid


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_ready) + "\n", encoding="utf-8")


def _directories(project: Path) -> tuple[Path, Path]:
    tables = project / "results" / "tables"
    derived = project / "data" / "derived"
    tables.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    return tables, derived


def prepare_bbbd_cache(config: dict, project: Path) -> Path:
    """Build the outcome-neutral 10-Hz ocular cache from verified archives."""

    _, derived = _directories(project)
    destination = derived / "bbbd_ocular_10hz.npz"
    if destination.exists():
        return destination
    archives = [
        (int(experiment), Path(path))
        for experiment, path in config["inputs"]["bbbd_archives"].items()
    ]
    cache, audit = load_bbbd_ocular_archives(archives)
    save_bbbd_ocular_cache(cache, destination)
    audit.to_csv(project / "provenance" / "bbbd_loaded_record_audit.csv", index=False)
    return destination


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def _calibration(predictions: pd.DataFrame) -> dict[str, float]:
    probability = np.clip(predictions["probability_attentive"].to_numpy(float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))
    design = sm.add_constant(logit)
    fitted = sm.GLM(
        predictions["truth_attentive"].to_numpy(int),
        design,
        family=sm.families.Binomial(),
    ).fit(cov_type="HC3")
    return {
        "calibration_intercept": float(fitted.params[0]),
        "calibration_slope": float(fitted.params[1]),
        "intercept_se_hc3": float(fitted.bse[0]),
        "slope_se_hc3": float(fitted.bse[1]),
    }


def _serialize_model(model: FrozenOcularModel) -> dict[str, object]:
    return {
        "feature_columns": list(model.feature_columns),
        "scaler_mean": model.scaler_mean,
        "scaler_scale": model.scaler_scale,
        "coefficient": model.coefficient,
        "intercept": model.intercept,
        "c_value": model.c_value,
    }


def load_frozen_model(path: Path) -> FrozenOcularModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FrozenOcularModel(
        feature_columns=tuple(payload["feature_columns"]),
        scaler_mean=np.asarray(payload["scaler_mean"], dtype=float),
        scaler_scale=np.asarray(payload["scaler_scale"], dtype=float),
        coefficient=np.asarray(payload["coefficient"], dtype=float),
        intercept=float(payload["intercept"]),
        c_value=float(payload["c_value"]),
    )


def run_bbbd_benchmark(config: dict, project: Path) -> dict[str, object]:
    """Run crossed outer validation, freeze the gate, and fit final parameters."""

    tables, _ = _directories(project)
    cache_path = prepare_bbbd_cache(config, project)
    cache = _load_npz(cache_path)
    data = cache["data"].astype(float)
    participant = cache["participant"].astype(str)
    video = cache["video"].astype(str)
    condition = cache["condition"].astype(str)
    validity = cache["original_valid_fraction"].astype(float)
    quality_features = build_attention_features(
        data,
        participant,
        video,
        condition,
        rate_hz=int(config["rate_hz"]),
        window_seconds=float(config["window_seconds"]),
        min_valid_fraction=float(config["bbbd"]["primary_min_original_valid_fraction"]),
        validity_records=validity,
    )
    primary_columns = tuple(config["primary_feature_columns"])
    quality_features["record_quality_complete"] = np.isfinite(
        quality_features.loc[:, primary_columns].to_numpy(float)
    ).all(axis=1)
    pair_complete = quality_features.groupby(["participant", "video"])[
        "record_quality_complete"
    ].transform("all")
    quality_features["pair_quality_complete"] = pair_complete
    quality_features.to_csv(tables / "bbbd_quality_audit.csv", index=False)
    keep_index = quality_features.loc[pair_complete, "record_index"].to_numpy(int)
    excluded = quality_features.loc[
        ~pair_complete,
        ["record_index", "participant", "video", "condition", *primary_columns],
    ]
    excluded.to_csv(tables / "bbbd_quality_exclusions.csv", index=False)
    data = data[keep_index]
    participant = participant[keep_index]
    video = video[keep_index]
    condition = condition[keep_index]
    validity = validity[keep_index]
    quality_summary = {
        "input_records": int(len(quality_features)),
        "eligible_records": int(len(keep_index)),
        "excluded_records": int(len(excluded)),
        "input_pairs": int(quality_features.groupby(["participant", "video"]).ngroups),
        "eligible_pairs": int(
            quality_features.loc[pair_complete].groupby(["participant", "video"]).ngroups
        ),
        "excluded_pairs": int(
            quality_features.loc[~pair_complete].groupby(["participant", "video"]).ngroups
        ),
        "eligible_participants": int(np.unique(participant).size),
        "rule": "exclude a paired participant-video unit when either session lacks any primary channel mean at the frozen original-valid window threshold",
    }
    _write_json(tables / "bbbd_quality_summary.json", quality_summary)
    benchmark = nested_attention_benchmark_from_records(
        data,
        participant,
        video,
        condition,
        rate_hz=int(config["rate_hz"]),
        window_seconds=float(config["window_seconds"]),
        min_valid_fraction=float(config["bbbd"]["primary_min_original_valid_fraction"]),
        feature_columns=primary_columns,
        c_grid=tuple(config["bbbd"]["regularization_grid"]),
        participant_folds=int(config["bbbd"]["outer_cv_participant_folds"]),
        seed=int(config["seed"]),
        validity_records=validity,
    )
    benchmark.predictions.to_csv(tables / "bbbd_outer_predictions.csv", index=False)
    benchmark.fold_metrics.to_csv(tables / "bbbd_outer_fold_metrics.csv", index=False)
    benchmark.selected_c.to_csv(tables / "bbbd_selected_c.csv", index=False)
    signflip = participant_signflip_test(
        benchmark.predictions,
        n_permutations=int(config["inference"]["label_permutations"]),
        seed=int(config["seed"]) + 11,
    )
    pd.DataFrame({"null_paired_probability_difference": signflip.pop("null")}).to_csv(
        tables / "bbbd_participant_signflip_null.csv", index=False
    )
    signflip.pop("paired_video_records").to_csv(tables / "bbbd_paired_predictions.csv", index=False)
    signflip.pop("participant_differences").to_csv(
        tables / "bbbd_participant_prediction_differences.csv", index=False
    )
    auc_bootstrap = two_way_auc_bootstrap(
        benchmark.predictions,
        n_bootstrap=int(config["inference"]["bootstrap_draws"]),
        seed=int(config["seed"]) + 12,
    )
    pd.DataFrame({"auc": auc_bootstrap.pop("draws")}).to_csv(
        tables / "bbbd_auc_bootstrap.csv", index=False
    )
    calibration = _calibration(benchmark.predictions)
    gate_passed = bool(signflip["p_value"] <= 0.05 and auc_bootstrap["ci_low"] > 0.5)
    validation = {
        **benchmark.aggregate_metrics,
        **signflip,
        **auc_bootstrap,
        **calibration,
        "gate_rule": "participant sign-flip p <= .05 and two-way bootstrap 95% AUC interval excludes .50",
        "gate_passed": gate_passed,
        "order_repetition_confounded": True,
    }
    _write_json(tables / "bbbd_validation.json", validation)

    full_features = build_attention_features(
        data,
        participant,
        video,
        condition,
        rate_hz=int(config["rate_hz"]),
        window_seconds=float(config["window_seconds"]),
        min_valid_fraction=float(config["bbbd"]["primary_min_original_valid_fraction"]),
        validity_records=validity,
    )
    full_features.to_csv(tables / "bbbd_full_features.csv", index=False)
    selection_counts = benchmark.selected_c["selected_c"].value_counts()
    highest = selection_counts.max()
    selected_c = float(min(selection_counts[selection_counts == highest].index))
    gate_record = {"gate_passed": gate_passed, "selected_c": selected_c}
    if gate_passed:
        model = fit_frozen_ocular_model(
            full_features,
            c_value=selected_c,
        feature_columns=primary_columns,
            seed=int(config["seed"]),
        )
        _write_json(tables / "frozen_ocular_model.json", _serialize_model(model))
        gate_record["model_path"] = str(tables / "frozen_ocular_model.json")
    else:
        gate_record["model_path"] = None
        gate_record["transport_skipped"] = True
    _write_json(tables / "bbbd_transport_gate.json", gate_record)
    return gate_record


def run_bbbd_threshold_sensitivity(config: dict, project: Path) -> dict[str, object]:
    """Evaluate the planned original-valid thresholds without changing transport.

    The primary .50 result is read from the frozen benchmark outputs.  Other
    thresholds repeat the same pair-completeness rule and crossed validation,
    but they do not refit or replace the transported Sherlock model.
    """

    tables, _ = _directories(project)
    cache = _load_npz(prepare_bbbd_cache(config, project))
    all_data = cache["data"].astype(float)
    all_participant = cache["participant"].astype(str)
    all_video = cache["video"].astype(str)
    all_condition = cache["condition"].astype(str)
    all_validity = cache["original_valid_fraction"].astype(float)
    primary_columns = tuple(config["primary_feature_columns"])
    primary_threshold = float(config["bbbd"]["primary_min_original_valid_fraction"])
    thresholds = sorted(
        set(
            [primary_threshold]
            + [float(value) for value in config["bbbd"]["sensitivity_original_valid_fractions"]]
        )
    )
    primary_validation = json.loads((tables / "bbbd_validation.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        quality = build_attention_features(
            all_data,
            all_participant,
            all_video,
            all_condition,
            rate_hz=int(config["rate_hz"]),
            window_seconds=float(config["window_seconds"]),
            min_valid_fraction=threshold,
            validity_records=all_validity,
        )
        quality["record_quality_complete"] = np.isfinite(
            quality.loc[:, primary_columns].to_numpy(float)
        ).all(axis=1)
        pair_complete = quality.groupby(["participant", "video"])[
            "record_quality_complete"
        ].transform("all")
        keep = quality.loc[pair_complete, "record_index"].to_numpy(int)
        base = {
            "min_original_valid_fraction": threshold,
            "eligible_records": int(len(keep)),
            "eligible_pairs": int(
                quality.loc[pair_complete].groupby(["participant", "video"]).ngroups
            ),
            "eligible_participants": int(np.unique(all_participant[keep]).size),
            "is_primary": bool(np.isclose(threshold, primary_threshold)),
        }
        if np.isclose(threshold, primary_threshold):
            rows.append(
                {
                    **base,
                    "status": "primary_frozen_result",
                    "auc": primary_validation["auc"],
                    "auc_ci_low": primary_validation["ci_low"],
                    "auc_ci_high": primary_validation["ci_high"],
                    "balanced_accuracy": primary_validation["balanced_accuracy"],
                    "brier": primary_validation["brier"],
                    "participant_signflip_p": primary_validation["p_value"],
                }
            )
            continue
        if len(keep) == 0 or np.unique(all_participant[keep]).size < 2:
            rows.append({**base, "status": "insufficient_quality_complete_data"})
            continue
        try:
            benchmark = nested_attention_benchmark_from_records(
                all_data[keep],
                all_participant[keep],
                all_video[keep],
                all_condition[keep],
                rate_hz=int(config["rate_hz"]),
                window_seconds=float(config["window_seconds"]),
                min_valid_fraction=threshold,
                feature_columns=primary_columns,
                c_grid=tuple(config["bbbd"]["regularization_grid"]),
                participant_folds=int(config["bbbd"]["outer_cv_participant_folds"]),
                seed=int(config["seed"]),
                validity_records=all_validity[keep],
            )
            signflip = participant_signflip_test(
                benchmark.predictions,
                n_permutations=int(config["inference"]["label_permutations"]),
                seed=int(config["seed"]) + 11,
            )
            auc_bootstrap = two_way_auc_bootstrap(
                benchmark.predictions,
                n_bootstrap=int(config["inference"]["bootstrap_draws"]),
                seed=int(config["seed"]) + 12,
            )
            rows.append(
                {
                    **base,
                    "status": "completed",
                    "auc": benchmark.aggregate_metrics["auc"],
                    "auc_ci_low": auc_bootstrap["ci_low"],
                    "auc_ci_high": auc_bootstrap["ci_high"],
                    "balanced_accuracy": benchmark.aggregate_metrics["balanced_accuracy"],
                    "brier": benchmark.aggregate_metrics["brier"],
                    "participant_signflip_p": signflip["p_value"],
                }
            )
        except (ValueError, RuntimeError) as error:
            rows.append({**base, "status": "not_estimable", "error": str(error)})
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "bbbd_validity_threshold_sensitivity.csv", index=False)
    result = {
        "status": "completed",
        "thresholds": len(frame),
        "all_nonprimary_estimable": bool((frame.loc[~frame["is_primary"], "status"] == "completed").all()),
        "transport_model_changed": False,
    }
    _write_json(tables / "bbbd_validity_threshold_sensitivity.json", result)
    return result


def run_eeg_convergence(config: dict, project: Path) -> dict[str, object]:
    """Validate convergence of held-out ocular and EEG synchrony scores."""

    tables, derived = _directories(project)
    eeg_config = config["bbbd"]["eeg_validation"]
    if not eeg_config.get("enabled", False):
        result = {"status": "disabled_in_frozen_configuration"}
        _write_json(tables / "bbbd_eeg_convergence.json", result)
        return result
    archives = [
        (int(experiment), Path(path))
        for experiment, path in config["inputs"]["bbbd_archives"].items()
    ]
    eeg_manifest_path = project / "provenance" / "bbbd_eeg_cache_manifest.csv"
    if eeg_manifest_path.exists():
        eeg_manifest = pd.read_csv(eeg_manifest_path)
    else:
        eeg_manifest = cache_bbbd_eeg(
            archives,
            derived / "bbbd_eeg_16hz",
            output_rate_hz=int(eeg_config["rate_hz"]),
        )
        eeg_manifest.to_csv(eeg_manifest_path, index=False)
    score_path = tables / "bbbd_eeg_corrca_scores.csv"
    if score_path.exists():
        eeg_scores = pd.read_csv(score_path)
    else:
        eeg_scores = cross_validated_corrca_scores(
            eeg_manifest,
            participant_folds=int(config["bbbd"]["outer_cv_participant_folds"]),
            components=int(eeg_config["components"]),
            seed=int(config["seed"]),
        )
        eeg_scores.to_csv(score_path, index=False)
    ocular = pd.read_csv(tables / "bbbd_outer_predictions.csv")
    joined = ocular.merge(
        eeg_scores,
        on=["participant", "video", "condition"],
        how="inner",
        validate="one_to_one",
    )
    participant_codes = pd.Categorical(joined["participant"]).codes
    video_codes = pd.Categorical(joined["video"]).codes
    fitted = ols(
        "eeg_corrca_isc ~ probability_attentive + C(condition) + C(participant) + C(video)",
        data=joined,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": np.column_stack([participant_codes, video_codes])},
    )
    interval = fitted.conf_int().loc["probability_attentive"]
    wide = joined.pivot_table(
        index=["participant", "video"],
        columns="condition",
        values=["probability_attentive", "eeg_corrca_isc"],
        aggfunc="mean",
    ).dropna()
    ocular_change = (
        wide[("probability_attentive", "attentive")]
        - wide[("probability_attentive", "distracted")]
    )
    eeg_change = (
        wide[("eeg_corrca_isc", "attentive")] - wide[("eeg_corrca_isc", "distracted")]
    )
    correlation = stats.pearsonr(ocular_change, eeg_change)
    joined.to_csv(tables / "bbbd_ocular_eeg_joined.csv", index=False)
    result = {
        "status": "completed",
        "n_records": len(joined),
        "n_participants": joined["participant"].nunique(),
        "n_videos": joined["video"].nunique(),
        "condition_adjusted_fixed_effect_coefficient": fitted.params["probability_attentive"],
        "two_way_cluster_se": fitted.bse["probability_attentive"],
        "ci_low": interval.iloc[0],
        "ci_high": interval.iloc[1],
        "p_value": fitted.pvalues["probability_attentive"],
        "paired_change_correlation_r": correlation.statistic,
        "paired_change_correlation_p": correlation.pvalue,
        "interpretation": "convergent association; not an independent attention effect",
    }
    _write_json(tables / "bbbd_eeg_convergence.json", result)
    return result


def _secondary_pairwise(values: np.ndarray, groups: np.ndarray, target: str, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for control in [group for group in dict.fromkeys(groups.tolist()) if group != target]:
        selected = np.isin(groups, [target, control])
        local_values = values[selected]
        local_groups = groups[selected]
        observed = float(local_values[local_groups == target].mean() - local_values[local_groups == control].mean())
        target_n = int(np.sum(local_groups == target))
        null = np.empty(10000, dtype=float)
        for iteration in range(10000):
            indices = rng.permutation(local_values.size)
            null[iteration] = float(
                local_values[indices[:target_n]].mean() - local_values[indices[target_n:]].mean()
            )
        p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / 10001
        rows.append({"comparison": f"{target} - {control}", "effect": observed, "p_value": p_value})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = multipletests(frame["p_value"], method="holm")[1]
    return frame


def _memory_associations(change: np.ndarray, sherlock: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "participant_id": sherlock["participant_id"].astype(int),
            "group": sherlock["group"].astype(str),
            "transported_boundary_change": change,
            "remember": sherlock["remember"].astype(float),
            "detail": sherlock["detail"].astype(float),
        }
    )
    rows = []
    for outcome in ("remember", "detail"):
        analysis = frame.dropna(subset=[outcome]).copy()
        fitted = ols(
            f"{outcome} ~ transported_boundary_change + C(group)", data=analysis
        ).fit(cov_type="HC3")
        interval = fitted.conf_int().loc["transported_boundary_change"]
        rows.append(
            {
                "outcome": outcome,
                "coefficient": fitted.params["transported_boundary_change"],
                "se_hc3": fitted.bse["transported_boundary_change"],
                "ci_low": interval.iloc[0],
                "ci_high": interval.iloc[1],
                "p_value": fitted.pvalues["transported_boundary_change"],
                "n": int(fitted.nobs),
                "interpretation": "association",
            }
        )
    return pd.DataFrame(rows)


def run_sherlock_transport(config: dict, project: Path) -> dict[str, object]:
    """Apply the frozen BBBD model and run the two necessary boundary tests."""

    tables, _ = _directories(project)
    gate = json.loads((tables / "bbbd_transport_gate.json").read_text(encoding="utf-8"))
    if not gate["gate_passed"]:
        result = {"status": "skipped_by_frozen_bbbd_gate"}
        _write_json(tables / "sherlock_transport_status.json", result)
        return result
    model = load_frozen_model(tables / "frozen_ocular_model.json")
    sherlock = _load_npz(Path(config["inputs"]["sherlock_cache"]))
    data = sherlock["data"].astype(float)
    groups = sherlock["group"].astype(str)
    participants = sherlock["participant_id"].astype(str)
    target = config["sherlock"]["target_group"]
    feature_frame = build_attention_features(
        data,
        participants,
        np.repeat("sherlock", len(participants)),
        np.repeat("unlabeled", len(participants)),
        rate_hz=int(config["rate_hz"]),
        window_seconds=float(config["window_seconds"]),
        min_valid_fraction=1.0,
    )
    global_score = model.predict_probability(
        feature_frame.loc[:, model.feature_columns].to_numpy(float)
    )
    global_source = feature_frame.copy()
    global_source["group"] = groups
    global_source["transported_score"] = global_score
    global_source.to_csv(tables / "sherlock_global_scores.csv", index=False)
    global_permutation = fixed_size_group_permutation(
        global_score,
        groups,
        target=target,
        n_permutations=int(config["inference"]["label_permutations"]),
        seed=int(config["seed"]) + 21,
    )
    pd.DataFrame({"effect": global_permutation.pop("null")}).to_csv(
        tables / "sherlock_global_label_null.csv", index=False
    )
    global_bootstrap = group_weighted_bootstrap(
        global_score,
        groups,
        target=target,
        n_bootstrap=int(config["inference"]["bootstrap_draws"]),
        seed=int(config["seed"]) + 22,
    )
    pd.DataFrame({"effect": global_bootstrap.pop("draws")}).to_csv(
        tables / "sherlock_global_bootstrap.csv", index=False
    )
    global_result = {**global_permutation, **global_bootstrap}
    _write_json(tables / "sherlock_global_inference.json", global_result)

    width = int(round(float(config["window_seconds"]) * int(config["rate_hz"])))
    window_features, probability_curve = transported_probability_curve(data, model, width=width)
    np.savez_compressed(
        project / "data" / "derived" / "sherlock_transported_curve.npz",
        window_features=window_features.astype(np.float32),
        probability_curve=probability_curve.astype(np.float32),
    )
    boundaries = list(config["sherlock"]["boundaries_10hz"])
    observed, group_means, change = boundary_probability_contrast(
        probability_curve,
        groups,
        boundaries,
        width=width,
        target=target,
    )
    boundary_label = fixed_size_group_permutation(
        change,
        groups,
        target=target,
        n_permutations=int(config["inference"]["label_permutations"]),
        seed=int(config["seed"]) + 23,
    )
    pd.DataFrame({"effect": boundary_label.pop("null")}).to_csv(
        tables / "sherlock_boundary_label_null.csv", index=False
    )
    temporal = boundary_location_test(
        probability_curve,
        groups,
        boundaries,
        width=width,
        target=target,
        n_shifts=int(config["inference"]["temporal_shifts"]),
        seed=int(config["seed"]) + 24,
    )
    pd.DataFrame({"shift": temporal.pop("shifts"), "effect": temporal.pop("null")}).to_csv(
        tables / "sherlock_boundary_temporal_null.csv", index=False
    )
    bootstrap = participant_boundary_bootstrap(
        probability_curve,
        groups,
        boundaries,
        width=width,
        target=target,
        n_bootstrap=int(config["inference"]["bootstrap_draws"]),
        seed=int(config["seed"]) + 25,
    )
    pd.DataFrame({"effect": bootstrap.pop("draws")}).to_csv(
        tables / "sherlock_boundary_bootstrap.csv", index=False
    )
    decision = crossed_null_decision(
        boundary_label["p_value"], temporal["p_value"], alpha=config["inference"]["alpha"]
    )
    boundary_result = {
        "observed": observed,
        "group_means": group_means,
        "label_p_value": boundary_label["p_value"],
        "temporal_p_value": temporal["p_value"],
        "intersection_union_p": decision.p_iut,
        "reject_both": decision.reject_both,
        **bootstrap,
    }
    _write_json(tables / "sherlock_boundary_inference.json", boundary_result)
    pd.DataFrame(
        {
            "participant_id": sherlock["participant_id"].astype(int),
            "group": groups,
            "global_score": global_score,
            "boundary_change": change,
        }
    ).to_csv(tables / "sherlock_participant_transported_scores.csv", index=False)
    _secondary_pairwise(change, groups, target, int(config["seed"]) + 26).to_csv(
        tables / "sherlock_secondary_group_contrasts.csv", index=False
    )
    _memory_associations(change, sherlock).to_csv(
        tables / "sherlock_memory_associations.csv", index=False
    )

    raw_specs = run_specification_grid(
        data,
        groups,
        boundaries,
        grid=config["specification_grid"],
        rate_hz=int(config["rate_hz"]),
        target_group=target,
    )
    raw_specs["score_family"] = "raw_isc"
    learned_rows = []
    for window_seconds in config["sensitivity_windows_seconds"]:
        local_width = int(round(float(window_seconds) * int(config["rate_hz"])))
        _, local_probability = transported_probability_curve(data, model, width=local_width)
        effect, means, _ = boundary_probability_contrast(
            local_probability,
            groups,
            boundaries,
            width=local_width,
            target=target,
        )
        learned_rows.append(
            {
                "channel": "learned_ocular_score",
                "estimator": "loo",
                "transform": "fisher_then_logit",
                "window_seconds": window_seconds,
                "reference": "all",
                "effect": effect,
                "direction_matches_primary_hypothesis": effect < 0,
                "score_family": "transported",
                **{f"change_{group}": value for group, value in means.items()},
            }
        )
    specifications = pd.concat([raw_specs, pd.DataFrame(learned_rows)], ignore_index=True)
    specifications.to_csv(tables / "sherlock_specification_grid.csv", index=False)
    return {"global": global_result, "boundary": boundary_result}


def run_dual_null_simulations(config: dict, project: Path) -> dict[str, object]:
    tables, _ = _directories(project)
    simulation_config = json.loads(
        (project / "config" / "simulation.json").read_text(encoding="utf-8")
    )
    replicates = simulate_dual_null_recovery(simulation_config)
    summary = summarize_dual_null_recovery(replicates)
    replicates.to_csv(tables / "dual_null_simulation_replicates.csv", index=False)
    summary.to_csv(tables / "dual_null_simulation_summary.csv", index=False)
    return {"rows": len(replicates), "states": summary.to_dict(orient="records")}
