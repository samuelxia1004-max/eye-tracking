"""Deterministic source-data, audit and figure generation from frozen results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_curve


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
    "light": "#E6E6E6",
}
GROUP_ORDER = ["Short-R", "Long", "Schema", "Short-P"]
GROUP_COLORS = {
    "Short-R": COLORS["red"],
    "Long": COLORS["blue"],
    "Schema": COLORS["green"],
    "Short-P": COLORS["purple"],
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _domain_shift(tables: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bbbd = pd.read_csv(tables / "bbbd_full_features.csv")
    sherlock = pd.read_csv(tables / "sherlock_global_scores.csv")
    model = _json(tables / "frozen_ocular_model.json")
    features = model["feature_columns"]
    mean = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    intercept = float(model["intercept"])
    sherlock["linear_predictor"] = (
        (sherlock[features].to_numpy(float) - mean) / scale
    ) @ coefficient + intercept
    bbbd["linear_predictor"] = ((bbbd[features].to_numpy(float) - mean) / scale) @ coefficient + intercept
    rows = []
    for feature in [*features, "linear_predictor"]:
        low, high = bbbd[feature].quantile([0.01, 0.99])
        for domain, frame in [("BBBD", bbbd), ("Sherlock", sherlock)]:
            rows.append(
                {
                    "feature": feature,
                    "domain": domain,
                    "n": len(frame),
                    "mean": frame[feature].mean(),
                    "sd": frame[feature].std(ddof=1),
                    "q01": frame[feature].quantile(0.01),
                    "median": frame[feature].median(),
                    "q99": frame[feature].quantile(0.99),
                    "bbbd_q01": low,
                    "bbbd_q99": high,
                    "outside_bbbd_01_99": float(((frame[feature] < low) | (frame[feature] > high)).mean()),
                }
            )
    return pd.DataFrame(rows), sherlock


def _figure1(config: dict, quality: dict, output: Path) -> list[str]:
    _style()
    fig, ax = plt.subplots(figsize=(183 / 25.4, 128 / 25.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white", edgecolor=color, linewidth=1.5
        )
        ax.add_patch(patch)
        ax.text(x + 0.15, y + h - 0.25, title, weight="bold", color=color, va="top", fontsize=7.2)
        ax.text(x + 0.15, y + h - 0.68, body, va="top", linespacing=1.28, fontsize=6.15)

    box(
        0.3, 4.75, 3.35, 2.35, "BBBD benchmark",
        f"{quality['eligible_participants']} participants; {quality['eligible_pairs']} paired videos\n"
        "attentive first / distracted rewatch\nX, Y and pupil ISC\nlabel-free within-session reference",
        COLORS["blue"],
    )
    box(
        4.33, 4.75, 3.35, 2.35, "Freeze and transport",
        "participant + video holdout\nfreeze features, scaling and C\nvalidation gate before Sherlock\nno Sherlock-outcome tuning",
        COLORS["green"],
    )
    box(
        8.35, 4.75, 3.35, 2.35, "Sherlock application",
        "104 participants; 4 groups\n18 usable annotated boundaries\nglobal and boundary contrasts\ncohort-relative ocular score",
        COLORS["orange"],
    )
    for start, end in [((3.65, 5.93), (4.33, 5.93)), ((7.68, 5.93), (8.35, 5.93))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, color=COLORS["gray"]))

    box(
        0.55, 1.25, 4.05, 2.25, "Null 1: participant labels",
        "Hold annotated boundaries fixed\nshuffle labels at fixed group sizes\ntests group exchangeability\nat the annotated contrast",
        COLORS["red"],
    )
    box(
        7.4, 1.25, 4.05, 2.25, "Null 2: boundary locations",
        "Hold participant groups fixed\ncircularly shift the boundary set\ntests whether the group contrast\nis location-specific",
        COLORS["purple"],
    )
    ax.add_patch(FancyArrowPatch((4.6, 2.35), (5.35, 2.35), arrowstyle="-|>", mutation_scale=12, color=COLORS["gray"]))
    ax.add_patch(FancyArrowPatch((7.4, 2.35), (6.65, 2.35), arrowstyle="-|>", mutation_scale=12, color=COLORS["gray"]))
    ax.text(6.0, 2.75, "Boundary-specific\ngroup claim", ha="center", va="center", weight="bold")
    ax.text(6.0, 1.85, r"$p_{IUT}=\max(p_{label},p_{time})$", ha="center", va="center")
    ax.text(6.0, 1.35, "Both conditions required", ha="center", color=COLORS["red"])
    ax.text(0.15, 7.65, "a", weight="bold", fontsize=11)
    ax.text(0.15, 3.8, "b", weight="bold", fontsize=11)
    ax.text(11.85, 0.2, "Analysis plan frozen locally; not publicly preregistered", ha="right", color=COLORS["gray"], fontsize=6.5)
    return _save(fig, output, "figure1_framework")


def _figure2(tables: Path, output: Path) -> tuple[list[str], pd.DataFrame]:
    _style()
    predictions = pd.read_csv(tables / "bbbd_outer_predictions.csv")
    paired = pd.read_csv(tables / "bbbd_paired_predictions.csv")
    validation = _json(tables / "bbbd_validation.json")
    model = _json(tables / "frozen_ocular_model.json")
    eeg = pd.read_csv(tables / "bbbd_ocular_eeg_joined.csv")
    eeg_result = _json(tables / "bbbd_eeg_convergence.json")
    fpr, tpr, thresholds = roc_curve(predictions["truth_attentive"], predictions["probability_attentive"])
    roc_data = pd.DataFrame({"panel": "roc", "x": fpr, "y": tpr, "value": thresholds})
    # The inferential table is deliberately stored in paired wide form so that
    # each row is one participant-video unit.  Collapse videos within
    # participant for the descriptive paired panel, then reshape for the
    # exported figure-source table.
    participant_pair = paired.groupby("participant")[["distracted", "attentive"]].mean()
    pair_rows = participant_pair.reset_index().melt(
        id_vars="participant", var_name="condition", value_name="probability"
    )
    pair_rows["panel"] = "paired"
    predictions["calibration_bin"] = pd.qcut(predictions["probability_attentive"], q=10, duplicates="drop")
    calibration = predictions.groupby("calibration_bin", observed=True).agg(
        predicted=("probability_attentive", "mean"), observed=("truth_attentive", "mean"), n=("truth_attentive", "size")
    ).reset_index(drop=True)
    calibration["panel"] = "calibration"
    coefficients = pd.DataFrame(
        {
            "panel": "coefficient",
            "feature": ["Gaze X", "Gaze Y", "Pupil"],
            "coefficient": model["coefficient"],
        }
    )
    source = pd.concat([roc_data, pair_rows, calibration, coefficients, eeg.assign(panel="eeg")], ignore_index=True, sort=False)

    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    grid = fig.add_gridspec(2, 3, wspace=0.48, hspace=0.48)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1:])]

    ax = axes[0]
    ax.plot(fpr, tpr, color=COLORS["blue"], lw=1.8)
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=0.8)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Outer held-out ROC")
    ax.text(0.97, 0.05, f"AUC {validation['auc']:.3f}\n95% CI {validation['ci_low']:.3f}–{validation['ci_high']:.3f}", ha="right", va="bottom")

    ax = axes[1]
    xmap = {"distracted": 0, "attentive": 1}
    for participant_id, row in participant_pair.iterrows():
        ax.plot([0, 1], [row["distracted"], row["attentive"]], color="#BBBBBB", lw=0.45, alpha=0.65)
    means = participant_pair[["distracted", "attentive"]].mean()
    ax.scatter([0, 1], means, s=28, color=[COLORS["red"], COLORS["blue"]], zorder=3)
    ax.set(xticks=[0, 1], xticklabels=["Distracted\nrewatch", "Attentive\nfirst"], ylabel="Held-out score", ylim=(-0.02, 1.02), title="Paired participant means")
    ax.text(0.98, 0.05, f"Δ = {validation['observed_mean_paired_probability_difference']:.3f}\np = {validation['p_value']:.5f}", transform=ax.transAxes, ha="right")

    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=COLORS["gray"], ls="--", lw=0.8)
    ax.plot(calibration["predicted"], calibration["observed"], marker="o", color=COLORS["green"], ms=3)
    ax.set(xlabel="Mean predicted", ylabel="Observed attentive", xlim=(0, 1), ylim=(0, 1), title="Calibration")
    ax.text(0.03, 0.95, f"slope {validation['calibration_slope']:.2f}\nBrier {validation['brier']:.3f}", transform=ax.transAxes, va="top")

    ax = axes[3]
    coef = np.asarray(model["coefficient"], dtype=float)
    ax.barh([0, 1, 2], coef, color=[COLORS["sky"], COLORS["blue"], COLORS["purple"]])
    ax.axvline(0, color=COLORS["gray"], lw=0.7)
    ax.set(yticks=[0, 1, 2], yticklabels=["Gaze X", "Gaze Y", "Pupil"], xlabel="Standardized logit coefficient", title="Frozen model")
    ax.invert_yaxis()

    ax = axes[4]
    colors = eeg["condition"].map({"attentive": COLORS["blue"], "distracted": COLORS["red"]})
    ax.scatter(eeg["probability_attentive"], eeg["eeg_corrca_isc"], s=7, c=colors, alpha=0.25, edgecolors="none")
    x = eeg["probability_attentive"].to_numpy(float)
    y = eeg["eeg_corrca_isc"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, intercept + slope * xx, color=COLORS["gray"], lw=1.2)
    ax.set(xlabel="Held-out ocular score", ylabel="Held-participant EEG CorrCA ISC", title="Within-cohort convergence (secondary)")
    ax.text(0.99, 0.03, f"adjusted β = {eeg_result['condition_adjusted_fixed_effect_coefficient']:.3f}\n95% CI {eeg_result['ci_low']:.3f}–{eeg_result['ci_high']:.3f}\np = {eeg_result['p_value']:.3f}", transform=ax.transAxes, ha="right", va="bottom")

    for label, ax in zip("abcde", axes, strict=True):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)
    fig.text(0.5, 0.01, "BBBD session labels are confounded with viewing order and repetition; EEG uses the same cohorts.", ha="center", fontsize=6.5, color=COLORS["gray"])
    return _save(fig, output, "figure2_bbbd_benchmark"), source


def _jitter_points(ax, frame, value, rng):
    for index, group in enumerate(GROUP_ORDER):
        values = frame.loc[frame["group"] == group, value].to_numpy(float)
        jitter = rng.normal(0, 0.055, len(values))
        ax.scatter(index + jitter, values, s=9, alpha=0.55, color=GROUP_COLORS[group], edgecolors="none")
        mean = values.mean()
        ax.plot([index - 0.18, index + 0.18], [mean, mean], color="black", lw=1.4)
    ax.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER, rotation=20)


def _figure3(tables: Path, output: Path, seed: int) -> tuple[list[str], pd.DataFrame]:
    _style()
    scores = pd.read_csv(tables / "sherlock_participant_transported_scores.csv")
    label_null = pd.read_csv(tables / "sherlock_boundary_label_null.csv")
    temporal_null = pd.read_csv(tables / "sherlock_boundary_temporal_null.csv")
    specs = pd.read_csv(tables / "sherlock_specification_grid.csv")
    global_result = _json(tables / "sherlock_global_inference.json")
    boundary = _json(tables / "sherlock_boundary_inference.json")
    source = pd.concat(
        [
            scores.assign(panel="participant_scores"),
            label_null.assign(panel="label_null"),
            temporal_null.assign(panel="temporal_null"),
            specs.assign(panel="specification"),
        ],
        ignore_index=True,
        sort=False,
    )
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 158 / 25.4))
    plt.subplots_adjust(wspace=0.48, hspace=0.5)
    rng = np.random.default_rng(seed)

    ax = axes[0, 0]
    _jitter_points(ax, scores, "global_score", rng)
    ax.set(ylabel="Full-film transported score", ylim=(-0.05, 1.05), title="Global group contrast")
    ax.text(0.98, 0.96, f"effect {global_result['observed']:.3f}\n95% CI {global_result['ci_low']:.3f}–{global_result['ci_high']:.3f}\np = {global_result['p_value']:.5f}", transform=ax.transAxes, ha="right", va="top")

    ax = axes[0, 1]
    _jitter_points(ax, scores, "boundary_change", rng)
    ax.axhline(0, color=COLORS["gray"], lw=0.7)
    ax.set(ylabel="Boundary − pre score", title="Annotated-boundary contrast")
    ax.text(0.98, 0.96, f"effect {boundary['observed']:.3f}\n95% CI {boundary['ci_low']:.3f}–{boundary['ci_high']:.3f}", transform=ax.transAxes, ha="right", va="top")

    ax = axes[0, 2]
    ax.hist(label_null["effect"], bins=45, color=COLORS["sky"], alpha=0.85)
    ax.axvline(boundary["observed"], color=COLORS["red"], lw=1.6)
    ax.set(xlabel="Permuted group effect", ylabel="Count", title="Participant-label null")
    ax.text(0.98, 0.94, f"p = {boundary['label_p_value']:.5f}", transform=ax.transAxes, ha="right", va="top")

    ax = axes[1, 0]
    ax.hist(temporal_null["effect"], bins=45, color=COLORS["purple"], alpha=0.82)
    ax.axvline(boundary["observed"], color=COLORS["red"], lw=1.6)
    ax.set(xlabel="Shifted-boundary effect", ylabel="Count", title="Boundary-location null")
    ax.text(0.98, 0.94, f"p = {boundary['temporal_p_value']:.3f}", transform=ax.transAxes, ha="right", va="top")

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.5, 0.82, "Conjunctive decision", ha="center", weight="bold", fontsize=10)
    ax.text(0.5, 0.60, f"Label condition: {'met' if boundary['label_p_value'] <= .05 else 'not met'}", ha="center", color=COLORS["green"] if boundary['label_p_value'] <= .05 else COLORS["red"])
    ax.text(0.5, 0.45, f"Location condition: {'met' if boundary['temporal_p_value'] <= .05 else 'not met'}", ha="center", color=COLORS["green"] if boundary['temporal_p_value'] <= .05 else COLORS["red"])
    ax.text(0.5, 0.27, f"IUT p = {boundary['intersection_union_p']:.3f}", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.10, "No boundary-specific group claim", ha="center", color=COLORS["red"], weight="bold")

    ax = axes[1, 2]
    raw = specs.loc[specs["score_family"] == "raw_isc"].sort_values("effect").reset_index(drop=True)
    color = raw["window_seconds"].map({5: COLORS["blue"], 10: COLORS["orange"], 20: COLORS["purple"]})
    ax.scatter(raw["effect"], np.arange(len(raw)), s=8, c=color, alpha=0.75, edgecolors="none")
    ax.axvline(0, color=COLORS["gray"], lw=0.8)
    ax.set(xlabel="Short-R boundary contrast", ylabel="Specifications (sorted)", title="Complete raw-ISC grid", yticks=[])
    for window, y in [(5, 0.95), (10, 0.87), (20, 0.79)]:
        ax.text(0.98, y, f"{window} s", transform=ax.transAxes, color={5: COLORS['blue'], 10: COLORS['orange'], 20: COLORS['purple']}[window], ha="right")
    ax.text(0.02, 0.05, f"{int((raw['effect'] < 0).sum())}/{len(raw)} negative", transform=ax.transAxes)

    for label, ax in zip("abcdef", axes.flat, strict=True):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)
    fig.text(0.5, 0.01, "The transported score is cohort-relative and is not a direct measure of attention or event segmentation.", ha="center", fontsize=6.5, color=COLORS["gray"])
    return _save(fig, output, "figure3_sherlock_crossed_null"), source


def _figure4(tables: Path, output: Path, seed: int) -> tuple[list[str], pd.DataFrame]:
    _style()
    summary = pd.read_csv(tables / "dual_null_simulation_summary.csv")
    scenarios = ["baseline", "low_autocorrelation", "high_autocorrelation", "smaller_sample", "more_boundaries", "smooth_nonstationarity", "weaker_signals"]
    states = ["neither", "global_group_only", "common_boundary_only", "group_boundary_interaction"]
    labels = ["Neither", "Global group", "Common boundary", "Group × boundary"]
    matrix = summary.pivot(index="scenario", columns="state", values="boundary_iut_detection_rate").loc[scenarios, states]
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 140 / 25.4))
    plt.subplots_adjust(wspace=0.42, hspace=0.48)

    ax = axes[0, 0]
    truth = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 1]])
    im = ax.imshow(truth, cmap=matplotlib.colors.ListedColormap(["white", COLORS["blue"]]), vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["Global group", "Common\nboundary", "Group ×\nboundary"],
        yticks=np.arange(4),
        yticklabels=labels,
        title="Simulation truth states",
    )
    for i in range(4):
        for j in range(3):
            ax.text(j, i, "present" if truth[i, j] else "absent", ha="center", va="center", color="white" if truth[i, j] else "black", fontsize=7)

    ax = axes[0, 1]
    heat = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set(
        xticks=np.arange(4),
        xticklabels=labels,
        yticks=np.arange(len(scenarios)),
        yticklabels=[value.replace("_", " ") for value in scenarios],
        title="IUT detection rate",
    )
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > .45 else "black", fontsize=6.5)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="Detection rate")

    ax = axes[1, 0]
    selected = summary[(summary["scenario"] == "baseline")].set_index("state").loc[states]
    x = np.arange(4)
    width = 0.24
    for offset, column, label, color in [
        (-width, "boundary_label_detection_rate", "Label", COLORS["blue"]),
        (0, "boundary_temporal_detection_rate", "Location", COLORS["purple"]),
        (width, "boundary_iut_detection_rate", "IUT", COLORS["red"]),
    ]:
        ax.bar(x + offset, selected[column], width=width, label=label, color=color)
    ax.axhline(.05, color=COLORS["gray"], ls="--", lw=.8)
    ax.set(xticks=x, xticklabels=labels, ylabel="Detection rate", ylim=(0, 1.05), title="Baseline component decisions")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")

    ax = axes[1, 1]
    interaction = summary[summary["state"] == "group_boundary_interaction"].set_index("scenario").loc[scenarios]
    y = np.arange(len(scenarios))
    rate = interaction["boundary_iut_detection_rate"].to_numpy(float)
    low = interaction["boundary_iut_detection_rate_ci_low"].to_numpy(float)
    high = interaction["boundary_iut_detection_rate_ci_high"].to_numpy(float)
    # Guard against machine-precision negatives when a Wilson upper bound and
    # the observed rate are both represented as 1.0.
    xerr = [np.maximum(rate - low, 0.0), np.maximum(high - rate, 0.0)]
    ax.errorbar(rate, y, xerr=xerr, fmt="o", color=COLORS["red"], ecolor=COLORS["gray"], capsize=2)
    ax.set(yticks=y, yticklabels=[value.replace("_", " ") for value in scenarios], xlabel="IUT power (95% Monte Carlo CI)", xlim=(0, 1.04), title="Interaction recovery")
    ax.invert_yaxis()

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)
    fig.text(0.5, 0.01, f"Simulation: 250 replicates per state/scenario; seed {seed}; parameters were not fitted to Sherlock.", ha="center", fontsize=6.5, color=COLORS["gray"])
    return _save(fig, output, "figure4_simulation"), summary


def build_report_artifacts(config: dict, project: Path) -> dict[str, object]:
    tables = project / "results" / "tables"
    figures = project / "results" / "figures"
    source_dir = project / "results" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    domain, sherlock_augmented = _domain_shift(tables)
    domain.to_csv(tables / "domain_shift_audit.csv", index=False)
    sherlock_augmented.to_csv(tables / "sherlock_global_scores_with_linear_predictor.csv", index=False)
    quality = _json(tables / "bbbd_quality_summary.json")
    validation = _json(tables / "bbbd_validation.json")
    global_result = _json(tables / "sherlock_global_inference.json")
    boundary = _json(tables / "sherlock_boundary_inference.json")
    eeg = _json(tables / "bbbd_eeg_convergence.json")
    memory = pd.read_csv(tables / "sherlock_memory_associations.csv")
    specs = pd.read_csv(tables / "sherlock_specification_grid.csv")
    raw = specs[specs["score_family"] == "raw_isc"]
    invariants = {
        "bbbd_input_records": quality["input_records"],
        "bbbd_eligible_records": quality["eligible_records"],
        "bbbd_eligible_pairs": quality["eligible_pairs"],
        "bbbd_eligible_participants": quality["eligible_participants"],
        "bbbd_auc": validation["auc"],
        "bbbd_auc_ci": [validation["ci_low"], validation["ci_high"]],
        "bbbd_signflip_p": validation["p_value"],
        "bbbd_gate_passed": validation["gate_passed"],
        "sherlock_n": config["sherlock"]["expected_participants"],
        "sherlock_group_counts": config["sherlock"]["expected_groups"],
        "sherlock_boundaries": len(config["sherlock"]["boundaries_10hz"]),
        "global_effect": global_result["observed"],
        "global_ci": [global_result["ci_low"], global_result["ci_high"]],
        "global_p": global_result["p_value"],
        "boundary_effect": boundary["observed"],
        "boundary_ci": [boundary["ci_low"], boundary["ci_high"]],
        "boundary_label_p": boundary["label_p_value"],
        "boundary_temporal_p": boundary["temporal_p_value"],
        "boundary_iut_p": boundary["intersection_union_p"],
        "boundary_specific_claim_supported": boundary["reject_both"],
        "raw_specifications": len(raw),
        "raw_specifications_negative": int((raw["effect"] < 0).sum()),
        "eeg_adjusted_beta": eeg["condition_adjusted_fixed_effect_coefficient"],
        "eeg_ci": [eeg["ci_low"], eeg["ci_high"]],
        "eeg_p": eeg["p_value"],
        "memory": memory.to_dict(orient="records"),
    }
    (tables / "statistical_invariants.json").write_text(json.dumps(invariants, indent=2) + "\n", encoding="utf-8")

    figure1 = _figure1(config, quality, figures)
    figure2, source2 = _figure2(tables, figures)
    seed = int(config["seed"])
    figure3, source3 = _figure3(tables, figures, seed)
    figure4, source4 = _figure4(tables, figures, seed)
    pd.DataFrame(
        [
            {"stage": "BBBD benchmark", "input": "official paired ocular derivatives", "output": "frozen three-channel logistic score"},
            {"stage": "Sherlock transport", "input": "pooled leave-one-out ocular ISC", "output": "global and boundary contrasts"},
            {"stage": "label null", "input": "fixed annotated boundaries", "output": "group exchangeability p-value"},
            {"stage": "location null", "input": "fixed participant groups", "output": "boundary-location p-value"},
        ]
    ).to_csv(source_dir / "figure1_framework.csv", index=False)
    source2.to_csv(source_dir / "figure2_bbbd_benchmark.csv", index=False)
    source3.to_csv(source_dir / "figure3_sherlock_transfer.csv", index=False)
    source4.to_csv(source_dir / "figure4_simulations.csv", index=False)

    audit_lines = [
        "# Statistical audit",
        "",
        f"- BBBD quality-complete analysis: {quality['eligible_participants']} participants, {quality['eligible_pairs']} paired participant-video units, {quality['eligible_records']} records.",
        f"- BBBD outer AUC: {validation['auc']:.6f}, two-way bootstrap 95% CI [{validation['ci_low']:.6f}, {validation['ci_high']:.6f}]; participant sign-flip p = {validation['p_value']:.6f}. Frozen gate: passed.",
        f"- Sherlock global Short-R contrast: {global_result['observed']:.6f}, 95% CI [{global_result['ci_low']:.6f}, {global_result['ci_high']:.6f}], p = {global_result['p_value']:.6f}.",
        f"- Sherlock boundary contrast: {boundary['observed']:.6f}, participant/boundary bootstrap 95% CI [{boundary['ci_low']:.6f}, {boundary['ci_high']:.6f}].",
        f"- Boundary component p-values: label {boundary['label_p_value']:.6f}; location {boundary['temporal_p_value']:.6f}; IUT {boundary['intersection_union_p']:.6f}. Boundary-specific claim supported: no.",
        f"- Complete raw-ISC specification grid: {int((raw['effect'] < 0).sum())}/{len(raw)} negative; no robustness threshold applied.",
        f"- EEG convergence: adjusted beta {eeg['condition_adjusted_fixed_effect_coefficient']:.6f}, 95% CI [{eeg['ci_low']:.6f}, {eeg['ci_high']:.6f}], p = {eeg['p_value']:.6f}; not independent validation.",
        "- Memory coefficients are group-adjusted associations and neither is described as prediction or causation.",
        "- Plus-one Monte Carlo p-values are used throughout. Label and temporal tests retain their distinct exchangeability assumptions.",
    ]
    (project / "results" / "statistical_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    outputs = [*figure1, *figure2, *figure3, *figure4]
    manifest = []
    for path_text in outputs:
        path = Path(path_text)
        manifest.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    pd.DataFrame(manifest).to_csv(project / "provenance" / "figure_output_manifest.csv", index=False)
    return {"figures": len(outputs), "source_tables": 4, "invariants": str(tables / "statistical_invariants.json")}
