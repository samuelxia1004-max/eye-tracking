"""Python-only publication figure generation."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PALETTE = {
    "Short-R": "#C44E52",
    "Long": "#4C72B0",
    "Schema": "#55A868",
    "Short-P": "#8172B3",
    "neutral": "#6B7280",
    "accent": "#D18F2B",
}


def _setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def figure1(root: Path) -> None:
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.55, wspace=0.35)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    labels = ["Public MAT files\n104 unique participants", "10-Hz traces\nX, Y, pupil", "18 event boundaries\nfixed windows", "ISC estimators\nand channel sets", "Permutation + bootstrap\nPseudo-boundary null"]
    xs = np.linspace(0.04, 0.82, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels, strict=True)):
        box = FancyBboxPatch((x, 0.38), 0.14, 0.32, boxstyle="round,pad=0.02", facecolor="#F4F6F8", edgecolor="#59636E", linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.07, 0.54, label, ha="center", va="center", fontsize=7)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.145, 0.54), (xs[i + 1] - 0.006, 0.54), arrowstyle="-|>", mutation_scale=8, color="#59636E", linewidth=0.8))
    ax.text(0.01, 0.96, "a", fontsize=8, fontweight="bold", va="top")
    ax.text(0.04, 0.16, "Frozen locally before inferential reanalysis; not a public preregistration", fontsize=6, color=PALETTE["neutral"])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ax2.text(0.02, 0.9, "Pairwise mean ISC", fontweight="bold", fontsize=8)
    ax2.text(0.02, 0.62, r"$ISC_g=\frac{2}{n_g(n_g-1)}\sum_{i<j}r_{ij}$", fontsize=11)
    ax2.text(0.02, 0.32, "Partners are reconstructed\nafter every label permutation.", fontsize=6.5, linespacing=1.4)
    _panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.text(0.02, 0.9, "Leave-one-out ISC", fontweight="bold", fontsize=8)
    ax3.text(0.02, 0.62, r"$ISC_i=r(x_i,\bar{x}_{-i})$", fontsize=11)
    ax3.text(0.02, 0.32, "The reference is within-group\nor all other participants.", fontsize=6.5, linespacing=1.4)
    _panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, 2])
    x = np.linspace(-1, 1, 100)
    ax4.plot(x, x, label="Pearson r", color=PALETTE["neutral"])
    ax4.plot(x, (1 + x) / 2, label="Legacy mean(corrcoef)", color=PALETTE["accent"])
    ax4.set(xlabel="Pearson r", ylabel="Reported scale", xlim=(-1, 1), ylim=(-1, 1))
    ax4.legend(loc="upper left")
    _panel_label(ax4, "d")
    _save(fig, root / "results" / "figures" / "figure1_method_workflow")


def figure2(root: Path) -> None:
    tables = root / "results" / "tables"
    participant = pd.read_csv(tables / "faithful_reproduction_participant.csv")
    summary = pd.read_csv(tables / "faithful_reproduction_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"wspace": 0.42})
    phases = ["pre", "boundary", "post"]
    for group in PALETTE:
        if group in {"neutral", "accent"}:
            continue
        part = summary[(summary.scale == "raw") & (summary.channel == "composite") & (summary.group == group)].set_index("phase").loc[phases]
        axes[0].errorbar(np.arange(3), part["mean"], yerr=part["std"] / np.sqrt(part["count"]), marker="o", linewidth=1.2, markersize=3, capsize=2, label=group, color=PALETTE[group])
    axes[0].set(xticks=np.arange(3), xticklabels=["Pre", "Boundary", "Post"], ylabel="Composite ISC (Pearson r)")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    _panel_label(axes[0], "a")

    raw = participant[(participant.scale == "raw") & (participant.channel == "composite")].pivot(index=["participant_id", "group"], columns="phase", values="isc").reset_index()
    raw["change"] = raw["boundary"] - raw["pre"]
    order = ["Short-R", "Long", "Schema", "Short-P"]
    rng = np.random.default_rng(20260816)
    for i, group in enumerate(order):
        values = raw.loc[raw.group == group, "change"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        axes[1].scatter(i + jitter, values, s=7, alpha=0.55, color=PALETTE[group], linewidth=0)
        axes[1].plot([i - 0.18, i + 0.18], [np.mean(values)] * 2, color="black", linewidth=1.2)
    axes[1].axhline(0, color="#A0A0A0", linewidth=0.6)
    axes[1].set(xticks=range(4), xticklabels=order, ylabel="Boundary minus pre ISC")
    plt.setp(axes[1].get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right")
    _panel_label(axes[1], "b")

    channel_change = participant[(participant.scale == "raw") & participant.channel.isin(["x", "y", "pupil"])].pivot(index=["participant_id", "group", "channel"], columns="phase", values="isc").reset_index()
    channel_change["change"] = channel_change["boundary"] - channel_change["pre"]
    means = channel_change.groupby(["group", "channel"])["change"].mean().unstack()
    x = np.arange(4)
    width = 0.22
    for j, channel in enumerate(["x", "y", "pupil"]):
        axes[2].bar(x + (j - 1) * width, means.loc[order, channel], width, label=channel.capitalize(), color=["#4C72B0", "#55A868", "#C9A227"][j])
    axes[2].axhline(0, color="#777777", linewidth=0.6)
    axes[2].set(xticks=x, xticklabels=order, ylabel="Mean channel change")
    plt.setp(axes[2].get_xticklabels(), rotation=35, rotation_mode="anchor", ha="right")
    axes[2].legend()
    _panel_label(axes[2], "c")
    _save(fig, root / "results" / "figures" / "figure2_reproduction_channels")


def figure3(root: Path) -> None:
    tables = root / "results" / "tables"
    specs = pd.read_csv(tables / "specification_grid.csv").sort_values("effect")
    pseudo = pd.read_csv(tables / "pseudo_boundary_null.csv")
    pseudo_test = json.loads((tables / "pseudo_boundary_test.json").read_text())
    primary = json.loads((tables / "primary_inference.json").read_text())
    fig = plt.figure(figsize=(7.2, 4.15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.55, 1], hspace=0.12, wspace=0.38)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_choices = fig.add_subplot(gs[1, 0], sharex=ax_curve)
    ax_null = fig.add_subplot(gs[:, 1])
    colors = specs["channel"].map({"x": "#4C72B0", "y": "#55A868", "pupil": "#C9A227", "gaze": "#8172B3", "composite": "#C44E52"})
    ax_curve.scatter(np.arange(len(specs)), specs["effect"], c=colors, s=12, linewidth=0, alpha=0.8)
    ax_curve.axhline(0, color="#666666", linewidth=0.7)
    ax_curve.set(ylabel="Short-R minus mean controls\n(boundary minus pre)")
    ax_curve.tick_params(axis="x", labelbottom=False)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=label) for label, color in [("X", "#4C72B0"), ("Y", "#55A868"), ("Pupil", "#C9A227"), ("Gaze", "#8172B3"), ("Composite", "#C44E52")]]
    ax_curve.legend(handles=handles, ncol=3, loc="lower right")
    _panel_label(ax_curve, "a")

    choice_rows = [
        ("Channel: X", specs["channel"].eq("x")),
        ("Channel: Y", specs["channel"].eq("y")),
        ("Channel: pupil", specs["channel"].eq("pupil")),
        ("Channel: gaze", specs["channel"].eq("gaze")),
        ("Channel: composite", specs["channel"].eq("composite")),
        ("Estimator: pairwise", specs["estimator"].eq("pairwise")),
        ("Estimator: LOO", specs["estimator"].eq("loo")),
        ("Aggregation: raw", specs["transform"].eq("raw")),
        ("Aggregation: Fisher z", specs["transform"].eq("fisher")),
        ("Window: 2 s", specs["window_seconds"].eq(2)),
        ("Window: 5 s", specs["window_seconds"].eq(5)),
        ("Window: 10 s", specs["window_seconds"].eq(10)),
        ("Reference: within", specs["reference"].eq("within")),
        ("Reference: all", specs["reference"].eq("all")),
    ]
    choice_matrix = np.vstack([row.to_numpy(dtype=float) for _, row in choice_rows])
    ax_choices.imshow(choice_matrix, aspect="auto", interpolation="nearest", cmap=mpl.colors.ListedColormap(["#F4F6F8", "#273746"]), vmin=0, vmax=1)
    ax_choices.set_yticks(np.arange(len(choice_rows)), [label for label, _ in choice_rows], fontsize=5.2)
    ax_choices.set_xlabel("Prespecified specifications, ordered by estimate")
    ax_choices.set_xticks([0, 29, 59, 89, 119], [1, 30, 60, 90, 120])
    ax_choices.tick_params(axis="y", length=0, pad=2)
    ax_choices.spines["left"].set_visible(False)
    ax_choices.spines["bottom"].set_visible(False)

    ax_null.hist(pseudo["pseudo_effect"], bins=45, color="#B9C0C7", edgecolor="white", linewidth=0.25)
    ax_null.axvline(pseudo_test["observed"], color=PALETTE["Short-R"], linewidth=1.4, label="Observed")
    ax_null.axvline(primary["observed"], color="#111111", linewidth=0.8, linestyle=":", label="Pairwise primary")
    ax_null.set(xlabel="Common-shift pseudo-boundary effect", ylabel="Count")
    ax_null.legend(loc="upper right")
    ax_null.text(0.03, 0.96, f"two-sided p = {pseudo_test['p_value']:.3g}", transform=ax_null.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    _panel_label(ax_null, "b")
    _save(fig, root / "results" / "figures" / "figure3_specification_pseudoboundary")


def figure4(root: Path) -> None:
    tables = root / "results" / "tables"
    simulation = pd.read_csv(tables / "simulation_summary.csv")
    memory = pd.read_csv(tables / "memory_associations.csv")
    participant = pd.read_csv(tables / "participant_boundary_response.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.48})
    null = (
        simulation[simulation.shared_signal == 0]
        .groupby(["ar1", "group_size"], as_index=False)["rejection_rate"]
        .mean()
    )
    alt = simulation[simulation.shared_signal > 0]
    for rho, group in null.groupby("ar1"):
        axes[0].plot(group["group_size"], group["rejection_rate"], marker="o", label=f"AR(1)={rho}")
    axes[0].axhline(0.05, color="#777777", linestyle=":", linewidth=0.8)
    axes[0].set(xlabel="Participants per group", ylabel="Type-I error", ylim=(0, max(0.25, null.rejection_rate.max() * 1.15)))
    axes[0].legend()
    _panel_label(axes[0], "a")

    affected = alt.groupby(["affected_channel", "group_size"])[["rejection_rate", "interval_coverage"]].mean().reset_index()
    for affected_channel, group in affected.groupby("affected_channel"):
        axes[1].plot(group["group_size"], group["rejection_rate"], marker="o", label=f"Power: {affected_channel}")
        axes[1].plot(group["group_size"], group["interval_coverage"], marker="s", linestyle="--", label=f"Coverage: {affected_channel}")
    axes[1].set(xlabel="Participants per group", ylabel="Rate", ylim=(0, 1.02))
    axes[1].legend(fontsize=5.5)
    _panel_label(axes[1], "b")

    plot_data = participant.dropna(subset=["boundary_response", "detail"]).copy()
    plot_data["response_residual"] = plot_data["boundary_response"] - plot_data.groupby("group")["boundary_response"].transform("mean")
    plot_data["detail_residual"] = plot_data["detail"] - plot_data.groupby("group")["detail"].transform("mean")
    colors = [PALETTE.get(group, PALETTE["neutral"]) for group in plot_data.group]
    axes[2].scatter(plot_data["response_residual"], plot_data["detail_residual"], c=colors, s=9, alpha=0.65, linewidth=0)
    fit = np.polyfit(plot_data["response_residual"], plot_data["detail_residual"], 1)
    xx = np.linspace(plot_data["response_residual"].min(), plot_data["response_residual"].max(), 100)
    axes[2].plot(xx, np.polyval(fit, xx), color="#222222", linewidth=0.9)
    row = memory[memory.outcome == "detail"].iloc[0]
    axes[2].set(xlabel="Boundary response residual", ylabel="Detail score residual")
    axes[2].text(0.03, 0.97, f"adjusted b={row.coefficient:.2f}\np={row.p_value:.3g}", transform=axes[2].transAxes, va="top")
    _panel_label(axes[2], "c")
    _save(fig, root / "results" / "figures" / "figure4_simulation_memory")


def generate_all_figures(root: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    _setup()
    figure1(root)
    figure2(root)
    figure3(root)
    figure4(root)
    source_dir = root / "results" / "figures" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    table_dir = root / "results" / "tables"
    source_map = {
        "faithful_reproduction_summary.csv": "figure2_summary.csv",
        "faithful_reproduction_participant.csv": "figure2_participant.csv",
        "specification_grid.csv": "figure3_specifications.csv",
        "pseudo_boundary_null.csv": "figure3_pseudo_boundary_null.csv",
        "simulation_summary.csv": "figure4_simulation_summary.csv",
        "participant_boundary_response.csv": "figure4_memory_points.csv",
        "memory_associations.csv": "figure4_memory_models.csv",
    }
    for source, destination in source_map.items():
        shutil.copyfile(table_dir / source, source_dir / destination)
