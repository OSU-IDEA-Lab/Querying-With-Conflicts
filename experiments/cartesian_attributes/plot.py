"""Plots matching the visual language of the bucket-count experiment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .scheme import DATASET_ORDER


INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SERIES = [
    ("Alg. 1", "#2a78d6", "o", "-"),
    ("Fast Alg. 1", "#eb6834", "s", "--"),
    ("Random credible pairs", "#1baf7a", "^", ":"),
    ("Single best pair", "#8e44ad", "D", "-."),
    ("Single random pair", "#7c746c", "P", (0, (3, 1, 1, 1))),
]
DATASET_COLORS = {
    "amazon": "#34549c",
    "pricerunner": "#c94f45",
    "flights": "#2eb16d",
    "census": "#f39c12",
    "compas": "#9a4caf",
}


def _style(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=0)


def _common_ranking_limits(path: Path, step: float = 0.05) -> tuple[float, float]:
    frames = []
    for candidate in sorted(path.parent.glob("*_utility_summary.csv")):
        frame = pd.read_csv(candidate)
        if {"strategy", "ci95_low", "ci95_high", "num_attributes"}.issubset(frame):
            frames.append(frame[frame["strategy"].isin([item[0] for item in SERIES])])
    combined = pd.concat(frames, ignore_index=True)
    low = min(0.0, float(combined["ci95_low"].min()))
    high = max(0.0, float(combined["ci95_high"].max()))
    return math.floor(low / step) * step, math.ceil(high / step) * step


def _common_query_limits(path: Path) -> tuple[float, float]:
    values: list[float] = []
    for candidate in sorted(path.parent.glob("*_runtime.csv")):
        frame = pd.read_csv(candidate)
        if {"strategy", "query_utility", "num_attributes"}.issubset(frame):
            selected = frame.loc[
                frame["strategy"].isin([item[0] for item in SERIES])
                & (frame["query_utility"] > 0),
                "query_utility",
            ]
            values.extend(selected.tolist())
    return (
        10.0 ** math.floor(math.log10(min(values))),
        10.0 ** math.ceil(math.log10(max(values))),
    )


def _query_summary(runtime: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runtime.groupby(["lambda", "strategy", "num_attributes"])["query_utility"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["sem"]
    return summary


def plot_results(
    utility_summary_path: str | Path,
    runtime_summary_path: str | Path,
    output_prefix: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    utility_path = Path(utility_summary_path)
    utility = pd.read_csv(utility_path)
    runtime = pd.read_csv(runtime_summary_path)
    raw_runtime_path = Path(f"{output_prefix}_runtime.csv")
    raw_runtime = pd.read_csv(raw_runtime_path)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    dataset = str(utility["dataset_label"].iloc[0])
    active = [item for item in SERIES if item[0] in set(utility["strategy"])]
    attributes = sorted(utility["num_attributes"].unique())

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "legend.frameon": False,
        }
    )

    lambdas = sorted(utility["lambda"].unique())
    figure, axes = plt.subplots(
        1, len(lambdas), figsize=(3.2 * len(lambdas), 3.4), sharey=True, squeeze=False
    )
    for axis, lam in zip(axes[0], lambdas):
        selected = utility[utility["lambda"] == lam]
        for name, color, marker, linestyle in active:
            data = selected[selected["strategy"] == name].sort_values("num_attributes")
            axis.fill_between(
                data["num_attributes"], data["ci95_low"], data["ci95_high"],
                color=color, alpha=0.12, linewidth=0,
            )
            axis.plot(
                data["num_attributes"], data["mean"], color=color, linewidth=2,
                marker=marker, markersize=5, markeredgecolor="white", markeredgewidth=1,
                linestyle=linestyle, label=name,
            )
        axis.set_xticks(attributes)
        axis.set_title(f"λ = {lam:g}")
        axis.set_xlabel("Number of attributes")
        axis.axhline(0.0, color=GRID, linewidth=1.0, zorder=0)
        _style(axis)
    axes[0, 0].set_ylabel("Ranking utility (Kendall τ-b)")
    axes[0, 0].set_ylim(*_common_ranking_limits(utility_path))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(labels),
        bbox_to_anchor=(0.5, 1.04), fontsize=8.5, labelcolor=INK,
    )
    figure.suptitle(f"{dataset}: utility by number of attributes", y=1.10, fontsize=11)
    figure.tight_layout()
    figure.savefig(f"{prefix}_utility.png", dpi=200, bbox_inches="tight")
    figure.savefig(f"{prefix}_utility.pdf", bbox_inches="tight")
    plt.close(figure)

    query = _query_summary(raw_runtime)
    figure, axes = plt.subplots(
        1, len(lambdas), figsize=(3.2 * len(lambdas), 3.4), sharey=True, squeeze=False
    )
    for axis, lam in zip(axes[0], lambdas):
        selected = query[query["lambda"] == lam]
        for name, color, marker, linestyle in active:
            data = selected[selected["strategy"] == name].sort_values("num_attributes")
            axis.fill_between(
                data["num_attributes"], data["ci95_low"].clip(lower=np.finfo(float).tiny),
                data["ci95_high"], color=color, alpha=0.12, linewidth=0,
            )
            axis.plot(
                data["num_attributes"], data["mean"], color=color, linewidth=2,
                marker=marker, markersize=5, markeredgecolor="white", markeredgewidth=1,
                linestyle=linestyle, label=name,
            )
        axis.set_yscale("log")
        axis.set_xticks(attributes)
        axis.set_title(f"λ = {lam:g}")
        axis.set_xlabel("Number of attributes")
        _style(axis)
    axes[0, 0].set_ylabel("User utility (information gain)")
    axes[0, 0].set_ylim(*_common_query_limits(raw_runtime_path))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=len(labels),
        bbox_to_anchor=(0.5, 1.04), fontsize=8.5, labelcolor=INK,
    )
    figure.suptitle(
        f"{dataset}: user utility by number of attributes", y=1.10, fontsize=11
    )
    figure.tight_layout()
    figure.savefig(f"{prefix}_query_utility.png", dpi=200, bbox_inches="tight")
    figure.savefig(f"{prefix}_query_utility.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.2, 3.0))
    for name, color, marker, linestyle in active:
        data = runtime[runtime["strategy"] == name].sort_values("num_attributes")
        axis.fill_between(
            data["num_attributes"], data["q25_seconds"], data["q75_seconds"],
            color=color, alpha=0.12, linewidth=0,
        )
        axis.plot(
            data["num_attributes"], data["median_seconds"], color=color, linewidth=2,
            marker=marker, markersize=5, markeredgecolor="white", markeredgewidth=1,
            linestyle=linestyle, label=name,
        )
    group_by_attribute = (
        runtime[["num_attributes", "observed_groups"]]
        .drop_duplicates()
        .set_index("num_attributes")["observed_groups"]
    )
    axis.set_yscale("log")
    axis.set_xticks(attributes)
    axis.set_xticklabels(
        [f"{value}\nm={int(group_by_attribute[value])}" for value in attributes]
    )
    axis.set_xlabel("Number of attributes (observed groups m)")
    axis.set_ylabel("Algorithm runtime (s)")
    axis.set_title(f"{dataset} (z = m)")
    _style(axis)
    axis.legend(fontsize=7.5, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    figure.savefig(f"{prefix}_runtime.png", dpi=200, bbox_inches="tight")
    figure.savefig(f"{prefix}_runtime.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_runtime_vs_m(output_dir: str | Path) -> None:
    """Plot the algorithm timings directly against observed group count m."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    frames = []
    for path in root.glob("*_runtime_summary.csv"):
        frame = pd.read_csv(path)
        if {"dataset", "observed_groups", "median_seconds"}.issubset(frame):
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), squeeze=False)
    for axis, dataset_key in zip(axes.ravel(), DATASET_ORDER):
        selected = combined[combined["dataset"] == dataset_key]
        dataset_label = str(selected["dataset_label"].iloc[0])
        active = [item for item in SERIES if item[0] in set(selected["strategy"])]
        for name, color, marker, linestyle in active:
            data = selected[selected["strategy"] == name].sort_values("observed_groups")
            axis.fill_between(
                data["observed_groups"], data["q25_seconds"], data["q75_seconds"],
                color=color, alpha=0.12, linewidth=0,
            )
            axis.plot(
                data["observed_groups"], data["median_seconds"], color=color,
                linewidth=2, marker=marker, markersize=5, markeredgecolor="white",
                markeredgewidth=1, linestyle=linestyle, label=name,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(dataset_label)
        axis.set_xlabel("Observed Cartesian groups, m")
        axis.set_ylabel("Algorithm runtime (s)")
        _style(axis)
    for axis in axes.ravel()[len(DATASET_ORDER):]:
        axis.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), frameon=False,
        bbox_to_anchor=(0.5, 1.01), fontsize=8.5,
    )
    fig.suptitle("Algorithm runtime versus m (current experiment uses z = m)", y=1.05)
    fig.tight_layout()
    fig.savefig(root / "runtime_vs_m.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "runtime_vs_m.pdf", bbox_inches="tight")
    plt.close(fig)


def _combined_frames(output_dir: str | Path, suffix: str) -> pd.DataFrame:
    frames = []
    for dataset_key in DATASET_ORDER:
        path = Path(output_dir) / f"{dataset_key}_{suffix}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if len(frames) != len(DATASET_ORDER):
        found = {str(frame["dataset"].iloc[0]) for frame in frames}
        missing = sorted(set(DATASET_ORDER) - found)
        raise FileNotFoundError(f"Missing result files for: {missing}")
    return pd.concat(frames, ignore_index=True)


def _style_iq_summary(axis, ylabel: str) -> None:
    axis.set_xticks((1, 2, 3))
    axis.set_xlabel("Number of Attributes", fontweight="bold")
    axis.set_ylabel(ylabel, fontweight="bold")
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", which="both", alpha=0.22)
    axis.tick_params(axis="both", which="both", length=0)
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    axis.legend(
        frameon=False, ncol=3, loc="upper left",
        prop={"weight": "bold", "size": 10},
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_iq_summaries(output_dir: str | Path) -> None:
    """Plot fresh IQ-only runtime and information gain for all five datasets."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    runtime = _combined_frames(root, "runtime")
    selected = runtime[runtime["strategy"] == "Alg. 1"]

    runtime_summary = (
        selected.groupby(["dataset", "dataset_label", "num_attributes"])["seconds"]
        .mean()
        .rename("time_seconds")
        .reset_index()
    )
    utility_summary = (
        selected.groupby(["dataset", "dataset_label", "num_attributes"])["query_utility"]
        .mean()
        .rename("utility_gain")
        .reset_index()
    )
    runtime_summary.to_csv(root / "iq_time_vs_num_attributes.csv", index=False)
    utility_summary.to_csv(root / "iq_utility_gain_vs_num_attributes.csv", index=False)

    for summary, value, ylabel, stem, log_scale in (
        (
            runtime_summary,
            "time_seconds",
            "Time(s) - log scale",
            "iq_time_vs_num_attributes",
            True,
        ),
        (
            utility_summary,
            "utility_gain",
            "Utility Gain - log scale",
            "iq_utility_gain_vs_num_attributes",
            True,
        ),
    ):
        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axis = plt.subplots(figsize=(7.6, 5.0))
        for dataset_key in DATASET_ORDER:
            data = summary[summary["dataset"] == dataset_key].sort_values(
                "num_attributes"
            )
            axis.plot(
                data["num_attributes"], data[value],
                color=DATASET_COLORS[dataset_key], linewidth=2.4,
                label=str(data["dataset_label"].iloc[0]),
            )
        if log_scale:
            axis.set_yscale("log")
        _style_iq_summary(axis, ylabel)
        figure.tight_layout(pad=0.25)
        figure.savefig(root / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(root / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
