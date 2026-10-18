"""Paper plots for the distribution-aware Cartesian-attribute experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .distributional import DEFAULT_PRIORS
from .scheme import DATASET_ORDER


GRID = "#e5e5e5"
LINE_WIDTH = 2.4
DATASET_STYLES = {
    "amazon": ("#222222", "o"),
    "pricerunner": ("#d73027", "s"),
    "flights": ("#00a6d6", "^"),
    "census": ("#c49a00", "D"),
    "compas": ("#777777", "P"),
}


def _style(axis, ylabel: str) -> None:
    axis.set_xticks((1, 2, 3))
    axis.set_xlabel("Number of Ranking Attributes", fontweight="bold", fontsize=11)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=11)
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_lines(axis, data: pd.DataFrame, metric: str) -> None:
    for dataset in DATASET_ORDER:
        selected = data[data["dataset"] == dataset].sort_values(
            "num_intent_attributes"
        )
        if selected.empty:
            continue
        color, marker = DATASET_STYLES[dataset]
        axis.plot(
            selected["num_intent_attributes"],
            selected[metric],
            color=color,
            linewidth=LINE_WIDTH,
            marker=marker,
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            label=str(selected["dataset_label"].iloc[0]),
        )


def average_bias_attributes(data: pd.DataFrame) -> pd.DataFrame:
    """Give each configured bias attribute equal weight in every plotted point."""

    if "bias_attribute" not in data.columns:
        return data
    keys = ["dataset", "dataset_label", "prior_model", "num_intent_attributes"]
    if "bias_attribute_count" in data.columns:
        joint_counts = data.groupby(keys)["bias_attribute"].nunique()
        if joint_counts.eq(1).all():
            return data
    metrics = [
        column
        for column in (
            "mean_seconds",
            "median_seconds",
            "mean_utility_gain",
            "mean_sampled_utility_gain",
            "mean_ranking_utility",
            "mean_babbling_ranking_utility",
            "mean_ranking_utility_gain",
            "mean_pairs",
            "mean_observed_groups",
        )
        if column in data.columns
    ]
    averaged = data.groupby(keys, as_index=False)[metrics].mean()
    counts = (
        data.groupby(keys, as_index=False)["bias_attribute"]
        .nunique()
        .rename(columns={"bias_attribute": "bias_attribute_count"})
    )
    averaged = averaged.merge(counts, on=keys, how="left", validate="one_to_one")
    if "samples" in data.columns:
        samples = data.groupby(keys, as_index=False)["samples"].sum()
        averaged = averaged.merge(
            samples, on=keys, how="left", validate="one_to_one"
        )
    return averaged


def _legend(axis, location: str = "upper left") -> None:
    axis.legend(
        frameon=False,
        ncol=2,
        loc=location,
        fontsize=6.5,
        handlelength=1.7,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def _save(figure, path: Path) -> None:
    figure.tight_layout(pad=0.15)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(
        path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.01
    )


def _utility_linthresh(data: pd.DataFrame) -> float:
    """Return a small linear region that preserves legitimate zero gain."""

    positive = data.loc[data["mean_utility_gain"] > 0, "mean_utility_gain"]
    if positive.empty:
        return 1e-3
    return max(float(positive.min()) / 2.0, 1e-12)


def plot_distributional_results(
    summary: pd.DataFrame | str | Path,
    output_dir: str | Path,
) -> None:
    """Write per-prior plots and a time/utility query-comparison figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(summary) if isinstance(summary, (str, Path)) else summary
    data = average_bias_attributes(data)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    utility_linthresh = _utility_linthresh(data)

    for prior in DEFAULT_PRIORS:
        selected = data[data["prior_model"] == prior]
        if selected.empty:
            continue

        figure, axis = plt.subplots(figsize=(3.15, 2.25))
        _plot_lines(axis, selected, "mean_seconds")
        axis.set_yscale("log")
        _style(axis, "MI Time (s)")
        _legend(axis)
        _save(figure, output / f"mi_execution_time_{prior}")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(3.15, 2.25))
        _plot_lines(axis, selected, "mean_utility_gain")
        if (selected["mean_utility_gain"] > 0).all():
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=utility_linthresh)
        _style(axis, "User Utility Gain")
        _legend(axis)
        _save(figure, output / f"mi_user_utility_{prior}")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(3.15, 2.25))
        _plot_lines(axis, selected, "mean_ranking_utility_gain")
        values = selected["mean_ranking_utility_gain"]
        span = float(values.max()) - float(values.min())
        pad = max(0.02, 0.08 * span)
        axis.set_ylim(
            min(0.0, float(values.min()) - pad),
            max(0.0, float(values.max()) + pad),
        )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        _style(axis, r"Ranking Gain ($\Delta\tau_b$)")
        _legend(axis, "upper right")
        _save(figure, output / f"mi_ranking_utility_gain_{prior}")
        plt.close(figure)

    available = [
        prior for prior in DEFAULT_PRIORS if not data[data["prior_model"] == prior].empty
    ]
    if not available:
        return

    figure, axes = plt.subplots(
        2,
        len(available),
        figsize=(3.15 * len(available), 4.65),
        sharex="col",
        sharey="row",
        squeeze=False,
    )
    for column, prior in enumerate(available):
        selected = data[data["prior_model"] == prior]
        runtime_axis = axes[0, column]
        utility_axis = axes[1, column]

        _plot_lines(runtime_axis, selected, "mean_seconds")
        runtime_axis.set_yscale("log")
        _style(runtime_axis, "MI Time (s)" if column == 0 else "")
        runtime_axis.set_xlabel("")
        runtime_axis.set_title(prior.capitalize(), fontweight="bold", fontsize=11)

        _plot_lines(utility_axis, selected, "mean_utility_gain")
        utility_axis.set_yscale("symlog", linthresh=utility_linthresh)
        _style(
            utility_axis,
            "User Utility Gain" if column == 0 else "",
        )
        utility_axis.set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    figure.supxlabel(
        "Number of Ranking Attributes", fontweight="bold", fontsize=11, y=0.01
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.92), pad=0.45)
    combined = output / "query_time_and_user_utility"
    figure.savefig(
        combined.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        combined.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
