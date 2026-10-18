"""Compact combined-dataset MI plots for the attribute-scalability queries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.bucket_attributes.sweep import kendall_tau_b_unique_intent

from .experiment import build_prefix_domains, make_component_biases, tuple_biases
from .scheme import DATASET_ORDER, load_dataset_scheme


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
    axis.set_xlabel("Number of Attributes", fontweight="bold", fontsize=11)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=11)
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _legend(axis, location: str) -> None:
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


def _plot_lines(axis, summary: pd.DataFrame, metric: str) -> None:
    for dataset in DATASET_ORDER:
        data = summary[summary["dataset"] == dataset].sort_values("num_attributes")
        axis.plot(
            data["num_attributes"],
            data[metric],
            color=DATASET_STYLES[dataset][0],
            linewidth=LINE_WIDTH,
            marker=DATASET_STYLES[dataset][1],
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=DATASET_STYLES[dataset][0],
            markeredgewidth=1.0,
            label=str(data["dataset_label"].iloc[0]),
        )


def ranking_gain_pairs(
    dataset: str,
    utility: pd.DataFrame,
    workload_path: Path,
    data_root: Path,
) -> pd.DataFrame:
    """Pair each MI Kendall score with its deterministic Babbling baseline."""
    scheme = load_dataset_scheme(dataset, data_root)
    prefixes = build_prefix_domains(scheme)
    workload = json.loads(workload_path.read_text())
    selected = utility[utility["strategy"] == "Alg. 1"]
    baseline_rows: list[dict[str, float | int | str]] = []

    for run in sorted(selected["run"].unique()):
        components = make_component_biases(scheme, run=int(run))
        for strength in sorted(selected["lambda"].unique()):
            for prefix in prefixes:
                biases = tuple_biases(prefix, components, lam=float(strength))
                groups = prefix.domain_size
                babbling = np.full(groups, (groups + 1) / 2.0) + biases
                intent = {
                    "ASC": np.arange(1, groups + 1),
                    "DESC": np.arange(groups, 0, -1),
                }
                for query_index, query in enumerate(workload["queries"]):
                    direction = query["direction"]
                    present = np.asarray(
                        query["groups_by_k"][str(prefix.num_attributes)],
                        dtype=np.int64,
                    )
                    baseline_rows.append(
                        {
                            "dataset": dataset,
                            "run": int(run),
                            "lambda": float(strength),
                            "num_attributes": prefix.num_attributes,
                            "query": query_index,
                            "query_groups": len(present),
                            "babbling_tau": kendall_tau_b_unique_intent(
                                babbling[present], intent[direction][present]
                            ),
                        }
                    )

    keys = ["dataset", "run", "lambda", "num_attributes", "query"]
    paired = selected[keys + ["ranking_utility"]].merge(
        pd.DataFrame(baseline_rows), on=keys, validate="one_to_one"
    )
    paired["ranking_utility_gain"] = (
        paired["ranking_utility"] - paired["babbling_tau"]
    )
    return paired


def _ranking_gain(
    dataset: str,
    utility: pd.DataFrame,
    workload_path: Path,
    data_root: Path,
) -> pd.DataFrame:
    """Average Kendall tau-b gain over queries, runs, and bias strengths."""
    paired = ranking_gain_pairs(dataset, utility, workload_path, data_root)
    cells = (
        paired.groupby(["run", "lambda", "num_attributes"], as_index=False)[
            "ranking_utility_gain"
        ]
        .mean()
    )
    return cells.groupby("num_attributes", as_index=False)[
        "ranking_utility_gain"
    ].mean()


def plot_query_metrics(
    results_dir: str | Path,
    data_root: str | Path = "data/real",
) -> None:
    """Write MI time, user utility, and Kendall gain across 1--3 attributes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    output = root / "query_paper_plots"
    output.mkdir(parents=True, exist_ok=True)
    time_rows: list[pd.DataFrame] = []
    user_rows: list[pd.DataFrame] = []
    ranking_rows: list[pd.DataFrame] = []

    for dataset in DATASET_ORDER:
        runtime = pd.read_csv(root / f"{dataset}_runtime.csv")
        utility = pd.read_csv(root / f"{dataset}_utility.csv")
        mi_runtime = runtime[runtime["strategy"] == "Alg. 1"]
        label = str(runtime["dataset_label"].iloc[0])
        time = (
            mi_runtime.groupby("num_attributes", as_index=False)["seconds"]
            .mean()
            .rename(columns={"seconds": "time_seconds"})
        )
        user = (
            mi_runtime.groupby("num_attributes", as_index=False)["query_utility"]
            .mean()
            .rename(columns={"query_utility": "user_utility"})
        )
        ranking = _ranking_gain(
            dataset,
            utility,
            root / f"{dataset}_queries.json",
            Path(data_root),
        )
        for frame in (time, user, ranking):
            frame.insert(0, "dataset_label", label)
            frame.insert(0, "dataset", dataset)
        time_rows.append(time)
        user_rows.append(user)
        ranking_rows.append(ranking)

    times = pd.concat(time_rows, ignore_index=True)
    users = pd.concat(user_rows, ignore_index=True)
    rankings = pd.concat(ranking_rows, ignore_index=True)
    times.to_csv(output / "mi_execution_time.csv", index=False)
    users.to_csv(output / "mi_user_utility.csv", index=False)
    rankings.to_csv(output / "mi_ranking_utility_gain.csv", index=False)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, times, "time_seconds")
    axis.set_yscale("log")
    _style(axis, "MI Time (s)")
    _legend(axis, "upper left")
    _save(figure, output / "mi_execution_time")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, users, "user_utility")
    axis.set_yscale("log")
    _style(axis, "User Utility Gain")
    _legend(axis, "upper left")
    _save(figure, output / "mi_user_utility")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, rankings, "ranking_utility_gain")
    span = float(rankings["ranking_utility_gain"].max()) - float(
        rankings["ranking_utility_gain"].min()
    )
    pad = max(0.02, 0.08 * span)
    low = min(0.0, float(rankings["ranking_utility_gain"].min()) - pad)
    high = max(0.0, float(rankings["ranking_utility_gain"].max()) + pad)
    axis.set_ylim(low, high)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    _style(axis, r"Ranking Utility Gain ($\Delta\tau_b$)")
    _legend(axis, "best")
    _save(figure, output / "mi_ranking_utility_gain")
    plt.close(figure)

    # One compact figure for the complete query-scalability experiment.  Every
    # panel overlays all datasets so prior, runtime, and utility trends can be
    # compared without moving between figures.
    figure, axes = plt.subplots(1, 3, figsize=(9.45, 2.45))
    _plot_lines(axes[0], times, "time_seconds")
    axes[0].set_yscale("log")
    _style(axes[0], "MI Time (s)")
    axes[0].set_title("Runtime", fontweight="bold", fontsize=10)

    _plot_lines(axes[1], users, "user_utility")
    axes[1].set_yscale("log")
    _style(axes[1], "User Utility Gain")
    axes[1].set_title("User utility", fontweight="bold", fontsize=10)

    _plot_lines(axes[2], rankings, "ranking_utility_gain")
    axes[2].set_ylim(low, high)
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    _style(axes[2], r"Ranking Utility Gain ($\Delta\tau_b$)")
    axes[2].set_title("Ranking utility", fontweight="bold", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        fontsize=8,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), pad=0.35)
    combined = output / "mi_query_scalability_combined"
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--data-root", default="data/real")
    arguments = parser.parse_args()
    plot_query_metrics(arguments.results_dir, arguments.data_root)
