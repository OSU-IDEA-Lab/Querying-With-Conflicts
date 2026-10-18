"""Preview MI metrics against domain and query-result cardinality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .plot import DATASET_COLORS
from .plot_query_metrics_paper import ranking_gain_pairs
from .scheme import DATASET_ORDER


GRID = "#e5e5e5"
LINE_WIDTH = 2.6


def _style(axis, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel, fontweight="bold", fontsize=11)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=11)
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _legend(axis, location: str = "best") -> None:
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


def _plot_lines(axis, data: pd.DataFrame, x: str, y: str) -> None:
    for dataset in DATASET_ORDER:
        curve = data[data["dataset"] == dataset].sort_values(x)
        axis.plot(
            curve[x],
            curve[y],
            color=DATASET_COLORS[dataset],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            label=str(curve["dataset_label"].iloc[0]),
        )


def plot_domain_metrics(
    results_dir: str | Path,
    data_root: str | Path = "data/real",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    output = root / "query_domain_preview"
    output.mkdir(parents=True, exist_ok=True)
    times: list[pd.DataFrame] = []
    users: list[pd.DataFrame] = []
    gains: list[pd.DataFrame] = []

    for dataset in DATASET_ORDER:
        runtime = pd.read_csv(root / f"{dataset}_runtime.csv")
        utility = pd.read_csv(root / f"{dataset}_utility.csv")
        selected = runtime[runtime["strategy"] == "Alg. 1"]
        time = (
            selected.groupby(
                ["dataset", "dataset_label", "observed_groups"], as_index=False
            )["seconds"]
            .median()
            .rename(columns={"seconds": "time_seconds"})
        )
        user = (
            selected.groupby(
                ["dataset", "dataset_label", "observed_groups"], as_index=False
            )["query_utility"]
            .mean()
            .rename(columns={"query_utility": "user_utility"})
        )
        paired = ranking_gain_pairs(
            dataset,
            utility,
            root / f"{dataset}_queries.json",
            Path(data_root),
        )
        per_query = (
            paired.groupby(["query", "query_groups"], as_index=False)[
                "ranking_utility_gain"
            ]
            .mean()
        )
        per_query["size_bin"] = np.floor(np.log2(per_query["query_groups"])).astype(int)
        gain = (
            per_query.groupby("size_bin", as_index=False)
            .agg(
                query_groups=("query_groups", "median"),
                ranking_utility_gain=("ranking_utility_gain", "mean"),
                queries=("query", "count"),
            )
        )
        gain.insert(0, "dataset_label", str(runtime["dataset_label"].iloc[0]))
        gain.insert(0, "dataset", dataset)
        times.append(time)
        users.append(user)
        gains.append(gain)

    time_data = pd.concat(times, ignore_index=True)
    user_data = pd.concat(users, ignore_index=True)
    gain_data = pd.concat(gains, ignore_index=True)
    time_data.to_csv(output / "mi_time_vs_m.csv", index=False)
    user_data.to_csv(output / "mi_user_utility_vs_m.csv", index=False)
    gain_data.to_csv(output / "mi_kendall_gain_vs_query_groups.csv", index=False)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, time_data, "observed_groups", "time_seconds")
    axis.set_xscale("log")
    axis.set_yscale("log")
    _style(axis, "Observed Groups ($m$)", "MI Time (s)")
    _legend(axis, "upper left")
    _save(figure, output / "mi_time_vs_m")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, user_data, "observed_groups", "user_utility")
    axis.set_xscale("log")
    axis.set_yscale("log")
    _style(axis, "Observed Groups ($m$)", "User Utility")
    _legend(axis, "upper left")
    _save(figure, output / "mi_user_utility_vs_m")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    _plot_lines(axis, gain_data, "query_groups", "ranking_utility_gain")
    axis.set_xscale("log")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    _style(axis, "Groups Returned by Query", r"Kendall Gain ($\Delta\tau_b$)")
    _legend(axis, "upper right")
    _save(figure, output / "mi_kendall_gain_vs_query_groups")
    plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--data-root", default="data/real")
    arguments = parser.parse_args()
    plot_domain_metrics(arguments.results_dir, arguments.data_root)
