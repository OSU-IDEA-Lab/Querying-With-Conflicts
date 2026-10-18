"""Compact paper plots for fixed-domain MI bucketization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("amazon", "pricerunner", "flights", "compas", "census")
ORANGE = "#eb6834"
LAMBDA_COLORS = ("#34549c", "#2eb16d", "#eb6834", "#8f4aa8")
GRID = "#e5e5e5"
LINE_WIDTH = 3.2


def _ticks(values: list[int]) -> list[int]:
    maximum = max(values)
    if maximum >= 1024:
        return [value for value in (1, 8, 64, 1024) if value in values]
    if maximum == 65:
        return [value for value in (1, 4, 16, 65) if value in values]
    return values


def _style(axis, ylabel: str, buckets: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Number of Buckets", fontweight="bold", fontsize=11)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=11)
    ticks = _ticks(buckets)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _save(figure, path: Path) -> None:
    figure.tight_layout(pad=0.15)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(
        path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.01
    )


def plot_results(results_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    paper = root / "paper_plots"
    paper.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(root / "combined_summary.csv")
    models = combined["bias_model"].dropna().unique().tolist()
    if len(models) != 1:
        raise ValueError("plot_results expects exactly one bias model per directory")
    strengths = sorted(combined["lambda"].dropna().unique().tolist())
    suffix = ""
    if models[0] == "seeded-iid-uniform":
        suffix = "_iid_sweep" if len(strengths) > 1 else "_iid"
    elif models[0] == "joint-conditional-iid-uniform":
        suffix = "_joint"
    aggregated = (
        combined.groupby(["dataset", "dataset_label", "buckets"], as_index=False)
        .agg(
            utility_recovery_mean=("utility_recovery_mean", "mean"),
            matching_seconds_mean=("matching_seconds_mean", "mean"),
        )
    )
    aggregated.to_csv(root / "lambda_aggregated_summary.csv", index=False)
    positive_times = combined.loc[
        combined["matching_seconds_mean"] > 0, "matching_seconds_mean"
    ]
    time_limits = (
        (
            float(positive_times.min()) * 0.5,
            float(positive_times.max()) * 2.0,
        )
        if not positive_times.empty
        else None
    )
    recovery_min = float(aggregated["utility_recovery_mean"].min())
    recovery_lower = min(0.0, recovery_min - 0.05 * max(1.0, abs(recovery_min)))

    for dataset in DATASETS:
        data = aggregated[aggregated["dataset"] == dataset].sort_values("buckets")
        if data.empty:
            continue
        buckets = data["buckets"].astype(int).tolist()

        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        axis.plot(
            data["buckets"], data["utility_recovery_mean"],
            color=ORANGE, linewidth=LINE_WIDTH, label="MI",
        )
        axis.set_ylim(recovery_lower, 1.04)
        _style(axis, "Utility Recovery", buckets)
        axis.legend(
            frameon=False, loc="lower right", fontsize=7.5,
            handlelength=2.1, handletextpad=0.45, borderaxespad=0.2,
        )
        _save(figure, paper / f"mi_bucket_{dataset}{suffix}_utility")
        plt.close(figure)

        if time_limits is not None:
            figure, axis = plt.subplots(figsize=(2.65, 2.05))
            axis.plot(
                data["buckets"], data["matching_seconds_mean"],
                color=ORANGE, linewidth=LINE_WIDTH, label="MI",
            )
            axis.set_yscale("log")
            axis.set_ylim(*time_limits)
            _style(axis, "Time (s)", buckets)
            axis.legend(
                frameon=False, loc="upper left", fontsize=7.5,
                handlelength=2.1, handletextpad=0.45, borderaxespad=0.2,
            )
            _save(figure, paper / f"mi_bucket_{dataset}{suffix}_time")
            plt.close(figure)

        if len(strengths) > 1:
            figure, axis = plt.subplots(figsize=(2.65, 2.05))
            for color, strength in zip(LAMBDA_COLORS, strengths):
                sensitivity = combined[
                    (combined["dataset"] == dataset)
                    & np.isclose(combined["lambda"], strength)
                ].sort_values("buckets")
                axis.plot(
                    sensitivity["buckets"],
                    sensitivity["utility_recovery_mean"],
                    color=color,
                    linewidth=2.4,
                    label=rf"$\lambda={strength:g}$",
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_ylim(
                min(-0.05, float(combined["utility_recovery_mean"].min()) - 0.05),
                1.04,
            )
            _style(axis, "Utility Recovery", buckets)
            axis.legend(
                frameon=False, loc="lower right", fontsize=6.8,
                handlelength=1.7, handletextpad=0.35, borderaxespad=0.2,
            )
            _save(
                figure,
                paper / f"mi_bucket_{dataset}{suffix}_lambda_sensitivity",
            )
            plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    args = parser.parse_args()
    plot_results(args.results_dir)
