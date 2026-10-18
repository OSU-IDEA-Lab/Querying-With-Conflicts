"""Compact per-dataset MI sensitivity plots for learned-bias strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("amazon", "pricerunner", "flights", "compas", "census")
LAMBDA_COLORS = ("#34549c", "#2eb16d", "#eb6834", "#8f4aa8")
GRID = "#e5e5e5"


def _ticks(values: list[int]) -> list[int]:
    maximum = max(values)
    if maximum >= 1024:
        return [value for value in (1, 8, 64, 1024) if value in values]
    if maximum == 65:
        return [value for value in (1, 4, 16, 65) if value in values]
    return values


def _style(axis, buckets: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Number of Buckets", fontweight="bold", fontsize=11)
    axis.set_ylabel("Utility Recovery", fontweight="bold", fontsize=11)
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


def plot_mi_lambda_sensitivity(
    results_dir: str | Path,
    *,
    strategy: str = "certified",
) -> None:
    """Plot exact MI utility recovery, separated by bias strength."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_dir)
    paper = root / "paper_plots"
    paper.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / "combined_summary.csv")
    data = data[data["buckets"] <= 1024].copy()
    strategy_columns = {
        "certified": ("utility_certified_exact_gain_mean", "mi", "MI"),
        "robust": ("utility_robust_exact_gain_mean", "robust", r"$z-1$"),
        "dhat": ("utility_dhat_exact_gain_mean", "dhat", r"$\hat{\delta}$"),
    }
    try:
        utility_column, filename_prefix, legend_title = strategy_columns[strategy]
    except KeyError as error:
        raise ValueError(f"Unknown strategy: {strategy}") from error
    data["utility_recovery"] = (
        data[utility_column] / data["utility_known_gain_mean"]
    )
    strengths = sorted(data["lambda"].dropna().unique().tolist())
    lower = min(-0.05, float(data["utility_recovery"].min()) - 0.05)

    for dataset in DATASETS:
        selected = data[data["dataset"] == dataset].copy()
        buckets = sorted(selected["buckets"].astype(int).unique().tolist())
        if not buckets:
            continue

        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        for color, strength in zip(LAMBDA_COLORS, strengths):
            curve = selected[np.isclose(selected["lambda"], strength)].sort_values(
                "buckets"
            )
            axis.plot(
                curve["buckets"],
                curve["utility_recovery"],
                color=color,
                linewidth=2.4,
                label=rf"$\lambda={strength:g}$",
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(lower, 1.04)
        _style(axis, buckets)
        axis.legend(
            frameon=False,
            loc="lower right",
            fontsize=6.8,
            handlelength=1.7,
            handletextpad=0.35,
            borderaxespad=0.2,
            title=legend_title,
            title_fontproperties={"weight": "bold", "size": 7.2},
        )
        _save(figure, paper / f"{filename_prefix}_{dataset}_lambda_sensitivity")
        plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument(
        "--strategy", choices=("certified", "robust", "dhat"), default="certified"
    )
    arguments = parser.parse_args()
    plot_mi_lambda_sensitivity(arguments.results_dir, strategy=arguments.strategy)
