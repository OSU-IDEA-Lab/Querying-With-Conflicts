"""Compact, single-metric figures for LaTeX papers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ORDER = ("amazon", "pricerunner", "flights", "compas", "census")
GRID = "#e5e5e5"
ORANGE = "#eb6834"
BLUE = "#34549c"
GRAY = "#777777"
LINE_WIDTH = 3.2


def _ticks(values: list[int], limit: int = 5) -> list[int]:
    ordered = sorted(set(values))
    if ordered[-1] >= 1024:
        return [value for value in (1, 8, 64, ordered[-1]) if value in ordered]
    if ordered[-1] == 65:
        return [value for value in (1, 4, 16, 65) if value in ordered]
    if len(ordered) <= limit:
        return ordered
    stride = math.ceil((len(ordered) - 1) / (limit - 1))
    selected = ordered[::stride]
    if selected[-1] != ordered[-1]:
        if len(selected) > 1 and ordered[-1] / selected[-1] < 1.35:
            selected.pop()
        selected.append(ordered[-1])
    return selected


def _compact_number(value: float, _position: int | None = None) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= scale:
            scaled = value / scale
            return f"{scaled:g}{suffix}"
    return f"{value:g}"


def _style(axis, ylabel: str, buckets: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Number of Buckets", fontweight="bold", fontsize=11)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=11)
    axis.set_xticks(_ticks(buckets))
    axis.set_xticklabels([str(value) for value in _ticks(buckets)])
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


def plot_paper_metrics(
    output_dir: str | Path,
    *,
    main_lambda: float | None = 0.25,
) -> None:
    """Write five datasets times three metrics as separate PDF/PNG figures.

    When ``main_lambda`` is ``None``, every curve gives equal weight to each
    available bias strength at a bucket level.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    root = Path(output_dir)
    paper_dir = root / "paper_plots"
    paper_dir.mkdir(parents=True, exist_ok=True)
    aggregated_rows = []

    for dataset in DATASET_ORDER:
        source = root / f"{dataset}_summary.csv"
        data = pd.read_csv(source)
        data = data[data["buckets"] <= 1024].copy()
        bias_scale = data["lambda"] * data["z"]
        data["bias_error_mean_fraction"] = data["bias_error_mean_mean"] / bias_scale
        data["bias_error_max_fraction"] = data["bias_error_max_mean"] / bias_scale
        data["bias_bound_fraction"] = data["bias_bound_mean"] / bias_scale
        data["utility_recovery"] = (
            data["utility_certified_exact_gain_mean"]
            / data["utility_known_gain_mean"]
        )
        utility_curves = data.copy()
        if main_lambda is None:
            data = (
                data.groupby("buckets", as_index=False)
                .agg(
                    bias_error_mean_fraction=("bias_error_mean_fraction", "mean"),
                    bias_error_max_fraction=("bias_error_max_fraction", "mean"),
                    bias_bound_fraction=("bias_bound_fraction", "mean"),
                    probe_queries_mean=("probe_queries_mean", "mean"),
                    utility_recovery=("utility_recovery", "mean"),
                    lambda_count=("lambda", "nunique"),
                )
            )
            data.insert(0, "dataset", dataset)
            aggregated_rows.append(data.copy())
        else:
            data = data[data["lambda"].sub(main_lambda).abs() < 1e-12].copy()
        data = data.sort_values("buckets")
        buckets = data["buckets"].astype(int).tolist()

        # 1. Effect on learned relative bias.
        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        axis.plot(
            data["buckets"], data["bias_error_mean_fraction"],
            color=BLUE, linewidth=LINE_WIDTH, label="mean",
        )
        axis.plot(
            data["buckets"], data["bias_error_max_fraction"],
            color=BLUE, linewidth=LINE_WIDTH, linestyle="--", label="maximum",
        )
        axis.plot(
            data["buckets"], data["bias_bound_fraction"],
            color=GRAY, linewidth=LINE_WIDTH, linestyle="-.", label=r"$\frac{1}{2}+\epsilon$",
        )
        axis.set_ylim(bottom=0.0)
        _style(axis, r"Error / ($\lambda z$)", buckets)
        axis.legend(
            frameon=False, loc="best", fontsize=7.5,
            handlelength=2.1, handletextpad=0.45, borderaxespad=0.2,
        )
        _save(figure, paper_dir / f"{dataset}_bias")
        plt.close(figure)

        # 2. Effect on the number of conflict-learning probes.
        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        axis.plot(
            data["buckets"], data["probe_queries_mean"],
            color=BLUE, linewidth=LINE_WIDTH,
        )
        axis.set_yscale("symlog", base=10, linthresh=1.0)
        axis.yaxis.set_major_formatter(FuncFormatter(_compact_number))
        _style(axis, "Probe Queries", buckets)
        _save(figure, paper_dir / f"{dataset}_queries")
        plt.close(figure)

        # 3. Effect on utility after matched-pair equilibrium certification.
        figure, axis = plt.subplots(figsize=(2.65, 2.05))
        if main_lambda is None:
            colors = ("#34549c", "#2eb16d", "#eb6834", "#8f4aa8")
            for color, strength in zip(
                colors, sorted(utility_curves["lambda"].unique())
            ):
                curve = utility_curves[
                    utility_curves["lambda"].sub(strength).abs() < 1e-12
                ].sort_values("buckets")
                axis.plot(
                    curve["buckets"],
                    curve["utility_recovery"],
                    color=color,
                    linewidth=LINE_WIDTH,
                    label=rf"$\lambda={strength:g}$",
                )
            axis.axhline(0.0, color="#777777", linewidth=0.8)
            axis.set_ylim(
                min(-0.05, float(utility_curves["utility_recovery"].min()) - 0.05),
                1.04,
            )
        else:
            axis.plot(
                data["buckets"], data["utility_recovery"],
                color=ORANGE, linewidth=LINE_WIDTH, label="MI",
            )
            axis.set_ylim(0.0, 1.04)
        _style(axis, "Utility Recovery", buckets)
        axis.legend(
            frameon=False, loc="lower right", fontsize=6.8,
            handlelength=1.7, handletextpad=0.35, borderaxespad=0.2,
        )
        _save(figure, paper_dir / f"{dataset}_utility")
        plt.close(figure)

    if aggregated_rows:
        pd.concat(aggregated_rows, ignore_index=True).to_csv(
            root / "lambda_aggregated_paper_summary.csv", index=False
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--lambda-bias", type=float, default=0.25)
    parser.add_argument("--aggregate-lambdas", action="store_true")
    arguments = parser.parse_args()
    plot_paper_metrics(
        arguments.output_dir,
        main_lambda=None if arguments.aggregate_lambdas else arguments.lambda_bias,
    )
