"""Three-panel paper plots for the scaled conflict-learning experiment."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


GRID = "#e5e5e5"
KNOWN = "#666666"
ROBUST = "#eb6834"
DHAT = "#1baf7a"
ERROR = "#34549c"


def _ticks(values: list[int], limit: int = 6) -> list[int]:
    ordered = sorted(set(values))
    if len(ordered) <= limit:
        return ordered
    stride = math.ceil((len(ordered) - 1) / (limit - 1))
    selected = ordered[::stride]
    if selected[-1] != ordered[-1]:
        if len(selected) > 1 and ordered[-1] / selected[-1] < 1.35:
            selected.pop()
        selected.append(ordered[-1])
    return selected


def _style(axis, ylabel: str) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Number of Learning Buckets", fontweight="bold")
    axis.set_ylabel(ylabel, fontweight="bold")
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", alpha=0.25, color=GRID)
    axis.tick_params(axis="both", which="both", length=0)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _band(axis, data, mean: str, std: str, color: str, style, label: str) -> None:
    values = data[mean].astype(float)
    spread = data[std].fillna(0.0).astype(float)
    axis.fill_between(
        data["buckets"], values - spread, values + spread,
        color=color, alpha=0.12, linewidth=0,
    )
    axis.plot(
        data["buckets"], values,
        color=color, linestyle=style, linewidth=2.2, label=label,
    )


def plot_exact_paper(
    output_dir: str | Path,
    *,
    main_lambda: float = 0.25,
) -> None:
    """Create one exact-matching, three-panel figure for every dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    summaries = [
        pd.read_csv(path)
        for path in sorted(root.glob("*_summary.csv"))
        if path.name != "combined_summary.csv"
    ]
    if not summaries:
        raise FileNotFoundError(f"No comparison summaries in {root}")

    summary = pd.concat(summaries, ignore_index=True)
    main = summary[summary["lambda"].sub(main_lambda).abs() < 1e-12].copy()
    denominator = main["utility_known_gain_mean"]
    main["certified_exact_recovery_mean"] = (
        main["utility_certified_exact_gain_mean"] / denominator
    )
    main["certified_exact_recovery_std"] = (
        main["utility_certified_exact_gain_std"] / denominator.abs()
    )
    main["certified_fast_recovery_mean"] = (
        main["utility_certified_fast_gain_mean"] / denominator
    )
    main["certified_fast_recovery_std"] = (
        main["utility_certified_fast_gain_std"] / denominator.abs()
    )

    for dataset in sorted(main["dataset"].unique()):
        data = main[main["dataset"] == dataset].sort_values("buckets")
        figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))

        axis = axes[0]
        axis.axhline(
            1.0, color=KNOWN, linewidth=1.4, linestyle="--", label="known bias"
        )
        _band(
            axis, data,
            "certified_exact_recovery_mean", "certified_exact_recovery_std",
            ROBUST, "-", "certified, exact matching",
        )
        _style(axis, "Utility Gain / Known-Bias Gain")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        axis = axes[1]
        axis.axhline(
            1.0, color=KNOWN, linewidth=1.4, linestyle="--", label="known bias"
        )
        _band(
            axis, data,
            "certified_fast_recovery_mean", "certified_fast_recovery_std",
            DHAT, "--", "certified, fast matching",
        )
        _style(axis, "Utility Gain / Known-Bias Gain")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        axis = axes[2]
        axis.plot(
            data["buckets"], data["bias_error_mean_mean"],
            color=ERROR, linewidth=2.2, label="mean error",
        )
        axis.plot(
            data["buckets"], data["bias_error_max_mean"],
            color=ERROR, linewidth=1.8, linestyle="--", label="maximum error",
        )
        axis.plot(
            data["buckets"], data["bias_bound_mean"],
            color="#777777", linewidth=1.5, linestyle="-.", label="½ + ε bound",
        )
        axis.set_ylim(bottom=0.0)
        _style(axis, "Relative-Bias Error")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        ticks = _ticks(data["buckets"].astype(int).tolist())
        for axis in axes:
            axis.set_xticks(ticks)
            axis.set_xticklabels([str(value) for value in ticks])

        figure.tight_layout(pad=0.3, w_pad=0.8)
        stem = root / f"{dataset}_utility_bias_lambda_{str(main_lambda).replace('.', 'p')}"
        figure.savefig(
            stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
        )
        figure.savefig(
            stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
        )
        plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--lambda-bias", type=float, default=0.25)
    arguments = parser.parse_args()
    plot_exact_paper(arguments.output_dir, main_lambda=arguments.lambda_bias)
