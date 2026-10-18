"""Plots for the fixed-domain conflict-learning experiment."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


INK = "#111111"
GRID = "#e5e5e5"
DATASET_STYLE = {
    "amazon": ("Amazon", "#34549c", "-"),
    "pricerunner": ("PriceRunner", "#c94f45", "--"),
    "flights": ("Flights", "#2eb16d", ":"),
    "compas": ("COMPAS", "#9a4caf", "-."),
    "census": ("Census", "#d99718", (0, (3, 1, 1, 1))),
}


def _ticks(values: list[int], limit: int = 7) -> list[int]:
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


def _style(axis, xlabel: str, ylabel: str, xscale: str | None = "log") -> None:
    if xscale is not None:
        axis.set_xscale(xscale)
    axis.set_xlabel(xlabel, fontweight="bold")
    axis.set_ylabel(ylabel, fontweight="bold")
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", alpha=0.25, color=GRID)
    axis.tick_params(axis="both", which="both", length=0)
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _line_with_band(
    axis,
    data,
    mean: str,
    std: str,
    color: str,
    style,
    label: str,
    x: str = "buckets",
):
    values = data[mean]
    spread = data[std].fillna(0.0)
    axis.fill_between(
        data[x], values - spread, values + spread,
        color=color, alpha=0.12, linewidth=0,
    )
    axis.plot(
        data[x], values,
        color=color, linestyle=style, linewidth=2.2, label=label,
    )


def plot_dataset(summary_path: str | Path, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(summary_path).sort_values("buckets")
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))

    axis = axes[0, 0]
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axhline(data["utility_known_gain_mean"].iloc[0], color="#555555", linestyle="--")
    _line_with_band(
        axis, data,
        "utility_certified_gain_mean", "utility_certified_gain_std",
        "#1baf7a", "-", "certified",
    )
    _style(axis, "Learning Buckets", "Quadratic Utility Gain")

    axis = axes[0, 1]
    axis.axhline(data["tau_no_query_mean"].iloc[0], color="#777777", linewidth=0.8)
    axis.axhline(data["tau_known_mean"].iloc[0], color="#555555", linestyle="--")
    _line_with_band(
        axis, data, "tau_certified_mean", "tau_certified_std",
        "#1baf7a", "-", "certified",
    )
    _style(axis, "Learning Buckets", "Kendall τ-b")

    axis = axes[0, 2]
    axis.plot(data["buckets"], data["probe_queries_mean"], color="#34549c", linewidth=2.2)
    axis.set_yscale("log")
    _style(axis, "Learning Buckets", "Probe Queries")

    axis = axes[1, 0]
    axis.plot(data["buckets"], data["bias_error_mean_mean"], color="#34549c", linewidth=2.2, label="mean")
    axis.plot(data["buckets"], data["bias_error_max_mean"], color="#34549c", linewidth=1.8, linestyle="--", label="maximum")
    axis.plot(data["buckets"], data["bias_bound_mean"], color="#777777", linewidth=1.4, linestyle="-.", label="½ + ε")
    _style(axis, "Learning Buckets", "Relative-bias Error")

    axis = axes[1, 1]
    axis.plot(data["buckets"], data["bias_coverage_mean"], color="#9a4caf", linewidth=2.2)
    axis.set_ylim(0.0, 1.02)
    _style(axis, "Learning Buckets", "Bound Coverage")

    axis = axes[1, 2]
    axis.plot(data["buckets"], data["seconds_median"], color="#c94f45", linewidth=2.2)
    axis.set_yscale("log")
    _style(axis, "Learning Buckets", "Learning + Matching Time (s)")

    ticks = _ticks(data["buckets"].astype(int).tolist())
    for axis in axes.flat:
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(value) for value in ticks])
        axis.minorticks_off()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output = Path(output_path)
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_dataset_paper(summary_path: str | Path, output_path: str | Path) -> None:
    """Paper figure: certified utility, probe cost, and learned-bias error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(summary_path).sort_values("buckets").copy()
    known = data["utility_known_gain_mean"].abs()
    data["certified_recovery"] = data["utility_certified_gain_mean"] / known
    data["certified_recovery_std"] = data["utility_certified_gain_std"] / known

    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))

    axis = axes[0]
    axis.axhline(1.0, color="#666666", linewidth=1.4, linestyle="--", label="known bias")
    _line_with_band(
        axis,
        data,
        "certified_recovery",
        "certified_recovery_std",
        "#1baf7a",
        "-",
        "certified",
    )
    _style(axis, "Number of Learning Buckets", "Utility Gain / Known-Bias Gain")
    axis.legend(frameon=False, loc="best", prop={"weight": "bold"})

    axis = axes[1]
    _line_with_band(
        axis,
        data,
        "probe_queries_mean",
        "probe_queries_std",
        "#34549c",
        "-",
        "learning + certification",
    )
    axis.set_yscale("log")
    _style(axis, "Number of Learning Buckets", "Probe Queries")
    axis.legend(frameon=False, loc="best", prop={"weight": "bold"})

    axis = axes[2]
    axis.plot(
        data["buckets"],
        data["bias_error_mean_mean"],
        color="#34549c",
        linewidth=2.2,
        label="mean error",
    )
    axis.plot(
        data["buckets"],
        data["bias_error_max_mean"],
        color="#34549c",
        linewidth=1.8,
        linestyle="--",
        label="maximum error",
    )
    axis.plot(
        data["buckets"],
        data["bias_bound_mean"],
        color="#777777",
        linewidth=1.5,
        linestyle="-.",
        label="½ + ε bound",
    )
    _style(axis, "Number of Learning Buckets", "Relative-Bias Error")
    axis.legend(frameon=False, loc="best", prop={"weight": "bold"})

    ticks = _ticks(data["buckets"].astype(int).tolist())
    for axis in axes:
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(value) for value in ticks])
        axis.minorticks_off()

    figure.tight_layout(pad=0.35, w_pad=1.0)
    output = Path(output_path)
    figure.savefig(
        output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def plot_combined(output_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    frames = []
    for dataset in DATASET_STYLE:
        path = root / f"{dataset}_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            # Backward-compatible with completed runs created before epsilon
            # was emitted as a first-class result column.
            if "epsilon_mean" not in frame:
                frame["epsilon_mean"] = frame["bias_bound_mean"] - 0.5
                frame["epsilon_std"] = frame["bias_bound_std"]
            if "bucket_width_groups" not in frame:
                frame["bucket_width_groups"] = np.ceil(
                    frame["m"] / frame["buckets"]
                ).astype(int)
            frames.append(frame)
            plot_dataset_paper(
                path,
                root / f"{dataset}_utility_bias_vs_buckets",
            )
    if not frames:
        raise FileNotFoundError(f"No conflict-learning summaries in {root}")
    combined = pd.concat(frames, ignore_index=True)
    combined["utility_recovery_mean"] = (
        combined["utility_certified_gain_mean"]
        / combined["utility_known_gain_mean"]
    )
    combined["utility_recovery_std"] = (
        combined["utility_certified_gain_std"]
        / combined["utility_known_gain_mean"].abs()
    )
    combined["utility_robust_recovery_mean"] = (
        combined["utility_robust_gain_mean"] / combined["utility_known_gain_mean"]
    )
    combined["utility_robust_recovery_std"] = (
        combined["utility_robust_gain_std"]
        / combined["utility_known_gain_mean"].abs()
    )
    combined.to_csv(root / "combined_summary.csv", index=False)

    plots = [
        ("utility_gain", "utility_certified_gain_mean", "utility_certified_gain_std", "Quadratic Utility Gain"),
        ("utility_recovery", "utility_recovery_mean", "utility_recovery_std", "Fraction of Known-Bias Utility Gain"),
        ("kendall_tau", "tau_certified_mean", "tau_certified_std", "Kendall τ-b"),
        ("probe_queries", "probe_queries_mean", "probe_queries_std", "Probe Queries"),
        ("bias_error", "bias_error_mean_mean", "bias_error_mean_std", "Mean Relative-bias Error"),
        ("runtime", "seconds_median", "seconds_std", "Learning + Matching Time (s)"),
    ]
    for stem, mean, std, ylabel in plots:
        figure, axis = plt.subplots(figsize=(7.6, 5.0))
        for dataset, (label, color, linestyle) in DATASET_STYLE.items():
            data = combined[combined["dataset"] == dataset].sort_values("buckets")
            if data.empty:
                continue
            _line_with_band(axis, data, mean, std, color, linestyle, label)
        if stem in {"probe_queries", "runtime"}:
            axis.set_yscale("log")
        ticks = _ticks(combined["buckets"].astype(int).tolist())
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(value) for value in ticks])
        axis.minorticks_off()
        _style(axis, "Number of Learning Buckets", ylabel)
        axis.legend(frameon=False, ncol=2, loc="best", prop={"weight": "bold"})
        figure.tight_layout(pad=0.25)
        figure.savefig(root / f"conflict_{stem}_vs_buckets.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(root / f"conflict_{stem}_vs_buckets.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)

    epsilon_plots = [
        ("utility_gain", "utility_certified_gain_mean", "utility_certified_gain_std", "Quadratic Utility Gain"),
        ("utility_recovery", "utility_recovery_mean", "utility_recovery_std", "Fraction of Known-Bias Utility Gain"),
        ("kendall_tau", "tau_certified_mean", "tau_certified_std", "Kendall τ-b"),
        ("probe_queries", "probe_queries_mean", "probe_queries_std", "Probe Queries"),
        ("runtime", "seconds_median", "seconds_std", "Learning + Matching Time (s)"),
    ]
    epsilon_ticks = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    epsilon_max = float(combined["epsilon_mean"].max())
    epsilon_ticks = [value for value in epsilon_ticks if value <= max(0.1, epsilon_max * 1.05)]
    for stem, mean, std, ylabel in epsilon_plots:
        figure, axis = plt.subplots(figsize=(7.6, 5.0))
        for dataset, (label, color, linestyle) in DATASET_STYLE.items():
            data = combined[combined["dataset"] == dataset].sort_values("epsilon_mean")
            if data.empty:
                continue
            _line_with_band(
                axis, data, mean, std, color, linestyle, label, x="epsilon_mean"
            )
        axis.set_xscale("symlog", base=2, linthresh=0.1)
        axis.set_xticks(epsilon_ticks)
        axis.set_xticklabels([f"{value:g}" for value in epsilon_ticks])
        if stem in {"probe_queries", "runtime"}:
            axis.set_yscale("log")
        axis.minorticks_off()
        _style(axis, "Bias Uncertainty ε", ylabel, xscale=None)
        axis.legend(frameon=False, ncol=2, loc="best", prop={"weight": "bold"})
        figure.tight_layout(pad=0.25)
        figure.savefig(root / f"conflict_{stem}_vs_epsilon.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(root / f"conflict_{stem}_vs_epsilon.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)

    robust_plots = [
        (
            "utility_gain",
            "utility_robust_gain_mean",
            "utility_robust_gain_std",
            "Learned Robust Quadratic Utility Gain",
        ),
        (
            "utility_recovery",
            "utility_robust_recovery_mean",
            "utility_robust_recovery_std",
            "Robust Utility Gain / Known-Bias Gain",
        ),
        (
            "kendall_tau",
            "tau_robust_mean",
            "tau_robust_std",
            "Learned Robust Kendall τ-b",
        ),
    ]
    for stem, mean, std, ylabel in robust_plots:
        for x, xlabel, suffix in (
            ("buckets", "Number of Learning Buckets", "buckets"),
            ("epsilon_mean", "Bias Uncertainty ε", "epsilon"),
            (
                "bucket_width_groups",
                "Maximum True Groups per Learning Bucket",
                "bucket_width",
            ),
        ):
            figure, axis = plt.subplots(figsize=(7.6, 5.0))
            for dataset, (label, color, linestyle) in DATASET_STYLE.items():
                data = combined[combined["dataset"] == dataset].sort_values(x)
                if data.empty:
                    continue
                _line_with_band(
                    axis, data, mean, std, color, linestyle, label, x=x
                )
            if x in {"buckets", "bucket_width_groups"}:
                ticks = _ticks(combined[x].astype(int).tolist())
                axis.set_xticks(ticks)
                axis.set_xticklabels([str(value) for value in ticks])
                _style(axis, xlabel, ylabel)
            else:
                axis.set_xscale("symlog", base=2, linthresh=0.1)
                axis.set_xticks(epsilon_ticks)
                axis.set_xticklabels([f"{value:g}" for value in epsilon_ticks])
                _style(axis, xlabel, ylabel, xscale=None)
            axis.minorticks_off()
            axis.legend(
                frameon=False, ncol=2, loc="best", prop={"weight": "bold"}
            )
            figure.tight_layout(pad=0.25)
            output = root / f"conflict_robust_{stem}_vs_{suffix}"
            figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
            figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
            plt.close(figure)

    shared_width_plots = [
        ("probe_queries", "probe_queries_mean", "probe_queries_std", "Probe Queries"),
        ("runtime", "seconds_median", "seconds_std", "Learning + Matching Time (s)"),
    ]
    width_ticks = _ticks(combined["bucket_width_groups"].astype(int).tolist())
    for stem, mean, std, ylabel in shared_width_plots:
        figure, axis = plt.subplots(figsize=(7.6, 5.0))
        for dataset, (label, color, linestyle) in DATASET_STYLE.items():
            data = combined[combined["dataset"] == dataset].sort_values(
                "bucket_width_groups"
            )
            if data.empty:
                continue
            _line_with_band(
                axis,
                data,
                mean,
                std,
                color,
                linestyle,
                label,
                x="bucket_width_groups",
            )
        axis.set_yscale("log")
        axis.set_xticks(width_ticks)
        axis.set_xticklabels([str(value) for value in width_ticks])
        axis.minorticks_off()
        _style(axis, "Maximum True Groups per Learning Bucket", ylabel)
        axis.legend(frameon=False, ncol=2, loc="best", prop={"weight": "bold"})
        figure.tight_layout(pad=0.25)
        output = root / f"conflict_{stem}_vs_bucket_width"
        figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
