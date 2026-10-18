"""Paper plots for the scaled exact-versus-fast comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INK = "#111111"
GRID = "#e5e5e5"
EXACT = "#34549c"
FAST = "#c94f45"
ROBUST = "#2eb16d"
KNOWN = "#666666"


def _ticks(values: list[int], stride: int = 2) -> list[int]:
    ordered = sorted(set(values))
    selected = ordered[::stride]
    if selected[-1] != ordered[-1]:
        if len(selected) > 1 and ordered[-1] / selected[-1] <= 2.1:
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


def _recovery_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    denominator = result["utility_known_gain_mean"]
    for method in ("dhat_exact", "dhat_fast", "robust_exact"):
        result[f"{method}_recovery_mean"] = (
            result[f"utility_{method}_gain_mean"] / denominator
        )
        result[f"{method}_recovery_std"] = (
            result[f"utility_{method}_gain_std"] / denominator.abs()
        )
    return result


def _finite_range(values: list[np.ndarray], pad: float = 0.04) -> tuple[float, float]:
    finite = np.concatenate([array[np.isfinite(array)] for array in values])
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, 1e-9)
    return low - pad * span, high + pad * span


def plot_comparison(
    output_dir: str | Path,
    *,
    main_lambda: float = 0.25,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(output_dir)
    summaries = []
    runtimes = []
    for path in sorted(root.glob("*_summary.csv")):
        if path.name == "combined_summary.csv":
            continue
        summaries.append(pd.read_csv(path))
        runtime_path = root / path.name.replace("_summary.csv", "_runtime.csv")
        if runtime_path.exists():
            runtimes.append(pd.read_csv(runtime_path))
    if not summaries:
        raise FileNotFoundError(f"No comparison summaries in {root}")
    summary = _recovery_columns(pd.concat(summaries, ignore_index=True))
    runtime = pd.concat(runtimes, ignore_index=True)
    main = summary[np.isclose(summary["lambda"], main_lambda)].copy()

    utility_ylim = _finite_range(
        [
            main["dhat_exact_recovery_mean"].to_numpy(float),
            main["dhat_fast_recovery_mean"].to_numpy(float),
            main["robust_exact_recovery_mean"].to_numpy(float),
            np.ones(len(main)),
        ]
    )
    tau_ylim = _finite_range(
        [
            main["tau_dhat_exact_mean"].to_numpy(float),
            main["tau_dhat_fast_mean"].to_numpy(float),
            main["tau_robust_exact_mean"].to_numpy(float),
            main["tau_known_mean"].to_numpy(float),
        ]
    )
    bias_ylim = _finite_range(
        [
            main["bias_error_mean_normalized_mean"].to_numpy(float),
            main["bias_error_max_normalized_mean"].to_numpy(float),
            main["bias_bound_normalized_mean"].to_numpy(float),
        ]
    )
    bias_ylim = (max(0.0, bias_ylim[0]), bias_ylim[1])

    for dataset in sorted(main["dataset"].unique()):
        data = main[main["dataset"] == dataset].sort_values("buckets")
        times = runtime[
            (runtime["dataset"] == dataset)
            & np.isclose(runtime["lambda"], main_lambda)
        ].sort_values("buckets")
        figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.6))

        axis = axes[0]
        axis.axhline(1.0, color=KNOWN, linestyle="--", linewidth=1.4, label="known bias")
        _band(axis, data, "dhat_exact_recovery_mean", "dhat_exact_recovery_std", EXACT, "-", r"$\hat{\delta}$ + exact")
        _band(axis, data, "dhat_fast_recovery_mean", "dhat_fast_recovery_std", FAST, "--", r"$\hat{\delta}$ + fast")
        _band(axis, data, "robust_exact_recovery_mean", "robust_exact_recovery_std", ROBUST, ":", r"$z-1$ + exact")
        axis.set_ylim(*utility_ylim)
        _style(axis, "Utility Gain / Known-Bias Gain")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        axis = axes[1]
        axis.plot(data["buckets"], data["tau_known_mean"], color=KNOWN, linestyle="--", linewidth=1.4, label="known bias")
        _band(axis, data, "tau_dhat_exact_mean", "tau_dhat_exact_std", EXACT, "-", r"$\hat{\delta}$ + exact")
        _band(axis, data, "tau_dhat_fast_mean", "tau_dhat_fast_std", FAST, "--", r"$\hat{\delta}$ + fast")
        _band(axis, data, "tau_robust_exact_mean", "tau_robust_exact_std", ROBUST, ":", r"$z-1$ + exact")
        axis.set_ylim(*tau_ylim)
        _style(axis, "Kendall τ-b")

        axis = axes[2]
        axis.plot(data["buckets"], data["bias_error_mean_normalized_mean"], color=EXACT, linewidth=2.2, label="mean error")
        axis.plot(data["buckets"], data["bias_error_max_normalized_mean"], color=EXACT, linestyle="--", linewidth=1.8, label="maximum error")
        axis.plot(data["buckets"], data["bias_bound_normalized_mean"], color=KNOWN, linestyle="-.", linewidth=1.5, label=r"$(1/2+\epsilon)/z$")
        axis.set_ylim(*bias_ylim)
        _style(axis, "Normalized Relative-Bias Error")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        axis = axes[3]
        axis.plot(times["buckets"], times["exact_matching_seconds"], color=EXACT, linewidth=2.2, label="exact matching")
        axis.plot(times["buckets"], times["fast_matching_seconds"], color=FAST, linestyle="--", linewidth=2.2, label="fast matching")
        axis.set_yscale("log")
        _style(axis, "Matching Time (s)")
        axis.legend(frameon=False, loc="best", prop={"weight": "bold", "size": 8})

        ticks = _ticks(data["buckets"].astype(int).tolist())
        for axis in axes:
            axis.set_xticks(ticks)
            axis.set_xticklabels([str(value) for value in ticks])
        figure.tight_layout(pad=0.3, w_pad=0.8)
        stem = root / f"{dataset}_exact_fast_lambda_{str(main_lambda).replace('.', 'p')}"
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)

        strengths = sorted(summary[summary["dataset"] == dataset]["lambda"].unique())
        figure, axes = plt.subplots(1, len(strengths), figsize=(4.4 * len(strengths), 3.5), squeeze=False)
        for axis, strength in zip(axes[0], strengths):
            subset = summary[
                (summary["dataset"] == dataset)
                & np.isclose(summary["lambda"], strength)
            ].sort_values("buckets")
            axis.axhline(1.0, color=KNOWN, linestyle="--", linewidth=1.4)
            _band(axis, subset, "dhat_exact_recovery_mean", "dhat_exact_recovery_std", EXACT, "-", r"$\hat{\delta}$ + exact")
            _band(axis, subset, "dhat_fast_recovery_mean", "dhat_fast_recovery_std", FAST, "--", r"$\hat{\delta}$ + fast")
            _style(axis, f"Utility Recovery (λ={strength:g})")
            ticks = _ticks(subset["buckets"].astype(int).tolist())
            axis.set_xticks(ticks)
            axis.set_xticklabels([str(value) for value in ticks])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, frameon=False, ncol=2, loc="upper center", prop={"weight": "bold"})
        figure.tight_layout(rect=(0, 0, 1, 0.91), pad=0.3)
        stem = root / f"{dataset}_lambda_sensitivity"
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
