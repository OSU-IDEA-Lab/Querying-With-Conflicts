"""Paper plots for trustworthy-answer detection scalability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_ORDER = ("amazon", "pricerunner", "flights", "census", "compas")
GRID = "#e5e5e5"
LINE_WIDTH = 2.4
DATASET_STYLES = {
    "amazon": ("#222222", "o"),
    "pricerunner": ("#d73027", "s"),
    "flights": ("#00a6d6", "^"),
    "census": ("#c49a00", "D"),
    "compas": ("#777777", "P"),
}


PANEL_SPECS = (
    ("uniform", "distinct", "Uniform: distinct"),
    ("uniform", "coinciding", "Uniform: coinciding"),
    ("gaussian", "distinct", "Gaussian: distinct"),
    ("gaussian", "coinciding", "Gaussian: coinciding"),
)


def _style_axis(axis, *, title: str | None = None) -> None:
    if title is not None:
        axis.set_title(title, fontweight="bold", fontsize=10)
    axis.set_yscale("log")
    axis.set_xlabel("Dataset Size (%)", fontweight="bold", fontsize=10)
    axis.set_ylabel("Time (s)", fontweight="bold", fontsize=10)
    axis.set_xticks((0.5, 2, 4, 6, 8, 10))
    axis.set_xticklabels(("0.5", "2", "4", "6", "8", "10"))
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=8.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_panel(axis, data: pd.DataFrame) -> None:
    for dataset in DATASET_ORDER:
        selected = data[data["dataset"] == dataset].sort_values(
            "dataset_size_percent"
        )
        if selected.empty:
            continue
        axis.plot(
            selected["dataset_size_percent"],
            selected["mean_seconds"],
            color=DATASET_STYLES[dataset][0],
            linewidth=LINE_WIDTH,
            marker=DATASET_STYLES[dataset][1],
            markevery=2,
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=DATASET_STYLES[dataset][0],
            markeredgewidth=0.9,
            label=str(selected["dataset_label"].iloc[0]),
        )


def _save(figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(
        destination.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.01,
    )


def plot_four_panels(source: str | Path, output: str | Path) -> None:
    """Write four individual panels and one combined preview."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for prior, setting, title in PANEL_SPECS:
        selected = data[
            data["prior_model"].eq(prior)
            & data["attribute_setting"].eq(setting)
        ]
        if selected.empty:
            continue
        figure, axis = plt.subplots(figsize=(3.15, 2.25))
        _draw_panel(axis, selected)
        _style_axis(axis)
        axis.legend(
            frameon=False,
            ncol=2,
            loc="upper left",
            fontsize=6.5,
            handlelength=1.7,
            columnspacing=0.65,
            handletextpad=0.35,
            borderaxespad=0.2,
        )
        figure.tight_layout(pad=0.15)
        _save(figure, output / f"detection_time_{prior}_{setting}")
        plt.close(figure)

    available_specs = [
        spec
        for spec in PANEL_SPECS
        if not data[
            data["prior_model"].eq(spec[0])
            & data["attribute_setting"].eq(spec[1])
        ].empty
    ]
    if not available_specs:
        return
    figure, axes = plt.subplots(
        1,
        len(available_specs),
        figsize=(3.15 * len(available_specs), 2.35),
        squeeze=False,
    )
    axes = axes[0]
    for axis, (prior, setting, title) in zip(axes, available_specs):
        selected = data[
            data["prior_model"].eq(prior)
            & data["attribute_setting"].eq(setting)
        ]
        _draw_panel(axis, selected)
        _style_axis(axis, title=title)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7.5,
    )
    figure.tight_layout(pad=0.2, w_pad=1.0)
    _save(figure, output / "detection_time_all")
    plt.close(figure)


def plot_scalability(source: str | Path, output: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(source)
    figure, axis = plt.subplots(figsize=(3.15, 2.25))
    for dataset in DATASET_ORDER:
        selected = data[data["dataset"] == dataset].sort_values("dataset_size_percent")
        axis.plot(
            selected["dataset_size_percent"],
            selected["time_seconds"],
            color=DATASET_STYLES[dataset][0],
            linewidth=LINE_WIDTH,
            marker=DATASET_STYLES[dataset][1],
            markevery=2,
            markersize=3.2,
            markerfacecolor="white",
            markeredgecolor=DATASET_STYLES[dataset][0],
            markeredgewidth=0.9,
            label=str(selected["dataset_label"].iloc[0]),
        )

    axis.set_yscale("log")
    axis.set_xlabel("Dataset Size (%)", fontweight="bold", fontsize=11)
    axis.set_ylabel("Time (s)", fontweight="bold", fontsize=11)
    axis.set_xticks((0.5, 2, 4, 6, 8, 10))
    axis.set_xticklabels(("0.5", "2", "4", "6", "8", "10"))
    axis.grid(False, axis="x", which="both")
    axis.grid(True, axis="y", color=GRID, linewidth=0.75)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5)
    axis.minorticks_off()
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=6.5,
        handlelength=1.7,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.2,
    )

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(pad=0.15)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    figure.savefig(
        destination.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(figure)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    arguments = parser.parse_args()
    plot_scalability(arguments.source, arguments.output)
