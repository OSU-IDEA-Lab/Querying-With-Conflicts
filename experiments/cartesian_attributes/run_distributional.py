"""Run the distribution-aware Cartesian intent-attribute experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .distributional import (
    DEFAULT_BIAS_ATTRIBUTE_COUNT,
    DEFAULT_GAUSSIAN_STD_FRACTION,
    DEFAULT_INTENTS,
    DEFAULT_LAMBDAS,
    DEFAULT_PRIORS,
    DEFAULT_RUNS,
    DEFAULT_SAMPLE_FRACTION,
    DEFAULT_ZIPF_EXPONENT,
    run_distributional_experiment,
    summarize_distributional_results,
)
from .plot_distributional import (
    average_bias_attributes,
    plot_distributional_results,
)
from .scheme import DATASET_ORDER, load_dataset_scheme


SCALABILITY_PRIORS = ("uniform", "gaussian")


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in _comma_strings(value)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/cartesian_attributes_distributional_distinct_bias"
        ),
    )
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASET_ORDER))
    parser.add_argument(
        "--priors", type=_comma_strings, default=list(SCALABILITY_PRIORS)
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--lambdas", type=_comma_floats, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--sample-fraction", type=float, default=DEFAULT_SAMPLE_FRACTION)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bias-seed", type=int, default=0)
    parser.add_argument(
        "--bias-attributes",
        type=int,
        default=DEFAULT_BIAS_ATTRIBUTE_COUNT,
        help="number of configured distinct attributes in the joint bias function",
    )
    parser.add_argument("--intents", type=int, default=DEFAULT_INTENTS)
    parser.add_argument("--intent-seed", type=int, default=0)
    parser.add_argument("--zipf-exponent", type=float, default=DEFAULT_ZIPF_EXPONENT)
    parser.add_argument(
        "--gaussian-std-fraction",
        type=float,
        default=DEFAULT_GAUSSIAN_STD_FRACTION,
    )
    parser.add_argument(
        "--gaussian-std-mode",
        choices=("fixed", "random"),
        default="random",
        help="draw sigma/z once per run, or use --gaussian-std-fraction",
    )
    parser.add_argument("--gaussian-prior-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    unknown_datasets = sorted(set(args.datasets) - set(DATASET_ORDER))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    unknown_priors = sorted(set(args.priors) - set(DEFAULT_PRIORS))
    if unknown_priors:
        raise ValueError(f"unknown priors: {unknown_priors}")
    if args.runs < 1:
        raise ValueError("runs must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for dataset in args.datasets:
        scheme = load_dataset_scheme(dataset, args.data_root)
        frame = run_distributional_experiment(
            scheme,
            priors=args.priors,
            runs=range(args.runs),
            lambdas=args.lambdas,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            bias_seed=args.bias_seed,
            intents=args.intents,
            intent_seed=args.intent_seed,
            zipf_exponent=args.zipf_exponent,
            gaussian_std_fraction=args.gaussian_std_fraction,
            gaussian_std_mode=args.gaussian_std_mode,
            gaussian_prior_seed=args.gaussian_prior_seed,
            bias_attribute_count=args.bias_attributes,
            batch_size=args.batch_size,
        )
        frame.to_csv(args.output_dir / f"{dataset}_raw.csv", index=False)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize_distributional_results(raw)
    plot_summary = average_bias_attributes(summary)
    raw.to_csv(args.output_dir / "all_results.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_summary.to_csv(args.output_dir / "plot_summary.csv", index=False)
    if not args.no_plots:
        plot_distributional_results(plot_summary, args.output_dir / "paper_plots")
    print(f"Done. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
