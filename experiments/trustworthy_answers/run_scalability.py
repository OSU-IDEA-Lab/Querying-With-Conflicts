"""Run trustworthy-answer detection on the existing five-dataset setup."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.cartesian_attributes.scheme import DATASET_ORDER, load_dataset_scheme

from .experiment import (
    DEFAULT_FRACTIONS_PERCENT,
    DEFAULT_BIAS_ATTRIBUTE_COUNT,
    DEFAULT_K,
    DEFAULT_LAMBDAS,
    DEFAULT_PRIORS,
    DEFAULT_RUNS,
    DEFAULT_SETTINGS,
    run_scalability_experiment,
    write_results,
)
from .plot_scalability import plot_four_panels


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in _comma_strings(value)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/real"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trustworthy_answers_joint"),
    )
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASET_ORDER))
    parser.add_argument(
        "--fractions-percent",
        type=_comma_floats,
        default=list(DEFAULT_FRACTIONS_PERCENT),
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--lambdas", type=_comma_floats, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--priors", type=_comma_strings, default=list(DEFAULT_PRIORS))
    parser.add_argument("--settings", type=_comma_strings, default=list(DEFAULT_SETTINGS))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bias-seed", type=int, default=0)
    parser.add_argument(
        "--bias-attributes",
        type=int,
        default=DEFAULT_BIAS_ATTRIBUTE_COUNT,
        help="number of distinct attributes in the joint bias function",
    )
    parser.add_argument("--gaussian-prior-seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    unknown = sorted(set(args.datasets) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for dataset in args.datasets:
        scheme = load_dataset_scheme(dataset, args.data_root)
        frame = run_scalability_experiment(
            scheme,
            fractions_percent=args.fractions_percent,
            k=args.k,
            priors=args.priors,
            settings=args.settings,
            runs=range(args.runs),
            lambdas=args.lambdas,
            sample_seed=args.sample_seed,
            bias_seed=args.bias_seed,
            gaussian_prior_seed=args.gaussian_prior_seed,
            bias_attribute_count=args.bias_attributes,
        )
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(
            args.output_dir / "checkpoint.csv", index=False
        )

    results = pd.concat(frames, ignore_index=True)
    write_results(args.output_dir, results)
    if not args.no_plots:
        plot_four_panels(
            args.output_dir / "summary.csv", args.output_dir / "paper_plots"
        )
    print(f"Done. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
