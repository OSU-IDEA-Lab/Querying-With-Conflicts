"""Run fixed-domain MI bucketization for all five datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.conflict_learning.scheme import DATASETS, load_fixed_domain
from experiments.joint_bias import load_joint_bias_distribution

from .experiment import run_fixed_domain_mi, summarize
from .plot import plot_results


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=root / "data" / "real")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to a prior-specific results directory",
    )
    parser.add_argument("--max-groups", type=int, default=1024)
    parser.add_argument("--domain-seed", type=int, default=17)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--lambda-bias", type=float, default=0.25)
    parser.add_argument(
        "--lambdas",
        type=_comma_floats,
        default=None,
        help="comma-separated bias strengths; overrides --lambda-bias",
    )
    parser.add_argument(
        "--bias-model",
        choices=("sorted", "iid", "joint"),
        default="joint",
        help="joint uses the empirical conditional distribution of three bias attributes",
    )
    parser.add_argument(
        "--prior-model",
        choices=("uniform", "gaussian", "gaussian-random"),
        default="uniform",
        help="rank prior used by MI's posterior and edge-weight calculations",
    )
    parser.add_argument(
        "--prior-variance",
        type=float,
        default=None,
        help="Gaussian kernel variance parameter in squared-rank units",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prior-seed", type=int, default=0)
    parser.add_argument("--exact-timeout", type=float, default=600.0)
    parser.add_argument(
        "--skip-general-timing",
        action="store_true",
        help="time the exact adjacent solver instead of the generic blossom solver",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args(argv)
    strengths = (
        arguments.lambdas
        if arguments.lambdas is not None
        else [arguments.lambda_bias]
    )
    if not strengths or any(strength <= 0 for strength in strengths):
        raise ValueError("Bias strengths must be positive")
    if arguments.prior_model == "gaussian":
        if arguments.prior_variance is None:
            parser.error("--prior-variance is required with --prior-model gaussian")
        if arguments.prior_variance <= 0:
            parser.error("--prior-variance must be positive")
    elif arguments.prior_variance is not None:
        parser.error("--prior-variance is only valid with --prior-model gaussian")

    if arguments.output_dir is None:
        directory_name = "mi_fixed_bucketization"
        if arguments.prior_model == "gaussian":
            directory_name += f"_gaussian_var_{arguments.prior_variance:g}"
        elif arguments.prior_model == "gaussian-random":
            directory_name += "_gaussian_random_lambda_sweep"
        if arguments.bias_model == "joint":
            directory_name += "_joint_bias"
        arguments.output_dir = root / "results" / directory_name

    unknown = sorted(set(arguments.datasets) - DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []

    for dataset in arguments.datasets:
        spec = DATASETS[dataset]
        domain = load_fixed_domain(
            spec,
            arguments.data_root,
            max_groups=arguments.max_groups,
            domain_seed=arguments.domain_seed,
            cluster_categorical=spec.categorical,
            transform_numeric=True,
        )
        joint_distribution = (
            load_joint_bias_distribution(domain, arguments.data_root)
            if arguments.bias_model == "joint"
            else None
        )
        dataset_results = []
        dataset_timings = []
        for strength in strengths:
            results, timings = run_fixed_domain_mi(
                domain,
                lambda_bias=strength,
                runs=arguments.runs,
                seed=arguments.seed,
                exact_timeout_seconds=arguments.exact_timeout,
                bias_model=arguments.bias_model,
                joint_bias_distribution=joint_distribution,
                prior_model=arguments.prior_model,
                prior_variance=arguments.prior_variance,
                prior_seed=arguments.prior_seed,
                benchmark_general=not arguments.skip_general_timing,
                progress=not arguments.quiet,
            )
            dataset_results.append(results)
            dataset_timings.append(timings)
        results = pd.concat(dataset_results, ignore_index=True)
        timings = pd.concat(dataset_timings, ignore_index=True)
        summary = summarize(results, timings)
        results.to_csv(arguments.output_dir / f"{dataset}_runs.csv", index=False)
        timings.to_csv(arguments.output_dir / f"{dataset}_runtime.csv", index=False)
        summary.to_csv(arguments.output_dir / f"{dataset}_summary.csv", index=False)
        combined.append(summary)

    pd.concat(combined, ignore_index=True).to_csv(
        arguments.output_dir / "combined_summary.csv", index=False
    )
    if not arguments.no_plots and set(arguments.datasets) == set(DATASETS):
        plot_results(arguments.output_dir)


if __name__ == "__main__":
    main()
