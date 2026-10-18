"""Run the scaled exact-versus-fast conflict-learning comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import run_dataset_comparison, summarize_comparison
from .plot_exact_paper import plot_exact_paper
from .scheme import DATASETS, load_fixed_domain
from experiments.joint_bias import load_joint_bias_distribution


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASETS))
    parser.add_argument(
        "--lambdas", type=_comma_floats, default=[0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument("--data-root", type=Path, default=root / "data" / "real")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--prior-model",
        choices=("uniform", "gaussian", "gaussian-random"),
        default="uniform",
        help="IID rank prior used to compute influential thresholds and gains.",
    )
    parser.add_argument(
        "--gaussian-std-fraction",
        type=float,
        default=0.25,
        help="Gaussian standard deviation as a fraction of z.",
    )
    parser.add_argument(
        "--prior-seed",
        type=int,
        default=0,
        help="Seed for the per-run Gaussian-width draws.",
    )
    parser.add_argument("--max-groups", type=int, default=2048)
    parser.add_argument("--domain-seed", type=int, default=17)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--intents", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bias-model",
        choices=("sorted", "joint"),
        default="joint",
        help="joint uses three distinct bias attributes and empirical P(A_b|e)",
    )
    parser.add_argument("--exact-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-buckets",
        type=int,
        default=1024,
        help="Largest learning-bucket level to run; the underlying m=z is unchanged.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    unknown = sorted(set(args.datasets) - DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    if args.prior_model == "gaussian" and not 0.0 < args.gaussian_std_fraction:
        raise ValueError("--gaussian-std-fraction must be positive")
    if args.output_dir is None:
        fraction = f"{args.gaussian_std_fraction:g}".replace(".", "p")
        suffix = (
            "conflict_learning_joint_bias"
            if args.bias_model == "joint"
            else "conflict_learning_sorted_bias"
        )
        if args.prior_model == "gaussian":
            suffix += f"_gaussian_std_{fraction}"
        elif args.prior_model == "gaussian-random":
            suffix += "_gaussian_random"
        args.output_dir = Path(__file__).resolve().parents[2] / "results" / suffix
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        domain = load_fixed_domain(
            spec,
            args.data_root,
            max_groups=args.max_groups,
            domain_seed=args.domain_seed,
            cluster_categorical=spec.categorical,
            transform_numeric=True,
        )
        joint_distribution = (
            load_joint_bias_distribution(domain, args.data_root)
            if args.bias_model == "joint"
            else None
        )
        print(
            f"{domain.dataset_label}: {domain.attribute_label}, "
            f"transform={domain.coordinate_transform}, "
            f"full={domain.full_cardinality}, fixed m=z={domain.m}, "
            f"prior={args.prior_model}",
            flush=True,
        )
        prior_variance = (
            (args.gaussian_std_fraction * domain.z) ** 2
            if args.prior_model == "gaussian"
            else None
        )
        dataset_results = []
        dataset_timings = []
        for strength in args.lambdas:
            results, timings = run_dataset_comparison(
                domain,
                lambda_bias=strength,
                prior_model=args.prior_model,
                prior_variance=prior_variance,
                prior_seed=args.prior_seed,
                runs=args.runs,
                intents_per_run=args.intents,
                seed=args.seed,
                exact_timeout_seconds=args.exact_timeout,
                max_learning_buckets=args.max_buckets,
                joint_bias_distribution=joint_distribution,
            )
            dataset_results.append(results)
            dataset_timings.append(timings)

        results = pd.concat(dataset_results, ignore_index=True)
        timings = pd.concat(dataset_timings, ignore_index=True)
        summary = summarize_comparison(results)
        results.to_csv(args.output_dir / f"{dataset}_runs.csv", index=False)
        timings.to_csv(args.output_dir / f"{dataset}_runtime.csv", index=False)
        summary.to_csv(args.output_dir / f"{dataset}_summary.csv", index=False)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(args.output_dir / "combined_summary.csv", index=False)
    if not args.no_plots and set(args.datasets) == set(DATASETS):
        plot_exact_paper(args.output_dir)
    print(f"Done. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
