"""Run the Cartesian-attribute version of the bucket experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.bucket_attributes.sweep import STRATEGY_ALIASES

from .experiment import (
    DEFAULT_LAMBDAS,
    DEFAULT_RUNS,
    DEFAULT_STRATEGIES,
    build_workload,
    run_experiment,
    scheme_summary,
    summarize_runtime,
    summarize_utility,
    write_results,
)
from .plot import plot_iq_summaries, plot_results, plot_runtime_vs_m
from .plot_query_metrics_paper import plot_query_metrics
from .scheme import DATASET_ORDER, load_dataset_scheme


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
        default=None,
        help="output directory; defaults to a prior-specific results directory",
    )
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASET_ORDER))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--query-seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--prior-model",
        choices=("uniform", "gaussian-random"),
        default="uniform",
        help="uniform, or a Gaussian whose standard deviation is sampled per run",
    )
    parser.add_argument(
        "--prior-seed",
        type=int,
        default=0,
        help="seed for Gaussian widths shared across datasets",
    )
    parser.add_argument("--lambdas", type=_comma_floats, default=list(DEFAULT_LAMBDAS))
    parser.add_argument(
        "--strategies",
        type=_comma_strings,
        default=["alg1", "fast", "random", "single", "single_any"],
        help="aliases: alg1, fast, random, single, single_any",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="merge selected strategies into compatible raw results already present",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.output_dir is None:
        directory = (
            "cartesian_attributes_gaussian_random"
            if args.prior_model == "gaussian-random"
            else "cartesian_attributes"
        )
        args.output_dir = Path("results") / directory
    unknown = sorted(set(args.datasets) - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    unknown_strategies = sorted(set(args.strategies) - set(STRATEGY_ALIASES))
    if unknown_strategies:
        raise ValueError(f"unknown strategy aliases: {unknown_strategies}")
    strategy_names = [STRATEGY_ALIASES[alias] for alias in args.strategies]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schemes = [load_dataset_scheme(key, args.data_root) for key in args.datasets]
    scheme_summary(schemes).to_csv(args.output_dir / "scheme_summary.csv", index=False)

    completed: list[Path] = []
    for scheme in schemes:
        print(
            f"Building {scheme.label} {args.queries}-query workload...", flush=True
        )
        payload, prefixes = build_workload(
            scheme,
            n_queries=args.queries,
            query_seed=args.query_seed,
        )
        workload_path = args.output_dir / f"{scheme.key}_queries.json"
        workload_path.write_text(json.dumps(payload, indent=2) + "\n")

        output_prefix = args.output_dir / scheme.key
        print(f"Running {scheme.label} Cartesian sweep...", flush=True)
        outputs = run_experiment(
            scheme,
            prefixes,
            payload,
            runs=range(args.runs),
            lambdas=args.lambdas,
            strategies=strategy_names,
            prior_model=args.prior_model,
            prior_seed=args.prior_seed,
            checkpoint_prefix=None if args.merge_existing else output_prefix,
        )
        if args.merge_existing:
            utility_path = Path(f"{output_prefix}_utility.csv")
            runtime_path = Path(f"{output_prefix}_runtime.csv")
            if utility_path.exists() and runtime_path.exists():
                results = pd.concat(
                    [pd.read_csv(utility_path), outputs[0]], ignore_index=True
                ).drop_duplicates(
                    [
                        "dataset",
                        "run",
                        "lambda",
                        "prior_model",
                        "num_attributes",
                        "strategy",
                        "query",
                    ],
                    keep="last",
                )
                timings = pd.concat(
                    [pd.read_csv(runtime_path), outputs[1]], ignore_index=True
                ).drop_duplicates(
                    [
                        "dataset",
                        "run",
                        "lambda",
                        "prior_model",
                        "num_attributes",
                        "strategy",
                    ],
                    keep="last",
                )
                outputs = (
                    results,
                    timings,
                    summarize_utility(results),
                    summarize_runtime(timings),
                )
        write_results(output_prefix, *outputs)
        completed.append(output_prefix)

    if not args.no_plots:
        for prefix in completed:
            plot_results(
                f"{prefix}_utility_summary.csv",
                f"{prefix}_runtime_summary.csv",
                prefix,
            )
        plot_runtime_vs_m(args.output_dir)
        if set(args.datasets) == set(DATASET_ORDER):
            plot_iq_summaries(args.output_dir)
            if "Alg. 1" in strategy_names:
                plot_query_metrics(args.output_dir, args.data_root)
    print(f"Done. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
