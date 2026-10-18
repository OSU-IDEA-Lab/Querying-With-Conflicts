"""Run fixed-domain conflict learning for every configured dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_dataset, summarize, write_results
from .plot import plot_combined, plot_dataset
from .scheme import DATASETS, load_fixed_domain


def _comma_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_comma_strings, default=list(DATASETS))
    parser.add_argument("--data-root", type=Path, default=root / "data" / "real")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "conflict_learning")
    parser.add_argument("--max-groups", type=int, default=1024)
    parser.add_argument("--domain-seed", type=int, default=17)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--intents", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    unknown = sorted(set(args.datasets) - DATASETS.keys())
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        domain = load_fixed_domain(
            DATASETS[dataset],
            args.data_root,
            max_groups=args.max_groups,
            domain_seed=args.domain_seed,
        )
        print(
            f"{domain.dataset_label}: {domain.attribute_label}, "
            f"full={domain.full_cardinality}, fixed m=z={domain.m}",
            flush=True,
        )
        results, timings = run_dataset(
            domain,
            alpha=args.alpha,
            runs=args.runs,
            intents_per_run=args.intents,
            seed=args.seed,
        )
        summary = summarize(results, timings)
        write_results(args.output_dir, dataset, results, timings, summary)
        if not args.no_plots:
            plot_dataset(
                args.output_dir / f"{dataset}_summary.csv",
                args.output_dir / f"{dataset}_overview",
            )

    if not args.no_plots and set(args.datasets) == set(DATASETS):
        plot_combined(args.output_dir)
    print(f"Done. Results are in {args.output_dir}")


if __name__ == "__main__":
    main()
