"""Merge independently run conflict-learning strength sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .comparison import summarize_comparison
from .scheme import DATASETS


def merge_results(source_dirs: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []

    for dataset in DATASETS:
        run_frames = [pd.read_csv(root / f"{dataset}_runs.csv") for root in source_dirs]
        runtime_frames = [
            pd.read_csv(root / f"{dataset}_runtime.csv") for root in source_dirs
        ]
        runs = (
            pd.concat(run_frames, ignore_index=True, sort=False)
            .sort_values(["lambda", "buckets", "run"])
            .reset_index(drop=True)
        )
        runtimes = (
            pd.concat(runtime_frames, ignore_index=True, sort=False)
            .sort_values(["lambda", "buckets"])
            .reset_index(drop=True)
        )
        summary = summarize_comparison(runs)

        runs.to_csv(output_dir / f"{dataset}_runs.csv", index=False)
        runtimes.to_csv(output_dir / f"{dataset}_runtime.csv", index=False)
        summary.to_csv(output_dir / f"{dataset}_summary.csv", index=False)
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv(
        output_dir / "combined_summary.csv", index=False
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("source_dirs", type=Path, nargs="+")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    merge_results(args.source_dirs, args.output_dir)
    print(f"Merged results are in {args.output_dir}")


if __name__ == "__main__":
    main()
