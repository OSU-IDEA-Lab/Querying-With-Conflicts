"""Resume exact-matching benchmarks for every MI bias realization.

The original utility sweep benchmarked general maximum-weight matching only
for run zero.  This utility-free resume pass fills the remaining timing rows,
checkpointing the CSV after every call so a long run can be restarted safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.conflict_learning.comparison import timed_exact_matching
from experiments.conflict_learning.experiment import ThresholdGrid, dyadic_levels, equilibrium
from experiments.conflict_learning.scheme import DATASETS, load_fixed_domain

from .experiment import _bucket_biases, seeded_biases, summarize


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "mi_fixed_bucketization_iid_lambda_sweep",
    )
    parser.add_argument("--data-root", type=Path, default=root / "data" / "real")
    parser.add_argument("--max-groups", type=int, default=1024)
    parser.add_argument("--domain-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    combined_summaries = []
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
        levels = {level.count: level for level in dyadic_levels(domain.m)}
        path = args.output_dir / f"{dataset}_runtime.csv"
        timings = pd.read_csv(path)
        pending = timings.index[timings["status"].eq("not_measured")].tolist()
        print(f"{dataset}: {len(pending)} timing calls pending", flush=True)

        cached_biases: dict[tuple[int, float], np.ndarray] = {}
        cached_grids: dict[tuple[str, float], ThresholdGrid] = {}
        for completed, index in enumerate(pending, start=1):
            row = timings.loc[index]
            run = int(row["run"])
            strength = float(row["lambda"])
            buckets = int(row["buckets"])
            key = (run, strength)
            if key not in cached_biases:
                cached_biases[key] = seeded_biases(
                    dataset,
                    domain.m,
                    strength,
                    run,
                    args.seed,
                    "iid",
                )
            bucket_biases = _bucket_biases(levels[buckets], cached_biases[key])
            prior_model = str(row.get("prior_model", "uniform"))
            if prior_model == "uniform":
                grid_key = (prior_model, float("nan"))
                grid = ThresholdGrid(domain.m)
            elif prior_model in {"gaussian", "gaussian-random"}:
                prior_variance = float(row["prior_variance"])
                grid_key = (prior_model, prior_variance)
                if grid_key not in cached_grids:
                    cached_grids[grid_key] = ThresholdGrid.gaussian(
                        domain.m, prior_variance
                    )
                grid = cached_grids[grid_key]
            else:
                raise ValueError(f"Unknown prior model in {path}: {prior_model}")

            def weight(left: int, right: int) -> float:
                source = equilibrium(left, right, bucket_biases, grid)
                return 0.0 if source is None else float(grid.v[source[2] - 1])

            _, seconds, status = timed_exact_matching(
                buckets, weight, args.timeout
            )
            timings.loc[index, "matching_seconds"] = seconds
            timings.loc[index, "status"] = status
            timings.to_csv(path, index=False)
            print(
                f"{dataset}: {completed}/{len(pending)} run={run} "
                f"lambda={strength:g} B={buckets} {status} {seconds:.3f}s",
                flush=True,
            )

        results = pd.read_csv(args.output_dir / f"{dataset}_runs.csv")
        summary = summarize(results, timings)
        summary.to_csv(args.output_dir / f"{dataset}_summary.csv", index=False)
        combined_summaries.append(summary)

    if set(args.datasets) == set(DATASETS):
        pd.concat(combined_summaries, ignore_index=True).to_csv(
            args.output_dir / "combined_summary.csv", index=False
        )


if __name__ == "__main__":
    main()
