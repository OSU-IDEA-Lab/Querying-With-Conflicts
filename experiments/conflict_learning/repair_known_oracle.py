"""Backfill the exact sorted known-bias oracle into completed comparison CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .comparison import known_bias_matching, summarize_comparison
from .experiment import ThresholdGrid, answer_matrix, equilibrium, mean_tau, pair_gain
from .plot_exact_paper import plot_exact_paper
from .scheme import DATASETS, load_fixed_domain


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data" / "real")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "conflict_learning_scaled",
    )
    parser.add_argument("--max-groups", type=int, default=2048)
    parser.add_argument("--domain-seed", type=int, default=17)
    parser.add_argument("--intents", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    summaries = []
    for dataset, spec in DATASETS.items():
        runs_path = args.output_dir / f"{dataset}_runs.csv"
        runtime_path = args.output_dir / f"{dataset}_runtime.csv"
        if not runs_path.exists():
            continue
        results = pd.read_csv(runs_path)
        timings = pd.read_csv(runtime_path)
        domain = load_fixed_domain(
            spec,
            args.data_root,
            max_groups=args.max_groups,
            domain_seed=args.domain_seed,
            cluster_categorical=spec.categorical,
            transform_numeric=True,
        )
        z = domain.z
        grid = ThresholdGrid(z)
        for strength in sorted(results["lambda"].unique()):
            biases = float(strength) * domain.coordinates
            started = time.perf_counter()
            matching = known_bias_matching(biases)
            oracle_seconds = time.perf_counter() - started
            pairs = []
            for u, v in matching:
                source = equilibrium(u, v, biases, grid)
                if source is not None:
                    pairs.append((source, source))
            cache = {}
            known_gain = float(
                sum(
                    pair_gain(z, biases, user, source, "fixed", cache)
                    for user, source in pairs
                )
            )
            utility_constant = float(
                domain.m * (z * z - 1.0) / 12.0 + np.sum(biases**2)
            )
            strength_mask = np.isclose(results["lambda"], strength)
            results.loc[strength_mask, "utility_known_gain"] = known_gain
            results.loc[strength_mask, "utility_known_raw"] = (
                known_gain - utility_constant
            )
            for run in sorted(results.loc[strength_mask, "run"].unique()):
                rng = np.random.default_rng(args.seed + 1000 + int(run))
                intents = rng.integers(
                    1, z + 1, size=(args.intents, domain.m)
                )
                tau = mean_tau(
                    intents, answer_matrix(z, biases, pairs, intents, "fixed")
                )
                mask = strength_mask & (results["run"] == run)
                results.loc[mask, "tau_known"] = tau
            timing_mask = np.isclose(timings["lambda"], strength)
            timings.loc[timing_mask, "known_matching_seconds"] = oracle_seconds
            timings.loc[timing_mask, "known_matching_status"] = "complete"
            print(
                f"{dataset} lambda={strength:g}: known oracle backfilled",
                flush=True,
            )

        results.to_csv(runs_path, index=False)
        timings.to_csv(runtime_path, index=False)
        summary = summarize_comparison(results)
        summary.to_csv(args.output_dir / f"{dataset}_summary.csv", index=False)
        summaries.append(summary)

    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_dir / "combined_summary.csv", index=False
    )
    plot_exact_paper(args.output_dir)


if __name__ == "__main__":
    main()
