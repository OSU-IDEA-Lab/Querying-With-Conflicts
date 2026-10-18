"""Run MI on nested buckets and evaluate utility on the fixed full domain."""

from __future__ import annotations

from typing import Any, Literal
import time
import zlib

import numpy as np
import pandas as pd

from experiments.conflict_learning.comparison import (
    known_bias_matching,
    timed_exact_matching,
)
from experiments.conflict_learning.experiment import (
    BucketLevel,
    Pair,
    ThresholdGrid,
    dyadic_levels,
    equilibrium,
    pair_gain,
)
from experiments.conflict_learning.scheme import FixedDomain
from experiments.joint_bias import JointBiasDistribution, joint_expected_biases


def seeded_sorted_biases(
    dataset: str,
    m: int,
    lambda_bias: float,
    run: int,
    seed: int,
) -> np.ndarray:
    """Draw fresh monotone biases while preserving the attribute order."""
    bias_seed = zlib.crc32(
        f"{dataset}|{seed}|{run}|fixed-domain-mi-bias".encode()
    )
    draws = np.random.default_rng(bias_seed).uniform(0.0, lambda_bias * m, size=m)
    return np.sort(draws)


def seeded_biases(
    dataset: str,
    m: int,
    lambda_bias: float,
    run: int,
    seed: int,
    bias_model: Literal["sorted", "iid"],
) -> np.ndarray:
    """Return paired draws for the smooth and no-order bias conditions.

    Both conditions use the same seeded uniform values.  The sorted condition
    assigns their order statistics along the attribute.  The IID condition
    preserves the draw order and shifts the values into [-lambda*m, 0].  The
    shift does not affect pairwise relative biases, so differences between the
    conditions are caused only by attribute--bias alignment.
    """
    bias_seed = zlib.crc32(
        f"{dataset}|{seed}|{run}|fixed-domain-mi-bias".encode()
    )
    width = lambda_bias * m
    draws = np.random.default_rng(bias_seed).uniform(0.0, width, size=m)
    if bias_model == "sorted":
        return np.sort(draws)
    if bias_model == "iid":
        return draws - width
    raise ValueError(f"Unknown bias model: {bias_model}")


def shared_gaussian_std_fraction(run: int, seed: int = 0) -> float:
    """Draw the Gaussian width used by the other random-prior MI sweeps."""
    draw_seed = zlib.crc32(f"query-gaussian-prior|{seed}|{run}".encode())
    draw = float(np.random.default_rng(draw_seed).uniform())
    return max(draw, np.finfo(float).eps)


def _bucket_biases(level: BucketLevel, biases: np.ndarray) -> np.ndarray:
    return np.asarray(
        [biases[start:stop].mean() for start, stop in level.intervals],
        dtype=float,
    )


def _mi_matching(
    bucket_biases: np.ndarray,
    grid: ThresholdGrid,
    timeout_seconds: float | None,
    *,
    benchmark_general: bool,
) -> tuple[list[tuple[int, int]] | None, float, str]:
    def weight(left: int, right: int) -> float:
        source = equilibrium(left, right, bucket_biases, grid)
        return 0.0 if source is None else float(grid.v[source[2] - 1])

    if grid.supports_adjacent_matching:
        # When g is nondecreasing and V is nonincreasing, edge weights are
        # nonincreasing with scalar bias distance.  The uncrossing argument then
        # makes consecutive pairs in bias order an exact maximum matching.
        started = time.perf_counter()
        exact = known_bias_matching(bucket_biases, weight)
        adjacent_seconds = time.perf_counter() - started
        if not benchmark_general:
            return exact, adjacent_seconds, "complete"
        _, seconds, status = timed_exact_matching(
            len(bucket_biases), weight, timeout_seconds
        )
        return exact, seconds, status

    # A supplied prior need not retain the uniform prior's monotone weights.
    # Fall back to the general exact solver rather than applying the shortcut
    # without its sufficient conditions.
    return timed_exact_matching(len(bucket_biases), weight, timeout_seconds)


def _lift_matching(
    level: BucketLevel,
    biases: np.ndarray,
    bucket_biases: np.ndarray,
    matching: list[tuple[int, int]],
    grid: ThresholdGrid,
) -> list[tuple[Pair, Pair | None]]:
    """Lift every matched bucket pair to one-to-one underlying group pairs."""
    lifted: list[tuple[Pair, Pair | None]] = []
    for left, right in matching:
        coarse = equilibrium(left, right, bucket_biases, grid)
        if coarse is None:
            continue
        first_bucket, _, threshold = coarse
        left_start, left_stop = level.intervals[left]
        right_start, right_stop = level.intervals[right]
        for left_group, right_group in zip(
            range(left_start, left_stop), range(right_start, right_stop)
        ):
            user = (
                (left_group, right_group, threshold)
                if first_bucket == left
                else (right_group, left_group, threshold)
            )
            source = equilibrium(left_group, right_group, biases, grid)
            lifted.append((user, source))
    return lifted


def run_fixed_domain_mi(
    domain: FixedDomain,
    *,
    lambda_bias: float = 0.25,
    runs: int = 5,
    seed: int = 0,
    exact_timeout_seconds: float | None = 600.0,
    bias_model: Literal["sorted", "iid", "joint"] = "sorted",
    joint_bias_distribution: JointBiasDistribution | None = None,
    prior_model: Literal[
        "uniform", "gaussian", "gaussian-random"
    ] = "uniform",
    prior_variance: float | None = None,
    prior_seed: int = 0,
    benchmark_general: bool = True,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bucket full groups, run MI on buckets, and evaluate on all full groups."""
    m = domain.m
    if prior_model == "uniform":
        if prior_variance is not None:
            raise ValueError("prior_variance is only valid for the Gaussian prior")
    elif prior_model == "gaussian":
        if prior_variance is None:
            raise ValueError("Gaussian MI requires prior_variance")
    elif prior_model == "gaussian-random":
        if prior_variance is not None:
            raise ValueError(
                "prior_variance is sampled for the random Gaussian prior"
            )
    else:  # pragma: no cover - protected by the type and CLI choices
        raise ValueError(f"Unknown prior model: {prior_model}")
    levels = dyadic_levels(m)
    result_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    gain_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

    for run in range(runs):
        prior_std_fraction = (
            shared_gaussian_std_fraction(run, prior_seed)
            if prior_model == "gaussian-random"
            else float("nan")
        )
        if prior_model == "uniform":
            grid = ThresholdGrid(m)
        elif prior_model == "gaussian":
            grid = ThresholdGrid.gaussian(m, float(prior_variance))
        else:
            grid = ThresholdGrid.gaussian(
                m, (prior_std_fraction * m) ** 2
            )
        if bias_model == "joint":
            if joint_bias_distribution is None:
                raise ValueError("joint bias model requires a joint distribution")
            biases = joint_expected_biases(
                joint_bias_distribution,
                m=m,
                lambda_bias=lambda_bias,
                run=run,
                seed=seed,
            )
        else:
            biases = seeded_biases(
                domain.dataset, m, lambda_bias, run, seed, bias_model
            )
        run_rows: list[dict[str, Any]] = []
        for level in levels:
            bucket_biases = _bucket_biases(level, biases)
            matching, matching_seconds, status = _mi_matching(
                bucket_biases,
                grid,
                exact_timeout_seconds,
                benchmark_general=benchmark_general,
            )
            metadata = {
                "dataset": domain.dataset,
                "dataset_label": domain.dataset_label,
                "attribute": domain.attribute,
                "attribute_label": domain.attribute_label,
                "coordinate_transform": domain.coordinate_transform,
                "full_cardinality": domain.full_cardinality,
                "m": m,
                "z": m,
                "run": run,
                "lambda": lambda_bias,
                "prior_model": prior_model,
                "prior_seed": prior_seed,
                "prior_std_fraction": prior_std_fraction,
                "prior_variance": grid.prior_variance,
                "prior_mean": grid.prior_mean,
                "prior_actual_variance": grid.prior_actual_variance,
                "buckets": level.count,
                "bucket_width_groups": max(
                    stop - start for start, stop in level.intervals
                ),
                "bias_model": (
                    "joint-conditional-iid-uniform"
                    if bias_model == "joint"
                    else (
                        "seeded-sorted-uniform"
                        if bias_model == "sorted"
                        else "seeded-iid-uniform"
                    )
                ),
                "bias_attribute": (
                    joint_bias_distribution.key
                    if joint_bias_distribution is not None
                    else ""
                ),
                "bias_attribute_label": (
                    joint_bias_distribution.label
                    if joint_bias_distribution is not None
                    else ""
                ),
                "bias_attribute_count": (
                    3 if joint_bias_distribution is not None else 1
                ),
                "joint_bias_values": (
                    joint_bias_distribution.joint_values
                    if joint_bias_distribution is not None
                    else m
                ),
                "distribution_rows": (
                    joint_bias_distribution.rows
                    if joint_bias_distribution is not None
                    else m
                ),
            }
            timing_rows.append(
                {
                    **metadata,
                    "matching_seconds": matching_seconds,
                    "status": status,
                }
            )
            if matching is None:
                lifted = []
                total_gain = float("nan")
                result_status = status
                matched_bucket_pairs = float("nan")
            else:
                lifted = _lift_matching(
                    level, biases, bucket_biases, matching, grid
                )
                total_gain = float(
                    sum(
                        pair_gain(
                            m,
                            biases,
                            user,
                            source,
                            "robust",
                            gain_cache,
                            grid=grid,
                        )
                        for user, source in lifted
                    )
                )
                result_status = "complete"
                matched_bucket_pairs = len(matching)
            run_rows.append(
                {
                    **metadata,
                    "total_quadratic_gain": total_gain,
                    "lifted_group_pairs": len(lifted),
                    "matched_bucket_pairs": matched_bucket_pairs,
                    "status": result_status,
                }
            )
            if progress:
                print(
                    f"{domain.dataset}: run={run}, B={level.count}/{m}, "
                    f"pairs={matched_bucket_pairs}, "
                    + (
                        f"general_time={matching_seconds:.3f}s"
                        if run == 0
                        else "general_time=not measured"
                    ),
                    flush=True,
                )

        full_rows = [row for row in run_rows if row["buckets"] == m]
        full_gain = (
            float(full_rows[0]["total_quadratic_gain"])
            if full_rows and full_rows[0]["status"] == "complete"
            else float("nan")
        )
        for row in run_rows:
            gain = float(row["total_quadratic_gain"])
            row["full_quadratic_gain"] = full_gain
            row["utility_recovery"] = (
                gain / full_gain
                if np.isfinite(gain) and np.isfinite(full_gain) and abs(full_gain) > 1e-12
                else float("nan")
            )
            result_rows.append(row)

    return pd.DataFrame(result_rows), pd.DataFrame(timing_rows)


def summarize(
    results: pd.DataFrame,
    timings: pd.DataFrame,
) -> pd.DataFrame:
    results = results.copy()
    timings = timings.copy()
    for frame in (results, timings):
        if "prior_model" not in frame:
            frame["prior_model"] = "uniform"
        if "prior_seed" not in frame:
            frame["prior_seed"] = 0
        if "prior_std_fraction" not in frame:
            frame["prior_std_fraction"] = np.nan
        if "prior_variance" not in frame:
            frame["prior_variance"] = np.nan
        if "prior_mean" not in frame:
            frame["prior_mean"] = (frame["z"] + 1.0) / 2.0
        if "prior_actual_variance" not in frame:
            frame["prior_actual_variance"] = (frame["z"] ** 2 - 1.0) / 12.0
        if "bias_attribute" not in frame:
            frame["bias_attribute"] = ""
        if "bias_attribute_label" not in frame:
            frame["bias_attribute_label"] = ""
        if "bias_attribute_count" not in frame:
            frame["bias_attribute_count"] = 1
        if "joint_bias_values" not in frame:
            frame["joint_bias_values"] = frame["m"]
        if "distribution_rows" not in frame:
            frame["distribution_rows"] = frame["m"]
    keys = [
        "dataset",
        "dataset_label",
        "attribute",
        "attribute_label",
        "coordinate_transform",
        "full_cardinality",
        "m",
        "z",
        "lambda",
        "prior_model",
        "prior_seed",
        "buckets",
        "bucket_width_groups",
        "bias_model",
        "bias_attribute",
        "bias_attribute_label",
        "bias_attribute_count",
        "joint_bias_values",
        "distribution_rows",
    ]
    metrics = [
        "total_quadratic_gain",
        "full_quadratic_gain",
        "utility_recovery",
        "lifted_group_pairs",
        "matched_bucket_pairs",
    ]
    aggregate = results.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    summary = aggregate.reset_index()
    summary = summary.rename(
        columns={
            "utility_recovery_mean": "utility_recovery_run_ratio_mean",
            "utility_recovery_std": "utility_recovery_run_ratio_std",
        }
    )
    denominator = summary["full_quadratic_gain_mean"]
    summary["utility_recovery_mean"] = (
        summary["total_quadratic_gain_mean"] / denominator
    )
    summary["utility_recovery_std"] = (
        summary["total_quadratic_gain_std"] / denominator.abs()
    )
    prior_summary = (
        results.groupby(keys, dropna=False)
        .agg(
            prior_std_fraction_mean=("prior_std_fraction", "mean"),
            prior_std_fraction_std=("prior_std_fraction", "std"),
            prior_variance=("prior_variance", "mean"),
            prior_variance_std=("prior_variance", "std"),
            prior_mean=("prior_mean", "mean"),
            prior_actual_variance=("prior_actual_variance", "mean"),
            prior_actual_variance_std=("prior_actual_variance", "std"),
        )
        .reset_index()
    )
    summary = summary.merge(
        prior_summary, on=keys, validate="one_to_one"
    )
    runtime = (
        timings[timings["status"] == "complete"]
        .groupby(keys, dropna=False)["matching_seconds"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "matching_seconds_mean",
                "median": "matching_seconds_median",
                "std": "matching_seconds_std",
            }
        )
    )
    return summary.merge(runtime, on=keys, how="left", validate="one_to_one")
