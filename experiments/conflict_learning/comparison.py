"""Scaled conflict-learning comparison with exact and fast strategies.

Conflict learning is shared. Its learned thresholds are then supplied to
either maximum-weight matching or the efficient sort-and-pair influential
strategy. Every selected pair is subsequently re-probed, its point-level
threshold is substituted into the final query, and unavailable pairs are
dropped. Consequently both strategies have identical representative-learning
probes and global bias error, while their matched-pair certification costs can
differ.
"""

from __future__ import annotations

from contextlib import contextmanager
import signal
import threading
import time
from typing import Any, Iterator
import zlib

import numpy as np
import pandas as pd

from .experiment import (
    BucketLevel,
    Pair,
    TOL,
    ThresholdGrid,
    answer_matrix,
    bias_error_metrics,
    certify_matching,
    dyadic_levels,
    equilibrium,
    learned_pair,
    learned_representatives,
    maximum_matching,
    mean_tau,
    pair_gain,
)
from .scheme import FixedDomain
from experiments.joint_bias import JointBiasDistribution, joint_expected_biases


class MatchingTimeoutError(TimeoutError):
    """Raised when one exact matching call exceeds its configured limit."""


def seeded_ordered_biases(
    dataset: str,
    m: int,
    lambda_bias: float,
    run: int,
    seed: int,
) -> np.ndarray:
    """Draw a fresh default-range bias vector and align it with attribute order.

    The base uniform draw is shared across bias strengths for a fixed dataset
    and run; ``lambda_bias`` only rescales it.  Sorting introduces the local
    structure needed by learning buckets without imposing an exactly linear
    bias function.
    """
    bias_seed = zlib.crc32(
        f"{dataset}|{seed}|{run}|conflict-learning-bias".encode()
    )
    unit_draws = np.random.default_rng(bias_seed).uniform(0.0, 1.0, size=m)
    return lambda_bias * m * (np.sort(unit_draws) - 1.0)


def shared_gaussian_std_fraction(run: int, seed: int = 0) -> float:
    """Draw the Gaussian width shared by all datasets and strengths in a run."""
    draw_seed = zlib.crc32(f"query-gaussian-prior|{seed}|{run}".encode())
    draw = float(np.random.default_rng(draw_seed).uniform())
    return max(draw, np.finfo(float).eps)


@contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    supported = (
        seconds is not None
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not supported:
        yield
        return

    def expired(_signum, _frame):
        raise MatchingTimeoutError(f"exact matching exceeded {seconds:g} seconds")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def timed_exact_matching(
    m: int,
    weight_function,
    timeout_seconds: float | None,
) -> tuple[list[tuple[int, int]] | None, float, str]:
    started = time.perf_counter()
    try:
        with _time_limit(timeout_seconds):
            matching = maximum_matching(m, weight_function)
        return matching, time.perf_counter() - started, "complete"
    except MatchingTimeoutError:
        return None, time.perf_counter() - started, "timeout"


def _interval_midpoint(grid: ThresholdGrid, pair: Pair) -> float:
    low, high = grid.interval(pair[2])
    return (low + high) / 2.0


def fast_influential_matching(
    level: BucketLevel,
    learned: dict[tuple[int, int], Pair | None],
    grid: ThresholdGrid,
) -> list[tuple[int, int]]:
    """Sort by learned relative bias to bucket zero, then pair consecutively."""
    bucket_scores = np.zeros(level.count, dtype=float)
    anchor_rep = int(level.representatives[0])
    for bucket in range(1, level.count):
        result = learned[(0, bucket)]
        if result is None:
            # No influential equilibrium exists for this anchor comparison.
            # Keep such buckets after the representable comparisons in their
            # stable domain order; do not clip the relative bias to the grid.
            bucket_scores[bucket] = float(bucket)
            continue
        midpoint = _interval_midpoint(grid, result)
        target_rep = int(level.representatives[bucket])
        bucket_scores[bucket] = midpoint if result[0] == target_rep else -midpoint
        if result[0] not in {anchor_rep, target_rep}:
            raise AssertionError("learned representative is outside the pair")

    scores = bucket_scores[level.group_to_bucket]
    order = np.lexsort((np.arange(len(scores), dtype=np.int64), scores))
    return [
        (int(order[index]), int(order[index + 1]))
        for index in range(0, len(order) - 1, 2)
    ]


def known_bias_matching(
    biases: np.ndarray, weight_function=None
) -> list[tuple[int, int]]:
    """Exact matching for the known one-dimensional bias order.

    Algorithm-2 edge weights are nonincreasing with bias distance.  The usual
    uncrossing argument therefore reduces the oracle to adjacent edges in
    sorted bias order. For an odd domain, a path dynamic program chooses which
    group to leave unmatched. This avoids running a general blossom
    implementation for the oracle.
    """
    order = np.argsort(biases, kind="stable")
    if weight_function is None:
        return [
            (int(order[index]), int(order[index + 1]))
            for index in range(0, len(order) - 1, 2)
        ]

    count = len(order)
    best = np.zeros(count + 1, dtype=float)
    take = np.zeros(count + 1, dtype=bool)
    for prefix in range(2, count + 1):
        pair_weight = best[prefix - 2] + float(
            weight_function(int(order[prefix - 2]), int(order[prefix - 1]))
        )
        if pair_weight > best[prefix - 1] + TOL:
            best[prefix] = pair_weight
            take[prefix] = True
        else:
            best[prefix] = best[prefix - 1]

    matching: list[tuple[int, int]] = []
    prefix = count
    while prefix >= 2:
        if take[prefix]:
            matching.append(
                (int(order[prefix - 2]), int(order[prefix - 1]))
            )
            prefix -= 2
        else:
            prefix -= 1
    matching.reverse()
    return matching


def _materialize_pairs(
    matching: list[tuple[int, int]] | None,
    level: BucketLevel,
    learned: dict[tuple[int, int], Pair | None],
    inside_flip: np.ndarray,
    biases: np.ndarray,
    grid: ThresholdGrid,
) -> list[tuple[Pair, Pair | None]] | None:
    """Materialize the legacy, uncertified learned-threshold strategy."""
    if matching is None:
        return None
    pairs: list[tuple[Pair, Pair | None]] = []
    for u, v in matching:
        user = learned_pair(u, v, level, learned, inside_flip)
        if user is not None:
            pairs.append((user, equilibrium(u, v, biases, grid)))
    return pairs


def _gain(
    pairs: list[tuple[Pair, Pair | None]] | None,
    z: int,
    biases: np.ndarray,
    mode: str,
    cache: dict[tuple[Any, ...], tuple[float, float]],
    grid: ThresholdGrid,
) -> float:
    if pairs is None:
        return float("nan")
    return float(
        sum(
            pair_gain(z, biases, user, source, mode, cache, grid=grid)
            for user, source in pairs
        )
    )


def _tau(
    pairs: list[tuple[Pair, Pair | None]] | None,
    z: int,
    biases: np.ndarray,
    intents: np.ndarray,
    mode: str,
    grid: ThresholdGrid,
) -> float:
    if pairs is None:
        return float("nan")
    return mean_tau(
        intents, answer_matrix(z, biases, pairs, intents, mode, grid=grid)
    )


def run_dataset_comparison(
    domain: FixedDomain,
    *,
    lambda_bias: float = 0.25,
    prior_model: str = "uniform",
    prior_variance: float | None = None,
    prior_seed: int = 0,
    runs: int = 5,
    intents_per_run: int = 1000,
    seed: int = 0,
    exact_timeout_seconds: float | None = 600.0,
    max_learning_buckets: int | None = None,
    joint_bias_distribution: JointBiasDistribution | None = None,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run conflict learning with a fresh ordered bias draw in every run."""
    if not 0.0 < lambda_bias <= 1.0:
        raise ValueError("lambda_bias must be in (0, 1]")
    if runs < 1:
        raise ValueError("runs must be positive")
    if intents_per_run < 1:
        raise ValueError("intents_per_run must be positive")
    if max_learning_buckets is not None and max_learning_buckets < 1:
        raise ValueError("max_learning_buckets must be positive")
    m = domain.m
    z = domain.z
    fixed_grid: ThresholdGrid | None = None
    if prior_model == "uniform":
        if prior_variance is not None:
            raise ValueError("prior_variance is only valid for the Gaussian prior")
        fixed_grid = ThresholdGrid(z)
    elif prior_model == "gaussian":
        if prior_variance is None:
            raise ValueError("prior_variance is required for the Gaussian prior")
        fixed_grid = ThresholdGrid.gaussian(z, prior_variance)
    elif prior_model == "gaussian-random":
        if prior_variance is not None:
            raise ValueError(
                "prior_variance is sampled for the random Gaussian prior"
            )
    else:
        raise ValueError(
            "prior_model must be 'uniform', 'gaussian', or 'gaussian-random'"
        )
    all_levels = dyadic_levels(m)
    levels = [
        level
        for level in all_levels
        if max_learning_buckets is None or level.count <= max_learning_buckets
    ]
    if not levels:
        raise ValueError("max_learning_buckets excludes every dyadic level")
    full_level = all_levels[-1] if levels[-1].count == m else None
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    structure_flip = np.zeros((m, m), dtype=np.int8)
    ordered_levels = [full_level, *levels[:-1]] if full_level is not None else levels

    for run in range(runs):
        prior_std_fraction = (
            shared_gaussian_std_fraction(run, prior_seed)
            if prior_model == "gaussian-random"
            else float("nan")
        )
        grid = (
            ThresholdGrid.gaussian(z, (prior_std_fraction * z) ** 2)
            if prior_model == "gaussian-random"
            else fixed_grid
        )
        assert grid is not None
        if not grid.supports_adjacent_matching:
            raise ValueError(
                f"{prior_model} prior does not support adjacent known-bias matching"
            )
        biases = (
            joint_expected_biases(
                joint_bias_distribution,
                m=m,
                lambda_bias=lambda_bias,
                run=run,
                seed=seed,
            )
            if joint_bias_distribution is not None
            else seeded_ordered_biases(
                domain.dataset, m, lambda_bias, run, seed
            )
        )
        gain_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

        def known_weight(u: int, v: int) -> float:
            result = equilibrium(u, v, biases, grid)
            return 0.0 if result is None else float(grid.v[result[2] - 1])

        known_started = time.perf_counter()
        oracle_matching = known_bias_matching(biases, known_weight)
        known_matching_seconds = time.perf_counter() - known_started
        known_pairs: list[tuple[Pair, Pair | None]] = []
        for u, v in oracle_matching:
            source = equilibrium(u, v, biases, grid)
            if source is not None:
                known_pairs.append((source, source))
        known_gain = _gain(known_pairs, z, biases, "fixed", gain_cache, grid)
        utility_constant = float(
            m * grid.prior_actual_variance + np.sum(biases**2)
        )

        rng = np.random.default_rng(seed + 1000 + run)
        if grid.prior_name == "uniform":
            intents = rng.integers(1, z + 1, size=(intents_per_run, m))
        else:
            intents = rng.choice(
                np.arange(1, z + 1),
                size=(intents_per_run, m),
                p=np.asarray(grid.prior, dtype=float),
            )
        # Keep within-bucket pair orientations identical to the original
        # uniform-prior experiment.  Previously they followed the uniform
        # intent draw in this seeded stream; replay that draw separately so a
        # change of intent prior cannot also change the learned query.
        orientation_rng = np.random.default_rng(seed + 1000 + run)
        orientation_rng.integers(1, z + 1, size=(intents_per_run, m))
        inside_flip = np.triu(
            orientation_rng.integers(0, 2, size=(m, m), dtype=np.int8), 1
        )
        known_tau = mean_tau(
            intents,
            answer_matrix(z, biases, known_pairs, intents, "fixed", grid=grid),
        )
        no_query_answers = np.tile(
            grid.prior_mean - biases, (len(intents), 1)
        )
        no_query_tau = mean_tau(intents, no_query_answers)

        for level in ordered_levels:
            learning_started = time.perf_counter()
            learned, queries = learned_representatives(level, biases, grid)
            learning_seconds = time.perf_counter() - learning_started
            variances = grid.v

            def learned_weight(u: int, v: int) -> float:
                user = learned_pair(u, v, level, learned, structure_flip)
                return 0.0 if user is None else float(variances[user[2] - 1])

            exact_matching, exact_seconds, exact_status = timed_exact_matching(
                m, learned_weight, exact_timeout_seconds
            )
            fast_started = time.perf_counter()
            fast_matching = fast_influential_matching(level, learned, grid)
            fast_seconds = time.perf_counter() - fast_started

            representative_bias = biases[level.representatives][
                level.group_to_bucket
            ]
            epsilon = 2.0 * float(
                np.max(np.abs(biases - representative_bias))
            )
            error_mean, error_max, coverage, bias_pairs = bias_error_metrics(
                level, biases, grid, learned, structure_flip, epsilon
            )
            unavailable_bucket_pairs = sum(
                value is None for value in learned.values()
            )
            metadata = {
                "dataset": domain.dataset,
                "dataset_label": domain.dataset_label,
                "attribute": domain.attribute,
                "attribute_label": domain.attribute_label,
                "coordinate_transform": domain.coordinate_transform,
                "full_cardinality": domain.full_cardinality,
                "m": m,
                "z": z,
                "run": run,
                "lambda": lambda_bias,
                "buckets": level.count,
                "bucket_width_groups": max(
                    stop - start for start, stop in level.intervals
                ),
                "bias_model": (
                    "joint-conditional-iid-uniform"
                    if joint_bias_distribution is not None
                    else "seeded-sorted-uniform"
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
                "prior_model": prior_model,
                "prior_seed": prior_seed,
                "prior_std_fraction": prior_std_fraction,
                "prior_variance": grid.prior_variance,
                "prior_actual_variance": grid.prior_actual_variance,
            }
            timings.append(
                {
                    **metadata,
                    "learning_probe_queries": queries,
                    "probe_queries": queries,
                    "unavailable_bucket_pairs": unavailable_bucket_pairs,
                    "known_matching_seconds": known_matching_seconds,
                    "known_matching_status": "complete",
                    "learning_seconds": learning_seconds,
                    "exact_matching_seconds": exact_seconds,
                    "fast_matching_seconds": fast_seconds,
                    "exact_total_seconds": learning_seconds + exact_seconds,
                    "fast_total_seconds": learning_seconds + fast_seconds,
                    "exact_matching_status": exact_status,
                }
            )

            exact_pairs = _materialize_pairs(
                exact_matching, level, learned, inside_flip, biases, grid
            )
            fast_pairs = _materialize_pairs(
                fast_matching, level, learned, inside_flip, biases, grid
            )
            certified_exact_pairs, exact_certification_queries, exact_dropped = (
                certify_matching(
                    exact_matching, level, learned, inside_flip, biases, grid
                )
            )
            certified_fast_pairs, fast_certification_queries, fast_dropped = (
                certify_matching(
                    fast_matching, level, learned, inside_flip, biases, grid
                )
            )
            timings[-1].update(
                {
                    "certification_probe_queries": exact_certification_queries,
                    "fast_certification_probe_queries": fast_certification_queries,
                    "probe_queries": queries + exact_certification_queries,
                    "fast_probe_queries": queries + fast_certification_queries,
                }
            )
            dhat_exact_gain = _gain(
                exact_pairs, z, biases, "fixed", gain_cache, grid
            )
            dhat_fast_gain = _gain(
                fast_pairs, z, biases, "fixed", gain_cache, grid
            )
            robust_exact_gain = _gain(
                exact_pairs, z, biases, "robust", gain_cache, grid
            )
            certified_exact_gain = _gain(
                certified_exact_pairs, z, biases, "fixed", gain_cache, grid
            )
            certified_fast_gain = _gain(
                certified_fast_pairs, z, biases, "fixed", gain_cache, grid
            )
            rows.append(
                {
                    **metadata,
                    "utility_known_gain": known_gain,
                    "utility_dhat_exact_gain": dhat_exact_gain,
                    "utility_dhat_fast_gain": dhat_fast_gain,
                    "utility_robust_exact_gain": robust_exact_gain,
                    "utility_certified_exact_gain": certified_exact_gain,
                    "utility_certified_fast_gain": certified_fast_gain,
                    "utility_known_raw": known_gain - utility_constant,
                    "utility_dhat_exact_raw": dhat_exact_gain - utility_constant,
                    "utility_dhat_fast_raw": dhat_fast_gain - utility_constant,
                    "utility_robust_exact_raw": robust_exact_gain - utility_constant,
                    "utility_certified_exact_raw": (
                        certified_exact_gain - utility_constant
                    ),
                    "utility_certified_fast_raw": (
                        certified_fast_gain - utility_constant
                    ),
                    "utility_no_query_raw": -utility_constant,
                    "tau_known": known_tau,
                    "tau_dhat_exact": _tau(
                        exact_pairs, z, biases, intents, "fixed", grid
                    ),
                    "tau_dhat_fast": _tau(
                        fast_pairs, z, biases, intents, "fixed", grid
                    ),
                    "tau_robust_exact": _tau(
                        exact_pairs, z, biases, intents, "robust", grid
                    ),
                    "tau_certified_exact": _tau(
                        certified_exact_pairs, z, biases, intents, "fixed", grid
                    ),
                    "tau_certified_fast": _tau(
                        certified_fast_pairs, z, biases, intents, "fixed", grid
                    ),
                    "tau_no_query": no_query_tau,
                    "learning_probe_queries": queries,
                    "certification_probe_queries": exact_certification_queries,
                    "fast_certification_probe_queries": fast_certification_queries,
                    "probe_queries": queries + exact_certification_queries,
                    "fast_probe_queries": queries + fast_certification_queries,
                    "unavailable_bucket_pairs": unavailable_bucket_pairs,
                    "bias_error_mean": error_mean,
                    "bias_error_max": error_max,
                    "bias_error_mean_normalized": error_mean / z,
                    "bias_error_max_normalized": error_max / z,
                    "epsilon": epsilon,
                    "epsilon_normalized": epsilon / z,
                    "bias_bound": 0.5 + epsilon,
                    "bias_bound_normalized": (0.5 + epsilon) / z,
                    "bias_coverage": coverage,
                    "bias_pairs": bias_pairs,
                    "exact_matched_pairs": (
                        float("nan") if exact_pairs is None else len(exact_pairs)
                    ),
                    "fast_matched_pairs": len(fast_pairs),
                    "certified_exact_matched_pairs": (
                        float("nan")
                        if certified_exact_pairs is None
                        else len(certified_exact_pairs)
                    ),
                    "certified_fast_matched_pairs": len(certified_fast_pairs),
                    "exact_dropped_pairs": exact_dropped,
                    "fast_dropped_pairs": fast_dropped,
                    "exact_matching_status": exact_status,
                }
            )
            if progress:
                print(
                    f"{domain.dataset} lambda={lambda_bias:g} run={run}: "
                    f"k={level.count}/{m}, "
                    f"probes={queries + exact_certification_queries} "
                    f"(learn={queries}, certify={exact_certification_queries}), "
                    f"learn={learning_seconds:.3f}s, "
                    f"exact={exact_status}:{exact_seconds:.3f}s, "
                    f"fast={fast_seconds:.3f}s",
                    flush=True,
                )

            if full_level is not None and level is full_level:
                # Release the O(m^2) learned-pair dictionary before lower levels.
                learned = {}

    return (
        pd.DataFrame(rows).sort_values(["lambda", "buckets", "run"]).reset_index(drop=True),
        pd.DataFrame(timings)
        .sort_values(["lambda", "buckets", "run"])
        .reset_index(drop=True),
    )


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if "prior_seed" not in results:
        results["prior_seed"] = 0
    if "prior_std_fraction" not in results:
        results["prior_std_fraction"] = np.nan
    if "bias_attribute" not in results:
        results["bias_attribute"] = ""
    if "bias_attribute_label" not in results:
        results["bias_attribute_label"] = ""
    if "bias_attribute_count" not in results:
        results["bias_attribute_count"] = 1
    if "joint_bias_values" not in results:
        results["joint_bias_values"] = results["m"]
    if "distribution_rows" not in results:
        results["distribution_rows"] = results["m"]
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
        "buckets",
        "bucket_width_groups",
        "bias_model",
        "bias_attribute",
        "bias_attribute_label",
        "bias_attribute_count",
        "joint_bias_values",
        "distribution_rows",
        "prior_model",
        "prior_seed",
        "exact_matching_status",
    ]
    prior_columns = {
        "prior_std_fraction",
        "prior_variance",
        "prior_actual_variance",
    }
    metrics = [
        column
        for column in results.columns
        if column not in {*keys, "run", *prior_columns}
    ]
    numeric = results[metrics].select_dtypes(include=[np.number]).columns.tolist()
    aggregate = results.groupby(keys, dropna=False)[numeric].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    summary = aggregate.reset_index()
    prior_summary = (
        results.groupby(keys, dropna=False)
        .agg(
            prior_std_fraction_mean=("prior_std_fraction", "mean"),
            prior_std_fraction_std=("prior_std_fraction", "std"),
            prior_variance=("prior_variance", "mean"),
            prior_variance_std=("prior_variance", "std"),
            prior_actual_variance=("prior_actual_variance", "mean"),
            prior_actual_variance_std=("prior_actual_variance", "std"),
        )
        .reset_index()
    )
    return summary.merge(prior_summary, on=keys, validate="one_to_one")
