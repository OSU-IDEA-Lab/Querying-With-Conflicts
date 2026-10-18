"""Run the bucketization sweep and record ranking and query utility.

This is the multi-dataset form of the attached Flights experiment. The source's
bias lives on fine groups. The user operates on coarser, consecutive buckets,
and each selected credible pair changes the source response. Ranking utility is
Kendall tau-b between that response and the intended fine-group order. Query
utility is the expected-information gain of the selected credible pairs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import signal
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from experiments.conflict_learning.experiment import ThresholdGrid

try:
    from scipy.stats import kendalltau as _scipy_kendalltau
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _scipy_kendalltau = None


DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 1.0)
DEFAULT_RUNS = 5
DEFAULT_TIMEOUT_SECONDS = 600.0

# Avoid constructing complete graphs/arrays large enough to exhaust memory
# before the wall-clock timeout can fire.
PAIR_LIMITS = {
    "Alg. 1": 1_000_000,
    "Single best pair": 5_000_000,
}


_CLOSED_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def closed(z: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form posterior gap and utility variance for every threshold."""
    if z not in _CLOSED_CACHE:
        distance = np.arange(1, z)
        area = (z - distance) * (z - distance + 1.0)
        q = 2.0 * z * z - area
        gap = 1 - distance + (z + 2 * distance - 1) * (z * z - area) / (3 * q)
        variance = area * (z + 2 * distance - 1) ** 2 / (18 * q)
        _CLOSED_CACHE[z] = (gap, variance)
    return _CLOSED_CACHE[z]


def shared_gaussian_std_fraction(run: int, seed: int = 0) -> float:
    """Return the seeded Gaussian width used across datasets in one run."""
    draw_seed = zlib.crc32(f"query-gaussian-prior|{seed}|{run}".encode())
    draw = float(np.random.default_rng(draw_seed).uniform())
    return max(draw, np.finfo(float).eps)


def dstar(bias_difference: np.ndarray | float, gap: np.ndarray) -> np.ndarray:
    """Smallest credible threshold for the oriented pair; zero if none exists."""
    bias_difference = np.asarray(bias_difference, dtype=float)
    if len(gap) == 0:
        return np.zeros_like(bias_difference, dtype=np.int64)
    index = np.searchsorted(-gap, -(bias_difference + 1) - 1e-12, side="left")
    clipped = np.minimum(index, len(gap) - 1)
    valid = (index < len(gap)) & (gap[clipped] >= bias_difference - 1e-12)
    return np.where(valid, clipped + 1, 0)


def orient(
    first: np.ndarray,
    second: np.ndarray,
    biases: np.ndarray,
    gap: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the higher-utility orientation for each candidate pair."""
    d_first = dstar(biases[first] - biases[second], gap)
    d_second = dstar(biases[second] - biases[first], gap)
    w_first = np.where(d_first > 0, variance[np.maximum(d_first, 1) - 1], 0.0)
    w_second = np.where(d_second > 0, variance[np.maximum(d_second, 1) - 1], 0.0)
    take_first = w_first >= w_second
    weights = np.maximum(w_first, w_second)
    upper = np.where(take_first, first, second)
    lower = np.where(take_first, second, first)
    delta = np.where(take_first, d_first, d_second)
    return weights, upper, lower, delta


Pair = tuple[int, int, int]
Strategy = Callable[..., list[Pair]]


def _threshold_statistics(
    z: int,
    gap: np.ndarray | None,
    utility_variance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if gap is None and utility_variance is None:
        return closed(z)
    if gap is None or utility_variance is None:
        raise ValueError("gap and utility_variance must be supplied together")
    gap = np.asarray(gap, dtype=float)
    utility_variance = np.asarray(utility_variance, dtype=float)
    if gap.shape != (z - 1,) or utility_variance.shape != (z - 1,):
        raise ValueError("threshold statistics must have length z-1")
    return gap, utility_variance


def _prior_statistics(
    z: int,
    prior_model: Literal["uniform", "gaussian-random"],
    prior_std_fraction: float,
) -> tuple[
    ThresholdGrid | None,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Build threshold and metadata values for one bucket-rank domain."""
    if prior_model == "uniform":
        gap, utility_variance = closed(z)
        return (
            None,
            gap,
            utility_variance,
            float("nan"),
            (z**2 - 1.0) / 12.0,
            (z + 1.0) / 2.0,
        )
    prior_variance = (prior_std_fraction * z) ** 2
    if z == 1:
        empty = np.empty(0, dtype=float)
        return None, empty, empty, prior_variance, 0.0, 1.0
    grid = ThresholdGrid.gaussian(z, prior_variance)
    return (
        grid,
        1.0 - grid.g,
        grid.v,
        prior_variance,
        grid.prior_actual_variance,
        grid.prior_mean,
    )


@dataclass(frozen=True)
class BucketLevel:
    """One nested, contiguous level of a balanced dyadic partition."""

    intervals: tuple[tuple[int, int], ...]
    fine_to_bucket: np.ndarray
    centers: np.ndarray

    @property
    def count(self) -> int:
        return len(self.intervals)


def dyadic_bucket_levels(fine_groups: int) -> list[BucketLevel]:
    """Return nested levels from one bucket to the full distinct domain."""
    if fine_groups < 1:
        raise ValueError("fine_groups must be positive")
    intervals: tuple[tuple[int, int], ...] = ((0, fine_groups),)
    levels: list[BucketLevel] = []
    while True:
        mapping = np.empty(fine_groups, dtype=np.int64)
        centers = np.empty(len(intervals), dtype=float)
        for bucket, (start, stop) in enumerate(intervals):
            mapping[start:stop] = bucket
            centers[bucket] = (start + 1 + stop) / 2.0
        levels.append(BucketLevel(intervals, mapping, centers))
        if len(intervals) == fine_groups:
            return levels
        refined: list[tuple[int, int]] = []
        for start, stop in intervals:
            if stop - start <= 1:
                refined.append((start, stop))
            else:
                midpoint = (start + stop) // 2
                refined.extend(((start, midpoint), (midpoint, stop)))
        intervals = tuple(refined)


class SweepTimeout(TimeoutError):
    """Raised when one method exhausts its per-run wall-clock budget."""


@contextmanager
def _wall_clock_limit(seconds: float):
    if seconds <= 0:
        raise SweepTimeout("wall-clock budget exhausted")
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle_timeout(_signum, _frame):
        raise SweepTimeout("wall-clock budget exhausted")

    previous = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def alg1(
    biases: np.ndarray,
    rng: np.random.Generator | None,
    *,
    gap: np.ndarray | None = None,
    utility_variance: np.ndarray | None = None,
) -> list[Pair]:
    """Algorithm 1 using an exact maximum-weight matching."""
    try:
        import networkx as nx
    except ImportError as error:
        raise RuntimeError(
            "Algorithm 1 requires networkx; install the repository requirements"
        ) from error

    if rng is not None:
        permutation = rng.permutation(len(biases))
        inverse = np.argsort(permutation)
        return [
            (int(inverse[upper]), int(inverse[lower]), delta)
            for upper, lower, delta in alg1(
                biases[inverse],
                None,
                gap=gap,
                utility_variance=utility_variance,
            )
        ]

    z = len(biases)
    gap, utility_variance = _threshold_statistics(
        z, gap, utility_variance
    )
    first, second = np.triu_indices(z, 1)
    weights, upper, lower, delta = orient(
        first, second, biases, gap, utility_variance
    )
    keep = weights > 0
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        zip(first[keep].tolist(), second[keep].tolist(), weights[keep].tolist())
    )
    position = {
        (a, b): index
        for index, (a, b) in enumerate(zip(first[keep].tolist(), second[keep].tolist()))
    }
    kept_upper, kept_lower, kept_delta = upper[keep], lower[keep], delta[keep]
    result: list[Pair] = []
    for a, b in nx.max_weight_matching(graph):
        index = position[(min(a, b), max(a, b))]
        result.append(
            (int(kept_upper[index]), int(kept_lower[index]), int(kept_delta[index]))
        )
    return result


def fast_alg1(
    biases: np.ndarray,
    _rng: np.random.Generator,
    *,
    gap: np.ndarray | None = None,
    utility_variance: np.ndarray | None = None,
) -> list[Pair]:
    z = len(biases)
    gap, utility_variance = _threshold_statistics(
        z, gap, utility_variance
    )
    order = np.argsort(biases, kind="stable")
    first, second = order[0 : z - 1 : 2], order[1:z:2]
    weights, upper, lower, delta = orient(
        first, second, biases, gap, utility_variance
    )
    return [
        (int(u), int(l), int(d))
        for u, l, d, weight in zip(upper, lower, delta, weights)
        if weight > 0
    ]


def random_credible(
    biases: np.ndarray,
    rng: np.random.Generator,
    *,
    gap: np.ndarray | None = None,
    utility_variance: np.ndarray | None = None,
) -> list[Pair]:
    z = len(biases)
    gap, utility_variance = _threshold_statistics(
        z, gap, utility_variance
    )
    permutation = rng.permutation(z)
    first, second = permutation[0 : z - 1 : 2], permutation[1:z:2]
    weights, upper, lower, delta = orient(
        first, second, biases, gap, utility_variance
    )
    return [
        (int(u), int(l), int(d))
        for u, l, d, weight in zip(upper, lower, delta, weights)
        if weight > 0
    ]


def single_best(
    biases: np.ndarray,
    rng: np.random.Generator,
    *,
    gap: np.ndarray | None = None,
    utility_variance: np.ndarray | None = None,
) -> list[Pair]:
    z = len(biases)
    gap, utility_variance = _threshold_statistics(
        z, gap, utility_variance
    )
    first, second = np.triu_indices(z, 1)
    weights, upper, lower, delta = orient(
        first, second, biases, gap, utility_variance
    )
    if len(weights) == 0 or weights.max() <= 0:
        return []
    best = np.flatnonzero(weights == weights.max())
    index = int(rng.choice(best))
    return [(int(upper[index]), int(lower[index]), int(delta[index]))]


def single_any(
    biases: np.ndarray,
    rng: np.random.Generator,
    *,
    gap: np.ndarray | None = None,
    utility_variance: np.ndarray | None = None,
) -> list[Pair]:
    """Return the first credible pair found in a seeded randomized scan.

    Candidates are evaluated one at a time so the scan stops as soon as it finds
    a usable pair. Unlike ``single_best``, this does not construct or score all
    ``m choose 2`` candidates merely to return one of them.
    """
    z = len(biases)
    if z < 2:
        return []
    gap, utility_variance = _threshold_statistics(
        z, gap, utility_variance
    )
    order = rng.permutation(z)
    for left in range(z - 1):
        for right in range(left + 1, z):
            first = np.asarray([order[left]])
            second = np.asarray([order[right]])
            weights, upper, lower, delta = orient(
                first, second, biases, gap, utility_variance
            )
            if weights[0] > 0:
                return [(int(upper[0]), int(lower[0]), int(delta[0]))]
    return []


STRATEGIES: dict[str, Strategy] = {
    "Alg. 1": alg1,
    "Fast Alg. 1": fast_alg1,
    "Random credible pairs": random_credible,
    "Single best pair": single_best,
    "Single random pair": single_any,
}

STRATEGY_ALIASES = {
    "alg1": "Alg. 1",
    "fast": "Fast Alg. 1",
    "random": "Random credible pairs",
    "single": "Single best pair",
    "single_any": "Single random pair",
}


def posterior(distance: int, sent: bool, z: int) -> tuple[float, float]:
    probability = (z - distance) * (z - distance + 1.0) / (2 * z * z)
    posterior_distance = (z + 2 * distance - 1) / 3.0
    if not sent:
        posterior_distance = -posterior_distance * probability / (1 - probability)
    return (
        (z + 1 - posterior_distance) / 2,
        (z + 1 + posterior_distance) / 2,
    )


def answer_fine(
    pairs: list[Pair],
    level: BucketLevel,
    fine_biases: np.ndarray,
    direction: str,
    grid: ThresholdGrid | None = None,
) -> np.ndarray:
    """Return the source response at fine-group resolution."""
    fine_groups = len(fine_biases)
    buckets = level.count
    if len(level.fine_to_bucket) != fine_groups:
        raise ValueError("bucket level and fine biases have different domains")
    ranks = (
        np.arange(1, buckets + 1)
        if direction == "ASC"
        else np.arange(buckets, 0, -1)
    )
    if grid is not None and grid.z != buckets:
        raise ValueError("grid and bucket level must use the same domain")
    mean = (buckets + 1) / 2 if grid is None else grid.prior_mean
    response = np.full(buckets, mean)
    for upper, lower, distance in pairs:
        sent = (ranks[lower] - ranks[upper]) >= distance
        if grid is None:
            response[upper], response[lower] = posterior(
                distance, sent, buckets
            )
        else:
            posterior_distance = (
                grid.d1[distance - 1] if sent else grid.d0[distance - 1]
            )
            response[upper] = mean - posterior_distance / 2.0
            response[lower] = mean + posterior_distance / 2.0
    bucket_ranks = np.arange(1, buckets + 1, dtype=float)
    response_fine = np.interp(response, bucket_ranks, level.centers)
    return response_fine[level.fine_to_bucket] + fine_biases


def kendall_tau_b_unique_intent(answer: np.ndarray, intent: np.ndarray) -> float:
    """Kendall tau-b optimized for the experiment's unique intent ranks.

    SciPy's compiled implementation avoids making metric bookkeeping dominate
    the large distinct-domain experiments. The local Fenwick-tree version is a
    dependency fallback with the same tau-b definition.
    """
    answer = np.asarray(answer)
    intent = np.asarray(intent)
    if len(answer) != len(intent):
        raise ValueError("answer and intent must have the same length")
    if len(answer) < 2:
        return float("nan")
    if _scipy_kendalltau is not None:
        return float(
            _scipy_kendalltau(answer, intent, method="auto", variant="b").statistic
        )

    ordered_answer = answer[np.argsort(intent, kind="stable")]
    _, encoded = np.unique(ordered_answer, return_inverse=True)
    tree = np.zeros(int(encoded.max()) + 2, dtype=np.int64)

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    discordant = 0
    seen = 0
    for rank_zero in encoded:
        rank = int(rank_zero) + 1
        discordant += seen - prefix(rank)
        index = rank
        while index < len(tree):
            tree[index] += 1
            index += index & -index
        seen += 1

    counts = np.bincount(encoded)
    ties_answer = int(np.sum(counts * (counts - 1) // 2))
    total_pairs = len(answer) * (len(answer) - 1) // 2
    concordant = total_pairs - discordant - ties_answer
    untied = concordant + discordant
    denominator = math.sqrt(total_pairs * untied)
    if denominator == 0:
        return float("nan")
    return float((concordant - discordant) / denominator)


def load_workload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        raise ValueError("legacy query lists lack dataset metadata; regenerate with workload.py")
    required = {
        "dataset",
        "ranking_attribute",
        "attribute_type",
        "fine_groups",
        "queries",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"workload is missing metadata: {', '.join(missing)}")
    fine_groups = int(payload["fine_groups"])
    for index, query in enumerate(payload["queries"]):
        groups = query.get("groups", [])
        if not groups or min(groups) < 0 or max(groups) >= fine_groups:
            raise ValueError(f"query {index} contains invalid fine-group indexes")
    return payload


def _select_levels(
    fine_groups: int, buckets: list[int] | None
) -> list[BucketLevel]:
    levels = dyadic_bucket_levels(fine_groups)
    if buckets is None:
        return levels
    by_count = {level.count: level for level in levels}
    invalid = sorted(set(buckets) - set(by_count))
    if invalid:
        raise ValueError(
            f"bucket counts must be nested dyadic levels for n={fine_groups}; "
            f"invalid={invalid}, valid={sorted(by_count)}"
        )
    return [by_count[count] for count in sorted(set(buckets))]


def summarize_utility(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "ranking_attribute",
        "attribute_type",
        "prior_model",
        "lambda",
        "strategy",
        "buckets",
    ]
    summary = (
        results.groupby(keys, dropna=False)["ranking_utility"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(summary["count"])
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["sem"]
    return summary


def summarize_runtime(timings: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "ranking_attribute",
        "attribute_type",
        "prior_model",
        "strategy",
        "buckets",
    ]
    completed = (
        timings[timings["status"] == "completed"]
        if "status" in timings
        else timings
    )
    return (
        completed.groupby(keys, dropna=False)
        .agg(
            median_seconds=("seconds", "median"),
            q25_seconds=("seconds", lambda values: values.quantile(0.25)),
            q75_seconds=("seconds", lambda values: values.quantile(0.75)),
            mean_pairs=("pairs", "mean"),
            mean_query_utility=("query_utility", "mean"),
        )
        .reset_index()
    )


def run_experiment(
    payload: dict[str, Any],
    *,
    runs: list[int],
    lambdas: list[float],
    buckets: list[int] | None = None,
    strategies: list[str] | None = None,
    prior_model: Literal["uniform", "gaussian-random"] = "uniform",
    prior_seed: int = 0,
    progress: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fine_groups = int(payload["fine_groups"])
    levels = _select_levels(fine_groups, buckets)
    strategy_names = strategies or list(STRATEGIES)
    unknown = sorted(set(strategy_names) - STRATEGIES.keys())
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}")
    if prior_model not in {"uniform", "gaussian-random"}:
        raise ValueError(f"unknown prior model: {prior_model}")

    queries = payload["queries"]
    present = [np.asarray(query["groups"], dtype=np.int64) for query in queries]
    directions = [query["direction"] for query in queries]
    intent = {
        "ASC": np.arange(1, fine_groups + 1),
        "DESC": np.arange(fine_groups, 0, -1),
    }
    metadata = (
        payload["dataset"],
        payload["ranking_attribute"],
        payload["attribute_type"],
    )
    result_rows: list[tuple[Any, ...]] = []
    timing_rows: list[tuple[Any, ...]] = []

    def utilities(answer_by_direction: dict[str, np.ndarray]) -> list[float]:
        return [
            kendall_tau_b_unique_intent(
                answer_by_direction[direction][groups], intent[direction][groups]
            )
            for groups, direction in zip(present, directions)
        ]

    for run in runs:
        prior_std_fraction = (
            shared_gaussian_std_fraction(run, prior_seed)
            if prior_model == "gaussian-random"
            else float("nan")
        )
        bias_seed = zlib.crc32(
            f"{payload['dataset']}|{1000 + run}|fine-bias".encode()
        )
        uniform = np.random.default_rng(bias_seed).uniform(size=fine_groups)
        (
            _,
            _,
            _,
            full_prior_variance,
            full_actual_variance,
            full_prior_mean,
        ) = _prior_statistics(
            fine_groups, prior_model, prior_std_fraction
        )
        full_prior_metadata = (
            prior_model,
            prior_seed,
            prior_std_fraction,
            full_prior_variance,
            full_actual_variance,
            full_prior_mean,
        )
        for lam in lambdas:
            fine_biases = -lam * fine_groups * uniform
            references = {
                "Babbling": {
                    direction: np.full(fine_groups, full_prior_mean) + fine_biases
                    for direction in intent
                },
                "Full revelation": {
                    direction: intent[direction] + fine_biases for direction in intent
                },
            }
            for strategy_name, answers in references.items():
                for query_index, utility in enumerate(utilities(answers)):
                    result_rows.append(
                        (
                            *metadata,
                            run,
                            lam,
                            *full_prior_metadata,
                            fine_groups,
                            strategy_name,
                            query_index,
                            directions[query_index],
                            utility,
                            utility,
                        )
                    )

        for strategy_name in strategy_names:
            deadline = time.monotonic() + timeout_seconds
            stopped = False
            for level in levels:
                bucket_count = level.count
                (
                    grid,
                    gap,
                    utility_variance,
                    prior_variance,
                    prior_actual_variance,
                    prior_mean,
                ) = _prior_statistics(
                    bucket_count, prior_model, prior_std_fraction
                )
                prior_metadata = (
                    prior_model,
                    prior_seed,
                    prior_std_fraction,
                    prior_variance,
                    prior_actual_variance,
                    prior_mean,
                )
                strategy_options = (
                    {"gap": gap, "utility_variance": utility_variance}
                    if prior_model == "gaussian-random"
                    else {}
                )
                candidate_pairs = bucket_count * (bucket_count - 1) // 2
                pair_limit = PAIR_LIMITS.get(strategy_name)
                if pair_limit is not None and candidate_pairs > pair_limit:
                    for lam in lambdas:
                        timing_rows.append(
                            (
                                *metadata,
                                run,
                                lam,
                                *prior_metadata,
                                bucket_count,
                                strategy_name,
                                0.0,
                                np.nan,
                                np.nan,
                                "resource_limit",
                            )
                        )
                    if progress:
                        print(
                            f"{payload['dataset']}: run {run}, {strategy_name} "
                            f"stopped at {bucket_count} buckets (pair limit)",
                            flush=True,
                        )
                    stopped = True
                    break

                component_means = np.asarray(
                    [uniform[start:stop].mean() for start, stop in level.intervals],
                    dtype=float,
                )
                for lam in lambdas:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timing_rows.append(
                            (
                                *metadata,
                                run,
                                lam,
                                *prior_metadata,
                                bucket_count,
                                strategy_name,
                                0.0,
                                np.nan,
                                np.nan,
                                "timeout",
                            )
                        )
                        stopped = True
                        break
                    fine_biases = -lam * fine_groups * uniform
                    bucket_biases = -lam * bucket_count * component_means
                    rng = np.random.default_rng(
                        zlib.crc32(
                            f"{payload['dataset']}|{run}|{lam}|{bucket_count}|{strategy_name}".encode()
                        )
                    )
                    started = time.perf_counter()
                    try:
                        with _wall_clock_limit(remaining):
                            pairs = STRATEGIES[strategy_name](
                                bucket_biases, rng, **strategy_options
                            )
                    except SweepTimeout:
                        seconds = time.perf_counter() - started
                        timing_rows.append(
                            (
                                *metadata,
                                run,
                                lam,
                                *prior_metadata,
                                bucket_count,
                                strategy_name,
                                seconds,
                                np.nan,
                                np.nan,
                                "timeout",
                            )
                        )
                        stopped = True
                        break
                    except MemoryError:
                        seconds = time.perf_counter() - started
                        timing_rows.append(
                            (
                                *metadata,
                                run,
                                lam,
                                *prior_metadata,
                                bucket_count,
                                strategy_name,
                                seconds,
                                np.nan,
                                np.nan,
                                "resource_limit",
                            )
                        )
                        stopped = True
                        break
                    seconds = time.perf_counter() - started
                    query_utility = float(
                        sum(
                            utility_variance[distance - 1]
                            for _, _, distance in pairs
                        )
                    )
                    timing_rows.append(
                        (
                            *metadata,
                            run,
                            lam,
                            *prior_metadata,
                            bucket_count,
                            strategy_name,
                            seconds,
                            len(pairs),
                            query_utility,
                            "completed",
                        )
                    )
                    answers = {
                        direction: answer_fine(
                            pairs, level, fine_biases, direction, grid=grid
                        )
                        for direction in intent
                    }
                    for query_index, utility in enumerate(utilities(answers)):
                        result_rows.append(
                            (
                                *metadata,
                                run,
                                lam,
                                *prior_metadata,
                                bucket_count,
                                strategy_name,
                                query_index,
                                directions[query_index],
                                utility,
                                utility,
                            )
                        )
                    if time.monotonic() >= deadline:
                        stopped = True
                        break
                if stopped:
                    if progress:
                        print(
                            f"{payload['dataset']}: run {run}, {strategy_name} "
                            f"reached {timeout_seconds:g}s limit",
                            flush=True,
                        )
                    break
            if progress and not stopped:
                print(
                    f"{payload['dataset']}: run {run}, {strategy_name} done",
                    flush=True,
                )

    results = pd.DataFrame(
        result_rows,
        columns=[
            "dataset",
            "ranking_attribute",
            "attribute_type",
            "run",
            "lambda",
            "prior_model",
            "prior_seed",
            "prior_std_fraction",
            "prior_variance",
            "prior_actual_variance",
            "prior_mean",
            "buckets",
            "strategy",
            "query",
            "direction",
            "tau",
            "ranking_utility",
        ],
    )
    timings = pd.DataFrame(
        timing_rows,
        columns=[
            "dataset",
            "ranking_attribute",
            "attribute_type",
            "run",
            "lambda",
            "prior_model",
            "prior_seed",
            "prior_std_fraction",
            "prior_variance",
            "prior_actual_variance",
            "prior_mean",
            "buckets",
            "strategy",
            "seconds",
            "pairs",
            "query_utility",
            "status",
        ],
    )
    return results, timings, summarize_utility(results), summarize_runtime(timings)


def write_results(
    output_prefix: str | Path,
    results: pd.DataFrame,
    timings: pd.DataFrame,
    utility_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(f"{prefix}_utility.csv", index=False)
    timings.to_csv(f"{prefix}_runtime.csv", index=False)
    utility_summary.to_csv(f"{prefix}_utility_summary.csv", index=False)
    runtime_summary.to_csv(f"{prefix}_runtime_summary.csv", index=False)


def _comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _strategy_names(value: str) -> list[str]:
    aliases = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(aliases) - STRATEGY_ALIASES.keys())
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown strategy aliases: {unknown}")
    return [STRATEGY_ALIASES[item] for item in aliases]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--runs", type=_comma_ints, default=list(range(DEFAULT_RUNS)))
    parser.add_argument("--lambdas", type=_comma_floats, default=list(DEFAULT_LAMBDAS))
    parser.add_argument(
        "--buckets",
        type=_comma_ints,
        default=None,
        help="optional dyadic bucket counts; default is every level from 1 to full",
    )
    parser.add_argument("--timeout-minutes", type=float, default=10.0)
    parser.add_argument(
        "--prior-model",
        choices=("uniform", "gaussian-random"),
        default="uniform",
    )
    parser.add_argument("--prior-seed", type=int, default=0)
    parser.add_argument(
        "--strategies",
        type=_strategy_names,
        default=list(STRATEGIES),
        help="comma-separated aliases: alg1,fast,random,single,single_any",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = load_workload(args.workload)
    outputs = run_experiment(
        payload,
        runs=args.runs,
        lambdas=args.lambdas,
        buckets=args.buckets,
        strategies=args.strategies,
        prior_model=args.prior_model,
        prior_seed=args.prior_seed,
        progress=not args.quiet,
        timeout_seconds=args.timeout_minutes * 60.0,
    )
    write_results(args.output_prefix, *outputs)
    utility_summary, runtime_summary = outputs[2], outputs[3]
    print(
        utility_summary.pivot_table(
            index=["lambda", "strategy"], columns="buckets", values="mean"
        ).round(3).to_string()
    )
    print(
        runtime_summary.pivot_table(
            index="strategy", columns="buckets", values="median_seconds"
        ).round(4).to_string()
    )


if __name__ == "__main__":
    main()
