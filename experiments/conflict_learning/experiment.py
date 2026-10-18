"""Learn source conflict thresholds while holding the true domain fixed.

This adapts the supplied conflict-learning reference experiment. The underlying
groups ``m`` and rank scale ``z=m`` remain fixed. Only the number of balanced,
nested learning buckets ``B`` changes from one to ``m``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .scheme import FixedDomain


TOL = 1e-9


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _discrete_gaussian_log_prior(z: int, variance: float) -> np.ndarray:
    if z < 2:
        raise ValueError("z must be at least 2")
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Gaussian prior variance must be positive and finite")
    ranks = np.arange(1, z + 1, dtype=float)
    center = (z + 1.0) / 2.0
    log_weights = -0.5 * ((ranks - center) ** 2) / variance
    if np.any(~np.isfinite(log_weights)):
        raise ValueError(
            "Gaussian prior variance is too small for this domain's numerical "
            "precision"
        )
    return log_weights - _logsumexp(log_weights)


def discrete_gaussian_prior(z: int, variance: float) -> np.ndarray:
    """Return a centered discrete Gaussian on ranks ``1, ..., z``.

    The Gaussian is evaluated at each integer rank and normalized after
    truncation to the finite rank domain.  Probabilities below float64's range
    are represented by its smallest positive value here; ``ThresholdGrid``
    retains their logarithms when computing conditional tail statistics.
    """
    log_prior = _discrete_gaussian_log_prior(z, variance)
    probabilities = np.exp(log_prior)
    probabilities = np.maximum(probabilities, np.nextafter(0.0, 1.0))
    return probabilities / probabilities.sum()


def difference_distribution(prior: np.ndarray) -> np.ndarray:
    """Return ``Pr(Y-X=k)`` for iid ranks with the supplied prior."""
    probabilities = np.asarray(prior, dtype=np.longdouble)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("prior must be a one-dimensional array of length at least 2")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise ValueError("prior probabilities must be positive and finite")
    probabilities = probabilities / probabilities.sum()
    return np.correlate(probabilities, probabilities, mode="full")


def area(z: int, d: np.ndarray | int) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return (z - d) * (z - d + 1.0)


def q_value(z: int, d: np.ndarray | int) -> np.ndarray:
    return 2.0 * z * z - area(z, d)


def g_value(z: int, d: np.ndarray | int) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    a = area(z, d)
    return d - (z + 2.0 * d - 1.0) * (z * z - a) / (3.0 * q_value(z, d))


def variance(z: int, d: np.ndarray | int) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    a = area(z, d)
    return a * (z + 2.0 * d - 1.0) ** 2 / (18.0 * q_value(z, d))


def cell_means(z: int, d: int) -> tuple[float, float]:
    p1 = float(area(z, d) / (2.0 * z * z))
    d1 = float((z + 2 * d - 1) / 3.0)
    return d1, -d1 * p1 / (1.0 - p1)


class ThresholdGrid:
    def __init__(
        self,
        z: int,
        prior: np.ndarray | None = None,
        *,
        prior_name: str = "uniform",
        prior_variance: float | None = None,
        _log_prior: np.ndarray | None = None,
    ):
        if z < 2:
            raise ValueError("z must be at least 2")
        self.z = z
        self.d = np.arange(1, z, dtype=np.int64)
        self.differences = np.arange(-(z - 1), z, dtype=np.int64)
        if prior is None:
            self.prior = np.full(z, 1.0 / z, dtype=np.longdouble)
            self.difference_probabilities = (
                z - np.abs(self.differences)
            ).astype(np.longdouble) / (z * z)
            self.g = g_value(z, self.d)
            self.v = variance(z, self.d)
            p = area(z, self.d) / (2.0 * z * z)
            self.d1 = (z + 2.0 * self.d - 1.0) / 3.0
            self.d0 = -self.d1 * p / (1.0 - p)
        elif _log_prior is None:
            probabilities = np.asarray(prior, dtype=np.longdouble)
            if probabilities.shape != (z,):
                raise ValueError(f"prior must contain exactly {z} probabilities")
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
                raise ValueError("prior probabilities must be positive and finite")
            probabilities = probabilities / probabilities.sum()
            if not np.allclose(
                probabilities,
                probabilities[::-1],
                rtol=1e-12,
                atol=1e-15,
            ):
                raise ValueError(
                    "the pairwise MI formulas require a prior symmetric about "
                    "(z+1)/2"
                )
            self.prior = probabilities
            self.difference_probabilities = difference_distribution(probabilities)
            positive_probabilities = self.difference_probabilities[z:]
            tail_probability = np.cumsum(positive_probabilities[::-1])[::-1]
            tail_moment = np.cumsum(
                (self.d.astype(np.longdouble) * positive_probabilities)[::-1]
            )[::-1]
            if np.any(tail_probability <= 0) or np.any(tail_probability >= 1):
                raise ValueError("prior does not give valid threshold-event probabilities")
            self.d1 = np.asarray(tail_moment / tail_probability, dtype=float)
            self.d0 = np.asarray(
                -tail_moment / (1 - tail_probability), dtype=float
            )
            p = np.asarray(tail_probability, dtype=float)
            self.g = self.d - 0.5 * (self.d0 + self.d1)
            self.v = 0.5 * p * (1.0 - p) * (self.d1 - self.d0) ** 2
        else:
            log_prior = np.asarray(_log_prior, dtype=float)
            if log_prior.shape != (z,) or np.any(~np.isfinite(log_prior)):
                raise ValueError(f"log prior must contain exactly {z} finite values")
            log_prior = log_prior - _logsumexp(log_prior)
            probabilities = np.exp(log_prior)
            probabilities = np.maximum(
                probabilities, np.nextafter(0.0, 1.0)
            )
            self.prior = probabilities / probabilities.sum()

            # Autocorrelation in log space keeps Gaussian tail events available
            # even when their probabilities are smaller than float64 can store.
            log_positive = np.empty(z, dtype=float)
            for difference in range(z):
                log_positive[difference] = _logsumexp(
                    log_prior[: z - difference]
                    + log_prior[difference:]
                )
            positive = np.exp(log_positive)
            self.difference_probabilities = np.concatenate(
                (positive[:0:-1], positive)
            )

            log_tail_probability = np.logaddexp.accumulate(
                log_positive[:0:-1]
            )[::-1]
            d1 = np.empty(z - 1, dtype=float)
            for index in range(z - 1):
                conditional = np.exp(
                    log_positive[index + 1 :] - log_tail_probability[index]
                )
                d1[index] = float(
                    np.dot(self.d[index:].astype(float), conditional)
                )
            tail_probability = np.exp(log_tail_probability)
            log_one_minus_tail = np.log1p(-tail_probability)
            odds = np.exp(log_tail_probability - log_one_minus_tail)
            self.d1 = d1
            self.d0 = -d1 * odds
            self.g = self.d - 0.5 * (self.d0 + self.d1)
            self.v = 0.5 * odds * d1**2
        self.prior_name = prior_name
        self.prior_variance = prior_variance
        ranks = np.arange(1, z + 1, dtype=np.longdouble)
        self.prior_mean = float(np.sum(ranks * self.prior))
        self.prior_actual_variance = float(
            np.sum((ranks - self.prior_mean) ** 2 * self.prior)
        )
        self.g_is_nondecreasing = bool(np.all(np.diff(self.g) >= -TOL))
        self.v_is_nonincreasing = bool(np.all(np.diff(self.v) <= TOL))
        self.influence_intervals_overlap = bool(
            0.0 <= self.g[0] <= 1.0 + TOL
            and np.all(np.diff(self.g) <= 1.0 + TOL)
        )
        self.supports_adjacent_matching = (
            self.g_is_nondecreasing
            and self.v_is_nonincreasing
            and self.influence_intervals_overlap
        )

    @classmethod
    def gaussian(cls, z: int, variance: float) -> "ThresholdGrid":
        log_prior = _discrete_gaussian_log_prior(z, variance)
        return cls(
            z,
            discrete_gaussian_prior(z, variance),
            prior_name="gaussian",
            prior_variance=float(variance),
            _log_prior=log_prior,
        )

    def cell_means(self, threshold: int) -> tuple[float, float]:
        if not 1 <= threshold < self.z:
            raise ValueError("threshold must be in 1, ..., z-1")
        index = threshold - 1
        return float(self.d1[index]), float(self.d0[index])

    def delta_star(self, relative_bias: float) -> int | None:
        if self.g_is_nondecreasing:
            index = int(np.searchsorted(self.g, relative_bias, side="left"))
            if index >= len(self.g) or self.g[index] > relative_bias + 1.0 + TOL:
                return None
            return index + 1
        valid = np.flatnonzero(
            (self.g + TOL >= relative_bias)
            & (self.g - 1.0 - TOL <= relative_bias)
        )
        return None if len(valid) == 0 else int(valid[0] + 1)

    def interval(self, threshold: int) -> tuple[float, float]:
        low = float(self.g[0] - 1.0 if threshold == 1 else self.g[threshold - 2])
        return low, float(self.g[threshold - 1])


_UNIFORM_GRID_CACHE: dict[int, ThresholdGrid] = {}


def _uniform_grid(z: int) -> ThresholdGrid:
    if z not in _UNIFORM_GRID_CACHE:
        _UNIFORM_GRID_CACHE[z] = ThresholdGrid(z)
    return _UNIFORM_GRID_CACHE[z]


@dataclass(frozen=True)
class BucketLevel:
    intervals: tuple[tuple[int, int], ...]
    group_to_bucket: np.ndarray
    representatives: np.ndarray

    @property
    def count(self) -> int:
        return len(self.intervals)


def dyadic_levels(m: int) -> list[BucketLevel]:
    if m < 1:
        raise ValueError("m must be positive")
    intervals: tuple[tuple[int, int], ...] = ((0, m),)
    levels: list[BucketLevel] = []
    while True:
        mapping = np.empty(m, dtype=np.int64)
        reps = np.empty(len(intervals), dtype=np.int64)
        for bucket, (start, stop) in enumerate(intervals):
            mapping[start:stop] = bucket
            reps[bucket] = (start + stop - 1) // 2
        levels.append(BucketLevel(intervals, mapping, reps))
        if len(intervals) == m:
            return levels
        refined: list[tuple[int, int]] = []
        for start, stop in intervals:
            if stop - start <= 1:
                refined.append((start, stop))
            else:
                midpoint = (start + stop) // 2
                refined.extend(((start, midpoint), (midpoint, stop)))
        intervals = tuple(refined)


Pair = tuple[int, int, int]


def equilibrium(
    u: int, v: int, biases: np.ndarray, grid: ThresholdGrid
) -> Pair | None:
    first, second = (u, v) if biases[u] >= biases[v] else (v, u)
    threshold = grid.delta_star(float(biases[first] - biases[second]))
    return None if threshold is None else (first, second, threshold)


def probe_count(z: int, threshold: int) -> int:
    count, low, high = 2, 0, z - 1
    while high - low > 1:
        midpoint = (low + high) // 2
        count += 1
        if midpoint < threshold:
            low = midpoint
        else:
            high = midpoint
    return count


def probe_equilibrium(
    u: int, v: int, biases: np.ndarray, grid: ThresholdGrid
) -> tuple[Pair | None, int]:
    """Probe one ordered pair and return its influential equilibrium and cost.

    The cost follows the same endpoint-plus-binary-search accounting used for
    representative learning. If the supplied orientation is not influential,
    two endpoint probes establish that fact before the reverse orientation is
    searched. Four endpoint probes certify that neither orientation works.
    """
    result = equilibrium(u, v, biases, grid)
    if result is None:
        return None, 4
    success = probe_count(grid.z, result[2])
    return result, success if result[0] == u else 2 + success


def learned_representatives(
    level: BucketLevel, biases: np.ndarray, grid: ThresholdGrid
) -> tuple[dict[tuple[int, int], Pair | None], int]:
    learned: dict[tuple[int, int], Pair | None] = {}
    queries = 0
    reps = level.representatives
    for left in range(level.count):
        for right in range(left + 1, level.count):
            eq, pair_queries = probe_equilibrium(
                int(reps[left]), int(reps[right]), biases, grid
            )
            learned[(left, right)] = eq
            queries += pair_queries
    return learned, queries


def certify_matching(
    matching: list[tuple[int, int]] | None,
    level: BucketLevel,
    learned: dict[tuple[int, int], Pair | None],
    inside_flip: np.ndarray,
    biases: np.ndarray,
    grid: ThresholdGrid,
) -> tuple[list[tuple[Pair, Pair]] | None, int, int]:
    """Re-probe matched pairs and materialize an influential final query.

    Matching is performed with representative-derived thresholds. This final
    pass implements the theorem's equilibrium correction: each candidate pair
    is probed using its learned orientation, the returned true threshold is
    substituted into the query, and a pair with no influential threshold is
    dropped. The returned counts are certification probes and dropped pairs.
    """
    if matching is None:
        return None, 0, 0
    pairs: list[tuple[Pair, Pair]] = []
    queries = 0
    dropped = 0
    for u, v in matching:
        candidate = learned_pair(u, v, level, learned, inside_flip)
        if candidate is None:
            dropped += 1
            continue
        certified, pair_queries = probe_equilibrium(
            candidate[0], candidate[1], biases, grid
        )
        queries += pair_queries
        if certified is None:
            dropped += 1
            continue
        pairs.append((certified, certified))
    return pairs, queries, dropped


def learned_pair(
    u: int,
    v: int,
    level: BucketLevel,
    learned: dict[tuple[int, int], Pair | None],
    inside_flip: np.ndarray,
) -> Pair | None:
    left, right = int(level.group_to_bucket[u]), int(level.group_to_bucket[v])
    if left == right:
        return (u, v, 1) if inside_flip[u, v] else (v, u, 1)
    key = (min(left, right), max(left, right))
    representative_result = learned[key]
    if representative_result is None:
        return None
    rep_first, _, threshold = representative_result
    first_bucket = left if int(level.representatives[left]) == rep_first else right
    return (u, v, threshold) if left == first_bucket else (v, u, threshold)


def maximum_matching(
    m: int,
    weight_function,
) -> list[tuple[int, int]]:
    graph = nx.Graph()
    graph.add_nodes_from(range(m))
    for u in range(m):
        for v in range(u + 1, m):
            weight = float(weight_function(u, v))
            if weight > 1e-12:
                graph.add_edge(u, v, weight=weight)
    return [tuple(sorted(pair)) for pair in nx.max_weight_matching(graph)]


def answer_matrix(
    z: int,
    biases: np.ndarray,
    pairs: list[tuple[Pair, Pair | None]],
    intents: np.ndarray,
    mode: str,
    grid: ThresholdGrid | None = None,
) -> np.ndarray:
    if grid is None:
        grid = _uniform_grid(z)
    elif grid.z != z:
        raise ValueError("grid and z must use the same rank domain")
    mean = grid.prior_mean
    answers = np.tile(mean - biases, (len(intents), 1))
    for user, source in pairs:
        if source is None:
            continue
        first, second, threshold = user
        differences = intents[:, second] - intents[:, first]
        high = differences >= threshold
        gaps = (
            np.where(high, threshold, threshold - 1)
            if mode == "fixed"
            else np.where(high, z - 1, 1 - z)
        )
        source_first, source_second, source_threshold = source
        same = (first, second) == (source_first, source_second)
        high_mean, low_mean = grid.cell_means(source_threshold)
        canonical = np.where(
            (gaps if same else -gaps) >= source_threshold, high_mean, low_mean
        )
        difference = canonical if same else -canonical
        answers[:, first] = mean - difference / 2.0 - biases[first]
        answers[:, second] = mean + difference / 2.0 - biases[second]
    return answers


def mean_tau(intents: np.ndarray, answers: np.ndarray) -> float:
    values = [
        kendalltau(intents[index], answers[index], variant="b").statistic
        for index in range(len(intents))
    ]
    return float(np.nanmean(values))


def _gain_coefficients(
    grid: ThresholdGrid,
    user: Pair,
    source: Pair | None,
    mode: str,
) -> tuple[float, float]:
    """Return intercept and relative-bias coefficient for expected gain."""
    if source is None:
        return 0.0, 0.0
    z = grid.z
    first, second, threshold = user
    source_first, source_second, source_threshold = source
    differences = grid.differences.astype(np.longdouble)
    probabilities = grid.difference_probabilities
    high = differences >= threshold
    gaps = (
        np.where(high, threshold, threshold - 1)
        if mode == "fixed"
        else np.where(high, z - 1, 1 - z)
    )
    same = (first, second) == (source_first, source_second)
    high_mean, low_mean = grid.cell_means(source_threshold)
    canonical = np.where(
        (gaps if same else -gaps) >= source_threshold, high_mean, low_mean
    )
    response_difference = canonical if same else -canonical
    intercept = float(
        np.sum(
            probabilities
            * (response_difference * differences - response_difference**2 / 2.0)
        )
    )
    coefficient = float(-np.sum(probabilities * response_difference))
    return intercept, coefficient


def pair_gain(
    z: int,
    biases: np.ndarray,
    user: Pair,
    source: Pair | None,
    mode: str,
    cache: dict[tuple[Any, ...], tuple[float, float]],
    grid: ThresholdGrid | None = None,
) -> float:
    if grid is None:
        grid = _uniform_grid(z)
    elif grid.z != z:
        raise ValueError("grid and z must use the same rank domain")
    same = source is not None and (user[0], user[1]) == (source[0], source[1])
    source_threshold = None if source is None else source[2]
    key = (grid, z, user[2], source_threshold, same, mode)
    if key not in cache:
        cache[key] = _gain_coefficients(grid, user, source, mode)
    intercept, coefficient = cache[key]
    relative_bias = float(biases[user[0]] - biases[user[1]])
    return intercept + coefficient * relative_bias


def bias_error_metrics(
    level: BucketLevel,
    biases: np.ndarray,
    grid: ThresholdGrid,
    learned: dict[tuple[int, int], Pair | None],
    inside_flip: np.ndarray,
    epsilon: float,
) -> tuple[float, float, float, int]:
    errors: list[float] = []
    inside: list[bool] = []
    for u in range(len(biases)):
        for v in range(u + 1, len(biases)):
            user = learned_pair(u, v, level, learned, inside_flip)
            if user is None:
                continue
            first, second, threshold = user
            low, high = grid.interval(threshold)
            error = abs((low + high) / 2.0 - (biases[first] - biases[second]))
            errors.append(error)
            inside.append(error <= (high - low) / 2.0 + epsilon + TOL)
    array = np.asarray(errors, dtype=float)
    return (
        float(array.mean()) if len(array) else float("nan"),
        float(array.max()) if len(array) else float("nan"),
        float(np.mean(inside)) if inside else float("nan"),
        len(array),
    )


def _structure_for_level(
    level: BucketLevel,
    biases: np.ndarray,
    grid: ThresholdGrid,
    inside_flip: np.ndarray,
) -> tuple[dict[tuple[int, int], Pair | None], int, list[tuple[int, int]]]:
    learned, queries = learned_representatives(level, biases, grid)
    variances = grid.v

    def weight(u: int, v: int) -> float:
        user = learned_pair(u, v, level, learned, inside_flip)
        return 0.0 if user is None else float(variances[user[2] - 1])

    matching = maximum_matching(len(biases), weight)
    return learned, queries, matching


def run_dataset(
    domain: FixedDomain,
    *,
    alpha: float = 0.05,
    runs: int = 5,
    intents_per_run: int = 1000,
    seed: int = 0,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every nested learning level for one fixed dataset domain."""
    m = domain.m
    z = domain.z
    biases = alpha * domain.coordinates
    grid = ThresholdGrid(z)
    levels = dyadic_levels(m)
    gain_cache: dict[tuple[Any, ...], tuple[float, float]] = {}

    def known_weight(u: int, v: int) -> float:
        relative = float(biases[u] - biases[v])
        thresholds = (
            grid.delta_star(relative),
            grid.delta_star(-relative),
        )
        return max(
            [float(grid.v[threshold - 1]) for threshold in thresholds if threshold]
            or [0.0]
        )

    known_matching = maximum_matching(
        m,
        known_weight,
    )
    known_pairs = []
    for u, v in known_matching:
        source = equilibrium(u, v, biases, grid)
        if source is not None:
            known_pairs.append((source, source))
    known_gain = sum(
        pair_gain(z, biases, user, source, "fixed", gain_cache)
        for user, source in known_pairs
    )
    utility_constant = float(m * (z * z - 1.0) / 12.0 + np.sum(biases**2))

    # Learning and matching depend on the fixed domain/bias and on thresholds,
    # not on sampled intents or the orientation of within-bucket threshold-one
    # pairs. Compute them once, then evaluate independent intent runs.
    structure_flip = np.zeros((m, m), dtype=np.int8)
    structures = []
    for level in levels:
        started = time.perf_counter()
        learned, queries, matching = _structure_for_level(
            level, biases, grid, structure_flip
        )
        structures.append(
            (level, learned, queries, matching, time.perf_counter() - started)
        )
        if progress:
            print(
                f"{domain.dataset}: learned B={level.count}/{m}, probes={queries}, "
                f"{structures[-1][4]:.3f}s",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for run in range(runs):
        rng = np.random.default_rng(seed + 1000 + run)
        intents = rng.integers(1, z + 1, size=(intents_per_run, m))
        inside_flip = np.triu(rng.integers(0, 2, size=(m, m), dtype=np.int8), 1)
        known_tau = mean_tau(intents, answer_matrix(z, biases, known_pairs, intents, "fixed"))
        no_query_answers = np.tile((z + 1.0) / 2.0 - biases, (len(intents), 1))
        no_query_tau = mean_tau(intents, no_query_answers)

        for level, learned, queries, matching, structure_seconds in structures:
            pairs, certification_queries, dropped_pairs = certify_matching(
                matching, level, learned, inside_flip, biases, grid
            )
            assert pairs is not None
            total_queries = queries + certification_queries

            robust_gain = sum(
                pair_gain(z, biases, user, source, "robust", gain_cache)
                for user, source in pairs
            )
            fixed_gain = sum(
                pair_gain(z, biases, user, source, "fixed", gain_cache)
                for user, source in pairs
            )
            robust_tau = mean_tau(
                intents, answer_matrix(z, biases, pairs, intents, "robust")
            )
            fixed_tau = mean_tau(
                intents, answer_matrix(z, biases, pairs, intents, "fixed")
            )
            representative_bias = biases[level.representatives][level.group_to_bucket]
            epsilon = 2.0 * float(np.max(np.abs(biases - representative_bias)))
            error_mean, error_max, coverage, bias_pairs = bias_error_metrics(
                level, biases, grid, learned, inside_flip, epsilon
            )
            metadata = {
                "dataset": domain.dataset,
                "dataset_label": domain.dataset_label,
                "attribute": domain.attribute,
                "attribute_label": domain.attribute_label,
                "full_cardinality": domain.full_cardinality,
                "m": m,
                "z": z,
                "run": run,
                "alpha": alpha,
                "buckets": level.count,
                "bucket_width_groups": max(
                    stop - start for start, stop in level.intervals
                ),
            }
            rows.append(
                {
                    **metadata,
                    "utility_known_gain": known_gain,
                    "utility_certified_gain": fixed_gain,
                    "utility_robust_gain": robust_gain,
                    "utility_dhat_gain": fixed_gain,
                    "utility_known_raw": known_gain - utility_constant,
                    "utility_certified_raw": fixed_gain - utility_constant,
                    "utility_robust_raw": robust_gain - utility_constant,
                    "utility_dhat_raw": fixed_gain - utility_constant,
                    "utility_no_query_raw": -utility_constant,
                    "tau_known": known_tau,
                    "tau_certified": fixed_tau,
                    "tau_robust": robust_tau,
                    "tau_dhat": fixed_tau,
                    "tau_no_query": no_query_tau,
                    "learning_probe_queries": queries,
                    "certification_probe_queries": certification_queries,
                    "probe_queries": total_queries,
                    "bias_error_mean": error_mean,
                    "bias_error_max": error_max,
                    "epsilon": epsilon,
                    "bias_bound": 0.5 + epsilon,
                    "bias_coverage": coverage,
                    "bias_pairs": bias_pairs,
                    "matched_pairs": len(pairs),
                    "dropped_pairs": dropped_pairs,
                }
            )
            timing_rows.append({**metadata, "seconds": structure_seconds})
            if progress:
                print(
                    f"{domain.dataset}: evaluated run {run}, B={level.count}/{m}",
                    flush=True,
                )

    return pd.DataFrame(rows), pd.DataFrame(timing_rows)


def summarize(results: pd.DataFrame, timings: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "dataset_label",
        "attribute",
        "attribute_label",
        "full_cardinality",
        "m",
        "z",
        "alpha",
        "buckets",
        "bucket_width_groups",
    ]
    metrics = [
        column
        for column in results.columns
        if column not in {*keys, "run"}
    ]
    aggregate = results.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    summary = aggregate.reset_index()
    runtime = (
        timings.groupby(keys, dropna=False)["seconds"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "seconds_mean",
                "median": "seconds_median",
                "std": "seconds_std",
            }
        )
    )
    return summary.merge(runtime, on=keys, validate="one_to_one")


def write_results(
    output_dir: str | Path,
    dataset: str,
    results: pd.DataFrame,
    timings: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    results.to_csv(root / f"{dataset}_runs.csv", index=False)
    timings.to_csv(root / f"{dataset}_runtime.csv", index=False)
    summary.to_csv(root / f"{dataset}_summary.csv", index=False)
