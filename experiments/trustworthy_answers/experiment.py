"""Scalability experiment for range-based trustworthy-answer detection.

The experiment retains the paper's original setup: ``k=80`` returned tuples
and ``z`` equal to 0.5--10 percent of each dataset.  An iid intent assigns a
rank to every observed intent group.  The data source orders the relevant
tuples by that rank adjusted by the tuple-level bias, and its first ``k``
tuples are the interpretation result beta(I) checked by the detector.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from experiments.cartesian_attributes.distributional import (
    bias_attributes,
    seeded_bias_values,
)
from experiments.cartesian_attributes.experiment import (
    DEFAULT_LAMBDAS,
    DEFAULT_RUNS,
    build_prefix_domains,
    shared_gaussian_std_fraction,
)
from experiments.cartesian_attributes.scheme import DatasetScheme


PriorModel = Literal["uniform", "gaussian"]
AttributeSetting = Literal["distinct", "coinciding"]

DEFAULT_FRACTIONS_PERCENT = tuple(np.arange(0.5, 10.0 + 0.5, 0.5))
DEFAULT_K = 80
DEFAULT_PRIORS: tuple[PriorModel, ...] = ("uniform", "gaussian")
DEFAULT_SETTINGS: tuple[AttributeSetting, ...] = ("distinct", "coinciding")
DEFAULT_BIAS_ATTRIBUTE_COUNT = 3


@dataclass(frozen=True)
class DetectionGrid:
    """Prior-dependent constants in the closed-form detection theorem."""

    z: int
    prior: np.ndarray
    rho: np.ndarray
    s: np.ndarray
    g_plus: np.ndarray
    g_minus: np.ndarray

    @classmethod
    def uniform(cls, z: int) -> "DetectionGrid":
        if z < 2:
            raise ValueError("z must be at least two")
        delta = np.arange(1, z, dtype=float)
        probability = (z - delta) * (z - delta + 1.0) / (2.0 * z * z)
        d1 = (z + 2.0 * delta - 1.0) / 3.0
        d0 = -d1 * probability / (1.0 - probability)
        g = delta - 0.5 * (d0 + d1)
        return cls(
            z=z,
            prior=np.full(z, 1.0 / z, dtype=float),
            rho=np.ones(z - 1, dtype=float),
            s=-d0,
            g_plus=g,
            g_minus=g - 1.0,
        )

    @classmethod
    def centered_gaussian(cls, z: int, std_fraction: float) -> "DetectionGrid":
        """Construct the exact symmetric-Gaussian constants in linear time.

        The positive difference probabilities have a closed form involving a
        prefix sum on integer and half-integer grids.  This avoids materializing
        the ``z`` by ``z`` pair table while evaluating the same posterior
        quantities as the theorem.
        """

        if z < 2:
            raise ValueError("z must be at least two")
        if not np.isfinite(std_fraction) or std_fraction <= 0:
            raise ValueError("std_fraction must be positive and finite")

        sigma = max(float(std_fraction) * z, np.finfo(float).eps)
        center = (z + 1.0) / 2.0
        ranks = np.arange(1, z + 1, dtype=float)
        log_weights = -0.5 * ((ranks - center) / sigma) ** 2
        log_weights -= float(np.max(log_weights))
        weights = np.exp(log_weights)
        normalizer = float(weights.sum())
        prior = weights / normalizer

        integer_kernel = np.exp(-((ranks - center) / sigma) ** 2)
        integer_prefix = np.concatenate(([0.0], np.cumsum(integer_kernel)))
        half_points = np.arange(1, z, dtype=float) + 0.5
        half_kernel = np.exp(-((half_points - center) / sigma) ** 2)
        half_prefix = np.concatenate(([0.0], np.cumsum(half_kernel)))

        differences = np.arange(1, z, dtype=np.int64)
        inner_sum = np.empty(z - 1, dtype=float)

        even = differences % 2 == 0
        even_d = differences[even]
        even_h = even_d // 2
        # Integer points 1+h, ..., z-h map to zero-based slice [h:z-h].
        inner_sum[even] = integer_prefix[z - even_h] - integer_prefix[even_h]

        odd = ~even
        odd_d = differences[odd]
        odd_h = (odd_d - 1) // 2
        # Half points 1.5+h, ..., z-h-0.5 map to [h:z-h-1].
        inner_sum[odd] = half_prefix[z - odd_h - 1] - half_prefix[odd_h]

        positive_difference = (
            np.exp(-(differences.astype(float) ** 2) / (4.0 * sigma**2))
            * inner_sum
            / (normalizer**2)
        )
        # Guard only against roundoff; every finite-support Gaussian cell is
        # strictly positive.
        positive_difference = np.maximum(
            positive_difference, np.nextafter(0.0, 1.0)
        )
        tail_probability = np.cumsum(positive_difference[::-1])[::-1]
        tail_moment = np.cumsum(
            (differences.astype(float) * positive_difference)[::-1]
        )[::-1]
        if np.any(tail_probability <= 0) or np.any(tail_probability >= 1):
            raise ArithmeticError("Gaussian threshold probabilities are invalid")
        d1 = tail_moment / tail_probability
        d0 = -tail_moment / (1.0 - tail_probability)
        g = differences.astype(float) - 0.5 * (d0 + d1)
        return cls(
            z=z,
            prior=prior,
            rho=np.ones(z - 1, dtype=float),
            s=-d0,
            g_plus=g,
            g_minus=g - 1.0,
        )


def general_range_filter(
    tuple_biases: np.ndarray,
    expected_group_biases: np.ndarray,
    bias_lower: float,
    bias_upper: float,
    grid: DetectionGrid,
    *,
    threshold_batch: int = 8192,
) -> tuple[np.ndarray, int]:
    """Return the trustworthy mask from the general range-based algorithm."""

    tuple_biases = np.asarray(tuple_biases, dtype=float)
    expected_group_biases = np.asarray(expected_group_biases, dtype=float)
    if tuple_biases.shape != expected_group_biases.shape or tuple_biases.ndim != 1:
        raise ValueError("returned tuple and group biases must be equal-length vectors")
    if bias_lower > bias_upper:
        raise ValueError("bias range is reversed")
    if threshold_batch < 1:
        raise ValueError("threshold_batch must be positive")

    untrustworthy = np.zeros(len(tuple_biases), dtype=bool)
    checks = 0
    for start in range(0, grid.z - 1, threshold_batch):
        active = np.flatnonzero(~untrustworthy)
        if len(active) == 0:
            break
        stop = min(start + threshold_batch, grid.z - 1)
        rho = grid.rho[start:stop][None, :]
        lower = (
            rho * expected_group_biases[active, None]
            - grid.g_plus[start:stop][None, :]
        )
        upper = (
            rho * expected_group_biases[active, None]
            - grid.g_minus[start:stop][None, :]
        )
        first = np.maximum(float(bias_lower), lower)
        condition_one = (first <= float(bias_upper)) & (first < upper)
        condition_two = float(bias_lower) < (
            tuple_biases[active, None] - grid.s[start:stop][None, :]
        )
        found = np.any(condition_one & condition_two, axis=1)
        untrustworthy[active[found]] = True
        checks += len(active) * (stop - start)
    return ~untrustworthy, checks


def _merged_special_intervals(grid: DetectionGrid) -> tuple[np.ndarray, np.ndarray]:
    """Merge the feasible relative-bias intervals for the symmetric setting."""

    if not np.allclose(grid.rho, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("the special detector requires a symmetric iid prior")
    lower = np.maximum(grid.g_minus, grid.s)
    upper = grid.g_plus
    valid = lower < upper
    valid_lower = lower[valid]
    valid_upper = upper[valid]
    if len(valid_lower) == 0:
        return np.empty(0), np.empty(0)
    # For the uniform and centered-Gaussian priors used here, the intervals
    # overlap in threshold order.  Recognize that common case in O(z); retain
    # the generic merge only as a defensive fallback.
    running_upper = np.maximum.accumulate(valid_upper)
    if np.all(valid_lower[1:] <= running_upper[:-1]):
        return (
            np.asarray([float(valid_lower.min())]),
            np.asarray([float(valid_upper.max())]),
        )
    intervals = sorted(
        zip(valid_lower, valid_upper), key=lambda pair: pair[0]
    )
    merged: list[list[float]] = [[float(intervals[0][0]), float(intervals[0][1])]]
    for current_lower, current_upper in intervals[1:]:
        if current_lower <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], float(current_upper))
        else:
            merged.append([float(current_lower), float(current_upper)])
    return (
        np.asarray([item[0] for item in merged], dtype=float),
        np.asarray([item[1] for item in merged], dtype=float),
    )


def special_threshold_filter(
    biases: np.ndarray,
    bias_lower: float,
    bias_upper: float,
    grid: DetectionGrid,
    *,
    intervals: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Return the trustworthy mask when tuple and expected group biases agree."""

    biases = np.asarray(biases, dtype=float)
    interval_lower, interval_upper = (
        _merged_special_intervals(grid) if intervals is None else intervals
    )
    if len(interval_lower) == 0:
        return np.ones(len(biases), dtype=bool), len(biases)

    # A possible competing bias x in [bias_lower,bias_upper] induces the
    # relative-bias range [b-bias_upper,b-bias_lower].  The merged intervals
    # are open on the left and closed on the right.
    difference_lower = biases - float(bias_upper)
    difference_upper = biases - float(bias_lower)
    indexes = np.searchsorted(interval_upper, difference_lower, side="left")
    untrustworthy = np.zeros(len(biases), dtype=bool)
    in_bounds = indexes < len(interval_lower)
    untrustworthy[in_bounds] = (
        difference_upper[in_bounds] > interval_lower[indexes[in_bounds]]
    )
    return ~untrustworthy, len(biases)


def _encode_values(values: np.ndarray) -> tuple[np.ndarray, tuple[object, ...]]:
    series = pd.Series(values)
    uniques = tuple(sorted(pd.unique(series).tolist(), key=lambda value: str(value)))
    lookup = {value: index for index, value in enumerate(uniques)}
    encoded = np.fromiter(
        (lookup[value] for value in values), dtype=np.int64, count=len(values)
    )
    return encoded, uniques


def _stable_unit_biases(
    count: int, *, dataset: str, run: int, attribute: str, seed: int
) -> np.ndarray:
    draw_seed = zlib.crc32(
        f"trustworthy-bias|{seed}|{dataset}|{run}|{attribute}".encode()
    )
    return np.random.default_rng(draw_seed).uniform(size=count)


def _query_result(
    sample_indexes: np.ndarray,
    group_ids: np.ndarray,
    group_ranks: np.ndarray,
    tuple_biases: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    """Return local indexes of the source's top-k interpretation result."""

    source_position = group_ranks[group_ids[sample_indexes]] - tuple_biases
    returned = min(int(k), len(sample_indexes))
    if returned == len(sample_indexes):
        candidates = np.arange(returned, dtype=np.int64)
    else:
        candidates = np.argpartition(source_position, returned - 1)[:returned]
    ordering = np.lexsort(
        (sample_indexes[candidates], source_position[candidates])
    )
    return candidates[ordering]


def _sample_group_ranks(
    grid: DetectionGrid,
    groups: int,
    *,
    dataset: str,
    run: int,
    fraction_percent: float,
    prior: str,
) -> np.ndarray:
    seed = zlib.crc32(
        f"trustworthy-intent|{dataset}|{run}|{fraction_percent:g}|{prior}".encode()
    )
    rng = np.random.default_rng(seed)
    if prior == "uniform":
        return rng.integers(1, grid.z + 1, size=groups).astype(float)
    return rng.choice(
        np.arange(1, grid.z + 1), size=groups, p=grid.prior
    ).astype(float)


def _joint_bias_encoding(
    scheme: DatasetScheme,
    *,
    count: int = DEFAULT_BIAS_ATTRIBUTE_COUNT,
) -> tuple[str, str, np.ndarray, int]:
    """Encode the joint values of the selected distinct bias attributes."""

    configured = bias_attributes(scheme)
    if not 1 <= count <= len(configured):
        raise ValueError(
            f"bias_attribute_count must lie in [1, {len(configured)}]"
        )
    selected = configured[:count]
    intent_columns = set(scheme.attributes)
    for attribute in selected:
        overlap = intent_columns.intersection(attribute.source_columns)
        if overlap:
            raise ValueError(
                f"bias attribute {attribute.key!r} reuses intent columns "
                f"{sorted(overlap)}"
            )

    joint_index = pd.MultiIndex.from_arrays(
        [attribute.values for attribute in selected],
        names=[attribute.key for attribute in selected],
    )
    encoded, unique_values = pd.factorize(joint_index, sort=True)
    key = " x ".join(attribute.key for attribute in selected)
    label = " × ".join(attribute.label for attribute in selected)
    return key, label, encoded.astype(np.int64, copy=False), len(unique_values)


def _general_joint_attribute(
    scheme: DatasetScheme,
    group_ids: np.ndarray,
    *,
    run: int,
    bias_seed: int,
    bias_attribute_count: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, int]:
    """Encode one joint bias function and its full-data group expectation."""

    groups = int(group_ids.max()) + 1
    counts = np.bincount(group_ids, minlength=groups).astype(float)
    key, label, encoded, joint_values = _joint_bias_encoding(
        scheme, count=bias_attribute_count
    )
    unit = _stable_unit_biases(
        joint_values,
        dataset=scheme.key,
        run=run,
        attribute=f"joint:{key}",
        seed=bias_seed,
    )
    weighted = np.bincount(group_ids, weights=unit[encoded], minlength=groups)
    expected = weighted / counts
    return key, label, encoded, unit, expected, joint_values


def _special_joint_attribute(
    scheme: DatasetScheme,
    group_ids: np.ndarray,
    *,
    run: int,
    bias_seed: int,
) -> tuple[str, str, np.ndarray, np.ndarray, int]:
    """Assign one bias to every observed joint intent-attribute value."""

    groups = int(group_ids.max()) + 1
    key = " x ".join(scheme.attributes)
    label = " × ".join(scheme.attribute_labels)
    unit = _stable_unit_biases(
        groups,
        dataset=scheme.key,
        run=run,
        attribute=f"joint-intent:{key}",
        seed=bias_seed,
    )
    return key, label, group_ids, unit, groups


def run_scalability_experiment(
    scheme: DatasetScheme,
    *,
    fractions_percent: Iterable[float] = DEFAULT_FRACTIONS_PERCENT,
    k: int = DEFAULT_K,
    priors: Iterable[PriorModel] = DEFAULT_PRIORS,
    settings: Iterable[AttributeSetting] = DEFAULT_SETTINGS,
    runs: Iterable[int] = range(DEFAULT_RUNS),
    lambdas: Iterable[float] = DEFAULT_LAMBDAS,
    sample_seed: int = 0,
    bias_seed: int = 0,
    gaussian_prior_seed: int = 0,
    bias_attribute_count: int = DEFAULT_BIAS_ATTRIBUTE_COUNT,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the paper's percentage-of-dataset trustworthy-answer benchmark."""

    fractions = tuple(float(value) for value in fractions_percent)
    prior_models = tuple(priors)
    attribute_settings = tuple(settings)
    if k < 1:
        raise ValueError("k must be positive")
    if any(not 0 < value <= 100 for value in fractions):
        raise ValueError("fractions must lie in (0,100]")
    if set(prior_models) - set(DEFAULT_PRIORS):
        raise ValueError("unsupported prior")
    if set(attribute_settings) - set(DEFAULT_SETTINGS):
        raise ValueError("unsupported attribute setting")

    prefix = build_prefix_domains(scheme)[-1]
    group_ids = prefix.group_ids
    groups = prefix.domain_size
    rows = len(scheme.frame)
    output: list[dict[str, object]] = []

    for run in runs:
        sample_rng = np.random.default_rng(
            zlib.crc32(f"trustworthy-sample|{sample_seed}|{scheme.key}|{run}".encode())
        )
        largest = min(
            rows,
            max(k, int(round(max(fractions) * rows / 100.0))),
        )
        nested_sample = sample_rng.choice(rows, size=largest, replace=False)
        general = (
            _general_joint_attribute(
                scheme,
                group_ids,
                run=int(run),
                bias_seed=bias_seed,
                bias_attribute_count=bias_attribute_count,
            )
            if "distinct" in attribute_settings
            else None
        )
        special = (
            _special_joint_attribute(
                scheme, group_ids, run=int(run), bias_seed=bias_seed
            )
            if "coinciding" in attribute_settings
            else None
        )
        gaussian_std_fraction = shared_gaussian_std_fraction(
            int(run), gaussian_prior_seed
        )

        for fraction in fractions:
            z = min(rows, max(k, int(round(fraction * rows / 100.0))))
            sample_indexes = nested_sample[:z]
            for prior_model in prior_models:
                precompute_started = time.perf_counter()
                grid = (
                    DetectionGrid.uniform(z)
                    if prior_model == "uniform"
                    else DetectionGrid.centered_gaussian(z, gaussian_std_fraction)
                )
                precompute_seconds = time.perf_counter() - precompute_started
                special_precompute_started = time.perf_counter()
                special_intervals = _merged_special_intervals(grid)
                special_precompute_seconds = (
                    precompute_seconds
                    + time.perf_counter()
                    - special_precompute_started
                )
                group_ranks = _sample_group_ranks(
                    grid,
                    groups,
                    dataset=scheme.key,
                    run=int(run),
                    fraction_percent=fraction,
                    prior=prior_model,
                )

                if "distinct" in attribute_settings:
                    assert general is not None
                    key, label, encoded, unit, expected, joint_values = general
                    for strength in lambdas:
                        scale = -float(strength) * z
                        tuple_bias = scale * unit[encoded[sample_indexes]]
                        returned_local = _query_result(
                            sample_indexes,
                            group_ids,
                            group_ranks,
                            tuple_bias,
                            k=k,
                        )
                        returned_rows = sample_indexes[returned_local]
                        started = time.perf_counter()
                        trustworthy, checks = general_range_filter(
                            tuple_bias[returned_local],
                            scale * expected[group_ids[returned_rows]],
                            -float(strength) * z,
                            0.0,
                            grid,
                        )
                        filter_seconds = time.perf_counter() - started
                        output.append(
                            {
                                "dataset": scheme.key,
                                "dataset_label": scheme.label,
                                "run": int(run),
                                "dataset_size_percent": fraction,
                                "z": z,
                                "k": min(k, z),
                                "prior_model": prior_model,
                                "gaussian_std_fraction": (
                                    gaussian_std_fraction
                                    if prior_model == "gaussian"
                                    else np.nan
                                ),
                                "attribute_setting": "distinct",
                                "bias_attribute": key,
                                "bias_attribute_label": label,
                                "bias_attribute_count": bias_attribute_count,
                                "joint_bias_values": joint_values,
                                "lambda": float(strength),
                                "precompute_seconds": precompute_seconds,
                                "filter_seconds": filter_seconds,
                                "time_seconds": precompute_seconds + filter_seconds,
                                "threshold_checks": checks,
                                "trustworthy_tuples": int(trustworthy.sum()),
                                "untrustworthy_tuples": int((~trustworthy).sum()),
                            }
                        )

                if "coinciding" in attribute_settings:
                    assert special is not None
                    key, label, encoded, unit, joint_values = special
                    for strength in lambdas:
                        scale = -float(strength) * z
                        tuple_bias = scale * unit[encoded[sample_indexes]]
                        returned_local = _query_result(
                            sample_indexes,
                            group_ids,
                            group_ranks,
                            tuple_bias,
                            k=k,
                        )
                        started = time.perf_counter()
                        trustworthy, checks = special_threshold_filter(
                            tuple_bias[returned_local],
                            -float(strength) * z,
                            0.0,
                            grid,
                            intervals=special_intervals,
                        )
                        filter_seconds = time.perf_counter() - started
                        output.append(
                            {
                                "dataset": scheme.key,
                                "dataset_label": scheme.label,
                                "run": int(run),
                                "dataset_size_percent": fraction,
                                "z": z,
                                "k": min(k, z),
                                "prior_model": prior_model,
                                "gaussian_std_fraction": (
                                    gaussian_std_fraction
                                    if prior_model == "gaussian"
                                    else np.nan
                                ),
                                "attribute_setting": "coinciding",
                                "bias_attribute": key,
                                "bias_attribute_label": label,
                                "bias_attribute_count": len(scheme.attributes),
                                "joint_bias_values": joint_values,
                                "lambda": float(strength),
                                "precompute_seconds": special_precompute_seconds,
                                "filter_seconds": filter_seconds,
                                "time_seconds": (
                                    special_precompute_seconds + filter_seconds
                                ),
                                "threshold_checks": checks,
                                "trustworthy_tuples": int(trustworthy.sum()),
                                "untrustworthy_tuples": int((~trustworthy).sum()),
                            }
                        )
                if progress:
                    print(
                        f"{scheme.label}: run {run}, {fraction:g}%, "
                        f"{prior_model}, z={z} done",
                        flush=True,
                    )
    return pd.DataFrame(output)


def summarize_scalability(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "dataset_label",
        "dataset_size_percent",
        "z",
        "k",
        "prior_model",
        "attribute_setting",
    ]
    return (
        results.groupby(keys, as_index=False)
        .agg(
            observations=("time_seconds", "size"),
            mean_seconds=("time_seconds", "mean"),
            median_seconds=("time_seconds", "median"),
            q25_seconds=("time_seconds", lambda values: values.quantile(0.25)),
            q75_seconds=("time_seconds", lambda values: values.quantile(0.75)),
            mean_precompute_seconds=("precompute_seconds", "mean"),
            mean_filter_seconds=("filter_seconds", "mean"),
            mean_threshold_checks=("threshold_checks", "mean"),
            mean_trustworthy_tuples=("trustworthy_tuples", "mean"),
        )
    )


def write_results(output: str | Path, results: pd.DataFrame) -> None:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "all_results.csv", index=False)
    summarize_scalability(results).to_csv(destination / "summary.csv", index=False)
