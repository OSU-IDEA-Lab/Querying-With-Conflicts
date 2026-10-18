"""Bucket-experiment methods over nested Cartesian attribute groups.

Each prefix of the configured ranking attributes defines a lexicographically
ordered SQL-style GROUP BY domain.  Bias is assigned once to each individual
attribute bucket for each run seed; a tuple receives the mean of its component
biases, scaled exactly like the original bucket experiment.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from experiments.bucket_attributes.sweep import (
    STRATEGIES,
    closed,
    kendall_tau_b_unique_intent,
    posterior,
)
from experiments.conflict_learning.experiment import ThresholdGrid

from .scheme import DatasetScheme


DEFAULT_LAMBDAS = (0.1, 0.25, 0.5, 1.0)
DEFAULT_RUNS = 5
DEFAULT_STRATEGIES = (
    "Alg. 1",
    "Fast Alg. 1",
    "Random credible pairs",
    "Single best pair",
    "Single random pair",
)


def shared_gaussian_std_fraction(run: int, seed: int = 0) -> float:
    """Draw one seeded normalized Gaussian width shared by every dataset."""
    draw_seed = zlib.crc32(f"query-gaussian-prior|{seed}|{run}".encode())
    draw = float(np.random.default_rng(draw_seed).uniform())
    return max(draw, np.finfo(float).eps)


@dataclass(frozen=True)
class PrefixDomain:
    attributes: tuple[str, ...]
    attribute_labels: tuple[str, ...]
    cardinalities: tuple[int, ...]
    domain: tuple[tuple[Any, ...], ...]
    group_ids: np.ndarray
    possible_groups: int

    @property
    def num_attributes(self) -> int:
        return len(self.attributes)

    @property
    def domain_size(self) -> int:
        """Number of observed SQL GROUP BY keys used by the algorithms."""

        return len(self.domain)

    @property
    def observed_groups(self) -> int:
        return self.domain_size

    @property
    def observed_density(self) -> float:
        return self.domain_size / self.possible_groups


def _sortable(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _tuple_sort_key(item: tuple[Any, ...]) -> tuple[tuple[int, Any], ...]:
    return tuple(_sortable(value) for value in item)


def build_prefix_domains(scheme: DatasetScheme) -> list[PrefixDomain]:
    """Build observed tuple domains for every nested attribute prefix."""

    prefixes: list[PrefixDomain] = []
    for width in range(1, len(scheme.attributes) + 1):
        attrs = scheme.attributes[:width]
        labels = scheme.attribute_labels[:width]
        tuples = [tuple(row) for row in scheme.frame.loc[:, attrs].itertuples(index=False)]
        domain = tuple(sorted(set(tuples), key=_tuple_sort_key))
        lookup = {item: index for index, item in enumerate(domain)}
        group_ids = np.fromiter(
            (lookup[item] for item in tuples), dtype=np.int64, count=len(tuples)
        )
        cards = tuple(int(scheme.frame[attr].nunique()) for attr in attrs)
        prefixes.append(
            PrefixDomain(
                attributes=attrs,
                attribute_labels=labels,
                cardinalities=cards,
                domain=domain,
                group_ids=group_ids,
                possible_groups=math.prod(cards),
            )
        )
    return prefixes


def make_component_biases(
    scheme: DatasetScheme, *, run: int
) -> dict[str, dict[Any, float]]:
    """Give every attribute bucket a stable U[0,1) draw for one run seed."""

    components: dict[str, dict[Any, float]] = {}
    for attr in scheme.attributes:
        values = sorted(pd.unique(scheme.frame[attr]).tolist(), key=_sortable)
        seed = zlib.crc32(f"{scheme.key}|{1000 + run}|{attr}".encode())
        draws = np.random.default_rng(seed).uniform(size=len(values))
        components[attr] = {
            value: float(draw) for value, draw in zip(values, draws)
        }
    return components


def tuple_biases(
    prefix: PrefixDomain,
    components: dict[str, dict[Any, float]],
    *,
    lam: float = 1.0,
) -> np.ndarray:
    """Aggregate fixed component biases, then apply original experiment scaling."""

    aggregate = np.asarray(
        [
            np.mean(
                [
                    components[attr][value]
                    for attr, value in zip(prefix.attributes, item)
                ]
            )
            for item in prefix.domain
        ],
        dtype=float,
    )
    return -float(lam) * prefix.domain_size * aggregate


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_queries(
    scheme: DatasetScheme,
    prefixes: list[PrefixDomain],
    *,
    n_queries: int = 100,
    seed: int = 7,
    min_groups: int = 3,
    max_attempts: int | None = None,
) -> list[dict[str, Any]]:
    """Generate one shared set of WHERE clauses for every attribute prefix.

    The construction mirrors the bucket experiment: select one to three filter
    attributes from a sampled row, use equality for categoricals and a seeded
    range for numerics, and retain predicates broad enough for ranking utility.
    """

    if n_queries < 1:
        raise ValueError("n_queries must be positive")
    categorical = list(scheme.categorical_filters)
    numeric = list(scheme.numeric_filters)
    attrs = categorical + numeric
    if not attrs:
        raise ValueError(f"{scheme.key} has no WHERE-clause attributes")

    frame = scheme.frame
    rng = np.random.default_rng(seed)
    queries: list[dict[str, Any]] = []
    attempts = 0
    limit = max_attempts or max(2_000, n_queries * 2_000)

    while len(queries) < n_queries and attempts < limit:
        attempts += 1
        tuple_row = frame.iloc[int(rng.integers(len(frame)))]
        size = min(len(attrs), int(rng.integers(1, 4)))
        chosen = rng.choice(attrs, size=size, replace=False).tolist()
        predicates: list[dict[str, Any]] = []
        mask = np.ones(len(frame), dtype=bool)

        for attribute in chosen:
            value = tuple_row[attribute]
            if pd.isna(value):
                break
            if attribute in categorical:
                predicates.append(
                    {"attr": attribute, "op": "=", "value": _json_scalar(value)}
                )
                mask &= frame[attribute].eq(value).fillna(False).to_numpy()
                continue

            values = pd.to_numeric(frame[attribute], errors="coerce")
            low_data = float(values.min())
            high_data = float(values.max())
            if not np.isfinite(low_data) or not np.isfinite(high_data):
                break
            if high_data == low_data:
                low = high = low_data
            else:
                width = float(rng.uniform(0.1, 0.5) * (high_data - low_data))
                low = max(float(value) - float(rng.uniform()) * width, low_data)
                high = min(low + width, high_data)
            predicates.append(
                {"attr": attribute, "op": "between", "value": [low, high]}
            )
            mask &= values.between(low, high).fillna(False).to_numpy()
        else:
            groups_by_k = {
                str(prefix.num_attributes): np.unique(prefix.group_ids[mask]).tolist()
                for prefix in prefixes
            }
            if all(len(groups) >= min_groups for groups in groups_by_k.values()):
                queries.append(
                    {
                        "direction": str(rng.choice(["ASC", "DESC"])),
                        "where": predicates,
                        "rows": int(mask.sum()),
                        "groups_by_k": groups_by_k,
                    }
                )

    if len(queries) != n_queries:
        raise RuntimeError(
            f"generated {len(queries)} of {n_queries} queries after {attempts} attempts; "
            f"review {scheme.key}'s WHERE-clause attributes"
        )
    return queries


def build_workload(
    scheme: DatasetScheme,
    *,
    n_queries: int = 100,
    query_seed: int = 7,
    min_groups: int = 3,
) -> tuple[dict[str, Any], list[PrefixDomain]]:
    prefixes = build_prefix_domains(scheme)
    effective_min_groups = min(
        min_groups, min(prefix.domain_size for prefix in prefixes)
    )
    queries = make_queries(
        scheme,
        prefixes,
        n_queries=n_queries,
        seed=query_seed,
        min_groups=effective_min_groups,
    )
    payload = {
        "schema_version": 1,
        "dataset": scheme.key,
        "dataset_label": scheme.label,
        "query_seed": query_seed,
        "minimum_returned_groups": effective_min_groups,
        "bias_assignment": "seeded per attribute bucket",
        "tuple_bias_aggregate": "arithmetic mean",
        "prefixes": [
            {
                "num_attributes": prefix.num_attributes,
                "attributes": list(prefix.attributes),
                "attribute_labels": list(prefix.attribute_labels),
                "cardinalities": list(prefix.cardinalities),
                "possible_groups": prefix.possible_groups,
                "observed_groups": prefix.domain_size,
                "observed_density": prefix.observed_density,
            }
            for prefix in prefixes
        ],
        "queries": queries,
    }
    return payload, prefixes


def answer_groups(
    pairs: list[tuple[int, int, int]],
    biases: np.ndarray,
    direction: str,
    grid: ThresholdGrid | None = None,
) -> np.ndarray:
    """Return the source response over Cartesian tuple groups."""

    groups = len(biases)
    ranks = (
        np.arange(1, groups + 1)
        if direction == "ASC"
        else np.arange(groups, 0, -1)
    )
    if grid is not None and grid.z != groups:
        raise ValueError("grid and biases must use the same group domain")
    mean = (groups + 1) / 2.0 if grid is None else grid.prior_mean
    response = np.full(groups, mean)
    for upper, lower, distance in pairs:
        sent = (ranks[lower] - ranks[upper]) >= distance
        if grid is None:
            response[upper], response[lower] = posterior(
                distance, sent, groups
            )
        else:
            posterior_distance = (
                grid.d1[distance - 1] if sent else grid.d0[distance - 1]
            )
            response[upper] = mean - posterior_distance / 2.0
            response[lower] = mean + posterior_distance / 2.0
    return response + biases


def summarize_utility(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "dataset_label",
        "prior_model",
        "lambda",
        "strategy",
        "num_attributes",
        "attribute_labels",
        "observed_groups",
        "possible_groups",
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
        "dataset_label",
        "prior_model",
        "strategy",
        "num_attributes",
        "attribute_labels",
        "observed_groups",
        "possible_groups",
    ]
    return (
        timings.groupby(keys, dropna=False)
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
    scheme: DatasetScheme,
    prefixes: list[PrefixDomain],
    payload: dict[str, Any],
    *,
    runs: Iterable[int] = range(DEFAULT_RUNS),
    lambdas: Iterable[float] = DEFAULT_LAMBDAS,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
    prior_model: Literal["uniform", "gaussian-random"] = "uniform",
    prior_seed: int = 0,
    progress: bool = True,
    checkpoint_prefix: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Algorithm 1, Fast Algorithm 1, and Random on every prefix."""

    strategy_names = list(strategies)
    unknown = sorted(set(strategy_names) - STRATEGIES.keys())
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}")
    if prior_model not in {"uniform", "gaussian-random"}:
        raise ValueError(f"unknown prior model: {prior_model}")
    queries = payload["queries"]
    result_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    def checkpoint() -> None:
        if checkpoint_prefix is None:
            return
        prefix = Path(checkpoint_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result_rows).to_csv(f"{prefix}_utility.csv", index=False)
        pd.DataFrame(timing_rows).to_csv(f"{prefix}_runtime.csv", index=False)

    for run in runs:
        components = make_component_biases(scheme, run=int(run))
        prior_std_fraction = (
            shared_gaussian_std_fraction(int(run), prior_seed)
            if prior_model == "gaussian-random"
            else float("nan")
        )
        for lam in lambdas:
            for prefix in prefixes:
                biases = tuple_biases(prefix, components, lam=float(lam))
                group_count = prefix.domain_size
                if prior_model == "gaussian-random":
                    prior_variance = (prior_std_fraction * group_count) ** 2
                    grid = ThresholdGrid.gaussian(group_count, prior_variance)
                    gap = 1.0 - grid.g
                    utility_variance = grid.v
                    strategy_options = {
                        "gap": gap,
                        "utility_variance": utility_variance,
                    }
                    prior_actual_variance = grid.prior_actual_variance
                    prior_mean = grid.prior_mean
                else:
                    prior_variance = float("nan")
                    grid = None
                    _, utility_variance = closed(group_count)
                    strategy_options = {}
                    prior_actual_variance = (group_count**2 - 1.0) / 12.0
                    prior_mean = (group_count + 1.0) / 2.0
                intent = {
                    "ASC": np.arange(1, group_count + 1),
                    "DESC": np.arange(group_count, 0, -1),
                }
                prefix_meta = {
                    "dataset": scheme.key,
                    "dataset_label": scheme.label,
                    "run": int(run),
                    "lambda": float(lam),
                    "prior_model": prior_model,
                    "prior_seed": int(prior_seed),
                    "prior_std_fraction": prior_std_fraction,
                    "prior_variance": prior_variance,
                    "prior_actual_variance": prior_actual_variance,
                    "prior_mean": prior_mean,
                    "num_attributes": prefix.num_attributes,
                    "attributes": json.dumps(prefix.attributes),
                    "attribute_labels": " × ".join(prefix.attribute_labels),
                    "cardinalities": json.dumps(prefix.cardinalities),
                    "observed_groups": prefix.domain_size,
                    "possible_groups": prefix.possible_groups,
                    "observed_density": prefix.observed_density,
                    "bias_aggregate": "mean",
                }
                for strategy_name in strategy_names:
                    rng = np.random.default_rng(
                        zlib.crc32(
                            f"{scheme.key}|{run}|{lam}|{prefix.num_attributes}|"
                            f"{strategy_name}".encode()
                        )
                    )
                    started = time.perf_counter()
                    pairs = STRATEGIES[strategy_name](
                        biases, rng, **strategy_options
                    )
                    seconds = time.perf_counter() - started
                    query_utility = float(
                        sum(
                            utility_variance[distance - 1]
                            for _, _, distance in pairs
                        )
                    )
                    timing_rows.append(
                        {
                            **prefix_meta,
                            "strategy": strategy_name,
                            "seconds": seconds,
                            "pairs": len(pairs),
                            "query_utility": query_utility,
                        }
                    )
                    answers = {
                        direction: answer_groups(
                            pairs, biases, direction, grid=grid
                        )
                        for direction in intent
                    }
                    for query_index, query in enumerate(queries):
                        direction = query["direction"]
                        groups = np.asarray(
                            query["groups_by_k"][str(prefix.num_attributes)],
                            dtype=np.int64,
                        )
                        ranking_utility = kendall_tau_b_unique_intent(
                            answers[direction][groups], intent[direction][groups]
                        )
                        result_rows.append(
                            {
                                **prefix_meta,
                                "strategy": strategy_name,
                                "query": query_index,
                                "direction": direction,
                                "ranking_utility": ranking_utility,
                                "tau": ranking_utility,
                            }
                        )
                checkpoint()
                if progress:
                    print(
                        f"{scheme.label}: run {run}, lambda {lam:g}, "
                        f"attributes {prefix.num_attributes}, groups {group_count} done",
                        flush=True,
                    )

    results = pd.DataFrame(result_rows)
    timings = pd.DataFrame(timing_rows)
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


def scheme_summary(schemes: Iterable[DatasetScheme]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scheme in schemes:
        for prefix in build_prefix_domains(scheme):
            rows.append(
                {
                    "dataset": scheme.key,
                    "dataset_label": scheme.label,
                    "source": str(scheme.source),
                    "num_attributes": prefix.num_attributes,
                    "attributes": json.dumps(prefix.attributes),
                    "attribute_labels": " × ".join(prefix.attribute_labels),
                    "cardinalities": json.dumps(prefix.cardinalities),
                    "observed_groups": prefix.domain_size,
                    "possible_groups": prefix.possible_groups,
                    "observed_density": prefix.observed_density,
                    "notes": scheme.notes,
                }
            )
    return pd.DataFrame(rows)
