"""Distribution-aware influential matching over Cartesian intent groups.

This module implements the general pairwise theorem for an arbitrary iid rank
prior and unequal empirical group probabilities.  It deliberately lives beside
the existing Cartesian workload rather than replacing it: the older
ASC/DESC/WHERE experiment and its result files remain reproducible unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
import zlib
from typing import Any, Iterable, Literal

import networkx as nx
import numpy as np
import pandas as pd

from experiments.bucket_attributes.sweep import kendall_tau_b_unique_intent

from .experiment import (
    DEFAULT_LAMBDAS,
    DEFAULT_RUNS,
    PrefixDomain,
    build_prefix_domains,
    shared_gaussian_std_fraction,
)
from .scheme import DatasetScheme


PriorModel = Literal["uniform", "zipfian", "gaussian"]
DEFAULT_PRIORS: tuple[PriorModel, ...] = ("uniform", "zipfian", "gaussian")
DEFAULT_SAMPLE_FRACTION = 1.0
DEFAULT_INTENTS = 100
DEFAULT_ZIPF_EXPONENT = 1.0
DEFAULT_GAUSSIAN_STD_FRACTION = 0.25
DEFAULT_BIAS_ATTRIBUTE_COUNT = 3
GaussianStdMode = Literal["fixed", "random"]


@dataclass(frozen=True)
class BiasAttribute:
    """One fixed, interpretable non-intent bias attribute."""

    key: str
    label: str
    source_columns: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class PriorMoments:
    """Pairwise posterior quantities for every threshold in ``1, ..., z-1``."""

    prior: np.ndarray
    thresholds: np.ndarray
    probability: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    mu_prime0: np.ndarray
    mu_prime1: np.ndarray
    alpha: np.ndarray
    alpha_prime: np.ndarray
    mean: float

    @property
    def z(self) -> int:
        return len(self.prior)


@dataclass(frozen=True)
class Constraint:
    """Best influential oriented threshold retained for a matched group pair."""

    first: int
    second: int
    threshold: int
    gain: float


def _sortable(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    return (1, str(value))


def _frequency_band(
    values: pd.Series,
    *,
    key: str,
    label: str,
    source_column: str,
) -> BiasAttribute:
    """Bucket values by their empirical listing frequency."""

    text = values.astype(str)
    counts = text.value_counts()
    bands = pd.cut(
        text.map(counts),
        bins=(0, 1, 2, 4, 9, np.inf),
        labels=(
            "1 listing",
            "2 listings",
            "3--4 listings",
            "5--9 listings",
            "10+ listings",
        ),
        include_lowest=True,
    )
    return BiasAttribute(
        key, label, (source_column,), bands.astype(str).to_numpy()
    )


def _length_band(
    values: pd.Series,
    *,
    key: str,
    label: str,
    source_column: str,
) -> BiasAttribute:
    """Create a stable four-level band from a text field's character length."""

    lengths = values.fillna("").astype(str).str.len()
    bands = pd.cut(
        lengths,
        bins=(-np.inf, 20, 40, 80, np.inf),
        labels=("short", "medium", "long", "very long"),
        include_lowest=True,
    )
    return BiasAttribute(
        key, label, (source_column,), bands.astype(str).to_numpy()
    )


def bias_attributes(scheme: DatasetScheme) -> tuple[BiasAttribute, ...]:
    """Return three fixed, interpretable non-intent bias attributes."""

    frame = scheme.frame
    if scheme.key == "amazon":
        review_count = pd.to_numeric(frame["reviews"], errors="coerce")
        review_band = pd.cut(
            review_count,
            bins=(-np.inf, 0, 9, 99, 999, np.inf),
            labels=(
                "0 reviews",
                "1--9 reviews",
                "10--99 reviews",
                "100--999 reviews",
                "1000+ reviews",
            ),
            include_lowest=True,
        )
        return (
            BiasAttribute(
                "category_id",
                "product category",
                ("category_id",),
                frame["category_id"].astype(str).to_numpy(),
            ),
            BiasAttribute(
                "isBestSeller",
                "best-seller status",
                ("isBestSeller",),
                frame["isBestSeller"].astype(str).to_numpy(),
            ),
            BiasAttribute(
                "review_count_band",
                "review-count band",
                ("reviews",),
                review_band.astype(str).to_numpy(),
            ),
        )
    if scheme.key == "pricerunner":
        # A source favoring already-popular products is a standard exposure
        # mechanism.  Frequency bands avoid a 13K-level sparse bias domain
        # while retaining a data-derived model-popularity attribute.
        return (
            _frequency_band(
                frame["ClusterID"],
                key="model_popularity_band",
                label="product-model popularity band",
                source_column="ClusterID",
            ),
            _length_band(
                frame["ProductTitle"],
                key="product_title_length_band",
                label="product-title length band",
                source_column="ProductTitle",
            ),
            _length_band(
                frame["ClusterLabel"],
                key="model_name_length_band",
                label="model-name length band",
                source_column="ClusterLabel",
            ),
        )
    if scheme.key == "compas":
        return (
            BiasAttribute(
                "race", "race", ("race",), frame["race"].astype(str).to_numpy()
            ),
            BiasAttribute(
                "sex", "sex", ("sex",), frame["sex"].astype(str).to_numpy()
            ),
            BiasAttribute(
                "c_charge_degree",
                "charge degree",
                ("c_charge_degree",),
                frame["c_charge_degree"].astype(str).to_numpy(),
            ),
        )
    if scheme.key == "flights":
        return (
            BiasAttribute(
                "class",
                "travel class",
                ("class",),
                frame["class"].astype(str).to_numpy(),
            ),
            BiasAttribute(
                "source_city",
                "source city",
                ("source_city",),
                frame["source_city"].astype(str).to_numpy(),
            ),
            BiasAttribute(
                "stops",
                "number of stops",
                ("stops",),
                frame["stops"].astype(str).to_numpy(),
            ),
        )
    if scheme.key == "census":
        age = pd.to_numeric(frame["age"], errors="coerce")
        labels = ("18--24", "25--34", "35--44", "45--54", "55--64", "65+")
        grouped = pd.cut(
            age,
            bins=(-np.inf, 24, 34, 44, 54, 64, np.inf),
            labels=labels,
            include_lowest=True,
        )
        return (
            BiasAttribute(
                "age_group",
                "age group",
                ("age",),
                grouped.astype(str).to_numpy(),
            ),
            BiasAttribute(
                "workclass",
                "work class",
                ("workclass",),
                frame["workclass"].astype(str).to_numpy(),
            ),
            BiasAttribute(
                "occupation",
                "occupation",
                ("occupation",),
                frame["occupation"].astype(str).to_numpy(),
            ),
        )
    raise KeyError(f"No distribution-aware bias attribute configured for {scheme.key!r}")


def bias_attribute(scheme: DatasetScheme) -> BiasAttribute:
    """Return the original single bias attribute for backward compatibility."""

    return bias_attributes(scheme)[0]


def prior_pmf(
    z: int,
    model: PriorModel,
    *,
    zipf_exponent: float = DEFAULT_ZIPF_EXPONENT,
    gaussian_std_fraction: float = DEFAULT_GAUSSIAN_STD_FRACTION,
) -> np.ndarray:
    """Construct a positive pmf on positions ``1, ..., z``."""

    if z < 2:
        raise ValueError("z must be at least 2")
    ranks = np.arange(1, z + 1, dtype=float)
    if model == "uniform":
        weights = np.ones(z, dtype=float)
    elif model == "zipfian":
        if not np.isfinite(zipf_exponent) or zipf_exponent <= 0:
            raise ValueError("zipf_exponent must be positive and finite")
        weights = ranks ** (-float(zipf_exponent))
    elif model == "gaussian":
        if not np.isfinite(gaussian_std_fraction) or gaussian_std_fraction <= 0:
            raise ValueError("gaussian_std_fraction must be positive and finite")
        center = (z + 1.0) / 2.0
        standard_deviation = max(float(gaussian_std_fraction) * z, 1e-6)
        weights = np.exp(-0.5 * ((ranks - center) / standard_deviation) ** 2)
    else:  # pragma: no cover - protected by CLI/type validation
        raise ValueError(f"unknown prior model: {model}")
    weights = np.maximum(weights, np.finfo(float).tiny)
    return weights / weights.sum()


def posterior_moments(prior: np.ndarray) -> PriorMoments:
    """Compute the four posterior means for every relative-rank threshold."""

    probabilities = np.asarray(prior, dtype=np.longdouble)
    if probabilities.ndim != 1 or len(probabilities) < 2:
        raise ValueError("prior must be a one-dimensional vector of length at least 2")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise ValueError("prior probabilities must be positive and finite")
    probabilities = probabilities / probabilities.sum()
    z = len(probabilities)
    ranks = np.arange(1, z + 1, dtype=np.longdouble)
    cdf = np.cumsum(probabilities)
    survival = np.cumsum(probabilities[::-1])[::-1]
    mean = np.sum(ranks * probabilities)

    p_values = np.empty(z - 1, dtype=np.longdouble)
    mu1 = np.empty(z - 1, dtype=np.longdouble)
    mu0 = np.empty(z - 1, dtype=np.longdouble)
    mu_prime1 = np.empty(z - 1, dtype=np.longdouble)
    mu_prime0 = np.empty(z - 1, dtype=np.longdouble)

    for threshold in range(1, z):
        x = np.arange(0, z - threshold)
        tail = survival[x + threshold]
        probability = np.sum(probabilities[x] * tail)
        if not 0 < probability < 1:
            raise ValueError(f"threshold {threshold} has a trivial event probability")
        p_values[threshold - 1] = probability
        mu1[threshold - 1] = np.sum(
            ranks[x] * probabilities[x] * tail
        ) / probability

        y = np.arange(threshold, z)
        lower = cdf[y - threshold]
        mu_prime1[threshold - 1] = np.sum(
            ranks[y] * probabilities[y] * lower
        ) / probability
        mu0[threshold - 1] = (mean - probability * mu1[threshold - 1]) / (
            1 - probability
        )
        mu_prime0[threshold - 1] = (
            mean - probability * mu_prime1[threshold - 1]
        ) / (1 - probability)

    alpha = mu0 - mu1
    alpha_prime = mu_prime1 - mu_prime0
    if np.any(alpha <= 0) or np.any(alpha_prime <= 0):
        raise ArithmeticError("nontrivial iid thresholds must have positive shifts")
    return PriorMoments(
        prior=np.asarray(probabilities, dtype=float),
        thresholds=np.arange(1, z, dtype=float),
        probability=np.asarray(p_values, dtype=float),
        mu0=np.asarray(mu0, dtype=float),
        mu1=np.asarray(mu1, dtype=float),
        mu_prime0=np.asarray(mu_prime0, dtype=float),
        mu_prime1=np.asarray(mu_prime1, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        alpha_prime=np.asarray(alpha_prime, dtype=float),
        mean=float(mean),
    )


def sample_row_indexes(
    rows: int,
    fraction: float,
    *,
    dataset: str,
    run: int,
    seed: int,
) -> np.ndarray:
    """Draw one uniform row sample without replacement for an experimental run."""

    if rows < 2:
        raise ValueError("at least two rows are required")
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must lie in (0, 1]")
    size = min(rows, max(2, int(round(fraction * rows))))
    if size == rows:
        return np.arange(rows, dtype=np.int64)
    draw_seed = zlib.crc32(f"distributional-sample|{seed}|{dataset}|{run}".encode())
    return np.sort(
        np.random.default_rng(draw_seed).choice(rows, size=size, replace=False)
    )


def seeded_bias_values(
    attribute: BiasAttribute,
    *,
    dataset: str,
    run: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the bias attribute and assign one stable unit bias to each value."""

    values = sorted(pd.unique(attribute.values).tolist(), key=_sortable)
    lookup = {value: index for index, value in enumerate(values)}
    encoded = np.fromiter(
        (lookup[value] for value in attribute.values),
        dtype=np.int64,
        count=len(attribute.values),
    )
    draw_seed = zlib.crc32(
        f"distributional-bias|{seed}|{dataset}|{run}|{attribute.key}".encode()
    )
    unit_bias = np.random.default_rng(draw_seed).uniform(size=len(values))
    return encoded, unit_bias


def seeded_joint_bias_values(
    attributes: Iterable[BiasAttribute],
    *,
    dataset: str,
    run: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, str, str, tuple[str, ...]]:
    """Encode a joint bias-attribute domain and draw one bias per joint value."""

    selected = tuple(attributes)
    if not selected:
        raise ValueError("at least one bias attribute is required")
    lengths = {len(attribute.values) for attribute in selected}
    if len(lengths) != 1:
        raise ValueError("joint bias attributes must have equal row counts")
    joint_index = pd.MultiIndex.from_arrays(
        [attribute.values for attribute in selected],
        names=[attribute.key for attribute in selected],
    )
    encoded, unique_values = pd.factorize(joint_index, sort=True)
    key = " x ".join(attribute.key for attribute in selected)
    label = " × ".join(attribute.label for attribute in selected)
    source_columns = tuple(
        column for attribute in selected for column in attribute.source_columns
    )
    draw_seed = zlib.crc32(
        f"distributional-joint-bias|{seed}|{dataset}|{run}|{key}".encode()
    )
    unit_bias = np.random.default_rng(draw_seed).uniform(size=len(unique_values))
    return (
        encoded.astype(np.int64, copy=False),
        unit_bias,
        key,
        label,
        source_columns,
    )


def empirical_group_statistics(
    prefix: PrefixDomain,
    sample_indexes: np.ndarray,
    encoded_bias_values: np.ndarray,
    unit_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ``P(e)`` and ``E[unit bias(A) | e]`` on one row sample."""

    groups = prefix.group_ids[sample_indexes]
    bias_ids = encoded_bias_values[sample_indexes]
    counts = np.bincount(groups, minlength=prefix.domain_size).astype(float)
    active = np.flatnonzero(counts > 0)
    if len(active) < 2:
        raise ValueError("sample contains fewer than two intent groups")
    weighted = np.bincount(
        groups,
        weights=unit_bias[bias_ids],
        minlength=prefix.domain_size,
    )
    probabilities = counts[active] / counts[active].sum()
    conditional_bias = weighted[active] / counts[active]
    return active, probabilities, conditional_bias


def _oriented_best(
    first: np.ndarray,
    second: np.ndarray,
    group_probabilities: np.ndarray,
    biases: np.ndarray,
    moments: PriorMoments,
    *,
    batch_size: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the best threshold and gain for the supplied pair orientations."""

    count = len(first)
    best_gain = np.zeros(count, dtype=float)
    best_threshold = np.zeros(count, dtype=np.int64)
    threshold = moments.thresholds[None, :]
    p_factor = (moments.probability * (1.0 - moments.probability))[None, :]
    center = (moments.mu0 + moments.mu1)[None, :] / 2.0
    center_prime = (moments.mu_prime0 + moments.mu_prime1)[None, :] / 2.0
    alpha = moments.alpha[None, :]
    alpha_prime = moments.alpha_prime[None, :]
    alpha_squared = alpha**2
    alpha_prime_squared = alpha_prime**2
    z = float(moments.z)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        left = first[start:stop]
        right = second[start:stop]
        probability_left = group_probabilities[left, None]
        probability_right = group_probabilities[right, None]
        weight = probability_left * alpha
        weight_prime = probability_right * alpha_prime
        k_value = weight * (biases[left, None] - center) - weight_prime * (
            biases[right, None] - center_prime
        )
        lower = np.maximum(
            weight_prime * threshold - weight,
            weight_prime * z - weight * (z - threshold + 1.0),
        )
        upper = np.minimum(
            weight_prime * (threshold + 1.0) - weight,
            weight_prime * z - weight * (z - threshold),
        )
        valid = (k_value >= lower - tolerance) & (k_value <= upper + tolerance)
        gain = p_factor * (
            probability_left * alpha_squared
            + probability_right * alpha_prime_squared
        )
        gain = np.where(valid, gain, 0.0)
        indexes = np.argmax(gain, axis=1)
        values = gain[np.arange(stop - start), indexes]
        best_gain[start:stop] = values
        best_threshold[start:stop] = np.where(values > 0, indexes + 1, 0)
    return best_gain, best_threshold


def influential_matching(
    group_probabilities: np.ndarray,
    biases: np.ndarray,
    moments: PriorMoments,
    *,
    batch_size: int = 2048,
    tolerance: float = 1e-10,
) -> list[Constraint]:
    """Compute the exact maximum-weight matching from the general theorem."""

    probabilities = np.asarray(group_probabilities, dtype=float)
    biases = np.asarray(biases, dtype=float)
    if probabilities.shape != biases.shape or probabilities.ndim != 1:
        raise ValueError("group probabilities and biases must be equal-length vectors")
    if len(probabilities) > moments.z:
        raise ValueError("the group domain cannot exceed the rank-prior support")
    if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("group probabilities must be positive and sum to one")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    first, second = np.triu_indices(len(probabilities), 1)
    forward_gain, forward_threshold = _oriented_best(
        first,
        second,
        probabilities,
        biases,
        moments,
        batch_size=batch_size,
        tolerance=tolerance,
    )
    reverse_gain, reverse_threshold = _oriented_best(
        second,
        first,
        probabilities,
        biases,
        moments,
        batch_size=batch_size,
        tolerance=tolerance,
    )
    reverse = reverse_gain > forward_gain
    gains = np.where(reverse, reverse_gain, forward_gain)
    oriented_first = np.where(reverse, second, first)
    oriented_second = np.where(reverse, first, second)
    thresholds = np.where(reverse, reverse_threshold, forward_threshold)
    keep = gains > 0

    graph = nx.Graph()
    graph.add_weighted_edges_from(
        zip(first[keep].tolist(), second[keep].tolist(), gains[keep].tolist())
    )
    lookup = {
        (int(left), int(right)): index
        for index, (left, right) in enumerate(zip(first.tolist(), second.tolist()))
    }
    constraints: list[Constraint] = []
    for left, right in nx.max_weight_matching(graph):
        index = lookup[(min(left, right), max(left, right))]
        constraints.append(
            Constraint(
                first=int(oriented_first[index]),
                second=int(oriented_second[index]),
                threshold=int(thresholds[index]),
                gain=float(gains[index]),
            )
        )
    return constraints


def evaluate_constraints_on_distribution(
    constraints: Iterable[Constraint],
    group_probabilities: np.ndarray,
    biases: np.ndarray,
    moments: PriorMoments,
    *,
    tolerance: float = 1e-10,
) -> tuple[float, int]:
    """Evaluate fixed constraints under a target population distribution.

    A constraint learned from a sample contributes its target-distribution
    information gain only when it remains influential under the target group
    masses and conditional biases.  Non-influential constraints fall back to
    the no-query response and contribute zero.  The second return value is the
    number of constraints that remain influential.
    """

    constraints = list(constraints)
    probabilities = np.asarray(group_probabilities, dtype=float)
    biases = np.asarray(biases, dtype=float)
    if probabilities.shape != biases.shape or probabilities.ndim != 1:
        raise ValueError("group probabilities and biases must be equal-length vectors")
    if len(probabilities) != moments.z:
        raise ValueError("the target distribution must cover the full rank support")
    if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("target group probabilities must be positive and sum to one")
    if not constraints:
        return 0.0, 0

    first = np.asarray([item.first for item in constraints], dtype=np.int64)
    second = np.asarray([item.second for item in constraints], dtype=np.int64)
    threshold_index = (
        np.asarray([item.threshold for item in constraints], dtype=np.int64) - 1
    )
    if (
        np.any(first < 0)
        or np.any(second < 0)
        or np.any(first >= len(probabilities))
        or np.any(second >= len(probabilities))
        or np.any(threshold_index < 0)
        or np.any(threshold_index >= len(moments.thresholds))
    ):
        raise ValueError("constraint index or threshold is outside the target domain")

    probability_left = probabilities[first]
    probability_right = probabilities[second]
    alpha = moments.alpha[threshold_index]
    alpha_prime = moments.alpha_prime[threshold_index]
    weight = probability_left * alpha
    weight_prime = probability_right * alpha_prime
    threshold = moments.thresholds[threshold_index]
    center = (moments.mu0[threshold_index] + moments.mu1[threshold_index]) / 2.0
    center_prime = (
        moments.mu_prime0[threshold_index] + moments.mu_prime1[threshold_index]
    ) / 2.0
    k_value = weight * (biases[first] - center) - weight_prime * (
        biases[second] - center_prime
    )
    z = float(moments.z)
    lower = np.maximum(
        weight_prime * threshold - weight,
        weight_prime * z - weight * (z - threshold + 1.0),
    )
    upper = np.minimum(
        weight_prime * (threshold + 1.0) - weight,
        weight_prime * z - weight * (z - threshold),
    )
    valid = (k_value >= lower - tolerance) & (k_value <= upper + tolerance)
    p_factor = moments.probability[threshold_index] * (
        1.0 - moments.probability[threshold_index]
    )
    gains = p_factor * (
        probability_left * alpha**2 + probability_right * alpha_prime**2
    )
    return float(gains[valid].sum()), int(valid.sum())


def information_gain_on_distribution(
    constraints: Iterable[Constraint],
    group_probabilities: np.ndarray,
    moments: PriorMoments,
) -> float:
    """Return target-population information gain without a credibility test."""

    constraints = list(constraints)
    probabilities = np.asarray(group_probabilities, dtype=float)
    if probabilities.shape != (moments.z,):
        raise ValueError("the target distribution must cover the full rank support")
    if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("target group probabilities must be positive and sum to one")
    gain = 0.0
    for item in constraints:
        index = item.threshold - 1
        p_factor = moments.probability[index] * (
            1.0 - moments.probability[index]
        )
        gain += p_factor * (
            probabilities[item.first] * moments.alpha[index] ** 2
            + probabilities[item.second] * moments.alpha_prime[index] ** 2
        )
    return float(gain)


def sample_iid_intents(
    moments: PriorMoments,
    *,
    intents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw the exact iid rank-vector intents used by the query workload."""

    if intents < 1:
        raise ValueError("intents must be positive")
    return rng.choice(
        np.arange(1, moments.z + 1),
        size=(intents, moments.z),
        p=moments.prior,
    )


def information_gain_for_intents(
    constraints: Iterable[Constraint],
    group_probabilities: np.ndarray,
    moments: PriorMoments,
    intent_ranks: np.ndarray,
) -> float:
    """Evaluate posterior-variance gain on a fixed set of sampled intents."""

    constraints = list(constraints)
    ranks = np.asarray(intent_ranks)
    if ranks.ndim != 2 or ranks.shape[1] != moments.z or len(ranks) < 1:
        raise ValueError("intent_ranks must have shape (n, z) with n positive")
    realized = np.zeros(len(ranks), dtype=float)
    for constraint in constraints:
        index = constraint.threshold - 1
        sent = (
            ranks[:, constraint.second] - ranks[:, constraint.first]
            >= constraint.threshold
        )
        first_posterior = np.where(sent, moments.mu1[index], moments.mu0[index])
        second_posterior = np.where(
            sent, moments.mu_prime1[index], moments.mu_prime0[index]
        )
        realized += group_probabilities[constraint.first] * (
            first_posterior - moments.mean
        ) ** 2
        realized += group_probabilities[constraint.second] * (
            second_posterior - moments.mean
        ) ** 2
    return float(realized.mean())


def sampled_information_gain(
    constraints: Iterable[Constraint],
    group_probabilities: np.ndarray,
    moments: PriorMoments,
    *,
    intents: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo check of the exact posterior-variance information gain."""

    ranks = sample_iid_intents(moments, intents=intents, rng=rng)
    return information_gain_for_intents(
        constraints, group_probabilities, moments, ranks
    )


def ranking_utility_for_intents(
    constraints: Iterable[Constraint],
    biases: np.ndarray,
    moments: PriorMoments,
    intent_ranks: np.ndarray,
) -> tuple[float, float, float]:
    """Return query, babbling, and gain in mean Kendall tau-b.

    Each sampled intent sends the conjunction determined by the selected
    disjoint constraints.  The source response uses the corresponding
    posterior mean for every matched group and its prior mean for every
    unmatched group, then subtracts the expected source bias ``b(e)``.
    """

    constraints = list(constraints)
    biases = np.asarray(biases, dtype=float)
    ranks = np.asarray(intent_ranks)
    if biases.shape != (moments.z,):
        raise ValueError("biases must have one value per intent group")
    if ranks.ndim != 2 or ranks.shape[1] != moments.z or len(ranks) < 1:
        raise ValueError("intent_ranks must have shape (n, z) with n positive")

    babbling = moments.mean - biases
    answers = np.tile(babbling, (len(ranks), 1))
    for constraint in constraints:
        index = constraint.threshold - 1
        sent = (
            ranks[:, constraint.second] - ranks[:, constraint.first]
            >= constraint.threshold
        )
        answers[:, constraint.first] = np.where(
            sent, moments.mu1[index], moments.mu0[index]
        ) - biases[constraint.first]
        answers[:, constraint.second] = np.where(
            sent, moments.mu_prime1[index], moments.mu_prime0[index]
        ) - biases[constraint.second]

    query_values = np.asarray(
        [
            kendall_tau_b_unique_intent(answer, intent)
            for answer, intent in zip(answers, ranks)
        ],
        dtype=float,
    )
    babbling_values = np.asarray(
        [kendall_tau_b_unique_intent(babbling, intent) for intent in ranks],
        dtype=float,
    )
    query_mean = (
        float(np.nanmean(query_values))
        if np.isfinite(query_values).any()
        else float("nan")
    )
    babbling_mean = (
        float(np.nanmean(babbling_values))
        if np.isfinite(babbling_values).any()
        else float("nan")
    )
    return query_mean, babbling_mean, query_mean - babbling_mean


def run_distributional_experiment(
    scheme: DatasetScheme,
    *,
    priors: Iterable[PriorModel] = DEFAULT_PRIORS,
    runs: Iterable[int] = range(DEFAULT_RUNS),
    lambdas: Iterable[float] = DEFAULT_LAMBDAS,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    sample_seed: int = 0,
    bias_seed: int = 0,
    intents: int = DEFAULT_INTENTS,
    intent_seed: int = 0,
    zipf_exponent: float = DEFAULT_ZIPF_EXPONENT,
    gaussian_std_fraction: float = DEFAULT_GAUSSIAN_STD_FRACTION,
    gaussian_std_mode: GaussianStdMode = "fixed",
    gaussian_prior_seed: int = 0,
    bias_attribute_count: int = DEFAULT_BIAS_ATTRIBUTE_COUNT,
    batch_size: int = 2048,
    progress: bool = True,
) -> pd.DataFrame:
    """Run exact distribution-aware MI for one real dataset."""

    prior_models = tuple(priors)
    invalid = sorted(set(prior_models) - set(DEFAULT_PRIORS))
    if invalid:
        raise ValueError(f"unknown priors: {invalid}")
    if gaussian_std_mode not in {"fixed", "random"}:
        raise ValueError("gaussian_std_mode must be 'fixed' or 'random'")
    prefixes = build_prefix_domains(scheme)
    configured_attributes = bias_attributes(scheme)
    if not 1 <= bias_attribute_count <= len(configured_attributes):
        raise ValueError(
            f"bias_attribute_count must lie in [1, {len(configured_attributes)}]"
        )
    selected_attributes = configured_attributes[:bias_attribute_count]
    intent_columns = set(scheme.attributes)
    for attribute in selected_attributes:
        overlap = intent_columns.intersection(attribute.source_columns)
        if overlap:
            raise ValueError(
                f"bias attribute {attribute.key!r} reuses intent columns {sorted(overlap)}"
            )
    rows: list[dict[str, Any]] = []

    for run in runs:
        indexes = sample_row_indexes(
            len(scheme.frame),
            sample_fraction,
            dataset=scheme.key,
            run=int(run),
            seed=sample_seed,
        )
        run_gaussian_std_fraction = (
            shared_gaussian_std_fraction(int(run), gaussian_prior_seed)
            if gaussian_std_mode == "random"
            else float(gaussian_std_fraction)
        )
        (
            encoded_bias,
            unit_bias,
            joint_key,
            joint_label,
            joint_source_columns,
        ) = seeded_joint_bias_values(
            selected_attributes,
            dataset=scheme.key,
            run=int(run),
            seed=bias_seed,
        )
        for prefix in prefixes:
            _, probabilities, conditional_unit_bias = empirical_group_statistics(
                prefix,
                indexes,
                encoded_bias,
                unit_bias,
            )
            groups = len(probabilities)
            if groups < 2:
                continue
            for prior_model in prior_models:
                prior = prior_pmf(
                    groups,
                    prior_model,
                    zipf_exponent=zipf_exponent,
                    gaussian_std_fraction=run_gaussian_std_fraction,
                )
                moments = posterior_moments(prior)
                rng_seed = zlib.crc32(
                    f"distributional-intent|{intent_seed}|{scheme.key}|{run}|"
                    f"{prefix.num_attributes}|{prior_model}".encode()
                )
                intent_ranks = sample_iid_intents(
                    moments,
                    intents=intents,
                    rng=np.random.default_rng(rng_seed),
                )
                for strength in lambdas:
                    biases = float(strength) * groups * conditional_unit_bias
                    started = time.perf_counter()
                    matching = influential_matching(
                        probabilities,
                        biases,
                        moments,
                        batch_size=batch_size,
                    )
                    seconds = time.perf_counter() - started
                    exact_gain = float(sum(item.gain for item in matching))
                    sampled_gain = information_gain_for_intents(
                        matching,
                        probabilities,
                        moments,
                        intent_ranks,
                    )
                    ranking, babbling_ranking, ranking_gain = (
                        ranking_utility_for_intents(
                            matching,
                            biases,
                            moments,
                            intent_ranks,
                        )
                    )
                    rows.append(
                        {
                            "dataset": scheme.key,
                            "dataset_label": scheme.label,
                            "run": int(run),
                            "lambda": float(strength),
                            "prior_model": prior_model,
                            "zipf_exponent": float(zipf_exponent),
                            "gaussian_std_mode": gaussian_std_mode,
                            "gaussian_std_fraction": float(
                                run_gaussian_std_fraction
                            ),
                            "gaussian_prior_seed": int(gaussian_prior_seed),
                            "num_intent_attributes": prefix.num_attributes,
                            "intent_attributes": " × ".join(prefix.attribute_labels),
                            "bias_attribute": joint_key,
                            "bias_attribute_label": joint_label,
                            "bias_source_columns": " × ".join(
                                joint_source_columns
                            ),
                            "bias_attribute_count": len(selected_attributes),
                            "joint_bias_values": len(unit_bias),
                            "sample_fraction": float(sample_fraction),
                            "sample_rows": len(indexes),
                            "observed_groups": groups,
                            "seconds": seconds,
                            "pairs": len(matching),
                            "expected_utility_gain": exact_gain,
                            "sampled_utility_gain": sampled_gain,
                            "ranking_utility": ranking,
                            "babbling_ranking_utility": babbling_ranking,
                            "ranking_utility_gain": ranking_gain,
                            "sampled_intents": int(intents),
                        }
                    )
                    if progress:
                        print(
                            f"{scheme.label}: run {run}, A={joint_key}, "
                            f"{prior_model}, k={prefix.num_attributes}, "
                            f"lambda={float(strength):g} done",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def summarize_distributional_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the two paper metrics over runs and bias strengths."""

    keys = [
        "dataset",
        "dataset_label",
        "prior_model",
        "num_intent_attributes",
        "bias_attribute",
        "bias_attribute_label",
        "bias_attribute_count",
    ]
    return (
        results.groupby(keys, as_index=False)
        .agg(
            mean_seconds=("seconds", "mean"),
            median_seconds=("seconds", "median"),
            mean_utility_gain=("expected_utility_gain", "mean"),
            mean_sampled_utility_gain=("sampled_utility_gain", "mean"),
            mean_ranking_utility=("ranking_utility", "mean"),
            mean_babbling_ranking_utility=("babbling_ranking_utility", "mean"),
            mean_ranking_utility_gain=("ranking_utility_gain", "mean"),
            mean_pairs=("pairs", "mean"),
            mean_observed_groups=("observed_groups", "mean"),
            samples=("seconds", "size"),
        )
    )
