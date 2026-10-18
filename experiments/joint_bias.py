"""Joint, distribution-aware bias functions for fixed-domain experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from experiments.conflict_learning.scheme import FixedDomain


@dataclass(frozen=True)
class JointBiasDistribution:
    """Empirical ``P(A_b | e)`` for one three-attribute joint bias domain."""

    dataset: str
    key: str
    label: str
    source_columns: tuple[str, ...]
    joint_values: int
    conditional: csr_matrix
    rows: int

    @property
    def groups(self) -> int:
        return int(self.conditional.shape[0])


def _top_values_plus_other(series: pd.Series, count: int = 10) -> pd.Series:
    values = series.fillna("Missing").astype(str)
    keep = set(values.value_counts().index[:count])
    return values.where(values.isin(keep), "Other")


def _frequency_band(series: pd.Series) -> pd.Series:
    values = series.fillna("Missing").astype(str)
    counts = values.value_counts()
    return pd.cut(
        values.map(counts),
        bins=(0, 1, 2, 4, 9, np.inf),
        labels=("1 listing", "2 listings", "3--4 listings", "5--9 listings", "10+ listings"),
        include_lowest=True,
    ).astype(str)


def _bias_columns(dataset: str, frame: pd.DataFrame) -> tuple[
    tuple[str, str, pd.Series], ...
]:
    if dataset == "amazon":
        reviews = pd.to_numeric(frame["reviews"], errors="coerce")
        review_band = pd.cut(
            reviews,
            bins=(-np.inf, 0, 9, 99, 999, np.inf),
            labels=("0 reviews", "1--9 reviews", "10--99 reviews", "100--999 reviews", "1000+ reviews"),
            include_lowest=True,
        ).astype(str)
        return (
            ("category_id", "product category", frame["category_id"].fillna("Missing").astype(str)),
            ("isBestSeller", "best-seller status", frame["isBestSeller"].fillna("Missing").astype(str)),
            ("review_count_band", "review-count band", review_band),
        )
    if dataset == "pricerunner":
        return (
            ("brand_top10", "brand", _top_values_plus_other(frame["brand"])),
            ("merchant_top10", "merchant", _top_values_plus_other(frame["MerchantID"])),
            ("category_label", "product category", frame["CategoryLabel"].fillna("Missing").astype(str)),
        )
    if dataset == "compas":
        return (
            ("race", "race", frame["race"].fillna("Missing").astype(str)),
            ("sex", "sex", frame["sex"].fillna("Missing").astype(str)),
            ("c_charge_degree", "charge degree", frame["c_charge_degree"].fillna("Missing").astype(str)),
        )
    if dataset == "flights":
        return (
            ("class", "travel class", frame["class"].fillna("Missing").astype(str)),
            ("source_city", "source city", frame["source_city"].fillna("Missing").astype(str)),
            ("stops", "number of stops", frame["stops"].fillna("Missing").astype(str)),
        )
    if dataset == "census":
        age = pd.to_numeric(frame["age"], errors="coerce")
        age_group = pd.cut(
            age,
            bins=(-np.inf, 24, 34, 44, 54, 64, np.inf),
            labels=("18--24", "25--34", "35--44", "45--54", "55--64", "65+"),
            include_lowest=True,
        ).astype(str)
        return (
            ("age_group", "age group", age_group),
            ("workclass", "work class", frame["workclass"].fillna("Missing").astype(str)),
            ("occupation", "occupation", frame["occupation"].fillna("Missing").astype(str)),
        )
    raise KeyError(f"No joint bias attributes configured for {dataset!r}")


def _required_columns(domain: FixedDomain) -> tuple[str, ...]:
    configured = {
        "amazon": ("category_id", "isBestSeller", "reviews"),
        "pricerunner": ("brand", "MerchantID", "CategoryLabel"),
        "compas": ("race", "sex", "c_charge_degree"),
        "flights": ("class", "source_city", "stops"),
        "census": ("age", "workclass", "occupation"),
    }
    return (domain.attribute, *configured[domain.dataset])


def load_joint_bias_distribution(
    domain: FixedDomain,
    data_root: str | Path,
) -> JointBiasDistribution:
    """Estimate the full-data conditional distribution of joint bias values."""

    filenames = {
        "amazon": "amazon_products.csv",
        "pricerunner": "pricerunner_simplified.csv",
        "compas": "compas.csv",
        "flights": "flights.csv",
        "census": "census.csv",
    }
    path = Path(data_root) / filenames[domain.dataset]
    frame = pd.read_csv(path, usecols=list(dict.fromkeys(_required_columns(domain))))

    if domain.categorical:
        ranking_values = frame[domain.attribute].astype(str)
        lookup = {str(value): index for index, value in enumerate(domain.values)}
    else:
        ranking_values = pd.to_numeric(frame[domain.attribute], errors="coerce")
        lookup = {float(value): index for index, value in enumerate(domain.values)}
    mapped = ranking_values.map(lookup)
    active = mapped.notna().to_numpy()
    group_ids = mapped[active].astype(np.int64).to_numpy()

    attributes = _bias_columns(domain.dataset, frame)
    joint_index = pd.MultiIndex.from_arrays(
        [values.to_numpy()[active] for _, _, values in attributes],
        names=[key for key, _, _ in attributes],
    )
    joint_ids, unique_values = pd.factorize(joint_index, sort=True)
    counts = csr_matrix(
        (
            np.ones(len(group_ids), dtype=float),
            (group_ids, joint_ids.astype(np.int64, copy=False)),
        ),
        shape=(domain.m, len(unique_values)),
    )
    row_counts = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(row_counts <= 0):
        missing = np.flatnonzero(row_counts <= 0).tolist()
        raise ValueError(f"selected intent groups have no distribution rows: {missing}")
    conditional = counts.multiply((1.0 / row_counts)[:, None]).tocsr()
    return JointBiasDistribution(
        dataset=domain.dataset,
        key=" x ".join(key for key, _, _ in attributes),
        label=" × ".join(label for _, label, _ in attributes),
        source_columns=tuple(column for column in _required_columns(domain)[1:]),
        joint_values=len(unique_values),
        conditional=conditional,
        rows=len(group_ids),
    )


def joint_expected_biases(
    distribution: JointBiasDistribution,
    *,
    m: int,
    lambda_bias: float,
    run: int,
    seed: int,
) -> np.ndarray:
    """Draw value-level biases and return ``E[b(A_b) | e]`` for every group."""

    if distribution.groups != m:
        raise ValueError("joint distribution and fixed domain disagree on m")
    draw_seed = zlib.crc32(
        f"{distribution.dataset}|{seed}|{run}|joint-bias|{distribution.key}".encode()
    )
    unit = np.random.default_rng(draw_seed).uniform(-1.0, 0.0, size=distribution.joint_values)
    return np.asarray(distribution.conditional @ unit).ravel() * (lambda_bias * m)
