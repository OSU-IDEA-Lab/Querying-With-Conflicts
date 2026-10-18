"""Dataset specifications for the conflict-learning experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConflictDataset:
    key: str
    label: str
    filename: str
    attribute: str
    attribute_label: str
    categorical: bool = False
    log_numeric: bool = False


DATASETS: dict[str, ConflictDataset] = {
    "amazon": ConflictDataset(
        "amazon", "Amazon", "amazon_products.csv", "price", "Price", log_numeric=True
    ),
    "pricerunner": ConflictDataset(
        "pricerunner",
        "PriceRunner",
        "pricerunner_simplified.csv",
        "ProductTitle",
        "Product model",
        categorical=True,
    ),
    "compas": ConflictDataset("compas", "COMPAS", "compas.csv", "age", "Age"),
    "flights": ConflictDataset(
        "flights", "Flights", "flights.csv", "price", "Price", log_numeric=True
    ),
    "census": ConflictDataset(
        "census", "Census", "census.csv", "education_num", "Education level"
    ),
}


@dataclass(frozen=True)
class FixedDomain:
    dataset: str
    dataset_label: str
    attribute: str
    attribute_label: str
    categorical: bool
    full_cardinality: int
    values: tuple[object, ...]
    coordinates: np.ndarray
    coordinate_transform: str = "linear-minmax"

    @property
    def m(self) -> int:
        return len(self.values)

    @property
    def z(self) -> int:
        return self.m


def _natural_numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").dropna().unique()
    return np.sort(numeric.astype(float))


def _categorical(values: pd.Series) -> np.ndarray:
    return np.asarray(sorted(values.dropna().astype(str).unique().tolist()), dtype=object)


def load_fixed_domain(
    spec: ConflictDataset,
    data_root: str | Path,
    *,
    max_groups: int = 1024,
    domain_seed: int = 17,
    cluster_categorical: bool = False,
    transform_numeric: bool = False,
) -> FixedDomain:
    """Load one reproducible, ordered, unbucketized group domain.

    Large domains are sampled once with a dataset-specific seed and then sorted.
    The selected values and their biases remain fixed across every run and every
    learning-bucket level.
    """
    if max_groups < 2:
        raise ValueError("max_groups must be at least two")
    path = Path(data_root) / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    values = pd.read_csv(path, usecols=[spec.attribute])[spec.attribute]
    distinct = _categorical(values) if spec.categorical else _natural_numeric(values)
    full_cardinality = len(distinct)
    if full_cardinality < 2:
        raise ValueError(f"{spec.key}.{spec.attribute} must have at least two values")

    if full_cardinality > max_groups:
        seed = zlib.crc32(f"{spec.key}|{domain_seed}|domain".encode())
        chosen = np.random.default_rng(seed).choice(
            full_cardinality, size=max_groups, replace=False
        )
        distinct = distinct[np.sort(chosen)]

    m = len(distinct)
    coordinate_transform = "linear-minmax"
    if spec.categorical and cluster_categorical:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        strings = distinct.astype(str)
        vectors = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            max_features=50000,
        ).fit_transform(strings)
        dimensions = min(16, m - 1, vectors.shape[1] - 1)
        if dimensions < 1:
            order = np.arange(m)
        else:
            embedded = TruncatedSVD(
                n_components=dimensions,
                random_state=domain_seed,
            ).fit_transform(vectors)
            tree = linkage(
                embedded,
                method="average",
                metric="cosine",
                optimal_ordering=True,
            )
            order = leaves_list(tree)
        distinct = distinct[order]
        coordinates = np.linspace(0.0, m - 1.0, m)
        coordinate_transform = "char-ngram-cluster-order"
    elif spec.categorical:
        coordinates = np.linspace(0.0, m - 1.0, m)
        coordinate_transform = "lexicographic-rank"
    else:
        numeric = distinct.astype(float)
        use_log = spec.log_numeric and transform_numeric
        transformed = np.log1p(numeric) if use_log else numeric
        span = float(transformed[-1] - transformed[0])
        coordinates = (
            np.linspace(0.0, m - 1.0, m)
            if span == 0
            else (transformed - transformed[0]) * (m - 1.0) / span
        )
        coordinate_transform = "log1p-minmax" if use_log else "linear-minmax"

    return FixedDomain(
        dataset=spec.key,
        dataset_label=spec.label,
        attribute=spec.attribute,
        attribute_label=spec.attribute_label,
        categorical=spec.categorical,
        full_cardinality=full_cardinality,
        values=tuple(distinct.tolist()),
        coordinates=np.asarray(coordinates, dtype=float),
        coordinate_transform=coordinate_transform,
    )
