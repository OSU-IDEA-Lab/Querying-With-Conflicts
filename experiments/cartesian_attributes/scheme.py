"""Fixed, auditable attribute schemes for Cartesian-domain experiments.

Prefixes of each attribute list form nested Cartesian domains.  The experiment
plots the number of included attributes while retaining the resulting Cartesian
cardinality as metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DATASET_ORDER = ("amazon", "pricerunner", "compas", "flights", "census")


@dataclass(frozen=True)
class DatasetScheme:
    key: str
    label: str
    source: Path
    frame: pd.DataFrame
    attributes: tuple[str, ...]
    attribute_labels: tuple[str, ...]
    categorical_filters: tuple[str, ...]
    numeric_filters: tuple[str, ...]
    notes: str

    @property
    def cardinalities(self) -> tuple[int, ...]:
        return tuple(int(self.frame[col].nunique()) for col in self.attributes)


def _read_selected(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path, usecols=columns)


def _load_amazon(data_root: Path) -> DatasetScheme:
    path = data_root / "amazon_products.csv"
    categorical = ("category_id", "isBestSeller")
    numeric = ("reviews", "listPrice")
    df = _read_selected(
        path,
        ["price", "stars", "boughtInLastMonth", *categorical, *numeric],
    )

    price = pd.to_numeric(df["price"], errors="coerce")
    stars = pd.to_numeric(df["stars"], errors="coerce")
    bought = pd.to_numeric(df["boughtInLastMonth"], errors="coerce").fillna(0)

    # Match the repository's existing Amazon scalability preparation.
    df["price_decile"] = pd.qcut(price, q=10, labels=False, duplicates="drop")
    df["stars_whole"] = np.floor(stars)
    df["bought_last_month"] = bought

    attrs = ("stars_whole", "price_decile", "bought_last_month")
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df.loc[:, attrs] = df.loc[:, attrs].astype("Int64")
    frame = df.dropna(subset=list(attrs)).reset_index(drop=True)
    return DatasetScheme(
        key="amazon",
        label="Amazon",
        source=path,
        frame=frame,
        attributes=attrs,
        attribute_labels=("Stars", "Price", "Purchases"),
        categorical_filters=categorical,
        numeric_filters=numeric,
        notes=(
            "Uses the existing preparation: six whole-star levels, ten price "
            "deciles, and thirty distinct bought-last-month values."
        ),
    )


def _top_values_plus_other(series: pd.Series, *, count: int) -> pd.Series:
    text = series.astype(str)
    keep = set(text.value_counts().index[:count])
    return text.where(text.isin(keep), "Other")


def _load_pricerunner(data_root: Path) -> DatasetScheme:
    path = data_root / "pricerunner_simplified.csv"
    categorical = ("CategoryLabel",)
    df = _read_selected(
        path,
        [
            "ProductTitle",
            "ClusterLabel",
            "brand",
            "MerchantID",
            "CategoryID",
            "ClusterID",
            *categorical,
        ],
    )
    df["brand_top10"] = _top_values_plus_other(df["brand"], count=10)
    df["merchant_top10"] = _top_values_plus_other(df["MerchantID"], count=10)
    df["category"] = df["CategoryID"].astype(str)

    attrs = ("brand_top10", "category", "merchant_top10")
    frame = df.dropna(subset=list(attrs)).reset_index(drop=True)
    return DatasetScheme(
        key="pricerunner",
        label="PriceRunner",
        source=path,
        frame=frame,
        attributes=attrs,
        attribute_labels=("Brand", "Category", "Merchant"),
        categorical_filters=categorical,
        numeric_filters=(),
        notes=(
            "Matches the existing loader: top-ten mappings for brand and merchant "
            "with an Other remainder, plus ten prepared categories."
        ),
    )


def _load_compas(data_root: Path) -> DatasetScheme:
    path = data_root / "compas_bucketized_for_cartesian.csv"
    attrs = ("priors_count", "age_cat_binned", "v_decile_score")
    categorical = ("sex", "race", "c_charge_degree", "score_text")
    numeric = ("decile_score", "juv_fel_count", "juv_misd_count", "juv_other_count")
    frame = _read_selected(path, [*attrs, *categorical, *numeric])
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(attrs)).reset_index(drop=True)
    return DatasetScheme(
        key="compas",
        label="COMPAS",
        source=path,
        frame=frame,
        attributes=attrs,
        attribute_labels=("Prior count", "Age", "Violence score"),
        categorical_filters=categorical,
        numeric_filters=numeric,
        notes=(
            "Uses the Cartesian-ready COMPAS file: two prior-count buckets, three "
            "age buckets, and ten violence-score levels."
        ),
    )


def _load_flights(data_root: Path) -> DatasetScheme:
    path = data_root / "flights_bucketized.csv"
    attrs = ("price_category", "airline", "days_category")
    categorical = (
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
    )
    numeric = ("duration",)
    frame = _read_selected(path, [*attrs, *categorical, *numeric])
    frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce")
    frame = frame.dropna(subset=list(attrs)).reset_index(drop=True)
    return DatasetScheme(
        key="flights",
        label="Flights",
        source=path,
        frame=frame,
        attributes=attrs,
        attribute_labels=("Price", "Airline", "Days left"),
        categorical_filters=categorical,
        numeric_filters=numeric,
        notes="Uses the repository's existing bucketized flight attributes.",
    )


def _load_census(data_root: Path) -> DatasetScheme:
    path = data_root / "census.csv"
    attrs = ("sex", "race", "education_num")
    categorical = (
        "workclass",
        "marital_status",
        "occupation",
        "relationship",
        "native_country",
    )
    numeric = ("age", "hours_per_week", "capital_gain", "capital_loss")
    frame = _read_selected(path, [*attrs, *categorical, *numeric])
    frame["education_num"] = pd.to_numeric(frame["education_num"], errors="coerce")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(attrs)).reset_index(drop=True)
    return DatasetScheme(
        key="census",
        label="Census",
        source=path,
        frame=frame,
        attributes=attrs,
        attribute_labels=("Sex", "Race", "Education"),
        categorical_filters=categorical,
        numeric_filters=numeric,
        notes=(
            "Uses the raw Census data with attributes ordered by increasing "
            "cardinality: sex, race, and education number."
        ),
    )


_LOADERS: dict[str, Callable[[Path], DatasetScheme]] = {
    "amazon": _load_amazon,
    "pricerunner": _load_pricerunner,
    "compas": _load_compas,
    "flights": _load_flights,
    "census": _load_census,
}


def load_dataset_scheme(key: str, data_root: str | Path) -> DatasetScheme:
    """Load one fixed scheme and validate its nested-domain invariants."""

    normalized = key.lower()
    if normalized not in _LOADERS:
        raise KeyError(f"Unknown dataset {key!r}; choose from {sorted(_LOADERS)}")
    scheme = _LOADERS[normalized](Path(data_root))
    cards = scheme.cardinalities
    if any(card < 2 for card in cards):
        raise ValueError(f"Every selected attribute must vary; got {cards}")
    if tuple(sorted(cards)) != cards:
        raise ValueError(
            f"Attributes must be ordered by nondecreasing cardinality; got {cards}"
        )
    return scheme
