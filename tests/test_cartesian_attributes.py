import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cartesian_attributes.experiment import (
    answer_groups,
    build_prefix_domains,
    build_workload,
    make_component_biases,
    run_experiment,
    scheme_summary,
    shared_gaussian_std_fraction,
    tuple_biases,
)
from experiments.cartesian_attributes.scheme import DatasetScheme


class CartesianAttributeTests(unittest.TestCase):
    def setUp(self):
        frame = pd.DataFrame(
            {
                "a": [0, 0, 1, 1],
                "b": ["x", "y", "x", "y"],
                "c": [1, 1, 1, 2],
                "filter": ["u", "u", "v", "v"],
            }
        )
        self.scheme = DatasetScheme(
            key="toy",
            label="Toy",
            source=Path("toy.csv"),
            frame=frame,
            attributes=("a", "b", "c"),
            attribute_labels=("A", "B", "C"),
            categorical_filters=("filter",),
            numeric_filters=(),
            notes="test",
        )

    def test_prefixes_use_observed_group_keys_and_record_possible_product(self):
        prefixes = build_prefix_domains(self.scheme)
        self.assertEqual([prefix.domain_size for prefix in prefixes], [2, 4, 4])
        self.assertEqual([prefix.possible_groups for prefix in prefixes], [2, 4, 8])
        self.assertEqual(prefixes[-1].observed_density, 0.5)

    def test_component_bias_is_fixed_and_mean_combined(self):
        prefix = build_prefix_domains(self.scheme)[1]
        first = make_component_biases(self.scheme, run=0)
        second = make_component_biases(self.scheme, run=0)
        self.assertEqual(first, second)
        combined = tuple_biases(prefix, first, lam=1.0)
        for item, value in zip(prefix.domain, combined):
            expected = -prefix.domain_size * np.mean(
                [first["a"][item[0]], first["b"][item[1]]]
            )
            self.assertAlmostEqual(value, expected)
            self.assertLessEqual(value, 0.0)
            self.assertGreaterEqual(value, -prefix.domain_size)

    def test_scheme_summary_records_domain_size(self):
        summary = scheme_summary([self.scheme])
        self.assertEqual(summary["observed_groups"].tolist(), [2, 4, 4])
        self.assertEqual(summary["possible_groups"].tolist(), [2, 4, 8])
        self.assertIn("observed_density", summary.columns)

    def test_group_answer_has_one_value_per_tuple(self):
        biases = np.array([-0.2, -0.5, -0.1])
        answer = answer_groups([], biases, "ASC")
        np.testing.assert_allclose(answer, np.full(3, 2.0) + biases)

    def test_gaussian_width_draw_is_seeded_and_shared(self):
        first = shared_gaussian_std_fraction(run=2, seed=11)
        repeated = shared_gaussian_std_fraction(run=2, seed=11)
        another_run = shared_gaussian_std_fraction(run=3, seed=11)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, another_run)
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)

    def test_small_workload_and_sweep_reuse_queries_across_prefixes(self):
        # A broader toy frame is needed to keep at least two groups after filtering.
        frame = pd.DataFrame(
            {
                "a": np.tile([0, 1, 2], 8),
                "b": np.repeat(["x", "y"], 12),
                "filter": np.tile(["u", "v"], 12),
            }
        )
        scheme = DatasetScheme(
            key="toy",
            label="Toy",
            source=Path("toy.csv"),
            frame=frame,
            attributes=("b", "a"),
            attribute_labels=("B", "A"),
            categorical_filters=("filter",),
            numeric_filters=(),
            notes="test",
        )
        payload, prefixes = build_workload(
            scheme, n_queries=4, query_seed=7, min_groups=2
        )
        self.assertEqual(len(payload["queries"]), 4)
        self.assertTrue(
            all(set(query["groups_by_k"]) == {"1", "2"} for query in payload["queries"])
        )
        results, timings, utility, runtime = run_experiment(
            scheme,
            prefixes,
            payload,
            runs=[0],
            lambdas=[0.5],
            strategies=["Fast Alg. 1", "Random credible pairs"],
            progress=False,
        )
        self.assertEqual(len(results), 4 * 2 * 2)
        self.assertEqual(len(timings), 2 * 2)
        self.assertIn("seconds", timings.columns)
        self.assertFalse(utility.empty)
        self.assertFalse(runtime.empty)

    def test_gaussian_random_prior_is_recorded_and_used_across_prefixes(self):
        frame = pd.DataFrame(
            {
                "a": np.tile([0, 1, 2], 8),
                "b": np.repeat(["x", "y"], 12),
                "filter": np.tile(["u", "v"], 12),
            }
        )
        scheme = DatasetScheme(
            key="toy",
            label="Toy",
            source=Path("toy.csv"),
            frame=frame,
            attributes=("b", "a"),
            attribute_labels=("B", "A"),
            categorical_filters=("filter",),
            numeric_filters=(),
            notes="test",
        )
        payload, prefixes = build_workload(
            scheme, n_queries=4, query_seed=7, min_groups=2
        )
        results, timings, utility, runtime = run_experiment(
            scheme,
            prefixes,
            payload,
            runs=[0],
            lambdas=[0.5],
            strategies=["Fast Alg. 1"],
            prior_model="gaussian-random",
            prior_seed=19,
            progress=False,
        )
        self.assertTrue((results["prior_model"] == "gaussian-random").all())
        self.assertEqual(results["prior_std_fraction"].nunique(), 1)
        for row in timings.itertuples():
            expected = (row.prior_std_fraction * row.observed_groups) ** 2
            self.assertAlmostEqual(row.prior_variance, expected)
            self.assertGreater(row.prior_actual_variance, 0.0)
        self.assertTrue(np.isfinite(timings["query_utility"]).all())
        self.assertFalse(utility.empty)
        self.assertFalse(runtime.empty)


if __name__ == "__main__":
    unittest.main()
