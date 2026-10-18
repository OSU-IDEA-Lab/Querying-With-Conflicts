import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiments.bucket_attributes.sweep as sweep
from experiments.bucket_attributes.plot import summarize_query_utility
from experiments.bucket_attributes.sweep import (
    dyadic_bucket_levels,
    kendall_tau_b_unique_intent,
    run_experiment,
    single_any,
    single_best,
)
from experiments.bucket_attributes.workload import (
    DATASETS,
    add_fine_groups,
    assign_categorical_fine_groups,
    assign_continuous_fine_groups,
    make_queries,
)


class FineGroupTests(unittest.TestCase):
    def test_continuous_groups_preserve_numeric_order(self):
        values = pd.Series([40.0, 10.0, 30.0, 20.0, 50.0, 60.0, 70.0, 80.0])
        groups, count = assign_continuous_fine_groups(values)
        ordered = pd.DataFrame({"value": values, "group": groups}).sort_values("value")
        self.assertEqual(count, 8)
        self.assertEqual(ordered["group"].tolist(), sorted(ordered["group"].tolist()))
        self.assertEqual(np.bincount(groups).tolist(), [1] * 8)

    def test_categorical_groups_are_lexicographic_and_cover_all_values(self):
        values = pd.Series(["zeta", "alpha", "gamma", "beta", "zeta"])
        groups, count = assign_categorical_fine_groups(values)
        mapping = dict(zip(values, groups))
        self.assertEqual(count, 4)
        self.assertLessEqual(mapping["alpha"], mapping["beta"])
        self.assertLessEqual(mapping["beta"], mapping["gamma"])
        self.assertLessEqual(mapping["gamma"], mapping["zeta"])

    def test_pricerunner_queries_use_product_model_as_ranking_attribute(self):
        spec = DATASETS["pricerunner"]
        frame = pd.DataFrame(
            {
                "ProductTitle": [f"model-{index:02d}" for index in range(20)],
                "MerchantID": [str(index % 2) for index in range(20)],
                "CategoryID": [str(index % 3) for index in range(20)],
                "CategoryLabel": [f"cat-{index % 3}" for index in range(20)],
                "brand": [f"brand-{index % 4}" for index in range(20)],
            }
        )
        grouped, count, _ = add_fine_groups(frame, spec)
        queries = make_queries(grouped, spec, n_queries=3, seed=7, min_groups=2)
        self.assertEqual(spec.ranking_label, "product model")
        self.assertEqual(count, 20)
        self.assertEqual(len(queries), 3)
        self.assertTrue(all(len(query["groups"]) >= 2 for query in queries))

    def test_dyadic_levels_are_nested_and_end_at_full_domain(self):
        levels = dyadic_bucket_levels(5)
        self.assertEqual([level.count for level in levels], [1, 2, 4, 5])
        np.testing.assert_array_equal(levels[0].fine_to_bucket, [0, 0, 0, 0, 0])
        np.testing.assert_array_equal(levels[-1].fine_to_bucket, [0, 1, 2, 3, 4])
        for coarse, fine in zip(levels, levels[1:]):
            for fine_bucket in range(fine.count):
                members = np.flatnonzero(fine.fine_to_bucket == fine_bucket)
                self.assertEqual(len(np.unique(coarse.fine_to_bucket[members])), 1)


class UtilityTests(unittest.TestCase):
    def test_single_pair_baselines_select_at_most_one_pair(self):
        biases = np.array([-0.1, -0.8, -1.5, -2.4, -3.0, -4.2])
        rng = np.random.default_rng(123)
        self.assertLessEqual(len(single_best(biases, rng)), 1)
        for _ in range(20):
            self.assertLessEqual(len(single_any(biases, rng)), 1)

    def test_single_any_stops_after_first_credible_pair(self):
        biases = np.zeros(8)
        rng = np.random.default_rng(7)
        original_orient = sweep.orient
        calls = []

        def recording_orient(first, second, *args):
            calls.append((len(first), len(second)))
            return original_orient(first, second, *args)

        with patch.object(sweep, "orient", side_effect=recording_orient):
            pairs = sweep.single_any(biases, rng)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(calls, [(1, 1)])

    def test_kendall_tau_b(self):
        intent = np.array([1, 2, 3])
        self.assertAlmostEqual(kendall_tau_b_unique_intent(intent, intent), 1.0)
        self.assertAlmostEqual(kendall_tau_b_unique_intent(intent[::-1], intent), -1.0)
        tied = kendall_tau_b_unique_intent(np.array([1, 1, 2]), intent)
        self.assertAlmostEqual(tied, 2 / math.sqrt(6))

    def test_sweep_records_both_utility_metrics(self):
        payload = {
            "dataset": "tiny",
            "ranking_attribute": "value",
            "attribute_type": "continuous",
            "fine_groups": 4,
            "queries": [
                {"direction": "ASC", "groups": [0, 1, 2, 3]},
                {"direction": "DESC", "groups": [0, 2, 3]},
            ],
        }
        results, timings, utility_summary, runtime_summary = run_experiment(
            payload,
            runs=[0],
            lambdas=[0.1],
            buckets=[1, 2, 4],
            strategies=["Fast Alg. 1"],
            progress=False,
        )
        self.assertIn("ranking_utility", results.columns)
        self.assertIn("tau", results.columns)
        self.assertTrue(np.allclose(results["ranking_utility"], results["tau"]))
        self.assertIn("query_utility", timings.columns)
        self.assertTrue((timings["status"] == "completed").all())
        self.assertIn("mean", utility_summary.columns)
        self.assertIn("mean_query_utility", runtime_summary.columns)

    def test_gaussian_random_prior_is_used_at_each_bucket_level(self):
        payload = {
            "dataset": "tiny",
            "ranking_attribute": "value",
            "attribute_type": "continuous",
            "fine_groups": 4,
            "queries": [
                {"direction": "ASC", "groups": [0, 1, 2, 3]},
                {"direction": "DESC", "groups": [0, 2, 3]},
            ],
        }
        results, timings, _, _ = run_experiment(
            payload,
            runs=[0],
            lambdas=[0.1],
            buckets=[1, 2, 4],
            strategies=["Fast Alg. 1"],
            prior_model="gaussian-random",
            prior_seed=0,
            progress=False,
        )
        self.assertTrue(results["prior_model"].eq("gaussian-random").all())
        self.assertEqual(timings["prior_std_fraction"].nunique(), 1)
        expected = (
            timings["prior_std_fraction"] * timings["buckets"]
        ) ** 2
        np.testing.assert_allclose(timings["prior_variance"], expected)
        self.assertTrue(np.isfinite(timings["query_utility"]).all())

    def test_query_utility_summary_keeps_lambda_dimension(self):
        runtime = pd.DataFrame(
            {
                "lambda": [0.1, 0.1, 0.5, 0.5],
                "strategy": ["Alg. 1"] * 4,
                "buckets": [10] * 4,
                "query_utility": [2.0, 4.0, 6.0, 8.0],
            }
        )
        summary = summarize_query_utility(runtime)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["lambda"].tolist(), [0.1, 0.5])
        self.assertEqual(summary["mean"].tolist(), [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
