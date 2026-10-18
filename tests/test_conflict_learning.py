import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from experiments.conflict_learning.experiment import (
    ThresholdGrid,
    bias_error_metrics,
    discrete_gaussian_prior,
    dyadic_levels,
    equilibrium,
    g_value,
    learned_representatives,
    probe_count,
    probe_equilibrium,
    run_dataset,
    variance,
)
from experiments.conflict_learning.scheme import FixedDomain
from experiments.conflict_learning.comparison import (
    fast_influential_matching,
    known_bias_matching,
    run_dataset_comparison,
    seeded_ordered_biases,
    shared_gaussian_std_fraction,
    summarize_comparison,
)
from experiments.mi_bucketization.experiment import (
    run_fixed_domain_mi,
    seeded_biases,
    summarize as summarize_mi_bucketization,
)
from experiments.joint_bias import JointBiasDistribution, joint_expected_biases


class ConflictLearningTests(unittest.TestCase):
    def test_joint_bias_uses_conditional_expectations_and_seeded_scaling(self):
        distribution = JointBiasDistribution(
            dataset="toy",
            key="a x b x c",
            label="A × B × C",
            source_columns=("a", "b", "c"),
            joint_values=3,
            conditional=csr_matrix(
                [[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]]
            ),
            rows=4,
        )
        first = joint_expected_biases(
            distribution, m=2, lambda_bias=0.25, run=0, seed=7
        )
        repeated = joint_expected_biases(
            distribution, m=2, lambda_bias=0.25, run=0, seed=7
        )
        stronger = joint_expected_biases(
            distribution, m=2, lambda_bias=0.5, run=0, seed=7
        )
        np.testing.assert_allclose(first, repeated)
        np.testing.assert_allclose(stronger, 2.0 * first)
        self.assertTrue((first >= -0.5).all() and (first <= 0.0).all())

    def test_conflict_biases_are_seeded_sorted_uniform_realizations(self):
        first = seeded_ordered_biases("toy", 16, 0.25, 0, 7)
        repeated = seeded_ordered_biases("toy", 16, 0.25, 0, 7)
        second = seeded_ordered_biases("toy", 16, 0.25, 1, 7)
        stronger = seeded_ordered_biases("toy", 16, 0.5, 0, 7)
        np.testing.assert_allclose(first, repeated)
        np.testing.assert_allclose(stronger, 2.0 * first)
        self.assertTrue((first >= -4.0).all() and (first <= 0.0).all())
        self.assertTrue((np.diff(first) >= 0.0).all())
        self.assertFalse(np.allclose(first, second))

    def test_mi_iid_bias_uses_same_draws_without_attribute_sorting(self):
        smooth = seeded_biases("toy", 16, 0.25, 2, 0, "sorted")
        iid = seeded_biases("toy", 16, 0.25, 2, 0, "iid")
        width = 0.25 * 16
        np.testing.assert_allclose(np.sort(iid + width), smooth)
        self.assertFalse(np.all(np.diff(iid) >= 0))

    def test_dyadic_levels_keep_fixed_underlying_domain(self):
        levels = dyadic_levels(5)
        self.assertEqual([level.count for level in levels], [1, 2, 4, 5])
        self.assertTrue(all(len(level.group_to_bucket) == 5 for level in levels))
        for coarse, fine in zip(levels, levels[1:]):
            for bucket in range(fine.count):
                members = np.flatnonzero(fine.group_to_bucket == bucket)
                self.assertEqual(len(np.unique(coarse.group_to_bucket[members])), 1)

    def test_threshold_search_matches_interval_definition(self):
        grid = ThresholdGrid(20)
        for relative_bias in np.linspace(0.0, 8.0, 41):
            threshold = grid.delta_star(float(relative_bias))
            self.assertIsNotNone(threshold)
            low, high = grid.interval(threshold)
            self.assertLessEqual(low - 1e-9, relative_bias)
            self.assertLessEqual(relative_bias, high + 1e-9)

    def test_generic_uniform_prior_matches_closed_forms(self):
        z = 20
        closed = ThresholdGrid(z)
        generic = ThresholdGrid(
            z,
            np.full(z, 1.0 / z),
            prior_name="generic-uniform",
        )
        np.testing.assert_allclose(generic.g, g_value(z, closed.d), rtol=1e-12)
        np.testing.assert_allclose(generic.v, variance(z, closed.d), rtol=1e-12)
        np.testing.assert_allclose(
            generic.difference_probabilities,
            closed.difference_probabilities,
            rtol=1e-12,
        )

    def test_discrete_gaussian_prior_is_centered_and_builds_valid_grid(self):
        z = 21
        prior = discrete_gaussian_prior(z, variance=9.0)
        np.testing.assert_allclose(prior, prior[::-1], rtol=1e-14)
        self.assertAlmostEqual(float(prior.sum()), 1.0)
        self.assertAlmostEqual(
            float(np.dot(np.arange(1, z + 1, dtype=np.longdouble), prior)),
            (z + 1) / 2,
        )
        grid = ThresholdGrid.gaussian(z, variance=9.0)
        self.assertGreater(grid.prior_actual_variance, 0.0)
        self.assertLess(grid.prior_actual_variance, 9.0)
        self.assertTrue(np.isfinite(grid.g).all())
        self.assertTrue(np.isfinite(grid.v).all())
        self.assertTrue((grid.v > 0).all())
        self.assertTrue(grid.g_is_nondecreasing)
        self.assertTrue(grid.v_is_nonincreasing)

    def test_wide_gaussian_converges_to_uniform_grid(self):
        z = 16
        uniform = ThresholdGrid(z)
        gaussian = ThresholdGrid.gaussian(z, variance=1e12)
        np.testing.assert_allclose(gaussian.g, uniform.g, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(gaussian.v, uniform.v, rtol=1e-10, atol=1e-10)

    def test_concentrated_gaussian_uses_stable_tail_statistics(self):
        grid = ThresholdGrid.gaussian(64, variance=0.25)
        self.assertTrue(np.isfinite(grid.d0).all())
        self.assertTrue(np.isfinite(grid.d1).all())
        self.assertTrue(np.isfinite(grid.g).all())
        self.assertTrue(np.isfinite(grid.v).all())
        self.assertTrue((grid.v >= 0).all())
        self.assertTrue(grid.supports_adjacent_matching)

    def test_probe_count_includes_endpoint_and_binary_queries(self):
        self.assertEqual(probe_count(8, 1), 4)
        self.assertEqual(probe_count(8, 7), 5)

    def test_matched_pair_probe_returns_true_threshold_and_counts_orientation(self):
        grid = ThresholdGrid(8)
        biases = np.array([0.0, 1.0])
        result, forward_queries = probe_equilibrium(1, 0, biases, grid)
        self.assertEqual(result, equilibrium(0, 1, biases, grid))
        self.assertEqual(forward_queries, probe_count(8, result[2]))
        reverse, reverse_queries = probe_equilibrium(0, 1, biases, grid)
        self.assertEqual(reverse, result)
        self.assertEqual(reverse_queries, forward_queries + 2)

    def test_small_run_keeps_m_and_z_fixed(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=6,
            values=tuple(range(6)),
            coordinates=np.arange(6, dtype=float),
        )
        results, timings = run_dataset(
            domain,
            alpha=0.05,
            runs=1,
            intents_per_run=10,
            progress=False,
        )
        self.assertEqual(sorted(results["buckets"].unique().tolist()), [1, 2, 4, 6])
        self.assertTrue((results["m"] == 6).all())
        self.assertTrue((results["z"] == 6).all())
        self.assertEqual(len(timings), 4)
        self.assertTrue((results["bias_coverage"] >= 0).all())
        self.assertTrue((results["bias_coverage"] <= 1).all())
        self.assertIn("utility_certified_gain", results)
        self.assertTrue(
            (
                results["probe_queries"]
                == results["learning_probe_queries"]
                + results["certification_probe_queries"]
            ).all()
        )

    def test_fast_matching_uses_shared_learned_structure(self):
        biases = np.linspace(0.0, 1.5, 6)
        grid = ThresholdGrid(6)
        level = dyadic_levels(6)[-1]
        learned, _ = learned_representatives(level, biases, grid)
        matching = fast_influential_matching(level, learned, grid)
        self.assertEqual(len(matching), 3)
        self.assertEqual(sorted(group for pair in matching for group in pair), list(range(6)))

    def test_known_bias_adjacent_matching_matches_general_exact_matching(self):
        from experiments.conflict_learning.experiment import maximum_matching

        biases = np.array([0.0, 0.3, 0.8, 1.7, 2.0, 3.2, 3.3, 4.1])
        for grid in (ThresholdGrid(16), ThresholdGrid.gaussian(16, 4.0)):
            with self.subTest(prior=grid.prior_name):
                self.assertTrue(grid.supports_adjacent_matching)

                def weight(u, v):
                    thresholds = (
                        grid.delta_star(float(biases[u] - biases[v])),
                        grid.delta_star(float(biases[v] - biases[u])),
                    )
                    return max(
                        [float(grid.v[value - 1]) for value in thresholds if value]
                        or [0.0]
                    )

                adjacent = known_bias_matching(biases, weight)
                general = maximum_matching(len(biases), weight)
                self.assertAlmostEqual(
                    sum(weight(*pair) for pair in adjacent),
                    sum(weight(*pair) for pair in general),
                )

    def test_known_bias_matching_optimizes_odd_domain_unmatched_group(self):
        from experiments.conflict_learning.experiment import maximum_matching

        biases = np.array([0.0, 0.2, 0.9, 1.0, 1.8])
        grid = ThresholdGrid(8)

        def weight(u, v):
            result = equilibrium(u, v, biases, grid)
            return 0.0 if result is None else float(grid.v[result[2] - 1])

        adjacent = known_bias_matching(biases, weight)
        general = maximum_matching(len(biases), weight)
        self.assertAlmostEqual(
            sum(weight(*pair) for pair in adjacent),
            sum(weight(*pair) for pair in general),
        )

    def test_comparison_shares_learning_between_exact_and_fast(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=6,
            values=tuple(range(6)),
            coordinates=np.arange(6, dtype=float),
        )
        results, timings = run_dataset_comparison(
            domain,
            lambda_bias=0.25,
            runs=1,
            intents_per_run=10,
            exact_timeout_seconds=None,
            progress=False,
        )
        self.assertIn("utility_dhat_exact_gain", results)
        self.assertIn("utility_dhat_fast_gain", results)
        self.assertIn("utility_certified_exact_gain", results)
        self.assertIn("utility_certified_fast_gain", results)
        self.assertIn("learning_seconds", timings)
        self.assertIn("exact_matching_seconds", timings)
        self.assertIn("fast_matching_seconds", timings)
        self.assertTrue((results["probe_queries"] >= 0).all())
        self.assertTrue(
            (
                results["probe_queries"]
                == results["learning_probe_queries"]
                + results["certification_probe_queries"]
            ).all()
        )
        self.assertTrue(
            results["certified_exact_matched_pairs"].le(
                results["exact_matched_pairs"]
            ).all()
        )
        finest = results[results["buckets"] == 6]
        np.testing.assert_allclose(
            finest["utility_certified_exact_gain"], finest["utility_known_gain"]
        )

    def test_comparison_allows_lambda_one_and_caps_only_learning_levels(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_dataset_comparison(
            domain,
            lambda_bias=1.0,
            runs=1,
            intents_per_run=10,
            exact_timeout_seconds=None,
            max_learning_buckets=4,
            progress=False,
        )
        self.assertEqual(sorted(results["buckets"].unique().tolist()), [1, 2, 4])
        self.assertTrue((results["m"] == 8).all())
        self.assertTrue((results["z"] == 8).all())
        self.assertTrue((results["lambda"] == 1.0).all())
        self.assertIn("unavailable_bucket_pairs", results)
        self.assertEqual(sorted(timings["buckets"].tolist()), [1, 2, 4])

    def test_comparison_supports_gaussian_rank_prior(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_dataset_comparison(
            domain,
            lambda_bias=0.25,
            prior_model="gaussian",
            prior_variance=4.0,
            runs=1,
            intents_per_run=10,
            exact_timeout_seconds=None,
            progress=False,
        )
        self.assertTrue(results["prior_model"].eq("gaussian").all())
        self.assertTrue(results["prior_variance"].eq(4.0).all())
        self.assertTrue((results["prior_actual_variance"] < 4.0).all())
        self.assertTrue(np.isfinite(results["utility_certified_exact_gain"]).all())
        self.assertTrue(
            (
                results["probe_queries"]
                == results["learning_probe_queries"]
                + results["certification_probe_queries"]
            ).all()
        )
        finest = results[results["buckets"] == 8]
        np.testing.assert_allclose(
            finest["utility_certified_exact_gain"], finest["utility_known_gain"]
        )
        self.assertTrue(timings["prior_model"].eq("gaussian").all())
        uniform, _ = run_dataset_comparison(
            domain,
            lambda_bias=0.25,
            runs=1,
            intents_per_run=10,
            exact_timeout_seconds=None,
            progress=False,
        )
        self.assertFalse(
            np.allclose(
                results["utility_known_gain"], uniform["utility_known_gain"]
            )
        )

    def test_comparison_averages_random_gaussian_widths_across_runs(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_dataset_comparison(
            domain,
            lambda_bias=0.25,
            prior_model="gaussian-random",
            prior_seed=11,
            runs=2,
            intents_per_run=10,
            exact_timeout_seconds=None,
            progress=False,
        )
        self.assertTrue(results["prior_model"].eq("gaussian-random").all())
        self.assertEqual(results["prior_std_fraction"].nunique(), 2)
        expected_fractions = {
            shared_gaussian_std_fraction(run, 11) for run in range(2)
        }
        self.assertEqual(set(results["prior_std_fraction"]), expected_fractions)
        np.testing.assert_allclose(
            results["prior_variance"],
            (results["prior_std_fraction"] * domain.m) ** 2,
        )
        self.assertTrue(timings["prior_model"].eq("gaussian-random").all())

        summary = summarize_comparison(results)
        self.assertEqual(len(summary), results["buckets"].nunique())
        self.assertTrue(summary["prior_std_fraction_mean"].notna().all())
        self.assertTrue(summary["prior_std_fraction_std"].gt(0).all())

    def test_fixed_domain_mi_recovers_full_utility_at_finest_level(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_fixed_domain_mi(
            domain,
            lambda_bias=0.25,
            runs=2,
            exact_timeout_seconds=None,
            progress=False,
        )
        finest = results[results["buckets"] == 8]
        self.assertTrue(np.allclose(finest["utility_recovery"], 1.0))
        self.assertTrue((results["m"] == 8).all())
        self.assertTrue((results["z"] == 8).all())
        self.assertTrue((timings["status"] == "complete").all())
        self.assertTrue(np.isfinite(timings["matching_seconds"]).all())

    def test_fixed_domain_mi_supports_gaussian_rank_prior(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_fixed_domain_mi(
            domain,
            lambda_bias=0.25,
            runs=1,
            exact_timeout_seconds=None,
            prior_model="gaussian",
            prior_variance=4.0,
            progress=False,
        )
        finest = results[results["buckets"] == 8]
        self.assertTrue(np.allclose(finest["utility_recovery"], 1.0))
        self.assertTrue((results["prior_model"] == "gaussian").all())
        self.assertTrue((results["prior_variance"] == 4.0).all())
        self.assertTrue((results["prior_actual_variance"] < 4.0).all())
        self.assertTrue((timings["status"] == "complete").all())

    def test_fixed_domain_mi_supports_random_gaussian_rank_prior(self):
        domain = FixedDomain(
            dataset="toy",
            dataset_label="Toy",
            attribute="value",
            attribute_label="Value",
            categorical=False,
            full_cardinality=8,
            values=tuple(range(8)),
            coordinates=np.arange(8, dtype=float),
        )
        results, timings = run_fixed_domain_mi(
            domain,
            lambda_bias=0.25,
            runs=2,
            exact_timeout_seconds=None,
            prior_model="gaussian-random",
            prior_seed=0,
            benchmark_general=False,
            progress=False,
        )
        finest = results[results["buckets"] == 8]
        self.assertTrue(np.allclose(finest["utility_recovery"], 1.0))
        self.assertTrue(results["prior_model"].eq("gaussian-random").all())
        self.assertEqual(results["prior_std_fraction"].nunique(), 2)
        expected = (results["prior_std_fraction"] * 8) ** 2
        np.testing.assert_allclose(results["prior_variance"], expected)
        self.assertTrue(np.isfinite(timings["matching_seconds"]).all())
        summary = summarize_mi_bucketization(results, timings)
        for _, row in summary.iterrows():
            selected = results[
                (results["lambda"] == row["lambda"])
                & (results["buckets"] == row["buckets"])
            ]
            expected_recovery = (
                selected["total_quadratic_gain"].mean()
                / selected["full_quadratic_gain"].mean()
            )
            self.assertAlmostEqual(
                row["utility_recovery_mean"], expected_recovery
            )
        self.assertIn("utility_recovery_run_ratio_mean", summary.columns)


if __name__ == "__main__":
    unittest.main()
