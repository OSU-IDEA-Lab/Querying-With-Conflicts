import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cartesian_attributes.distributional import (
    bias_attributes,
    empirical_group_statistics,
    evaluate_constraints_on_distribution,
    influential_matching,
    information_gain_on_distribution,
    posterior_moments,
    prior_pmf,
    ranking_utility_for_intents,
    run_distributional_experiment,
    sample_iid_intents,
    sampled_information_gain,
)
from experiments.cartesian_attributes.sample_recovery import (
    nested_sample_indexes,
    run_sample_recovery_experiment,
)
from experiments.cartesian_attributes.experiment import build_prefix_domains
from experiments.cartesian_attributes.scheme import DatasetScheme


class DistributionalCartesianTests(unittest.TestCase):
    def setUp(self):
        frame = pd.DataFrame(
            {
                "a": np.tile([0, 1], 12),
                "b": np.repeat(["x", "y", "z"], 8),
                "c": np.tile([0, 0, 1, 1], 6),
                "race": np.tile(["r1", "r2", "r1", "r3"], 6),
                "sex": np.tile(["f", "m"], 12),
                "c_charge_degree": np.tile(["F", "M", "M"], 8),
            }
        )
        self.scheme = DatasetScheme(
            key="compas",
            label="Toy COMPAS",
            source=Path("toy.csv"),
            frame=frame,
            attributes=("a", "b", "c"),
            attribute_labels=("A", "B", "C"),
            categorical_filters=("race",),
            numeric_filters=(),
            notes="test",
        )

    def test_arbitrary_prior_moments_retain_asymmetric_shifts(self):
        uniform = posterior_moments(prior_pmf(7, "uniform"))
        np.testing.assert_allclose(uniform.alpha, uniform.alpha_prime, rtol=1e-12)

        zipfian = posterior_moments(
            prior_pmf(7, "zipfian", zipf_exponent=1.2)
        )
        self.assertTrue((zipfian.alpha > 0).all())
        self.assertTrue((zipfian.alpha_prime > 0).all())
        self.assertFalse(np.allclose(zipfian.alpha, zipfian.alpha_prime))

    def test_empirical_statistics_use_sample_group_mass_and_conditional_bias(self):
        prefix = build_prefix_domains(self.scheme)[0]
        indexes = np.arange(len(self.scheme.frame))
        encoded = np.where(self.scheme.frame["race"].to_numpy() == "r1", 0, 1)
        unit_bias = np.array([0.0, 1.0])
        _, probability, conditional = empirical_group_statistics(
            prefix, indexes, encoded, unit_bias
        )
        np.testing.assert_allclose(probability, [0.5, 0.5])
        # In this fixture, group a=0 contains only r1 rows and group a=1
        # contains only non-r1 rows, so the conditional expectation differs
        # even though both groups have equal mass.
        np.testing.assert_allclose(conditional, [0.0, 1.0])

    def test_matching_gain_agrees_with_sampled_iid_intents(self):
        probability = np.array([0.2, 0.3, 0.1, 0.4])
        biases = np.array([-0.1, -0.4, -1.0, -1.4])
        moments = posterior_moments(prior_pmf(4, "zipfian"))
        matching = influential_matching(
            probability, biases, moments, batch_size=2
        )
        exact = sum(item.gain for item in matching)
        sampled = sampled_information_gain(
            matching,
            probability,
            moments,
            intents=100_000,
            rng=np.random.default_rng(7),
        )
        self.assertAlmostEqual(sampled, exact, delta=max(0.01, 0.03 * exact))

        evaluated, credible = evaluate_constraints_on_distribution(
            matching, probability, biases, moments
        )
        self.assertAlmostEqual(evaluated, exact)
        self.assertEqual(credible, len(matching))
        self.assertAlmostEqual(
            information_gain_on_distribution(matching, probability, moments),
            exact,
        )

    def test_nested_samples_and_full_domain_recovery(self):
        samples = nested_sample_indexes(
            len(self.scheme.frame),
            [0.2, 0.5, 1.0],
            dataset=self.scheme.key,
            run=0,
            seed=0,
        )
        self.assertTrue(set(samples[0.2]).issubset(set(samples[0.5])))
        self.assertTrue(set(samples[0.5]).issubset(set(samples[1.0])))

        results = run_sample_recovery_experiment(
            self.scheme,
            fractions=[0.5, 1.0],
            priors=["uniform"],
            runs=[0],
            lambdas=[0.25],
            batch_size=8,
            progress=False,
        )
        self.assertEqual(len(results), 6)
        full = results[results["sample_fraction"] == 1.0]
        np.testing.assert_allclose(full["utility_recovery"], 1.0)
        self.assertTrue(
            (results["sample_observed_groups"] <= results["full_observed_groups"]).all()
        )

    def test_ranking_utility_uses_the_same_iid_intent_vectors(self):
        probability = np.array([0.2, 0.3, 0.1, 0.4])
        biases = np.array([0.1, 0.4, 1.0, 1.4])
        moments = posterior_moments(prior_pmf(4, "zipfian"))
        matching = influential_matching(
            probability, biases, moments, batch_size=2
        )
        intents = sample_iid_intents(
            moments, intents=200, rng=np.random.default_rng(11)
        )
        query, babbling, gain = ranking_utility_for_intents(
            matching, biases, moments, intents
        )
        self.assertTrue(np.isfinite([query, babbling, gain]).all())
        self.assertAlmostEqual(gain, query - babbling)
        self.assertGreaterEqual(query, -1.0)
        self.assertLessEqual(query, 1.0)

    def test_small_end_to_end_run_covers_three_priors_and_prefixes(self):
        results = run_distributional_experiment(
            self.scheme,
            runs=[0],
            lambdas=[0.25],
            sample_fraction=1.0,
            intents=100,
            batch_size=8,
            progress=False,
        )
        self.assertEqual(len(results), 9)
        self.assertEqual(set(results["prior_model"]), {"uniform", "zipfian", "gaussian"})
        self.assertEqual(set(results["num_intent_attributes"]), {1, 2, 3})
        self.assertTrue(
            (results["bias_attribute"] == "race x sex x c_charge_degree").all()
        )
        self.assertTrue((results["bias_attribute_count"] == 3).all())
        self.assertTrue((results["sample_rows"] == len(self.scheme.frame)).all())
        self.assertTrue((results["expected_utility_gain"] >= 0).all())
        self.assertTrue(results["ranking_utility"].between(-1.0, 1.0).all())
        self.assertTrue(
            results["babbling_ranking_utility"].between(-1.0, 1.0).all()
        )

    def test_three_distinct_bias_attributes_and_random_gaussian_width(self):
        self.assertEqual(
            [attribute.key for attribute in bias_attributes(self.scheme)],
            ["race", "sex", "c_charge_degree"],
        )
        results = run_distributional_experiment(
            self.scheme,
            priors=["uniform", "gaussian"],
            runs=[0],
            lambdas=[0.25],
            gaussian_std_mode="random",
            bias_attribute_count=3,
            intents=10,
            batch_size=8,
            progress=False,
        )
        self.assertEqual(len(results), 6)
        self.assertEqual(results["bias_attribute"].nunique(), 1)
        self.assertTrue((results["bias_attribute_count"] == 3).all())
        self.assertEqual(results["gaussian_std_fraction"].nunique(), 1)
        self.assertTrue((results["gaussian_std_mode"] == "random").all())


if __name__ == "__main__":
    unittest.main()
