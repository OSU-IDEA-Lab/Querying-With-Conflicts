import unittest

import numpy as np

from experiments.cartesian_attributes.distributional import posterior_moments, prior_pmf
from experiments.trustworthy_answers.experiment import (
    DetectionGrid,
    _joint_bias_encoding,
    general_range_filter,
    special_threshold_filter,
)
from experiments.cartesian_attributes.scheme import load_dataset_scheme


class TrustworthyAnswerTests(unittest.TestCase):
    def test_joint_bias_encoding_uses_all_three_distinct_attributes(self):
        scheme = load_dataset_scheme("compas", "data/real")
        key, label, encoded, joint_values = _joint_bias_encoding(scheme)
        self.assertEqual(key, "race x sex x c_charge_degree")
        self.assertIn("race", label)
        self.assertEqual(len(encoded), len(scheme.frame))
        self.assertEqual(len(np.unique(encoded)), joint_values)

    def test_uniform_grid_matches_general_posterior_formulas(self):
        z = 17
        moments = posterior_moments(prior_pmf(z, "uniform"))
        grid = DetectionGrid.uniform(z)
        rho = moments.alpha / moments.alpha_prime
        delta = moments.thresholds
        expected_plus = (
            0.5 * rho * (moments.mu0 + moments.mu1)
            - 0.5 * (moments.mu_prime0 + moments.mu_prime1)
            + np.minimum(delta + 1.0 - rho, z - rho * (z - delta))
        )
        expected_minus = expected_plus - 1.0 + np.maximum(
            (1.0 - rho) * (z - delta),
            (rho - 1.0) * (z - delta - 1.0),
        )
        np.testing.assert_allclose(grid.rho, rho)
        np.testing.assert_allclose(grid.s, moments.mu0 - moments.mu_prime0)
        np.testing.assert_allclose(grid.g_plus, expected_plus)
        np.testing.assert_allclose(grid.g_minus, expected_minus)

    def test_fast_gaussian_grid_matches_quadratic_reference(self):
        z = 31
        std_fraction = 0.27
        moments = posterior_moments(
            prior_pmf(z, "gaussian", gaussian_std_fraction=std_fraction)
        )
        grid = DetectionGrid.centered_gaussian(z, std_fraction)
        np.testing.assert_allclose(grid.prior, moments.prior, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            grid.s, moments.mu0 - moments.mu_prime0, rtol=1e-10, atol=1e-11
        )
        rho = moments.alpha / moments.alpha_prime
        delta = moments.thresholds
        expected_plus = (
            0.5 * rho * (moments.mu0 + moments.mu1)
            - 0.5 * (moments.mu_prime0 + moments.mu_prime1)
            + np.minimum(delta + 1.0 - rho, z - rho * (z - delta))
        )
        np.testing.assert_allclose(grid.g_plus, expected_plus, rtol=1e-10, atol=1e-11)

    def test_general_batched_filter_matches_scalar_test(self):
        grid = DetectionGrid.uniform(23)
        tuple_bias = np.array([-2.0, -9.0, -15.0])
        expected = np.array([-4.0, -8.0, -12.0])
        actual, checks = general_range_filter(
            tuple_bias, expected, -23.0, 0.0, grid, threshold_batch=4
        )
        reference = []
        for bt, be in zip(tuple_bias, expected):
            unsafe = False
            for rho, s, gp, gm in zip(
                grid.rho, grid.s, grid.g_plus, grid.g_minus
            ):
                lower = rho * be - gp
                upper = rho * be - gm
                first = max(-23.0, lower)
                if first <= 0.0 and first < upper and -23.0 < bt - s:
                    unsafe = True
                    break
            reference.append(not unsafe)
        np.testing.assert_array_equal(actual, reference)
        self.assertGreater(checks, 0)

    def test_special_filter_matches_exhaustive_delta_test(self):
        grid = DetectionGrid.centered_gaussian(29, 0.4)
        biases = np.linspace(-29.0, 0.0, 25)
        actual, checks = special_threshold_filter(biases, -29.0, 0.0, grid)
        reference = []
        candidate = np.linspace(-29.0, 0.0, 20_001)
        for bias in biases:
            unsafe = False
            differences = bias - candidate
            for s, gp, gm in zip(grid.s, grid.g_plus, grid.g_minus):
                if np.any((differences > gm) & (differences <= gp) & (differences > s)):
                    unsafe = True
                    break
            reference.append(not unsafe)
        np.testing.assert_array_equal(actual, reference)
        self.assertEqual(checks, len(biases))


if __name__ == "__main__":
    unittest.main()
