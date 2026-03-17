#!/usr/bin/env python3
"""
Tests for evaluation metrics.
"""

import sys
import os
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.metrics import (
    state_fidelity,
    total_variation_distance,
    success_probability,
    energy_error,
    approximation_ratio,
    counts_to_probabilities,
)


class TestStateFidelity(unittest.TestCase):

    def test_identical_states(self):
        sv = np.array([1, 0], dtype=complex)
        self.assertAlmostEqual(state_fidelity(sv, sv), 1.0)

    def test_orthogonal_states(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([0, 1], dtype=complex)
        self.assertAlmostEqual(state_fidelity(sv1, sv2), 0.0)

    def test_superposition(self):
        sv1 = np.array([1, 0], dtype=complex)
        sv2 = np.array([1, 1], dtype=complex) / math.sqrt(2)
        self.assertAlmostEqual(state_fidelity(sv1, sv2), 0.5)

    def test_bell_state(self):
        bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        self.assertAlmostEqual(state_fidelity(bell, bell), 1.0)


class TestTVD(unittest.TestCase):

    def test_identical_distributions(self):
        p = {"00": 0.5, "11": 0.5}
        self.assertAlmostEqual(total_variation_distance(p, p), 0.0)

    def test_disjoint_distributions(self):
        p = {"00": 1.0}
        q = {"11": 1.0}
        self.assertAlmostEqual(total_variation_distance(p, q), 1.0)

    def test_arrays(self):
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        self.assertAlmostEqual(total_variation_distance(p, q), 0.5)


class TestSuccessProbability(unittest.TestCase):

    def test_all_success(self):
        counts = {"111": 100}
        self.assertAlmostEqual(success_probability(counts, ["111"]), 1.0)

    def test_no_success(self):
        counts = {"000": 100}
        self.assertAlmostEqual(success_probability(counts, ["111"]), 0.0)

    def test_partial(self):
        counts = {"111": 30, "000": 70}
        self.assertAlmostEqual(success_probability(counts, ["111"]), 0.3)


class TestEnergyError(unittest.TestCase):

    def test_exact(self):
        self.assertAlmostEqual(energy_error(-1.5, -1.5), 0.0)

    def test_error(self):
        self.assertAlmostEqual(energy_error(-1.4, -1.5), 0.1)


class TestApproximationRatio(unittest.TestCase):

    def test_optimal(self):
        self.assertAlmostEqual(approximation_ratio(4, 4), 1.0)

    def test_half(self):
        self.assertAlmostEqual(approximation_ratio(2, 4), 0.5)

    def test_zero_optimal(self):
        self.assertAlmostEqual(approximation_ratio(0, 0), 1.0)


class TestCountsToProbs(unittest.TestCase):

    def test_basic(self):
        counts = {"00": 50, "11": 50}
        probs = counts_to_probabilities(counts)
        self.assertAlmostEqual(probs["00"], 0.5)
        self.assertAlmostEqual(probs["11"], 0.5)


if __name__ == "__main__":
    unittest.main()
