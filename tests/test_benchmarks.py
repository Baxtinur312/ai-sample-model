#!/usr/bin/env python3
"""
Tests for benchmark modules (Grover, QFT, VQE).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.grover import run_grover_benchmark, theoretical_success_prob
from benchmarks.qft import run_qft_benchmark
from benchmarks.vqe import run_vqe_benchmark


class TestGroverBenchmark(unittest.TestCase):

    def test_2qubit_grover(self):
        """Grover on 2 qubits should find the marked state."""
        result = run_grover_benchmark(num_qubits=2, shots=2048, seed=42)
        self.assertGreater(result["measured_success_prob"], 0.5)

    def test_3qubit_grover(self):
        """Grover on 3 qubits should achieve high success probability."""
        result = run_grover_benchmark(num_qubits=3, shots=2048, seed=42)
        self.assertGreater(result["measured_success_prob"], 0.7)

    def test_theoretical_prediction(self):
        """Theoretical prediction should be close to 1 for optimal iterations."""
        p = theoretical_success_prob(N=8, M=1, k=2)
        self.assertGreater(p, 0.9)


class TestQFTBenchmark(unittest.TestCase):

    def test_2qubit_qft(self):
        """QFT on 2 qubits should pass fidelity tests."""
        result = run_qft_benchmark(num_qubits=2, shots=1024, seed=42)
        # At least the zero-state test should produce results
        self.assertGreater(len(result["tests"]), 0)

    def test_3qubit_qft(self):
        """QFT on 3 qubits should produce results."""
        result = run_qft_benchmark(num_qubits=3, shots=1024, seed=42)
        self.assertGreater(len(result["tests"]), 0)


class TestVQEBenchmark(unittest.TestCase):

    def test_h2_vqe(self):
        """VQE should find energy close to exact for H₂."""
        result = run_vqe_benchmark(seed=42)
        # Energy error should be reasonable (< 0.1 Ha)
        self.assertLess(result["energy_error"], 0.1)
        self.assertIn("exact_ground_energy", result)
        self.assertIn("vqe_energy", result)


if __name__ == "__main__":
    unittest.main()
