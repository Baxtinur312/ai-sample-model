#!/usr/bin/env python3
"""
Tests for the Statevector backend.
"""

import sys
import os
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend


class TestStatevectorBackend(unittest.TestCase):

    def setUp(self):
        self.backend = StatevectorBackend()

    def test_initial_state(self):
        """Empty circuit should give |0⟩."""
        qc = QuantumCircuit(1)
        result = self.backend.run(qc, shots=100, seed=42)
        np.testing.assert_allclose(result.statevector, [1, 0], atol=1e-12)
        self.assertEqual(result.counts.get("0", 0), 100)

    def test_hadamard_superposition(self):
        """H|0⟩ should give equal superposition."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = self.backend.run(qc, shots=10000, seed=42)
        sv = result.statevector
        np.testing.assert_allclose(np.abs(sv), [1/math.sqrt(2)] * 2, atol=1e-12)
        # Counts should be ~50/50
        p0 = result.counts.get("0", 0) / 10000
        self.assertAlmostEqual(p0, 0.5, delta=0.05)

    def test_bell_state(self):
        """H-CNOT should create Bell state."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        result = self.backend.run(qc, shots=10000, seed=42)
        # Only |00⟩ and |11⟩ should appear
        self.assertEqual(result.counts.get("01", 0), 0)
        self.assertEqual(result.counts.get("10", 0), 0)
        self.assertGreater(result.counts.get("00", 0), 4000)
        self.assertGreater(result.counts.get("11", 0), 4000)

    def test_x_gate_flip(self):
        """X|0⟩ = |1⟩."""
        qc = QuantumCircuit(1)
        qc.x(0)
        result = self.backend.run(qc, shots=100, seed=42)
        np.testing.assert_allclose(result.statevector, [0, 1], atol=1e-12)
        self.assertEqual(result.counts.get("1", 0), 100)

    def test_two_x_identity(self):
        """XX = I."""
        qc = QuantumCircuit(1)
        qc.x(0).x(0)
        result = self.backend.run(qc, shots=100, seed=42)
        np.testing.assert_allclose(result.statevector, [1, 0], atol=1e-12)

    def test_ry_rotation(self):
        """RY(π/2) on |0⟩ should give equal superposition."""
        qc = QuantumCircuit(1)
        qc.ry(0, math.pi / 2)
        result = self.backend.run(qc, shots=10000, seed=42)
        probs = np.abs(result.statevector) ** 2
        np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-12)

    def test_seed_reproducibility(self):
        """Same seed should give same counts."""
        qc = QuantumCircuit(1)
        qc.h(0)
        r1 = self.backend.run(qc, shots=1000, seed=123)
        r2 = self.backend.run(qc, shots=1000, seed=123)
        self.assertEqual(r1.counts, r2.counts)

    def test_multi_qubit(self):
        """GHZ state on 3 qubits."""
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).cnot(1, 2)
        result = self.backend.run(qc, shots=10000, seed=42)
        # Only |000⟩ and |111⟩
        for bs in ["001", "010", "011", "100", "101", "110"]:
            self.assertEqual(result.counts.get(bs, 0), 0)
        self.assertGreater(result.counts.get("000", 0), 4000)
        self.assertGreater(result.counts.get("111", 0), 4000)

    def test_validation_too_many_qubits(self):
        """Should raise for circuits exceeding max qubits."""
        backend = StatevectorBackend(max_qubits=5)
        qc = QuantumCircuit(10)
        with self.assertRaises(RuntimeError):
            backend.run(qc)


if __name__ == "__main__":
    unittest.main()
