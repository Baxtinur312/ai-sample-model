#!/usr/bin/env python3
"""
Tests for the Stabilizer backend.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.stabilizer import StabilizerBackend


class TestStabilizerBackend(unittest.TestCase):

    def setUp(self):
        self.backend = StabilizerBackend()

    def test_initial_state(self):
        """Empty circuit should give all |0⟩."""
        qc = QuantumCircuit(2)
        result = self.backend.run(qc, shots=100, seed=42)
        self.assertEqual(result.counts.get("00", 0), 100)

    def test_x_flip(self):
        """X gate should flip qubit."""
        qc = QuantumCircuit(1)
        qc.x(0)
        result = self.backend.run(qc, shots=100, seed=42)
        self.assertEqual(result.counts.get("1", 0), 100)

    def test_bell_state(self):
        """H-CNOT on stabilizer should give ~50/50 entangled."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        result = self.backend.run(qc, shots=10000, seed=42)
        # Should only get 00 and 11
        self.assertEqual(result.counts.get("01", 0), 0)
        self.assertEqual(result.counts.get("10", 0), 0)
        total = sum(result.counts.values())
        self.assertEqual(total, 10000)

    def test_non_clifford_rejected(self):
        """T gate should be rejected."""
        qc = QuantumCircuit(1)
        qc.t(0)
        with self.assertRaises(RuntimeError):
            self.backend.run(qc)

    def test_ry_rejected(self):
        """RY gate should be rejected."""
        qc = QuantumCircuit(1)
        qc.ry(0, 0.5)
        with self.assertRaises(RuntimeError):
            self.backend.run(qc)

    def test_larger_clifford(self):
        """Should handle 10+ qubits efficiently."""
        qc = QuantumCircuit(20)
        for i in range(20):
            qc.h(i)
        for i in range(19):
            qc.cnot(i, i + 1)
        result = self.backend.run(qc, shots=100, seed=42)
        self.assertEqual(sum(result.counts.values()), 100)

    def test_no_statevector(self):
        """Stabilizer does not provide statevector."""
        qc = QuantumCircuit(1)
        qc.h(0)
        result = self.backend.run(qc, shots=10, seed=42)
        self.assertIsNone(result.statevector)


if __name__ == "__main__":
    unittest.main()
