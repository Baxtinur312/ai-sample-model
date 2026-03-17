#!/usr/bin/env python3
"""
Tests for the Circuit IR module.
"""

import sys
import os
import unittest
import hashlib
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import (
    QuantumCircuit, QuantumGate, GATE_MATRICES, CLIFFORD_GATES,
    _embed_gate,
)


class TestQuantumGate(unittest.TestCase):
    """Tests for QuantumGate."""

    def test_single_qubit_gate_matrix(self):
        """H gate matrix should be correct."""
        g = QuantumGate("H", (0,))
        H = g.matrix
        # H² = I
        np.testing.assert_allclose(H @ H, np.eye(2), atol=1e-12)

    def test_parametric_gate(self):
        """RY(π) should be equivalent to Y (up to phase)."""
        g = QuantumGate("RY", (0,), (math.pi,))
        mat = g.matrix
        # RY(π) = [[0, -1], [1, 0]] = -iY
        expected = np.array([[0, -1], [1, 0]], dtype=complex)
        np.testing.assert_allclose(mat, expected, atol=1e-12)

    def test_cnot_matrix(self):
        """CNOT should have the correct 4x4 matrix."""
        g = QuantumGate("CNOT", (0, 1))
        mat = g.matrix
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
        np.testing.assert_allclose(mat, expected, atol=1e-12)

    def test_is_clifford(self):
        assert QuantumGate("H", (0,)).is_clifford
        assert QuantumGate("CNOT", (0, 1)).is_clifford
        assert not QuantumGate("T", (0,)).is_clifford
        assert not QuantumGate("RY", (0,), (0.5,)).is_clifford


class TestQuantumCircuit(unittest.TestCase):
    """Tests for QuantumCircuit."""

    def test_basic_construction(self):
        qc = QuantumCircuit(3)
        qc.h(0).cnot(0, 1).x(2)
        self.assertEqual(qc.num_qubits, 3)
        self.assertEqual(len(qc.gates), 3)

    def test_invalid_qubit(self):
        qc = QuantumCircuit(2)
        with self.assertRaises(IndexError):
            qc.h(5)

    def test_circuit_hash_deterministic(self):
        qc1 = QuantumCircuit(2)
        qc1.h(0).cnot(0, 1)
        qc2 = QuantumCircuit(2)
        qc2.h(0).cnot(0, 1)
        self.assertEqual(qc1.circuit_hash(), qc2.circuit_hash())

    def test_circuit_hash_differs(self):
        qc1 = QuantumCircuit(2)
        qc1.h(0).cnot(0, 1)
        qc2 = QuantumCircuit(2)
        qc2.h(1).cnot(0, 1)
        self.assertNotEqual(qc1.circuit_hash(), qc2.circuit_hash())

    def test_is_clifford_all_clifford(self):
        qc = QuantumCircuit(2)
        qc.h(0).s(0).cnot(0, 1)
        self.assertTrue(qc.is_clifford)

    def test_is_clifford_with_t(self):
        qc = QuantumCircuit(2)
        qc.h(0).t(0).cnot(0, 1)
        self.assertFalse(qc.is_clifford)

    def test_depth(self):
        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2)  # All parallel → depth 1
        self.assertEqual(qc.depth, 1)
        qc.cnot(0, 1)       # depth 2
        self.assertEqual(qc.depth, 2)

    def test_to_unitary_identity(self):
        """Empty circuit should give identity."""
        qc = QuantumCircuit(2)
        U = qc.to_unitary()
        np.testing.assert_allclose(U, np.eye(4), atol=1e-12)

    def test_to_unitary_x_gate(self):
        """X gate should flip |0⟩ to |1⟩."""
        qc = QuantumCircuit(1)
        qc.x(0)
        U = qc.to_unitary()
        state_0 = np.array([1, 0])
        result = U @ state_0
        np.testing.assert_allclose(result, [0, 1], atol=1e-12)

    def test_draw(self):
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        drawing = qc.draw()
        self.assertIn("H", drawing)
        self.assertIn("CNOT", drawing)


if __name__ == "__main__":
    unittest.main()
