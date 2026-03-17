#!/usr/bin/env python3
"""
Tests for the MPS backend.
"""

import sys
import os
import unittest
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from quantum_kernel.backends.mps import MPSBackend
from benchmarks.metrics import state_fidelity


class TestMPSBackend(unittest.TestCase):

    def setUp(self):
        self.sv_backend = StatevectorBackend()
        self.mps_backend = MPSBackend(max_bond_dim=64)

    def test_initial_state(self):
        """Empty circuit should give |00⟩."""
        qc = QuantumCircuit(2)
        result = self.mps_backend.run(qc, shots=100, seed=42)
        self.assertEqual(result.counts.get("00", 0), 100)

    def test_single_qubit_h(self):
        """H gate should match statevector."""
        qc = QuantumCircuit(1)
        qc.h(0)
        sv_result = self.sv_backend.run(qc, shots=1, seed=42)
        mps_result = self.mps_backend.run(qc, shots=1, seed=42)
        fid = state_fidelity(sv_result.statevector, mps_result.statevector)
        self.assertGreater(fid, 0.999)

    def test_bell_state_mps_vs_sv(self):
        """MPS Bell state should match statevector."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        sv_result = self.sv_backend.run(qc, shots=1, seed=42)
        mps_result = self.mps_backend.run(qc, shots=1, seed=42)
        fid = state_fidelity(sv_result.statevector, mps_result.statevector)
        self.assertGreater(fid, 0.99)

    def test_multi_qubit_agreement(self):
        """MPS should agree with statevector for small circuits."""
        qc = QuantumCircuit(4)
        qc.h(0).cnot(0, 1).h(2).cnot(2, 3)
        sv_result = self.sv_backend.run(qc, shots=1, seed=42)
        mps_result = self.mps_backend.run(qc, shots=1, seed=42)
        fid = state_fidelity(sv_result.statevector, mps_result.statevector)
        self.assertGreater(fid, 0.99)

    def test_truncation_error_tracking(self):
        """Truncation error should be tracked."""
        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
        result = self.mps_backend.run(qc, shots=10, seed=42)
        # For small circuits, truncation should be minimal
        self.assertIn("cumulative_truncation_error", result.metadata)

    def test_rejects_3qubit_gates(self):
        """MPS should reject Toffoli gates."""
        qc = QuantumCircuit(3)
        qc.toffoli(0, 1, 2)
        with self.assertRaises(RuntimeError):
            self.mps_backend.run(qc)


if __name__ == "__main__":
    unittest.main()
