"""
Dense Statevector Simulator
Exact simulation of quantum circuits via full statevector evolution.
Memory grows as 2^n complex amplitudes (16 bytes each for complex128).
Dense statevector simulation backend.

Simulates arbitrary quantum circuits exactly using a 2^n complex vector.
Supports all gates defined in circuit_ir and returns measurement counts
sampled from the probability distribution.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit, QuantumGate
from .base import Backend, SimulationResult


class StatevectorBackend(Backend):
    """Full statevector (Schrödinger) simulation."""

    name = "statevector"
    method = "dense_statevector"

    def __init__(self, max_qubits: int = 20):
        self.max_qubits = max_qubits

    def validate_circuit(self, circuit: QuantumCircuit) -> List[str]:
        issues = []
        if circuit.num_qubits > self.max_qubits:
            issues.append(
                f"Circuit has {circuit.num_qubits} qubits, exceeding the "
                f"statevector limit of {self.max_qubits}. Memory required: "
                f"~{2**circuit.num_qubits * 16 / 1e9:.1f} GB"
            )
        return issues

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        issues = self.validate_circuit(circuit)
        if issues:
            raise RuntimeError(
                "Statevector validation failed:\n" + "\n".join(issues)
            )

        n = circuit.num_qubits
        dim = 2 ** n

        sv = np.zeros(dim, dtype=complex)
        sv[0] = 1.0

        for gate in circuit.gates:
            sv = self._apply_gate(sv, gate, n)

        probs = np.abs(sv) ** 2
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        rng = np.random.default_rng(seed)
        samples = rng.choice(dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for s in samples:
            bitstring = format(s, f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return SimulationResult(
            counts=counts,
            statevector=sv.copy(),
            probabilities=probs,
            metadata={
                "backend": self.name,
                "method": self.method,
                "num_qubits": n,
                "num_gates": len(circuit.gates),
                "circuit_depth": circuit.depth,
            },
            execution_label="ideal_simulation",
        )

    def _apply_gate(
        self, sv: np.ndarray, gate: QuantumGate, n: int
    ) -> np.ndarray:
        """Apply a gate to the statevector using tensor reshaping."""
        mat = gate.matrix
        nq = gate.num_qubits

        if nq == 1:
            return self._apply_single_qubit_gate(sv, mat, gate.qubits[0], n)
        elif nq == 2:
            return self._apply_two_qubit_gate(
                sv, mat, gate.qubits[0], gate.qubits[1], n
            )
        elif nq == 3:
            return self._apply_three_qubit_gate(
                sv, mat, gate.qubits[0], gate.qubits[1], gate.qubits[2], n
            )
        else:
            raise NotImplementedError(
                f"{nq}-qubit gate application not implemented"
            )

    @staticmethod
    def _apply_single_qubit_gate(
        sv: np.ndarray, mat: np.ndarray, qubit: int, n: int
    ) -> np.ndarray:
        """Efficient single-qubit gate via tensor reshape."""
        sv = sv.reshape([2] * n)
        sv = np.moveaxis(sv, qubit, 0)
        sv = np.tensordot(mat, sv, axes=([1], [0]))
        sv = np.moveaxis(sv, 0, qubit)
        return sv.reshape(-1)

    @staticmethod
    def _apply_two_qubit_gate(
        sv: np.ndarray, mat: np.ndarray, q0: int, q1: int, n: int
    ) -> np.ndarray:
        """Two-qubit gate via tensor operations."""
        sv = sv.reshape([2] * n)
        perm = list(range(n))
        perm.remove(q0)
        perm.remove(q1)
        perm = [q0, q1] + perm
        sv = sv.transpose(perm)
        sv_shape = sv.shape
        sv = sv.reshape(4, -1)
        sv = mat.reshape(4, 4) @ sv
        sv = sv.reshape(sv_shape)
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        sv = sv.transpose(inv_perm)
        return sv.reshape(-1)

    @staticmethod
    def _apply_three_qubit_gate(
        sv: np.ndarray, mat: np.ndarray, q0: int, q1: int, q2: int, n: int
    ) -> np.ndarray:
        """Three-qubit gate via tensor operations."""
        sv = sv.reshape([2] * n)
        perm = list(range(n))
        perm.remove(q0)
        perm.remove(q1)
        perm.remove(q2)
        perm = [q0, q1, q2] + perm
        sv = sv.transpose(perm)
        sv_shape = sv.shape
        sv = sv.reshape(8, -1)
        sv = mat.reshape(8, 8) @ sv
        sv = sv.reshape(sv_shape)
        inv_perm = [0] * n
        for i, p in enumerate(perm):
            inv_perm[p] = i
        sv = sv.transpose(inv_perm)
        return sv.reshape(-1)
