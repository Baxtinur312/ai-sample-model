"""
Matrix Product State (MPS) Simulator
Approximate simulation of quantum circuits using MPS tensor-network
representation with SVD-based truncation. Effective for circuits with
limited entanglement or near-1D topologies.

Bond dimension controls the accuracy/speed trade-off.
Matrix Product State (MPS) simulation backend.

Implements a lightweight bond-dimension-truncated MPS simulator for
1-2 qubit gates. Suitable for weakly-entangled circuits on up to ~100 qubits
when the entanglement entropy is bounded.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import svd as scipy_svd

from quantum_kernel.circuit_ir import QuantumCircuit, QuantumGate
from .base import Backend, SimulationResult


class MPSState:
    """
    Matrix Product State representation of an n-qubit state.

    Each qubit is represented by a 3-tensor: M[i] with shape
    (bond_left, physical=2, bond_right).

    Initial state |00...0⟩ has bond dimension 1 everywhere.
    """

    def __init__(self, n: int, max_bond_dim: int = 64,
                 truncation_threshold: float = 1e-10):
        self.n = n
        self.max_bond_dim = max_bond_dim
        self.truncation_threshold = truncation_threshold
        self.cumulative_truncation_error = 0.0

        # Initialize: each tensor is shape (1, 2, 1)
        self.tensors: List[np.ndarray] = []
        for i in range(n):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0  # |0⟩ state
            self.tensors.append(t)

    def apply_single_qubit_gate(self, mat: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate (2x2 matrix) at the given qubit."""
        # M[qubit] has shape (bl, 2, br)
        t = self.tensors[qubit]
        # Contract along physical index: new[bl, p', br] = sum_p mat[p', p] * t[bl, p, br]
        self.tensors[qubit] = np.einsum("ij,kjl->kil", mat, t)

    def apply_two_qubit_gate(self, mat: np.ndarray, q0: int, q1: int) -> None:
        """
        Apply a two-qubit gate. The qubits must be adjacent (|q0 - q1| == 1).
        For non-adjacent qubits, SWAP gates should be used first.

        The gate matrix is 4x4, reshaped to (2, 2, 2, 2).
        """
        if abs(q0 - q1) != 1:
            # For non-adjacent: swap chain
            self._apply_non_adjacent_two_qubit(mat, q0, q1)
            return

        # Ensure q0 < q1
        if q0 > q1:
            # Swap qubit labels and adjust gate matrix
            mat = self._swap_gate_qubits(mat)
            q0, q1 = q1, q0

        left = self.tensors[q0]   # (bl, 2, bm)
        right = self.tensors[q1]  # (bm, 2, br)

        bl = left.shape[0]
        bm = left.shape[2]
        br = right.shape[2]

        # left: (bl, 2, bm), right: (bm, 2, br)
        # Contract to theta: (bl, 2, 2, br)
        theta = np.einsum("ijk,klm->ijlm", left, right)
        
        # Apply gate: mat is (4, 4) acting on the two physical indices
        # We need to reshape theta to contract with mat
        # theta currently has axes: (bl, p0, p1, br)
        # Reshape to Contract: (bl, 4, br)
        theta = theta.reshape(bl, 4, br)
        # mat is (4, 4), mat @ theta along axis 1 -> (bl, 4, br)
        theta = np.einsum("ab,ibc->iac", mat, theta)
        
        # Reshape for SVD: (bl*2, 2*br)
        theta = theta.reshape(bl * 2, 2 * br)

        # SVD and truncate
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)

        # Truncate based on bond dimension and threshold
        k = len(S)
        if k > self.max_bond_dim:
            trunc_error = np.sum(S[self.max_bond_dim:] ** 2)
            self.cumulative_truncation_error += trunc_error
            k = self.max_bond_dim
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]

        # Remove small singular values
        mask = S > self.truncation_threshold
        if not np.all(mask):
            trunc_error = np.sum(S[~mask] ** 2)
            self.cumulative_truncation_error += trunc_error
            k = max(1, int(np.sum(mask)))
            U = U[:, :k]
            S = S[:k]
            Vh = Vh[:k, :]

        # Absorb singular values into right tensor (left-canonical form)
        S_diag = np.diag(S)
        self.tensors[q0] = U.reshape(bl, 2, k)
        self.tensors[q1] = (S_diag @ Vh).reshape(k, 2, br)

    def _apply_non_adjacent_two_qubit(
        self, mat: np.ndarray, q0: int, q1: int
    ) -> None:
        """Apply a two-qubit gate on non-adjacent qubits via SWAP chain."""
        swap_mat = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)

        # Move q0 next to q1 using SWAPs
        if q0 < q1:
            path = list(range(q0, q1 - 1))
            for i in path:
                self.apply_two_qubit_gate(swap_mat, i, i + 1)
            # Now apply gate at (q1-1, q1)
            self.apply_two_qubit_gate(mat, q1 - 1, q1)
            # Undo SWAPs
            for i in reversed(path):
                self.apply_two_qubit_gate(swap_mat, i, i + 1)
        else:
            path = list(range(q0, q1 + 1, -1))
            for i in path:
                self.apply_two_qubit_gate(swap_mat, i - 1, i)
            self.apply_two_qubit_gate(mat, q1, q1 + 1)
            for i in reversed(path):
                self.apply_two_qubit_gate(swap_mat, i - 1, i)

    @staticmethod
    def _swap_gate_qubits(mat: np.ndarray) -> np.ndarray:
        """Swap the qubit ordering of a 4x4 gate matrix."""
        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)
        return swap @ mat @ swap

    def get_statevector(self) -> np.ndarray:
        """
        Contract the MPS into a full statevector.
        WARNING: Exponential in qubit count!
        """
        if self.n > 20:
            raise RuntimeError(
                f"Cannot extract statevector for {self.n} qubits (too large)"
            )
        result = self.tensors[0]  # (1, 2, b1)
        for i in range(1, self.n):
            t = self.tensors[i]   # (b_{i-1}, 2, b_i)
            # Contract: result (1, 2^i, b_i) × t (b_i, 2, b_{i+1})
            result = np.einsum("...i,ijk->...jk", result, t)
        return result.reshape(-1)

    def sample(self, rng: np.random.Generator) -> str:
        """Sample a single bitstring from the MPS."""
        n = self.n
        bits = []
        # Left-to-right sampling
        # For simplicity, compute probabilities from statevector for small n
        if n <= 20:
            sv = self.get_statevector()
            probs = np.abs(sv) ** 2
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            idx = rng.choice(len(probs), p=probs)
            return format(idx, f"0{n}b")
        else:
            # For large n, use sequential sampling (approximate)
            # This is a simplified version
            remaining = self.tensors[0].copy()  # (1, 2, b1)
            for i in range(n):
                # Compute probability of 0 and 1 for qubit i
                if i == 0:
                    t = self.tensors[0]  # (1, 2, b1)
                    p0 = np.sum(np.abs(t[0, 0, :]) ** 2)
                    p1 = np.sum(np.abs(t[0, 1, :]) ** 2)
                else:
                    p0 = np.sum(np.abs(remaining[:, 0, :]) ** 2)
                    p1 = np.sum(np.abs(remaining[:, 1, :]) ** 2)

                total = p0 + p1
                if total < 1e-15:
                    bits.append(0)
                    continue
                p0 /= total
                bit = 0 if rng.random() < p0 else 1
                bits.append(bit)

                # Condition on this measurement
                if i < n - 1:
                    if i == 0:
                        conditional = self.tensors[0][0, bit, :]  # (b1,)
                        norm = np.linalg.norm(conditional)
                        if norm > 1e-15:
                            conditional = conditional / norm
                        remaining = np.einsum(
                            "i,ijk->jk", conditional, self.tensors[i + 1]
                        )
                        remaining = remaining[np.newaxis, :, :]
                    else:
                        conditional = remaining[:, bit, :]  # (..., bm)
                        norm = np.linalg.norm(conditional)
                        if norm > 1e-15:
                            conditional = conditional / norm
                        if i + 1 < n:
                            remaining = np.einsum(
                                "...i,ijk->...jk", conditional,
                                self.tensors[i + 1]
                            )

            return "".join(str(b) for b in bits)



class MPSBackend(Backend):
    """MPS simulation backend with configurable bond dimension χ."""

    name = "mps"
    method = "matrix_product_state"

    def __init__(self, max_qubits: int = 100, max_bond_dim: int = 64,
                 truncation_threshold: float = 1e-10,
                 warn_truncation_error: float = 1e-6):
        self.max_qubits = max_qubits
        self.max_bond_dim = max_bond_dim
        self.truncation_threshold = truncation_threshold
        self.warn_truncation_error = warn_truncation_error

    def validate_circuit(self, circuit: QuantumCircuit) -> List[str]:
        issues = []
        if circuit.num_qubits > self.max_qubits:
            issues.append(
                f"Circuit has {circuit.num_qubits} qubits exceeding MPS "
                f"limit of {self.max_qubits}"
            )
        for g in circuit.gates:
            if g.num_qubits > 2:
                issues.append(
                    f"MPS backend does not support {g.num_qubits}-qubit "
                    f"gate '{g.name}'. Decompose into 1- and 2-qubit gates."
                )
                break
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
                "MPS validation failed:\n" + "\n".join(issues)
            )

        n = circuit.num_qubits
        rng = np.random.default_rng(seed)

        # Build the MPS state
        mps = MPSState(n, self.max_bond_dim, self.truncation_threshold)

        # Apply gates
        for gate in circuit.gates:
            mat = gate.matrix
            if gate.num_qubits == 1:
                mps.apply_single_qubit_gate(mat, gate.qubits[0])
            elif gate.num_qubits == 2:
                mps.apply_two_qubit_gate(mat, gate.qubits[0], gate.qubits[1])
            else:
                raise RuntimeError(
                    f"MPS: {gate.num_qubits}-qubit gates not supported"
                )

        # Warn on high truncation error
        if mps.cumulative_truncation_error > self.warn_truncation_error:
            warnings.warn(
                f"MPS truncation error = {mps.cumulative_truncation_error:.2e} "
                f"exceeds threshold {self.warn_truncation_error:.2e}. "
                f"Results may be inaccurate. Consider increasing bond dimension.",
                RuntimeWarning,
            )

        # Sample measurements
        counts: Dict[str, int] = {}
        for _ in range(shots):
            bitstring = mps.sample(rng)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Extract statevector if small enough
        statevector = None
        probs = None
        if n <= 20:
            try:
                statevector = mps.get_statevector()
                probs = np.abs(statevector) ** 2
                probs = np.maximum(probs, 0)
                total = probs.sum()
                if total > 1e-15:
                    probs /= total
            except Exception:
                pass

        return SimulationResult(
            counts=counts,
            statevector=statevector,
            probabilities=probs,
            metadata={
                "backend": self.name,
                "method": self.method,
                "num_qubits": n,
                "num_gates": len(circuit.gates),
                "max_bond_dimension": self.max_bond_dim,
                "cumulative_truncation_error": mps.cumulative_truncation_error,
            },
            execution_label="ideal_simulation",
        )
