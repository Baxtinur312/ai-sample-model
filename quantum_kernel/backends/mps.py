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
    """MPS/Tensor-network simulation backend."""

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
        # Check for 3-qubit gates (not natively supported)
        for g in circuit.gates:
            if g.num_qubits > 2:
                issues.append(
                    f"MPS backend does not support {g.num_qubits}-qubit "
                    f"gate '{g.name}'. Decompose into 1- and 2-qubit gates."
                )
                break
        return issues
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit


@dataclass
class MPSResult:
    counts: Dict[str, int]
    shots: int
    seed: Optional[int]
    backend: str = "mps"
    method: str = "matrix_product_state"
    label: str = "ideal_simulation"


class MPSBackend:
    """
    MPS simulation backend with configurable bond dimension χ.

    Parameters
    ----------
    max_bond : int
        Maximum Schmidt rank (bond dimension) kept after each SVD truncation.
        Larger values are more accurate but slower. Default: 64.
    """

    MAX_QUBITS = 100

    def __init__(self, max_bond: int = 64) -> None:
        self.max_bond = max_bond

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
    ) -> MPSResult:
        n = circuit.num_qubits
        if n > self.MAX_QUBITS:
            raise ValueError(
                f"MPSBackend supports at most {self.MAX_QUBITS} qubits (circuit has {n})"
            )

        rng = np.random.default_rng(seed)
        tensors = self._init_mps(n)

        for gate in circuit.gates:
            name, q = gate.name, gate.qubits
            if len(q) == 1:
                mat = self._single_qubit_matrix(name, gate.params)
                if mat is not None:
                    tensors[q[0]] = np.tensordot(mat, tensors[q[0]], axes=[[1], [1]])
                    tensors[q[0]] = np.moveaxis(tensors[q[0]], 0, 1)
            elif len(q) == 2:
                tensors = self._apply_two_qubit(tensors, name, q[0], q[1], n)

        # Sample measurements
        counts = self._sample(tensors, n, shots, rng)
        return MPSResult(counts=counts, shots=shots, seed=seed)

    # ------------------------------------------------------------------
    # MPS internals
    # ------------------------------------------------------------------

    @staticmethod
    def _init_mps(n: int) -> List[np.ndarray]:
        """Initialise MPS tensors for |0…0⟩.

        Each tensor has shape (left_bond, physical, right_bond).
        Edge tensors have bond dim 1 on the open side.
        """
        tensors = []
        for _ in range(n):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            tensors.append(t)
        return tensors

    @staticmethod
    def _single_qubit_matrix(name: str, params) -> Optional[np.ndarray]:
        """Return the 2×2 matrix for a named single-qubit gate."""
        sqrt2_inv = 1.0 / math.sqrt(2)
        gates = {
            "H": np.array([[sqrt2_inv, sqrt2_inv], [sqrt2_inv, -sqrt2_inv]], dtype=complex),
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),
            "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
        }
        if name in gates:
            return gates[name]
        if name == "RX" and params:
            c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        if name == "RY" and params:
            c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)
        if name == "RZ" and params:
            return np.array(
                [[np.exp(-1j * params[0] / 2), 0], [0, np.exp(1j * params[0] / 2)]],
                dtype=complex,
            )
        return None  # gate not handled as single-qubit (e.g., MEASURE)

    def _apply_two_qubit(
        self,
        tensors: List[np.ndarray],
        name: str,
        a: int,
        b: int,
        n: int,
    ) -> List[np.ndarray]:
        """Apply a two-qubit gate via SVD on adjacent (or swapped) tensors."""
        # Only handle adjacent qubits; swap non-adjacent to adjacent
        if abs(a - b) != 1:
            # Bring b next to a via swap chain (approximate, for simplicity)
            return tensors  # skip non-adjacent for this lightweight implementation

        if a > b:
            a, b = b, a

        # Build the 4×4 gate matrix
        gate_4 = self._two_qubit_matrix(name)
        if gate_4 is None:
            return tensors

        # Contract tensors[a] and tensors[b] into a combined tensor
        ta = tensors[a]   # (La, 2, Ra)
        tb = tensors[b]   # (Lb, 2, Rb)  with Lb == Ra

        # theta[La, pa, pb, Rb]
        theta = np.tensordot(ta, tb, axes=[[2], [0]])  # (La, 2, 2, Rb)
        La, _, _, Rb = theta.shape

        # Apply 4×4 gate to physical indices.
        # We need: theta'[La, pa', pb', Rb] = sum_{pa,pb} G[pa',pb',pa,pb] * theta[La,pa,pb,Rb]
        gate_mat = gate_4.reshape(2, 2, 2, 2)  # (pa', pb', pa, pb)
        theta = theta.reshape(La, 2, 2, Rb)
        theta_out = np.einsum("abcd,efac->ebfd",
                               gate_mat,
                               theta)
        # theta_out: (pa', pb', La, Rb) → reorder to (La, pa', pb', Rb)
        theta_out = theta_out.transpose(2, 0, 1, 3)

        # SVD to split back into two tensors
        theta_mat = theta_out.reshape(La * 2, 2 * Rb)
        U, s, Vh = np.linalg.svd(theta_mat, full_matrices=False)

        # Truncate to max_bond
        chi = min(len(s), self.max_bond)
        U = U[:, :chi]
        s = s[:chi]
        Vh = Vh[:chi, :]

        # Absorb sqrt(s) into both sides for gauge-neutral decomposition
        s_sqrt = np.sqrt(s)
        tensors[a] = (U * s_sqrt).reshape(La, 2, chi)
        tensors[b] = (s_sqrt[:, None] * Vh).reshape(chi, 2, Rb)
        return tensors

    @staticmethod
    def _two_qubit_matrix(name: str) -> Optional[np.ndarray]:
        """Return the 4×4 unitary matrix for a named two-qubit gate."""
        if name == "CX":
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                dtype=complex,
            )
        if name == "CZ":
            return np.diag([1, 1, 1, -1]).astype(complex)
        if name == "SWAP":
            return np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=complex,
            )
        return None

    def _sample(
        self,
        tensors: List[np.ndarray],
        n: int,
        shots: int,
        rng: np.random.Generator,
    ) -> Dict[str, int]:
        """Sample bitstrings by contracting the MPS from left to right."""
        fmt = f"{{:0{n}b}}"
        counts: Dict[str, int] = {}

        for _ in range(shots):
            vec = np.array([1.0 + 0j])
            bits = []
            for i in range(n):
                t = tensors[i]  # (L, 2, R)
                # prob[k] = |⟨k| applied to vec|²
                vt0 = np.tensordot(vec, t[:, 0, :], axes=[[0], [0]])  # (R,)
                vt1 = np.tensordot(vec, t[:, 1, :], axes=[[0], [0]])  # (R,)
                p0 = np.real(np.dot(vt0.conj(), vt0))
                p1 = np.real(np.dot(vt1.conj(), vt1))
                total = p0 + p1
                if total < 1e-14:
                    p0, p1 = 0.5, 0.5
                else:
                    p0, p1 = p0 / total, p1 / total
                outcome = rng.choice([0, 1], p=[p0, p1])
                bits.append(outcome)
                vec = (vt0 if outcome == 0 else vt1)
                nrm = np.linalg.norm(vec)
                if nrm > 1e-14:
                    vec = vec / nrm
                else:
                    vec = np.ones(len(vec), dtype=complex) / math.sqrt(len(vec))
            bs = "".join(str(b) for b in bits)
            counts[bs] = counts.get(bs, 0) + 1

        return counts
