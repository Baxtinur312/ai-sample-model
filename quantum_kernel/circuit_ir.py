"""
Circuit Intermediate Representation (IR)
Defines QuantumGate and QuantumCircuit classes with a built-in gate library.
All standard gates are represented as unitary matrices.
Quantum Circuit Intermediate Representation.

Provides a gate library, circuit builder with method chaining,
ASCII art drawing, and a hash for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Gate matrix library
# ---------------------------------------------------------------------------

_SQRT2_INV = 1.0 / math.sqrt(2.0)

GATE_MATRICES: Dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": np.array([[_SQRT2_INV, _SQRT2_INV],
                    [_SQRT2_INV, -_SQRT2_INV]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
    "CNOT": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex),
    "CZ": np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=complex),
    "SWAP": np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex),
    "TOFFOLI": np.eye(8, dtype=complex),
}
# Toffoli: flip the |111⟩↔|110⟩ entries
GATE_MATRICES["TOFFOLI"][6, 6] = 0
GATE_MATRICES["TOFFOLI"][7, 7] = 0
GATE_MATRICES["TOFFOLI"][6, 7] = 1
GATE_MATRICES["TOFFOLI"][7, 6] = 1

# Clifford gate set (used by stabilizer backend)
CLIFFORD_GATES = {"I", "X", "Y", "Z", "H", "S", "CNOT", "CZ", "SWAP"}


def _rx(theta: float) -> np.ndarray:
    """Rotation about X axis."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    """Rotation about Y axis."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    """Rotation about Z axis."""
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=complex)


PARAMETRIC_GATE_BUILDERS = {
    "RX": _rx,
    "RY": _ry,
    "RZ": _rz,
}


# ---------------------------------------------------------------------------
# QuantumGate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumGate:
    """A single quantum gate application."""

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        """Return the unitary matrix for this gate."""
        if self.name in GATE_MATRICES:
            return GATE_MATRICES[self.name].copy()
        if self.name in PARAMETRIC_GATE_BUILDERS:
            if len(self.params) != 1:
                raise ValueError(
                    f"{self.name} requires exactly 1 parameter, got {len(self.params)}"
                )
            return PARAMETRIC_GATE_BUILDERS[self.name](self.params[0])
        raise ValueError(f"Unknown gate: {self.name}")

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def is_clifford(self) -> bool:
        return self.name in CLIFFORD_GATES

    def __repr__(self) -> str:
        p = f", params={self.params}" if self.params else ""
        return f"QuantumGate({self.name}, qubits={self.qubits}{p})"
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Gate definitions
# ---------------------------------------------------------------------------

GATE_NAMES = {
    "H": "H",
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "S": "S",
    "T": "T",
    "CX": "CX",   # CNOT
    "CZ": "CZ",
    "SWAP": "SW",
    "RX": "Rx",
    "RY": "Ry",
    "RZ": "Rz",
    "U3": "U3",
    "MEASURE": "M",
    "MCX": "MCX",
    "MCZ": "MCZ",
}


class Gate:
    """Immutable description of a single gate application."""

    __slots__ = ("name", "qubits", "params", "label")

    def __init__(
        self,
        name: str,
        qubits: Tuple[int, ...],
        params: Tuple[float, ...] = (),
        label: Optional[str] = None,
    ) -> None:
        self.name = name
        self.qubits = tuple(qubits)
        self.params = tuple(params)
        self.label = label or GATE_NAMES.get(name, name)

    def __repr__(self) -> str:
        p = f"({', '.join(f'{v:.4g}' for v in self.params)})" if self.params else ""
        return f"{self.name}{p} {self.qubits}"


# ---------------------------------------------------------------------------
# QuantumCircuit
# ---------------------------------------------------------------------------

class QuantumCircuit:
    """
    A quantum circuit represented as an ordered list of gate operations.

    Example
    -------
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cnot(0, 1)
    >>> print(qc)
    QuantumCircuit(2 qubits, 2 gates)
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("Circuit must have at least 1 qubit")
        self.num_qubits = num_qubits
        self.gates: List[QuantumGate] = []
        self._metadata: Dict = {}

    # -- Single-qubit gates --------------------------------------------------

    def i(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("I", (qubit,))
        return self

    def x(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("X", (qubit,))
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("Y", (qubit,))
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("Z", (qubit,))
        return self

    def h(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("H", (qubit,))
        return self

    def s(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("S", (qubit,))
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        self._add_gate("T", (qubit,))
        return self

    def rx(self, qubit: int, theta: float) -> "QuantumCircuit":
        self._add_gate("RX", (qubit,), (theta,))
        return self

    def ry(self, qubit: int, theta: float) -> "QuantumCircuit":
        self._add_gate("RY", (qubit,), (theta,))
        return self

    def rz(self, qubit: int, theta: float) -> "QuantumCircuit":
        self._add_gate("RZ", (qubit,), (theta,))
        return self

    # -- Multi-qubit gates ---------------------------------------------------

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        self._add_gate("CNOT", (control, target))
        return self

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Alias for CNOT."""
        return self.cnot(control, target)

    def cz(self, qubit0: int, qubit1: int) -> "QuantumCircuit":
        self._add_gate("CZ", (qubit0, qubit1))
        return self

    def swap(self, qubit0: int, qubit1: int) -> "QuantumCircuit":
        self._add_gate("SWAP", (qubit0, qubit1))
        return self

    def toffoli(self, c0: int, c1: int, target: int) -> "QuantumCircuit":
        self._add_gate("TOFFOLI", (c0, c1, target))
        return self

    def ccx(self, c0: int, c1: int, target: int) -> "QuantumCircuit":
        """Alias for Toffoli."""
        return self.toffoli(c0, c1, target)

    # -- Generic gate --------------------------------------------------------

    def add_gate(self, name: str, qubits: Sequence[int],
                 params: Sequence[float] = ()) -> "QuantumCircuit":
        """Add an arbitrary named gate."""
        self._add_gate(name, tuple(qubits), tuple(params))
        return self

    # -- Internal helpers ----------------------------------------------------

    def _add_gate(self, name: str, qubits: Tuple[int, ...],
                  params: Tuple[float, ...] = ()) -> None:
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                raise IndexError(
                    f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit"
                )
        self.gates.append(QuantumGate(name=name, qubits=qubits, params=params))

    # -- Properties ----------------------------------------------------------

    @property
    def depth(self) -> int:
        """Approximate circuit depth (layers of non-overlapping gates)."""
        if not self.gates:
            return 0
        qubit_time = [0] * self.num_qubits
        for gate in self.gates:
            t = max(qubit_time[q] for q in gate.qubits) + 1
            for q in gate.qubits:
                qubit_time[q] = t
        return max(qubit_time)

    @property
    def gate_counts(self) -> Dict[str, int]:
        """Count of each gate type in the circuit."""
        counts: Dict[str, int] = {}
        for g in self.gates:
            counts[g.name] = counts.get(g.name, 0) + 1
        return counts

    @property
    def is_clifford(self) -> bool:
        """True if all gates are in the Clifford group."""
        return all(g.is_clifford for g in self.gates)

    def to_gate_list(self) -> List[Dict]:
        """Serialize the circuit as a list of gate dictionaries."""
        return [
            {
                "name": g.name,
                "qubits": list(g.qubits),
                "params": list(g.params),
            }
            for g in self.gates
        ]

    def circuit_hash(self) -> str:
        """Deterministic SHA-256 hash of the circuit for provenance."""
        data = json.dumps({
            "num_qubits": self.num_qubits,
            "gates": self.to_gate_list(),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_unitary(self) -> np.ndarray:
        """
        Compute the full unitary matrix of the circuit.
        WARNING: Exponential in qubit count. Only for small circuits.
        """
        if self.num_qubits > 10:
            raise RuntimeError(
                f"to_unitary() is infeasible for {self.num_qubits} qubits "
                f"(matrix would be {2**self.num_qubits}x{2**self.num_qubits})"
            )
        dim = 2 ** self.num_qubits
        U = np.eye(dim, dtype=complex)
        for gate in self.gates:
            U = _embed_gate(gate, self.num_qubits) @ U
        return U

    def draw(self) -> str:
        """Simple ASCII drawing of the circuit."""
        lines = [f"q{i}: " for i in range(self.num_qubits)]
        for gate in self.gates:
            # Determine column width
            label = gate.name
            if gate.params:
                label += f"({','.join(f'{p:.2f}' for p in gate.params)})"
            col_width = max(len(label) + 2, 5)
            for i in range(self.num_qubits):
                if i == gate.qubits[0]:
                    cell = f"[{label}]"
                elif i in gate.qubits:
                    cell = f"[{'*' * len(label)}]"
                else:
                    cell = "-" * (len(label) + 2)
                lines[i] += cell.ljust(col_width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"QuantumCircuit({self.num_qubits} qubits, {len(self.gates)} gates)"

    def __len__(self) -> int:
        return len(self.gates)

    def copy(self) -> "QuantumCircuit":
        """Return a deep copy of this circuit."""
        qc = QuantumCircuit(self.num_qubits)
        qc.gates = list(self.gates)
        qc._metadata = dict(self._metadata)
        return qc


# ---------------------------------------------------------------------------
# Gate embedding helper
# ---------------------------------------------------------------------------

def _embed_gate(gate: QuantumGate, num_qubits: int) -> np.ndarray:
    """
    Embed a gate's unitary into the full Hilbert space.
    Handles arbitrary qubit orderings for multi-qubit gates.
    """
    mat = gate.matrix
    n = num_qubits
    dim = 2 ** n

    if gate.num_qubits == 1:
        # Single-qubit: tensor product  I ⊗ ... ⊗ U ⊗ ... ⊗ I
        q = gate.qubits[0]
        ops = [np.eye(2, dtype=complex)] * n
        ops[q] = mat
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    elif gate.num_qubits == 2:
        # Two-qubit gate: build permutation to canonical qubit order
        q0, q1 = gate.qubits
        full = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                # Extract the bits at positions q0, q1
                bi0 = (i >> (n - 1 - q0)) & 1
                bi1 = (i >> (n - 1 - q1)) & 1
                bj0 = (j >> (n - 1 - q0)) & 1
                bj1 = (j >> (n - 1 - q1)) & 1
                # Other bits must match
                i_rest = i & ~((1 << (n - 1 - q0)) | (1 << (n - 1 - q1)))
                j_rest = j & ~((1 << (n - 1 - q0)) | (1 << (n - 1 - q1)))
                if i_rest != j_rest:
                    continue
                row_2q = bi0 * 2 + bi1
                col_2q = bj0 * 2 + bj1
                full[i, j] += mat[row_2q, col_2q]
        return full

    elif gate.num_qubits == 3:
        # Three-qubit gate (e.g., Toffoli)
        q0, q1, q2 = gate.qubits
        full = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            for j in range(dim):
                bi0 = (i >> (n - 1 - q0)) & 1
                bi1 = (i >> (n - 1 - q1)) & 1
                bi2 = (i >> (n - 1 - q2)) & 1
                bj0 = (j >> (n - 1 - q0)) & 1
                bj1 = (j >> (n - 1 - q1)) & 1
                bj2 = (j >> (n - 1 - q2)) & 1
                mask = ((1 << (n - 1 - q0)) | (1 << (n - 1 - q1)) |
                        (1 << (n - 1 - q2)))
                if (i & ~mask) != (j & ~mask):
                    continue
                row_3q = bi0 * 4 + bi1 * 2 + bi2
                col_3q = bj0 * 4 + bj1 * 2 + bj2
                full[i, j] += mat[row_3q, col_3q]
        return full
    else:
        raise NotImplementedError(
            f"Gate embedding for {gate.num_qubits}-qubit gates not implemented"
        )

