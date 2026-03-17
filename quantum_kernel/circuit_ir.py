"""
Quantum Circuit Intermediate Representation.

Provides a gate library, circuit builder with method chaining,
ASCII art drawing, and a hash for reproducibility tracking.
"""

from __future__ import annotations

import hashlib
import json
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
    Classical-emulation quantum circuit builder.

    Supports method chaining for gate application:

        qc = QuantumCircuit(2)
        qc.h(0).cnot(0, 1)
    """

    def __init__(self, num_qubits: int) -> None:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
        self.num_qubits: int = num_qubits
        self._gates: List[Gate] = []

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------

    def _check_qubit(self, q: int) -> None:
        if not (0 <= q < self.num_qubits):
            raise ValueError(
                f"Qubit index {q} out of range for {self.num_qubits}-qubit circuit"
            )

    def _add(self, name: str, qubits: Tuple[int, ...], params: Tuple[float, ...] = ()) -> "QuantumCircuit":
        for q in qubits:
            self._check_qubit(q)
        self._gates.append(Gate(name, qubits, params))
        return self

    def h(self, qubit: int) -> "QuantumCircuit":
        """Hadamard gate."""
        return self._add("H", (qubit,))

    def x(self, qubit: int) -> "QuantumCircuit":
        """Pauli-X (NOT) gate."""
        return self._add("X", (qubit,))

    def y(self, qubit: int) -> "QuantumCircuit":
        """Pauli-Y gate."""
        return self._add("Y", (qubit,))

    def z(self, qubit: int) -> "QuantumCircuit":
        """Pauli-Z gate."""
        return self._add("Z", (qubit,))

    def s(self, qubit: int) -> "QuantumCircuit":
        """S (phase) gate."""
        return self._add("S", (qubit,))

    def t(self, qubit: int) -> "QuantumCircuit":
        """T gate (π/8 phase)."""
        return self._add("T", (qubit,))

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Rotation around X axis by theta radians."""
        return self._add("RX", (qubit,), (theta,))

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Rotation around Y axis by theta radians."""
        return self._add("RY", (qubit,), (theta,))

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Rotation around Z axis by theta radians."""
        return self._add("RZ", (qubit,), (theta,))

    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> "QuantumCircuit":
        """Generic U3 single-qubit rotation."""
        return self._add("U3", (qubit,), (theta, phi, lam))

    def mcx(self, controls: list, target: int) -> "QuantumCircuit":
        """Multi-controlled X gate (generalised Toffoli)."""
        for q in controls:
            self._check_qubit(q)
        self._check_qubit(target)
        all_qubits = tuple(controls) + (target,)
        self._gates.append(Gate("MCX", all_qubits))
        return self

    def measure(self, qubit: int) -> "QuantumCircuit":
        """Measurement gate (collapses qubit in Z basis)."""
        return self._add("MEASURE", (qubit,))

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Controlled-NOT (CX) gate."""
        return self._add("CX", (control, target))

    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Alias for cnot."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Controlled-Z gate."""
        return self._add("CZ", (control, target))

    def swap(self, qubit_a: int, qubit_b: int) -> "QuantumCircuit":
        """SWAP gate."""
        return self._add("SWAP", (qubit_a, qubit_b))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @property
    def gates(self) -> List[Gate]:
        return list(self._gates)

    @property
    def depth(self) -> int:
        """Circuit depth (layers) under a greedy scheduling heuristic."""
        qubit_time = [0] * self.num_qubits
        for gate in self._gates:
            t = max(qubit_time[q] for q in gate.qubits) + 1
            for q in gate.qubits:
                qubit_time[q] = t
        return max(qubit_time) if qubit_time else 0

    def digest(self) -> str:
        """SHA-256 fingerprint of the gate sequence (for provenance)."""
        payload = json.dumps(
            [(g.name, g.qubits, g.params) for g in self._gates], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # ASCII drawing
    # ------------------------------------------------------------------

    def draw(self) -> str:
        """Return a text-mode circuit diagram."""
        if not self._gates:
            return "\n".join(f"q{i}: ─────" for i in range(self.num_qubits))

        # Assign each gate to the earliest column where all its qubits are free
        col_of: List[int] = []
        qubit_col = [0] * self.num_qubits
        for gate in self._gates:
            col = max(qubit_col[q] for q in gate.qubits)
            col_of.append(col)
            for q in gate.qubits:
                qubit_col[q] = col + 1

        num_cols = max(qubit_col)

        # Build a grid: rows = qubits, cols = columns
        grid: List[List[str]] = [["─────"] * num_cols for _ in range(self.num_qubits)]

        for gate, col in zip(self._gates, col_of):
            if len(gate.qubits) == 1:
                q = gate.qubits[0]
                label = gate.label[:3]
                grid[q][col] = f"[{label:<3}]"
            elif len(gate.qubits) == 2:
                ctrl, tgt = gate.qubits[0], gate.qubits[1]
                if gate.name in ("CX",):
                    grid[ctrl][col] = "[ ● ]"
                    grid[tgt][col] = "[ ⊕ ]"
                elif gate.name == "CZ":
                    grid[ctrl][col] = "[ ● ]"
                    grid[tgt][col] = "[ Z ]"
                elif gate.name == "SWAP":
                    grid[ctrl][col] = "[ × ]"
                    grid[tgt][col] = "[ × ]"
                else:
                    label = gate.label[:3]
                    grid[ctrl][col] = f"[{label:<3}]"
                    grid[tgt][col] = f"[{label:<3}]"

        lines = []
        for i, row in enumerate(grid):
            lines.append(f"q{i}: " + "──".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"QuantumCircuit(n={self.num_qubits}, "
            f"gates={len(self._gates)}, depth={self.depth})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self._gates)
