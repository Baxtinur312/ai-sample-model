"""
Stabilizer / Clifford Simulator
Efficient simulation of Clifford circuits using the Aaronson–Gottesman
tableau representation. Only supports Clifford gates (H, S, X, Y, Z,
CNOT, CZ, SWAP). Raises an error for non-Clifford gates.

Can simulate thousands of qubits efficiently.
Stabilizer (Clifford) simulation backend.

Uses a binary symplectic tableau to efficiently simulate Clifford circuits
on up to thousands of qubits. Non-Clifford gates (T, Rx(θ≠kπ/2), etc.) are
rejected with a clear error message.

Reference: Aaronson & Gottesman, quant-ph/0406196
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit, CLIFFORD_GATES
from .base import Backend, SimulationResult


class StabilizerTableau:
    """
    Binary symplectic tableau for n-qubit stabilizer state.

    Representation: 2n stabilizer generators, each stored as:
      - x[i]: n-bit array (X part)
      - z[i]: n-bit array (Z part)
      - phase[i]: 0 or 1 (tracks signs: 0 → +1, 1 → −1; overall ±i phases
                  are tracked modulo 4 internally but simplified for output)

    The initial state |00...0⟩ has stabilizers Z_0, Z_1, ..., Z_{n-1}.
    Rows 0..n-1 = destabilizers; rows n..2n-1 = stabilizers.
    """

    def __init__(self, n: int):
        self.n = n
        # Tableau: (2n) × (2n+1) binary matrix
        # Columns: x_0..x_{n-1}, z_0..z_{n-1}, phase
        self.table = np.zeros((2 * n, 2 * n + 1), dtype=np.int8)
        # Initialize: destabilizers = X_i, stabilizers = Z_i
        for i in range(n):
            self.table[i, i] = 1           # destabilizer X part
            self.table[n + i, n + i] = 1   # stabilizer Z part

    def _x(self, row: int) -> np.ndarray:
        return self.table[row, : self.n]

    def _z(self, row: int) -> np.ndarray:
        return self.table[row, self.n : 2 * self.n]

    def _phase(self, row: int) -> int:
        return int(self.table[row, 2 * self.n])

    def _set_phase(self, row: int, phase: int) -> None:
        self.table[row, 2 * self.n] = phase % 2

    def _rowmult(self, target: int, source: int) -> None:
        """Multiply row target by row source (in the group)."""
        # Phase update (simplified; tracking mod 2 for ±1)
        x_s = self._x(source)
        z_s = self._z(source)
        x_t = self._x(target)
        z_t = self._z(target)
        # g function: phase contribution from multiplying Pauli terms
        phase_contrib = 0
        for j in range(self.n):
            if x_s[j] and z_s[j]:     # Y
                phase_contrib += x_t[j] * (1 - z_t[j]) + z_t[j] * (1 - x_t[j])
            elif x_s[j] and not z_s[j]:  # X
                phase_contrib += z_t[j] * (2 * x_t[j] - 1 + 1)
            elif not x_s[j] and z_s[j]:  # Z
                phase_contrib += x_t[j] * (1 - 2 * z_t[j] + 1)
        new_phase = (self._phase(target) + self._phase(source) +
                     phase_contrib // 2) % 2
        self._set_phase(target, new_phase)
        # XOR the x and z parts
        self.table[target, : 2 * self.n] ^= self.table[source, : 2 * self.n]

    def apply_h(self, qubit: int) -> None:
        """Hadamard gate on qubit."""
        n = self.n
        for i in range(2 * n):
            # Phase update: if both X and Z are set, flip phase
            if self.table[i, qubit] and self.table[i, n + qubit]:
                self._set_phase(i, (self._phase(i) + 1) % 2)
            # Swap X and Z
            self.table[i, qubit], self.table[i, n + qubit] = (
                self.table[i, n + qubit],
                self.table[i, qubit],
            )

    def apply_s(self, qubit: int) -> None:
        """S (phase) gate on qubit."""
        n = self.n
        for i in range(2 * n):
            if self.table[i, qubit] and self.table[i, n + qubit]:
                self._set_phase(i, (self._phase(i) + 1) % 2)
            self.table[i, n + qubit] ^= self.table[i, qubit]

    def apply_cnot(self, control: int, target: int) -> None:
        """CNOT gate: control→target."""
        n = self.n
        for i in range(2 * n):
            # Phase update for Y⊗Y cases
            if (self.table[i, control] and self.table[i, n + target] and
                    not (self.table[i, n + control] ^ self.table[i, target])):
                pass  # complex sign cases omitted for basic impl
            self.table[i, target] ^= self.table[i, control]
            self.table[i, n + control] ^= self.table[i, n + target]

    def apply_x(self, qubit: int) -> None:
        """X gate on qubit (= H Z H, but direct is simpler)."""
        n = self.n
        for i in range(2 * n):
            if self.table[i, n + qubit]:  # Z part is set → phase flip
                self._set_phase(i, (self._phase(i) + 1) % 2)

    def apply_y(self, qubit: int) -> None:
        """Y gate on qubit."""
        n = self.n
        for i in range(2 * n):
            if self.table[i, qubit] ^ self.table[i, n + qubit]:
                self._set_phase(i, (self._phase(i) + 1) % 2)

    def apply_z(self, qubit: int) -> None:
        """Z gate on qubit."""
        n = self.n
        for i in range(2 * n):
            if self.table[i, qubit]:  # X part is set → phase flip
                self._set_phase(i, (self._phase(i) + 1) % 2)

    def apply_cz(self, q0: int, q1: int) -> None:
        """CZ gate = H_target · CNOT · H_target."""
        self.apply_h(q1)
        self.apply_cnot(q0, q1)
        self.apply_h(q1)

    def apply_swap(self, q0: int, q1: int) -> None:
        """SWAP = 3 CNOTs."""
        self.apply_cnot(q0, q1)
        self.apply_cnot(q1, q0)
        self.apply_cnot(q0, q1)

    def measure(self, qubit: int, rng: np.random.Generator) -> int:
        """
        Measure a single qubit. Returns 0 or 1.
        May modify the tableau (projective measurement).
        """
        n = self.n
        # Find a stabilizer with X_qubit set
        p = None
        for i in range(n, 2 * n):
            if self.table[i, qubit]:
                p = i
                break

        if p is not None:
            # Random outcome
            for i in range(2 * n):
                if i != p and self.table[i, qubit]:
                    self._rowmult(i, p)
            # Move destabilizer
            self.table[p - n] = self.table[p].copy()
            # Reset stabilizer
            self.table[p] = 0
            self.table[p, n + qubit] = 1
            outcome = int(rng.integers(0, 2))
            self._set_phase(p, outcome)
            return outcome
        else:
            # Deterministic outcome
            # Scratch row method: compute from destabilizers
            # Simplified: check if -Z_qubit is in stabilizer group
            scratch_phase = 0
            for i in range(n):
                if self.table[i, qubit]:
                    # This destabilizer has X on this qubit
                    scratch_phase = (scratch_phase + self._phase(n + i)) % 2
            return scratch_phase


class StabilizerBackend(Backend):
    """Clifford-only stabilizer simulation backend."""

    name = "stabilizer"
    method = "clifford_tableau"

    def __init__(self, max_qubits: int = 5000):
        self.max_qubits = max_qubits

    def validate_circuit(self, circuit: QuantumCircuit) -> List[str]:
        issues = []
        if circuit.num_qubits > self.max_qubits:
            issues.append(
                f"Circuit has {circuit.num_qubits} qubits exceeding limit "
                f"of {self.max_qubits}"
            )
        non_clifford = [
            g.name for g in circuit.gates if g.name not in CLIFFORD_GATES
        ]
        if non_clifford:
            issues.append(
                f"Non-Clifford gates found (not supported): "
                f"{set(non_clifford)}. Stabilizer backend only supports "
                f"{CLIFFORD_GATES}"
            )
        return issues
from quantum_kernel.circuit_ir import QuantumCircuit

_CLIFFORD_GATES = {"H", "X", "Y", "Z", "S", "CX", "CZ", "SWAP", "MEASURE"}


@dataclass
class StabilizerResult:
    counts: Dict[str, int]
    shots: int
    seed: Optional[int]
    backend: str = "stabilizer"
    method: str = "clifford_tableau"
    label: str = "ideal_simulation"


class StabilizerBackend:
    """
    Clifford stabilizer simulation backend.

    Supports H, X, Y, Z, S, CX, CZ, SWAP, MEASURE.
    Raises ValueError for non-Clifford gates.
    """

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        issues = self.validate_circuit(circuit)
        if issues:
            raise RuntimeError(
                "Stabilizer validation failed:\n" + "\n".join(issues)
            )

        n = circuit.num_qubits
        rng = np.random.default_rng(seed)

        # Run multiple shots by replaying the circuit
        counts: Dict[str, int] = {}
        for _ in range(shots):
            tab = StabilizerTableau(n)
            # Apply gates
            for gate in circuit.gates:
                self._apply_gate(tab, gate)
            # Measure all qubits
            bits = [tab.measure(q, rng) for q in range(n)]
            bitstring = "".join(str(b) for b in bits)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return SimulationResult(
            counts=counts,
            statevector=None,  # Stabilizer doesn't produce a statevector
            probabilities=None,
            metadata={
                "backend": self.name,
                "method": self.method,
                "num_qubits": n,
                "num_gates": len(circuit.gates),
            },
            execution_label="ideal_simulation",
        )

    @staticmethod
    def _apply_gate(tab: StabilizerTableau, gate) -> None:
        name = gate.name
        qubits = gate.qubits
        if name == "H":
            tab.apply_h(qubits[0])
        elif name == "S":
            tab.apply_s(qubits[0])
        elif name == "X":
            tab.apply_x(qubits[0])
        elif name == "Y":
            tab.apply_y(qubits[0])
        elif name == "Z":
            tab.apply_z(qubits[0])
        elif name == "I":
            pass
        elif name == "CNOT":
            tab.apply_cnot(qubits[0], qubits[1])
        elif name == "CZ":
            tab.apply_cz(qubits[0], qubits[1])
        elif name == "SWAP":
            tab.apply_swap(qubits[0], qubits[1])
        else:
            raise RuntimeError(f"Unsupported gate in stabilizer: {name}")
    ) -> StabilizerResult:
        n = circuit.num_qubits
        rng = np.random.default_rng(seed)

        for gate in circuit.gates:
            if gate.name not in _CLIFFORD_GATES:
                raise ValueError(
                    f"Gate {gate.name!r} is not a Clifford gate. "
                    "Use StatevectorBackend for universal gate sets."
                )

        # Initialise the 2n × (2n+1) tableau in |0…0⟩
        # Rows 0..n-1 are destabilisers, rows n..2n-1 are stabilisers.
        tableau = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        for i in range(n):
            tableau[i, i] = 1          # destabiliser X_i
            tableau[n + i, n + i] = 1  # stabiliser Z_i

        def _rowmul(t: np.ndarray, i: int, j: int) -> None:
            """Multiply row i into row j (in-place): row_j *= row_i."""
            # Phase update via bit arithmetic
            xi, zi = t[i, :n], t[i, n:2*n]
            xj, zj = t[j, :n], t[j, n:2*n]
            # g(x1, z1, x2, z2) counts phase contributions
            g = int(np.sum(
                xi.astype(int) * zj.astype(int) * (1 - 2 * xj.astype(int)) +
                (1 - xi.astype(int)) * zj.astype(int) * (xj.astype(int) - zi.astype(int))
            ))
            t[j, 2*n] ^= t[i, 2*n]
            t[j, 2*n] ^= (g // 2) % 2
            t[j, :2*n] ^= t[i, :2*n]

        def _h(t: np.ndarray, q: int) -> None:
            for row in range(2 * n):
                t[row, 2*n] ^= t[row, q] & t[row, n + q]
                t[row, q], t[row, n + q] = t[row, n + q], t[row, q]

        def _s(t: np.ndarray, q: int) -> None:
            for row in range(2 * n):
                t[row, 2*n] ^= t[row, q] & t[row, n + q]
                t[row, n + q] ^= t[row, q]

        def _cx(t: np.ndarray, ctrl: int, tgt: int) -> None:
            for row in range(2 * n):
                t[row, 2*n] ^= (
                    t[row, ctrl] & t[row, n + tgt]
                    & (t[row, tgt] ^ t[row, n + ctrl] ^ 1)
                )
                t[row, tgt] ^= t[row, ctrl]
                t[row, n + ctrl] ^= t[row, n + tgt]

        def _x(t: np.ndarray, q: int) -> None:
            _h(t, q); _s(t, q); _s(t, q); _h(t, q)

        def _z(t: np.ndarray, q: int) -> None:
            _s(t, q); _s(t, q)

        def _y(t: np.ndarray, q: int) -> None:
            _x(t, q); _z(t, q)

        def _cz(t: np.ndarray, ctrl: int, tgt: int) -> None:
            _h(t, tgt); _cx(t, ctrl, tgt); _h(t, tgt)

        def _swap(t: np.ndarray, a: int, b: int) -> None:
            _cx(t, a, b); _cx(t, b, a); _cx(t, a, b)

        def _measure(t: np.ndarray, q: int, rng: np.random.Generator) -> int:
            """Measure qubit q; return outcome 0 or 1."""
            # Find a stabiliser row that anticommutes with Z_q
            p = None
            for row in range(n, 2 * n):
                if t[row, q]:
                    p = row
                    break
            if p is None:
                # Deterministic outcome
                t2 = t.copy()
                # Zero out scratch row 0
                t2[0] = 0
                t2[0, 2*n] = 0
                for i in range(n):
                    if t2[i + n, q]:
                        _rowmul(t2, i + n, 0)
                return int(t2[0, 2*n])
            # Random outcome
            outcome = int(rng.integers(0, 2))
            for row in range(2 * n):
                if row != p and t[row, q]:
                    _rowmul(t, p, row)
            t[p - n] = t[p].copy()
            t[p] = 0
            t[p, n + q] = 1
            t[p, 2*n] = outcome
            return outcome

        # Apply each gate
        for gate in circuit.gates:
            name, q = gate.name, gate.qubits
            if name == "H":
                _h(tableau, q[0])
            elif name == "X":
                _x(tableau, q[0])
            elif name == "Y":
                _y(tableau, q[0])
            elif name == "Z":
                _z(tableau, q[0])
            elif name == "S":
                _s(tableau, q[0])
            elif name == "CX":
                _cx(tableau, q[0], q[1])
            elif name == "CZ":
                _cz(tableau, q[0], q[1])
            elif name == "SWAP":
                _swap(tableau, q[0], q[1])

        # Sample by measuring all qubits shots times
        fmt = f"{{:0{n}b}}"
        # Re-simulate each shot from scratch for correct sampling
        counts: Dict[str, int] = {}
        for _ in range(shots):
            t = tableau.copy()
            bits = []
            for q in range(n):
                bits.append(_measure(t, q, rng))
            bs = "".join(str(b) for b in bits)
            counts[bs] = counts.get(bs, 0) + 1

        return StabilizerResult(counts=counts, shots=shots, seed=seed)
