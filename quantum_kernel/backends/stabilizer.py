"""
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
