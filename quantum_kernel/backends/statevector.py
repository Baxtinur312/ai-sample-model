"""
Dense statevector simulation backend.

Simulates arbitrary quantum circuits exactly using a 2^n complex vector.
Supports all gates defined in circuit_ir and returns measurement counts
sampled from the probability distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit


# ---------------------------------------------------------------------------
# Gate matrices
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
        ],
        dtype=complex,
    )


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _apply_single(sv: np.ndarray, mat: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2×2 unitary to one qubit of the statevector."""
    sv = sv.reshape([2] * n)
    sv = np.tensordot(mat, sv, axes=[[1], [qubit]])
    # tensordot puts the contracted axis first; move it back
    sv = np.moveaxis(sv, 0, qubit)
    return sv.reshape(-1)


def _apply_cnot(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    """Apply CX (CNOT) gate."""
    sv = sv.reshape([2] * n)
    # Indices where the control qubit is |1⟩
    idx_ctrl1 = [slice(None)] * n
    idx_ctrl1[ctrl] = 1
    idx_ctrl1 = tuple(idx_ctrl1)
    # sv[idx_ctrl1] has shape (2,) * (n-1); tgt position shifts if tgt > ctrl
    tgt_in_slice = tgt if tgt < ctrl else tgt - 1
    sv_ctrl1 = sv[idx_ctrl1]
    sv_ctrl1 = np.tensordot(_X, sv_ctrl1, axes=[[1], [tgt_in_slice]])
    sv_ctrl1 = np.moveaxis(sv_ctrl1, 0, tgt_in_slice)
    sv[idx_ctrl1] = sv_ctrl1
    return sv.reshape(-1)


def _apply_cz(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    """Apply CZ gate."""
    sv = sv.reshape([2] * n)
    idx = [slice(None)] * n
    idx[ctrl] = 1
    idx[tgt] = 1
    sv[tuple(idx)] *= -1
    return sv.reshape(-1)


def _apply_mcx(sv: np.ndarray, controls: list, target: int, n: int) -> np.ndarray:
    """Apply a multi-controlled X gate (generalised Toffoli)."""
    sv = sv.reshape([2] * n)
    # Build index tuple where all control qubits are 1
    idx = [slice(None)] * n
    for c in controls:
        idx[c] = 1
    idx = tuple(idx)
    # Apply X to target within the selected slice
    sv_slice = sv[idx]
    # Move target axis to front, apply X, move back
    # Need to find which axis in sv_slice corresponds to `target`
    # Axes in sv_slice: all original axes except those fixed to 1 by controls
    # Build mapping from original qubit index to slice axis
    slice_axes = [i for i in range(n) if i not in controls]
    tgt_axis = slice_axes.index(target)
    sv_slice_new = np.tensordot(_X, sv_slice, axes=[[1], [tgt_axis]])
    sv_slice_new = np.moveaxis(sv_slice_new, 0, tgt_axis)
    sv[idx] = sv_slice_new
    return sv.reshape(-1)


def _apply_swap(sv: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """Apply SWAP gate."""
    sv = sv.reshape([2] * n)
    sv = np.swapaxes(sv, a, b)
    return sv.reshape(-1).copy()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StatevectorResult:
    """Result of a statevector simulation run."""

    statevector: np.ndarray
    counts: Dict[str, int]
    shots: int
    seed: Optional[int]
    backend: str = "statevector"
    method: str = "dense_statevector"
    label: str = "ideal_simulation"

    def probabilities(self) -> np.ndarray:
        return np.abs(self.statevector) ** 2


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class StatevectorBackend:
    """
    Exact dense-statevector simulation backend.

    Supports up to ~20 qubits comfortably (memory: 2^n complex128 values).
    """

    MAX_QUBITS = 20

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> StatevectorResult:
        """
        Simulate *circuit* and return a StatevectorResult.

        Parameters
        ----------
        circuit : QuantumCircuit
        shots   : int — number of measurement samples
        seed    : int | None — RNG seed for reproducibility
        """
        n = circuit.num_qubits
        if n > self.MAX_QUBITS:
            raise ValueError(
                f"StatevectorBackend supports at most {self.MAX_QUBITS} qubits "
                f"(circuit has {n})"
            )

        rng = np.random.default_rng(seed)

        # Initialize |0…0⟩
        sv = np.zeros(2**n, dtype=complex)
        sv[0] = 1.0

        # Apply gates
        for gate in circuit.gates:
            name = gate.name
            q = gate.qubits

            if name == "H":
                sv = _apply_single(sv, _H, q[0], n)
            elif name == "X":
                sv = _apply_single(sv, _X, q[0], n)
            elif name == "Y":
                sv = _apply_single(sv, _Y, q[0], n)
            elif name == "Z":
                sv = _apply_single(sv, _Z, q[0], n)
            elif name == "S":
                sv = _apply_single(sv, _S, q[0], n)
            elif name == "T":
                sv = _apply_single(sv, _T, q[0], n)
            elif name == "RX":
                sv = _apply_single(sv, _rx(gate.params[0]), q[0], n)
            elif name == "RY":
                sv = _apply_single(sv, _ry(gate.params[0]), q[0], n)
            elif name == "RZ":
                sv = _apply_single(sv, _rz(gate.params[0]), q[0], n)
            elif name == "U3":
                sv = _apply_single(sv, _u3(*gate.params[:3]), q[0], n)
            elif name == "CX":
                sv = _apply_cnot(sv, q[0], q[1], n)
            elif name == "CZ":
                sv = _apply_cz(sv, q[0], q[1], n)
            elif name == "SWAP":
                sv = _apply_swap(sv, q[0], q[1], n)
            elif name == "MCX":
                sv = _apply_mcx(sv, list(q[:-1]), q[-1], n)
            elif name == "MEASURE":
                pass  # handled at sampling stage
            else:
                raise ValueError(f"Unknown gate: {name!r}")

        # Sample measurements
        probs = np.abs(sv) ** 2
        probs = probs / probs.sum()  # normalise floating-point drift
        indices = rng.choice(len(probs), size=shots, p=probs)
        fmt = f"{{:0{n}b}}"
        counts: Dict[str, int] = {}
        for idx in indices:
            bs = fmt.format(idx)
            counts[bs] = counts.get(bs, 0) + 1

        return StatevectorResult(
            statevector=sv,
            counts=counts,
            shots=shots,
            seed=seed,
        )
