"""
Quantum Fourier Transform (QFT) benchmark.

Constructs the QFT circuit, runs it on several input states, and
compares the output statevectors against the classical DFT result.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from benchmarks.metrics import state_fidelity


# ---------------------------------------------------------------------------
# QFT circuit builder
# ---------------------------------------------------------------------------


def _build_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """Return a QuantumCircuit implementing the QFT."""
    n = num_qubits
    qc = QuantumCircuit(n)

    for j in range(n):
        qc.h(j)
        for k in range(j + 1, n):
            theta = math.pi / (2 ** (k - j))
            # Controlled phase via RZ decomposition: CP(θ) ≈ CX · RZ · CX
            qc.rz(theta / 2, k)
            qc.cnot(j, k)
            qc.rz(-theta / 2, k)
            qc.cnot(j, k)
            qc.rz(theta / 2, j)

    # Bit-reversal permutation
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)

    return qc


def _prepare_input_state(num_qubits: int, state_index: int) -> QuantumCircuit:
    """Prepare a computational basis input state |state_index⟩."""
    n = num_qubits
    qc = QuantumCircuit(n)
    for q in range(n):
        if (state_index >> q) & 1:
            qc.x(q)
    return qc


def _combine(prep: QuantumCircuit, qft: QuantumCircuit) -> QuantumCircuit:
    """Concatenate a preparation circuit with the QFT circuit."""
    n = prep.num_qubits
    combined = QuantumCircuit(n)
    for gate in prep.gates + qft.gates:
        combined._gates.append(gate)
    return combined


def _classical_qft(input_vec: np.ndarray) -> np.ndarray:
    """Compute QFT output using numpy's FFT (exact reference)."""
    N = len(input_vec)
    return np.fft.fft(input_vec) / math.sqrt(N)


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_qft_benchmark(
    num_qubits: int = 3,
    shots: int = 1024,
    seed: Optional[int] = None,
    test_states: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run QFT benchmark on several input states.

    Parameters
    ----------
    num_qubits : int
    shots : int
    seed : int | None
    test_states : list of int | None
        Computational basis states to test. Defaults to [0, 1, 2].

    Returns
    -------
    dict with:
        num_qubits, qft_depth, tests (list of per-state results)
    """
    n = num_qubits
    N = 2 ** n
    if test_states is None:
        test_states = [0, 1, 2]

    qft_circ = _build_qft_circuit(n)
    backend = StatevectorBackend()

    tests = []
    for idx in test_states:
        if idx >= N:
            continue
        prep = _prepare_input_state(n, idx)
        full = _combine(prep, qft_circ)
        result = backend.run(full, shots=shots, seed=seed)
        sv_sim = result.statevector

        # Classical reference
        input_vec = np.zeros(N, dtype=complex)
        input_vec[idx] = 1.0
        sv_ref = _classical_qft(input_vec)

        fid = state_fidelity(sv_sim, sv_ref)

        tests.append(
            {
                "input_state": idx,
                "input_bitstring": f"{{:0{n}b}}".format(idx),
                "fidelity": fid,
                "statevector_norm": float(np.linalg.norm(sv_sim)),
            }
        )

    avg_fidelity = sum(t["fidelity"] for t in tests) / len(tests) if tests else 0.0

    return {
        "num_qubits": n,
        "qft_depth": qft_circ.depth,
        "average_fidelity": avg_fidelity,
        "tests": tests,
    }
