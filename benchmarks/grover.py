"""
Grover's search algorithm benchmark.

Constructs the Grover oracle and diffusion operator for a given number
of qubits and a target bitstring, runs the circuit on the statevector
backend, and returns success probabilities and measured counts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend


# ---------------------------------------------------------------------------
# Oracle and diffusion
# ---------------------------------------------------------------------------


def _build_grover_circuit(num_qubits: int, target: int) -> QuantumCircuit:
    """
    Return a QuantumCircuit implementing Grover's algorithm.

    The target state is the integer *target* encoded in binary.
    The number of Grover iterations is set to the theoretically optimal value.
    """
    n = num_qubits
    N = 2 ** n
    num_iters = max(1, math.floor(math.pi / 4 * math.sqrt(N)))

    qc = QuantumCircuit(n)

    # Initialise uniform superposition
    for q in range(n):
        qc.h(q)

    for _ in range(num_iters):
        # ----- Oracle: flip phase of |target⟩ -----
        # Apply X to qubits where target bit is 0
        for q in range(n):
            if not (target >> q) & 1:
                qc.x(q)
        # Multi-controlled Z via conjugation with H on last qubit
        qc.h(n - 1)
        _apply_mcx(qc, list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q in range(n):
            if not (target >> q) & 1:
                qc.x(q)

        # ----- Diffusion: 2|ψ⟩⟨ψ| − I -----
        for q in range(n):
            qc.h(q)
        for q in range(n):
            qc.x(q)
        qc.h(n - 1)
        _apply_mcx(qc, list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for q in range(n):
            qc.x(q)
        for q in range(n):
            qc.h(q)

    return qc


def _apply_mcx(qc: QuantumCircuit, controls: list, target: int) -> None:
    """Apply a multi-controlled X gate."""
    qc.mcx(controls, target)


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_grover_benchmark(
    num_qubits: int = 3,
    shots: int = 2048,
    seed: Optional[int] = None,
    target: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run Grover's search benchmark.

    Parameters
    ----------
    num_qubits : int
        Number of search qubits (database size = 2^num_qubits).
    shots : int
        Number of measurement samples.
    seed : int | None
        RNG seed for reproducibility.
    target : int | None
        Index of the marked item. Defaults to 2^num_qubits - 1.

    Returns
    -------
    dict with keys:
        num_qubits, N, target, target_bitstring,
        measured_success_prob, theoretical_success_prob, prob_error,
        top_counts, circuit_depth
    """
    n = num_qubits
    N = 2 ** n
    if target is None:
        target = N - 1

    qc = _build_grover_circuit(n, target)
    backend = StatevectorBackend()
    result = backend.run(qc, shots=shots, seed=seed)

    fmt = f"{{:0{n}b}}"
    target_bs = fmt.format(target)
    success_count = result.counts.get(target_bs, 0)
    measured_prob = success_count / shots

    # Theoretical success probability after optimal iterations
    num_iters = max(1, math.floor(math.pi / 4 * math.sqrt(N)))
    theta = math.asin(1.0 / math.sqrt(N))
    theoretical_prob = math.sin((2 * num_iters + 1) * theta) ** 2

    top_counts = dict(
        sorted(result.counts.items(), key=lambda x: -x[1])[:8]
    )

    return {
        "num_qubits": n,
        "N": N,
        "target": target,
        "target_bitstring": target_bs,
        "measured_success_prob": measured_prob,
        "theoretical_success_prob": theoretical_prob,
        "prob_error": abs(measured_prob - theoretical_prob),
        "top_counts": top_counts,
        "circuit_depth": qc.depth,
    }
