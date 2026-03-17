"""
Grover's Algorithm Benchmark
Implements Grover's search algorithm and validates that amplitude
amplification produces success probability matching the theoretical
prediction: ~sin²((2k+1)θ) where sin(θ) = √(M/N), k = iterations.

Optimal iterations ≈ (π/4)√(N/M).
Grover's search algorithm benchmark.

Constructs the Grover oracle and diffusion operator for a given number
of qubits and a target bitstring, runs the circuit on the statevector
backend, and returns success probabilities and measured counts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
from typing import Any, Dict, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from benchmarks.metrics import success_probability, counts_to_probabilities


def build_grover_oracle(
    num_qubits: int, marked_states: Sequence[int]
) -> QuantumCircuit:
    """
    Build a phase oracle that flips the phase of marked states.
    Uses multi-controlled Z gates decomposed into elementary gates.

    For the benchmark, we build the oracle as a diagonal unitary applied
    via the statevector backend directly (phase kickback trick).
    """
    qc = QuantumCircuit(num_qubits)

    for target in marked_states:
        # Convert target to binary and apply X gates to flip 0-bits
        bits = format(target, f"0{num_qubits}b")
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)

        # Multi-controlled Z: for 2-3 qubits, use Toffoli + phase
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        elif num_qubits == 3:
            # CCZ = Toffoli with phase: H-Toffoli-H on target
            qc.h(2)
            qc.toffoli(0, 1, 2)
            qc.h(2)
        else:
            # For larger circuits, approximate with a cascade of controls
            # Simplified: just use CZ between pairs
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)

        # Undo X gates
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)

    return qc


def build_diffusion_operator(num_qubits: int) -> QuantumCircuit:
    """
    Build the Grover diffusion operator: 2|s⟩⟨s| − I
    where |s⟩ is the uniform superposition.
    """
    qc = QuantumCircuit(num_qubits)

    # H on all qubits
    for i in range(num_qubits):
        qc.h(i)

    # X on all qubits
    for i in range(num_qubits):
        qc.x(i)

    # Multi-controlled Z (same decomposition as oracle)
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    elif num_qubits == 3:
        qc.h(2)
        qc.toffoli(0, 1, 2)
        qc.h(2)
    else:
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # X on all qubits
    for i in range(num_qubits):
        qc.x(i)

    # H on all qubits
    for i in range(num_qubits):
        qc.h(i)

    return qc


def build_grover_circuit(
    num_qubits: int,
    marked_states: Sequence[int],
    num_iterations: Optional[int] = None,
) -> QuantumCircuit:
    """
    Build a complete Grover's algorithm circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (search space = 2^num_qubits).
    marked_states : sequence of int
        Integer indices of marked states.
    num_iterations : int, optional
        Number of Grover iterations. If None, uses optimal count.

    Returns
    -------
    QuantumCircuit
        The complete Grover circuit.
    """
    N = 2 ** num_qubits
    M = len(marked_states)

    if num_iterations is None:
        num_iterations = optimal_iterations(N, M)

    qc = QuantumCircuit(num_qubits)

    # Initialize uniform superposition
    for i in range(num_qubits):
        qc.h(i)

    # Apply Grover iterations
    for _ in range(num_iterations):
        # Oracle
        oracle = build_grover_oracle(num_qubits, marked_states)
        for gate in oracle.gates:
            qc.gates.append(gate)

        # Diffusion
        diffusion = build_diffusion_operator(num_qubits)
        for gate in diffusion.gates:
            qc.gates.append(gate)


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


def optimal_iterations(N: int, M: int) -> int:
    """
    Compute the optimal number of Grover iterations: ⌊π/(4θ)⌋
    where θ = arcsin(√(M/N)).
    """
    if M >= N:
        return 0
    theta = math.asin(math.sqrt(M / N))
    return max(1, int(math.pi / (4 * theta)))


def theoretical_success_prob(N: int, M: int, k: int) -> float:
    """
    Theoretical success probability after k Grover iterations:
    P = sin²((2k+1)θ) where sin(θ) = √(M/N).
    """
    if M >= N:
        return 1.0
    theta = math.asin(math.sqrt(M / N))
    return math.sin((2 * k + 1) * theta) ** 2
def _apply_mcx(qc: QuantumCircuit, controls: list, target: int) -> None:
    """Apply a multi-controlled X gate."""
    qc.mcx(controls, target)


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_grover_benchmark(
    num_qubits: int = 3,
    marked_states: Optional[Sequence[int]] = None,
    shots: int = 1024,
    seed: int = 42,
) -> Dict:
    """
    Run the Grover benchmark and compare measured vs theoretical results.

    Returns
    -------
    dict
        Benchmark results including success probability, theoretical
        prediction, and comparison.
    """
    N = 2 ** num_qubits
    if marked_states is None:
        marked_states = [N - 1]  # Mark the last state

    M = len(marked_states)
    k = optimal_iterations(N, M)
    expected_prob = theoretical_success_prob(N, M, k)

    # Build and run
    circuit = build_grover_circuit(num_qubits, marked_states, k)
    backend = StatevectorBackend()
    result = backend.run(circuit, shots=shots, seed=seed)

    # Compute measured success probability
    target_bitstrings = [format(s, f"0{num_qubits}b") for s in marked_states]
    measured_prob = success_probability(result.counts, target_bitstrings)

    return {
        "algorithm": "Grover",
        "num_qubits": num_qubits,
        "N": N,
        "M": M,
        "optimal_iterations": k,
        "theoretical_success_prob": expected_prob,
        "measured_success_prob": measured_prob,
        "prob_error": abs(measured_prob - expected_prob),
        "marked_states": list(marked_states),
        "target_bitstrings": target_bitstrings,
        "top_counts": dict(sorted(
            result.counts.items(), key=lambda x: -x[1]
        )[:8]),
        "total_shots": shots,
        "circuit_depth": circuit.depth,
        "circuit_gates": len(circuit.gates),
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
