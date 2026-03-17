"""
Quantum Approximate Optimization Algorithm (QAOA) Benchmark
=============================================================
Implements QAOA for MaxCut on small graphs.

MaxCut: partition graph vertices into two sets to maximize edges between sets.
QAOA uses alternating problem (cost) and mixer unitaries parameterized by
(γ, β) to approximate the optimal solution.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from benchmarks.metrics import approximation_ratio


# ============================================================================
# MaxCut Problem
# ============================================================================

def maxcut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    """
    Compute the MaxCut value for a given bitstring partition.
    An edge (u, v) is cut if bits u and v differ.
    """
    cut = 0
    for u, v in edges:
        if bitstring[u] != bitstring[v]:
            cut += 1
    return cut


def brute_force_maxcut(
    num_nodes: int, edges: List[Tuple[int, int]]
) -> Tuple[int, str]:
    """
    Brute-force optimal MaxCut (exponential, small graphs only).

    Returns
    -------
    tuple
        (optimal_cut_value, optimal_bitstring)
    """
    best_cut = 0
    best_bs = "0" * num_nodes
    for i in range(2 ** num_nodes):
        bs = format(i, f"0{num_nodes}b")
        cut = maxcut_value(bs, edges)
        if cut > best_cut:
            best_cut = cut
            best_bs = bs
    return best_cut, best_bs


def maxcut_cost_hamiltonian(
    num_qubits: int, edges: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Build the MaxCut cost Hamiltonian:
    C = Σ_{(u,v) ∈ E} ½(I − Z_u Z_v)

    The ground state (maximum eigenvalue) corresponds to the MaxCut solution.
    """
    dim = 2 ** num_qubits
    H = np.zeros((dim, dim), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    for u, v in edges:
        # Build Z_u Z_v
        ops = [I2] * num_qubits
        ops[u] = Z
        ops[v] = Z
        ZZ = ops[0]
        for op in ops[1:]:
            ZZ = np.kron(ZZ, op)
        H += 0.5 * (np.eye(dim) - ZZ)

    return H


# ============================================================================
# QAOA Circuit
# ============================================================================

def build_qaoa_circuit(
    num_qubits: int,
    edges: List[Tuple[int, int]],
    gammas: np.ndarray,
    betas: np.ndarray,
) -> QuantumCircuit:
    """
    Build the QAOA circuit for MaxCut.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (= graph nodes).
    edges : list of (int, int)
        Graph edges.
    gammas : np.ndarray
        Problem unitary parameters (one per QAOA layer).
    betas : np.ndarray
        Mixer unitary parameters (one per QAOA layer).
    """
    p = len(gammas)  # QAOA depth
    qc = QuantumCircuit(num_qubits)

    # Initial state: uniform superposition
    for i in range(num_qubits):
        qc.h(i)

    # QAOA layers
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        # Problem unitary: exp(-i γ C) = Π_{(u,v)} exp(-i γ (1-Z_u Z_v)/2)
        for u, v in edges:
            # ZZ interaction: RZZ(γ) = CNOT · RZ(γ) · CNOT
            qc.cnot(u, v)
            qc.rz(v, gamma)
            qc.cnot(u, v)

        # Mixer unitary: exp(-i β Σ X_j) = Π_j RX(2β)
        for i in range(num_qubits):
            qc.rx(i, 2 * beta)

    return qc


# ============================================================================
# QAOA Optimization
# ============================================================================

def qaoa_expectation(
    params: np.ndarray,
    num_qubits: int,
    edges: List[Tuple[int, int]],
    p: int,
    backend: StatevectorBackend,
) -> float:
    """
    Compute ⟨ψ(γ,β)|C|ψ(γ,β)⟩ for QAOA.
    We negate since we want to maximize the cut but minimizers minimize.
    """
    gammas = params[:p]
    betas = params[p:]

    qc = build_qaoa_circuit(num_qubits, edges, gammas, betas)
    result = backend.run(qc, shots=1, seed=0)
    sv = result.statevector

    H = maxcut_cost_hamiltonian(num_qubits, edges)
    expectation = np.real(sv.conj() @ H @ sv)
    return -float(expectation)  # Negate for minimization


def run_qaoa(
    num_qubits: int,
    edges: List[Tuple[int, int]],
    p: int = 1,
    max_iterations: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Run QAOA optimization for MaxCut.

    Parameters
    ----------
    num_qubits : int
        Number of graph nodes/qubits.
    edges : list of (int, int)
        Graph edges.
    p : int
        QAOA depth (number of layers).
    max_iterations : int
        Maximum optimizer iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Optimization results.
    """
    backend = StatevectorBackend()
    rng = np.random.default_rng(seed)

    # Random initial parameters
    initial_params = rng.uniform(0, math.pi, 2 * p)

    # Optimize
    opt_result = minimize(
        qaoa_expectation,
        initial_params,
        args=(num_qubits, edges, p, backend),
        method="COBYLA",
        options={"maxiter": max_iterations},
    )

    # Best energy (negate back)
    best_energy = -opt_result.fun

    # Brute-force optimal
    optimal_cut, optimal_bitstring = brute_force_maxcut(num_qubits, edges)

    # Sample from optimal circuit
    best_gammas = opt_result.x[:p]
    best_betas = opt_result.x[p:]
    best_circuit = build_qaoa_circuit(num_qubits, edges, best_gammas, best_betas)
    sample_result = backend.run(best_circuit, shots=1024, seed=seed)

    # Find best sampled bitstring
    best_sampled = max(sample_result.counts, key=sample_result.counts.get)
    sampled_cut = maxcut_value(best_sampled, edges)

    return {
        "algorithm": "QAOA",
        "problem": "MaxCut",
        "num_qubits": num_qubits,
        "num_edges": len(edges),
        "qaoa_depth_p": p,
        "optimal_cut": optimal_cut,
        "optimal_bitstring": optimal_bitstring,
        "qaoa_energy": best_energy,
        "best_sampled_bitstring": best_sampled,
        "best_sampled_cut": sampled_cut,
        "approximation_ratio": approximation_ratio(best_energy, optimal_cut),
        "sampled_approx_ratio": approximation_ratio(sampled_cut, optimal_cut),
        "optimizer": "COBYLA",
        "converged": opt_result.success,
        "top_counts": dict(sorted(
            sample_result.counts.items(), key=lambda x: -x[1]
        )[:8]),
    }


def run_qaoa_benchmark(
    seed: int = 42,
) -> Dict:
    """
    Run the QAOA MaxCut benchmark on a small triangle + pendant graph.

    Graph:    0 — 1
              |   |
              3 — 2

    This is a 4-node cycle graph with optimal MaxCut = 4 (bipartite).
    """
    num_qubits = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    result = run_qaoa(num_qubits, edges, p=2, seed=seed)
    result["benchmark"] = "MaxCut_4node_cycle"
    return result
