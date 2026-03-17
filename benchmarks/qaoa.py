"""
QAOA (Quantum Approximate Optimisation Algorithm) benchmark.

Solves the MaxCut problem on a 4-node cycle graph using QAOA.

Graph: 0 — 1 — 2 — 3 — 0  (cycle C₄)
Optimal MaxCut value = 4 (alternating colouring)

The QAOA circuit alternates between the cost unitary U_C and the mixer
unitary U_B = exp(-iβ ΣX_i).  Classical optimisation finds β, γ ∈ ℝ.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

_EDGES: List[Tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 0)]
_NUM_NODES = 4
_OPTIMAL_CUT = 4


def _maxcut_value(bitstring: str, edges: List[Tuple[int, int]]) -> int:
    """Return the MaxCut value for a given bitstring (0/1 assignment)."""
    return sum(1 for u, v in edges if bitstring[u] != bitstring[v])


# ---------------------------------------------------------------------------
# QAOA circuit builder
# ---------------------------------------------------------------------------


def _build_qaoa_circuit(
    gamma: float,
    beta: float,
    edges: List[Tuple[int, int]],
    num_nodes: int,
    p: int = 1,
) -> QuantumCircuit:
    """Build a p-layer QAOA circuit."""
    qc = QuantumCircuit(num_nodes)

    # Initialise uniform superposition
    for q in range(num_nodes):
        qc.h(q)

    for _ in range(p):
        # Cost layer: exp(-i γ/2 (I - Z_u Z_v)) for each edge
        for u, v in edges:
            qc.cnot(u, v)
            qc.rz(gamma, v)
            qc.cnot(u, v)

        # Mixer layer: exp(-i β X_i) = Rx(2β) on each qubit
        for q in range(num_nodes):
            qc.rx(2 * beta, q)

    return qc


# ---------------------------------------------------------------------------
# Cost expectation value
# ---------------------------------------------------------------------------


def _qaoa_energy(params: np.ndarray, backend: StatevectorBackend) -> float:
    """Return negative expected MaxCut value (for minimisation)."""
    gamma, beta = params
    qc = _build_qaoa_circuit(gamma, beta, _EDGES, _NUM_NODES, p=1)
    result = backend.run(qc, shots=4096, seed=0)
    sv = result.statevector
    n = _NUM_NODES

    # Compute ⟨ψ|H_C|ψ⟩ exactly from statevector
    probs = np.abs(sv) ** 2
    energy = 0.0
    for idx, prob in enumerate(probs):
        bs = f"{idx:0{n}b}"
        energy += prob * _maxcut_value(bs, _EDGES)
    return -energy  # negate for minimisation


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_qaoa_benchmark(
    seed: Optional[int] = None,
    p: int = 1,
) -> Dict[str, Any]:
    """
    Run QAOA MaxCut benchmark on a 4-node cycle graph.

    Returns
    -------
    dict with:
        optimal_cut, qaoa_energy, approximation_ratio,
        best_sampled_cut, best_sampled_bitstring,
        gamma, beta, iterations
    """
    rng = np.random.default_rng(seed)
    backend = StatevectorBackend()

    x0 = rng.uniform(0, math.pi, size=2)
    res = minimize(
        _qaoa_energy,
        x0,
        args=(backend,),
        method="COBYLA",
        options={"maxiter": 300, "rhobeg": 0.5},
    )

    gamma, beta = float(res.x[0]), float(res.x[1])
    qaoa_energy = -float(res.fun)  # restore sign

    # Sample the optimised circuit
    qc = _build_qaoa_circuit(gamma, beta, _EDGES, _NUM_NODES, p=1)
    sample_result = backend.run(qc, shots=2048, seed=seed)
    counts = sample_result.counts

    # Find best sampled bitstring (highest MaxCut value)
    best_bs = max(counts, key=lambda bs: _maxcut_value(bs, _EDGES))
    best_cut = _maxcut_value(best_bs, _EDGES)

    approx_ratio = qaoa_energy / _OPTIMAL_CUT

    return {
        "optimal_cut": _OPTIMAL_CUT,
        "qaoa_energy": qaoa_energy,
        "approximation_ratio": approx_ratio,
        "best_sampled_cut": best_cut,
        "best_sampled_bitstring": best_bs,
        "gamma": gamma,
        "beta": beta,
        "iterations": int(res.nfev),
        "counts": dict(sorted(counts.items(), key=lambda x: -x[1])[:8]),
    }
