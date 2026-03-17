"""
Variational Quantum Eigensolver (VQE) benchmark.

Finds the ground state of the H₂ molecule Hamiltonian using a classical
optimizer and a parameterised ansatz circuit. All arithmetic is classical
(no real QPU required).

H₂ Hamiltonian (STO-3G, Jordan-Wigner, equilibrium bond length ≈ 0.74 Å):

    H = g₀ I + g₁ Z₀ + g₂ Z₁ + g₃ Z₀Z₁ + g₄ Y₀Y₁ + g₅ X₀X₁

Coefficients from Kandala et al. (2017) / Qiskit textbook values:
    g₀ = -0.81054
    g₁ = +0.17218
    g₂ = -0.22575
    g₃ = +0.12091
    g₄ = -0.04523
    g₅ = -0.04523
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from benchmarks.metrics import state_fidelity


# ---------------------------------------------------------------------------
# H₂ Hamiltonian
# ---------------------------------------------------------------------------

_G = {
    "I":   -0.81054,
    "Z0":  +0.17218,
    "Z1":  -0.22575,
    "Z0Z1": +0.12091,
    "Y0Y1": -0.04523,
    "X0X1": -0.04523,
}

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _h2_matrix() -> np.ndarray:
    """Return the 4×4 H₂ Hamiltonian matrix."""
    kron = np.kron
    H = (
        _G["I"]    * np.eye(4, dtype=complex)
        + _G["Z0"]   * kron(_Z, _I2)
        + _G["Z1"]   * kron(_I2, _Z)
        + _G["Z0Z1"] * kron(_Z, _Z)
        + _G["Y0Y1"] * kron(_Y, _Y)
        + _G["X0X1"] * kron(_X, _X)
    )
    return H


# ---------------------------------------------------------------------------
# Ansatz circuit
# ---------------------------------------------------------------------------


def _ansatz(params: np.ndarray) -> QuantumCircuit:
    """
    Hardware-efficient ansatz for H₂.

    Two qubits, one Ry layer, CX entangler, second Ry layer.
    params : array of length 4 (θ₀, θ₁, θ₂, θ₃)
    """
    qc = QuantumCircuit(2)
    qc.ry(float(params[0]), 0)
    qc.ry(float(params[1]), 1)
    qc.cnot(0, 1)
    qc.ry(float(params[2]), 0)
    qc.ry(float(params[3]), 1)
    return qc


# ---------------------------------------------------------------------------
# Energy estimation
# ---------------------------------------------------------------------------


def _energy(params: np.ndarray, backend: StatevectorBackend) -> float:
    """Compute ⟨ψ(θ)|H|ψ(θ)⟩ exactly from the statevector."""
    qc = _ansatz(params)
    result = backend.run(qc, shots=65536, seed=0)
    sv = result.statevector
    H = _h2_matrix()
    e = float(np.real(sv.conj() @ H @ sv))
    return e


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------


def run_vqe_benchmark(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run VQE to find the H₂ ground-state energy.

    Returns
    -------
    dict with:
        exact_ground_energy, vqe_energy, energy_error,
        state_fidelity, iterations, chemical_accuracy, params
    """
    rng = np.random.default_rng(seed)
    backend = StatevectorBackend()

    # Exact diagonalisation
    H = _h2_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    exact_energy = float(eigenvalues[0])
    exact_gs = eigenvectors[:, 0]

    # VQE optimisation
    x0 = rng.uniform(-math.pi, math.pi, size=4)
    res = minimize(
        _energy,
        x0,
        args=(backend,),
        method="COBYLA",
        options={"maxiter": 500, "rhobeg": 0.5},
    )

    vqe_params = res.x
    vqe_energy = float(res.fun)
    error = abs(vqe_energy - exact_energy)

    # Compute fidelity of VQE state with exact ground state
    qc = _ansatz(vqe_params)
    sv_vqe = backend.run(qc, shots=65536, seed=0).statevector
    fid = state_fidelity(sv_vqe, exact_gs)

    return {
        "exact_ground_energy": exact_energy,
        "vqe_energy": vqe_energy,
        "energy_error": error,
        "state_fidelity": fid,
        "iterations": int(res.nfev),
        "chemical_accuracy": error < 1.6e-3,
        "params": vqe_params.tolist(),
    }
