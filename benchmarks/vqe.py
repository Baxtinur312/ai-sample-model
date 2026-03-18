"""
Variational Quantum Eigensolver (VQE) Benchmark
Implements a simple VQE for the H₂ molecule (2-qubit minimal model).
Compares the hybrid optimization result to exact diagonalization.

VQE's core idea: parameterized quantum circuit + classical optimizer
minimizes ⟨ψ(θ)|H|ψ(θ)⟩ to find the ground-state energy.
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
from typing import Callable, Dict, List, Optional, Tuple
from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from benchmarks.metrics import energy_error


# ============================================================================
# Hamiltonians
# ============================================================================

def h2_hamiltonian() -> np.ndarray:
    """
    Minimal 2-qubit Hamiltonian for H₂ at equilibrium bond length.

    H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1

    Using standard coefficients (STO-3G basis, R=0.735 Å):
    """
    I = np.eye(4, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    ZI = np.kron(Z, np.eye(2))
    IZ = np.kron(np.eye(2), Z)
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)

    # Coefficients for H₂ (approximate, STO-3G at R ≈ 0.735 Å)
    g0 = -0.4804
    g1 = +0.3435
    g2 = -0.4347
    g3 = +0.5716
    g4 = +0.0910
    g5 = +0.0910

    H = g0 * I + g1 * ZI + g2 * IZ + g3 * ZZ + g4 * XX + g5 * YY
    return H


def exact_ground_state_energy(H: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute the exact ground-state energy and eigenvector via diagonalization.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = np.argmin(eigenvalues)
    return float(eigenvalues[idx]), eigenvectors[:, idx]


# ============================================================================
# VQE Ansatz
# ============================================================================

def build_vqe_ansatz(
    num_qubits: int, params: np.ndarray, depth: int = 1
) -> QuantumCircuit:
    """
    Build a hardware-efficient ansatz: layers of RY rotations + CNOT entanglers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    params : np.ndarray
        Variational parameters. Length = num_qubits * (depth + 1).
    depth : int
        Number of entangling layers.
    """
    qc = QuantumCircuit(num_qubits)
    param_idx = 0

    # Initial rotation layer
    for i in range(num_qubits):
        if param_idx < len(params):
            qc.ry(i, params[param_idx])
            param_idx += 1

    # Entangling layers
    for d in range(depth):
        # CNOT ladder
        for i in range(num_qubits - 1):
            qc.cnot(i, i + 1)
        # Rotation layer
        for i in range(num_qubits):
            if param_idx < len(params):
                qc.ry(i, params[param_idx])
                param_idx += 1

    return qc


# ============================================================================
# VQE Optimization
# ============================================================================

def vqe_energy(
    params: np.ndarray,
    hamiltonian: np.ndarray,
    num_qubits: int,
    ansatz_depth: int,
    backend: StatevectorBackend,
) -> float:
    """
    Compute ⟨ψ(θ)|H|ψ(θ)⟩ for given parameters.
    """
    qc = build_vqe_ansatz(num_qubits, params, ansatz_depth)
    result = backend.run(qc, shots=1, seed=0)  # Only need statevector
    sv = result.statevector
    energy = np.real(sv.conj() @ hamiltonian @ sv)
    return float(energy)


def run_vqe(
    hamiltonian: np.ndarray,
    num_qubits: int = 2,
    ansatz_depth: int = 1,
    max_iterations: int = 200,
    seed: int = 42,
) -> Dict:
    """
    Run the VQE optimization loop.

    Returns
    -------
    dict
        Optimization result including final energy, exact energy, and
        convergence history.
    """
    backend = StatevectorBackend()
    rng = np.random.default_rng(seed)

    # Number of parameters
    num_params = num_qubits * (ansatz_depth + 1)
    initial_params = rng.uniform(0, 2 * math.pi, num_params)

    # Track convergence
    energy_history: List[float] = []

    def objective(params):
        e = vqe_energy(params, hamiltonian, num_qubits, ansatz_depth, backend)
        energy_history.append(e)
        return e

    # Run optimizer
    opt_result = minimize(
        objective,
        initial_params,
        method="COBYLA",
        options={"maxiter": max_iterations, "rhobeg": 0.5},
    )

    # Exact comparison
    exact_energy, exact_state = exact_ground_state_energy(hamiltonian)
    final_energy = opt_result.fun

    # Final state fidelity
    final_qc = build_vqe_ansatz(num_qubits, opt_result.x, ansatz_depth)
    final_result = backend.run(final_qc, shots=1, seed=0)
    from benchmarks.metrics import state_fidelity
    fidelity = state_fidelity(final_result.statevector, exact_state)

    return {
        "algorithm": "VQE",
        "num_qubits": num_qubits,
        "ansatz_depth": ansatz_depth,
        "num_params": num_params,
        "exact_ground_energy": exact_energy,
        "vqe_energy": final_energy,
        "energy_error": energy_error(final_energy, exact_energy),
        "state_fidelity": fidelity,
        "optimizer": "COBYLA",
        "iterations": len(energy_history),
        "converged": opt_result.success,
        "energy_history_first_last": (
            [energy_history[0], energy_history[-1]] if energy_history else []
        ),
    }


def run_vqe_benchmark(
    shots: int = 1024,
    seed: int = 42,
) -> Dict:
    """
    Run the H₂ VQE benchmark.
    """
    H = h2_hamiltonian()
    result = run_vqe(H, num_qubits=2, ansatz_depth=2, seed=seed)

    result["benchmark"] = "H2_VQE"
    result["chemical_accuracy"] = result["energy_error"] < 0.0016  # 1 kcal/mol
    result["hamiltonian"] = "H2_STO3G_R0.735"

    return result
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
    qc.ry(0, float(params[0]))
    qc.ry(1, float(params[1]))
    qc.cnot(0, 1)
    qc.ry(0, float(params[2]))
    qc.ry(1, float(params[3]))
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
