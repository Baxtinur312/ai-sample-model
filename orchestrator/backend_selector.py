"""
Backend Selector
================
Rule-based automatic selection of simulation backend based on circuit
properties. Implements the fallback chain from the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from quantum_kernel.circuit_ir import QuantumCircuit, CLIFFORD_GATES


@dataclass
class BackendRecommendation:
    """Result of backend selection."""
    backend_name: str
    method: str
    justification: str
    warnings: List[str]
    fallback_from: Optional[str] = None


def select_backend(
    circuit: QuantumCircuit,
    need_statevector: bool = False,
    noise_required: bool = False,
    performance_target: str = "balanced",
    max_statevector_qubits: int = 20,
    max_mps_qubits: int = 100,
) -> BackendRecommendation:
    """
    Select the best simulation backend for a given circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to simulate.
    need_statevector : bool
        Whether full statevector access is needed.
    noise_required : bool
        Whether noise simulation is required.
    performance_target : str
        One of "fast", "accurate", "balanced".
    max_statevector_qubits : int
        Maximum qubits for statevector simulation.
    max_mps_qubits : int
        Maximum qubits for MPS simulation.

    Returns
    -------
    BackendRecommendation
        The recommended backend with justification.
    """
    n = circuit.num_qubits
    is_clifford = circuit.is_clifford
    warnings = []

    # ------------------------------------------------------------------
    # Decision tree
    # ------------------------------------------------------------------

    # 1. If small enough for statevector and accuracy is needed → statevector
    if n <= max_statevector_qubits:
        if noise_required:
            warnings.append(
                "Noise is applied via probabilistic Pauli insertion on the "
                "statevector. For density-matrix noise, a dedicated backend "
                "would be needed."
            )
        return BackendRecommendation(
            backend_name="statevector",
            method="dense_statevector",
            justification=(
                f"Circuit has {n} qubits (≤ {max_statevector_qubits}), "
                f"which is within statevector memory limits. This provides "
                f"exact simulation with full state access."
            ),
            warnings=warnings,
        )

    # 2. If Clifford-only → stabilizer
    if is_clifford and not need_statevector:
        return BackendRecommendation(
            backend_name="stabilizer",
            method="clifford_tableau",
            justification=(
                f"Circuit has {n} qubits and uses only Clifford gates. "
                f"Stabilizer simulation is efficient (polynomial) for this "
                f"gate family."
            ),
            warnings=[
                "Stabilizer backend does not provide statevector access."
            ] if need_statevector else [],
            fallback_from="statevector" if n > max_statevector_qubits else None,
        )

    # 3. MPS for moderate circuits with structure
    if n <= max_mps_qubits:
        warnings.append(
            "MPS accuracy depends on circuit entanglement and bond dimension. "
            f"For {n} qubits, results may be approximate."
        )
        if need_statevector and n > 20:
            warnings.append(
                "Statevector extraction from MPS is only available for ≤20 qubits."
            )
        return BackendRecommendation(
            backend_name="mps",
            method="matrix_product_state",
            justification=(
                f"Circuit has {n} qubits with non-Clifford gates. "
                f"MPS provides approximate simulation with controllable "
                f"accuracy via bond dimension truncation."
            ),
            warnings=warnings,
            fallback_from="statevector",
        )

    # 4. Too large for all available backends
    return BackendRecommendation(
        backend_name="none",
        method="none",
        justification=(
            f"Circuit has {n} qubits, which exceeds all available backend "
            f"limits (statevector: {max_statevector_qubits}, "
            f"MPS: {max_mps_qubits}). Consider: "
            f"(1) reducing the problem size, "
            f"(2) using a Clifford approximation if applicable, or "
            f"(3) running on quantum hardware."
        ),
        warnings=[
            "No suitable classical simulation backend available.",
            "Consider hardware execution or problem-size reduction.",
        ],
    )


def estimate_entanglement(circuit: QuantumCircuit) -> str:
    """
    Heuristic entanglement estimate based on two-qubit gate density.

    Returns 'low', 'medium', or 'high'.
    """
    total_gates = len(circuit.gates)
    if total_gates == 0:
        return "low"
    two_qubit_gates = sum(1 for g in circuit.gates if g.num_qubits >= 2)
    ratio = two_qubit_gates / total_gates
    if ratio < 0.2:
        return "low"
    elif ratio < 0.5:
        return "medium"
    else:
        return "high"
