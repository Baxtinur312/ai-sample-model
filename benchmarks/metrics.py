"""
Evaluation Metrics
Quantum fidelity metrics and distributional distance measures
for benchmarking quantum simulation accuracy.
Evaluation metrics for quantum state and distribution comparison.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence
from typing import Dict

import numpy as np


# ============================================================================
# State-level metrics (require statevector access)
# ============================================================================

def state_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """
    Compute the fidelity between two pure states: F = |⟨ψ|φ⟩|².

    Parameters
    ----------
    sv1, sv2 : np.ndarray
        Complex statevectors (must be same length).

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    sv1 = np.asarray(sv1, dtype=complex).ravel()
    sv2 = np.asarray(sv2, dtype=complex).ravel()
    if len(sv1) != len(sv2):
        raise ValueError(
            f"Statevector lengths differ: {len(sv1)} vs {len(sv2)}"
        )
    return float(abs(np.vdot(sv1, sv2)) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the trace distance ½‖ρ − σ‖₁ between two density matrices.

    For pure states |ψ⟩ and |φ⟩ this equals √(1 − F) where F is the fidelity.
    """
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def density_matrix_fidelity(
    rho: np.ndarray, sigma: np.ndarray
) -> float:
    """
    Fidelity between two density matrices:
    F(ρ, σ) = (Tr √(√ρ σ √ρ))².

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    from scipy.linalg import sqrtm

    sqrt_rho = sqrtm(rho)
    product = sqrt_rho @ sigma @ sqrt_rho
    sqrt_product = sqrtm(product)
    fidelity = np.real(np.trace(sqrt_product)) ** 2
    return float(np.clip(fidelity, 0.0, 1.0))


# ============================================================================
# Distribution-level metrics (work with measurement counts)
# ============================================================================

def total_variation_distance(
    p: Dict[str, float] | np.ndarray,
    q: Dict[str, float] | np.ndarray,
) -> float:
    """
    Total Variation Distance: TVD(p, q) = ½ Σ|p_i − q_i|.

    Accepts either:
    - Dict[str, float]: probability distributions as {bitstring: probability}
    - np.ndarray: probability vectors (same length)

    Returns
    -------
    float
        TVD in [0, 1].
    """
    if isinstance(p, dict) and isinstance(q, dict):
        all_keys = set(p.keys()) | set(q.keys())
        return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in all_keys)
    else:
        p_arr = np.asarray(p, dtype=float).ravel()
        q_arr = np.asarray(q, dtype=float).ravel()
        if len(p_arr) != len(q_arr):
            raise ValueError(
                f"Probability vector lengths differ: {len(p_arr)} vs {len(q_arr)}"
            )
        return float(0.5 * np.sum(np.abs(p_arr - q_arr)))


def counts_to_probabilities(
    counts: Dict[str, int], num_qubits: Optional[int] = None
) -> Dict[str, float]:
    """
    Convert measurement counts to a probability distribution.

    Parameters
    ----------
    counts : dict
        Measurement counts {bitstring: count}.
    num_qubits : int, optional
        If given, ensures all 2^n bitstrings are represented.
    """
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    if num_qubits is not None:
        for i in range(2 ** num_qubits):
            key = format(i, f"0{num_qubits}b")
            if key not in probs:
                probs[key] = 0.0

    return probs


# ============================================================================
# Algorithm-specific metrics
# ============================================================================

def success_probability(
    counts: Dict[str, int],
    target_states: Sequence[str],
) -> float:
    """
    Compute the success probability: fraction of measurements landing on
    target states.

    Parameters
    ----------
    counts : dict
        Measurement counts.
    target_states : list of str
        Bitstrings that count as "success."

    Returns
    -------
    float
        Success probability in [0, 1].
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    success = sum(counts.get(s, 0) for s in target_states)
    return success / total


def energy_error(estimated: float, exact: float) -> float:
    """
    Absolute energy error.

    Parameters
    ----------
    estimated : float
        Estimated energy (e.g., from VQE).
    exact : float
        Exact ground-state energy.

    Returns
    -------
    float
        |estimated − exact|
    """
    return abs(estimated - exact)


def approximation_ratio(
    achieved_value: float, optimal_value: float
) -> float:
    """
    Approximation ratio for optimization problems (e.g., QAOA/MaxCut).

    Returns achieved / optimal (higher is better, max 1.0).
    """
    if optimal_value == 0:
        return 1.0 if achieved_value == 0 else 0.0
    return achieved_value / optimal_value
    eigenvalues = np.linalg.eigvalsh((diff + diff.conj().T) / 2)
    return 0.5 * float(np.sum(np.abs(eigenvalues)))



def counts_to_probs(counts: Dict[str, int]) -> Dict[str, float]:
    """Normalise a counts dictionary to probabilities."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def hellinger_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute the Hellinger fidelity (Bhattacharyya coefficient) between two
    probability distributions.

    Returns
    -------
    float in [0, 1] — 1 means identical distributions.
    """
    keys = set(p) | set(q)
    return float(sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys))
