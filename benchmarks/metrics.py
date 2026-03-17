"""
Evaluation metrics for quantum state and distribution comparison.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def state_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """
    Compute |⟨ψ₁|ψ₂⟩|² (pure-state fidelity).

    Parameters
    ----------
    sv1, sv2 : array-like
        Statevectors (need not be normalised — they will be normalised
        internally before computing the inner product).

    Returns
    -------
    float in [0, 1]
    """
    sv1 = np.asarray(sv1, dtype=complex)
    sv2 = np.asarray(sv2, dtype=complex)
    n1 = np.linalg.norm(sv1)
    n2 = np.linalg.norm(sv2)
    if n1 < 1e-14 or n2 < 1e-14:
        return 0.0
    inner = np.dot(sv1.conj() / n1, sv2 / n2)
    return float(np.real(inner * np.conj(inner)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the trace distance ½‖ρ − σ‖₁ between two density matrices.

    For pure states |ψ⟩ and |φ⟩ this equals √(1 − F) where F is the fidelity.
    """
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh((diff + diff.conj().T) / 2)
    return 0.5 * float(np.sum(np.abs(eigenvalues)))


def total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute the total variation distance ½ Σ|p(x) − q(x)|.

    Parameters
    ----------
    p, q : dict mapping bitstring → probability
    """
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


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
