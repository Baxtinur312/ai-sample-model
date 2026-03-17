"""
Noise Model Module
==================
Simulated noise channels for quantum circuits.
Supports depolarizing, bit-flip, and phase-flip channels.

IMPORTANT: These are simulated noise models, NOT calibration-derived
from real hardware. They provide approximate behavioral modeling only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit, QuantumGate


@dataclass
class NoiseModel:
    """
    A noise model that can be applied to quantum circuits.

    Noise is modeled as probabilistic Pauli errors inserted after each gate.
    This is a simplified (but widely used) noise abstraction.

    Attributes
    ----------
    depolarizing_rate : float
        Probability of a single-qubit depolarizing error per gate.
    bit_flip_rate : float
        Probability of a bit-flip (X) error per gate.
    phase_flip_rate : float
        Probability of a phase-flip (Z) error per gate.
    two_qubit_error_factor : float
        Multiplier for error rates on two-qubit gates (typically 2-10x).
    measurement_error_rate : float
        Probability of a measurement readout error.
    """

    depolarizing_rate: float = 0.001
    bit_flip_rate: float = 0.0005
    phase_flip_rate: float = 0.0005
    two_qubit_error_factor: float = 5.0
    measurement_error_rate: float = 0.01

    def describe(self) -> str:
        return (
            f"NoiseModel(depol={self.depolarizing_rate}, "
            f"bit_flip={self.bit_flip_rate}, "
            f"phase_flip={self.phase_flip_rate}, "
            f"2q_factor={self.two_qubit_error_factor}, "
            f"meas_err={self.measurement_error_rate}) "
            f"[SIMULATED — not calibration-derived]"
        )


def apply_noise_to_counts(
    counts: Dict[str, int],
    noise_model: NoiseModel,
    num_qubits: int,
    rng: np.random.Generator,
) -> Dict[str, int]:
    """
    Apply measurement noise to a counts dictionary.
    Each measured bit has a probability of being flipped.
    """
    if noise_model.measurement_error_rate <= 0:
        return counts

    noisy_counts: Dict[str, int] = {}
    for bitstring, count in counts.items():
        for _ in range(count):
            bits = list(bitstring)
            for i in range(len(bits)):
                if rng.random() < noise_model.measurement_error_rate:
                    bits[i] = "1" if bits[i] == "0" else "0"
            noisy_bs = "".join(bits)
            noisy_counts[noisy_bs] = noisy_counts.get(noisy_bs, 0) + 1
    return noisy_counts


def inject_gate_noise(
    statevector: np.ndarray,
    gate: QuantumGate,
    noise_model: NoiseModel,
    num_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply probabilistic Pauli noise after a gate operation.

    For each qubit involved in the gate, with probability p, apply a
    random Pauli error (X, Y, or Z with equal probability for depolarizing).
    """
    factor = (noise_model.two_qubit_error_factor
              if gate.num_qubits >= 2 else 1.0)

    for qubit in gate.qubits:
        # Depolarizing noise
        if rng.random() < noise_model.depolarizing_rate * factor:
            pauli = rng.choice(["X", "Y", "Z"])
            statevector = _apply_pauli(statevector, pauli, qubit, num_qubits)

        # Bit-flip noise
        if rng.random() < noise_model.bit_flip_rate * factor:
            statevector = _apply_pauli(statevector, "X", qubit, num_qubits)

        # Phase-flip noise
        if rng.random() < noise_model.phase_flip_rate * factor:
            statevector = _apply_pauli(statevector, "Z", qubit, num_qubits)

    return statevector


def _apply_pauli(
    sv: np.ndarray, pauli: str, qubit: int, n: int
) -> np.ndarray:
    """Apply a single Pauli operator to the statevector."""
    sv = sv.reshape([2] * n)

    if pauli == "X":
        # Swap the 0 and 1 amplitudes along this qubit's axis
        indices_0 = [slice(None)] * n
        indices_1 = [slice(None)] * n
        indices_0[qubit] = 0
        indices_1[qubit] = 1
        sv_copy = sv.copy()
        sv[tuple(indices_0)] = sv_copy[tuple(indices_1)]
        sv[tuple(indices_1)] = sv_copy[tuple(indices_0)]

    elif pauli == "Z":
        # Negate the |1⟩ amplitudes along this qubit's axis
        indices_1 = [slice(None)] * n
        indices_1[qubit] = 1
        sv[tuple(indices_1)] *= -1

    elif pauli == "Y":
        # Y = iXZ
        indices_0 = [slice(None)] * n
        indices_1 = [slice(None)] * n
        indices_0[qubit] = 0
        indices_1[qubit] = 1
        sv_copy = sv.copy()
        sv[tuple(indices_0)] = -1j * sv_copy[tuple(indices_1)]
        sv[tuple(indices_1)] = 1j * sv_copy[tuple(indices_0)]

    return sv.reshape(-1)
