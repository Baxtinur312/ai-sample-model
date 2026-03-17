"""
Noise models for quantum circuit simulation.

Provides depolarising, amplitude-damping, and phase-damping channels
that can be applied post-gate to statevector results via a density-matrix
approximation, as well as a simple probabilistic bit-flip layer for
stabilizer and MPS backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class NoiseModel:
    """
    Parameterised noise model.

    Parameters
    ----------
    depolarising_rate : float
        Single-qubit depolarising probability p per gate (0 ≤ p ≤ 1).
    t1 : float
        Amplitude-damping time constant (arbitrary units). 0 = no damping.
    t2 : float
        Dephasing time constant (arbitrary units). 0 = no dephasing.
    gate_time : float
        Duration of one gate cycle (same units as t1/t2). Default 1e-7.
    readout_error : float
        Probability of a readout bit-flip per qubit (0 ≤ p ≤ 1).
    """

    depolarising_rate: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    gate_time: float = 1e-7
    readout_error: float = 0.0
    label: str = "noisy_simulation"

    def __post_init__(self) -> None:
        for name, val in [
            ("depolarising_rate", self.depolarising_rate),
            ("readout_error", self.readout_error),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    @classmethod
    def ideal(cls) -> "NoiseModel":
        """Return a noiseless model (all parameters zero)."""
        return cls(label="ideal_simulation")

    @classmethod
    def ibmq_like(cls) -> "NoiseModel":
        """Return a rough IBMQ-like noise model (illustrative only)."""
        return cls(
            depolarising_rate=1e-3,
            t1=100e-6,
            t2=80e-6,
            gate_time=100e-9,
            readout_error=0.02,
            label="noisy_simulation (ibmq-like)",
        )

    # ------------------------------------------------------------------
    # Application helpers
    # ------------------------------------------------------------------

    def apply_readout_errors(
        self, counts: Dict[str, int], num_qubits: int, rng: np.random.Generator
    ) -> Dict[str, int]:
        """Flip each readout bit with probability `readout_error`."""
        if self.readout_error == 0.0:
            return counts
        noisy: Dict[str, int] = {}
        for bs, cnt in counts.items():
            for _ in range(cnt):
                bits = list(bs)
                for i in range(num_qubits):
                    if rng.random() < self.readout_error:
                        bits[i] = "1" if bits[i] == "0" else "0"
                key = "".join(bits)
                noisy[key] = noisy.get(key, 0) + 1
        return noisy

    def amplitude_damping_kraus(self, t: Optional[float] = None) -> list:
        """Amplitude-damping Kraus operators for a single qubit."""
        t = t or self.gate_time
        gamma = 1.0 - math.exp(-t / self.t1) if self.t1 > 0 else 0.0
        K0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex)
        return [K0, K1]

    def __repr__(self) -> str:
        return (
            f"NoiseModel(p_dep={self.depolarising_rate}, "
            f"T1={self.t1}, T2={self.t2}, "
            f"p_ro={self.readout_error})"
        )
