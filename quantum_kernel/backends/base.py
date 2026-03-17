"""
Abstract Backend Interface
==========================
All simulation backends implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit


@dataclass
class SimulationResult:
    """Result of a circuit simulation run."""

    counts: Dict[str, int]                         # Measurement outcome counts
    statevector: Optional[np.ndarray] = None       # Final statevector (if available)
    probabilities: Optional[np.ndarray] = None     # |amplitude|² distribution
    metadata: Dict = field(default_factory=dict)   # Backend-specific metadata
    execution_label: str = "ideal_simulation"      # "ideal_simulation" / "noisy_simulation" / "hardware"

    @property
    def total_shots(self) -> int:
        return sum(self.counts.values())

    def top_counts(self, n: int = 10) -> Dict[str, int]:
        """Return top-n most frequent measurement outcomes."""
        return dict(sorted(self.counts.items(), key=lambda x: -x[1])[:n])


class Backend(ABC):
    """Abstract base class for quantum simulation backends."""

    name: str = "base"
    method: str = "unknown"

    @abstractmethod
    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Execute a circuit and return sampled measurement results."""
        ...

    def validate_circuit(self, circuit: QuantumCircuit) -> List[str]:
        """
        Return a list of warnings/errors about this circuit for this backend.
        Empty list means the circuit is fully supported.
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method})"
