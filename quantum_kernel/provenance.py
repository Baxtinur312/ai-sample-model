"""
Reproducibility and provenance tracking.

Every simulation run should be wrapped in a ReproducibilityBlock so that
results are fully auditable and re-runnable.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit


@dataclass
class ReproducibilityBlock:
    """
    Immutable record linking a simulation run to its configuration.

    Attributes
    ----------
    circuit_digest : str
        SHA-256 fingerprint of the gate sequence.
    backend : str
        Backend name (e.g. ``"statevector"``).
    method : str
        Simulation method (e.g. ``"dense_statevector"``).
    shots : int
    seed : int | None
    timestamp : str
        ISO-8601 UTC timestamp of when the block was created.
    python_version : str
    numpy_version : str
    label : str
        Always ``"ideal_simulation"`` or ``"noisy_simulation"``.
    """

    circuit_digest: str
    backend: str
    method: str
    shots: int
    seed: Optional[int]
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"))
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    numpy_version: str = field(default_factory=lambda: np.__version__)
    label: str = "ideal_simulation"

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        circuit: QuantumCircuit,
        backend: str,
        method: str,
        shots: int,
        seed: Optional[int],
        label: str = "ideal_simulation",
    ) -> "ReproducibilityBlock":
        """Create a block from a completed simulation run."""
        return cls(
            circuit_digest=circuit.digest(),
            backend=backend,
            method=method,
            shots=shots,
            seed=seed,
            label=label,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "circuit_digest": self.circuit_digest,
            "backend": self.backend,
            "method": self.method,
            "shots": self.shots,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "label": self.label,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ReproducibilityBlock":
        return cls(**d)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"ReproducibilityBlock("
            f"digest={self.circuit_digest}, "
            f"backend={self.backend}/{self.method}, "
            f"shots={self.shots}, seed={self.seed}, "
            f"ts={self.timestamp}, label={self.label})"
        )

    def __repr__(self) -> str:
        return self.__str__()
