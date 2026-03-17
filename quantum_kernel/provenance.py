"""
Reproducibility & Provenance Module
=====================================
Tracks all parameters needed to reproduce a quantum simulation run.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ReproducibilityBlock:
    """
    Immutable record of all parameters for a simulation run.
    Required in every output per the system architecture spec.
    """

    # Circuit identity
    circuit_hash: str = ""
    num_qubits: int = 0
    num_gates: int = 0
    circuit_depth: int = 0

    # Execution config
    backend_name: str = ""
    simulation_method: str = ""
    shots: int = 0
    seed: Optional[int] = None

    # Noise (if any)
    noise_model: Optional[str] = None

    # Environment
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(
        default_factory=lambda: f"{platform.system()} {platform.release()}"
    )
    numpy_version: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Execution label
    execution_label: str = "ideal_simulation"

    def __post_init__(self):
        try:
            import numpy as np
            self.numpy_version = np.__version__
        except ImportError:
            self.numpy_version = "not installed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str | Path) -> None:
        """Save the reproducibility block as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_run(
        cls,
        circuit,
        backend_name: str,
        method: str,
        shots: int,
        seed: Optional[int] = None,
        noise_model=None,
        execution_label: str = "ideal_simulation",
    ) -> "ReproducibilityBlock":
        """Create a ReproducibilityBlock from a circuit and run parameters."""
        return cls(
            circuit_hash=circuit.circuit_hash(),
            num_qubits=circuit.num_qubits,
            num_gates=len(circuit.gates),
            circuit_depth=circuit.depth,
            backend_name=backend_name,
            simulation_method=method,
            shots=shots,
            seed=seed,
            noise_model=str(noise_model) if noise_model else None,
            execution_label=execution_label,
        )

    def __repr__(self) -> str:
        return (
            f"ReproducibilityBlock(backend={self.backend_name}, "
            f"method={self.simulation_method}, shots={self.shots}, "
            f"seed={self.seed}, qubits={self.num_qubits})"
        )


def save_experiment(
    result_data: Dict[str, Any],
    provenance: ReproducibilityBlock,
    output_dir: str = "./results",
    experiment_name: str = "experiment",
) -> Path:
    """
    Save an experiment's results and provenance as JSON artifacts.

    Returns the path to the saved result file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"

    artifact = {
        "experiment_name": experiment_name,
        "results": _make_serializable(result_data),
        "provenance": provenance.to_dict(),
    }

    filepath = out / filename
    filepath.write_text(
        json.dumps(artifact, indent=2, default=str), encoding="utf-8"
    )
    return filepath


def _make_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON types to serializable form."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj
