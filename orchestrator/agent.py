"""
Quantum Emulation Agent
========================
Orchestrates the Reason → Act → Verify → Explain loop for quantum
circuit experiments. Uses the Quantum Behavior Kernel backends and
produces structured results with reproducibility blocks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.base import SimulationResult
from quantum_kernel.backends.statevector import StatevectorBackend
from quantum_kernel.backends.stabilizer import StabilizerBackend
from quantum_kernel.backends.mps import MPSBackend
from quantum_kernel.noise import NoiseModel, apply_noise_to_counts, inject_gate_noise
from quantum_kernel.provenance import ReproducibilityBlock, save_experiment
from orchestrator.backend_selector import select_backend, BackendRecommendation


@dataclass
class ExperimentResult:
    """Structured result from an orchestrated experiment."""

    phenomenon: str
    circuit_description: str
    physics_expectation: str
    results: Dict[str, Dict[str, Any]]  # mode → {counts, statevector, ...}
    comparison: Dict[str, Any]           # Cross-mode comparisons
    reproducibility: Dict[str, Any]      # Provenance blocks per mode
    backend_selection: Dict[str, str]    # Selection justification

    def summary(self) -> str:
        """Human-readable summary of the experiment."""
        lines = [
            f"═══ Quantum Experiment: {self.phenomenon} ═══",
            f"",
            f"Physics Expectation:",
            f"  {self.physics_expectation}",
            f"",
            f"Circuit: {self.circuit_description}",
            f"",
        ]
        for mode, data in self.results.items():
            lines.append(f"── {mode} ──")
            if "counts" in data:
                top = dict(sorted(
                    data["counts"].items(), key=lambda x: -x[1]
                )[:8])
                lines.append(f"  Top counts: {top}")
            if "energy" in data:
                lines.append(f"  Energy: {data['energy']:.6f}")
            lines.append("")

        if self.comparison:
            lines.append("── Comparison ──")
            for key, val in self.comparison.items():
                if isinstance(val, float):
                    lines.append(f"  {key}: {val:.6f}")
                else:
                    lines.append(f"  {key}: {val}")

        return "\n".join(lines)


class QuantumEmulationAgent:
    """
    Main orchestrator agent implementing the Reason → Act → Verify → Explain
    pattern for quantum experiments.
    """

    def __init__(
        self,
        default_shots: int = 1024,
        default_seed: int = 42,
        noise_model: Optional[NoiseModel] = None,
        output_dir: str = "./results",
    ):
        self.default_shots = default_shots
        self.default_seed = default_seed
        self.noise_model = noise_model
        self.output_dir = output_dir

        # Initialize backends
        self.backends = {
            "statevector": StatevectorBackend(),
            "stabilizer": StabilizerBackend(),
            "mps": MPSBackend(),
        }

    def run_experiment(
        self,
        circuit: QuantumCircuit,
        phenomenon: str = "general",
        physics_expectation: str = "",
        modes: Optional[List[str]] = None,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> ExperimentResult:
        """
        Run a quantum experiment across one or more execution modes.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit to execute.
        phenomenon : str
            What the circuit demonstrates (e.g., "entanglement").
        physics_expectation : str
            What should happen and why.
        modes : list of str
            Execution modes: "ideal", "noisy", or specific backend names.
        shots : int
            Number of measurement shots.
        seed : int
            Random seed for reproducibility.
        noise_model : NoiseModel
            Noise model for noisy execution (overrides default).
        """
        shots = shots or self.default_shots
        seed = seed or self.default_seed
        noise = noise_model or self.noise_model or NoiseModel()
        modes = modes or ["ideal"]

        # REASON: Select backend
        recommendation = select_backend(circuit)
        backend_name = recommendation.backend_name
        if backend_name == "none":
            raise RuntimeError(
                f"No suitable backend: {recommendation.justification}"
            )

        results: Dict[str, Dict[str, Any]] = {}
        provenance_blocks: Dict[str, Any] = {}

        for mode in modes:
            # ACT: Execute the circuit
            if mode == "ideal":
                result = self._run_ideal(circuit, backend_name, shots, seed)
            elif mode == "noisy":
                result = self._run_noisy(circuit, noise, shots, seed)
            elif mode in self.backends:
                result = self.backends[mode].run(circuit, shots, seed)
            else:
                raise ValueError(f"Unknown execution mode: {mode}")

            # Store results
            mode_data: Dict[str, Any] = {
                "counts": result.counts,
                "execution_label": result.execution_label,
                "metadata": result.metadata,
            }
            if result.statevector is not None:
                mode_data["statevector_norm"] = float(
                    np.linalg.norm(result.statevector)
                )
            if result.probabilities is not None:
                mode_data["top_probabilities"] = {
                    format(i, f"0{circuit.num_qubits}b"): float(p)
                    for i, p in enumerate(result.probabilities)
                    if p > 0.01
                }
            results[mode] = mode_data

            # VERIFY: Create provenance
            prov = ReproducibilityBlock.from_run(
                circuit=circuit,
                backend_name=backend_name,
                method=self.backends.get(backend_name, self.backends["statevector"]).method,
                shots=shots,
                seed=seed,
                noise_model=noise if mode == "noisy" else None,
                execution_label=result.execution_label,
            )
            provenance_blocks[mode] = prov.to_dict()

        # COMPARE: Cross-mode comparison
        comparison = self._compare_modes(results, circuit.num_qubits)

        experiment = ExperimentResult(
            phenomenon=phenomenon,
            circuit_description=str(circuit),
            physics_expectation=physics_expectation,
            results=results,
            comparison=comparison,
            reproducibility=provenance_blocks,
            backend_selection={
                "chosen": recommendation.backend_name,
                "method": recommendation.method,
                "justification": recommendation.justification,
            },
        )

        # Save artifact
        try:
            save_experiment(
                result_data={
                    "phenomenon": phenomenon,
                    "results": {
                        k: {kk: vv for kk, vv in v.items()
                             if kk != "statevector"}
                        for k, v in results.items()
                    },
                    "comparison": comparison,
                },
                provenance=ReproducibilityBlock.from_run(
                    circuit, backend_name,
                    self.backends.get(backend_name, self.backends["statevector"]).method,
                    shots, seed
                ),
                output_dir=self.output_dir,
                experiment_name=phenomenon,
            )
        except Exception:
            pass  # Don't fail experiment if saving fails

        return experiment

    def _run_ideal(
        self, circuit: QuantumCircuit, backend_name: str,
        shots: int, seed: int
    ) -> SimulationResult:
        """Run on the ideal (noiseless) backend."""
        backend = self.backends[backend_name]
        return backend.run(circuit, shots, seed)

    def _run_noisy(
        self, circuit: QuantumCircuit, noise: NoiseModel,
        shots: int, seed: int
    ) -> SimulationResult:
        """
        Run with noise applied.
        Uses statevector backend with probabilistic Pauli noise injection.
        """
        backend = self.backends["statevector"]
        # Validate
        issues = backend.validate_circuit(circuit)
        if issues:
            raise RuntimeError(
                "Cannot run noisy simulation:\n" + "\n".join(issues)
            )

        n = circuit.num_qubits
        dim = 2 ** n
        rng = np.random.default_rng(seed)

        # Initialize |00...0⟩
        sv = np.zeros(dim, dtype=complex)
        sv[0] = 1.0

        # Apply gates with noise
        for gate in circuit.gates:
            sv = backend._apply_gate(sv, gate, n)
            sv = inject_gate_noise(sv, gate, noise, n, rng)

        # Normalize (noise can slightly denormalize)
        norm = np.linalg.norm(sv)
        if norm > 1e-15:
            sv /= norm

        # Probabilities
        probs = np.abs(sv) ** 2
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        # Sample
        samples = rng.choice(dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for s in samples:
            bitstring = format(s, f"0{n}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        # Apply measurement noise
        counts = apply_noise_to_counts(counts, noise, n, rng)

        return SimulationResult(
            counts=counts,
            statevector=sv.copy(),
            probabilities=probs,
            metadata={
                "backend": "statevector",
                "method": "dense_statevector_noisy",
                "noise_model": noise.describe(),
                "num_qubits": n,
            },
            execution_label="noisy_simulation",
        )

    @staticmethod
    def _compare_modes(
        results: Dict[str, Dict[str, Any]], num_qubits: int
    ) -> Dict[str, Any]:
        """Compare results across execution modes using TVD."""
        comparison = {}
        mode_names = list(results.keys())

        if len(mode_names) < 2:
            return comparison

        for i in range(len(mode_names)):
            for j in range(i + 1, len(mode_names)):
                m1, m2 = mode_names[i], mode_names[j]
                c1 = results[m1]["counts"]
                c2 = results[m2]["counts"]

                # Total variation distance
                all_keys = set(c1.keys()) | set(c2.keys())
                total1 = sum(c1.values())
                total2 = sum(c2.values())
                tvd = 0.5 * sum(
                    abs(c1.get(k, 0) / total1 - c2.get(k, 0) / total2)
                    for k in all_keys
                )
                comparison[f"tvd_{m1}_vs_{m2}"] = tvd

        return comparison
