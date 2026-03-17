#!/usr/bin/env python3
"""
Demo: Bell State — Entanglement Demonstration
==============================================
Creates the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 and demonstrates:
- Superposition via Hadamard gate
- Entanglement via CNOT gate
- Correlated measurement outcomes

Expected output: ~50% |00⟩ and ~50% |11⟩, never |01⟩ or |10⟩ (ideal).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from quantum_kernel.provenance import ReproducibilityBlock
from benchmarks.metrics import state_fidelity

import numpy as np


def main():
    print("=" * 60)
    print("  Quantum Entanglement Demo: Bell State |Φ+⟩")
    print("=" * 60)

    # Build Bell state circuit: H on qubit 0, then CNOT(0→1)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cnot(0, 1)

    print(f"\nCircuit:\n{qc.draw()}")
    print(f"\nCircuit hash: {qc.circuit_hash()[:16]}...")

    # Run on statevector backend
    backend = StatevectorBackend()
    shots = 1024
    seed = 42
    result = backend.run(qc, shots=shots, seed=seed)

    # Expected Bell state: (|00⟩ + |11⟩) / √2
    expected_sv = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    fidelity = state_fidelity(result.statevector, expected_sv)

    print(f"\n── Results ──")
    print(f"Statevector: {np.round(result.statevector, 4)}")
    print(f"Expected:    {np.round(expected_sv, 4)}")
    print(f"Fidelity:    {fidelity:.6f}")
    print(f"\nMeasurement counts ({shots} shots):")
    for bitstring, count in sorted(result.counts.items()):
        bar = "█" * (count * 40 // shots)
        print(f"  |{bitstring}⟩: {count:4d} ({count/shots*100:5.1f}%)  {bar}")

    # Physics explanation
    print(f"\n── Physics ──")
    print(f"The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 is maximally entangled.")
    print(f"It cannot be written as a product of single-qubit states.")
    print(f"Measurements are perfectly correlated: both qubits always agree.")
    if result.counts.get("01", 0) == 0 and result.counts.get("10", 0) == 0:
        print(f"✓ Confirmed: no |01⟩ or |10⟩ outcomes observed.")
    else:
        anti = result.counts.get("01", 0) + result.counts.get("10", 0)
        print(f"⚠ {anti} anti-correlated outcomes (noise or finite statistics).")

    # Reproducibility
    prov = ReproducibilityBlock.from_run(
        circuit=qc, backend_name="statevector",
        method="dense_statevector", shots=shots, seed=seed,
    )
    print(f"\n── Reproducibility ──")
    print(f"Backend:  {prov.backend_name} ({prov.simulation_method})")
    print(f"Shots:    {prov.shots}")
    print(f"Seed:     {prov.seed}")
    print(f"Circuit:  {prov.circuit_hash[:16]}...")
    print(f"Python:   {prov.python_version.split()[0]}")
    print(f"NumPy:    {prov.numpy_version}")
    print(f"Label:    {prov.execution_label}")


if __name__ == "__main__":
    main()
