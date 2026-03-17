"""Demo: Bell state entanglement."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from quantum_kernel.provenance import ReproducibilityBlock
from benchmarks.metrics import state_fidelity

qc = QuantumCircuit(2)
qc.h(0).cnot(0, 1)

print("Circuit:")
print(qc.draw())

backend = StatevectorBackend()
result = backend.run(qc, shots=1024, seed=42)

expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
fidelity = state_fidelity(result.statevector, expected)

print(f"\nStatevector: {np.round(result.statevector, 4)}")
print(f"Fidelity:    {fidelity:.6f}")
print(f"\nMeasurement counts:")
for bs, count in sorted(result.counts.items()):
    bar = "█" * (count * 40 // 1024)
    print(f"  |{bs}⟩: {count:4d} ({count/1024*100:5.1f}%)  {bar}")

prov = ReproducibilityBlock.from_run(qc, "statevector", "dense_statevector", 1024, 42)
print(f"\n[Provenance] {prov}")
