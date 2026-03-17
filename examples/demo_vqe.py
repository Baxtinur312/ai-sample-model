#!/usr/bin/env python3
"""
Demo: Variational Quantum Eigensolver (VQE) for H₂
Runs VQE to estimate the ground-state energy of H₂ (hydrogen molecule)
using a 2-qubit model with a hardware-efficient ansatz.
Compares to exact diagonalization.
"""

"""Demo: VQE for H₂ ground-state energy."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.vqe import run_vqe_benchmark


def main():
    print("=" * 60)
    print("  VQE Benchmark: H₂ Ground-State Energy")
    print("=" * 60)

    result = run_vqe_benchmark(seed=42)

    print(f"\n── Configuration ──")
    print(f"Hamiltonian:    {result['hamiltonian']}")
    print(f"Qubits:         {result['num_qubits']}")
    print(f"Ansatz depth:   {result['ansatz_depth']}")
    print(f"Parameters:     {result['num_params']}")
    print(f"Optimizer:      {result['optimizer']}")

    print(f"\n── Results ──")
    print(f"Exact ground energy:  {result['exact_ground_energy']:.6f} Ha")
    print(f"VQE energy:           {result['vqe_energy']:.6f} Ha")
    print(f"Energy error:         {result['energy_error']:.6f} Ha")
    print(f"State fidelity:       {result['state_fidelity']:.6f}")
    print(f"Iterations:           {result['iterations']}")
    print(f"Converged:            {result['converged']}")

    if result.get("chemical_accuracy"):
        print(f"\n✓ Chemical accuracy achieved (error < 1.6 mHa = 1 kcal/mol)")
    else:
        print(f"\n⚠ Chemical accuracy NOT achieved (error ≥ 1.6 mHa)")

    print(f"\n── Physics ──")
    print(f"VQE uses a hybrid quantum-classical loop:")
    print(f"  1. Quantum circuit prepares trial state |ψ(θ)⟩")
    print(f"  2. Measure ⟨ψ(θ)|H|ψ(θ)⟩")
    print(f"  3. Classical optimizer updates θ to minimize energy")
    print(f"  4. Repeat until convergence")
    print(f"This approach requires shorter quantum coherence times")
    print(f"compared to quantum phase estimation.")


if __name__ == "__main__":
    main()
r = run_vqe_benchmark(seed=42)
print(f"Exact energy:  {r['exact_ground_energy']:.6f} Ha")
print(f"VQE energy:    {r['vqe_energy']:.6f} Ha")
print(f"Energy error:  {r['energy_error']:.6f} Ha")
print(f"Fidelity:      {r['state_fidelity']:.6f}")
print(f"Iterations:    {r['iterations']}")
if r.get("chemical_accuracy"):
    print("✓ Chemical accuracy achieved!")
else:
    print("⚠ Chemical accuracy not reached")
