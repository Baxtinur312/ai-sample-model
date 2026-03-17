"""Demo: VQE for H₂ ground-state energy."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.vqe import run_vqe_benchmark

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
