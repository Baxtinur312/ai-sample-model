"""Demo: Grover's search algorithm."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.grover import run_grover_benchmark

for n in [2, 3, 4]:
    r = run_grover_benchmark(num_qubits=n, shots=2048, seed=42)
    print(f"\n{n} qubits (N={r['N']}, target=|{r['target_bitstring']}⟩):")
    print(f"  Measured success prob:     {r['measured_success_prob']:.4f}")
    print(f"  Theoretical success prob:  {r['theoretical_success_prob']:.4f}")
    print(f"  Error:                     {r['prob_error']:.4f}")
    top = sorted(r["top_counts"].items(), key=lambda x: -x[1])[:3]
    for bs, count in top:
        print(f"  |{bs}⟩: {count}")
