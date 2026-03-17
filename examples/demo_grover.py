#!/usr/bin/env python3
"""
Demo: Grover's Search Algorithm
Runs Grover's algorithm on a 3-qubit search space (8 items),
searching for a single marked state. Demonstrates amplitude
amplification and compares to theoretical predictions.
"""

"""Demo: Grover's search algorithm."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.grover import run_grover_benchmark


def main():
    print("=" * 60)
    print("  Grover's Search Algorithm Benchmark")
    print("=" * 60)

    for n_qubits in [2, 3, 4]:
        N = 2 ** n_qubits
        print(f"\n── {n_qubits} qubits, N={N}, searching for |{'1'*n_qubits}⟩ ──")

        result = run_grover_benchmark(
            num_qubits=n_qubits,
            marked_states=[N - 1],
            shots=2048,
            seed=42,
        )

        print(f"  Optimal iterations:       {result['optimal_iterations']}")
        print(f"  Theoretical success prob:  {result['theoretical_success_prob']:.4f}")
        print(f"  Measured success prob:     {result['measured_success_prob']:.4f}")
        print(f"  Error:                     {result['prob_error']:.4f}")
        print(f"  Circuit depth:             {result['circuit_depth']}")
        print(f"  Circuit gates:             {result['circuit_gates']}")
        print(f"  Top counts:")
        for bs, count in sorted(
            result["top_counts"].items(), key=lambda x: -x[1]
        )[:5]:
            bar = "█" * (count * 30 // 2048)
            print(f"    |{bs}⟩: {count:4d}  {bar}")

    print(f"\n── Summary ──")
    print(f"Grover's algorithm amplifies the marked state amplitude,")
    print(f"achieving ~O(√N) query complexity vs O(N) classical search.")


if __name__ == "__main__":
    main()
for n in [2, 3, 4]:
    r = run_grover_benchmark(num_qubits=n, shots=2048, seed=42)
    print(f"\n{n} qubits (N={r['N']}, target=|{r['target_bitstring']}⟩):")
    print(f"  Measured success prob:     {r['measured_success_prob']:.4f}")
    print(f"  Theoretical success prob:  {r['theoretical_success_prob']:.4f}")
    print(f"  Error:                     {r['prob_error']:.4f}")
    top = sorted(r["top_counts"].items(), key=lambda x: -x[1])[:3]
    for bs, count in top:
        print(f"  |{bs}⟩: {count}")
