
import sys
import os
import json

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_bell_demo():
    """Run the Bell state entanglement demo."""
    from quantum_kernel.circuit_ir import QuantumCircuit
    from quantum_kernel.backends.statevector import StatevectorBackend
    from quantum_kernel.provenance import ReproducibilityBlock
    from benchmarks.metrics import state_fidelity
    import numpy as np

    print("\n" + "=" * 60)
    print("  🔬 Bell State — Entanglement Demonstration")
    print("=" * 60)

    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)

    print(f"\nCircuit ({qc}):")
    print(qc.draw())

    backend = StatevectorBackend()
    result = backend.run(qc, shots=1024, seed=42)

    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    fidelity = state_fidelity(result.statevector, expected)

    print(f"\nStatevector: {np.round(result.statevector, 4)}")
    print(f"Fidelity:    {fidelity:.6f}")
    print(f"\nMeasurement counts (1024 shots):")
    for bs, count in sorted(result.counts.items()):
        bar = "█" * (count * 40 // 1024)
        print(f"  |{bs}⟩: {count:4d} ({count/1024*100:5.1f}%)  {bar}")

    prov = ReproducibilityBlock.from_run(qc, "statevector", "dense_statevector", 1024, 42)
    print(f"\n[Provenance] {prov}")
    return fidelity > 0.999


def run_grover_demo():
    """Run Grover's search algorithm demo."""
    from benchmarks.grover import run_grover_benchmark

    print("\n" + "=" * 60)
    print("  🔍 Grover's Search Algorithm")
    print("=" * 60)

    results = []
    for n in [2, 3, 4]:
        N = 2 ** n
        r = run_grover_benchmark(num_qubits=n, shots=2048, seed=42)
        results.append(r)
        print(f"\n  {n} qubits (N={N}): "
              f"measured={r['measured_success_prob']:.3f}, "
              f"theory={r['theoretical_success_prob']:.3f}, "
              f"error={r['prob_error']:.3f}")
        top = sorted(r["top_counts"].items(), key=lambda x: -x[1])[:3]
        for bs, count in top:
            print(f"    |{bs}⟩: {count}")

    return all(r["measured_success_prob"] > 0.5 for r in results)


def run_vqe_demo():
    """Run VQE for H₂ molecule demo."""
    from benchmarks.vqe import run_vqe_benchmark

    print("\n" + "=" * 60)
    print("  ⚛️  VQE — H₂ Ground-State Energy")
    print("=" * 60)

    r = run_vqe_benchmark(seed=42)
    print(f"\n  Exact energy:  {r['exact_ground_energy']:.6f} Ha")
    print(f"  VQE energy:    {r['vqe_energy']:.6f} Ha")
    print(f"  Energy error:  {r['energy_error']:.6f} Ha")
    print(f"  Fidelity:      {r['state_fidelity']:.6f}")
    print(f"  Iterations:    {r['iterations']}")

    if r.get("chemical_accuracy"):
        print(f"  ✓ Chemical accuracy achieved!")
    else:
        print(f"  ⚠ Chemical accuracy not reached")

    return r["energy_error"] < 0.1


def run_qaoa_demo():
    """Run QAOA MaxCut demo."""
    from benchmarks.qaoa import run_qaoa_benchmark

    print("\n" + "=" * 60)
    print("  📊 QAOA — MaxCut Optimization")
    print("=" * 60)

    r = run_qaoa_benchmark(seed=42)
    print(f"\n  Graph:              4-node cycle")
    print(f"  Optimal cut:        {r['optimal_cut']}")
    print(f"  QAOA energy:        {r['qaoa_energy']:.4f}")
    print(f"  Approx ratio:       {r['approximation_ratio']:.4f}")
    print(f"  Best sampled cut:   {r['best_sampled_cut']}")
    print(f"  Best bitstring:     |{r['best_sampled_bitstring']}⟩")

    return r["approximation_ratio"] > 0.5


def run_all_benchmarks():
    """Run all benchmarks and produce a summary."""
    print("\n" + "=" * 60)
    print("  📋 Full Benchmark Suite")
    print("=" * 60)

    results = {}

    print("\n▶ Running Grover benchmark...")
    from benchmarks.grover import run_grover_benchmark
    results["grover"] = run_grover_benchmark(num_qubits=3, shots=2048, seed=42)

    print("▶ Running QFT benchmark...")
    from benchmarks.qft import run_qft_benchmark
    results["qft"] = run_qft_benchmark(num_qubits=3, shots=1024, seed=42)

    print("▶ Running VQE benchmark...")
    from benchmarks.vqe import run_vqe_benchmark
    results["vqe"] = run_vqe_benchmark(seed=42)

    print("▶ Running QAOA benchmark...")
    from benchmarks.qaoa import run_qaoa_benchmark
    results["qaoa"] = run_qaoa_benchmark(seed=42)

    print("\n" + "─" * 60)
    print("  Benchmark Results Summary")
    print("─" * 60)
    print(f"  {'Benchmark':<12} {'Metric':<25} {'Value':<12} {'Pass'}")
    print(f"  {'─'*12} {'─'*25} {'─'*12} {'─'*6}")

    g = results["grover"]
    print(f"  {'Grover':<12} {'Success probability':<25} {g['measured_success_prob']:<12.4f} {'✓' if g['measured_success_prob'] > 0.7 else '✗'}")

    q = results["qft"]
    qft_fid = q["tests"][0]["fidelity"] if q["tests"] else 0
    print(f"  {'QFT':<12} {'State fidelity (|0⟩)':<25} {qft_fid:<12.4f} {'✓' if qft_fid > 0.99 else '✗'}")

    v = results["vqe"]
    print(f"  {'VQE':<12} {'Energy error (Ha)':<25} {v['energy_error']:<12.6f} {'✓' if v['energy_error'] < 0.1 else '✗'}")

    a = results["qaoa"]
    print(f"  {'QAOA':<12} {'Approximation ratio':<25} {a['approximation_ratio']:<12.4f} {'✓' if a['approximation_ratio'] > 0.5 else '✗'}")

    print()
    return results


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        Quantum-Like AI System v0.1.0                    ║")
    print("║        Classical Emulation of Quantum Circuits          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  ⚠ This is NOT a quantum computer.                     ║")
    print("║  ⚠ Outputs are labeled: ideal_simulation               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    command = args[0].lower()

    if command == "bell":
        run_bell_demo()
    elif command == "grover":
        run_grover_demo()
    elif command == "vqe":
        run_vqe_demo()
    elif command == "qaoa":
        run_qaoa_demo()
    elif command == "benchmarks":
        run_all_benchmarks()
    elif command == "all":
        passed = []
        passed.append(("Bell State", run_bell_demo()))
        passed.append(("Grover", run_grover_demo()))
        passed.append(("VQE", run_vqe_demo()))
        passed.append(("QAOA", run_qaoa_demo()))

        print("\n" + "=" * 60)
        print("  Final Summary")
        print("=" * 60)
        for name, ok in passed:
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"  {name:<20} {status}")

        run_all_benchmarks()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python ai.sample.py [bell|grover|vqe|qaoa|benchmarks|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
