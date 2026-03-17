# Quantum-Like AI System

A modular Python system that faithfully emulates quantum circuit behavior using classical simulation, provides LLM prompt templates for a "Quantum Emulation Agent," and includes benchmark suites for canonical quantum algorithms.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│             AI Orchestration Layer (B)               │
│  ┌──────────┐  ┌─────────────────┐  ┌────────────┐  │
│  │  Prompt   │  │Backend Selector │  │   Agent    │  │
│  │ Templates │  │(auto-fallback)  │  │(R→A→V→E)  │  │
│  └──────────┘  └─────────────────┘  └────────────┘  │
├─────────────────────────────────────────────────────┤
│           Quantum Behavior Kernel (A)                │
│  ┌──────────┐  ┌──────────────────────────────────┐  │
│  │Circuit IR│  │         Backends                 │  │
│  │(gate lib │  │  ┌────────────┐ ┌────────────┐   │  │
│  │ hashing  │  │  │Statevector │ │ Stabilizer │   │  │
│  │ drawing) │  │  │  (exact)   │ │ (Clifford) │   │  │
│  └──────────┘  │  └────────────┘ └────────────┘   │  │
│  ┌──────────┐  │  ┌────────────┐                  │  │
│  │  Noise   │  │  │    MPS     │                  │  │
│  │  Model   │  │  │(tensor net)│                  │  │
│  └──────────┘  │  └────────────┘                  │  │
│  ┌──────────┐  └──────────────────────────────────┘  │
│  │Provenance│                                        │
│  └──────────┘                                        │
├─────────────────────────────────────────────────────┤
│              Benchmarks & Evaluation                 │
│   Grover │ QFT │ VQE │ QAOA │ Metrics               │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Demos

```bash
# Bell state (entanglement)
python examples/demo_bell_state.py

# Grover's search
python examples/demo_grover.py

# VQE for H₂ molecule
python examples/demo_vqe.py
```

### Run Benchmarks Programmatically

```python
from benchmarks.grover import run_grover_benchmark
from benchmarks.qft import run_qft_benchmark
from benchmarks.vqe import run_vqe_benchmark
from benchmarks.qaoa import run_qaoa_benchmark

# Grover (3 qubits, searching for |111⟩)
print(run_grover_benchmark(num_qubits=3))

# QFT (3 qubits, verified against DFT)
print(run_qft_benchmark(num_qubits=3))

# VQE (H₂ ground state)
print(run_vqe_benchmark())

# QAOA (MaxCut on 4-node cycle)
print(run_qaoa_benchmark())
```

## Simulation Backends

| Backend | Method | Qubits | Gate Support | State Access |
|---------|--------|--------|-------------|-------------|
| `StatevectorBackend` | Dense statevector | ≤ 20 | All gates | Full |
| `StabilizerBackend` | Clifford tableau | ≤ 5000 | Clifford only | None |
| `MPSBackend` | Matrix Product State | ≤ 100 | 1-2 qubit gates | Full (small) |

## Prompt Templates

The `orchestrator/prompts.py` module provides:
- **Tier A** — Tool-verified quantum emulation (recommended for production)
- **Tier B** — Textual amplitude-ledger reasoning (pedagogical only)

## Evaluation Metrics

- **State fidelity** — |⟨ψ|φ⟩|²
- **Trace distance** — ½‖ρ−σ‖₁
- **Total variation distance** — ½Σ|p−q|
- **Success probability** — for algorithm benchmarks
- **Energy error** — |E_estimated − E_exact|
- **Approximation ratio** — for optimization (QAOA)

## ⚠️ Important Disclaimers

- **This is NOT a quantum computer.** It is a classical emulation of the quantum circuit model.
- **Simulator outputs do not prove anything about nature** — only real quantum hardware can.
- All outputs are labeled as `ideal_simulation` or `noisy_simulation`.
- Noise models are **simulated, not calibration-derived** from real hardware.
- Every run includes a reproducibility block for auditability.

## License

MIT
# ai-sample-model
