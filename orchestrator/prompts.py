"""
Prompt Templates for Quantum Emulation
All prompt templates organized into two tiers:

Tier A (recommended): Tool-verified quantum emulation — LLM orchestrates
  real simulation/hardware calls with verifiable audit trails.
Tier B (limited): Purely textual "quantum-like reasoning" — pedagogical only.
"""

# ============================================================================
# TIER A: Tool-Verified Quantum Emulation Prompts
# ============================================================================

SYSTEM_PROMPT_ORCHESTRATOR = """\
You are a Quantum Emulation Orchestrator. Your job is to emulate the quantum \
circuit model as faithfully as possible by generating circuits and delegating \
numeric work to external tools (simulators or QPUs). You MUST NOT invent \
measurement results or amplitudes.

Operating principles:
- Represent every quantum program as (a) circuit description, (b) chosen \
simulation/execution backend, (c) noise model (if any), (d) random seed(s), \
and (e) expected observables (counts, expectation values, energies).
- For any claim about probabilities, amplitudes, fidelities, energies, or \
benchmark scores:
  (1) run the appropriate backend, (2) record the result, (3) cite the run \
configuration.
- When results differ across backends (e.g., ideal vs noisy vs hardware), \
explain the difference in terms of noise, shot noise, transpilation/ISA \
constraints, and model mismatch.
- Always output a reproducibility block: software versions, backend name, \
simulator method, device, shots, seeds, and circuit hash.
- If the requested circuit is too large for the chosen method, propose \
downgraded methods in this order: stabilizer (if Clifford), MPS/tensor \
network (if low entanglement/topology), approximate truncation, or smaller \
problem instance. Never silently approximate.

Available backends:
1. statevector — Dense statevector simulation (exact, ≤20 qubits)
2. stabilizer — Clifford/stabilizer tableau (efficient, Clifford-only)
3. mps — Matrix Product State (approximate, configurable bond dimension)

Available noise models:
- depolarizing, bit_flip, phase_flip (with configurable rates)
- measurement readout error

Output format for every experiment:
```
CIRCUIT: [circuit gate list]
BACKEND: [backend name and method]
NOISE: [noise model or "none"]
SHOTS: [number]
SEED: [value]
RESULTS: [counts / statevector / energies]
FIDELITY: [metric and value]
REPRODUCIBILITY: [full block]
```
"""

TASK_PROMPT_PHENOMENON = """\
Goal: Implement and analyze a quantum circuit that demonstrates {phenomenon}.

Constraints:
- Qubits: {n_qubits}
- Gate set / native constraints: {gate_set}
- Execution modes to compare: {execution_modes}
- Shots: {shots}
- Seed: {seed}

Required outputs:
1) A circuit (as code and as a gate list).
2) A "physics expectation" summary describing what should happen and why.
3) Results for each execution mode (counts, expectation values, energies).
4) Fidelity-oriented comparison across modes using:
   - state fidelity if state access exists
   - otherwise distribution distance (e.g., TVD) on sampled outcomes
5) A reproducibility block.
"""

TASK_PROMPT_BACKEND_SELECTION = """\
Given a circuit family described as:
- n_qubits = {n_qubits}
- depth = {depth}
- gate_types = {gate_types}
- expected_entanglement = {entanglement}
- need_statevector_access = {need_statevector}
- noise_required = {noise_required}
- performance_target = {performance_target}

Select a simulation method and justify:
- statevector vs stabilizer vs MPS
Include failure modes and what evidence would cause you to switch methods.
"""

TASK_PROMPT_BENCHMARK = """\
Run the {benchmark_name} benchmark suite:

Parameters:
- Qubit range: {qubit_range}
- Shots per run: {shots}
- Seed: {seed}
- Backends to compare: {backends}

Required:
1) Construct the canonical circuit for each qubit count.
2) Run on each backend and collect results.
3) Compute evaluation metrics:
   - For Grover: success probability vs theoretical (π/4)√(N/M) iterations
   - For QFT: state fidelity vs DFT matrix
   - For VQE: energy error vs exact diagonalization
   - For QAOA: approximation ratio vs brute-force optimal
4) Generate a comparison table and summary.
5) Include reproducibility blocks for all runs.
"""

# ============================================================================
# TIER B: Textual "Quantum-Like Reasoning" (Toy-Scale Only)
# ============================================================================

TIER_B_AMPLITUDE_LEDGER = """\
Simulate a toy quantum-like reasoning process using an Amplitude Ledger with \
K branches.

Rules:
- Maintain K branches {{b_i}} with complex amplitudes a_i (use polar form \
r_i * exp(i*theta_i)).
- Ensure normalization: sum_i |a_i|^2 = 1.
- Each step applies a specified KxK unitary U to the amplitude vector a \
(you may use small K=2..8 only).
- After T steps, perform a measurement: sample one branch according to \
|a_i|^2 and output that branch as the final answer.
- Also output the full final probability distribution and an explanation \
of interference effects (which branches were amplified/suppressed).

If K or U is too large to compute exactly, refuse and ask for smaller K or \
simpler U.

K = {k}
T = {t}
Initial amplitudes = {initial_amplitudes}
Unitary operations per step = {unitaries}
"""

# ============================================================================
# Helper
# ============================================================================

def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with the given keyword arguments.
    Missing keys are left as {key} placeholders.
    """
    for key, value in kwargs.items():
        placeholder = "{" + key + "}"
        template = template.replace(placeholder, str(value))
    return template
AI Orchestration Layer — Prompt templates and backend selector.

Provides two tiers of prompt templates for an LLM acting as a
"Quantum Emulation Agent":

  Tier A — Tool-verified quantum emulation (recommended for production)
  Tier B — Textual amplitude-ledger reasoning (pedagogical / low-fidelity)
"""

from __future__ import annotations

from typing import Optional

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend
from quantum_kernel.backends.stabilizer import StabilizerBackend
from quantum_kernel.backends.mps import MPSBackend


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------


class BackendSelector:
    """
    Automatically select the most appropriate simulation backend for a circuit.

    Policy
    ------
    - ≤ 20 qubits, any gates       → StatevectorBackend (exact)
    - All Clifford gates            → StabilizerBackend (scalable)
    - 20 < n ≤ 100                 → MPSBackend (approximate)
    """

    _CLIFFORD = {"H", "X", "Y", "Z", "S", "CX", "CZ", "SWAP", "MEASURE"}

    @classmethod
    def select(cls, circuit: QuantumCircuit):
        """Return an instantiated backend appropriate for *circuit*."""
        n = circuit.num_qubits
        gate_names = {g.name for g in circuit.gates}
        non_clifford = gate_names - cls._CLIFFORD

        if n <= StatevectorBackend.MAX_QUBITS:
            return StatevectorBackend()
        if not non_clifford:
            return StabilizerBackend()
        return MPSBackend()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TIER_A = """\
You are a Quantum Emulation Agent (Tier A — Tool-Verified).

Your role is to reason about quantum circuit behaviour using a classical
simulation tool. You must NEVER fabricate amplitudes or measurement outcomes.
All quantum results must come from the simulation backend.

Workflow (R → A → V → E):
  R — Receive the user's circuit or algorithm description.
  A — Assemble a QuantumCircuit using the circuit IR.
  V — Verify by running the circuit on the StatevectorBackend.
  E — Explain the simulation result in plain language.

Rules:
  1. Always state: "This is NOT a quantum computer — results are ideal_simulation."
  2. Show the circuit diagram produced by qc.draw().
  3. Report statevector amplitudes rounded to 4 decimal places.
  4. Report measurement counts and probabilities.
  5. Include the ReproducibilityBlock in every response.
  6. Never claim the simulation proves anything about real quantum hardware.
"""

SYSTEM_PROMPT_TIER_B = """\
You are a Quantum Emulation Agent (Tier B — Amplitude Ledger).

You reason step-by-step about quantum states using symbolic amplitude tracking.
This mode is PEDAGOGICAL ONLY and may accumulate floating-point errors.

Rules:
  1. Track amplitudes as fractions of 1/√2^n where possible.
  2. Clearly mark all outputs as PEDAGOGICAL — NOT verified by simulation.
  3. For production use, always prefer Tier A (tool-verified).
"""

USER_PROMPT_TEMPLATE = """\
Please simulate the following quantum circuit and explain the result:

{circuit_description}

Number of qubits: {num_qubits}
Shots: {shots}
Seed: {seed}
"""


def build_user_prompt(
    circuit_description: str,
    num_qubits: int,
    shots: int = 1024,
    seed: Optional[int] = None,
) -> str:
    """Format a user-facing simulation request prompt."""
    return USER_PROMPT_TEMPLATE.format(
        circuit_description=circuit_description,
        num_qubits=num_qubits,
        shots=shots,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Agent (synchronous, tool-calling)
# ---------------------------------------------------------------------------


class QuantumEmulationAgent:
    """
    A simple synchronous agent that assembles circuits, runs them, and
    produces a structured result. In a production setting this would be
    driven by an LLM; here it provides the tool-calling skeleton.
    """

    def __init__(self, tier: str = "A") -> None:
        if tier not in ("A", "B"):
            raise ValueError(f"tier must be 'A' or 'B', got {tier!r}")
        self.tier = tier
        self.system_prompt = (
            SYSTEM_PROMPT_TIER_A if tier == "A" else SYSTEM_PROMPT_TIER_B
        )

    def run(self, circuit: QuantumCircuit, shots: int = 1024, seed: Optional[int] = None) -> dict:
        """
        Run a circuit through the agent pipeline (R → A → V → E).

        Returns a structured report dict.
        """
        from quantum_kernel.provenance import ReproducibilityBlock

        backend = BackendSelector.select(circuit)
        result = backend.run(circuit, shots=shots, seed=seed)

        prov = ReproducibilityBlock.from_run(
            circuit,
            backend.__class__.__name__.replace("Backend", "").lower(),
            getattr(result, "method", "unknown"),
            shots,
            seed,
        )

        top = dict(sorted(result.counts.items(), key=lambda x: -x[1])[:8])

        return {
            "system_prompt_tier": self.tier,
            "label": getattr(result, "label", "ideal_simulation"),
            "circuit": str(circuit),
            "circuit_diagram": circuit.draw(),
            "top_counts": top,
            "total_shots": shots,
            "provenance": str(prov),
        }
