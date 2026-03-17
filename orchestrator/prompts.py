"""
Prompt Templates for Quantum Emulation
=======================================
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
