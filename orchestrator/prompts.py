"""
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
