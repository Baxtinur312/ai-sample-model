import os
import time
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from quantum_kernel.circuit_ir import QuantumCircuit
from orchestrator.agent import QuantumEmulationAgent

# Context to hold the current draft circuit across tool calls
_shared_context = {
    "circuit": None,
    "circuit_logs": [],
}

class QuantumAgentTools:
    """
    Tools exposed to the Gemini model so it can build and run quantum circuits.
    """
    
    @staticmethod
    def initialize_circuit(num_qubits: int) -> str:
        """
        Create a new clean quantum circuit with the specified number of qubits.
        Call this first before adding any gates.
        
        Args:
            num_qubits: The number of qubits in the circuit (e.g., 2 for a Bell state).
        """
        _shared_context["circuit"] = QuantumCircuit(num_qubits)
        _shared_context["circuit_logs"] = [f"Initialized circuit with {num_qubits} qubits."]
        return f"Successfully initialized a quantum circuit with {num_qubits} qubits."

    @staticmethod
    def apply_single_qubit_gate(gate: str, target: int, param: Optional[float] = None) -> str:
        """
        Apply a single-qubit gate (H, X, Y, Z, S, T, rz, ry, rx) to a target qubit.
        
        Args:
            gate: The name of the gate (e.g., 'h', 'x', 'rz').
            target: The index of the qubit to apply the gate to (0-indexed).
            param: Optional rotation parameter for parameterized gates like rx, ry rz.
        """
        circ = _shared_context.get("circuit")
        if not circ:
            return "Error: You must initialize the circuit first using initialize_circuit."
            
        gate = gate.lower()
        try:
            if param is not None:
                getattr(circ, gate)(target, param)
                _shared_context["circuit_logs"].append(f"Applied {gate.upper()}({param:.3f}) on q[{target}]")
                return f"Applied {gate.upper()}({param}) to qubit {target}."
            else:
                getattr(circ, gate)(target)
                _shared_context["circuit_logs"].append(f"Applied {gate.upper()} on q[{target}]")
                return f"Applied {gate.upper()} to qubit {target}."
        except Exception as e:
            return f"Error applying gate: {str(e)}"

    @staticmethod
    def apply_two_qubit_gate(gate: str, control: int, target: int) -> str:
        """
        Apply a two-qubit gate (like cnot, cz, swap) between two qubits.
        
        Args:
            gate: The name of the gate (e.g., 'cnot', 'cz').
            control: The index of the control qubit.
            target: The index of the target qubit.
        """
        circ = _shared_context.get("circuit")
        if not circ:
            return "Error: You must initialize the circuit first using initialize_circuit."
            
        gate = gate.lower()
        try:
            getattr(circ, gate)(control, target)
            _shared_context["circuit_logs"].append(f"Applied {gate.upper()} on q[{control}] -> q[{target}]")
            return f"Applied {gate.upper()} with control {control} and target {target}."
        except Exception as e:
            return f"Error applying gate: {str(e)}"

    @staticmethod
    def run_simulation(phenomenon: str, physics_expectation: str) -> str:
        """
        Run the fully constructed quantum circuit on the emulation kernel, gathering 
        both ideal and noisy results to explain the phenomenon.
        
        Args:
            phenomenon: A short title of what is being demonstrated (e.g., "Bell State").
            physics_expectation: A brief explanation of what the physical outputs of this circuit should be.
        """
        circ = _shared_context.get("circuit")
        if not circ:
            return "Error: You must build a circuit before running a simulation."
            
        try:
            agent = QuantumEmulationAgent(output_dir="/tmp")
            result = agent.run_experiment(
                circuit=circ,
                phenomenon=phenomenon,
                physics_expectation=physics_expectation,
                modes=["ideal", "noisy"],
                shots=1024
            )
            summary = result.summary()
            
            # Reset the context for the next request
            _shared_context["circuit"] = None
            _shared_context["circuit_logs"] = []
            
            return f"Simulation complete. Results:\n{summary}"
        except Exception as e:
            return f"Simulation failed: {str(e)}"


class GeminiOrchestrator:
    """
    A true LLM-backed orchestrator that reasons about user prompts and 
    invokes the proper Quantum Behavior Kernel tools.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY not found. LLM agent will fail if called.")
            self.client = None
        else:
            self.client = genai.Client()

    def process_prompt(self, user_prompt: str) -> str:
        """
        Send a user request to Gemini and allow it to use tools to build 
        and simulate a quantum circuit before returning the final response.
        """
        if not self.client:
            return (
                "Error: GEMINI_API_KEY environment variable is missing. "
                "Unable to connect to the Gemini LLM for reasoning. "
                "Please set it and restart the server."
            )

        # Clear shared state before starting
        _shared_context["circuit"] = None
        _shared_context["circuit_logs"] = []

        system_instruction = (
            "You are a highly capable Quantum AI Emulation Orchestrator. "
            "A user will give you a phenomenon or problem. You must think step-by-step, "
            "design a quantum circuit using your tools, and run a simulation to prove it. "
            "ALWAYS Follow these steps in order:\n"
            "1. Call `initialize_circuit(num_qubits)`\n"
            "2. Call `apply_single_qubit_gate` or `apply_two_qubit_gate` repeatedly to build the circuit.\n"
            "3. Call `run_simulation(phenomenon, expectation)` to execute it on the Quantum Behavior Kernel.\n"
            "4. Summarize the results back to the user based on the tool output.\n"
            "Explain what the quantum physics implies and how the classical simulation verified it."
        )

        tools = [
            QuantumAgentTools.initialize_circuit,
            QuantumAgentTools.apply_single_qubit_gate,
            QuantumAgentTools.apply_two_qubit_gate,
            QuantumAgentTools.run_simulation
        ]
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.2
        )

        try:
            chat = self.client.chats.create(model="gemini-2.5-flash", config=config)
            
            # The SDK handles the tool loop natively if we just send the message
            # But the user might need streaming text in a real app. For Server.py, 
            # we will just block on response for simplicity, then return the text.
            response = chat.send_message(user_prompt)
            
            # The chat history will perform the function calling ping-pong internally.
            final_text = response.text
            
            return final_text
            
        except Exception as e:
            return f"LLM Execution Error: {str(e)}"
