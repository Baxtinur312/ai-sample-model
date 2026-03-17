import numpy as np
import time
import sys
import os
from typing import Tuple, List, Callable, Dict, Any
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_kernel.circuit_ir import QuantumCircuit
from quantum_kernel.backends.statevector import StatevectorBackend

class QMLClassifier:
    """
    A Proof-of-concept Quantum Neural Network (QNN) that acts as a binary classifier.
    It builds a Parameterized Quantum Circuit (PQC) and trains it using classical optimization.
    
    Structure:
    1. Data Encoding: RX and RY rotations to embed classical data into the quantum state.
    2. Trainable Ansatz: Layered RY/RZ rotations structured with CNOT entanglers.
    3. Measurement: Expectation value of Pauli-Z on the first qubit.
    """
    
    def __init__(self, num_qubits: int = 2, layers: int = 2, seed: int = 42):
        self.num_qubits = num_qubits
        self.layers = layers
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Each layer needs 2 parameters per qubit (RY, RZ)
        self.num_params = self.layers * self.num_qubits * 2
        self.params = self.rng.uniform(-np.pi, np.pi, self.num_params)
        
        self.backend = StatevectorBackend()

    def _build_circuit(self, x: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        """
        Build the quantum circuit for a single data point `x` given current weights `params`.
        Assumes x is a 2D feature vector implicitly mapped into self.num_qubits.
        """
        circuit = QuantumCircuit(self.num_qubits)
        
        # 1. Data Encoding (Feature Map)
        # Assuming x has 2 features. We map x[0] to RX and x[1] to RY on all qubits
        for i in range(self.num_qubits):
            circuit.rx(i, x[0])
            circuit.ry(i, x[1])
            
        # 2. Trainable Ansatz (Variational Form)
        param_idx = 0
        for _ in range(self.layers):
            # Rotations
            for i in range(self.num_qubits):
                circuit.ry(i, params[param_idx])
                circuit.rz(i, params[param_idx + 1])
                param_idx += 2
                
            # Entangling layer (Linear topology)
            for i in range(self.num_qubits - 1):
                circuit.cnot(i, i + 1)
                
        return circuit

    def _compute_expectation(self, statevector: np.ndarray) -> float:
        """
        Compute the expectation value of the Pauli Z operator on qubit 0.
        Z = |0><0| - |1><1|. So E[Z] = P(|0>) - P(|1>) on qubit 0.
        """
        # Tracing out to get probabilities for qubit 0
        # Since qubit 0 is the most significant bit in our ordering (or least, depending on convention)
        # Let's compute directly: Z expectation = sum_{i where q0 is 0} |c_i|^2 - sum_{i where q0 is 1} |c_i|^2
        
        # In our statevector backend, the convention is typical lexicographical. Default is q0 is least significant bit in Qiskit, but our kernel is standard.
        # Let's just manually compute the density of bit 0 being 0 vs 1.
        
        prob_0 = 0.0
        prob_1 = 0.0
        for i, val in enumerate(statevector):
            # Check if 0th bit of integer `i` is 0 or 1
            if (i & 1) == 0:
                prob_0 += np.abs(val)**2
            else:
                prob_1 += np.abs(val)**2
                
        return prob_0 - prob_1

    def forward(self, x: np.ndarray, params: Optional[np.ndarray] = None) -> float:
        """
        Forward pass for a single data point. Returns output in [-1, 1].
        """
        if params is None:
            params = self.params
            
        circuit = self._build_circuit(x, params)
        result = self.backend.run(circuit, shots=0) # Only statevector needed
        return self._compute_expectation(result.statevector)

    def predict(self, X: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
        """ Predict class labels (0 or 1) for a batch of data. """
        predictions = []
        for x in X:
            expectation = self.forward(x, params)
            # Map [-1, 1] expectation to class 0 or 1.
            # If Z > 0, we can say class 0, else class 1.
            pred = 0 if expectation > 0 else 1
            predictions.append(pred)
        return np.array(predictions)

    def cost_function(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """ Mean Squared Error loss mapping y in {0,1} to target expectations {1, -1}. """
        loss = 0.0
        for idx in range(len(X)):
            pred_z = self.forward(X[idx], params)
            target_z = 1.0 if y[idx] == 0 else -1.0
            loss += (pred_z - target_z) ** 2
        return loss / len(X)

    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 50) -> Dict[str, Any]:
        """ Train the QNN. """
        history = []
        
        def callback(xk):
            current_loss = self.cost_function(xk, X, y)
            history.append(current_loss)
            
        print(f"Training QNN on {len(X)} samples with {maxiter} iterations...")
        start_time = time.time()
        
        # Initial loss
        history.append(self.cost_function(self.params, X, y))
        
        res = minimize(
            self.cost_function,
            self.params,
            args=(X, y),
            method='COBYLA',
            options={'maxiter': maxiter},
            callback=callback
        )
        
        self.params = res.x
        end_time = time.time()
        
        # Compute final accuracy
        preds = self.predict(X)
        accuracy = np.mean(preds == y)
        
        return {
            "success": res.success,
            "message": res.message,
            "final_loss": res.fun,
            "accuracy": float(accuracy),
            "iterations": res.nfev,
            "loss_history": [float(h) for h in history],
            "time_taken_sec": end_time - start_time,
            "final_params": self.params.tolist()
        }

# Helper to generate a toy dataset (e.g., XOR or clusters)
def generate_toy_data(samples: int = 40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Generate 2 blobs
    X1 = rng.multivariate_normal([0.5, 0.5], [[0.1, 0], [0, 0.1]], samples // 2)
    y1 = np.zeros(samples // 2)
    
    X2 = rng.multivariate_normal([-0.5, -0.5], [[0.1, 0], [0, 0.1]], samples // 2)
    y2 = np.ones(samples // 2)
    
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    
    # Shuffle
    indices = np.arange(samples)
    rng.shuffle(indices)
    
    return X[indices], y[indices]

if __name__ == "__main__":
    X, y = generate_toy_data()
    qnn = QMLClassifier(num_qubits=2, layers=2, seed=42)
    print("Initial accuracy:", np.mean(qnn.predict(X) == y))
    
    res = qnn.fit(X, y, maxiter=40)
    print("Optimization Result:", res)
