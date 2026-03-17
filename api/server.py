import sys
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Ensure project root is on the path so we can import quantum_kernel and benchmarks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.grover import run_grover_benchmark
from benchmarks.qft import run_qft_benchmark
from benchmarks.vqe import run_vqe_benchmark
from benchmarks.qaoa import run_qaoa_benchmark

app = FastAPI(title="Quantum-Like AI System API")

# Setup CORS for development purposes (if frontend runs separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GroverRequest(BaseModel):
    num_qubits: int = 3
    shots: int = 2048
    seed: int = 42

class QFTRequest(BaseModel):
    num_qubits: int = 3
    shots: int = 1024
    seed: int = 42

class BasicSeedRequest(BaseModel):
    seed: int = 42

@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "Quantum Behavior Kernel API is online."}

@app.post("/api/benchmarks/grover")
def extract_grover(req: GroverRequest):
    try:
        if req.num_qubits < 2 or req.num_qubits > 10:
            raise HTTPException(status_code=400, detail="Grover requires 2-10 qubits.")
        
        result = run_grover_benchmark(
            num_qubits=req.num_qubits,
            shots=req.shots,
            seed=req.seed
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmarks/qft")
def extract_qft(req: QFTRequest):
    try:
        if req.num_qubits < 2 or req.num_qubits > 10:
            raise HTTPException(status_code=400, detail="QFT requires 2-10 qubits.")
            
        result = run_qft_benchmark(
            num_qubits=req.num_qubits,
            shots=req.shots,
            seed=req.seed
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmarks/vqe")
def extract_vqe(req: BasicSeedRequest):
    try:
        # VQE uses a fixed 2-qubit hardware efficient ansatz for H2
        result = run_vqe_benchmark(seed=req.seed)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmarks/qaoa")
def extract_qaoa(req: BasicSeedRequest):
    try:
        # QAOA uses a fixed 4-node cycle graph
        result = run_qaoa_benchmark(seed=req.seed)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class QMLRequest(BaseModel):
    num_qubits: int = 2
    layers: int = 2
    samples: int = 40
    maxiter: int = 50
    seed: int = 42

@app.post("/api/benchmarks/qml")
def run_qml(req: QMLRequest):
    try:
        from benchmarks.qml_classifier import QMLClassifier, generate_toy_data
        
        # Generate data
        X, y = generate_toy_data(samples=req.samples, seed=req.seed)
        
        # Initialize and train
        qnn = QMLClassifier(num_qubits=req.num_qubits, layers=req.layers, seed=req.seed)
        res = qnn.fit(X, y, maxiter=req.maxiter)
        
        # Format dataset for returning
        dataset = []
        for i in range(len(X)):
            dataset.append({
                "x": float(X[i][0]),
                "y": float(X[i][1]),
                "label": int(y[i])
            })
            
        res["dataset"] = dataset
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
def chat_agent(req: ChatRequest):
    try:
        from orchestrator.gemini_agent import GeminiOrchestrator
        
        agent = GeminiOrchestrator()
        result_text = agent.process_prompt(req.message)
        
        return {"response": result_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
os.makedirs(frontend_dir, exist_ok=True)

# Mount static files and handle root redirect
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Frontend not found", "path": index_path}

# Allow simple execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
