// Base URL for API calls
const API_BASE = '/api/benchmarks';

// Navigation Logic
document.querySelectorAll('.nav-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Update active nav
        document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        // Update active panel
        const targetId = button.getAttribute('data-target');
        document.querySelectorAll('.benchmark-panel').forEach(panel => {
            panel.style.display = 'none';
            panel.classList.remove('active');
        });
        
        const targetPanel = document.getElementById(targetId);
        targetPanel.style.display = 'block';
        // Small delay to allow display:block to apply before animating opacity
        setTimeout(() => targetPanel.classList.add('active'), 10);
    });
});

// UI Helpers
const showLoading = () => document.getElementById('loading-overlay').classList.remove('hidden');
const hideLoading = () => document.getElementById('loading-overlay').classList.add('hidden');

const formatJSON = (obj) => {
    return `<div class="json-output">${JSON.stringify(obj, null, 2)}</div>`;
};

const createMetricBox = (title, value, isSuccess = false) => {
    // Format float values appropriately
    let displayValue = value;
    if (typeof value === 'number') {
        displayValue = Number.isInteger(value) ? value : value.toFixed(4);
    }
    
    return `
        <div class="metric-box">
            <div class="metric-title">${title}</div>
            <div class="metric-value ${isSuccess ? 'success' : ''}">${displayValue}</div>
        </div>
    `;
};

// --- QML Classifier ---
async function runQML() {
    const layers = parseInt(document.getElementById('qml-layers').value);
    const iters = parseInt(document.getElementById('qml-iters').value);
    const resultsContainer = document.getElementById('qml-results');
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/qml`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ layers: layers, maxiter: iters, samples: 40, seed: Math.floor(Math.random() * 1000) })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        // Render Results
        let html = '<div class="result-card">';
        html += createMetricBox('Accuracy', (data.accuracy * 100).toFixed(1) + '%', data.accuracy > 0.8);
        html += createMetricBox('Final Loss (MSE)', data.final_loss.toFixed(4));
        html += createMetricBox('Iterations (COBYLA)', data.iterations);
        html += createMetricBox('Training Time', data.time_taken_sec.toFixed(2) + 's');
        html += '</div>';
        
        html += `
            <div class="metric-box" style="margin-top: 1rem;">
                <div class="metric-title">Optimization History</div>
                <div style="font-family: var(--font-mono); margin-top: 0.5rem; color: var(--text-primary)">
                    Initial Loss: ${data.loss_history[0].toFixed(4)}<br>
                    Final Loss: &nbsp;&nbsp;${data.final_loss.toFixed(4)}<br>
                    Status: ${data.message}
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        
    } catch (e) {
        resultsContainer.innerHTML = `<div class="metric-box" style="border-color: var(--danger)">
            <div class="metric-title" style="color: var(--danger)">Error</div>
            <div>${e.message}</div>
        </div>`;
    } finally {
        hideLoading();
    }
}

// --- Grover's Algorithm ---
async function runGrover() {
    const qubits = parseInt(document.getElementById('grover-qubits').value);
    const shots = parseInt(document.getElementById('grover-shots').value);
    const resultsContainer = document.getElementById('grover-results');
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/grover`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_qubits: qubits, shots: shots, seed: 42 })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        // Render Results
        let html = '<div class="result-card">';
        html += createMetricBox('Theoretical Success', data.theoretical_success_prob);
        html += createMetricBox('Measured Success', data.measured_success_prob, data.measured_success_prob > 0.5);
        html += createMetricBox('Optimal Iterations', data.optimal_iterations);
        html += createMetricBox('Error', data.prob_error);
        html += '</div>';
        
        // Render Distribution Bars
        html += '<h3>Measurement Distribution (Top 8)</h3>';
        html += '<div class="bar-chart">';
        
        const maxCount = Math.max(...Object.values(data.top_counts));
        for (const [bitstring, count] of Object.entries(data.top_counts)) {
            const percentage = (count / maxCount) * 100;
            const absolutePercent = ((count / shots) * 100).toFixed(1);
            html += `
                <div class="bar-row">
                    <div class="bar-label">|${bitstring}⟩</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${percentage}%"></div>
                    </div>
                    <div class="bar-value">${absolutePercent}%</div>
                </div>
            `;
        }
        html += '</div>';
        
        resultsContainer.innerHTML = html;
        
    } catch (e) {
        resultsContainer.innerHTML = `<div class="metric-box" style="border-color: var(--danger)">
            <div class="metric-title" style="color: var(--danger)">Error</div>
            <div>${e.message}</div>
        </div>`;
    } finally {
        hideLoading();
    }
}

// --- Quantum Fourier Transform ---
async function runQFT() {
    const qubits = parseInt(document.getElementById('qft-qubits').value);
    const resultsContainer = document.getElementById('qft-results');
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/qft`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_qubits: qubits, shots: 1024, seed: 42 })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        let html = '<div class="result-card">';
        html += createMetricBox('Overall Passed', data.overall_passed ? 'YES' : 'NO', data.overall_passed);
        html += createMetricBox('State Space (N)', data.N);
        html += '</div>';
        
        html += '<h3>Test Case Verifications</h3>';
        data.tests.forEach(test => {
            html += `
                <div class="metric-box" style="margin-bottom: 1rem; border-color: ${test.passed ? 'var(--success)' : 'var(--danger)'}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <strong>${test.name}</strong>
                        <span style="color: ${test.passed ? 'var(--success)' : 'var(--danger)'}">${test.passed ? 'PASS' : 'FAIL'}</span>
                    </div>
                    <div class="metric-title">${test.description}</div>
                    <div style="font-family: var(--font-mono); margin-top: 0.5rem;">Fidelity: ${test.fidelity.toFixed(6)}</div>
                </div>
            `;
        });
        
        if (data.unitary_comparison) {
            html += `
                <div class="metric-box" style="margin-top: 1rem;">
                    <div class="metric-title">Full Unitary Comparison vs DFT Matrix</div>
                    <div style="font-family: var(--font-mono); margin-top: 0.5rem;">
                        Frobenius Error: ${data.unitary_comparison.frobenius_error.toFixed(6)}<br>
                        Passed: ${data.unitary_comparison.passed}
                    </div>
                </div>
            `;
        }
        
        resultsContainer.innerHTML = html;
        
    } catch (e) {
        resultsContainer.innerHTML = `<div class="metric-box" style="border-color: var(--danger)">
            <div class="metric-title" style="color: var(--danger)">Error</div>
            <div>${e.message}</div>
        </div>`;
    } finally {
        hideLoading();
    }
}

// --- VQE (Variational Quantum Eigensolver) ---
async function runVQE() {
    const resultsContainer = document.getElementById('vqe-results');
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/vqe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: Math.floor(Math.random() * 1000) })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        let html = '<div class="result-card">';
        html += createMetricBox('Exact Energy (Ha)', data.exact_ground_energy);
        html += createMetricBox('VQE Energy (Ha)', data.vqe_energy, data.chemical_accuracy);
        html += createMetricBox('Energy Error', data.energy_error);
        html += createMetricBox('Chemical Accuracy', data.chemical_accuracy ? 'ACHIEVED' : 'FAILED', data.chemical_accuracy);
        html += createMetricBox('State Fidelity', data.state_fidelity);
        html += createMetricBox('Optimizer Iterations', data.iterations);
        html += '</div>';
        
        html += `
            <div class="metric-box" style="margin-top: 1rem;">
                <div class="metric-title">Optimization History</div>
                <div style="font-family: var(--font-mono); margin-top: 0.5rem; color: var(--text-primary)">
                    Initial Energy: ${data.energy_history_first_last[0]?.toFixed(6) || 'N/A'} Ha<br>
                    Final Energy: &nbsp;&nbsp;${data.energy_history_first_last[1]?.toFixed(6) || 'N/A'} Ha<br>
                    Ansatz: ${data.num_qubits} qubits, depth ${data.ansatz_depth} (${data.num_params} parameters)
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        
    } catch (e) {
        resultsContainer.innerHTML = `<div class="metric-box" style="border-color: var(--danger)">
            <div class="metric-title" style="color: var(--danger)">Error</div>
            <div>${e.message}</div>
        </div>`;
    } finally {
        hideLoading();
    }
}

// --- QAOA (MaxCut) ---
async function runQAOA() {
    const resultsContainer = document.getElementById('qaoa-results');
    
    showLoading();
    try {
        const response = await fetch(`${API_BASE}/qaoa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: Math.floor(Math.random() * 1000) })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        let html = '<div class="result-card">';
        html += createMetricBox('Optimal Cut', data.optimal_cut);
        html += createMetricBox('Best Sampled Cut', data.best_sampled_cut, data.best_sampled_cut === data.optimal_cut);
        html += createMetricBox('Approximation Ratio', data.approximation_ratio, data.approximation_ratio > 0.8);
        html += createMetricBox('Sampled Approx Ratio', data.sampled_approx_ratio, data.sampled_approx_ratio > 0.8);
        html += createMetricBox('QAOA Energy', data.qaoa_energy);
        html += createMetricBox('Best Bitstring', `|${data.best_sampled_bitstring}⟩`);
        html += '</div>';
        
        // Render Distribution Bars
        html += '<h3>Measurement Distribution (Top 8)</h3>';
        html += '<div class="bar-chart">';
        
        const maxCount = Math.max(...Object.values(data.top_counts));
        for (const [bitstring, count] of Object.entries(data.top_counts)) {
            const percentage = (count / maxCount) * 100;
            // Rough percentage assuming 1024 shots
            const absolutePercent = ((count / 1024) * 100).toFixed(1);
            const isOptimal = bitstring === data.optimal_bitstring || bitstring === data.best_sampled_bitstring;
            
            html += `
                <div class="bar-row">
                    <div class="bar-label" style="${isOptimal ? 'color: var(--success); font-weight: bold;' : ''}">|${bitstring}⟩</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: ${percentage}%; ${isOptimal ? 'background: var(--success)' : ''}"></div>
                    </div>
                    <div class="bar-value">${absolutePercent}%</div>
                </div>
            `;
        }
        html += '</div>';
        
        html += `
            <div class="metric-box" style="margin-top: 1.5rem;">
                <div class="metric-title">Problem Details</div>
                <div style="font-family: var(--font-mono); margin-top: 0.5rem; color: var(--text-primary)">
                    Target: ${data.problem} on ${data.num_qubits}-node cycle graph (${data.num_edges} edges)<br>
                    QAOA Depth (p): ${data.qaoa_depth_p}<br>
                    Optimizer: ${data.optimizer} (Converged: ${data.converged})
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = html;
        
    } catch (e) {
        resultsContainer.innerHTML = `<div class="metric-box" style="border-color: var(--danger)">
            <div class="metric-title" style="color: var(--danger)">Error</div>
            <div>${e.message}</div>
        </div>`;
    } finally {
        hideLoading();
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // Optional: trigger the first tab's run on load, or just leave it empty.
    // For now, we leave the placeholder text as requested by the UI design.
});

// --- Chatbot Logic ---
function handleChatKey(event) {
    if (event.key === 'Enter') {
        sendChatMessage();
    }
}

async function sendChatMessage() {
    const inputField = document.getElementById('chat-input');
    const message = inputField.value.trim();
    if (!message) return;
    
    const historyContainer = document.getElementById('chat-history');
    
    // Add user message to UI
    historyContainer.innerHTML += `
        <div class="chat-message user-msg">
            <div class="msg-avatar">U</div>
            <div class="msg-bubble">${message}</div>
        </div>
    `;
    inputField.value = '';
    
    // Scroll to bottom
    historyContainer.scrollTop = historyContainer.scrollHeight;
    
    // Add loading indicator
    const loadingId = 'loading-' + Date.now();
    historyContainer.innerHTML += `
        <div class="chat-message agent-msg" id="${loadingId}">
            <div class="msg-avatar">Q</div>
            <div class="msg-bubble" style="color: var(--text-secondary); font-style: italic;">
                Orchestrating simulation...
            </div>
        </div>
    `;
    historyContainer.scrollTop = historyContainer.scrollHeight;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });
        
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        
        // Remove loading and add agent response
        document.getElementById(loadingId).remove();
        
        // Format the text summary into pre-wrapped HTML to preserve the console-like formatting
        const formattedResponse = `<pre style="font-family: var(--font-mono); font-size: 0.875rem; overflow-x: auto; white-space: pre-wrap; margin: 0;">${data.response}</pre>`;
        
        historyContainer.innerHTML += `
            <div class="chat-message agent-msg">
                <div class="msg-avatar">Q</div>
                <div class="msg-bubble">${formattedResponse}</div>
            </div>
        `;
    } catch (e) {
        document.getElementById(loadingId).remove();
        historyContainer.innerHTML += `
            <div class="chat-message agent-msg">
                <div class="msg-avatar" style="background: var(--danger)">!</div>
                <div class="msg-bubble" style="border-color: var(--danger); color: var(--danger)">
                    Error: ${e.message}
                </div>
            </div>
        `;
    }
    historyContainer.scrollTop = historyContainer.scrollHeight;
}
