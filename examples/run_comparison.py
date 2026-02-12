
"""
TRIỂN KHAI PAPER: Quantum Natural Gradient (NeurIPS 2019)
Phần: Comparison Script (Refactored using `modules`)

Mục tiêu:
1. So sánh hiệu quả giữa Vanilla Gradient Descent (GD) và Quantum Natural Gradient (QNG).
2. Minh họa dòng chảy Quantum Gradient Flow (QGF).
"""

import pennylane as qml
from pennylane import numpy as np
import time
import matplotlib.pyplot as plt

# Import from our new modules
from qng.ansatz import circuit_layer
from qng.qng import QNGOptimizer, metric_tensor_block_diag
from qng.qgf import simulate_qgf
from qng.visual import plot_convergence_comparison

# ==========================================
# CONFIGURATION
# ==========================================
np.random.seed(42)
n_qubits = 9
n_layers = 5
step_size = 0.01
steps = 40 # Demo steps

dev = qml.device("default.qubit", wires=n_qubits)

# ==========================================
# DEFINE COST CIRCUIT (Specific to this experiment)
# ==========================================
@qml.qnode(dev)
def cost_circuit(params):
    # 1. Encoding
    for i in range(n_qubits):
        qml.RY(np.pi / 4, wires=i)
        
    # 2. Variational Ansatz
    # params shape: (n_layers, n_qubits, 3)
    for layer_weights in params:
        circuit_layer(layer_weights, n_qubits)
        
    # 3. Measurement (Hamiltonian Z0*Z1)
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

# Initialize Parameters
init_params = np.random.uniform(low=0, high=2 * np.pi, size=(n_layers, n_qubits, 3), requires_grad=True)
print(f"Initialized {n_qubits} qubits, {n_layers} layers.")

# ==========================================
# 1. VANILLA SGD
# ==========================================
print("\n--- Training with Vanilla SGD ---")
gd_params = init_params.copy()
gd_opt = qml.GradientDescentOptimizer(stepsize=step_size)
gd_cost_history = []
gd_params_history = [gd_params.flatten()]

start_time = time.time()
for i in range(steps):
    gd_params, value = gd_opt.step_and_cost(cost_circuit, gd_params)
    gd_cost_history.append(value)
    gd_params_history.append(gd_params.flatten())
    if i % 10 == 0: print(f"SGD Step {i}: {value:.4f}")
print(f"SGD Final Cost: {gd_cost_history[-1]:.4f}")

# ==========================================
# 2. QUANTUM NATURAL GRADIENT
# ==========================================
print("\n--- Training with Quantum Natural Gradient ---")
qng_params = init_params.copy()
qng_opt = QNGOptimizer(stepsize=step_size, approx="block-diag")
qng_cost_history = []
qng_params_history = [qng_params.flatten()]

start_time = time.time()
for i in range(steps):
    qng_params, value = qng_opt.step_and_cost(cost_circuit, qng_params)
    qng_cost_history.append(value)
    qng_params_history.append(qng_params.flatten())
    if i % 10 == 0: print(f"QNG Step {i}: {value:.4f}")
print(f"QNG Final Cost: {qng_cost_history[-1]:.4f}")

# ==========================================
# 3. ADAM OPTIMIZER (NEW)
# ==========================================
print("\n--- Training with Adam Optimizer ---")
adam_params = init_params.copy()
adam_opt = qml.AdamOptimizer(stepsize=step_size)
adam_cost_history = []
adam_params_history = [adam_params.flatten()]

start_time = time.time()
for i in range(steps):
    adam_params, value = adam_opt.step_and_cost(cost_circuit, adam_params)
    adam_cost_history.append(value)
    adam_params_history.append(adam_params.flatten())
    if i % 10 == 0: print(f"Adam Step {i}: {value:.4f}")
print(f"Adam Final Cost: {adam_cost_history[-1]:.4f}")

# ==========================================
# 4. QUANTUM GRADIENT FLOW (ODE)
# ==========================================
print("\n--- Simulating Quantum Gradient Flow (ODE) ---")
t_eval = np.linspace(0, steps * step_size, 50)
t_span = (0, steps * step_size)

qgf_params_history = []

try:
    sol = simulate_qgf(cost_circuit, init_params, t_span, t_eval, n_layers, n_qubits)
    
    qgf_cost_history = []
    print("Calculating Flow Cost...")
    qgf_params_history = sol.y.T # Already (n_steps, n_params)
    
    for params_flat in sol.y.T:
        p = params_flat.reshape((n_layers, n_qubits, 3))
        qgf_cost_history.append(cost_circuit(p))
    print("QGF Simulation Complete.")
except Exception as e:
    print(f"QGF Failed: {e}")
    qgf_cost_history = []
    sol = type('obj', (object,), {'t': np.array([])})

# ==========================================
# 5. VISUALIZATION
# ==========================================
print("\n--- Plotting Results ---")
steps_flow = sol.t / step_size if len(qgf_cost_history) > 0 else []

# 5.1 Convergence Plot
plot_convergence_comparison(
    gd_cost_history, 
    qng_cost_history, 
    qgf_cost_history, 
    steps_flow,
    adam_history=adam_cost_history,
    filename="reports/qng_adam_sgd_comparison.png"
)

# 5.2 Loss Landscape Plot (PCA)
from qng.visual import plot_loss_landscape_pca

# Prepare data
trajectories = [np.array(gd_params_history), np.array(qng_params_history), np.array(adam_params_history)]
labels = ["SGD", "QNG", "Adam"]

if len(qgf_params_history) > 0:
    trajectories.append(qgf_params_history)
    labels.append("QGF (Continuous)")

# Helper for reshape in lambda
init_shape = init_params.shape
def cost_wrapper(flat_params):
    p = flat_params.reshape(init_shape)
    return cost_circuit(p)

# Call with wrapper
plot_loss_landscape_pca(
    cost_fn=cost_wrapper,
    trajectories=trajectories,
    labels=labels,
    filename="reports/loss_landscape_pca.png",
    grid_size=20 # Reduced for speed
)

# Draw Circuit
qml.draw_mpl(cost_circuit)(init_params)
plt.savefig("reports/qng_circuit.png")
