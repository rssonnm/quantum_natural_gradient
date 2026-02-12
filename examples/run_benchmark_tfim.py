import pennylane as qml
from pennylane import numpy as np
import time
from qng.models import SimpleQNN
from qng.utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

def run_tfim_benchmark(n_qubits=4, n_layers=2, steps=100):
    """
    Benchmark QNG on the Transverse Field Ising Model (TFIM).
    Finds the ground state energy of the Hamiltonian:
    H = - sum(Z_i Z_{i+1}) - h * sum(X_i)
    """
    h = 1.0 # Critical field
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Define Hamiltonian
    coeffs = []
    obs = []
    # Coupling terms (ZZ)
    for i in range(n_qubits):
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
    # Field terms (X)
    for i in range(n_qubits):
        coeffs.append(-h)
        obs.append(qml.PauliX(i))
    
    H = qml.Hamiltonian(coeffs, obs)
    
    @qml.qnode(dev)
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        return qml.expval(H)

    # Optimization with QNG
    logger.info("Starting TFIM Benchmark with QNG...")
    params_init = np.random.uniform(0, 2*np.pi, qml.StronglyEntanglingLayers.shape(n_layers, n_qubits), requires_grad=True)
    
    opt_qng = qml.QNGOptimizer(stepsize=0.05, approx="block-diag")
    params_qng = params_init.copy()
    
    start = time.time()
    for i in range(steps):
        params_qng, cost = opt_qng.step_and_cost(circuit, params_qng)
        if i % 20 == 0:
            logger.info(f"QNG Step {i}: Energy = {cost:.6f}")
    qng_time = time.time() - start
    logger.info(f"QNG Finished in {qng_time:.2f}s. Final Energy: {cost:.6f}")
    
    # Optimization with Adam
    logger.info("\nStarting TFIM Benchmark with Adam...")
    params_adam = params_init.copy()
    opt_adam = qml.AdamOptimizer(stepsize=0.05)
    
    start = time.time()
    for i in range(steps):
        params_adam, cost_adam = opt_adam.step_and_cost(circuit, params_adam)
        if i % 20 == 0:
            logger.info(f"Adam Step {i}: Energy = {cost_adam:.6f}")
    adam_time = time.time() - start
    logger.info(f"Adam Finished in {adam_time:.2f}s. Final Energy: {cost_adam:.6f}")

if __name__ == "__main__":
    run_tfim_benchmark(n_qubits=4, n_layers=2, steps=50)
