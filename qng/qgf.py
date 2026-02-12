from typing import Any, Callable, List, Optional, Tuple, Union

from pennylane import numpy as np
import pennylane as qml
from scipy.integrate import solve_ivp

from .utils import get_logger

logger = get_logger(__name__)

def natural_gradient_flow(t: float, params_flat: np.ndarray, cost_circuit: qml.QNode, n_layers: int, n_qubits: int) -> np.ndarray:
    """
    ODE function for Quantum Gradient Flow (QGF).
    Equation: d(theta)/dt = - g^{-1} * grad(L)
    
    Args:
        t (float): Time variable (required by ODE solver).
        params_flat (np.ndarray): Flattened parameters vector.
        cost_circuit (qml.QNode): PennyLane QNode representing the cost function.
        n_layers (int): Number of layers in the ansatz.
        n_qubits (int): Number of qubits.
        
    Returns:
        np.ndarray: The derivative d(theta)/dt (flattened).
    """
    # Reshape params
    params = params_flat.reshape((n_layers, n_qubits, 3))
    params_pennylane = np.array(params, requires_grad=True)
    
    # Gradient calculation
    grad_fn = qml.grad(cost_circuit)
    current_grad = grad_fn(params_pennylane)
    
    # Metric Tensor calculation (Block-diag)
    # Note: approx="block-diag" provides the Fubini-Study metric approximation
    metric_fn = qml.metric_tensor(cost_circuit, approx="block-diag")
    g_matrix_all = metric_fn(params_pennylane)
    
    v_list = []
    
    # Solve system per layer/qubit
    for l in range(n_layers):
        
        v_layer = np.zeros_like(current_grad[l])
        
        for q in range(n_qubits):
            g_vec = current_grad[l, q]
            try:
                # Correct slicing for (L, N, 3, L, N, 3) tensor
                # Extract the 3x3 block for layer l, qubit q
                mat = g_matrix_all[l, q, :, l, q, :]
                
                # Regularization to avoid singular matrix
                mat = mat + 1e-6 * np.eye(3) 
                
                # Solve linear system: mat * v = -grad
                # We want d(theta)/dt = - g^-1 * grad
                # So mat * d(theta)/dt = -grad
                sol = np.linalg.solve(mat, -g_vec)
                v_layer[q] = sol
                
            except (IndexError, TypeError, np.linalg.LinAlgError):
                # Fallback to standard gradient descent direction if metric is singular or ill-formed
                v_layer[q] = -g_vec 
        
        v_list.append(v_layer)
        
    return np.array(v_list).flatten()

def simulate_qgf(cost_circuit: qml.QNode, init_params: np.ndarray, t_span: Tuple[float, float], t_eval: np.ndarray, n_layers: int, n_qubits: int) -> Any:
    """
    Run the ODE solver to simulate Quantum Gradient Flow.
    
    Args:
        cost_circuit (qml.QNode): The cost function QNode.
        init_params (np.ndarray): Initial parameters.
        t_span (Tuple[float, float]): Integration interval (t0, tf).
        t_eval (np.ndarray): Time points to evaluate the solution at.
        n_layers (int): Number of layers.
        n_qubits (int): Number of qubits.
        
    Returns:
        scipy.integrate.OdeResult: Object containing the solution structure.
    """
    init_params_flat = init_params.flatten()
    
    # Wrapper for ODE solver because solve_ivp expects f(t, y, *args) structure
    # but we pass args clearly via lambda or wrapper
    def ode_func(t: float, y: np.ndarray) -> np.ndarray:
        return natural_gradient_flow(t, y, cost_circuit, n_layers, n_qubits)
        
    sol = solve_ivp(fun=ode_func, t_span=t_span, y0=init_params_flat, 
                    t_eval=t_eval, method='RK45')
    
    return sol
