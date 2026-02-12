import pytest
import pennylane as qml
from pennylane import numpy as np
from qng.qng import metric_tensor_block_diag, qng_update_step_manual

def test_metric_tensor_block_diag():
    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))
    
    params = np.array([0.1, 0.2], requires_grad=True)
    mt = metric_tensor_block_diag(circuit, params)
    assert mt.shape == (2, 2)

def test_qng_update_step_manual():
    # Mock data for (n_layers=1, n_qubits=1, 3)
    params = np.array([[[0.1, 0.2, 0.3]]])
    grad = np.array([[[0.01, 0.01, 0.01]]])
    # Mock metric tensor block (3x3)
    # PennyLane 6D structure simulation: (1, 1, 3, 1, 1, 3)
    mt_blocks = np.zeros((1, 1, 3, 1, 1, 3))
    for i in range(3):
        mt_blocks[0, 0, i, 0, 0, i] = 0.25 # Identity scaling
        
    new_params = qng_update_step_manual(params, grad, mt_blocks, step_size=0.01)
    assert new_params.shape == params.shape
    # Check if updated (params - eta * g^-1 * grad)
    # g = 0.25*I -> g^-1 = 4*I
    # new = 0.1 - 0.01 * 4 * 0.01 = 0.1 - 0.0004 = 0.0996
    assert np.allclose(new_params[0, 0, 0], 0.1 - 0.01 * 4 * 0.01)
