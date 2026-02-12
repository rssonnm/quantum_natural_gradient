import pennylane as qml
from pennylane import numpy as np
from typing import Any, List, Optional, Tuple, Union

def circuit_layer(weights: np.ndarray, n_qubits: int) -> None:
    """
    Xây dựng một layer biến phân (Variational Layer) theo cấu trúc trong paper QNG.

    Cấu trúc mỗi layer bao gồm:
    1. Các cổng quay (Rotation) trên từng qubit: Rot(phi, theta, omega).
    2. Các cổng Entangling (CZ) để tạo rối lượng tử giữa các qubit lân cận.

    Args:
        weights (np.ndarray): Tham số biến phân cho layer này. Shape: (n_qubits, 3).
        n_qubits (int): Số lượng qubits trong mạch.
    """
    # 1. Single-qubit rotations
    for i in range(n_qubits):
        qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)

    # 2. Entangling layer (CZ gates)
    for i in range(n_qubits):
        qml.CZ(wires=[i, (i + 1) % n_qubits])

def StrongEntanglerLayers(params: np.ndarray, n_qubits: int) -> None:
    """
    Wrapper cho StrongEntanglingLayers của PennyLane.

    Args:
        params (np.ndarray): Tham số biến phân. Shape: (n_layers, n_qubits, 3).
        n_qubits (int): Số lượng qubits trong mạch.
    """
    qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
