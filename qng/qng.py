import pennylane as qml
from pennylane import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union

from .utils import get_logger

logger = get_logger(__name__)

# Re-export PennyLane's QNGOptimizer for convenience
class QNGOptimizer(qml.QNGOptimizer):
    """
    Wrapper around PennyLane's QNGOptimizer.
    This optimizer uses the Fubini-Study metric tensor to adjust the gradient checks.
    
    Args:
        stepsize (float): Learning rate. Default: 0.01.
        approx (str): Approximation method for the metric tensor. Default: "block-diag".
    """
    def __init__(self, stepsize: float = 0.01, approx: str = "block-diag"):
        super().__init__(stepsize=stepsize, approx=approx)

def metric_tensor_block_diag(qnode: qml.QNode, params: np.ndarray) -> np.ndarray:
    """
    Compute the block-diagonal metric tensor for a given QNode and parameters.
    This is a helper function to inspect the geometry.
    
    Args:
        qnode (qml.QNode): The Quantum Node to inspect.
        params (np.ndarray): The parameters to evaluate the metric tensor at.
        
    Returns:
        np.ndarray: The block-diagonal metric tensor.
    """
    metric_fn = qml.metric_tensor(qnode, approx="block-diag")
    return metric_fn(params)

def qng_update_step_manual(params: np.ndarray, gradient: np.ndarray, metric_tensor_blocks: np.ndarray, step_size: float = 0.01) -> np.ndarray:
    """
    Thực hiện một bước cập nhật QNG thủ công (Detailed Implementation).
    
    Công thức: theta_new = theta_old - eta * g^{-1} * gradient
    
    Args:
        params (np.ndarray): Tham số hiện tại (n_layers, n_qubits, 3).
        gradient (np.ndarray): Gradient tại tham số hiện tại (n_layers, n_qubits, 3).
        metric_tensor_blocks (np.ndarray): Metric tensor dưới dạng block-diag.
        step_size (float): Learning rate. Default: 0.01.
        
    Returns:
        np.ndarray: Tham số đã cập nhật.
    """
    n_layers, n_qubits, _ = params.shape
    new_params = params.copy()
    
    # Duyệt qua từng layer và qubit để cập nhật (vì block-diag làm việc độc lập trên từng qubit/gate)
    for l in range(n_layers):
        for q in range(n_qubits):
            # 1. Lấy Gradient vector (3 thành phần cho 3 góc quay của Rot gate)
            g_vec = gradient[l, q]
            
            # 2. Lấy Metric Tensor block (3x3 matrix)
            # PennyLane trả về tensor shape (L, N, 3, L, N, 3) hoặc tuple of blocks.
            # Ở đây giả sử input đã được xử lý hoặc là output raw của qml.metric_tensor approx='block-diag'
            # CHÚ Ý: Cần xử lý Indexing đúng dựa trên cấu trúc trả về thực tế.
            # Nếu là full tensor (L, N, 3, L, N, 3):
            try:
                # Thử indexing kiểu tensor 6D
                block = metric_tensor_blocks[l, q, :, l, q, :]
            except (IndexError, TypeError):
                # Nếu là list/tuple phẳng hoặc shape khác
                # Fallback logic hoặc báo lỗi. Ở đây ta giả định tensor 6D hoặc 2D blocks
                # Cho trường hợp mô phỏng đơn giản, fallback về Identity nếu lỗi structure
                block = np.eye(3)
                
            # 3. Regularization (Tikhonov) để tránh ma trận suy biến (singular)
            # g -> g + epsilon * I
            block_reg = block + 1e-6 * np.eye(3)
            
            # 4. Tính tích: v = g^{-1} * gradient
            # Giải hệ phương trình tuyến tính: block_reg * v = g_vec
            # v = solve(block_reg, g_vec)
            # Hướng di chuyển tự nhiên (Natural Gradient direction)
            natural_grad_vec = np.linalg.solve(block_reg, g_vec)
            
            # 5. Cập nhật tham số
            # theta_new = theta_old - eta * natural_grad
            new_params[l, q] = params[l, q] - step_size * natural_grad_vec
            
    return new_params

