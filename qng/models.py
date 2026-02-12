import time
from typing import Any, Callable, List, Optional, Tuple, Union

import pennylane as qml
from pennylane import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

class SimpleQNN:
    """
    Quantum Neural Network Classifier với API giống Scikit-learn.
    Hỗ trợ huấn luyện với QNG (Quantum Natural Gradient) hoặc Adam.

    Args:
        n_qubits (int): Số lượng qubits. Default: 4.
        n_layers (int): Số lượng layers biến phân. Default: 2.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params: Optional[np.ndarray] = None
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define internal QNode
        @qml.qnode(self.dev)
        def _circuit(params: np.ndarray, x: np.ndarray) -> np.ndarray:
            # Encoding
            qml.AngleEmbedding(x, wires=range(n_qubits))
            # Variational Layers (StronglyEntangling)
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            # Measurement
            return qml.probs(wires=0)
            
        self.qnode = _circuit

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 100, step_size: float = 0.01, optimizer: str = "adam", verbose: bool = True) -> None:
        """
        Huấn luyện mô hình.

        Args:
            X (np.ndarray): Dữ liệu đầu vào (features).
            y (np.ndarray): Nhãn (labels).
            steps (int): Số bước huấn luyện. Default: 100.
            step_size (float): Learning rate. Default: 0.01.
            optimizer (str): "adam" hoặc "qng". Default: "adam".
            verbose (bool): In tien do huan luyen. Default: True.
        """
        # Init params
        shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.n_qubits)
        self.params = np.random.uniform(low=0, high=2*np.pi, size=shape, requires_grad=True)
        
        # Cost function
        def cost(params: np.ndarray) -> float:
            predictions = [self.qnode(params, x_i)[1] for x_i in X] # Prob of class 1
            loss = np.mean((y - np.stack(predictions)) ** 2)
            return loss

        # Metric Tensor approximation for QNG
        def metric_tensor_fn(params: np.ndarray) -> np.ndarray:
            # Compute on representative sample (X[0])
            return qml.metric_tensor(self.qnode, approx="block-diag")(params, X[0])

        # Select Optimizer
        opt: Any
        if optimizer == "qng":
            opt = qml.QNGOptimizer(stepsize=step_size, approx="block-diag")
        else:
            opt = qml.AdamOptimizer(stepsize=step_size)
            
        # Training Loop
        start_time = time.time()
        for i in range(steps):
            current_loss: float
            if optimizer == "qng":
                self.params, current_loss = opt.step_and_cost(cost, self.params, metric_tensor_fn=metric_tensor_fn)
            else:
                self.params, current_loss = opt.step_and_cost(cost, self.params)
                
            if verbose and i % 10 == 0:
                logger.info(f"Step {i:3d}: Loss = {current_loss:.6f}")
                
        if verbose:
            logger.info(f"Huấn luyện hoàn tất sau {time.time() - start_time:.2f}s. Final Loss: {current_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn cho dữ liệu mới.

        Args:
            X (np.ndarray): Dữ liệu đầu vào.

        Returns:
            np.ndarray: Nhãn dự đoán (Boolean hoặc 0/1).
        """
        if self.params is None:
            raise ValueError("Model chưa được huấn luyện. Gọi fit() trước.")
            
        predictions = [self.qnode(self.params, x_i)[1] for x_i in X]
        return np.array(predictions) > 0.5


class QuantumAutoencoder:
    """
    Variational Quantum Autoencoder for Data Compression.
    
    Args:
        n_qubits (int): Tổng số lượng qubits. Default: 2.
        n_latent (int): Số lượng qubits ẩn (latent space). Default: 1.
    """
    def __init__(self, n_qubits: int = 2, n_latent: int = 1):
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_trash = n_qubits - n_latent
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params: Optional[np.ndarray] = None
        
        @qml.qnode(self.dev)
        def _circuit(params: np.ndarray, feature_vector: np.ndarray) -> np.ndarray:
            qml.AmplitudeEmbedding(features=feature_vector, wires=range(n_qubits), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            trash_wires = [int(w) for w in (n_latent + np.arange(self.n_trash))]
            return qml.probs(wires=trash_wires)
        
        self.qnode = _circuit

    def fit(self, data: np.ndarray, steps: int = 100, step_size: float = 0.05) -> None:
        """
        Huấn luyện Autoencoder.

        Args:
            data (np.ndarray): Dữ liệu cần nén.
            steps (int): Số bước huấn luyện.
            step_size (float): Learning rate.
        """
        n_layers = 3
        shape = (n_layers, self.n_qubits, 3)
        self.params = np.random.uniform(low=0, high=2*np.pi, size=shape, requires_grad=True)
        
        opt = qml.AdamOptimizer(stepsize=step_size)
        
        def cost(params: np.ndarray) -> float:
            loss = 0.0
            for f in data:
                probs = self.qnode(params, f)
                loss += (1 - probs[0]) # Minimize 1 - P(|0>_trash)
            return loss / len(data)

        for i in range(steps):
            self.params, val = opt.step_and_cost(cost, self.params)
            if i % 10 == 0:
                logger.info(f"Step {i:3d}: Loss = {val:.6f}")
                
    def compress(self, feature_vector: np.ndarray) -> float:
        """
        Mô phỏng nén dữ liệu và trả về độ trung thực (fidelity).
        
        Args:
            feature_vector (np.ndarray): Dữ liệu đầu vào.
            
        Returns:
            float: Fidelity (độ trung thực) của nén.
        """
        if self.params is None:
            raise ValueError("Model chưa được huấn luyện.")
            
        # In simulator, we can't easily get the partial trace state vector without logic.
        # But we can return the fidelity metric.
        probs = self.qnode(self.params, feature_vector)
        return probs[0] # Fidelity


class RegressionQNN:
    """
    Quantum Neural Network for Regression tasks.
    Output: Expectation value of PauliZ on wire 0 (Range [-1, 1]).
    
    Args:
        n_qubits (int): Số lượng qubits. Default: 4.
        n_layers (int): Số lượng layers. Default: 3.
        data_reuploading (bool): Sử dụng cơ chế Data Re-uploading. Default: True.
        trainable_scaling (bool): Sử dụng tham số scaling (trainable encoding). Default: False.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 3, data_reuploading: bool = True, trainable_scaling: bool = False):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.data_reuploading = data_reuploading
        self.trainable_scaling = trainable_scaling
        self.params: Optional[np.ndarray] = None
        # Allocate extra wire for Full Metric Tensor (Hadamard Test)
        # Wires 0 to n_qubits-1 are used for circuit. Wire n_qubits is aux.
        self.dev = qml.device("default.qubit", wires=n_qubits + 1)
        self.loss_history: List[float] = []
        
        @qml.qnode(self.dev)
        def _circuit(params: np.ndarray, x: Union[float, np.ndarray, List[float]]) -> float:

            # Reshape flat params
            # var_shape = (n_layers, n_qubits, 3) -> size = n_layers * n_qubits * 3
            n_var = n_layers * n_qubits * 3
            
            var_params: np.ndarray
            embed_params: Optional[np.ndarray] = None

            if self.trainable_scaling:
                # embed_shape = (n_layers, n_qubits, 2) -> size = n_layers * n_qubits * 2
                var_flat = params[:n_var]
                embed_flat = params[n_var:]
                
                var_params = var_flat.reshape((n_layers, n_qubits, 3))
                embed_params = embed_flat.reshape((n_layers, n_qubits, 2))
            else:
                var_params = params.reshape((n_layers, n_qubits, 3))
                
            # Prepare x vector
            x_vec: np.ndarray
            if np.ndim(x) == 0:
                x_vec = np.array([x] * n_qubits)
            else:
                x_vec = np.array(x)
                if len(x_vec) < n_qubits:
                    x_vec = np.pad(x_vec, (0, n_qubits - len(x_vec)), mode='constant')
            
            # Data Re-uploading Architecture
            for i in range(n_layers):
                if self.data_reuploading or i == 0:
                    if self.trainable_scaling and embed_params is not None:
                        # Trainable Encoding: RX(x * scale + bias)
                        for q in range(n_qubits):
                            scale = embed_params[i, q, 0]
                            bias = embed_params[i, q, 1]
                            val = x_vec[q] * scale + bias
                            qml.RX(val, wires=q)
                    else:
                        qml.AngleEmbedding(x_vec, wires=range(n_qubits))
                        
                qml.StronglyEntanglingLayers(np.expand_dims(var_params[i], axis=0), wires=range(n_qubits))
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
            
        self.qnode = _circuit

    def fit(self, X: np.ndarray, y: np.ndarray, steps: int = 100, step_size: float = 0.01, optimizer: str = "qng", verbose: bool = True) -> None:
        """
        Fit model to data X, y.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target labels.
            steps (int): Number of training steps.
            step_size (float): Learning rate.
            optimizer (str): Optimizer choice ("qng" or "adam").
            verbose (bool): Print progress.
        """
        # Init params
        var_shape = (self.n_layers, self.n_qubits, 3)
        var_size = np.prod(var_shape)
        
        if self.params is None:
            var_params = np.random.uniform(low=0, high=2*np.pi, size=var_size, requires_grad=True)
            
            if self.trainable_scaling:
                # Initialize scales to 1.0 and biases to 0.0
                embed_size = self.n_layers * self.n_qubits * 2
                
                # Set scales closer to 1 (randomly distributed around 1)
                embed_params = np.random.uniform(low=0.5, high=1.5, size=embed_size, requires_grad=True)
                
                self.params = np.concatenate([var_params, embed_params])
            else:
                self.params = var_params
            
        self.loss_history = []
        
        # Optimizer
        opt: Any
        if optimizer == "qng":
            # NOTE: block-diag approximation is standard and doesn't require aux wires.
            opt = qml.QNGOptimizer(stepsize=step_size, approx="block-diag")
        else:
            opt = qml.AdamOptimizer(stepsize=step_size)
            
        # Cost Function (MSE)
        def cost(params: np.ndarray) -> float:
            preds = [self.qnode(params, x_i) for x_i in X]
            loss = np.mean((y - np.stack(preds)) ** 2)
            return loss
        
        # Metric Tensor for QNG
        def metric_tensor_fn(params: np.ndarray) -> np.ndarray:
            # approx="block-diag" works with flat params
            return qml.metric_tensor(self.qnode, approx="block-diag")(params, X[0])
            
        logger.info(f"Training RegressionQNN ({optimizer})...")
        start = time.time()
        
        for i in range(steps):
            val: float
            if optimizer == "qng":
                self.params, val = opt.step_and_cost(cost, self.params, metric_tensor_fn=metric_tensor_fn)
            else:
                self.params, val = opt.step_and_cost(cost, self.params)
            
            self.loss_history.append(val)
            if verbose and i % 20 == 0:
                logger.info(f"Step {i}: Loss={val:.4f}")
                
        logger.info(f"Finished in {time.time()-start:.2f}s. Final Loss: {self.loss_history[-1]:.4f}")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá trị cho dữ liệu mới.
        
        Args:
            X (np.ndarray): Dữ liệu đầu vào.
            
        Returns:
            np.ndarray: Giá trị dự đoán.
        """
        if self.params is None:
            raise ValueError("Model chưa được huấn luyện.")
        return np.array([self.qnode(self.params, x) for x in X])
