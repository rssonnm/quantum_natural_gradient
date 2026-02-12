import pytest
import numpy as np
import pennylane as qml
from qng.models import SimpleQNN, RegressionQNN, QuantumAutoencoder

def test_simple_qnn_init():
    model = SimpleQNN(n_qubits=2, n_layers=1)
    assert model.n_qubits == 2
    assert model.n_layers == 1
    assert model.params is None

def test_simple_qnn_fit_predict():
    model = SimpleQNN(n_qubits=2, n_layers=1)
    X = np.random.uniform(0, np.pi, (4, 2))
    y = np.array([0, 1, 0, 1])
    
    # Test fit
    model.fit(X, y, steps=2, step_size=0.1, optimizer="adam", verbose=False)
    assert model.params is not None
    
    # Test predict
    y_pred = model.predict(X)
    assert len(y_pred) == 4
    assert y_pred.dtype == bool

def test_regression_qnn_init():
    model = RegressionQNN(n_qubits=2, n_layers=1)
    assert model.n_qubits == 2
    assert model.n_layers == 1

def test_regression_qnn_fit():
    model = RegressionQNN(n_qubits=2, n_layers=1)
    X = np.random.uniform(0, 1, (4, 2))
    y = np.random.uniform(-1, 1, (4,))
    
    model.fit(X, y, steps=2, step_size=0.1, optimizer="qng", verbose=False)
    assert model.params is not None
    
    y_pred = model.predict(X)
    assert len(y_pred) == 4

def test_autoencoder_fit():
    model = QuantumAutoencoder(n_qubits=2, n_latent=1)
    data = np.random.uniform(0, 1, (4, 4)) # Amplitude embedding needs 2^n
    # Normalize data for amplitude embedding
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    model.fit(data, steps=2, step_size=0.1)
    assert model.params is not None
    
    fidelity = model.compress(data[0])
    assert 0 <= fidelity <= 1.0
