
"""
BENCHMARK: Quantum Regression
Mục tiêu: Đánh giá khả năng xấp xỉ hàm số liên tục của QNN (Regression).
Hàm mục tiêu: y = 0.5 * sin(3x) + 0.3 * cos(5x) + noise
Data range: x in [-pi, pi] mapped to [0, pi] for embedding.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

from qng.models import RegressionQNN

# ==========================================
# 1. GENERATE REGRESSION DATA
# ==========================================
np.random.seed(42)
n_samples = 100
X = np.linspace(-np.pi, np.pi, n_samples).reshape(-1, 1)
# Target function
y = 0.5 * np.sin(3 * X) + 0.3 * np.cos(5 * X)
# Add noise
y += 0.05 * np.random.normal(size=y.shape)

# Normalize X to [0, pi] for AngleEmbedding
scaler_x = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_x.fit_transform(X)

# Normalize y to [-0.8, 0.8] to fit within PauliZ range [-1, 1] comfortably
scaler_y = MinMaxScaler(feature_range=(-0.8, 0.8))
y_scaled = scaler_y.fit_transform(y).flatten() # Fit expects 1D array for y

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ==========================================
# 2. TRAIN MODELS
# ==========================================
# Single Qubit Re-uploading is sufficient and faster for 1D regression
n_qubits = 1
n_layers = 10 
steps = 150

print(f"\n--- fitting RegressionQNN ({n_qubits} qubits, {n_layers} layers, Re-uploading=True, TrainableScaling=True) ---")

# Train QNG
# Note: Trainable Scaling adds parameters (scale, bias) to embedding, allowing learning of frequencies.
model_qng = RegressionQNN(n_qubits=n_qubits, n_layers=n_layers, data_reuploading=True, trainable_scaling=True)
model_qng.fit(X_train, y_train, steps=steps, optimizer="qng", step_size=0.05)

# Train Adam
model_adam = RegressionQNN(n_qubits=n_qubits, n_layers=n_layers, data_reuploading=True, trainable_scaling=True)
model_adam.fit(X_train, y_train, steps=steps, optimizer="adam", step_size=0.05)

# ==========================================
# 3. EVALUATE & PLOT
# ==========================================
y_pred_qng = model_qng.predict(X_scaled)
y_pred_adam = model_adam.predict(X_scaled)

# Inverse transform y for plotting
y_pred_qng_orig = scaler_y.inverse_transform(y_pred_qng.reshape(-1, 1))
y_pred_adam_orig = scaler_y.inverse_transform(y_pred_adam.reshape(-1, 1))
y_orig = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))

# MSE on Test set
mse_qng = np.mean((y_test - model_qng.predict(X_test))**2)
mse_adam = np.mean((y_test - model_adam.predict(X_test))**2)

print(f"\nTest MSE (QNG): {mse_qng:.6f}")
print(f"Test MSE (Adam): {mse_adam:.6f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Function Approximation
ax1.scatter(X, scaler_y.inverse_transform(y), color='gray', alpha=0.5, label='Data')
ax1.plot(X, y_pred_qng_orig, 'r-', linewidth=2, label=f'QNG (MSE={mse_qng:.4f})')
ax1.plot(X, y_pred_adam_orig, 'b--', linewidth=2, label=f'Adam (MSE={mse_adam:.4f})')
ax1.set_title("Function Approximation")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss History
ax2.plot(model_qng.loss_history, 'r-', label='QNG')
ax2.plot(model_adam.loss_history, 'b--', label='Adam')
ax2.set_title("Training Loss (MSE)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Loss")
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/benchmark_regression.png")
print("Plot saved to reports/benchmark_regression.png")
