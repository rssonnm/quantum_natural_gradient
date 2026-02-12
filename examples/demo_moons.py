
"""
BENCHMARK: QNN Classification on Non-Linear Data (Two Moons)
Mục tiêu: So sánh khả năng hội tụ của QNG vs Adam trên dữ liệu phi tuyến.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

from qng.models import SimpleQNN

# ==========================================
# 1. GENERATE COMPLEX DATA (MOONS)
# ==========================================
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
# Rescale to [0, pi] for angle embedding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)
y = 2 * y - 1 # {-1, 1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize Data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k')
plt.title("Two Moons Dataset (Rescaled)")
plt.savefig("reports/moons_dataset.png")

# ==========================================
# 2. TRAIN MODELS
# ==========================================
n_qubits = 2 # 2 features -> 2 qubits is enough for basic embedding, but let's use 4 for expressivity
# Actually SimpleQNN uses AngleEmbedding on range(n_qubits).
# Since only 2 features, limit n_qubits=2 for direct mapping, or pad features.
# SimpleQNN logic: `qml.AngleEmbedding(x, wires=range(n_qubits))`
# If x has 2 features and n_qubits=4, it might error or pad?
# PennyLane AngleEmbedding expects features to match wires.
# Let's check modules/models.py
# `qml.AngleEmbedding(x, wires=range(n_qubits))`
# If len(x) < n_qubits, it recycles? No, usually exception.
# Let's stick to n_qubits=2.
n_qubits = 2 
n_layers = 4 # Deeper layers for non-linearity

print(f"\n--- Training QNN on Moons ({n_qubits} Qubits, {n_layers} Layers) ---")

def train_and_eval(name, optimizer, steps=60):
    print(f"\nOptimizer: {name}")
    model = SimpleQNN(n_qubits=n_qubits, n_layers=n_layers)
    
    # Custom fit loop to capture history if needed, but simple .fit is enough for proof
    # Wait, SimpleQNN.fit prints loss every 10 steps.
    # We want loss history to plot.
    # SimpleQNN doesn't return history currently.
    # I should modify SimpleQNN or just rely on console output?
    # Let's modify modules/models.py to return history or access it.
    # checking models.py -> it has `self.loss_history` in the snippet?
    # No, the refactored version does NOT have `self.loss_history`.
    # It prints loss.
    # I will modify modules/models.py quickly to add history attribute.
    
    model.fit(X_train, y_train, steps=steps, optimizer=optimizer, step_size=0.05)
    
    # Eval
    acc = np.mean((2*(model.predict(X_test)) - 1) == y_test)
    print(f"Test Accuracy ({name}): {acc*100:.2f}%")
    return model

# Train QNG
model_qng = train_and_eval("QNG", "qng")

# Train Adam
model_adam = train_and_eval("Adam", "adam")

# Compare Decision Boundaries
print("\nPlotting decision boundaries...")
xx, yy = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, np.pi, 30))
X_grid = np.c_[xx.ravel(), yy.ravel()]

pred_qng = model_qng.predict(X_grid).reshape(xx.shape)
pred_adam = model_adam.predict(X_grid).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].contourf(xx, yy, pred_qng, alpha=0.8, cmap='coolwarm')
ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k')
ax[0].set_title("QNG Decision Boundary")

ax[1].contourf(xx, yy, pred_adam, alpha=0.8, cmap='coolwarm')
ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor='k')
ax[1].set_title("Adam Decision Boundary")

plt.savefig("reports/benchmark_moons_boundary.png")
print("Saved to reports/benchmark_moons_boundary.png")
