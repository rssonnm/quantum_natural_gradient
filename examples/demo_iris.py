
"""
DEMO: Huấn luyện Quantum Neural Network (QNN) sử dụng `modules`.
Dataset: Iris (Binary classification: Setosa vs Versicolor).
Optimizer: QNG vs Adam.

Mục tiêu:
1. Load dữ liệu Iris.
2. Preprocess (Scale về [0, pi] cho Angle Embedding).
3. Khởi tạo `SimpleQNN` từ `modules.models`.
4. Train và so sánh độ chính xác.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# IMPORT MODULE TỰ VIẾT
from qng.models import SimpleQNN

# ==========================================
# 1. LOAD & PREPROCESS DATA
# ==========================================
print("--- Loading Iris Dataset ---")
iris = load_iris()
X = iris.data[:100]  # Only first 2 classes (Setosa, Versicolor)
y = iris.target[:100]
# Convert labels to {-1, 1}
y = 2 * y - 1

# Scale data to [0, pi] for Angle Embedding
scaler = MinMaxScaler(feature_range=(0, np.pi))
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data shape: {X_train.shape}")
print(f"Labels: {np.unique(y_train)}")

# ==========================================
# 2. TRAIN WITH QUANTUM NATURAL GRADIENT
# ==========================================
print("\n--- Training with Quantum Natural Gradient (QNG) ---")
model_qng = SimpleQNN(n_qubits=4, n_layers=2)
# Train nhanh 30 steps
model_qng.fit(X_train, y_train, steps=30, step_size=0.02, optimizer="qng")

# Evaluate
y_prob_qng = model_qng.predict(X_test)
# SimpleQNN.predict trả về Xác suất class 1.
# Convert prob > 0.5 -> Label 1, else -1
y_pred_qng = np.where(y_prob_qng > 0.5, 1, -1)
acc_qng = np.mean(y_pred_qng == y_test)
print(f"QNG Test Accuracy: {acc_qng * 100:.2f}%")

# ==========================================
# 3. TRAIN WITH ADAM (BASELINE)
# ==========================================
print("\n--- Training with Adam (Baseline) ---")
model_adam = SimpleQNN(n_qubits=4, n_layers=2)
model_adam.fit(X_train, y_train, steps=30, step_size=0.02, optimizer="adam")

y_prob_adam = model_adam.predict(X_test)
y_pred_adam = np.where(y_prob_adam > 0.5, 1, -1)
acc_adam = np.mean(y_pred_adam == y_test)
print(f"Adam Test Accuracy: {acc_adam * 100:.2f}%")
