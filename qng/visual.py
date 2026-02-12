import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Any, Callable

from .utils import get_logger

logger = get_logger(__name__)

def plot_convergence_comparison(
    gd_history: List[float], 
    qng_history: List[float], 
    qgf_history: List[float], 
    steps_flow: List[float], 
    adam_history: Optional[List[float]] = None, 
    filename: str = "convergence.png"
) -> None:
    """
    Plot convergence comparison between GD, QNG, QGF, and Adam.
    
    Args:
        gd_history (List[float]): Loss history for Gradient Descent.
        qng_history (List[float]): Loss history for Quantum Natural Gradient.
        qgf_history (List[float]): Loss history for Quantum Gradient Flow.
        steps_flow (List[float]): Time steps for QGF.
        adam_history (Optional[List[float]]): Loss history for Adam optimizer.
        filename (str): Output filename.
    """
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. QGF (Continuous) - Background
    if len(qgf_history) > 0:
        plt.plot(steps_flow, qgf_history, label="Quantum Gradient Flow (Continuous)", 
                 color="green", linewidth=6, alpha=0.3)

    # 4. Adam Optimizer - Middle (New)
    if adam_history is not None:
        plt.plot(adam_history, label="Adam Optimizer", color="orange", linewidth=2.5, linestyle='-.')

    # 2. QNG (Discrete) - Middle
    plt.plot(qng_history, label="Quantum Natural Gradient (Discrete)", 
             color="red", linewidth=2.5, linestyle='--')

    # 3. Vanilla GD - Top
    plt.plot(gd_history, label="Vanilla SGD", 
             color="blue", linewidth=2, alpha=1.0)

    # Baseline
    plt.axhline(y=-1.0, color="black", linestyle="--", label="Ground State Energy")

    plt.title("Optimization Convergence Comparison", fontsize=14)
    plt.xlabel("Optimization Steps (Time / Learning Rate)", fontsize=12)
    plt.ylabel("Cost Function Expectation Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=300)
    logger.info(f"Plot saved to: {filename}")

def plot_loss_landscape_pca(
    cost_fn: Callable[[np.ndarray], float], 
    trajectories: List[np.ndarray], 
    labels: List[str], 
    filename: str = "loss_landscape.png", 
    grid_size: int = 30
) -> None:
    """
    Visualize the loss landscape using PCA projection of the optimization trajectories.
    
    Args:
        cost_fn (Callable): The cost function J(theta).
        trajectories (List[np.ndarray]): List of parameter histories [hist1, hist2, ...].
        labels (List[str]): Labels for each trajectory (e.g., ["SGD", "QNG"]).
        filename (str): Output filename.
        grid_size (int): Resolution of the grid.
    """
    from sklearn.decomposition import PCA # Local import
    logger.info("Generating Loss Landscape (PCA)...")
    
    # 1. Collect all points to fit PCA
    # Flatten if trajectory list contains lists of arrays
    all_points_list = []
    for t in trajectories:
        if isinstance(t, list):
            all_points_list.append(np.array(t))
        else:
            all_points_list.append(t)
            
    all_points = np.vstack(all_points_list)
    
    # 2. PCA Projection to 2D
    pca = PCA(n_components=2)
    pca.fit(all_points)
    
    # Define grid limits based on projected points
    points_2d = pca.transform(all_points)
    x_min, x_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    y_min, y_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    
    # Add margin
    margin_x = (x_max - x_min) * 0.2
    margin_y = (y_max - y_min) * 0.2
    x_range = np.linspace(x_min - margin_x, x_max + margin_x, grid_size)
    y_range = np.linspace(y_min - margin_y, y_max + margin_y, grid_size)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    # 3. Evaluate Cost on Grid
    # We need to inverse transform (2D -> High Dim) to evaluate cost
    # PCA Inverse: X_orig = X_2d * Components + Mean
    
    total_grid_points = grid_size * grid_size
    logger.info(f"Evaluating cost function on {total_grid_points} grid points...")
    
    # Sample shape from first point
    sample_shape = trajectories[0][0].shape
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Point in 2D PCA space
            point_2d = np.array([[X[i, j], Y[i, j]]])
            # Back to parameter space
            point_hd = pca.inverse_transform(point_2d).reshape(sample_shape)
            # Evaluate Cost
            Z[i, j] = cost_fn(point_hd)
            
    # 4. Plot
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-white')
    
    # Contour Plot
    cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(cp, label='Cost Value')
    
    # Plot Trajectories
    colors = ['blue', 'red', 'orange', 'green']
    markers = ['o', 'x', '^', '.']
    
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        # Project trajectory to 2D
        traj_array = np.array(traj) if isinstance(traj, list) else traj
        traj_2d = pca.transform(traj_array)
        
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=c, label=label, linewidth=2, marker=m, markersize=4, alpha=0.8)
        # Mark Start (Black Star)
        plt.plot(traj_2d[0, 0], traj_2d[0, 1], 'k*', markersize=10)
        # Mark End (White Circle)
        plt.plot(traj_2d[-1, 0], traj_2d[-1, 1], 'wo', markersize=6, markeredgecolor='k')

    plt.title("Optimization Trajectories in PCA Subspace", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.legend()
    
    plt.savefig(filename, dpi=300)
    logger.info(f"Loss landscape saved to: {filename}")

