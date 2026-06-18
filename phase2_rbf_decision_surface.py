import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import sys

# Ensure the parent directory is in the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface

def main():
    print("=" * 60)
    print("Running Phase 2: SVM RBF Decision Function Surface Visualization")
    print("=" * 60)
    
    # 1. Setup paths
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Generate data with noise
    C_val = 10.0
    gamma_val = 1.0
    noise_val = 0.08
    seed_val = 7
    
    X, y = generate_ring_dataset(
        n_inner=35,
        n_outer=45,
        inner_radius_range=(0.0, 1.0),
        outer_radius_range=(1.6, 2.5),
        noise=noise_val,
        random_seed=seed_val
    )
    
    # 3. Train real RBF SVM
    clf = train_svm(X, y, kernel="rbf", C=C_val, gamma=gamma_val)
    
    # Calculate accuracy
    accuracy = clf.score(X, y)
    print(f"Model: SVM with RBF Kernel (C={C_val}, gamma={gamma_val})")
    print(f"Number of Support Vectors: {len(clf.support_)}")
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
    
    # 4. Create decision grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy, grid_points = make_decision_grid(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=150
    )
    
    # 5. Compute decision function surface values
    Z = compute_decision_surface(clf, grid_points, xx, yy)
    
    # Get support vectors details
    sv_indices = clf.support_
    X_sv = X[sv_indices]
    y_sv = y[sv_indices]
    
    # Compute decision scores for all points
    decision_scores = clf.decision_function(X)
    
    # Colormap settings
    # We use custom HSL-tailored colors for premium aesthetics:
    # Inner class (Blue): HSL (210, 80%, 50%)
    # Outer class (Red): HSL (350, 80%, 50%)
    c_blue = '#1E88E5'
    c_red = '#E53935'
    c_boundary = '#FFD54F' # Yellow
    c_margin = '#FFEB3B'   # Light Yellow
    
    # 6. Plot 2D Decision Boundary
    fig_2d, ax_2d = plt.subplots(figsize=(7, 6))
    
    # Background contour filled to represent decision confidence
    contour_fill = ax_2d.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.3)
    cbar = fig_2d.colorbar(contour_fill, ax=ax_2d)
    cbar.set_label("Decision Score f(x, y)", rotation=270, labelpad=15)
    
    # Decision boundary (f=0) and margins (f=-1, f=1)
    contour_lines = ax_2d.contour(
        xx, yy, Z, levels=[-1.0, 0.0, 1.0],
        colors=[c_margin, c_boundary, c_margin],
        linestyles=["dashed", "solid", "dashed"],
        linewidths=[1.5, 2.5, 1.5]
    )
    ax_2d.clabel(contour_lines, inline=True, fmt={-1.0: "-1 (Margin)", 0.0: "0 (Boundary)", 1.0: "+1 (Margin)"}, fontsize=9)
    
    # Scatter points
    ax_2d.scatter(X[y == 0, 0], X[y == 0, 1], c=c_blue, label="Inner Class (y=0)", edgecolors='k', s=50, zorder=3)
    ax_2d.scatter(X[y == 1, 0], X[y == 1, 1], c=c_red, label="Outer Class (y=1)", edgecolors='k', s=50, zorder=3)
    
    # Highlight support vectors with double rings
    ax_2d.scatter(
        X_sv[:, 0], X_sv[:, 1], s=120, facecolors='none',
        edgecolors='white', linewidths=2.0, label="Support Vectors", zorder=4
    )
    ax_2d.scatter(
        X_sv[:, 0], X_sv[:, 1], s=180, facecolors='none',
        edgecolors='black', linewidths=1.0, zorder=4
    )
    
    ax_2d.set_title(f"2D RBF SVM Decision Boundary (Accuracy: {accuracy*100:.1f}%)")
    ax_2d.set_xlabel("Feature x")
    ax_2d.set_ylabel("Feature y")
    ax_2d.legend(loc="upper right")
    ax_2d.grid(True, linestyle="--", alpha=0.5)
    
    path_2d = os.path.join(output_dir, "phase2_decision_boundary_2d.png")
    fig_2d.savefig(path_2d, dpi=150, bbox_inches="tight")
    plt.close(fig_2d)
    
    # 7. Plot 3D Decision Function Surface
    fig_3d = plt.figure(figsize=(8, 7))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot decision surface z = f(x, y)
    # Use RdBu_r colormap so blue represents negative decision scores and red represents positive
    surf = ax_3d.plot_surface(
        xx, yy, Z, cmap="RdBu_r", alpha=0.5,
        linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    
    # Plot reference plane z=0 (semi-transparent yellow)
    # We create a constant grid of 0s
    z_zero = np.zeros_like(xx)
    ax_3d.plot_surface(
        xx, yy, z_zero, color=c_boundary, alpha=0.15,
        shade=False, label="Separating Plane (z=0)"
    )
    
    # Plot 3D data points at (x, y, f(x, y))
    ax_3d.scatter(
        X[y == 0, 0], X[y == 0, 1], decision_scores[y == 0],
        c=c_blue, edgecolors='k', s=40, depthshade=False, label="Inner Class (y=0)"
    )
    ax_3d.scatter(
        X[y == 1, 0], X[y == 1, 1], decision_scores[y == 1],
        c=c_red, edgecolors='k', s=40, depthshade=False, label="Outer Class (y=1)"
    )
    
    # Highlight support vectors in 3D
    ax_3d.scatter(
        X_sv[:, 0], X_sv[:, 1], clf.decision_function(X_sv),
        s=100, facecolors='none', edgecolors='yellow', linewidths=2.0,
        depthshade=False, label="Support Vectors (3D)"
    )
    
    # Add labels and formatting
    ax_3d.set_title("3D SVM RBF Decision Function Surface z = f(x, y)")
    ax_3d.set_xlabel("Feature x")
    ax_3d.set_ylabel("Feature y")
    ax_3d.set_zlabel("Decision Score f(x, y)")
    
    # Adjust viewing angle for best visualization
    ax_3d.view_init(elev=28, azim=-48)
    
    path_3d = os.path.join(output_dir, "phase2_decision_surface_3d.png")
    fig_3d.savefig(path_3d, dpi=150, bbox_inches="tight")
    plt.close(fig_3d)
    
    # 8. Create combined image
    fig_comb, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    # Left: 2D Boundary
    contour_fill_c = ax1.contourf(xx, yy, Z, levels=50, cmap="RdBu_r", alpha=0.3)
    ax1.contour(
        xx, yy, Z, levels=[-1.0, 0.0, 1.0],
        colors=[c_margin, c_boundary, c_margin],
        linestyles=["dashed", "solid", "dashed"],
        linewidths=[1.5, 2.5, 1.5]
    )
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], c=c_blue, edgecolors='k', s=45, label="Inner Class")
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], c=c_red, edgecolors='k', s=45, label="Outer Class")
    ax1.scatter(X_sv[:, 0], X_sv[:, 1], s=100, facecolors='none', edgecolors='white', linewidths=2, label="Support Vectors")
    ax1.set_title("2D Projection: Decision Boundary & Margins")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Right: 3D Surface
    ax2_3d = fig_comb.add_subplot(1, 2, 2, projection='3d')
    ax2_3d.plot_surface(xx, yy, Z, cmap="RdBu_r", alpha=0.4, linewidth=0)
    ax2_3d.plot_surface(xx, yy, z_zero, color=c_boundary, alpha=0.1, shade=False)
    ax2_3d.scatter(X[y == 0, 0], X[y == 0, 1], decision_scores[y == 0], c=c_blue, edgecolors='k', s=30, depthshade=False)
    ax2_3d.scatter(X[y == 1, 0], X[y == 1, 1], decision_scores[y == 1], c=c_red, edgecolors='k', s=30, depthshade=False)
    ax2_3d.scatter(X_sv[:, 0], X_sv[:, 1], clf.decision_function(X_sv), s=80, facecolors='none', edgecolors='yellow', linewidths=1.5, depthshade=False)
    ax2_3d.set_title("3D View: Decision Surface z = f(x, y)")
    ax2_3d.set_xlabel("x")
    ax2_3d.set_ylabel("y")
    ax2_3d.set_zlabel("f(x, y)")
    ax2_3d.view_init(elev=28, azim=-48)
    
    # Add mathematical teaching note text at bottom of figure
    note_text = (
        "Note: The toy mapping z = x² + y² (Phase 1) is for basic geometric intuition.\n"
        "The real RBF kernel corresponds to an infinite-dimensional feature space; "
        "what we plot above is the 3D decision function f(x, y) = Σ α_i y_i K(x_i, x) + b."
    )
    fig_comb.text(0.5, 0.02, note_text, ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))
    
    path_combined = os.path.join(output_dir, "phase2_combined_visualization.png")
    fig_comb.savefig(path_combined, dpi=150, bbox_inches="tight")
    plt.close(fig_comb)
    
    print("\nSaved visualizations:")
    print(f"  - 2D Boundary: {path_2d}")
    print(f"  - 3D Surface:  {path_3d}")
    print(f"  - Combined:    {path_combined}")
    print("=" * 60)

if __name__ == "__main__":
    main()
