import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# Ensure the parent directory is in the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_generator import generate_ring_dataset
from utils.svm_utils import train_svm, make_decision_grid, compute_decision_surface

# Page configuration
st.set_page_config(
    page_title="SVM Kernel Trick 3D Interactive Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        /* Import Outfit Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Banner Card */
        .banner-card {
            background: linear-gradient(135deg, #1E3A8A 0%, #0D1B2A 100%);
            color: white;
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .banner-card h1 {
            color: #FFD54F !important;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .banner-card p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Glassmorphic Cards */
        .info-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Math warning callout */
        .math-warning {
            background-color: rgba(255, 213, 79, 0.08);
            border-left: 4px solid #FFD54F;
            padding: 1rem 1.5rem;
            border-radius: 4px 12px 12px 4px;
            margin: 1rem 0;
            color: #E2E8F0;
        }
        .math-warning-title {
            color: #FFD54F;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Caching Functions
# ---------------------------------------------------------
@st.cache_data
def cached_generate_data(n_inner, n_outer, noise, seed):
    return generate_ring_dataset(
        n_inner=n_inner,
        n_outer=n_outer,
        inner_radius_range=(0.0, 1.0),
        outer_radius_range=(1.6, 2.5),
        noise=noise,
        random_seed=seed
    )

@st.cache_data
def cached_make_grid(x_min, x_max, y_min, y_max, resolution):
    return make_decision_grid(
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        resolution=resolution
    )

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=64)
st.sidebar.title("Configuration")

st.sidebar.subheader("SVM Parameters")
kernel = st.sidebar.selectbox(
    "Kernel Function",
    options=["rbf", "linear", "poly", "sigmoid"],
    index=0,
    help="Determines the mathematical function used to map inputs."
)

C = st.sidebar.slider(
    "C (Regularization)",
    min_value=0.01,
    max_value=100.0,
    value=10.0,
    step=0.05,
    help="Penalty parameter C of the error term. Smaller C allows a wider margin but more errors."
)

# Conditionally display Gamma
if kernel in ["rbf", "poly", "sigmoid"]:
    gamma = st.sidebar.slider(
        "Gamma",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.05,
        help="Kernel coefficient. High gamma leads to tight decision boundaries (overfitting risk)."
    )
else:
    gamma = 1.0  # fallback ignored by linear

# Conditionally display Degree
if kernel == "poly":
    degree = st.sidebar.slider(
        "Polynomial Degree",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Degree of the polynomial kernel function."
    )
else:
    degree = 3  # fallback

st.sidebar.subheader("Data Generation")
number_of_points = st.sidebar.slider(
    "Total Data Points",
    min_value=40,
    max_value=300,
    value=120,
    step=10,
    help="Total number of points generated for the two classes combined."
)

noise = st.sidebar.slider(
    "Data Noise",
    min_value=0.0,
    max_value=0.5,
    value=0.08,
    step=0.01,
    help="Standard deviation of Gaussian noise added to the point coordinates."
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=1000,
    value=7,
    step=1,
    help="Seed to guarantee data reproducibility."
)

# ---------------------------------------------------------
# Main Panel UI
# ---------------------------------------------------------
st.markdown("""
    <div class="banner-card">
        <h1>SVM Kernel Trick 3D Interactive Demo</h1>
        <p>Explore Support Vector Machines visually. Learn how kernel transformations map non-linearly separable data into decision spaces where they become linearly separable.</p>
    </div>
""", unsafe_allow_html=True)

# Concept Introduction Panel
with st.expander("🎓 Learn the Core Educational Concept", expanded=True):
    st.markdown("""
        ### Why do we need the Kernel Trick?
        1. **2D Nonlinearity**: In the original 2D space, the center blue points and outer red ring points **cannot** be separated by any straight line.
        2. **Feature Mapping**: If we map each point $(x, y)$ to 3D space using a mapping like $\\phi(x, y) = (x, y, x^2 + y^2)$, the inner blue points stay low, while the outer red points rise high.
        3. **3D Hyperplane Separation**: In this 3D feature space, we can easily slide a flat horizontal plane between the two classes.
        4. **Nonlinear Decision Boundary**: When we project the intersection of this 3D plane back onto the 2D plane, it forms a circle (a non-linear decision boundary).
    """)
    st.markdown("""
        <div class="math-warning">
            <div class="math-warning-title">⚠️ Crucial Mathematical Note</div>
            The toy mapping $z = x^2 + y^2$ (used in the Phase 1 animation) is a simplified visual representation.
            In practice, the <b>Radial Basis Function (RBF) Kernel</b> does not map points explicitly to 3D. Rather, it corresponds to an 
            <b>infinite-dimensional</b> feature space. The 3D surface visualized below represents the SVM's decision function value 
            $z = f(x, y)$ (the classification score/margin distance), not the feature space itself.
        </div>
    """, unsafe_allow_html=True)

# Generate Data
n_inner = int(number_of_points * 0.45)
n_outer = int(number_of_points * 0.55)
X, y = cached_generate_data(n_inner, n_outer, noise, random_seed)

# Train SVM Model
try:
    clf = train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)
    model_success = True
except Exception as e:
    st.error(f"Failed to train SVM model with these parameters. Error: {e}")
    model_success = False

if model_success:
    # Compute mesh grid bounds and grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy, grid_points = cached_make_grid(x_min, x_max, y_min, y_max, resolution=80)
    
    # Compute decision surface Z
    Z = compute_decision_surface(clf, grid_points, xx, yy)
    decision_scores = clf.decision_function(X)
    
    # Support vector indices
    sv_indices = clf.support_
    X_sv = X[sv_indices]
    
    # Colors
    c_blue = '#1E88E5'
    c_red = '#E53935'
    c_boundary = '#FFD54F'
    c_margin = '#FFEB3B'
    
    # Accuracy
    accuracy = clf.score(X, y)
    
    # Layout with columns
    col1, col2 = st.columns(2)
    
    # ---------------------------------------------------------
    # Column 1: 2D Plotly Chart
    # ---------------------------------------------------------
    with col1:
        st.subheader("2D Projection: Decision Boundary & Margins")
        
        fig_2d = go.Figure()
        
        # Background Heatmap showing decision confidence
        fig_2d.add_trace(go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            colorscale='RdBu',
            reversescale=True,
            showscale=True,
            colorbar=dict(title='f(x, y) score', title_side='right'),
            contours_coloring='heatmap',
            opacity=0.35,
            hoverinfo='skip'
        ))
        
        # Decision Boundary (f(x, y) = 0) and Margins (f(x, y) = +/-1)
        fig_2d.add_trace(go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            showscale=False,
            contours=dict(
                coloring='none',
                start=-1,
                end=1,
                size=1,
                showlabels=True,
                labelfont=dict(size=11, color='white')
            ),
            line=dict(color='yellow', width=2),
            hoverinfo='skip'
        ))
        
        # Inner Class (0) Scatter
        fig_2d.add_trace(go.Scatter(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            mode='markers',
            marker=dict(color=c_blue, size=9, line=dict(color='black', width=1.5)),
            name='Inner Class (y=0)',
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra>Class 0 (Blue)</extra>'
        ))
        
        # Outer Class (1) Scatter
        fig_2d.add_trace(go.Scatter(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            mode='markers',
            marker=dict(color=c_red, size=9, line=dict(color='black', width=1.5)),
            name='Outer Class (y=1)',
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra>Class 1 (Red)</extra>'
        ))
        
        # Support Vectors Ring Highlight
        fig_2d.add_trace(go.Scatter(
            x=X_sv[:, 0],
            y=X_sv[:, 1],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=16,
                line=dict(color='white', width=2)
            ),
            name='Support Vector Highlight',
            hovertemplate='SV Coordinates: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
        
        fig_2d.update_layout(
            xaxis_title="Feature x",
            yaxis_title="Feature y",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_2d, use_container_width=True)
        
    # ---------------------------------------------------------
    # Column 2: 3D Plotly Chart
    # ---------------------------------------------------------
    with col2:
        st.subheader("3D View: Decision Function Surface z = f(x, y)")
        
        fig_3d = go.Figure()
        
        # Decision Surface
        fig_3d.add_trace(go.Surface(
            x=xx[0, :],
            y=yy[:, 0],
            z=Z,
            colorscale='RdBu',
            reversescale=True,
            opacity=0.6,
            showscale=False,
            colorbar=dict(title='f(x, y)'),
            name='Decision Surface'
        ))
        
        # Zero-Plane z=0 (Separating plane)
        fig_3d.add_trace(go.Surface(
            x=xx[0, :],
            y=yy[:, 0],
            z=np.zeros_like(Z),
            colorscale=[[0, 'rgba(255, 213, 79, 0.12)'], [1, 'rgba(255, 213, 79, 0.12)']],
            showscale=False,
            hoverinfo='none',
            name='Zero Reference Plane'
        ))
        
        # Class 0 Data Points in 3D
        fig_3d.add_trace(go.Scatter3d(
            x=X[y == 0, 0],
            y=X[y == 0, 1],
            z=decision_scores[y == 0],
            mode='markers',
            marker=dict(color=c_blue, size=5, line=dict(color='black', width=1)),
            name='Inner Class (y=0)',
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Score: %{z:.2f}<extra></extra>'
        ))
        
        # Class 1 Data Points in 3D
        fig_3d.add_trace(go.Scatter3d(
            x=X[y == 1, 0],
            y=X[y == 1, 1],
            z=decision_scores[y == 1],
            mode='markers',
            marker=dict(color=c_red, size=5, line=dict(color='black', width=1)),
            name='Outer Class (y=1)',
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Score: %{z:.2f}<extra></extra>'
        ))
        
        # Support Vectors in 3D (Gold diamonds)
        fig_3d.add_trace(go.Scatter3d(
            x=X_sv[:, 0],
            y=X_sv[:, 1],
            z=decision_scores[sv_indices],
            mode='markers',
            marker=dict(
                color='#FFD54F',
                symbol='diamond',
                size=7,
                line=dict(color='black', width=1.5)
            ),
            name='Support Vector (3D)',
            hovertemplate='SV x: %{x:.2f}<br>SV y: %{y:.2f}<br>Score: %{z:.2f}<extra></extra>'
        ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="Score f(x,y)",
                camera=dict(
                    eye=dict(x=1.35, y=-1.35, z=0.85)
                )
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
    # ---------------------------------------------------------
    # Metrics Panel
    # ---------------------------------------------------------
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="Training Accuracy",
            value=f"{accuracy * 100:.1f}%"
        )
    with m_col2:
        st.metric(
            label="Number of Support Vectors",
            value=f"{len(sv_indices)} / {number_of_points}"
        )
    with m_col3:
        st.metric(
            label="Active Kernel",
            value=kernel.upper()
        )
    with m_col4:
        st.metric(
            label="Regularization (C)",
            value=str(C)
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ---------------------------------------------------------
    # Dynamic Teaching & Parameter Intuition Notes
    # ---------------------------------------------------------
    st.subheader("💡 Parameter Analysis & Dynamic Teaching Notes")
    
    analysis_tips = []
    
    # Gamma Analysis
    if kernel in ["rbf", "poly", "sigmoid"]:
        if gamma < 0.2:
            analysis_tips.append("🟢 **Gamma is small (smooth boundary)**: The radius of influence for each support vector is wide. The decision boundary is smooth and generalized. It models structural trends rather than noise.")
        elif gamma > 3.0:
            analysis_tips.append("🔴 **Gamma is large (overfitting risk)**: The radius of influence is extremely narrow. The decision boundary becomes a set of local bubbles enclosing individual data points. High accuracy on this dataset, but extremely poor generalization on new data.")
        else:
            analysis_tips.append("🟡 **Gamma is balanced**: The model strikes a compromise, capturing structural shape changes without isolating individual data points.")
            
    # C Analysis
    if C < 1.0:
        analysis_tips.append("🟢 **C is small (soft margin)**: The optimization tolerates some training errors to widen the margin corridor. This keeps the boundary simpler and less sensitive to individual outliers.")
    elif C > 20.0:
        analysis_tips.append("🔴 **C is large (hard margin)**: The optimization aggressively attempts to classify every single training point correctly. This may cause the decision boundary to wiggle and warp significantly, increasing overfitting risk.")
    else:
        analysis_tips.append("🟡 **C is balanced**: The classifier balances classification accuracy with margin width.")
        
    # Display Tips
    for tip in analysis_tips:
        st.markdown(tip)
        
    # Suggested classroom exercises
    with st.expander("📝 Classroom Teaching Suggestions & Exercises"):
        st.markdown("""
            **Experiment 1: RBF Local Bubbles**
            1. Set the kernel to **RBF**.
            2. Keep **C = 10** and increase **Gamma** to **8.0** or higher.
            3. Observe how the decision boundary wraps tightly around individual points (forming localized islands). This is a textbook example of **overfitting**.
            
            **Experiment 2: Linear Kernel Limitation**
            1. Set the kernel to **Linear**.
            2. Notice that the model fails to separate the concentric dataset, with accuracy sitting around 50%.
            3. Observe that the 3D surface is a flat plane tilted in space. Cutting a paraboloid-like concentric distribution with a flat plane naturally fails.
            
            **Experiment 3: Soft Margin vs Hard Margin**
            1. Set the kernel to **RBF** and **Gamma = 1.0**.
            2. Add noise to the data (e.g., set **Noise = 0.25** or higher) to cause classes to overlap.
            3. Compare the boundary shape with **C = 0.1** (smooth boundary, some misplaced dots) versus **C = 100** (jagged, complicated boundary trying to snake around overlapping dots).
        """)

else:
    st.warning("Could not execute SVM visualization. Please adjust the sidebar settings to valid parameter bounds.")
